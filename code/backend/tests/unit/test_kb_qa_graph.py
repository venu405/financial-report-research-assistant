"""LangGraph 问答图测试：FakeLLM + FakeEmbedding（不联网、毫秒级）。"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.kb.qa_graph import (
    _llm_invoke,
    _llm_stream,
    _normalize_citations,
    build_qa_graph,
    run_qa,
    run_qa_stream,
)
from services.kb.retriever import HybridRetriever
from services.kb.vector_store import VectorStore
from tests.mocks import FakeEmbedding, FakeLLM


class _CaptureOpenAI:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content="stream-ok")
                            )
                        ]
                    )
                ]
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="invoke-ok"))]
        )


def test_openai_invoke_default_keeps_request_shape():
    llm = _CaptureOpenAI()

    assert _llm_invoke(llm, [{"role": "user", "content": "hi"}]) == "invoke-ok"

    request = llm.calls[0]
    assert request["model"] == "deepseek-chat"
    assert "extra_body" not in request


def test_openai_invoke_reasoning_effort_merges_extra_body():
    llm = _CaptureOpenAI()

    _llm_invoke(
        llm,
        [{"role": "user", "content": "hi"}],
        reasoning_effort="none",
        extra_body={"keep_me": "yes"},
    )

    assert llm.calls[0]["extra_body"] == {
        "keep_me": "yes",
        "reasoning_effort": "none",
    }


def test_openai_stream_reasoning_effort_is_sent_in_extra_body():
    llm = _CaptureOpenAI()

    assert list(
        _llm_stream(
            llm,
            [{"role": "user", "content": "hi"}],
            reasoning_effort="none",
        )
    ) == ["stream-ok"]

    request = llm.calls[0]
    assert request["stream"] is True
    assert request["extra_body"] == {"reasoning_effort": "none"}


def _make_env(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(["采购金额超过5万元必须公开招投标"]),
        texts=["采购金额超过5万元必须公开招投标"],
        doc_id="d1", doc_title="采购制度", source_type="md",
        chunk_indices=[0], kb_id="default",
    )
    return store, emb


def test_qa_graph_answer_with_citations_and_score(tmp_path):
    store, emb = _make_env(tmp_path)
    llm = FakeLLM(route={
        "仅基于以下资料": "根据制度，超过5万元必须公开招投标。[1]",
        "RAG 质量评估员": "10",
    })
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)
    result = run_qa(graph, question="超过多少万元必须招投标", kb_id="default")
    assert "5万元" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["score"] == 10
    assert result["retries"] == 0
    generation_prompt = next(prompt for prompt in llm.prompts if "仅基于以下资料" in prompt)
    assert "ST简称或股票代码时视为主体明确" in generation_prompt
    assert "优先回答合并口径" in generation_prompt
    assert "禁止用母公司数值替代公司整体数值" in generation_prompt


def test_reason_question_prompt_requires_evidence_coverage_and_citations(tmp_path):
    store, emb = _make_env(tmp_path)
    llm = FakeLLM(
        route={
            "仅基于以下资料": "收入下降，主要原因是销售减少。[1]",
            "RAG 质量评估员": "10 10",
        }
    )
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    run_qa(graph, question="收入下降的主要原因是什么？", kb_id="default")

    generation_prompt = next(
        prompt for prompt in llm.prompts if "仅基于以下资料" in prompt
    )
    assert "若问题询问原因、为何或变动原因" in generation_prompt
    assert "覆盖证据中明确写出的主要原因，并为每个原因逐项引用" in generation_prompt
    assert "不要编造证据未明确写出的原因" in generation_prompt


def test_retry_without_verification_reasons_gets_completeness_correction(
    tmp_path, monkeypatch
):
    store, emb = _make_env(tmp_path)
    monkeypatch.setattr(
        "services.kb.qa_graph._answer_verification_reasons",
        lambda question, answer, evidence: [],
    )
    llm = FakeLLM(
        route={
            "仅基于以下资料": "收入下降。[1]",
            "RAG 质量评估员": "2 2",
        }
    )
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    result = run_qa(graph, question="收入下降的主要原因是什么？", kb_id="default")

    generation_prompts = [
        prompt for prompt in llm.prompts if "仅基于以下资料" in prompt
    ]
    assert result["retries"] == 1
    assert len(generation_prompts) == 2
    assert "完整性/相关性质量评估" in generation_prompts[1]
    assert "原因题尤其不要只写变化结果或单一表层原因" in generation_prompts[1]
    assert "补齐资料明确支持的要点" in generation_prompts[1]


def test_non_reason_question_gets_conditional_reason_guidance_only(tmp_path):
    store, emb = _make_env(tmp_path)
    llm = FakeLLM(
        route={
            "仅基于以下资料": "超过5万元必须公开招投标。[1]",
            "RAG 质量评估员": "10 10",
        }
    )
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    run_qa(graph, question="超过多少万元必须招投标？", kb_id="default")

    generation_prompt = next(
        prompt for prompt in llm.prompts if "仅基于以下资料" in prompt
    )
    reason_rule = next(
        line for line in generation_prompt.splitlines() if line.startswith("10. ")
    )
    assert reason_rule.startswith("10. 若问题询问原因、为何或变动原因")
    assert "必须列出多个原因" not in generation_prompt


def test_retry_with_verification_reasons_keeps_targeted_reason(tmp_path, monkeypatch):
    store, emb = _make_env(tmp_path)
    reason = "数字未与字段绑定"
    monkeypatch.setattr(
        "services.kb.qa_graph._answer_verification_reasons",
        lambda question, answer, evidence: [reason],
    )
    llm = FakeLLM(route={"仅基于以下资料": "收入下降。[1]"})
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    run_qa(graph, question="收入下降的主要原因是什么？", kb_id="default")

    generation_prompts = [
        prompt for prompt in llm.prompts if "仅基于以下资料" in prompt
    ]
    assert len(generation_prompts) == 2
    assert "未通过确定性核验" in generation_prompts[1]
    assert reason in generation_prompts[1]
    assert "不要补造资料、改写数字单位或省略口径标签" in generation_prompts[1]


def test_citations_are_reindexed_and_answer_is_rewritten():
    all_citations = [
        {"index": index, "chunk_id": f"c{index}"} for index in range(1, 6)
    ]

    answer, citations = _normalize_citations(
        "采购人、供应商和代理机构。[1][2][5] 无效来源[9]", all_citations
    )

    assert answer == "采购人、供应商和代理机构。[1][2][3] 无效来源9"
    assert [item["index"] for item in citations] == [1, 2, 3]
    assert [item["chunk_id"] for item in citations] == ["c1", "c2", "c5"]


def test_qa_graph_retry_on_low_quality(tmp_path):
    store, emb = _make_env(tmp_path)
    llm = FakeLLM(route={
        "仅基于以下资料": "乱编的答案[1]",
        "RAG 质量评估员": "2",  # 持续低分 → 重试 1 次（MAX_RETRY=1）后结束
    })
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)
    result = run_qa(graph, question="采购规则", kb_id="default")
    assert result["retries"] == 1
    assert result["score"] <= 5  # 低分保留（供前端展示"质量不佳"）


def test_qa_graph_no_results_honest(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))  # 空库
    emb = FakeEmbedding()
    llm = FakeLLM()  # 仅护栏分类调 1 次 LLM；generate/evaluate 不硬编、不调 LLM
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)
    result = run_qa(graph, question="随便问问", kb_id="default")
    assert "未检索到" in result["answer"]
    assert result["citations"] == []
    assert result["score"] == 10  # 诚实声明（无资料）满分
    assert llm.call_count == 1  # 只有护栏分类 1 次，generate/evaluate 不调 LLM


def test_qa_graph_expands_neighbor_context_but_keeps_matched_citation(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    texts = ["前置定义", "核心证据", "后续限制"]
    store.add_chunks(
        embeddings=emb.embed_texts(texts),
        texts=texts,
        doc_id="neighbor-doc",
        doc_title="相邻上下文",
        source_type="md",
        chunk_indices=[0, 1, 2],
        kb_id="default",
        extra_metadata=[
            {"chunk_type": "child", "next_chunk_id": "neighbor-doc-1"},
            {
                "chunk_type": "child",
                "previous_chunk_id": "neighbor-doc-0",
                "next_chunk_id": "neighbor-doc-2",
            },
            {"chunk_type": "child", "previous_chunk_id": "neighbor-doc-1"},
        ],
    )
    llm = FakeLLM(
        route={
            "客服意图分类器": "kb_question",
            "仅基于以下资料": "回答。[1]",
            "RAG 质量评估员": "10",
        }
    )
    graph = build_qa_graph(
        llm=llm,
        embeddings=emb,
        vector_store=store,
        top_k=1,
        recall_k=3,
        neighbor_expansion=1,
    )

    result = run_qa(graph, question="核心证据", kb_id="default")

    generation_prompt = next(prompt for prompt in llm.prompts if "仅基于以下资料" in prompt)
    assert "前置定义\n\n核心证据\n\n后续限制" in generation_prompt
    assert result["citations"][0]["chunk_id"] == "neighbor-doc-1"


def test_qa_graph_keeps_original_query_and_expands_recall(tmp_path, monkeypatch):
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    llm = FakeLLM(
        responses=["kb_question", "安全改写后的 2026 查询"],
        route={
            "仅基于以下资料": "根据资料回答。[1]",
            "RAG 质量评估员": "10 10",
        },
    )
    calls = []

    def fake_search_queries(self, queries, **kwargs):
        calls.append((list(queries), kwargs))
        return [
            {
                "chunk_id": "c1",
                "text": "2026 年制度证据",
                "metadata": {"doc_id": "d1", "source_type": "pdf", "kb_id": "default"},
                "score": 0.9,
            }
        ]

    class CaptureReranker:
        mode = "off"

        def rerank(self, query, hits, top_n):
            assert "原始问题 2026" in query
            assert "安全改写后的 2026 查询" in query
            return hits[:top_n]

    monkeypatch.setattr(HybridRetriever, "search_queries", fake_search_queries)
    graph = build_qa_graph(
        llm=llm,
        embeddings=emb,
        vector_store=store,
        top_k=1,
        recall_k=7,
        reranker=CaptureReranker(),
        metadata_filters={"chunk_type": "child"},
    )
    result = run_qa(
        graph,
        question="原始问题 2026",
        history=[{"role": "user", "content": "上一轮上下文"}],
        kb_id="default",
        metadata_filters={"source_type": "pdf"},
    )

    assert calls[0][0] == ["原始问题 2026", "安全改写后的 2026 查询"]
    assert calls[0][1]["top_k"] == 7
    assert calls[0][1]["metadata_filters"] == {
        "$and": [{"chunk_type": "child"}, {"source_type": "pdf"}]
    }
    assert result["search_meta"]["recall_raw"]
    assert result["search_meta"]["contexts"] == result["contexts"]


@pytest.mark.parametrize(
    ("question", "fact"),
    [
        ("上市时间是什么时候", "上市时间"),
        ("请问上市时间是什么时候", "上市时间"),
        ("想知道联系电话", "联系电话"),
        ("帮我查一下董事长是谁", "董事长"),
        ("公司股票上市交易所是哪里", "上市交易所"),
    ],
)
def test_company_fact_without_subject_clarifies_before_retrieval_or_generation(
    tmp_path, monkeypatch, question, fact
):
    """没有公司主体的上市时间问题必须在护栏直接澄清。"""
    store, emb = _make_env(tmp_path)
    llm = FakeLLM()

    def fail_retrieve(*args, **kwargs):
        raise AssertionError("ambiguous company fact must not retrieve")

    monkeypatch.setattr(HybridRetriever, "search", fail_retrieve)
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    result = run_qa(graph, question=question, kb_id="default")

    assert result["answer"] == f"请问您想查询哪家公司的{fact}？请提供公司名称或股票代码。"
    assert result["needs_clarification"] is True
    assert result["citations"] == []
    assert result["escalate"] is False
    assert result["search_meta"]["attempts"] == 0
    assert llm.call_count == 0


def test_ambiguous_user_history_cannot_be_confirmed_by_assistant_answer(tmp_path, monkeypatch):
    """助手曾回答交易所不能反向确认用户要查询的公司。"""
    store, emb = _make_env(tmp_path)
    llm = FakeLLM()

    monkeypatch.setattr(
        HybridRetriever,
        "search",
        lambda *args, **kwargs: pytest.fail("ambiguous company fact must not retrieve"),
    )
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    result = run_qa(
        graph,
        question="上市时间是什么时候",
        history=[
            {"role": "user", "content": "公司股票上市交易所是哪里"},
            {"role": "assistant", "content": "上海证券交易所"},
        ],
        kb_id="default",
    )

    assert result["needs_clarification"] is True
    assert result["answer"].startswith("请问您想查询哪家公司的上市时间？")
    assert result["search_meta"]["attempts"] == 0
    assert llm.call_count == 0


def test_explicit_company_in_user_history_keeps_original_flow(tmp_path):
    """用户历史明确公司后，追问上市时间不走澄清。"""
    store, emb = _make_env(tmp_path)
    llm = FakeLLM(
        route={
            "仅基于以下资料": "根据资料回答。[1]",
            "RAG 质量评估员": "10 10",
        }
    )
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    result = run_qa(
        graph,
        question="上市时间呢",
        history=[
            {"role": "user", "content": "我想查询深圳市广道数字技术股份有限公司"},
        ],
        kb_id="default",
    )

    assert result["needs_clarification"] is False
    assert llm.call_count > 0
    evaluator_prompt = next(prompt for prompt in llm.prompts if "RAG 质量评估员" in prompt)
    assert "【用户历史】" in evaluator_prompt
    assert "深圳市广道数字技术股份有限公司" in evaluator_prompt


@pytest.mark.parametrize(
    "question",
    [
        "深圳市广道数字技术股份有限公司的固定电话是什么",
        "ST广道的上市时间是什么时候",
        "000001的上市交易所是什么",
    ],
)
def test_explicit_company_forms_are_not_blocked(tmp_path, monkeypatch, question):
    """公司全称、ST 简称和六位股票代码都应通过主体门禁。"""
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    llm = FakeLLM()
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    result = run_qa(graph, question=question, kb_id="default")

    assert result["needs_clarification"] is False


def test_stream_final_transmits_clarification_without_retrieval(tmp_path, monkeypatch):
    """流式 final 同步透传 needs_clarification，且没有检索节点。"""
    store, emb = _make_env(tmp_path)
    llm = FakeLLM()
    monkeypatch.setattr(
        HybridRetriever,
        "search",
        lambda *args, **kwargs: pytest.fail("ambiguous company fact must not retrieve"),
    )
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)

    events = list(run_qa_stream(graph, question="上市时间是什么时候", kb_id="default"))
    final = next(event for event in events if event.get("type") == "final")
    nodes = {event.get("node") for event in events if event.get("type") == "node"}

    assert final["needs_clarification"] is True
    assert final["citations"] == []
    assert final["escalate"] is False
    assert not nodes.intersection({"faq_lookup", "rewrite", "retrieve", "generate"})
    assert llm.call_count == 0
