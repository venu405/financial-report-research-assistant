"""LangGraph 问答图测试：FakeLLM + FakeEmbedding（不联网、毫秒级）。"""
from __future__ import annotations

from services.kb.qa_graph import build_qa_graph, run_qa
from services.kb.vector_store import VectorStore
from tests.mocks import FakeEmbedding, FakeLLM


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
        "你是企业知识库助手": "根据制度，超过5万元必须公开招投标。[1]",
        "RAG 质量评估员": "10",
    })
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)
    result = run_qa(graph, question="超过多少万元必须招投标", kb_id="default")
    assert "5万元" in result["answer"]
    assert len(result["citations"]) == 1
    assert result["score"] == 10
    assert result["retries"] == 0


def test_qa_graph_retry_on_low_quality(tmp_path):
    store, emb = _make_env(tmp_path)
    llm = FakeLLM(route={
        "你是企业知识库助手": "乱编的答案[1]",
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
