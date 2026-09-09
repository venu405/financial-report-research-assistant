"""LLM 重排器的提示内容与解析回归测试。"""

import json

from services.kb.qa_graph import _protect_reranked_financial_candidates
from services.kb.reranker import CrossEncoderReranker, LLMReranker
from tests.mocks import FakeLLM


def test_reranker_sees_title_and_end_of_chunk():
    tail_fact = "政府采购应当遵循公开透明、公平竞争、公正和诚实信用原则"
    llm = FakeLLM(default="0:9")
    reranker = LLMReranker(llm)
    hit = {
        "text": "前置内容" * 100 + tail_fact,
        "metadata": {"doc_title": "中华人民共和国政府采购法"},
    }

    result = reranker.rerank("政府采购应当遵循哪些基本原则？", [hit], 1)

    assert result[0]["rerank_score"] == 9
    prompt = llm.prompts[-1]
    assert "中华人民共和国政府采购法" in prompt
    assert tail_fact in prompt
    assert "属于其他公司，必须评为 0" in prompt


def test_llm_bad_order_keeps_exact_financial_candidate_in_top_n():
    good = {
        "chunk_id": "exact-financial-fact",
        "text": "华微电子2024年营业收入为100万元。",
        "metadata": {
            "doc_id": "huadian-2024",
            "doc_title": "华微电子 2024年年度报告",
            "report_period": "2024年度",
            "statement_scope": "consolidated",
            "financial_facts_json": json.dumps(
                [
                    {
                        "metric": "revenue",
                        "raw_value": "100",
                        "canonical_value": "1000000",
                        "unit": "万元",
                        "report_period": "2024年度",
                        "statement_scope": "consolidated",
                    }
                ],
                ensure_ascii=False,
            ),
        },
    }
    unrelated = {
        "chunk_id": "unrelated",
        "text": "华微电子的组织架构和联系方式。",
        "metadata": {"doc_id": "huadian-2024", "doc_title": "公司概况"},
    }
    candidates = [unrelated, good]
    reranked = LLMReranker(FakeLLM(default="0:9\n1:1")).rerank(
        "华微电子2024年营业收入是多少？", candidates, 1
    )

    result = _protect_reranked_financial_candidates(
        "华微电子2024年营业收入是多少？", candidates, reranked, 1
    )

    assert [hit["chunk_id"] for hit in result] == ["exact-financial-fact"]


def test_financial_protection_keeps_reranker_order_for_ordinary_questions():
    hits = [
        {"chunk_id": "first", "text": "制度甲"},
        {"chunk_id": "second", "text": "制度乙"},
        {"chunk_id": "third", "text": "制度丙"},
    ]

    result = _protect_reranked_financial_candidates(
        "采购制度的适用范围是什么？", hits, [hits[2], hits[0], hits[1]], 2
    )

    assert [hit["chunk_id"] for hit in result] == ["third", "first"]


class _FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, documents))
        return self.scores


def test_crossencoder_is_lazy_and_has_stable_normalized_scores(monkeypatch):
    model = _FakeCrossEncoder([0.0, 2.0, 2.0])
    reranker = CrossEncoderReranker(model_name="local-test")
    load_count = []

    def load_model():
        load_count.append(True)
        return model

    monkeypatch.setattr(reranker, "_ensure_model", load_model)
    hits = [
        {"chunk_id": "c1", "text": "一"},
        {"chunk_id": "c2", "text": "二"},
        {"chunk_id": "c3", "text": "三"},
    ]

    assert reranker._model is None
    result = reranker.rerank("问题", hits, 3)

    assert load_count == [True]
    assert model.calls == [("问题", ["一", "二", "三"])]
    # c2/c3 同分时保持输入顺序；分数均为有限的 0-10 数值。
    assert [hit["chunk_id"] for hit in result] == ["c2", "c3", "c1"]
    assert result[0]["rerank_score"] == result[1]["rerank_score"] == 8.8
    assert result[2]["rerank_score"] == 5.0
    assert all(0.0 <= hit["rerank_norm"] <= 1.0 for hit in result)


def test_crossencoder_invalid_score_and_exception_degrade_stably(monkeypatch):
    assert CrossEncoderReranker._normalize_score(float("inf")) == 0.0
    assert CrossEncoderReranker._normalize_score("not-a-score") == 0.0

    reranker = CrossEncoderReranker()
    monkeypatch.setattr(reranker, "_ensure_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    hits = [{"chunk_id": "c1", "text": "一"}, {"chunk_id": "c2", "text": "二"}]
    result = reranker.rerank("问题", hits, 2)
    assert [hit["chunk_id"] for hit in result] == ["c1", "c2"]
