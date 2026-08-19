"""护栏分类 + 检索日志回归测试（收尾第 4 步）。"""
from __future__ import annotations

from services.kb.qa_graph import _normalize_category
from services.kb.retrieval_log import RetrievalLogStore


def test_normalize_category_six_labels():
    """六类标签归一化 + 越狱拦截 + 未知回退 kb_question。"""
    # kb_question
    assert _normalize_category("kb_question") == "kb_question"
    assert _normalize_category("知识库") == "kb_question"
    # faq
    assert _normalize_category("faq") == "faq"
    assert _normalize_category("常见问题") == "faq"
    # smalltalk
    assert _normalize_category("smalltalk") == "smalltalk"
    assert _normalize_category("闲聊") == "smalltalk"
    # ticket_intent
    assert _normalize_category("ticket_intent") == "ticket_intent"
    assert _normalize_category("工单") == "ticket_intent"
    # human_request
    assert _normalize_category("human_request") == "human_request"
    assert _normalize_category("转人工") == "human_request"
    # out_of_scope（越狱/无关）
    assert _normalize_category("out_of_scope") == "out_of_scope"
    assert _normalize_category("越狱") == "out_of_scope"


def test_normalize_category_unknown_falls_back():
    """分类失败/乱码 → 默认 kb_question（不影响检索链路）。"""
    assert _normalize_category("") == "kb_question"
    assert _normalize_category("随机乱码xyz") == "kb_question"
    assert _normalize_category("Hello there") == "kb_question"


def test_normalize_category_case_insensitive():
    assert _normalize_category("KB_QUESTION") == "kb_question"
    assert _normalize_category("SmallTalk") == "smalltalk"


# ---------- 检索日志 ----------

def test_retrieval_log_roundtrip(tmp_path):
    store = RetrievalLogStore(str(tmp_path / "r.db"))
    store.record(
        kb_id="default",
        question="采购招投标门槛",
        rewritten="招投标金额门槛",
        rerank_mode="llm",
        answerable=True,
        evidence_score=0.75,
        escalate=False,
        attempts=1,
        latency_ms={"retrieve": 10.0, "rerank": 5.0},
        hits=[{"chunk_id": "c1", "text": "内容", "score": 0.8, "metadata": {"doc_title": "采购制度"}}],
        final_hits=[{"chunk_id": "c1", "text": "内容", "rerank_score": 8.0, "metadata": {"doc_title": "采购制度"}}],
        faithfulness=8,
        answer="超过5万须招投标",
    )
    entries = store.recent()
    assert len(entries) == 1
    e = entries[0]
    assert e["question"] == "采购招投标门槛"
    assert e["rewritten"] == "招投标金额门槛"
    assert e["answerable"] is True
    assert e["evidence_score"] == 0.75
    assert e["faithfulness"] == 8
    # hits JSON 解析
    assert e["hits"][0]["chunk_id"] == "c1"
    assert e["hits"][0]["doc_title"] == "采购制度"
    assert e["final_hits"][0]["rerank_score"] == 8.0


def test_retrieval_log_escalate_and_count(tmp_path):
    store = RetrievalLogStore(str(tmp_path / "r.db"))
    store.record(kb_id="default", question="q1", escalate=True, answerable=False)
    store.record(kb_id="default", question="q2", escalate=False, answerable=True)
    assert store.count() == 2
    # escalate 记录正确
    escalated = [e for e in store.recent() if e["escalate"]]
    assert len(escalated) == 1
    assert escalated[0]["question"] == "q1"
