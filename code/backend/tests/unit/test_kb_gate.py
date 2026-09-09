"""gate 可回答性门槛回归测试（收尾第 1/4 步）。

覆盖 P0 修复的 gate 证据分字段 + P1 的阈值关闭态：
  1. rerank_score 归一化（0-10 → 0-1）
  2. 余弦 score 回退
  3. BM25-only 命中中性分（不误判）
  4. 阈值=0 关闭态直接 pass（空检索也不 escalate）
"""
from __future__ import annotations

from services.kb.qa_graph import (
    _answerability_threshold,
    _evidence_score,
    build_qa_graph,
    run_qa,
)
from services.kb.vector_store import VectorStore
from tests.mocks import FakeEmbedding, FakeLLM


def test_evidence_score_uses_rerank_score():
    """rerank_score 优先，且 0-10 量纲归一到 0-1。"""
    assert _evidence_score({"rerank_score": 8}) == 0.8
    assert _evidence_score({"rerank_score": 10}) == 1.0
    assert _evidence_score({"rerank_score": 3}) == 0.3


def test_evidence_score_falls_back_to_cosine():
    """无 rerank_score 时回退余弦 score（0-1）。"""
    assert _evidence_score({"score": 0.7}) == 0.7
    # 有 rerank_score 时忽略余弦
    assert _evidence_score({"rerank_score": 5, "score": 0.9}) == 0.5


def test_evidence_score_bm25_only_neutral():
    """BM25-only 命中（无向量分）给中性分 0.5，不按 0 分误判答不了。"""
    assert _evidence_score({"score": 0.0}) == 0.5
    assert _evidence_score({"score": None}) == 0.5
    assert _evidence_score({}) == 0.5


def test_threshold_zero_means_disabled(monkeypatch, tmp_path):
    """阈值=0（默认关闭）时空检索也 pass，不触发 escalate。"""
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    assert _answerability_threshold() == 0.0

    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    llm = FakeLLM(route={"客服意图分类器": "kb_question"})
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)
    result = run_qa(graph, question="随便问问", kb_id="default")
    # 阈值关闭：空库不转人工，而是诚实"未检索到"
    assert result["escalate"] is False
    assert "未检索到" in result["answer"]


def test_threshold_enabled_escalates_on_empty(monkeypatch, tmp_path):
    """阈值开启（如 0.5）时，空检索证据不足 → 两次降级后 escalate。"""
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0.5")
    assert _answerability_threshold() == 0.5

    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    llm = FakeLLM(route={"客服意图分类器": "kb_question"})
    graph = build_qa_graph(llm=llm, embeddings=emb, vector_store=store, top_k=3)
    result = run_qa(graph, question="库是空的随便问", kb_id="default")
    assert result["escalate"] is True
