"""FastAPI TestClient 端点测试（收尾清单 P2-4）。

覆盖：匿名 ask、visitor ask（P0-1 回归）、越权 403、429 限流、流式 final 字段（P0-2 回归）。
用 mock 组件替换外部依赖（Ollama/Chroma/OpenAI），不碰真实服务。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main as main_mod
from tests.mocks import FakeEmbedding, FakeLLM


class MockConfig:
    def __init__(self, tmp_path):
        self.kb_chroma_dir = str(tmp_path / "chroma")
        self.kb_ollama_host = "http://localhost:11434"
        self.kb_embedding_model = "bge-m3"
        self.kb_collection = "test_col"
        self.llm_api_key = "sk-test"
        self.llm_base_url = None
        self.llm_model_id = "test-model"
        self.kb_top_k = 3
        self.cors_origins = "*"
        self.admin_api_key = ""
        self.kb_embedding_mode = "local"


class MockVectorStore:
    def __init__(self, *a, **k):
        pass

    def migrate_default_kb_id(self):
        pass

    def search(self, *a, **k):
        return []

    def delete_kb(self, kb_id):
        return 0

    def list_docs(self, *a, **k):
        return [], 0

    def get_doc_ids(self, *a, **k):
        return []

    def get_doc_kb_id(self, *a, **k):
        return None


class MockReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


def _result(escalate: bool = False) -> dict:
    return {
        "answer": "测试回答",
        "citations": [],
        "contexts": [],
        "retries": 0,
        "score": 8,
        "escalate": escalate,
        "intent": "kb_question",
        "faq_hit": False,
        "rewritten": "",
        "recall_raw": [],
        "retrieval_attempts": 0,
    }


class MockGraph:
    def __init__(self, escalate: bool = False):
        self._escalate = escalate

    def invoke(self, initial, config=None):
        return _result(self._escalate)

    def stream(self, initial, config=None, stream_mode=None):
        # 模拟 LangGraph 在多 stream_mode=["updates","custom"] 时 yield (kind, data) 元组
        if isinstance(stream_mode, list):
            yield ("updates", {"generate": _result(self._escalate)})
        else:
            yield {"generate": _result(self._escalate)}


def _make_client(monkeypatch, tmp_path, *, escalate: bool = False, rate_limit: int = 20) -> TestClient:
    monkeypatch.setattr(
        main_mod.Configuration, "from_env",
        classmethod(lambda cls, overrides=None: MockConfig(tmp_path)),
    )
    monkeypatch.setattr("services.kb.embeddings.EmbeddingClient", lambda **k: FakeEmbedding())
    monkeypatch.setattr("services.kb.vector_store.VectorStore", MockVectorStore)
    monkeypatch.setattr("openai.OpenAI", lambda **k: FakeLLM(default="ok"))
    monkeypatch.setattr("services.kb.reranker.build_reranker", lambda *a, **k: MockReranker())
    monkeypatch.setattr("services.kb.qa_graph.build_qa_graph", lambda **k: MockGraph(escalate))
    monkeypatch.setenv("KB_ASK_RATE_LIMIT", str(rate_limit))
    app = main_mod.create_app()
    return TestClient(app)


def test_anonymous_ask_ok(monkeypatch, tmp_path):
    """不传 user_id 的匿名 ask → 200（读操作放行）。"""
    client = _make_client(monkeypatch, tmp_path)
    resp = client.post("/kb/ask", json={"question": "hi", "kb_id": "default"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "测试回答"


def test_visitor_ask_not_403(monkeypatch, tmp_path):
    """P0-1 回归：visitor-* 未注册身份不再 403，当匿名放行。"""
    client = _make_client(monkeypatch, tmp_path)
    resp = client.post(
        "/kb/ask", json={"question": "hi", "kb_id": "default", "user_id": "visitor-abc123"}
    )
    assert resp.status_code == 200
    assert resp.json()["answer"] == "测试回答"


def test_registered_user_without_access_403(monkeypatch, tmp_path):
    """已注册但无该库权限的用户 → 403（RBAC 仍然生效）。"""
    from services.kb.auth import AuthStore

    auth = AuthStore(str(tmp_path / "kb_users.db"))
    uid, _ = auth.create_user("bob")
    auth.grant_access(uid, "other")  # 只授 other，不授 default

    client = _make_client(monkeypatch, tmp_path)
    resp = client.post(
        "/kb/ask", json={"question": "hi", "kb_id": "default", "user_id": uid}
    )
    assert resp.status_code == 403


def test_ask_rate_limit_429(monkeypatch, tmp_path):
    """超过限流阈值 → 429。"""
    client = _make_client(monkeypatch, tmp_path, rate_limit=1)
    first = client.post("/kb/ask", json={"question": "hi", "kb_id": "default"})
    assert first.status_code == 200
    second = client.post("/kb/ask", json={"question": "hi", "kb_id": "default"})
    assert second.status_code == 429


def test_stream_final_has_conversation_id(monkeypatch, tmp_path):
    """P0-2 回归：escalate 时流式 final 事件应回传 conversation_id / status。"""
    client = _make_client(monkeypatch, tmp_path, escalate=True)
    resp = client.post(
        "/kb/ask/stream",
        json={"question": "库外问题", "kb_id": "default", "user_id": "visitor-xyz"},
    )
    assert resp.status_code == 200
    body = resp.text
    # 解析 SSE 事件，找 final
    events = []
    for chunk in body.split("\n\n"):
        line = chunk.strip()
        if line.startswith("data:"):
            import json

            events.append(json.loads(line[5:]))
    final = next(e for e in events if e.get("type") == "final")
    assert final["escalate"] is True
    assert final.get("conversation_id") is not None
    assert final.get("status") == "waiting"
