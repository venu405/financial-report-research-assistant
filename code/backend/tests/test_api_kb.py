"""FastAPI TestClient 端点测试（收尾清单 P2-4）。

覆盖：匿名 ask、visitor ask（P0-1 回归）、越权 403、429 限流、流式 final 字段（P0-2 回归）。
用 mock 组件替换外部依赖（Ollama/Chroma/OpenAI），不碰真实服务。
"""
from __future__ import annotations

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
        self.llm_reasoning_effort = None
        self.kb_top_k = 3
        self.kb_chunk_size = 800
        self.kb_chunk_overlap = 100
        self.cors_origins = "*"
        self.admin_api_key = ""
        self.kb_embedding_mode = "local"
        self.app_env = "development"


class MockVectorStore:
    last_instance = None

    def __init__(self, *a, **k):
        self.docs = {}
        self.add_calls = []
        MockVectorStore.last_instance = self

    def migrate_default_kb_id(self):
        pass

    def search(self, *a, **k):
        return []

    def delete_kb(self, kb_id):
        return 0

    def list_docs(self, *a, **k):
        return [], 0

    def get_doc_ids(self, *a, **k):
        doc_id = a[0] if a else k.get("doc_id", "")
        return list(self.docs.get(doc_id, {}).get("ids", []))

    def get_doc_kb_id(self, *a, **k):
        doc_id = a[0] if a else k.get("doc_id", "")
        return self.docs.get(doc_id, {}).get("kb_id")

    def find_doc_by_hash(self, *a, **k):
        return None

    def add_chunks(
        self, *, embeddings, texts, doc_id, doc_title, source_type,
        chunk_indices, kb_id, content_hash=None, extra_metadata=None,
    ):
        ids = [f"{doc_id}-{index}" for index in chunk_indices]
        self.add_calls.append({"doc_title": doc_title, "texts": list(texts)})
        self.docs[doc_id] = {
            "ids": ids,
            "documents": list(texts),
            "embeddings": list(embeddings),
            "metadatas": list(extra_metadata or []),
            "doc_title": doc_title,
            "source_type": source_type,
            "kb_id": kb_id,
            "content_hash": content_hash,
        }
        return ids

    def delete_chunk_ids(self, chunk_ids, **kwargs):
        removed = 0
        for doc in self.docs.values():
            kept = [index for index in range(len(doc["ids"])) if doc["ids"][index] not in chunk_ids]
            removed += len(doc["ids"]) - len(kept)
            for key in ("ids", "documents", "embeddings", "metadatas"):
                doc[key] = [doc[key][index] for index in kept]
        return removed

    def list_kbs(self):
        return ["default", "internal"]

    def snapshot_doc(self, doc_id):
        doc = self.docs.get(doc_id, {})
        return {
            "ids": list(doc.get("ids", [])),
            "documents": list(doc.get("documents", [])),
            "metadatas": list(doc.get("metadatas", [])),
            "embeddings": list(doc.get("embeddings", [])),
        }

    def restore_doc_snapshot(self, snapshot):
        return len(snapshot.get("ids", []))


class MockReranker:
    def rerank(self, query, candidates, top_k):
        return candidates[:top_k]


class RecordingEmbedding(FakeEmbedding):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def embed_texts(self, texts):
        self.inputs.extend(texts)
        return super().embed_texts(texts)


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
    last_initial: dict | None = None

    def __init__(self, escalate: bool = False):
        self._escalate = escalate

    def invoke(self, initial, config=None):
        MockGraph.last_initial = initial
        return _result(self._escalate)

    def stream(self, initial, config=None, stream_mode=None):
        # 模拟 LangGraph 在多 stream_mode=["updates","custom"] 时 yield (kind, data) 元组
        if isinstance(stream_mode, list):
            yield ("updates", {"generate": _result(self._escalate)})
        else:
            yield {"generate": _result(self._escalate)}


def _make_client(
    monkeypatch, tmp_path, *, escalate: bool = False, rate_limit: int = 20, embedding=None
) -> TestClient:
    monkeypatch.setattr(
        main_mod.Configuration, "from_env",
        classmethod(lambda cls, overrides=None: MockConfig(tmp_path)),
    )
    embedding = embedding or FakeEmbedding()
    monkeypatch.setattr("services.kb.embeddings.EmbeddingClient", lambda **k: embedding)
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


def test_ask_passes_metadata_filters_to_graph(monkeypatch, tmp_path):
    client = _make_client(monkeypatch, tmp_path)
    filters = {"source_type": "pdf", "page_start": {"$gte": 2}}

    resp = client.post(
        "/kb/ask",
        json={"question": "第二页内容", "kb_id": "default", "metadata_filters": filters},
    )

    assert resp.status_code == 200
    assert MockGraph.last_initial is not None
    assert MockGraph.last_initial["metadata_filters"] == filters


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


def test_ingest_update_and_stage_use_original_filename_for_title_and_embedding(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore
    from services.kb.ingest import DocumentChunk

    prepared = []
    build_titles = []

    def fake_build_chunks(path, **kwargs):
        build_titles.append(kwargs.get("document_title"))
        chunk = DocumentChunk(
            text="财务正文",
            doc_id=kwargs["doc_id"],
            doc_title=path.stem,
            chunk_index=0,
            source_type=path.suffix.lstrip("."),
            kb_id=kwargs["kb_id"],
            metadata={},
        )
        prepared.append(chunk)
        return [chunk]

    monkeypatch.setattr("services.kb.ingest.build_chunks", fake_build_chunks)
    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("入库管理员", role="admin")
    auth.grant_access(user_id, "default")
    embedding = RecordingEmbedding()
    client = _make_client(monkeypatch, tmp_path, embedding=embedding)
    headers = {"X-Api-Token": token}

    uploaded = client.post(
        "/kb/ingest",
        headers=headers,
        files={"file": ("年度报告.pdf", b"placeholder", "application/pdf")},
    )
    assert uploaded.status_code == 200, uploaded.text
    doc_id = uploaded.json()["doc_id"]
    assert uploaded.json()["title"] == "年度报告"

    updated = client.put(
        f"/kb/docs/{doc_id}",
        headers=headers,
        files={"file": ("更新报告.pdf", b"placeholder-2", "application/pdf")},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["title"] == "更新报告"

    staged = client.post(
        f"/kb/docs/{doc_id}/versions",
        headers=headers,
        files={"file": ("暂存报告.pdf", b"placeholder-3", "application/pdf")},
    )
    assert staged.status_code == 200, staged.text
    assert staged.json()["title"] == "暂存报告"

    explicit = client.post(
        "/kb/ingest",
        headers=headers,
        data={"title": "手工财务标题"},
        files={"file": ("另一个报告.pdf", b"placeholder-4", "application/pdf")},
    )
    assert explicit.status_code == 200, explicit.text
    assert explicit.json()["title"] == "手工财务标题"

    assert [chunk.doc_title for chunk in prepared] == [
        "年度报告",
        "更新报告",
        "暂存报告",
        "手工财务标题",
    ]
    assert build_titles == ["年度报告", "更新报告", "暂存报告", "手工财务标题"]
    assert all("_upload_" not in text for text in embedding.inputs)
    assert all(not text.startswith("_upload_") for text in embedding.inputs)
    assert embedding.inputs[-1].startswith("手工财务标题\n")
    assert [call["doc_title"] for call in MockVectorStore.last_instance.add_calls] == [
        "年度报告",
        "更新报告",
        "手工财务标题",
    ]


def test_reload_restores_messages_and_blocks_other_visitor(monkeypatch, tmp_path):
    """刷新后可从服务端恢复消息，另一访客即使猜到 thread_id 也不能读取。"""
    client = _make_client(monkeypatch, tmp_path)
    visitor = "kbv_" + "a" * 32
    asked = client.post(
        "/kb/ask",
        json={"question": "记住我的合同编号", "kb_id": "default", "visitor_token": visitor},
    )
    assert asked.status_code == 200
    thread_id = asked.json()["thread_id"]

    restored = client.get(
        f"/kb/conversations/{thread_id}/messages",
        params={"visitor_token": visitor},
    )
    assert restored.status_code == 200
    assert [m["role"] for m in restored.json()["messages"]] == ["user", "assistant"]

    listed = client.get(
        "/kb/conversations",
        params={"kb_id": "default", "visitor_token": visitor},
    )
    assert listed.status_code == 200
    assert [c["thread_id"] for c in listed.json()["conversations"]] == [thread_id]

    stolen = client.get(
        f"/kb/conversations/{thread_id}/messages",
        params={"visitor_token": "kbv_" + "b" * 32},
    )
    assert stolen.status_code == 403

    deleted = client.delete(
        f"/kb/conversations/{thread_id}",
        params={"visitor_token": visitor},
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True


def test_human_conversation_supports_bidirectional_messages(monkeypatch, tmp_path):
    """转人工后访客继续发言，坐席回复落在同一条消息时间线。"""
    client = _make_client(monkeypatch, tmp_path, escalate=True)
    visitor = "kbv_" + "c" * 32
    asked = client.post(
        "/kb/ask",
        json={"question": "我要人工", "kb_id": "default", "visitor_token": visitor},
    )
    data = asked.json()
    assert data["status"] == "waiting"

    sent = client.post(
        f"/kb/conversations/{data['thread_id']}/messages",
        json={"content": "补充：订单号 A100", "visitor_token": visitor},
    )
    assert sent.status_code == 200
    restored = client.get(
        f"/kb/conversations/{data['thread_id']}/messages",
        params={"visitor_token": visitor},
    ).json()
    assert restored["messages"][-1]["content"] == "补充：订单号 A100"

    closed = client.post(
        f"/kb/conversations/{data['thread_id']}/close",
        params={"visitor_token": visitor},
    )
    assert closed.status_code == 200
    assert closed.json()["closed"] is True
    after_close = client.post(
        f"/kb/conversations/{data['thread_id']}/messages",
        json={"content": "还能继续吗", "visitor_token": visitor},
    )
    assert after_close.status_code == 409


def test_internal_kb_rejects_anonymous_when_visibility_enforced(monkeypatch, tmp_path):
    from services.kb.kb_meta_store import KbMetaStore

    KbMetaStore(tmp_path / "kb_meta.db").create("internal", "内部制度")
    monkeypatch.setenv("KB_ENFORCE_KB_VISIBILITY", "1")
    client = _make_client(monkeypatch, tmp_path)

    denied = client.post("/kb/ask", json={"question": "hi", "kb_id": "internal"})
    assert denied.status_code == 401


def test_agent_role_can_use_customer_service_queue(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    agent_id, agent_token = auth.create_user("一号坐席", role="agent")
    member_id, member_token = auth.create_user("普通成员", role="member")
    auth.grant_access(agent_id, "default")
    auth.grant_access(member_id, "default")
    client = _make_client(monkeypatch, tmp_path, escalate=True)
    client.post(
        "/kb/ask",
        json={
            "question": "转人工",
            "kb_id": "default",
            "visitor_token": "kbv_" + "d" * 32,
        },
    )

    allowed = client.get("/kb/agent/queue", headers={"X-Api-Token": agent_token})
    denied = client.get("/kb/agent/queue", headers={"X-Api-Token": member_token})
    assert allowed.status_code == 200
    assert len(allowed.json()["conversations"]) == 1
    assert denied.status_code == 403


def test_explicit_memory_consent_and_delete(monkeypatch, tmp_path):
    client = _make_client(monkeypatch, tmp_path)
    visitor = "kbv_" + "e" * 32
    before = client.get("/kb/memory", params={"visitor_token": visitor})
    assert before.status_code == 200
    assert before.json() == {"consent": False, "memories": []}

    denied = client.post(
        "/kb/memory", json={"visitor_token": visitor, "key": "company", "value": "星河科技"}
    )
    assert denied.status_code == 409
    consent = client.put(
        "/kb/memory/consent", json={"visitor_token": visitor, "enabled": True}
    )
    assert consent.status_code == 200
    saved = client.post(
        "/kb/memory",
        json={"visitor_token": visitor, "key": "contact", "value": "13800138000"},
    )
    assert saved.status_code == 200
    assert "13800138000" not in saved.json()["value"]
    memory_id = client.get("/kb/memory", params={"visitor_token": visitor}).json()["memories"][0]["id"]
    deleted = client.delete(f"/kb/memory/{memory_id}", params={"visitor_token": visitor})
    assert deleted.json()["deleted"] is True


def test_available_agent_is_auto_assigned_and_dashboard_available(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    agent_id, agent_token = auth.create_user("自动坐席", role="agent")
    supervisor_id, supervisor_token = auth.create_user("主管", role="supervisor")
    auth.grant_access(agent_id, "default")
    auth.grant_access(supervisor_id, "default")
    client = _make_client(monkeypatch, tmp_path, escalate=True)
    online = client.put(
        "/kb/agent/availability",
        headers={"X-Api-Token": agent_token},
        json={"available": True},
    )
    assert online.status_code == 200

    asked = client.post(
        "/kb/ask",
        json={
            "question": "需要人工",
            "kb_id": "default",
            "visitor_token": "kbv_" + "f" * 32,
        },
    )
    assert asked.status_code == 200
    assert asked.json()["status"] == "human"
    queue = client.get("/kb/agent/queue", headers={"X-Api-Token": agent_token})
    assert queue.json()["conversations"][0]["agent_id"] == agent_id
    dashboard = client.get(
        "/kb/operations/dashboard", headers={"X-Api-Token": supervisor_token}
    )
    assert dashboard.status_code == 200
    assert set(dashboard.json()) == {
        "generated_at", "conversations", "tickets", "feedback", "rag", "open_alerts"
    }


def test_draft_versions_require_identity_and_kb_access(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore
    from services.kb.governance_store import GovernanceStore

    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, user_token = auth.create_user("制度管理员", role="member")
    auth.grant_access(user_id, "secret")
    governance = GovernanceStore(tmp_path / "kb_governance.db")
    governance.create_version(
        doc_id="draft-only", kb_id="secret", title="未发布制度",
        snapshot={"ids": ["draft-only-0"], "documents": ["secret"],
                  "metadatas": [{"doc_id": "draft-only", "kb_id": "secret"}],
                  "embeddings": [[0.1]]},
        created_by=user_id,
    )
    client = _make_client(monkeypatch, tmp_path)

    assert client.get("/kb/docs/draft-only/versions").status_code == 401
    allowed = client.get(
        "/kb/docs/draft-only/versions", headers={"X-Api-Token": user_token}
    )
    assert allowed.status_code == 200
    assert allowed.json()["versions"][0]["title"] == "未发布制度"


def test_supervisor_cannot_review_version_from_another_kb(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore
    from services.kb.governance_store import GovernanceStore

    auth = AuthStore(tmp_path / "kb_users.db")
    supervisor_id, supervisor_token = auth.create_user("A库主管", role="supervisor")
    auth.grant_access(supervisor_id, "kb-a")
    version = GovernanceStore(tmp_path / "kb_governance.db").create_version(
        doc_id="b-doc", kb_id="kb-b", title="B库制度",
        snapshot={"ids": ["b-doc-0"], "documents": ["secret"],
                  "metadatas": [{"doc_id": "b-doc", "kb_id": "kb-b"}],
                  "embeddings": [[0.1]]},
        created_by="b-user", state="review",
    )
    client = _make_client(monkeypatch, tmp_path)
    denied = client.post(
        f"/kb/docs/versions/{version['id']}/review",
        headers={"X-Api-Token": supervisor_token},
        data={"action": "approve", "note": "不应成功"},
    )
    assert denied.status_code == 403


def test_ticket_endpoints_enforce_kb_isolation(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    finance_id, finance_token = auth.create_user("财务客服", role="agent")
    default_id, default_token = auth.create_user("默认客服", role="agent")
    supervisor_id, supervisor_token = auth.create_user("财务主管", role="supervisor")
    auth.grant_access(finance_id, "finance")
    auth.grant_access(default_id, "default")
    auth.grant_access(supervisor_id, "finance")
    client = _make_client(monkeypatch, tmp_path)

    created = client.post(
        "/kb/tickets", headers={"X-Api-Token": finance_token},
        json={"kb_id": "finance", "title": "财务问题", "description": "敏感内容"},
    )
    assert created.status_code == 200
    ticket_id = created.json()["ticket_id"]
    assert client.get(
        f"/kb/tickets/{ticket_id}", headers={"X-Api-Token": default_token}
    ).status_code == 403
    assert client.get(
        "/kb/tickets", headers={"X-Api-Token": default_token}
    ).json()["tickets"] == []
    offline = client.post(
        f"/kb/tickets/{ticket_id}/assign", headers={"X-Api-Token": supervisor_token},
        data={"assignee": finance_id},
    )
    assert offline.status_code == 409
    assert client.put(
        "/kb/agent/availability", headers={"X-Api-Token": finance_token},
        json={"available": True},
    ).status_code == 200
    assigned = client.post(
        f"/kb/tickets/{ticket_id}/assign", headers={"X-Api-Token": supervisor_token},
        data={"assignee": finance_id},
    )
    assert assigned.status_code == 200


def test_http_source_revalidates_redirect_host(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    _, admin_token = auth.create_user("管理员", role="admin")
    monkeypatch.setenv("KB_SYNC_ALLOWED_HOSTS", "allowed.example")
    client = _make_client(monkeypatch, tmp_path)
    created = client.post(
        "/kb/sources", headers={"X-Api-Token": admin_token},
        json={"name": "制度源", "kb_id": "default", "source_type": "http",
              "location": "https://allowed.example/rules.txt", "interval_minutes": 60},
    )
    assert created.status_code == 200

    import urllib.error

    redirect_blocked = {"value": False}

    class RejectingOpener:
        def __init__(self, handler):
            self.handler = handler

        def open(self, request, timeout):
            redirect_blocked["value"] = self.handler.redirect_request(
                request, None, 302, "Found", {}, "http://127.0.0.1/private"
            ) is None
            raise urllib.error.HTTPError(request.full_url, 302, "Found", {}, None)

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda handler: RejectingOpener(handler)
    )
    synced = client.post(
        f"/kb/sources/{created.json()['id']}/run",
        headers={"X-Api-Token": admin_token},
    )
    assert synced.status_code == 400
    assert redirect_blocked["value"] is True


def test_file_source_sync_requires_review_before_publish(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    _, admin_token = auth.create_user("管理员", role="admin")
    sync_root = tmp_path / "sync_sources"
    sync_root.mkdir()
    (sync_root / "rules.txt").write_text("采购必须经过审批。", encoding="utf-8")
    monkeypatch.setenv("KB_SYNC_ALLOWED_ROOT", str(sync_root))
    client = _make_client(monkeypatch, tmp_path)
    created = client.post(
        "/kb/sources", headers={"X-Api-Token": admin_token},
        json={"name": "采购制度", "kb_id": "default", "source_type": "file",
              "location": "rules.txt", "interval_minutes": 60},
    )
    synced = client.post(
        f"/kb/sources/{created.json()['id']}/run", headers={"X-Api-Token": admin_token}
    )
    assert synced.status_code == 200, synced.text
    assert synced.json()["status"] == "pending_review"
    version_id = synced.json()["version_id"]
    pending = client.get(
        "/kb/docs/versions/pending", headers={"X-Api-Token": admin_token}
    )
    assert pending.json()["versions"][0]["state"] == "review"
    approved = client.post(
        f"/kb/docs/versions/{version_id}/review", headers={"X-Api-Token": admin_token},
        data={"action": "approve", "note": "内容正确"},
    )
    assert approved.json()["state"] == "approved"
    published = client.post(
        f"/kb/docs/versions/{version_id}/publish", headers={"X-Api-Token": admin_token}
    )
    assert published.status_code == 200
    assert published.json()["state"] == "published"
