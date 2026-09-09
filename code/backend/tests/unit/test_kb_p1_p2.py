from __future__ import annotations

import sqlite3

import pytest

from services.kb.alert_store import AlertStore
from services.kb.governance_store import GovernanceStore
from services.kb.privacy import PrivacyStore, redact_text, run_retention_cleanup
from services.kb.source_store import SourceStore


def _snapshot(doc_id: str = "d1", text: str = "正文") -> dict:
    return {
        "ids": [f"{doc_id}-0"],
        "documents": [text],
        "metadatas": [{"doc_id": doc_id, "kb_id": "default", "chunk_index": 0}],
        "embeddings": [[0.1, 0.2]],
    }


def test_document_governance_workflow_and_rollback(tmp_path):
    store = GovernanceStore(tmp_path / "governance.db")
    first = store.create_version(
        doc_id="d1", kb_id="default", title="制度", snapshot=_snapshot(),
        created_by="u1", state="draft", original_name="制度.pdf",
        original_path="document_originals/d1/file.pdf",
    )
    assert first["version_no"] == 1
    assert first["original_name"] == "制度.pdf"
    assert first["original_path"] == "document_originals/d1/file.pdf"
    listed = store.list_versions("d1")[0]
    assert listed["has_original"] is True
    assert "original_path" not in listed
    assert store.transition(first["id"], "review", actor="u1")["state"] == "review"
    assert store.transition(first["id"], "approved", actor="supervisor")["state"] == "approved"
    assert store.transition(first["id"], "published", actor="supervisor")["state"] == "published"

    rolled = store.create_rollback(first["id"], actor="supervisor")
    assert rolled and rolled["version_no"] == 2
    assert rolled["state"] == "published"
    assert rolled["snapshot"]["documents"] == ["正文"]
    assert rolled["original_path"] == "document_originals/d1/file.pdf"

    with pytest.raises(ValueError, match="不可"):
        store.transition(first["id"], "review", actor="u1")


def test_privacy_consent_redaction_and_policy(tmp_path):
    store = PrivacyStore(tmp_path / "privacy.db")
    owner = "user:u1"
    with pytest.raises(PermissionError):
        store.set_memory(owner, "company", "星河科技")
    store.set_consent(owner, True)
    memory = store.set_memory(owner, "contact", "电话 13800138000，邮箱 a@example.com")
    assert "13800138000" not in memory["value"]
    assert "a@example.com" not in memory["value"]
    assert len(store.list_memory(owner)) == 1
    store.set_consent(owner, False)
    assert store.list_memory(owner) == []

    updated = store.update_policy({"conversation_days": 30})
    assert updated["conversation_days"] == 30
    with pytest.raises(ValueError):
        store.update_policy({"conversation_days": 0})
    assert "sk_abcdefghijklmnopqrstuvwxyz" not in redact_text("sk_abcdefghijklmnopqrstuvwxyz")


def test_retention_cleanup_dry_run_and_execute(tmp_path):
    db = tmp_path / "kb_feedback.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE feedback (id INTEGER PRIMARY KEY, ts TEXT, rating INTEGER)")
    conn.execute("INSERT INTO feedback(ts,rating) VALUES('2020-01-01',0)")
    conn.commit()
    conn.close()
    policy = PrivacyStore.DEFAULT_POLICY | {"feedback_days": 30}

    assert run_retention_cleanup(tmp_path, policy, dry_run=True)["feedback"] == 1
    assert run_retention_cleanup(tmp_path, policy, dry_run=False)["feedback"] == 1
    with sqlite3.connect(db) as check:
        assert check.execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0


def test_retention_cleanup_removes_expired_memory(tmp_path):
    privacy = PrivacyStore(tmp_path / "kb_privacy.db")
    privacy.set_consent("visitor:1", True)
    privacy.set_memory("visitor:1", "company", "星河科技")
    with sqlite3.connect(tmp_path / "kb_privacy.db") as conn:
        conn.execute("UPDATE user_memory SET expires_at='2020-01-01T00:00:00+00:00'")
        conn.commit()

    result = run_retention_cleanup(tmp_path, privacy.policy(), dry_run=False)
    assert result["memory"] == 1
    assert privacy.list_memory("visitor:1") == []


def test_retention_cleanup_removes_stale_active_conversation_and_checkpoints(tmp_path):
    with sqlite3.connect(tmp_path / "kb_conversations.db") as conn:
        conn.execute(
            "CREATE TABLE conversations (id INTEGER PRIMARY KEY, thread_id TEXT, status TEXT, updated_at TEXT)"
        )
        conn.execute("CREATE TABLE messages (conversation_id INTEGER, content TEXT)")
        conn.execute("INSERT INTO conversations VALUES(1,'old-thread','human','2020-01-01')")
        conn.execute("INSERT INTO messages VALUES(1,'secret')")
    with sqlite3.connect(tmp_path / "kb_checkpoints.db") as conn:
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT, value TEXT)")
        conn.execute("CREATE TABLE writes (thread_id TEXT, value TEXT)")
        conn.execute("INSERT INTO checkpoints VALUES('old-thread','state')")
        conn.execute("INSERT INTO writes VALUES('old-thread','write')")

    result = run_retention_cleanup(tmp_path, PrivacyStore.DEFAULT_POLICY, dry_run=False)
    assert result["conversations"] == 1
    assert result["checkpoints"] == 2
    with sqlite3.connect(tmp_path / "kb_conversations.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    with sqlite3.connect(tmp_path / "kb_checkpoints.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM writes").fetchone()[0] == 0


def test_source_schedule_and_result(tmp_path):
    store = SourceStore(tmp_path / "sources.db")
    source = store.create(
        name="制度目录", kb_id="default", source_type="file",
        location=str(tmp_path / "rules.md"), interval_minutes=30,
    )
    assert [item["id"] for item in store.due()] == [source["id"]]
    assert store.claim(source["id"]) is True
    first_lease = store.get(source["id"])["lease_until"]
    assert store.renew(source["id"], lease_minutes=10) is True
    assert store.get(source["id"])["lease_until"] > first_lease
    assert store.claim(source["id"]) is False
    assert store.due() == []
    store.mark_result(source["id"], ok=True, content_hash="abc", doc_id="d1")
    updated = store.get(source["id"])
    assert updated and updated["last_status"] == "ok" and updated["doc_id"] == "d1"
    assert store.due() == []

    with pytest.raises(ValueError, match="不能为空"):
        store.create(
            name="", kb_id="default", source_type="file",
            location="rules.md", interval_minutes=30,
        )


def test_alert_dedup_and_ack(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    first = store.emit(
        dedupe_key="sla-1", alert_type="conversation_sla", severity="high",
        target="1", message="超时",
    )
    second = store.emit(
        dedupe_key="sla-1", alert_type="conversation_sla", severity="critical",
        target="1", message="仍然超时",
    )
    assert first == second
    assert store.count_open() == 1
    assert store.acknowledge(first, "supervisor") is True
    assert store.list()[0]["status"] == "acknowledged"
    assert store.resolve("sla-1") is True
    assert store.list()[0]["status"] == "resolved"
    reopened = store.emit(
        dedupe_key="sla-1", alert_type="conversation_sla", severity="high",
        target="1", message="再次超时",
    )
    assert reopened == first
    assert store.list()[0]["status"] == "open"
