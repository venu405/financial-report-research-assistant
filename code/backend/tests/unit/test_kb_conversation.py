"""会话状态机回归测试（收尾第 4 步）。

覆盖 P1 修复：
  1. transfer 踢坐席场景（human 状态不覆盖）
  2. claim 原子性（已领取不再被他人领走）
  3. thread_id 唯一性（并发建不出重复会话）
"""
from __future__ import annotations

import pytest

from services.kb.conversation_store import (
    STATUS_AI,
    STATUS_CLOSED,
    STATUS_HUMAN,
    ConversationStore,
)


def test_transfer_does_not_kick_agent(tmp_path):
    """已转人工（human）后再次 escalate 不能把坐席踢出（状态回 waiting）。"""
    store = ConversationStore(str(tmp_path / "c.db"))
    conv = store.get_or_create("t1")
    assert conv["status"] == STATUS_AI
    # ai -> waiting
    assert store.transfer_to_human(conv["id"], "门槛两次不过") is True
    # waiting -> human（坐席领取）
    assert store.claim(conv["id"], "agent-1") is True
    # human 状态再次 transfer 不覆盖
    assert store.transfer_to_human(conv["id"], "再次转人工") is False
    c = store.get(conv["id"])
    assert c["status"] == STATUS_HUMAN
    assert c["agent_id"] == "agent-1"


def test_claim_atomic(tmp_path):
    """claim 只在 waiting 态生效，第二个坐席抢不到。"""
    store = ConversationStore(str(tmp_path / "c.db"))
    conv = store.get_or_create("t1")
    store.transfer_to_human(conv["id"], "转人工")
    assert store.claim(conv["id"], "agent-1") is True
    assert store.claim(conv["id"], "agent-2") is False
    assert store.get(conv["id"])["agent_id"] == "agent-1"
    assert store.get(conv["id"])["first_response_at"] is None
    store.add_message(conv["id"], "agent", "您好，我来处理")
    assert store.get(conv["id"])["first_response_at"] is not None


def test_thread_id_unique(tmp_path):
    """同 thread_id 的 get_or_create 返回同一会话（不建重复）。"""
    store = ConversationStore(str(tmp_path / "c.db"))
    c1 = store.get_or_create("t1", visitor_id="v1")
    c2 = store.get_or_create("t1", visitor_id="v1")
    assert c1["id"] == c2["id"]
    # 列表里只有一条
    all_conv = store.list_by_status()
    assert len([c for c in all_conv if c["thread_id"] == "t1"]) == 1


def test_thread_cannot_be_reused_by_another_owner(tmp_path):
    store = ConversationStore(str(tmp_path / "c.db"))
    store.get_or_create("t1", visitor_id="owner-1")

    with pytest.raises(PermissionError, match="不属于"):
        store.get_or_create("t1", visitor_id="owner-2")


def test_recent_model_history_includes_agent_as_assistant(tmp_path):
    store = ConversationStore(str(tmp_path / "c.db"))
    conv = store.get_or_create("t1", visitor_id="owner-1")
    store.add_message(conv["id"], "user", "我的公司是星河科技")
    store.add_message(conv["id"], "assistant", "好的")
    store.add_message(conv["id"], "agent", "人工补充说明")

    assert store.recent_model_history(conv["id"]) == [
        {"role": "user", "content": "我的公司是星河科技"},
        {"role": "assistant", "content": "好的"},
        {"role": "assistant", "content": "人工补充说明"},
    ]


def test_delete_owned_removes_conversation_and_messages(tmp_path):
    store = ConversationStore(str(tmp_path / "c.db"))
    conv = store.get_or_create("t1", visitor_id="owner-1")
    store.add_message(conv["id"], "user", "需要删除")

    assert store.delete_owned("t1", "owner-2") is False
    assert store.delete_owned("t1", "owner-1") is True
    assert store.get_by_thread("t1") is None
    assert store.list_messages(conv["id"]) == []


def test_close_then_transfer_noop(tmp_path):
    """已关闭会话不再被 transfer 复活为 waiting。"""
    store = ConversationStore(str(tmp_path / "c.db"))
    conv = store.get_or_create("t1")
    store.transfer_to_human(conv["id"], "转人工")
    store.claim(conv["id"], "agent-1")
    store.close(conv["id"])
    assert store.get(conv["id"])["status"] == STATUS_CLOSED
    assert store.transfer_to_human(conv["id"], "又转") is False
    assert store.get(conv["id"])["status"] == STATUS_CLOSED


def test_conversation_stats_include_total(tmp_path):
    store = ConversationStore(str(tmp_path / "c.db"))
    store.get_or_create("t1")
    second = store.get_or_create("t2")
    store.transfer_to_human(second["id"], "需要人工")

    stats = store.stats()
    assert stats["total"] == 2
    assert stats["ai"] == 1
    assert stats["waiting"] == 1
