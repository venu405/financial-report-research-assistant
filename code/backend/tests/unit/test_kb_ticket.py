"""工单状态机回归测试（收尾第 1/4 步）。

覆盖 P1 修复：
  1. 状态白名单（非法状态拒绝）
  2. 合法流转表（倒流拒绝）
  3. 工单不存在返回 False
  4. 发号器并发不撞号（IntegrityError 重试）
"""
from __future__ import annotations

import sqlite3
import threading

import pytest

from services.kb.ticket_store import (
    _TRANSITIONS,
    STATUS_CLOSED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_RESOLVED,
    VALID_STATUS,
    TicketStore,
)


def test_valid_status_whitelist():
    assert STATUS_PENDING in VALID_STATUS
    assert STATUS_PROCESSING in VALID_STATUS
    assert STATUS_RESOLVED in VALID_STATUS
    assert STATUS_CLOSED in VALID_STATUS
    assert "bogus" not in VALID_STATUS


def test_transitions_no_backflow():
    # resolved 不能回 processing / pending
    assert STATUS_PROCESSING not in _TRANSITIONS[STATUS_RESOLVED]
    assert STATUS_PENDING not in _TRANSITIONS[STATUS_RESOLVED]
    # closed 不可再流转
    assert _TRANSITIONS[STATUS_CLOSED] == set()


def test_ticket_status_machine(tmp_path):
    store = TicketStore(str(tmp_path / "t.db"))
    t = store.create(title="标题", description="描述")
    tid = t["ticket_id"]
    # 正向流转
    assert store.update_status(tid, STATUS_PROCESSING) is True
    assert store.update_status(tid, STATUS_RESOLVED) is True
    assert store.update_status(tid, STATUS_CLOSED) is True
    # 非法状态
    with pytest.raises(ValueError):
        store.update_status(tid, "bogus")
    # 已关闭不可再流转
    with pytest.raises(ValueError):
        store.update_status(tid, STATUS_PROCESSING)
    # 不存在返回 False
    assert store.update_status(99999, STATUS_PROCESSING) is False


def test_assign_closed_ticket_rejected(tmp_path):
    store = TicketStore(str(tmp_path / "t.db"))
    t = store.create(title="x")
    tid = t["ticket_id"]
    store.update_status(tid, STATUS_RESOLVED)
    store.update_status(tid, STATUS_CLOSED)
    with pytest.raises(ValueError):
        store.assign(tid, "agent-1")


def test_ticket_create_validates_priority_and_due_hours(tmp_path):
    store = TicketStore(str(tmp_path / "t.db"))
    with pytest.raises(ValueError, match="优先级"):
        store.create(title="x", priority="invalid")
    with pytest.raises(ValueError, match="时限"):
        store.create(title="x", due_hours=0)
    with pytest.raises(ValueError, match="时限"):
        store.create(title="x", due_hours=8761)


def test_ticket_no_generator_concurrent(tmp_path):
    """发号器并发不撞号（COUNT/MAX + INSERT 竞态由 IntegrityError 重试兜底）。"""
    store = TicketStore(str(tmp_path / "t.db"))
    nos: list[str] = []
    lock = threading.Lock()

    def _create():
        r = store.create(title="并发")
        with lock:
            nos.append(r["ticket_no"])

    threads = [threading.Thread(target=_create) for _ in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # 20 个并发建单，编号必须唯一
    assert len(nos) == 20
    assert len(set(nos)) == 20


def test_ticket_stats_include_total(tmp_path):
    store = TicketStore(str(tmp_path / "t.db"))
    first = store.create(title="一", kb_id="finance")
    store.create(title="二")
    store.update_status(first["ticket_id"], STATUS_PROCESSING)

    stats = store.stats()
    assert stats["total"] == 2
    assert stats["pending"] == 1
    assert stats["processing"] == 1
    assert store.get(first["ticket_id"])["kb_id"] == "finance"


def test_legacy_ticket_kb_is_backfilled_from_conversation(tmp_path):
    ticket_db = tmp_path / "kb_tickets.db"
    conversation_db = tmp_path / "kb_conversations.db"
    with sqlite3.connect(conversation_db) as conn:
        conn.execute("CREATE TABLE conversations (id INTEGER PRIMARY KEY, kb_id TEXT)")
        conn.execute("INSERT INTO conversations VALUES(7,'finance')")
    store = TicketStore(str(ticket_db))
    created = store.create(conversation_id=7, title="历史", kb_id="default")

    # Re-opening runs the compatibility backfill used for pre-kb_id ticket rows.
    reopened = TicketStore(str(ticket_db))
    assert reopened.get(created["ticket_id"])["kb_id"] == "finance"
