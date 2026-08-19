"""轻量工单系统（客服改造第 10 项）。

tickets：编号发号器 + 状态流（待处理 → 处理中 → 已解决 → 已关闭）
ticket_progress：时间线，每次状态变更/备注留痕。

转人工时一键带会话上下文建单（标题=问题摘要，描述=最近消息）。
"""
from __future__ import annotations

import datetime as _dt
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATUS_PENDING = "pending"       # 待处理
STATUS_PROCESSING = "processing"  # 处理中
STATUS_RESOLVED = "resolved"     # 已解决
STATUS_CLOSED = "closed"         # 已关闭


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class TicketStore:
    """工单（SQLite）。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_no       TEXT NOT NULL UNIQUE,
                conversation_id INTEGER,
                title           TEXT NOT NULL DEFAULT '',
                description     TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'pending',
                assignee        TEXT NOT NULL DEFAULT '',
                created_at      TEXT DEFAULT (datetime('now')),
                updated_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS ticket_progress (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket_id   INTEGER NOT NULL,
                action      TEXT NOT NULL DEFAULT '',
                note        TEXT NOT NULL DEFAULT '',
                created_at  TEXT DEFAULT (datetime('now'))
            );
            """
        )
        self._conn.commit()

    def _next_no(self) -> str:
        """编号发号器：T+日期+三位序号（如 T20260819-001）。"""
        date = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
        row = self._conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE ticket_no LIKE ?", (f"T{date}-%",)
        ).fetchone()
        seq = (row[0] if row else 0) + 1
        return f"T{date}-{seq:03d}"

    def create(
        self,
        *,
        conversation_id: int | None = None,
        title: str = "",
        description: str = "",
    ) -> dict[str, Any]:
        """建单。返回 {ticket_id, ticket_no}。"""
        ticket_no = self._next_no()
        cur = self._conn.execute(
            "INSERT INTO tickets(ticket_no, conversation_id, title, description) "
            "VALUES(?,?,?,?)",
            (ticket_no, conversation_id, title[:200], description[:2000]),
        )
        self._conn.commit()
        tid = cur.lastrowid
        self.add_progress(tid, "create", "工单创建")
        return {"ticket_id": tid, "ticket_no": ticket_no}

    def add_progress(self, ticket_id: int, action: str, note: str = "") -> None:
        self._conn.execute(
            "INSERT INTO ticket_progress(ticket_id, action, note) VALUES(?,?,?)",
            (ticket_id, action, note),
        )
        self._conn.commit()

    def update_status(self, ticket_id: int, status: str, note: str = "") -> None:
        """变更状态 + 记时间线。"""
        self._conn.execute(
            "UPDATE tickets SET status=?, updated_at=? WHERE id=?",
            (status, _utc_now_iso(), ticket_id),
        )
        self._conn.commit()
        self.add_progress(ticket_id, status, note)

    def assign(self, ticket_id: int, assignee: str) -> None:
        self._conn.execute(
            "UPDATE tickets SET assignee=?, status=?, updated_at=? WHERE id=?",
            (assignee, STATUS_PROCESSING, _utc_now_iso(), ticket_id),
        )
        self._conn.commit()
        self.add_progress(ticket_id, "assign", f"指派给 {assignee}")

    def get(self, ticket_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id, ticket_no, conversation_id, title, description, status, "
            "assignee, created_at, updated_at FROM tickets WHERE id=?",
            (ticket_id,),
        ).fetchone()
        if not row:
            return None
        d = {
            "id": row[0], "ticket_no": row[1], "conversation_id": row[2],
            "title": row[3], "description": row[4], "status": row[5],
            "assignee": row[6], "created_at": row[7], "updated_at": row[8],
        }
        d["progress"] = self.list_progress(ticket_id)
        return d

    def list(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = (
            "SELECT id, ticket_no, conversation_id, title, description, status, "
            "assignee, created_at, updated_at FROM tickets"
        )
        params: tuple = ()
        if status:
            sql += " WHERE status=?"
            params = (status,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [
            {
                "id": r[0], "ticket_no": r[1], "conversation_id": r[2],
                "title": r[3], "description": r[4], "status": r[5],
                "assignee": r[6], "created_at": r[7], "updated_at": r[8],
            }
            for r in rows
        ]

    def list_progress(self, ticket_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT action, note, created_at FROM ticket_progress "
            "WHERE ticket_id=? ORDER BY id",
            (ticket_id,),
        ).fetchall()
        return [{"action": r[0], "note": r[1], "created_at": r[2]} for r in rows]
