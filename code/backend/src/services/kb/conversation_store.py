"""会话与消息模型（客服改造第 8 项）：人工协作闭环的地基。

conversations：会话状态机（AI 服务中 → 待接入 → 人工服务中 → 已结束）
messages：会话内消息（user / assistant / agent 三种角色）

复用现有 thread_id 体系：kb 问答的 thread_id 就是会话的 thread_id，
转人工后坐席在"客服工作台"里领取、回复、结束。
"""
from __future__ import annotations

import datetime as _dt
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 会话状态
STATUS_AI = "ai"           # AI 服务中
STATUS_WAITING = "waiting"  # 待接入（已转人工，等坐席领取）
STATUS_HUMAN = "human"      # 人工服务中（坐席已领取）
STATUS_CLOSED = "closed"    # 已结束


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class ConversationStore:
    """会话与消息（SQLite）。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id       TEXT NOT NULL,
                kb_id           TEXT NOT NULL DEFAULT 'default',
                visitor_id      TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'ai',
                transfer_reason TEXT NOT NULL DEFAULT '',
                agent_id        TEXT NOT NULL DEFAULT '',
                unread_count    INTEGER NOT NULL DEFAULT 0,
                tag             TEXT NOT NULL DEFAULT '',
                created_at      TEXT DEFAULT (datetime('now')),
                updated_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                created_at      TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_conv_status ON conversations(status);
            """
        )
        self._conn.commit()

    # ---------- 会话 ----------
    def get_or_create(
        self, thread_id: str, *, kb_id: str = "default", visitor_id: str = ""
    ) -> dict[str, Any]:
        """按 thread_id 取会话，不存在则建。返回会话 dict。"""
        row = self._conn.execute(
            "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
            "agent_id, unread_count, tag, created_at, updated_at "
            "FROM conversations WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
        if row:
            return self._row_to_dict(row)
        cur = self._conn.execute(
            "INSERT INTO conversations(thread_id, kb_id, visitor_id) VALUES(?,?,?)",
            (thread_id, kb_id, visitor_id),
        )
        self._conn.commit()
        return self.get_or_create(thread_id, kb_id=kb_id, visitor_id=visitor_id)

    def update_status(self, conv_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE conversations SET status=?, updated_at=? WHERE id=?",
            (status, _utc_now_iso(), conv_id),
        )
        self._conn.commit()

    def transfer_to_human(self, conv_id: int, reason: str) -> None:
        """转人工：状态 → waiting，记录原因，清零未读。"""
        self._conn.execute(
            "UPDATE conversations SET status=?, transfer_reason=?, unread_count=0, "
            "updated_at=? WHERE id=?",
            (STATUS_WAITING, reason, _utc_now_iso(), conv_id),
        )
        self._conn.commit()

    def claim(self, conv_id: int, agent_id: str) -> bool:
        """坐席领取会话：waiting → human。已被别人领走返回 False。"""
        cur = self._conn.execute(
            "UPDATE conversations SET status=?, agent_id=?, updated_at=? "
            "WHERE id=? AND status=?",
            (STATUS_HUMAN, agent_id, _utc_now_iso(), conv_id, STATUS_WAITING),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def close(self, conv_id: int) -> None:
        self.update_status(conv_id, STATUS_CLOSED)

    def list_by_status(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """按状态列会话（工作台：待接入池 / 人工服务中）。"""
        if status:
            rows = self._conn.execute(
                "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
                "agent_id, unread_count, tag, created_at, updated_at "
                "FROM conversations WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
                "agent_id, unread_count, tag, created_at, updated_at "
                "FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def set_tag(self, conv_id: int, tag: str) -> None:
        self._conn.execute("UPDATE conversations SET tag=? WHERE id=?", (tag, conv_id))
        self._conn.commit()

    # ---------- 消息 ----------
    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO messages(conversation_id, role, content) VALUES(?,?,?)",
            (conversation_id, role, content),
        )
        self._conn.commit()
        # 访客消息则未读 +1（坐席视角）
        if role == "user":
            self._conn.execute(
                "UPDATE conversations SET unread_count=unread_count+1, updated_at=? WHERE id=?",
                (_utc_now_iso(), conversation_id),
            )
            self._conn.commit()
        return cur.lastrowid

    def list_messages(self, conversation_id: int, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [
            {"role": r[0], "content": r[1], "created_at": r[2]} for r in reversed(rows)
        ]

    @staticmethod
    def _row_to_dict(r: tuple) -> dict[str, Any]:
        return {
            "id": r[0],
            "thread_id": r[1],
            "kb_id": r[2],
            "visitor_id": r[3],
            "status": r[4],
            "transfer_reason": r[5],
            "agent_id": r[6],
            "unread_count": r[7],
            "tag": r[8],
            "created_at": r[9],
            "updated_at": r[10],
        }
