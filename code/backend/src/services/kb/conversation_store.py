"""会话与消息模型（客服改造第 8 项）：人工协作闭环的地基。

conversations：会话状态机（AI 服务中 → 待接入 → 人工服务中 → 已结束）
messages：会话内消息（user / assistant / agent 三种角色）

复用现有 thread_id 体系：kb 问答的 thread_id 就是会话的 thread_id，
转人工后坐席在"客服工作台"里领取、回复、结束。
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import secrets
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
                thread_id       TEXT NOT NULL UNIQUE,
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
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(conversations)")}
        migrations = {
            "priority": "TEXT NOT NULL DEFAULT 'normal'",
            "sla_due_at": "TEXT",
            "first_response_at": "TEXT",
            "last_customer_at": "TEXT",
        }
        for name, definition in migrations.items():
            if name not in columns:
                self._conn.execute(f"ALTER TABLE conversations ADD COLUMN {name} {definition}")
        self._conn.commit()

    # ---------- 会话 ----------
    def get_or_create(
        self, thread_id: str, *, kb_id: str = "default", visitor_id: str = ""
    ) -> dict[str, Any]:
        """按 thread_id 取会话，不存在则建。返回会话 dict。

        P2：并发下 INSERT 可能撞 UNIQUE(thread_id)，用 INSERT OR IGNORE 兜底，
        撞了就查回来（不会抛 IntegrityError，也不会建出重复会话）。
        """
        row = self._conn.execute(
            "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
            "agent_id, unread_count, tag, created_at, updated_at, priority, sla_due_at, "
            "first_response_at, last_customer_at "
            "FROM conversations WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
        if row:
            existing = self._row_to_dict(row)
            if visitor_id and existing["visitor_id"] and not secrets.compare_digest(
                visitor_id, existing["visitor_id"]
            ):
                raise PermissionError("会话不属于当前用户")
            if visitor_id and not existing["visitor_id"]:
                self._conn.execute(
                    "UPDATE conversations SET visitor_id=? WHERE id=?",
                    (visitor_id, existing["id"]),
                )
                self._conn.commit()
                existing["visitor_id"] = visitor_id
            return existing
        self._conn.execute(
            "INSERT OR IGNORE INTO conversations(thread_id, kb_id, visitor_id) VALUES(?,?,?)",
            (thread_id, kb_id, visitor_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
            "agent_id, unread_count, tag, created_at, updated_at, priority, sla_due_at, "
            "first_response_at, last_customer_at "
            "FROM conversations WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else {
            "id": 0, "thread_id": thread_id, "kb_id": kb_id, "visitor_id": visitor_id,
            "status": STATUS_AI, "transfer_reason": "", "agent_id": "",
            "unread_count": 0, "tag": "", "created_at": "", "updated_at": "",
            "priority": "normal", "sla_due_at": None, "first_response_at": None,
            "last_customer_at": None,
        }

    def get_by_thread(self, thread_id: str) -> dict[str, Any] | None:
        """按 thread_id 查会话。"""
        row = self._conn.execute(
            "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
            "agent_id, unread_count, tag, created_at, updated_at, priority, sla_due_at, "
            "first_response_at, last_customer_at "
            "FROM conversations WHERE thread_id=?",
            (thread_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_for_owner(self, owner_key: str, limit: int = 100) -> list[dict[str, Any]]:
        """列出某个登录用户或签名访客拥有的会话。"""
        rows = self._conn.execute(
            "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
            "agent_id, unread_count, tag, created_at, updated_at, priority, sla_due_at, "
            "first_response_at, last_customer_at "
            "FROM conversations WHERE visitor_id=? ORDER BY updated_at DESC LIMIT ?",
            (owner_key, limit),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete_owned(self, thread_id: str, owner_key: str) -> bool:
        """删除归属匹配的会话及其全部消息。"""
        row = self._conn.execute(
            "SELECT id FROM conversations WHERE thread_id=? AND visitor_id=?",
            (thread_id, owner_key),
        ).fetchone()
        if not row:
            return False
        self._conn.execute("DELETE FROM messages WHERE conversation_id=?", (row[0],))
        self._conn.execute("DELETE FROM conversations WHERE id=?", (row[0],))
        self._conn.commit()
        return True

    def update_status(self, conv_id: int, status: str) -> None:
        self._conn.execute(
            "UPDATE conversations SET status=?, updated_at=? WHERE id=?",
            (status, _utc_now_iso(), conv_id),
        )
        self._conn.commit()

    def transfer_to_human(self, conv_id: int, reason: str) -> bool:
        """转人工：ai → waiting。仅 ai 状态可转，human/closed 不覆盖（避免把坐席踢出）。

        返回是否成功转移（False = 已是人工/已关闭，不覆盖）。
        """
        minutes = max(1, int(os.getenv("KB_AGENT_SLA_MINUTES", "10")))
        due_at = (
            _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(minutes=minutes)
        ).isoformat()
        cur = self._conn.execute(
            "UPDATE conversations SET status=?, transfer_reason=?, unread_count=0, "
            "sla_due_at=?, updated_at=? WHERE id=? AND status=?",
            (STATUS_WAITING, reason, due_at, _utc_now_iso(), conv_id, STATUS_AI),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get(self, conv_id: int) -> dict[str, Any] | None:
        """按 id 查会话，不存在返回 None。"""
        row = self._conn.execute(
            "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
            "agent_id, unread_count, tag, created_at, updated_at, priority, sla_due_at, "
            "first_response_at, last_customer_at "
            "FROM conversations WHERE id=?",
            (conv_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def mark_read(self, conv_id: int) -> None:
        """清零会话未读数（坐席已读）。"""
        self._conn.execute(
            "UPDATE conversations SET unread_count=0 WHERE id=?", (conv_id,)
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
                "agent_id, unread_count, tag, created_at, updated_at, priority, sla_due_at, "
                "first_response_at, last_customer_at "
                "FROM conversations WHERE status=? ORDER BY updated_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
                "agent_id, unread_count, tag, created_at, updated_at, priority, sla_due_at, "
                "first_response_at, last_customer_at "
                "FROM conversations ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def set_tag(self, conv_id: int, tag: str) -> None:
        self._conn.execute("UPDATE conversations SET tag=? WHERE id=?", (tag, conv_id))
        self._conn.commit()

    # ---------- 消息 ----------
    def add_message(self, conversation_id: int, role: str, content: str) -> int:
        if role not in {"user", "assistant", "agent", "system"}:
            raise ValueError(f"非法消息角色: {role}")
        cur = self._conn.execute(
            "INSERT INTO messages(conversation_id, role, content) VALUES(?,?,?)",
            (conversation_id, role, content),
        )
        self._conn.commit()
        # 访客消息则未读 +1（坐席视角）
        if role == "user":
            self._conn.execute(
                "UPDATE conversations SET unread_count=unread_count+1,last_customer_at=?,updated_at=? WHERE id=?",
                (_utc_now_iso(), _utc_now_iso(), conversation_id),
            )
            self._conn.commit()
        elif role == "agent":
            self._conn.execute(
                "UPDATE conversations SET first_response_at=COALESCE(first_response_at,?),"
                "updated_at=? WHERE id=?",
                (_utc_now_iso(), _utc_now_iso(), conversation_id),
            )
            self._conn.commit()
        return cur.lastrowid

    def list_messages(self, conversation_id: int, limit: int = 200) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
            (conversation_id, limit),
        ).fetchall()
        return [
            {"id": r[0], "role": r[1], "content": r[2], "created_at": r[3]}
            for r in reversed(rows)
        ]

    def recent_model_history(
        self, conversation_id: int, limit: int = 20
    ) -> list[dict[str, str]]:
        """读取最近消息并转换成模型可接受的 user/assistant 历史。"""
        history: list[dict[str, str]] = []
        for message in self.list_messages(conversation_id, limit=limit):
            role = message["role"]
            if role == "user":
                history.append({"role": "user", "content": message["content"]})
            elif role in {"assistant", "agent"}:
                history.append({"role": "assistant", "content": message["content"]})
        return history

    def set_priority(self, conv_id: int, priority: str) -> bool:
        if priority not in {"low", "normal", "high", "urgent"}:
            raise ValueError("非法会话优先级")
        cur = self._conn.execute(
            "UPDATE conversations SET priority=?,updated_at=? WHERE id=?",
            (priority, _utc_now_iso(), conv_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def sla_breaches(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, thread_id, kb_id, visitor_id, status, transfer_reason, "
            "agent_id, unread_count, tag, created_at, updated_at, priority, sla_due_at, "
            "first_response_at, last_customer_at FROM conversations "
            "WHERE status IN ('waiting','human') AND first_response_at IS NULL "
            "AND sla_due_at IS NOT NULL AND sla_due_at < ?",
            (_utc_now_iso(),),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def active_load(self, agent_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE status='human' AND agent_id=?",
            (agent_id,),
        ).fetchone()
        return int(row[0] if row else 0)

    def stats(self) -> dict[str, int]:
        result = {"total": 0, "ai": 0, "waiting": 0, "human": 0, "closed": 0, "sla_breached": 0}
        for status, count in self._conn.execute(
            "SELECT status,COUNT(*) FROM conversations GROUP BY status"
        ):
            result[str(status)] = int(count)
            result["total"] += int(count)
        result["sla_breached"] = len(self.sla_breaches())
        return result

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
            "priority": r[11],
            "sla_due_at": r[12],
            "first_response_at": r[13],
            "last_customer_at": r[14],
        }
