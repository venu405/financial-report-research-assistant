"""快捷回复话术库（客服改造第 11 项）。

坐席常用话术，管理面板维护，工作台里一键插入。
会话标签复用 conversation_store.set_tag（无需单独表）。
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class QuickReplyStore:
    """快捷回复（SQLite）。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS quick_replies (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL,
                content     TEXT NOT NULL,
                created_at  TEXT DEFAULT (datetime('now'))
            );
            """
        )
        self._conn.commit()

    def add(self, title: str, content: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO quick_replies(title, content) VALUES(?,?)", (title, content)
        )
        self._conn.commit()
        return cur.lastrowid

    def list(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, title, content FROM quick_replies ORDER BY id"
        ).fetchall()
        return [{"id": r[0], "title": r[1], "content": r[2]} for r in rows]

    def delete(self, reply_id: int) -> int:
        cur = self._conn.execute("DELETE FROM quick_replies WHERE id=?", (reply_id,))
        self._conn.commit()
        return cur.rowcount
