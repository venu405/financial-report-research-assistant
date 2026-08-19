"""满意度评价闭环（客服改造第 7 项）。

每条 AI 回答可 👍/👎 + 可选文字反馈，落库关联到该次检索日志（按 question 定位）。
差评自动标记会话，供人工复盘（管理接口查差评列表）。
"""
from __future__ import annotations

import datetime as _dt
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class FeedbackStore:
    """满意度反馈（SQLite）。rating: 1=👍 满意 / 0=👎 不满意。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT NOT NULL,
                kb_id       TEXT NOT NULL DEFAULT 'default',
                question    TEXT NOT NULL DEFAULT '',
                answer      TEXT NOT NULL DEFAULT '',
                rating      INTEGER NOT NULL DEFAULT 1,
                comment     TEXT NOT NULL DEFAULT ''
            );
            """
        )
        self._conn.commit()

    def record(
        self,
        *,
        kb_id: str,
        question: str,
        answer: str,
        rating: int,
        comment: str = "",
    ) -> int:
        """记一条反馈。失败不阻断主流程。"""
        try:
            cur = self._conn.execute(
                "INSERT INTO feedback(ts, kb_id, question, answer, rating, comment) "
                "VALUES(?,?,?,?,?,?)",
                (_utc_now_iso(), kb_id, question, answer[:500], 1 if rating else 0, comment[:500]),
            )
            self._conn.commit()
            return cur.lastrowid
        except Exception as exc:
            logger.warning("满意度反馈写入失败: %s", exc)
            return 0

    def recent(self, limit: int = 100, *, negative_only: bool = False) -> list[dict[str, Any]]:
        """最近反馈；negative_only=True 只看差评（人工复盘用）。"""
        sql = "SELECT ts, kb_id, question, answer, rating, comment FROM feedback"
        if negative_only:
            sql += " WHERE rating = 0"
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self._conn.execute(sql, (limit,)).fetchall()
        return [
            {
                "ts": r[0],
                "kb_id": r[1],
                "question": r[2],
                "answer": r[3],
                "rating": r[4],
                "comment": r[5],
            }
            for r in rows
        ]

    def count(self, *, negative_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM feedback"
        if negative_only:
            sql += " WHERE rating = 0"
        row = self._conn.execute(sql).fetchone()
        return row[0] if row else 0
