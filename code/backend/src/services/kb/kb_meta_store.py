"""知识库元数据（客服改造第 15 项）：建/改/删库 + 名称描述。

FAQ 与文档都挂在库下，管理面板需要能建库、改名称描述、删库。
实际向量/文档数据仍由 VectorStore（chroma）管理，这里只存元数据。
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KbMetaStore:
    """知识库元数据（SQLite）。"""

    VALID_VISIBILITY = {"internal", "public"}

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS kb_meta (
                kb_id       TEXT PRIMARY KEY,
                name        TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                visibility  TEXT NOT NULL DEFAULT 'internal',
                created_at  TEXT DEFAULT (datetime('now'))
            );
            """
        )
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(kb_meta)")}
        if "visibility" not in columns:
            self._conn.execute(
                "ALTER TABLE kb_meta ADD COLUMN visibility TEXT NOT NULL DEFAULT 'internal'"
            )
        self._conn.commit()

    def create(
        self,
        kb_id: str,
        name: str,
        description: str = "",
        visibility: str = "internal",
    ) -> dict[str, Any]:
        """建库。已存在则抛 ValueError（kb_id 唯一）。"""
        exists = self._conn.execute(
            "SELECT 1 FROM kb_meta WHERE kb_id=?", (kb_id,)
        ).fetchone()
        if exists:
            raise ValueError(f"知识库 {kb_id} 已存在")
        if visibility not in self.VALID_VISIBILITY:
            raise ValueError(f"非法可见范围: {visibility}")
        self._conn.execute(
            "INSERT INTO kb_meta(kb_id, name, description, visibility) VALUES(?,?,?,?)",
            (kb_id, name, description, visibility),
        )
        self._conn.commit()
        return {
            "kb_id": kb_id,
            "name": name,
            "description": description,
            "visibility": visibility,
        }

    def update(
        self,
        kb_id: str,
        name: str | None = None,
        description: str | None = None,
        visibility: str | None = None,
    ) -> bool:
        """改名称/描述（只更新传入的字段）。"""
        if name is None and description is None and visibility is None:
            return False
        if visibility is not None and visibility not in self.VALID_VISIBILITY:
            raise ValueError(f"非法可见范围: {visibility}")
        if name is not None:
            self._conn.execute(
                "UPDATE kb_meta SET name=? WHERE kb_id=?", (name, kb_id)
            )
        if description is not None:
            self._conn.execute(
                "UPDATE kb_meta SET description=? WHERE kb_id=?", (description, kb_id)
            )
        if visibility is not None:
            self._conn.execute(
                "UPDATE kb_meta SET visibility=? WHERE kb_id=?", (visibility, kb_id)
            )
        self._conn.commit()
        return True

    def delete(self, kb_id: str) -> int:
        cur = self._conn.execute("DELETE FROM kb_meta WHERE kb_id=?", (kb_id,))
        self._conn.commit()
        return cur.rowcount

    def list(self) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT kb_id, name, description, visibility, created_at "
            "FROM kb_meta ORDER BY created_at"
        ).fetchall()
        return [
            {
                "kb_id": r[0],
                "name": r[1],
                "description": r[2],
                "visibility": r[3],
                "created_at": r[4],
            }
            for r in rows
        ]

    def get(self, kb_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT kb_id, name, description, visibility, created_at "
            "FROM kb_meta WHERE kb_id=?",
            (kb_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "kb_id": row[0],
            "name": row[1],
            "description": row[2],
            "visibility": row[3],
            "created_at": row[4],
        }

    def is_public(self, kb_id: str) -> bool:
        row = self._conn.execute(
            "SELECT visibility FROM kb_meta WHERE kb_id=?", (kb_id,)
        ).fetchone()
        return bool(row and row[0] == "public")

    def public_ids(self) -> set[str]:
        rows = self._conn.execute(
            "SELECT kb_id FROM kb_meta WHERE visibility='public'"
        ).fetchall()
        return {row[0] for row in rows}
