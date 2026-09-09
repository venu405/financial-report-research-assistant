"""文档版本与审核发布流程。"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

VALID_STATES = {"draft", "review", "approved", "published", "rejected"}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class GovernanceStore:
    """保存每个文档版本的向量快照和审批状态。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.Lock()
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS document_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                version_no INTEGER NOT NULL,
                title TEXT NOT NULL DEFAULT '',
                state TEXT NOT NULL DEFAULT 'draft',
                snapshot_json TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT '',
                reviewed_by TEXT NOT NULL DEFAULT '',
                review_note TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                published_at TEXT,
                original_name TEXT NOT NULL DEFAULT '',
                original_path TEXT NOT NULL DEFAULT '',
                UNIQUE(doc_id, version_no)
            );
            CREATE INDEX IF NOT EXISTS idx_doc_versions_doc
                ON document_versions(doc_id, version_no DESC);
            CREATE INDEX IF NOT EXISTS idx_doc_versions_state
                ON document_versions(state, updated_at DESC);
            """
        )
        columns = {
            row[1] for row in self._conn.execute("PRAGMA table_info(document_versions)")
        }
        if "original_name" not in columns:
            self._conn.execute(
                "ALTER TABLE document_versions ADD COLUMN original_name TEXT NOT NULL DEFAULT ''"
            )
        if "original_path" not in columns:
            self._conn.execute(
                "ALTER TABLE document_versions ADD COLUMN original_path TEXT NOT NULL DEFAULT ''"
            )
        self._conn.commit()

    def create_version(
        self,
        *,
        doc_id: str,
        kb_id: str,
        title: str,
        snapshot: dict[str, Any],
        created_by: str,
        state: str = "draft",
        review_note: str = "",
        original_name: str = "",
        original_path: str = "",
    ) -> dict[str, Any]:
        if state not in VALID_STATES:
            raise ValueError(f"非法文档状态: {state}")
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version_no), 0) FROM document_versions WHERE doc_id=?",
                (doc_id,),
            ).fetchone()
            version_no = int(row[0] or 0) + 1
            now = _now()
            cur = self._conn.execute(
                "INSERT INTO document_versions(doc_id,kb_id,version_no,title,state,"
                "snapshot_json,created_by,review_note,created_at,updated_at,published_at,"
                "original_name,original_path) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    doc_id,
                    kb_id,
                    version_no,
                    title[:300],
                    state,
                    json.dumps(snapshot, ensure_ascii=False),
                    created_by,
                    review_note[:1000],
                    now,
                    now,
                    now if state == "published" else None,
                    original_name[:300],
                    original_path[:2000],
                ),
            )
            self._conn.commit()
        return self.get(cur.lastrowid) or {}

    def get(self, version_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT id,doc_id,kb_id,version_no,title,state,snapshot_json,created_by,"
            "reviewed_by,review_note,created_at,updated_at,published_at,original_name,original_path "
            "FROM document_versions WHERE id=?",
            (version_id,),
        ).fetchone()
        return self._row(row) if row else None

    def list_versions(self, doc_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id,doc_id,kb_id,version_no,title,state,snapshot_json,created_by,"
            "reviewed_by,review_note,created_at,updated_at,published_at,original_name,original_path "
            "FROM document_versions WHERE doc_id=? ORDER BY version_no DESC",
            (doc_id,),
        ).fetchall()
        return [self._row(row, include_snapshot=False) for row in rows]

    def list_pending(self, kb_id: str | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT id,doc_id,kb_id,version_no,title,state,snapshot_json,created_by,"
            "reviewed_by,review_note,created_at,updated_at,published_at,original_name,original_path "
            "FROM document_versions WHERE state IN ('review','approved')"
        )
        params: tuple[Any, ...] = ()
        if kb_id:
            sql += " AND kb_id=?"
            params = (kb_id,)
        sql += " ORDER BY updated_at"
        return [self._row(row, include_snapshot=False) for row in self._conn.execute(sql, params)]

    def transition(
        self, version_id: int, target: str, *, actor: str, note: str = ""
    ) -> dict[str, Any] | None:
        transitions = {
            "draft": {"review"},
            "review": {"approved", "rejected"},
            "approved": {"published", "rejected"},
            "published": set(),
            "rejected": {"review"},
        }
        if target not in VALID_STATES:
            raise ValueError(f"非法文档状态: {target}")
        current = self.get(version_id)
        if not current:
            return None
        if target not in transitions[current["state"]]:
            raise ValueError(f"文档版本不可从 {current['state']} 流转到 {target}")
        now = _now()
        self._conn.execute(
            "UPDATE document_versions SET state=?,reviewed_by=?,review_note=?,"
            "updated_at=?,published_at=? WHERE id=?",
            (target, actor, note[:1000], now, now if target == "published" else None, version_id),
        )
        self._conn.commit()
        return self.get(version_id)

    def create_rollback(self, version_id: int, *, actor: str) -> dict[str, Any] | None:
        source = self.get(version_id)
        if not source:
            return None
        return self.create_version(
            doc_id=source["doc_id"],
            kb_id=source["kb_id"],
            title=source["title"],
            snapshot=source["snapshot"],
            created_by=actor,
            state="published",
            review_note=f"回滚自版本 v{source['version_no']}",
            original_name=source["original_name"],
            original_path=source["original_path"],
        )

    @staticmethod
    def _row(row: tuple[Any, ...], *, include_snapshot: bool = True) -> dict[str, Any]:
        result = {
            "id": row[0],
            "doc_id": row[1],
            "kb_id": row[2],
            "version_no": row[3],
            "title": row[4],
            "state": row[5],
            "created_by": row[7],
            "reviewed_by": row[8],
            "review_note": row[9],
            "created_at": row[10],
            "updated_at": row[11],
            "published_at": row[12],
            "original_name": row[13],
            "has_original": bool(row[14]),
        }
        if include_snapshot:
            result["snapshot"] = json.loads(row[6])
            result["original_path"] = row[14]
        return result
