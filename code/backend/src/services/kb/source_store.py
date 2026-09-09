"""外部知识数据源配置与定时同步状态（不包含客服渠道）。"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class SourceStore:
    VALID_TYPES = {"file", "http"}

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sync_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kb_id TEXT NOT NULL,
                source_type TEXT NOT NULL,
                location TEXT NOT NULL,
                interval_minutes INTEGER NOT NULL DEFAULT 60,
                enabled INTEGER NOT NULL DEFAULT 1,
                doc_id TEXT NOT NULL DEFAULT '',
                last_hash TEXT NOT NULL DEFAULT '',
                last_status TEXT NOT NULL DEFAULT 'never',
                last_error TEXT NOT NULL DEFAULT '',
                last_sync_at TEXT,
                next_sync_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                lease_until TEXT
            );
            """
        )
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(sync_sources)")}
        if "lease_until" not in columns:
            self._conn.execute("ALTER TABLE sync_sources ADD COLUMN lease_until TEXT")
        self._conn.commit()

    def create(self, *, name: str, kb_id: str, source_type: str, location: str, interval_minutes: int) -> dict[str, Any]:
        if not name.strip() or not kb_id.strip() or not location.strip():
            raise ValueError("数据源名称、知识库和位置不能为空")
        if source_type not in self.VALID_TYPES:
            raise ValueError(f"不支持的数据源类型: {source_type}")
        if interval_minutes < 5 or interval_minutes > 10080:
            raise ValueError("同步周期必须在 5 分钟到 7 天之间")
        now = _now()
        cur = self._conn.execute(
            "INSERT INTO sync_sources(name,kb_id,source_type,location,interval_minutes,next_sync_at,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (name.strip()[:200], kb_id.strip(), source_type, location.strip()[:2000], interval_minutes, now.isoformat(), now.isoformat()),
        )
        self._conn.commit()
        return self.get(cur.lastrowid) or {}

    def get(self, source_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM sync_sources WHERE id=?", (source_id,)).fetchone()
        return self._row(row) if row else None

    def list(self) -> list[dict[str, Any]]:
        return [self._row(row) for row in self._conn.execute("SELECT * FROM sync_sources ORDER BY id DESC")]

    def due(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT * FROM sync_sources WHERE enabled=1 AND next_sync_at<=? "
            "AND (lease_until IS NULL OR lease_until<=?) ORDER BY next_sync_at LIMIT ?",
            (_now().isoformat(), _now().isoformat(), limit),
        ).fetchall()
        return [self._row(row) for row in rows]

    def claim(self, source_id: int, *, lease_minutes: int = 5) -> bool:
        now = _now()
        lease_until = (now + dt.timedelta(minutes=lease_minutes)).isoformat()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute(
                "UPDATE sync_sources SET lease_until=? WHERE id=? AND enabled=1 "
                "AND (lease_until IS NULL OR lease_until<=?)",
                (lease_until, source_id, now.isoformat()),
            )
            self._conn.commit()
            return cur.rowcount > 0
        except Exception:
            self._conn.rollback()
            raise

    def claim_due(self, limit: int = 20, *, lease_minutes: int = 5) -> list[dict[str, Any]]:
        claimed: list[dict[str, Any]] = []
        for source in self.due(limit=limit):
            if self.claim(source["id"], lease_minutes=lease_minutes):
                current = self.get(source["id"])
                if current:
                    claimed.append(current)
        return claimed

    def renew(self, source_id: int, *, lease_minutes: int = 5) -> bool:
        lease_until = (_now() + dt.timedelta(minutes=lease_minutes)).isoformat()
        with sqlite3.connect(self._db_path) as connection:
            connection.execute("PRAGMA busy_timeout=5000")
            cur = connection.execute(
                "UPDATE sync_sources SET lease_until=? WHERE id=? AND lease_until IS NOT NULL",
                (lease_until, source_id),
            )
            return cur.rowcount > 0

    def mark_result(self, source_id: int, *, ok: bool, content_hash: str = "", doc_id: str = "", error: str = "") -> None:
        source = self.get(source_id)
        if not source:
            return
        now = _now()
        next_sync = now + dt.timedelta(minutes=source["interval_minutes"])
        self._conn.execute(
            "UPDATE sync_sources SET last_hash=CASE WHEN ?='' THEN last_hash ELSE ? END,"
            "doc_id=CASE WHEN ?='' THEN doc_id ELSE ? END,last_status=?,last_error=?,"
            "last_sync_at=?,next_sync_at=?,lease_until=NULL WHERE id=?",
            (content_hash, content_hash, doc_id, doc_id, "ok" if ok else "error", error[:1000],
             now.isoformat(), next_sync.isoformat(), source_id),
        )
        self._conn.commit()

    def delete(self, source_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM sync_sources WHERE id=?", (source_id,))
        self._conn.commit()
        return cur.rowcount > 0

    @staticmethod
    def _row(row: tuple[Any, ...]) -> dict[str, Any]:
        keys = ["id", "name", "kb_id", "source_type", "location", "interval_minutes", "enabled",
                "doc_id", "last_hash", "last_status", "last_error", "last_sync_at", "next_sync_at", "created_at",
                "lease_until"]
        return dict(zip(keys, row, strict=True))
