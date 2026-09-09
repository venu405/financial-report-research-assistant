"""客服 SLA 与运行告警。"""
from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path
from typing import Any


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class AlertStore:
    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dedupe_key TEXT NOT NULL UNIQUE,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                target TEXT NOT NULL DEFAULT '',
                message TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                acknowledged_by TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_alert_status ON alerts(status,created_at DESC);
            """
        )
        self._conn.commit()

    def emit(self, *, dedupe_key: str, alert_type: str, severity: str, target: str, message: str) -> int:
        now = _now()
        self._conn.execute(
            "INSERT INTO alerts(dedupe_key,alert_type,severity,target,message,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(dedupe_key) DO UPDATE SET "
            "severity=excluded.severity,message=excluded.message,updated_at=excluded.updated_at,"
            "status=CASE WHEN alerts.status='resolved' THEN 'open' ELSE alerts.status END",
            (dedupe_key, alert_type, severity, target, message[:1000], now, now),
        )
        self._conn.commit()
        row = self._conn.execute("SELECT id FROM alerts WHERE dedupe_key=?", (dedupe_key,)).fetchone()
        return int(row[0])

    def list(self, *, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        sql = "SELECT id,alert_type,severity,target,message,status,created_at,updated_at,acknowledged_by FROM alerts"
        params: tuple[Any, ...] = ()
        if status:
            sql += " WHERE status=?"
            params = (status,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self._conn.execute(sql, (*params, limit)).fetchall()
        return [
            {"id": r[0], "type": r[1], "severity": r[2], "target": r[3], "message": r[4],
             "status": r[5], "created_at": r[6], "updated_at": r[7], "acknowledged_by": r[8]}
            for r in rows
        ]

    def acknowledge(self, alert_id: int, actor: str) -> bool:
        cur = self._conn.execute(
            "UPDATE alerts SET status='acknowledged',acknowledged_by=?,updated_at=? WHERE id=?",
            (actor, _now(), alert_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def resolve(self, dedupe_key: str) -> bool:
        cur = self._conn.execute(
            "UPDATE alerts SET status='resolved',updated_at=? "
            "WHERE dedupe_key=? AND status!='resolved'",
            (_now(), dedupe_key),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def resolve_missing(self, prefix: str, active_keys: set[str]) -> int:
        rows = self._conn.execute(
            "SELECT dedupe_key FROM alerts WHERE dedupe_key LIKE ? AND status!='resolved'",
            (f"{prefix}%",),
        ).fetchall()
        stale = [str(row[0]) for row in rows if str(row[0]) not in active_keys]
        if not stale:
            return 0
        placeholders = ",".join("?" for _ in stale)
        cur = self._conn.execute(
            f"UPDATE alerts SET status='resolved',updated_at=? "
            f"WHERE dedupe_key IN ({placeholders})",
            (_now(), *stale),
        )
        self._conn.commit()
        return cur.rowcount

    def count_open(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM alerts WHERE status='open'").fetchone()
        return int(row[0] if row else 0)
