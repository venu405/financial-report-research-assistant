"""敏感信息脱敏、长期记忆授权与数据保留策略。"""
from __future__ import annotations

import datetime as dt
import re
import sqlite3
from pathlib import Path
from typing import Any

_PATTERNS = [
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[手机号已隐藏]"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[身份证号已隐藏]"),
    (re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[银行卡号已隐藏]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[邮箱已隐藏]"),
    (re.compile(r"\b(?:sk|kb)_[A-Za-z0-9_-]{16,}\b"), "[密钥已隐藏]"),
]


def redact_text(value: str) -> str:
    result = value
    for pattern, replacement in _PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class PrivacyStore:
    """显式同意后才保存的结构化长期记忆和保留期限配置。"""

    DEFAULT_POLICY = {
        "conversation_days": 365,
        "retrieval_days": 180,
        "feedback_days": 365,
        "audit_days": 730,
        "memory_days": 365,
        "ticket_days": 730,
    }

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_consent (
                owner_key TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_key TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_key, memory_key)
            );
            CREATE TABLE IF NOT EXISTS retention_policy (
                policy_key TEXT PRIMARY KEY,
                days INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        for key, days in self.DEFAULT_POLICY.items():
            self._conn.execute(
                "INSERT OR IGNORE INTO retention_policy(policy_key,days,updated_at) VALUES(?,?,?)",
                (key, days, _now()),
            )
        self._conn.commit()

    def set_consent(self, owner_key: str, enabled: bool) -> None:
        self._conn.execute(
            "INSERT INTO memory_consent(owner_key,enabled,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(owner_key) DO UPDATE SET enabled=excluded.enabled,updated_at=excluded.updated_at",
            (owner_key, 1 if enabled else 0, _now()),
        )
        if not enabled:
            self._conn.execute("DELETE FROM user_memory WHERE owner_key=?", (owner_key,))
        self._conn.commit()

    def has_consent(self, owner_key: str) -> bool:
        row = self._conn.execute(
            "SELECT enabled FROM memory_consent WHERE owner_key=?", (owner_key,)
        ).fetchone()
        return bool(row and row[0])

    def set_memory(
        self, owner_key: str, key: str, value: str, *, expires_days: int | None = None
    ) -> dict[str, Any]:
        if not self.has_consent(owner_key):
            raise PermissionError("用户尚未同意保存长期记忆")
        clean_key = re.sub(r"[^\w\u4e00-\u9fff.-]", "", key)[:64]
        if not clean_key:
            raise ValueError("记忆字段名不能为空")
        now = dt.datetime.now(dt.timezone.utc)
        days = expires_days or self.policy()["memory_days"]
        expires_at = (now + dt.timedelta(days=days)).isoformat()
        self._conn.execute(
            "INSERT INTO user_memory(owner_key,memory_key,memory_value,expires_at,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(owner_key,memory_key) DO UPDATE SET "
            "memory_value=excluded.memory_value,expires_at=excluded.expires_at,updated_at=excluded.updated_at",
            (owner_key, clean_key, redact_text(value)[:500], expires_at, now.isoformat(), now.isoformat()),
        )
        self._conn.commit()
        return {"key": clean_key, "value": redact_text(value)[:500], "expires_at": expires_at}

    def list_memory(self, owner_key: str) -> list[dict[str, Any]]:
        self.cleanup_expired_memory()
        rows = self._conn.execute(
            "SELECT id,memory_key,memory_value,expires_at,updated_at FROM user_memory "
            "WHERE owner_key=? ORDER BY id",
            (owner_key,),
        ).fetchall()
        return [
            {"id": r[0], "key": r[1], "value": r[2], "expires_at": r[3], "updated_at": r[4]}
            for r in rows
        ]

    def delete_memory(self, owner_key: str, memory_id: int) -> bool:
        cur = self._conn.execute(
            "DELETE FROM user_memory WHERE id=? AND owner_key=?", (memory_id, owner_key)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def cleanup_expired_memory(self) -> int:
        cur = self._conn.execute(
            "DELETE FROM user_memory WHERE expires_at IS NOT NULL AND expires_at < ?", (_now(),)
        )
        self._conn.commit()
        return cur.rowcount

    def policy(self) -> dict[str, int]:
        return {
            row[0]: int(row[1])
            for row in self._conn.execute("SELECT policy_key,days FROM retention_policy")
        }

    def update_policy(self, values: dict[str, int]) -> dict[str, int]:
        allowed = set(self.DEFAULT_POLICY)
        for key, value in values.items():
            if key not in allowed:
                raise ValueError(f"未知保留策略: {key}")
            days = int(value)
            if days < 1 or days > 3650:
                raise ValueError("保留期限必须在 1 到 3650 天之间")
            self._conn.execute(
                "UPDATE retention_policy SET days=?,updated_at=? WHERE policy_key=?",
                (days, _now(), key),
            )
        self._conn.commit()
        return self.policy()


def run_retention_cleanup(
    data_dir: str | Path, policy: dict[str, int], *, dry_run: bool = True
) -> dict[str, int]:
    """按白名单表执行保留期限清理；默认仅统计，不删除。"""
    root = Path(data_dir).resolve()
    now = dt.datetime.now(dt.timezone.utc)
    result: dict[str, int] = {}

    def cutoff(key: str) -> str:
        return (now - dt.timedelta(days=int(policy[key]))).isoformat()

    specs = [
        ("retrieval", root / "kb_retrieval_log.db", "retrieval_log", "ts", "retrieval_days", "1=1"),
        ("feedback", root / "kb_feedback.db", "feedback", "ts", "feedback_days", "1=1"),
        ("audit", root / "kb_audit.db", "audit_log", "ts", "audit_days", "1=1"),
        ("tickets", root / "kb_tickets.db", "tickets", "updated_at", "ticket_days", "status='closed'"),
    ]
    for label, path, table, time_col, policy_key, extra_where in specs:
        if not path.is_file():
            result[label] = 0
            continue
        conn = sqlite3.connect(path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if not exists:
                result[label] = 0
                continue
            where = f"{extra_where} AND datetime({time_col}) < datetime(?)"
            count = int(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", (cutoff(policy_key),)).fetchone()[0])
            result[label] = count
            if not dry_run and count:
                conn.execute(f"DELETE FROM {table} WHERE {where}", (cutoff(policy_key),))
                conn.commit()
        finally:
            conn.close()

    privacy_path = root / "kb_privacy.db"
    result["memory"] = 0
    if privacy_path.is_file():
        conn = sqlite3.connect(privacy_path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='user_memory'"
            ).fetchone()
            if exists:
                where = (
                    "(expires_at IS NOT NULL AND datetime(expires_at) < datetime(?)) OR "
                    "(expires_at IS NULL AND datetime(updated_at) < datetime(?))"
                )
                params = (_now(), cutoff("memory_days"))
                result["memory"] = int(
                    conn.execute(f"SELECT COUNT(*) FROM user_memory WHERE {where}", params).fetchone()[0]
                )
                if not dry_run and result["memory"]:
                    conn.execute(f"DELETE FROM user_memory WHERE {where}", params)
                    conn.commit()
        finally:
            conn.close()

    conv_path = root / "kb_conversations.db"
    result["conversations"] = 0
    if conv_path.is_file():
        conn = sqlite3.connect(conv_path)
        try:
            old_conversations = [
                (row[0], row[1])
                for row in conn.execute(
                    "SELECT id,thread_id FROM conversations "
                    "WHERE datetime(updated_at) < datetime(?)",
                    (cutoff("conversation_days"),),
                )
            ]
            old_ids = [row[0] for row in old_conversations]
            result["conversations"] = len(old_ids)
            if not dry_run and old_ids:
                placeholders = ",".join("?" for _ in old_ids)
                conn.execute(
                    f"DELETE FROM messages WHERE conversation_id IN ({placeholders})", old_ids
                )
                conn.execute(f"DELETE FROM conversations WHERE id IN ({placeholders})", old_ids)
                conn.commit()
        finally:
            conn.close()

        thread_ids = [row[1] for row in old_conversations]
        checkpoint_path = root / "kb_checkpoints.db"
        result["checkpoints"] = 0
        if checkpoint_path.is_file() and thread_ids:
            conn = sqlite3.connect(checkpoint_path)
            try:
                placeholders = ",".join("?" for _ in thread_ids)
                for table in ("checkpoints", "writes"):
                    exists = conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                    ).fetchone()
                    if not exists:
                        continue
                    count = int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE thread_id IN ({placeholders})",
                            thread_ids,
                        ).fetchone()[0]
                    )
                    result["checkpoints"] += count
                    if not dry_run and count:
                        conn.execute(
                            f"DELETE FROM {table} WHERE thread_id IN ({placeholders})",
                            thread_ids,
                        )
                if not dry_run:
                    conn.commit()
            finally:
                conn.close()
    else:
        result["checkpoints"] = 0
    return result
