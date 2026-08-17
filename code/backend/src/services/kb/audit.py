"""操作审计——写操作留痕（生产合规要求）。

企业知识库的写操作（入库/更新/删除/建号/授权/角色变更）都要可追溯：
谁、在什么时间、对什么目标、做了什么。落 SQLite 审计表 + 同步打到日志，
双通道保证：日志丢了有表，表被清有日志。

表结构：audit_log(id PK AUTOINCREMENT, ts, user_id, action, target, detail)
  - ts：UTC ISO 时间
  - user_id：操作者（token/user_id 解析出的身份；未认证操作记 "anonymous"）
  - action：动作名（ingest / update / delete / create_user / grant / revoke /
            set_role / reset_token / admin_bootstrap）
  - target：操作对象（doc_id / kb_id / user_id）
  - detail：补充信息（JSON 字符串，如分块数、库名）
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class AuditStore:
    """审计日志（SQLite）。同一进程串行写，WAL 防锁。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action  TEXT NOT NULL,
                target  TEXT NOT NULL DEFAULT '',
                detail  TEXT NOT NULL DEFAULT ''
            );
            """
        )
        self._conn.commit()

    def record(
        self,
        *,
        user_id: str,
        action: str,
        target: str = "",
        detail: dict[str, Any] | None = None,
    ) -> None:
        """写一条审计记录。失败不阻断主流程（审计是增强，不是单点）。"""
        detail_s = json.dumps(detail, ensure_ascii=False) if detail else ""
        try:
            self._conn.execute(
                "INSERT INTO audit_log(ts, user_id, action, target, detail) "
                "VALUES(?, ?, ?, ?, ?)",
                (_utc_now_iso(), user_id or "anonymous", action, target, detail_s),
            )
            self._conn.commit()
            logger.info(
                "AUDIT user=%s action=%s target=%s detail=%s",
                user_id or "anonymous", action, target, detail_s,
            )
        except Exception as exc:  # 审计失败只记日志，不影响业务
            logger.warning("审计写入失败: %s", exc)

    def recent(self, limit: int = 200) -> list[dict[str, Any]]:
        """最近 N 条审计（管理界面/排障用）。"""
        rows = self._conn.execute(
            "SELECT ts, user_id, action, target, detail FROM audit_log "
            "ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "ts": r[0],
                "user_id": r[1],
                "action": r[2],
                "target": r[3],
                "detail": r[4],
            }
            for r in rows
        ]

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()
        return row[0] if row else 0
