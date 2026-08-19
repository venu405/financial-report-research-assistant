"""AI 客服人设与对外话术（客服改造第 6 项）。

企业信息、服务时间、回答风格、拒答话术、转人工话术模板化，存库不硬编码。
管理面板可编辑；generate / direct_reply 节点拼 system prompt 时读取。

存储：SQLite 单表（kb_id -> config JSON）。默认人设兜底（未配置时用）。
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 默认人设（未配置时兜底，保证客服话术永远可用）
DEFAULT_PERSONA: dict[str, str] = {
    "company_name": "本公司",
    "service_hours": "工作日 9:00-18:00",
    "tone": "专业、友好、简洁",
    "refuse_message": "抱歉，我只能回答与本公司业务相关的问题。",
    "transfer_message": "好的，已为您转接人工客服，请稍候。",
}


class PersonaStore:
    """人设配置存储（SQLite）。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS persona (
                kb_id   TEXT PRIMARY KEY,
                config  TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        self._conn.commit()

    def get_persona(self, kb_id: str) -> dict[str, str]:
        """取人设配置；未配置或解析失败回退默认人设。"""
        row = self._conn.execute(
            "SELECT config FROM persona WHERE kb_id=?", (kb_id,)
        ).fetchone()
        if not row:
            return dict(DEFAULT_PERSONA)
        try:
            cfg = json.loads(row[0] or "{}")
        except Exception:
            return dict(DEFAULT_PERSONA)
        merged = dict(DEFAULT_PERSONA)
        merged.update({k: v for k, v in cfg.items() if v})
        return merged

    def set_persona(self, kb_id: str, config: dict[str, Any]) -> None:
        """保存人设配置（upsert）。"""
        cfg = {k: str(v) for k, v in (config or {}).items() if k in DEFAULT_PERSONA}
        self._conn.execute(
            "INSERT INTO persona(kb_id, config) VALUES(?, ?) "
            "ON CONFLICT(kb_id) DO UPDATE SET config=excluded.config",
            (kb_id, json.dumps(cfg, ensure_ascii=False)),
        )
        self._conn.commit()
        logger.info("更新人设 kb=%s", kb_id)

    def build_system_prompt(self, kb_id: str) -> str:
        """把 persona 拼成 system prompt（generate 节点用）。"""
        p = self.get_persona(kb_id)
        return (
            f"你是{p['company_name']}的智能客服，服务时间{p['service_hours']}。\n"
            f"回答风格：{p['tone']}。\n"
            "要求：仅基于提供的资料回答，不编造；资料不足时明确说明。\n"
            f"无法回答时的话术：{p['refuse_message']}\n"
            f"转人工时的话术：{p['transfer_message']}"
        )
