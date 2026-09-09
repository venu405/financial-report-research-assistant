"""检索日志留痕--每次问答的检索全过程可回放（质量治理基建）。

为什么需要？（agent-desk 的 RetrieveLog 思路）
  - 客服答错了，第一件事是回放"当时检索到了什么、分数多少"，
    而不是猜"是不是模型幻觉"。
  - 调参（换 chunk_size / top_k / rerank 阈值）前后，用真实日志对比
    命中变化，而不是凭感觉。

一条日志 = 一次问答的检索侧全貌：
  question / rewritten / retrieval_query（第二次尝试的查询）
  answerable / evidence_score（可回答性判定）
  hits（rerank 前候选 + rerank 后 top-k，含各自分数）
  latency_ms（分阶段耗时）/ faithfulness（生成后忠实度评分回填）

表结构与 AuditStore 同风格：SQLite + WAL，写失败只告警不阻断（增强组件）。
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# hits JSON 里每条候选最多保留的文本长度（控表体积，排障看片段足够）
_HIT_TEXT_LEN = 120


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _slim_hit(h: dict[str, Any]) -> dict[str, Any]:
    """把检索命中压缩成日志需要的最小字段（控体积）。"""
    meta = h.get("metadata", {}) or {}
    return {
        "chunk_id": h.get("chunk_id", ""),
        "doc_title": meta.get("doc_title", ""),
        "text": (h.get("text") or "")[:_HIT_TEXT_LEN],
        "vec_score": h.get("score"),
        "rrf_score": h.get("rrf_score"),
        "rerank_score": h.get("rerank_score"),
    }


class RetrievalLogStore:
    """检索日志（SQLite）。同一进程串行写，WAL 防锁。"""

    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS retrieval_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT NOT NULL,
                kb_id           TEXT NOT NULL DEFAULT '',
                thread_id       TEXT NOT NULL DEFAULT '',
                question        TEXT NOT NULL DEFAULT '',
                rewritten       TEXT NOT NULL DEFAULT '',
                retrieval_query TEXT NOT NULL DEFAULT '',
                rerank_mode     TEXT NOT NULL DEFAULT 'off',
                answerable      INTEGER NOT NULL DEFAULT 1,
                evidence_score  REAL NOT NULL DEFAULT 0,
                escalate        INTEGER NOT NULL DEFAULT 0,
                attempts        INTEGER NOT NULL DEFAULT 1,
                latency_ms      TEXT NOT NULL DEFAULT '{}',
                hits            TEXT NOT NULL DEFAULT '[]',
                final_hits      TEXT NOT NULL DEFAULT '[]',
                faithfulness    INTEGER,
                answer          TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_log_ts ON retrieval_log(ts);
            CREATE INDEX IF NOT EXISTS idx_retrieval_log_kb ON retrieval_log(kb_id);
            """
        )
        self._conn.commit()

    def record(
        self,
        *,
        kb_id: str = "",
        thread_id: str = "",
        question: str = "",
        rewritten: str = "",
        retrieval_query: str = "",
        rerank_mode: str = "off",
        answerable: bool = True,
        evidence_score: float = 0.0,
        escalate: bool = False,
        attempts: int = 1,
        latency_ms: dict[str, float] | None = None,
        hits: list[dict[str, Any]] | None = None,
        final_hits: list[dict[str, Any]] | None = None,
        faithfulness: int | None = None,
        answer: str = "",
    ) -> None:
        """写一条检索日志。失败不阻断主流程（留痕是增强，不是单点）。"""
        try:
            self._conn.execute(
                "INSERT INTO retrieval_log(ts, kb_id, thread_id, question, rewritten, "
                "retrieval_query, rerank_mode, answerable, evidence_score, escalate, "
                "attempts, latency_ms, hits, final_hits, faithfulness, answer) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _utc_now_iso(),
                    kb_id,
                    thread_id,
                    question,
                    rewritten,
                    retrieval_query,
                    rerank_mode,
                    1 if answerable else 0,
                    round(float(evidence_score), 4),
                    1 if escalate else 0,
                    attempts,
                    json.dumps(latency_ms or {}, ensure_ascii=False),
                    json.dumps([_slim_hit(h) for h in (hits or [])], ensure_ascii=False),
                    json.dumps([_slim_hit(h) for h in (final_hits or [])], ensure_ascii=False),
                    faithfulness,
                    answer[:500],
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("检索日志写入失败: %s", exc)

    def recent(
        self, limit: int = 50, kb_id: str | None = None
    ) -> list[dict[str, Any]]:
        """最近 N 条（管理界面/排障用）。kb_id 指定时只查该库。"""
        sql = (
            "SELECT ts, kb_id, thread_id, question, rewritten, retrieval_query, "
            "rerank_mode, answerable, evidence_score, escalate, attempts, "
            "latency_ms, hits, final_hits, faithfulness, answer FROM retrieval_log"
        )
        params: tuple = ()
        if kb_id:
            sql += " WHERE kb_id = ?"
            params = (kb_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self._conn.execute(sql + "", (*params, limit)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_faithfulness(self, *, question: str, score: int) -> None:
        """回填忠实度评分（生成完成后质量闭环）。按 question 定位最近一条。"""
        try:
            self._conn.execute(
                "UPDATE retrieval_log SET faithfulness = ? WHERE rowid = "
                "(SELECT rowid FROM retrieval_log WHERE question = ? "
                "ORDER BY rowid DESC LIMIT 1)",
                (score, question),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("检索日志回填忠实度失败: %s", exc)

    @staticmethod
    def _row_to_dict(r: tuple) -> dict[str, Any]:
        try:
            latency = json.loads(r[11]) if r[11] else {}
            hits = json.loads(r[12]) if r[12] else []
            final_hits = json.loads(r[13]) if r[13] else []
        except json.JSONDecodeError:
            latency, hits, final_hits = {}, [], []
        return {
            "ts": r[0],
            "kb_id": r[1],
            "thread_id": r[2],
            "question": r[3],
            "rewritten": r[4],
            "retrieval_query": r[5],
            "rerank_mode": r[6],
            "answerable": bool(r[7]),
            "evidence_score": r[8],
            "escalate": bool(r[9]),
            "attempts": r[10],
            "latency_ms": latency,
            "hits": hits,
            "final_hits": final_hits,
            "faithfulness": r[14],
            "answer": r[15],
        }

    def count(self, kb_id: str | None = None) -> int:
        if kb_id:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM retrieval_log WHERE kb_id = ?", (kb_id,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM retrieval_log").fetchone()
        return row[0] if row else 0

    def stats(self) -> dict[str, float | int]:
        row = self._conn.execute(
            "SELECT COUNT(*),AVG(CASE WHEN faithfulness IS NOT NULL THEN faithfulness END),"
            "AVG(escalate),AVG(evidence_score) FROM retrieval_log"
        ).fetchone()
        return {
            "total": int(row[0] or 0) if row else 0,
            "avg_faithfulness": round(float(row[1] or 0), 2) if row else 0.0,
            "escalation_rate": round(float(row[2] or 0), 4) if row else 0.0,
            "avg_evidence_score": round(float(row[3] or 0), 4) if row else 0.0,
        }
