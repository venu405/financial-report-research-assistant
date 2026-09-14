"""可核验财务指标的独立 SQLite 存储。"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
import threading
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

METRIC_CODES = {
    "revenue",
    "net_profit_parent",
    "operating_cash_flow",
    "total_assets",
    "total_liabilities",
}
PERIOD_TYPES = {"annual", "semiannual", "quarterly"}
STATEMENT_SCOPES = {"consolidated", "parent", "unknown"}
EXTRACTION_STATUSES = {"verified", "missing", "conflict", "failed", "corrected"}
UNIT_MULTIPLIERS = {
    "元": Decimal("1"),
    "万元": Decimal("10000"),
    "亿元": Decimal("100000000"),
}

METRIC_COLUMNS = (
    "id", "kb_id", "company_name", "company_code", "report_period", "period_type",
    "metric_code", "metric_name", "raw_value", "raw_unit", "normalized_value",
    "normalized_unit", "statement_scope", "source_doc_id", "source_title",
    "source_page", "source_page_end", "source_chunk_id", "source_text",
    "extraction_status", "created_by", "created_at", "updated_at",
)
CREATE_COLUMNS = METRIC_COLUMNS[1:-3]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _clean_decimal_text(value: str) -> str:
    text = str(value).strip().replace(",", "")
    if not text:
        raise ValueError("raw_value 不能为空")
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("raw_value 必须是合法十进制数值") from exc
    if not number.is_finite():
        raise ValueError("raw_value 不能是 NaN 或无穷大")
    return str(number)


def _canonical_decimal(number: Decimal) -> str:
    text = format(number, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def normalize_financial_value(
    raw_value: str | None,
    raw_unit: str | None,
    extraction_status: str,
) -> tuple[str | None, str]:
    """把允许的人民币单位确定性归一为元。"""
    if extraction_status not in EXTRACTION_STATUSES:
        raise ValueError(f"非法 extraction_status: {extraction_status}")
    value_text = "" if raw_value is None else str(raw_value).strip()
    unit = "" if raw_unit is None else str(raw_unit).strip()
    if extraction_status == "missing":
        if value_text:
            raise ValueError("extraction_status=missing 时 raw_value 必须为空")
        return None, ""
    if not value_text:
        if extraction_status == "failed":
            return None, ""
        raise ValueError(f"extraction_status={extraction_status} 时 raw_value 不能为空")
    if unit not in UNIT_MULTIPLIERS:
        raise ValueError("raw_unit 只支持：元、万元、亿元")
    cleaned = _clean_decimal_text(value_text)
    normalized = Decimal(cleaned) * UNIT_MULTIPLIERS[unit]
    if not normalized.is_finite():
        raise ValueError("归一化后的数值无效")
    return _canonical_decimal(normalized), "元"


def _require_enum(value: Any, allowed: set[str], field: str) -> str:
    text = str(value or "").strip()
    if text not in allowed:
        raise ValueError(f"非法 {field}: {value}")
    return text


def _require_text(value: Any, field: str, *, required: bool = True) -> str:
    text = "" if value is None else str(value).strip()
    if required and not text:
        raise ValueError(f"{field} 不能为空")
    return text


def _page(value: Any, field: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是正整数") from exc
    if number < 1:
        raise ValueError(f"{field} 必须是正整数")
    return number


class FinancialMetricStore:
    """独立的财务指标库，连接在应用生命周期内复用。"""

    def __init__(self, db_path: str | Path):
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.RLock()
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS financial_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT NOT NULL,
                company_name TEXT NOT NULL,
                company_code TEXT NOT NULL DEFAULT '',
                report_period TEXT NOT NULL,
                period_type TEXT NOT NULL,
                metric_code TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                raw_value TEXT,
                raw_unit TEXT NOT NULL DEFAULT '',
                normalized_value TEXT,
                normalized_unit TEXT NOT NULL DEFAULT '',
                statement_scope TEXT NOT NULL,
                source_doc_id TEXT NOT NULL DEFAULT '',
                source_title TEXT NOT NULL DEFAULT '',
                source_page INTEGER,
                source_page_end INTEGER,
                source_chunk_id TEXT NOT NULL DEFAULT '',
                source_text TEXT NOT NULL DEFAULT '',
                extraction_status TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_financial_metrics_filter
                ON financial_metrics(kb_id, company_name, report_period, metric_code);
            CREATE TABLE IF NOT EXISTS financial_metric_revisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_id INTEGER NOT NULL,
                old_snapshot_json TEXT NOT NULL,
                new_snapshot_json TEXT NOT NULL,
                actor TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(metric_id) REFERENCES financial_metrics(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_financial_metric_revisions_metric
                ON financial_metric_revisions(metric_id, id DESC);
            """
        )
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create(self, data: dict[str, Any]) -> dict[str, Any]:
        prepared = self._prepare_create(data)
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO financial_metrics(" + ",".join(CREATE_COLUMNS) + ",created_by,created_at,updated_at) "
                "VALUES(" + ",".join("?" for _ in CREATE_COLUMNS) + ",?,?,?)",
                tuple(prepared[column] for column in CREATE_COLUMNS)
                + (prepared["created_by"], now, now),
            )
            self._conn.commit()
            metric_id = int(cur.lastrowid)
        return self.get(metric_id) or {}

    def get(self, metric_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT " + ",".join(METRIC_COLUMNS) + " FROM financial_metrics WHERE id=?",
            (metric_id,),
        ).fetchone()
        return self._row(row) if row else None

    def list(
        self,
        *,
        kb_id: str,
        company_name: str | None = None,
        report_period: str | None = None,
        metric_code: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        if limit < 1 or limit > 200:
            raise ValueError("limit 必须在 1 到 200 之间")
        if offset < 0:
            raise ValueError("offset 不能小于 0")
        clauses = ["kb_id=?"]
        params: list[Any] = [_require_text(kb_id, "kb_id")]
        for field, value in (("company_name", company_name), ("report_period", report_period), ("metric_code", metric_code)):
            if value is not None and str(value).strip():
                clauses.append(f"{field}=?")
                params.append(str(value).strip())
        where = " AND ".join(clauses)
        total = int(self._conn.execute(f"SELECT COUNT(*) FROM financial_metrics WHERE {where}", params).fetchone()[0])
        rows = self._conn.execute(
            "SELECT " + ",".join(METRIC_COLUMNS) + f" FROM financial_metrics WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [self._row(row) for row in rows], total

    def distinct_company_names(self, *, kb_id: str) -> list[str]:
        """返回知识库内出现过的公司全称（去重排序），供简称解析用。"""
        rows = self._conn.execute(
            "SELECT DISTINCT company_name FROM financial_metrics "
            "WHERE kb_id=? AND company_name<>'' ORDER BY company_name",
            [_require_text(kb_id, "kb_id")],
        ).fetchall()
        return [str(row[0]) for row in rows]

    def update(self, metric_id: int, patch: dict[str, Any], *, actor: str, reason: str) -> dict[str, Any] | None:
        actor_text = _require_text(actor, "actor")
        reason_text = _require_text(reason, "reason")
        allowed = {
            "raw_value", "raw_unit", "statement_scope", "source_doc_id", "source_title",
            "source_page", "source_page_end", "source_chunk_id", "source_text", "extraction_status",
        }
        unknown = set(patch) - allowed
        if unknown:
            raise ValueError(f"不允许修订字段: {', '.join(sorted(unknown))}")
        if not patch:
            raise ValueError("至少提供一个可修订字段")

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                old = self._get_unlocked(metric_id)
                if not old:
                    self._conn.rollback()
                    return None
                new = dict(old)
                for key, value in patch.items():
                    if key in {"source_page", "source_page_end"}:
                        new[key] = _page(value, key)
                    elif key == "statement_scope":
                        new[key] = _require_enum(value, STATEMENT_SCOPES, key)
                    elif key == "extraction_status":
                        new[key] = _require_enum(value, EXTRACTION_STATUSES, key)
                    elif key == "raw_value":
                        new[key] = None if value is None else str(value).strip()
                    elif key == "raw_unit":
                        new[key] = "" if value is None else str(value).strip()
                    else:
                        new[key] = "" if value is None else str(value).strip()

                normalized, normalized_unit = normalize_financial_value(
                    new["raw_value"], new["raw_unit"], new["extraction_status"]
                )
                if normalized is None:
                    new["raw_value"] = None
                    new["raw_unit"] = ""
                else:
                    new["raw_value"] = _clean_decimal_text(new["raw_value"])
                new["normalized_value"] = normalized
                new["normalized_unit"] = normalized_unit
                new["updated_at"] = _now()
                direct_keys = [key for key in patch if key not in {"raw_value", "raw_unit"}]
                assignments = [f"{key}=?" for key in direct_keys]
                assignments.extend(["raw_value=?", "raw_unit=?", "normalized_value=?", "normalized_unit=?", "updated_at=?"])
                values = [new[key] for key in direct_keys]
                values.extend([new["raw_value"], new["raw_unit"], normalized, normalized_unit, new["updated_at"], metric_id])
                self._conn.execute(
                    "UPDATE financial_metrics SET " + ",".join(assignments) + " WHERE id=?", values
                )
                self._conn.execute(
                    "INSERT INTO financial_metric_revisions(metric_id,old_snapshot_json,new_snapshot_json,actor,reason,created_at) VALUES(?,?,?,?,?,?)",
                    (metric_id, json.dumps(old, ensure_ascii=False, sort_keys=True),
                     json.dumps({**new, "id": metric_id}, ensure_ascii=False, sort_keys=True),
                     actor_text, reason_text, new["updated_at"]),
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
        return self.get(metric_id)

    def revisions(self, metric_id: int) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id,metric_id,old_snapshot_json,new_snapshot_json,actor,reason,created_at "
            "FROM financial_metric_revisions WHERE metric_id=? ORDER BY id DESC", (metric_id,)
        ).fetchall()
        return [{
            "id": row[0], "metric_id": row[1], "old_snapshot": json.loads(row[2]),
            "new_snapshot": json.loads(row[3]), "actor": row[4], "reason": row[5], "created_at": row[6],
        } for row in rows]

    @staticmethod
    def _prepare_create(data: dict[str, Any]) -> dict[str, Any]:
        metric_code = _require_enum(data.get("metric_code"), METRIC_CODES, "metric_code")
        period_type = _require_enum(data.get("period_type"), PERIOD_TYPES, "period_type")
        statement_scope = _require_enum(data.get("statement_scope", "unknown"), STATEMENT_SCOPES, "statement_scope")
        extraction_status = _require_enum(data.get("extraction_status", "verified"), EXTRACTION_STATUSES, "extraction_status")
        normalized, normalized_unit = normalize_financial_value(data.get("raw_value"), data.get("raw_unit", ""), extraction_status)
        raw_value = None if normalized is None else _clean_decimal_text(data["raw_value"])
        return {
            "kb_id": _require_text(data.get("kb_id"), "kb_id"),
            "company_name": _require_text(data.get("company_name"), "company_name"),
            "company_code": _require_text(data.get("company_code"), "company_code", required=False),
            "report_period": _require_text(data.get("report_period"), "report_period"),
            "period_type": period_type, "metric_code": metric_code,
            "metric_name": _require_text(data.get("metric_name") or metric_code, "metric_name"),
            "raw_value": raw_value,
            "raw_unit": "" if normalized is None else str(data.get("raw_unit") or "").strip(),
            "normalized_value": normalized, "normalized_unit": normalized_unit,
            "statement_scope": statement_scope,
            "source_doc_id": _require_text(data.get("source_doc_id"), "source_doc_id", required=False),
            "source_title": _require_text(data.get("source_title"), "source_title", required=False),
            "source_page": _page(data.get("source_page"), "source_page"),
            "source_page_end": _page(data.get("source_page_end"), "source_page_end"),
            "source_chunk_id": _require_text(data.get("source_chunk_id"), "source_chunk_id", required=False),
            "source_text": _require_text(data.get("source_text"), "source_text", required=False),
            "extraction_status": extraction_status,
            "created_by": _require_text(data.get("created_by"), "created_by"),
        }

    def _get_unlocked(self, metric_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT " + ",".join(METRIC_COLUMNS) + " FROM financial_metrics WHERE id=?", (metric_id,)
        ).fetchone()
        return self._row(row) if row else None

    @staticmethod
    def _row(row: tuple[Any, ...]) -> dict[str, Any]:
        return dict(zip(METRIC_COLUMNS, row, strict=True))
