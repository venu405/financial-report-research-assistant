"""用户选定公司的同期间确定性指标对比，不推断同行关系。"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Sequence

from .financial_metric_store import FinancialMetricStore

METRIC_ORDER = (
    "revenue",
    "net_profit_parent",
    "operating_cash_flow",
    "total_assets",
    "total_liabilities",
)
METRIC_NAMES = {
    "revenue": "营业收入",
    "net_profit_parent": "归属于上市公司股东的净利润",
    "operating_cash_flow": "经营活动产生的现金流量净额",
    "total_assets": "资产总额",
    "total_liabilities": "负债总额",
}
VALID_CODES = set(METRIC_ORDER)


def normalize_company_names(company_names: Sequence[str] | None) -> list[str]:
    if company_names is None:
        raise ValueError("company_names 必须选择 2-3 家公司")
    names = [str(name).strip() for name in company_names]
    if any(not name for name in names):
        raise ValueError("company_names 不能包含空名称")
    if len(names) < 2 or len(names) > 3:
        raise ValueError("company_names 必须选择 2-3 家公司")
    if len(set(names)) != len(names):
        raise ValueError("company_names 不能包含重复名称")
    return names


def normalize_metric_codes(metric_codes: Sequence[str] | None) -> list[str]:
    if not metric_codes:
        return list(METRIC_ORDER)
    requested = [str(code).strip() for code in metric_codes]
    if any(not code for code in requested):
        raise ValueError("metric_codes 不能包含空值")
    invalid = [code for code in requested if code not in VALID_CODES]
    if invalid:
        raise ValueError(f"metric_codes 包含非法指标: {', '.join(invalid)}")
    selected = set(requested)
    return [code for code in METRIC_ORDER if code in selected]


def _decimal(value: Any) -> Decimal | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() else None


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _percent_text(value: Decimal) -> str:
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return _decimal_text(rounded) or "0"


def _latest(records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    if not records:
        return None, False
    return max(records, key=lambda item: (str(item.get("updated_at") or ""), int(item["id"]))), len(records) > 1


def _pending(code: str, company_name: str | None, metric_code: str | None, message: str) -> dict[str, Any]:
    return {"code": code, "company_name": company_name, "metric_code": metric_code, "message": message}


def _row(company_name: str, record: dict[str, Any] | None, *, note: str, comparable: bool) -> dict[str, Any]:
    if record is None:
        return {
            "company_name": company_name,
            "company_code": None,
            "value": None,
            "statement_scope": None,
            "period_type": None,
            "extraction_status": None,
            "source_title": "",
            "source_page": None,
            "source_page_end": None,
            "bar_percent": None,
            "comparable": False,
            "note": note,
        }
    value = record.get("normalized_value")
    return {
        "company_name": company_name,
        "company_code": str(record.get("company_code") or "").strip() or None,
        "value": value if _decimal(value) is not None else None,
        "statement_scope": record.get("statement_scope"),
        "period_type": record.get("period_type"),
        "extraction_status": record.get("extraction_status"),
        "source_title": record.get("source_title") or "",
        "source_page": record.get("source_page"),
        "source_page_end": record.get("source_page_end"),
        "bar_percent": None,
        "comparable": comparable,
        "note": note,
    }


def compare_companies(
    store: FinancialMetricStore,
    *,
    kb_id: str,
    company_names: Sequence[str],
    report_period: str,
    metric_codes: Sequence[str] | None = None,
) -> dict[str, Any]:
    names = normalize_company_names(company_names)
    period = str(report_period).strip()
    if not period:
        raise ValueError("report_period 不能为空")
    codes = normalize_metric_codes(metric_codes)
    pending: list[dict[str, Any]] = []
    selected_by_company: dict[str, dict[str, dict[str, Any] | None]] = {}
    company_rows: list[dict[str, Any]] = []

    for name in names:
        records, _ = store.list(kb_id=kb_id, company_name=name, report_period=period, limit=200)
        groups = {code: [] for code in codes}
        for record in records:
            if record.get("metric_code") in groups:
                groups[record["metric_code"]].append(record)
        selected: dict[str, dict[str, Any] | None] = {}
        for code in codes:
            selected_record, duplicate = _latest(groups[code])
            selected[code] = selected_record
            if duplicate:
                pending.append(_pending("duplicate_metric", name, code, f"{name} 的 {period} {METRIC_NAMES[code]} 有重复记录，已选择 updated_at/id 最新记录"))
        selected_by_company[name] = selected
        company_code = None
        for record in records:
            if str(record.get("company_code") or "").strip():
                company_code = str(record["company_code"]).strip()
                break
        company_rows.append({"name": name, "code": company_code, "found": bool(records)})
        if not records:
            pending.append(_pending("missing_company", name, None, f"未找到 {name} 在 {period} 的指标记录"))

    metrics: list[dict[str, Any]] = []
    for code in codes:
        selected_records = [selected_by_company[name][code] for name in names]
        values = [_decimal(record.get("normalized_value")) if record else None for record in selected_records]
        period_types = {record.get("period_type") for record in selected_records if record}
        scopes = {record.get("statement_scope") for record in selected_records if record}
        statuses = {record.get("extraction_status") for record in selected_records if record}
        has_missing = any(record is None or value is None for record, value in zip(selected_records, values, strict=True))
        status_conflict = any(status in {"conflict", "failed"} for status in statuses)
        period_mismatch = len(period_types) > 1
        scope_mismatch = len(scopes) > 1
        comparable = not has_missing and not status_conflict and not period_mismatch and not scope_mismatch

        if has_missing:
            note = "存在缺失指标或空值，无法比较"
        elif status_conflict:
            note = "存在冲突或失败记录，无法比较"
        elif period_mismatch:
            note = "期间类型不一致，无法比较"
        elif scope_mismatch:
            note = "报表口径不一致，无法比较"
        else:
            note = "按用户所选公司同期间、同口径比较"

        rows: list[dict[str, Any]] = []
        for name, record, value in zip(names, selected_records, values, strict=True):
            if record is None:
                pending.append(_pending("missing_metric", name, code, f"{name} 缺少 {METRIC_NAMES[code]}"))
                row_note = "指标缺失"
            elif value is None:
                pending.append(_pending("missing_metric_value", name, code, f"{name} 的 {METRIC_NAMES[code]} 数值为空"))
                row_note = "指标数值为空"
            elif record.get("extraction_status") in {"conflict", "failed"}:
                pending.append(_pending("metric_status_conflict", name, code, f"{name} 的 {METRIC_NAMES[code]} 状态为 {record.get('extraction_status')}"))
                row_note = "指标状态不可核验"
            elif period_mismatch:
                row_note = "期间类型不一致"
                pending.append(_pending("period_type_mismatch", name, code, f"{name} 的 {METRIC_NAMES[code]} 期间类型与其他所选公司不一致"))
            elif scope_mismatch:
                row_note = "报表口径不一致"
                pending.append(_pending("statement_scope_mismatch", name, code, f"{name} 的 {METRIC_NAMES[code]} 报表口径与其他所选公司不一致"))
            else:
                row_note = "已纳入同期间同口径比较"
            rows.append(_row(name, record, note=row_note, comparable=comparable))

        if comparable:
            maximum = max(abs(value) for value in values if value is not None)
            if maximum == 0:
                bars = ["0"] * len(values)
            else:
                bars = [_percent_text(abs(value) / maximum * Decimal("100")) for value in values if value is not None]
            for row, bar in zip(rows, bars, strict=True):
                row["bar_percent"] = bar

        metrics.append({
            "metric_code": code,
            "metric_name": METRIC_NAMES[code],
            "unit": "元",
            "comparable": comparable,
            "comparison_note": note,
            "rows": rows,
        })

    return {
        "report_period": period,
        "companies": company_rows,
        "metrics": metrics,
        "pending_items": pending,
        "selection_note": "用户选择的公司样本仅用于本次同期间指标并列展示，不代表整个行业。",
    }
