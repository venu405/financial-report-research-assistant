"""不调用模型的单公司财务指标确定性分析。"""
from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from .financial_metric_store import FinancialMetricStore
from .peer_comparison import (
    normalize_report_period_input,
    report_period_candidates,
    resolve_company_name,
)

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
YEAR_PERIOD_RE = re.compile(r"^(?P<year>\d{4})(?P<suffix>年度)?$")


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


def _percent_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return _decimal_text(rounded)


def _period_before(period: str) -> str | None:
    match = YEAR_PERIOD_RE.fullmatch(period.strip())
    if not match:
        return None
    previous = str(int(match.group("year")) - 1)
    return previous + (match.group("suffix") or "")


def _latest(records: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, bool]:
    if not records:
        return None, False
    selected = max(records, key=lambda item: (str(item.get("updated_at") or ""), int(item["id"])))
    return selected, len(records) > 1


def _pending(code: str, metric_code: str | None, message: str) -> dict[str, Any]:
    return {"code": code, "metric_code": metric_code, "message": message}


def _records_by_metric(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result = {code: [] for code in METRIC_ORDER}
    for record in records:
        if record.get("metric_code") in result:
            result[record["metric_code"]].append(record)
    return result


def _comparison_note(current: dict[str, Any] | None, comparison: dict[str, Any] | None) -> tuple[bool, str]:
    if current is None:
        return False, "当前期间指标缺失"
    if comparison is None:
        return False, "比较期间指标缺失"
    if current.get("period_type") != comparison.get("period_type"):
        return False, "期间类型不一致，无法比较"
    if current.get("statement_scope") != comparison.get("statement_scope"):
        return False, "报表口径不一致，无法比较"
    if _decimal(current.get("normalized_value")) is None or _decimal(comparison.get("normalized_value")) is None:
        return False, "指标数值缺失，无法比较"
    return True, "已按相同期间类型和报表口径比较"


def analyze_company(
    store: FinancialMetricStore,
    *,
    kb_id: str,
    company_name: str,
    report_period: str,
    comparison_period: str | None = None,
    derive_comparison: bool = True,
) -> dict[str, Any]:
    """按固定规则生成单公司指标、现金流对照和待办项。

    derive_comparison=False 时只取当前报告期的 5 项指标，不做任何期间比较
    （不自动推导上年、不产出比较相关的待核实项）。
    """
    queried_name = company_name.strip()
    if not queried_name:
        raise ValueError("company_name 不能为空")
    # 简称解析：全称精确匹配优先，唯一子串匹配兜底；多候选报错不猜测。
    resolved_name, candidates = resolve_company_name(store, kb_id=kb_id, name=queried_name)
    if not resolved_name:
        if candidates:
            raise ValueError(
                f"「{queried_name}」匹配到多家公司：{'、'.join(candidates)}，请改用完整公司名称"
            )
        resolved_name = queried_name
    company_name = resolved_name
    report_period = normalize_report_period_input(report_period)
    if not report_period:
        raise ValueError("report_period 不能为空")

    pending: list[dict[str, Any]] = []
    current_records: list[dict[str, Any]] = []
    for option in report_period_candidates(report_period):
        current_records, _ = store.list(
            kb_id=kb_id, company_name=company_name, report_period=option, limit=200
        )
        if current_records:
            break
    if queried_name != company_name:
        pending.append(_pending("name_resolved", None, f"已按简称「{queried_name}」匹配到 {company_name}"))

    explicit = comparison_period.strip() if comparison_period and comparison_period.strip() else ""
    derived_period: str | None = explicit or (_period_before(report_period) if derive_comparison else None)
    comparison_records: list[dict[str, Any]] = []
    if derived_period is not None:
        comparison_records, _ = store.list(
            kb_id=kb_id, company_name=company_name, report_period=derived_period, limit=200
        )
    elif comparison_period is None and derive_comparison:
        pending.append(_pending("comparison_period_unavailable", None, "报告期间不是可推导四位年份，未设置比较期间"))

    current_groups = _records_by_metric(current_records)
    comparison_groups = _records_by_metric(comparison_records)
    selected_current: dict[str, dict[str, Any] | None] = {}
    selected_comparison: dict[str, dict[str, Any] | None] = {}

    for code in METRIC_ORDER:
        selected, duplicate = _latest(current_groups[code])
        selected_current[code] = selected
        if duplicate:
            pending.append(_pending("duplicate_metric", code, f"{report_period} 存在重复指标，已选择 updated_at/id 最新记录"))
        if derived_period is None:
            selected_comparison[code] = None
            continue
        selected, duplicate = _latest(comparison_groups[code])
        selected_comparison[code] = selected
        if duplicate:
            pending.append(_pending("duplicate_metric", code, f"{derived_period} 存在重复指标，已选择 updated_at/id 最新记录"))

    if not current_records:
        for code in METRIC_ORDER:
            pending.append(_pending("missing_current_metric", code, f"当前期间缺少 {METRIC_NAMES[code]}"))
    else:
        for code in METRIC_ORDER:
            record = selected_current[code]
            if record is None:
                pending.append(_pending("missing_current_metric", code, f"当前期间缺少 {METRIC_NAMES[code]}"))
            elif _decimal(record.get("normalized_value")) is None:
                pending.append(_pending("missing_current_value", code, f"当前期间 {METRIC_NAMES[code]} 数值为空"))

    if derived_period is not None and not comparison_records:
        pending.append(_pending("comparison_period_not_found", None, f"未找到比较期间 {derived_period} 的指标记录"))
    if derived_period is not None:
        for code in METRIC_ORDER:
            record = selected_comparison[code]
            if record is None:
                pending.append(_pending("missing_comparison_metric", code, f"比较期间缺少 {METRIC_NAMES[code]}"))
            elif _decimal(record.get("normalized_value")) is None:
                pending.append(_pending("missing_comparison_value", code, f"比较期间 {METRIC_NAMES[code]} 数值为空"))

    metrics: list[dict[str, Any]] = []
    for code in METRIC_ORDER:
        current = selected_current[code]
        comparison = selected_comparison[code]
        comparable, note = _comparison_note(current, comparison)
        if current is not None and comparison is not None:
            if current.get("period_type") != comparison.get("period_type"):
                pending.append(_pending("period_type_mismatch", code, f"{METRIC_NAMES[code]} 期间类型不一致"))
            elif current.get("statement_scope") != comparison.get("statement_scope"):
                pending.append(_pending("statement_scope_mismatch", code, f"{METRIC_NAMES[code]} 报表口径不一致"))
        change_amount: str | None = None
        change_rate: str | None = None
        if comparable:
            current_value = _decimal(current["normalized_value"])
            comparison_value = _decimal(comparison["normalized_value"])
            assert current_value is not None and comparison_value is not None
            difference = current_value - comparison_value
            change_amount = _decimal_text(difference)
            if comparison_value == 0:
                note = "比较期数值为零，无法计算变化率"
                pending.append(_pending("zero_comparison_base", code, f"{METRIC_NAMES[code]} 比较期数值为零"))
            else:
                change_rate = _percent_text(difference / abs(comparison_value) * Decimal("100"))
        metrics.append({
            "metric_code": code,
            "metric_name": METRIC_NAMES[code],
            "current": current,
            "comparison": comparison,
            "change_amount": change_amount,
            "change_rate_percent": change_rate,
            "comparable": comparable,
            "comparison_note": note,
        })

    profit = selected_current["net_profit_parent"]
    cash_flow = selected_current["operating_cash_flow"]
    profit_value = _decimal(profit.get("normalized_value")) if profit else None
    cash_value = _decimal(cash_flow.get("normalized_value")) if cash_flow else None
    same_scope = bool(
        profit and cash_flow
        and profit.get("period_type") == cash_flow.get("period_type")
        and profit.get("statement_scope") == cash_flow.get("statement_scope")
    )
    if profit_value is not None and cash_value is not None and same_scope:
        difference = cash_value - profit_value
        if difference > 0:
            relation = "higher"
            note = "经营活动现金流量净额高于归母净利润"
        elif difference < 0:
            relation = "lower"
            note = "经营活动现金流量净额低于归母净利润"
        else:
            relation = "equal"
            note = "经营活动现金流量净额等于归母净利润"
        cash_flow_result = {
            "net_profit_value": _decimal_text(profit_value),
            "operating_cash_flow_value": _decimal_text(cash_value),
            "difference": _decimal_text(difference),
            "relation": relation,
            "comparable": True,
            "note": note,
        }
    else:
        if profit is None or profit_value is None:
            pending.append(_pending("missing_profit_cash_flow_input", "net_profit_parent", "当前归母净利润缺失或数值为空"))
        if cash_flow is None or cash_value is None:
            pending.append(_pending("missing_profit_cash_flow_input", "operating_cash_flow", "当前经营活动现金流量净额缺失或数值为空"))
        if profit and cash_flow and not same_scope:
            pending.append(_pending("profit_cash_flow_not_comparable", None, "利润和现金流期间类型或报表口径不一致"))
        cash_flow_result = {
            "net_profit_value": _decimal_text(profit_value),
            "operating_cash_flow_value": _decimal_text(cash_value),
            "difference": None,
            "relation": "unavailable",
            "comparable": False,
            "note": "当前期间归母净利润和经营活动现金流量净额不可比或缺失",
        }

    company_code = None
    for record in list(selected_current.values()) + list(selected_comparison.values()):
        if record and str(record.get("company_code") or "").strip():
            company_code = str(record["company_code"]).strip()
            break
    disclosures = []
    for code in METRIC_ORDER:
        record = selected_current[code]
        if record is not None:
            disclosures.append({
                "metric_code": code,
                "source_title": record.get("source_title") or "",
                "source_page": record.get("source_page"),
                "source_page_end": record.get("source_page_end"),
                "source_text": record.get("source_text") or "",
            })

    return {
        "company": {"name": company_name, "queried_name": queried_name, "code": company_code},
        "report_period": report_period,
        "comparison_period": derived_period,
        "metrics": metrics,
        "profit_cash_flow": cash_flow_result,
        "disclosures": disclosures,
        "pending_items": pending,
    }
