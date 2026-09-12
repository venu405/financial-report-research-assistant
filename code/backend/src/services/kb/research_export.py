"""Deterministic Markdown exports for the structured research results."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

MISSING = "未提供/不可计算"


def _cell(value: Any) -> str:
    if value is None:
        return MISSING
    text = str(value).strip()
    if not text:
        return MISSING
    return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def _yes_no(value: Any) -> str:
    if value is None:
        return MISSING
    return "是" if bool(value) else "否"


def _page_range(start: Any, end: Any) -> str:
    if start is None and end is None:
        return MISSING
    if start is None:
        return _cell(end)
    if end is None or start == end:
        return _cell(start)
    return f"{_cell(start)}-{_cell(end)}"


def _generated_at(value: str | datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return str(value)


def _pending_lines(items: list[dict[str, Any]] | None) -> list[str]:
    if not items:
        return ["- 无"]
    lines = []
    for item in items:
        parts = [_cell(item.get("code"))]
        if item.get("company_name") is not None:
            parts.append(f"公司：{_cell(item.get('company_name'))}")
        if item.get("metric_code") is not None:
            parts.append(f"指标：{_cell(item.get('metric_code'))}")
        parts.append(_cell(item.get("message")))
        lines.append("- " + "；".join(parts))
    return lines


def _disclosure_line(disclosure: dict[str, Any]) -> list[str]:
    lines = [
        f"- {_cell(disclosure.get('metric_code'))}：文档 {_cell(disclosure.get('source_title'))}；页码 {_page_range(disclosure.get('source_page'), disclosure.get('source_page_end'))}"
    ]
    source_text = disclosure.get("source_text")
    if source_text is not None and str(source_text).strip():
        lines.append(f"  - 原文：{_cell(source_text)}")
    return lines


def render_company_analysis(
    result: dict[str, Any],
    *,
    kb_id: str,
    generated_at: str | datetime | None = None,
) -> str:
    company = result.get("company") or {}
    lines = [
        "# 单公司财报研究底稿",
        "",
        f"- UTC生成时间：{_generated_at(generated_at)}",
        f"- 资料库：{_cell(kb_id)}",
        f"- 公司：{_cell(company.get('name'))}（代码：{_cell(company.get('code'))}）",
        f"- 报告期：{_cell(result.get('report_period'))}",
        f"- 比较期：{_cell(result.get('comparison_period'))}",
        "",
        "## 数据口径与边界",
        "",
        "- 金额按结构化指标记录的元口径展示；缺失值、空值或不可计算结果统一标记为“未提供/不可计算”。",
        "- 本底稿只复述确定性计算结果和已有来源定位，不补写未提供的原文，不推断指标变化原因。",
        "- 本底稿仅供资料核验与研究整理，明确不构成投资建议。",
        "",
        "## 指标表",
        "",
        "| 指标 | 当前值（元） | 比较值（元） | 变动额（元） | 变动率（%） | 可比 | 说明 |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for metric in result.get("metrics") or []:
        current = metric.get("current") or {}
        comparison = metric.get("comparison") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _cell(metric.get("metric_name") or metric.get("metric_code")),
                    _cell(current.get("normalized_value")),
                    _cell(comparison.get("normalized_value")),
                    _cell(metric.get("change_amount")),
                    _cell(metric.get("change_rate_percent")),
                    _yes_no(metric.get("comparable")),
                    _cell(metric.get("comparison_note")),
                ]
            )
            + " |"
        )

    cash_flow = result.get("profit_cash_flow") or {}
    lines.extend(
        [
            "",
            "## 利润与经营活动现金流对照",
            "",
            "| 归母净利润（元） | 经营活动现金流净额（元） | 现金流减利润（元） | 关系 | 可比 | 说明 |",
            "| ---: | ---: | ---: | --- | --- | --- |",
            "| "
            + " | ".join(
                [
                    _cell(cash_flow.get("net_profit_value")),
                    _cell(cash_flow.get("operating_cash_flow_value")),
                    _cell(cash_flow.get("difference")),
                    _cell(cash_flow.get("relation")),
                    _yes_no(cash_flow.get("comparable")),
                    _cell(cash_flow.get("note")),
                ]
            )
            + " |",
            "",
            "## 来源文档与页码",
            "",
        ]
    )
    disclosures = result.get("disclosures") or []
    if disclosures:
        for disclosure in disclosures:
            lines.extend(_disclosure_line(disclosure))
    else:
        lines.append("- 无可用来源记录")

    lines.extend(["", "## 待核实事项", ""])
    lines.extend(_pending_lines(result.get("pending_items")))
    return "\n".join(lines) + "\n"


def render_peer_comparison(
    result: dict[str, Any],
    *,
    kb_id: str,
    generated_at: str | datetime | None = None,
) -> str:
    companies = result.get("companies") or []
    sample = "、".join(_cell(item.get("name")) for item in companies) or MISSING
    lines = [
        "# 用户选定公司同期间指标对比研究底稿",
        "",
        f"- UTC生成时间：{_generated_at(generated_at)}",
        f"- 资料库：{_cell(kb_id)}",
        f"- 所选样本：{sample}",
        f"- 报告期：{_cell(result.get('report_period'))}",
        "- 比较期：同一报告期内的所选公司记录",
        "",
        "## 数据口径与边界",
        "",
        "- 金额单位固定为元；缺失值、空值或不可计算结果统一标记为“未提供/不可计算”，不以 0 代替。",
        "- 仅在所选公司记录齐全、金额可用且期间类型和报表口径一致时标记为可比。",
        "- 本次结果只描述用户选定的公司样本，不代表整个行业；不作公司优劣顺序判断，不推断严格同行关系。",
        "- 本底稿仅供资料核验与研究整理，明确不构成投资建议。",
        "",
        "## 指标表",
        "",
        "| 指标 | 公司 | 指标值（元） | 报表口径 | 期间类型 | 提取状态 | 柱值（%） | 可比 | 说明 |",
        "| --- | --- | ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for metric in result.get("metrics") or []:
        for row in metric.get("rows") or []:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _cell(metric.get("metric_name") or metric.get("metric_code")),
                        _cell(row.get("company_name")),
                        _cell(row.get("value")),
                        _cell(row.get("statement_scope")),
                        _cell(row.get("period_type")),
                        _cell(row.get("extraction_status")),
                        _cell(row.get("bar_percent")),
                        _yes_no(row.get("comparable")),
                        _cell(row.get("note")),
                    ]
                )
                + " |"
            )
        if metric.get("comparison_note"):
            lines.append(f"- { _cell(metric.get('metric_code')) }：{_cell(metric.get('comparison_note'))}")

    lines.extend(["", "## 来源文档与页码", ""])
    source_lines = []
    for metric in result.get("metrics") or []:
        for row in metric.get("rows") or []:
            source_lines.append(
                f"- {_cell(metric.get('metric_code'))} / {_cell(row.get('company_name'))}：文档 {_cell(row.get('source_title'))}；页码 {_page_range(row.get('source_page'), row.get('source_page_end'))}"
            )
    lines.extend(source_lines or ["- 无可用来源记录"])
    lines.extend(["", "## 待核实事项", ""])
    lines.extend(_pending_lines(result.get("pending_items")))
    return "\n".join(lines) + "\n"


__all__ = ["MISSING", "render_company_analysis", "render_peer_comparison"]
