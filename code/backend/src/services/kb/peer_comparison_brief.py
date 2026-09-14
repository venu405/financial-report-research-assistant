"""同业对比简报：只消费 ``compare_companies`` 的结构化结果。"""
from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import Any

from loguru import logger

from .qa_graph import _llm_invoke


def _decimal(value: Any) -> Decimal | None:
    try:
        value = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return value if value.is_finite() else None


def _number(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _template_brief(result: dict[str, Any]) -> str:
    """从既有对比结果确定性地生成简报，不调用外部服务。"""
    companies = result.get("companies") or []
    sample = "、".join(str(item.get("name") or "未提供") for item in companies)
    lines = [
        "# 同业对比简报",
        "",
        "## 样本边界",
        "",
        f"- 报告期：{result.get('report_period') or '未提供'}",
        f"- 所选样本：{sample or '未提供'}",
        f"- {result.get('selection_note') or '本次结果只基于用户选定样本。'}",
        "- 本简报仅复述给定指标的直接数值比较，不构成投资建议。",
        "",
        "## 逐指标对比",
        "",
    ]
    for metric in result.get("metrics") or []:
        name = metric.get("metric_name") or metric.get("metric_code") or "未命名指标"
        rows = metric.get("rows") or []
        rendered = "；".join(
            f"{row.get('company_name') or '未提供'}：{row.get('value') if row.get('value') is not None else '未提供/不可计算'}"
            for row in rows
        )
        unit = str(metric.get("unit") or "").strip() or "未标注单位"
        lines.extend([f"### {name}", "", f"- 指标值（{unit}）：{rendered or '未提供/不可计算'}"])
        if metric.get("comparable"):
            values = [
                (str(row.get("company_name") or "未提供"), _decimal(row.get("value")))
                for row in rows
            ]
            valid = [(company, value) for company, value in values if value is not None]
            if len(valid) == len(rows) and valid:
                high = max(value for _, value in valid)
                low = min(value for _, value in valid)
                leaders = "、".join(company for company, value in valid if value == high)
                if high == low:
                    lines.append(f"- 领先方：{leaders}（数值并列，差额为 0 元）。")
                else:
                    lows = "、".join(company for company, value in valid if value == low)
                    lines.append(
                        f"- 领先方：{leaders}，直接数值高于 {lows}，差额为 {_number(high - low)} 元。"
                    )
            else:
                lines.append("- 不可比项说明：指标标记为可比，但数值记录不完整，未作领先方判断。")
        else:
            notes = [
                f"{row.get('company_name') or '未提供'}：{row.get('note') or '不可比'}"
                for row in rows
                if not row.get("comparable") or row.get("value") is None
            ]
            detail = "；".join(notes) or str(metric.get("comparison_note") or "不可比")
            lines.append(f"- 不可比项说明：{detail}。")
        lines.append("")

    lines.extend(["## 待核实或不可比事项", ""])
    pending = result.get("pending_items") or []
    if not pending:
        lines.append("- 无")
    else:
        for item in pending:
            lines.append(f"- {item.get('message') or item.get('code') or '未提供说明'}")
    return "\n".join(lines).rstrip() + "\n"


def _brief_prompt(result: dict[str, Any]) -> list[dict[str, str]]:
    facts = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    return [
        {
            "role": "system",
            "content": (
                "你生成中文 Markdown 同业对比简报。只能使用用户给出的结构化指标数值、公司、"
                "可比性和待核实事项；禁止使用外部知识、补充数字、推断原因或给出投资建议。"
                "优劣判断仅能写同口径且 comparable=true 的直接数值比较。每项不可比或缺失必须"
                "显式列出。必须包含：两家或三家公司逐指标对比、各指标领先方、不可比项说明、"
                "样本边界声明。"
            ),
        },
        {"role": "user", "content": f"给定事实（唯一来源）：\n{facts}"},
    ]


def generate_peer_comparison_brief(
    result: dict[str, Any], *, llm: Any, model: str, reasoning_effort: str | None = None
) -> dict[str, str]:
    """优先使用项目统一 LLM 调用；任何异常（含超时）回退确定性模板。"""
    try:
        brief = _llm_invoke(
            llm,
            _brief_prompt(result),
            model=model,
            reasoning_effort=reasoning_effort,
        )
        if brief.strip():
            return {"brief": brief, "brief_source": "llm"}
    except Exception as exc:
        # 简报是增强功能，调用失败不能影响已核验的结构化比较结果；
        # 但降级原因必须留痕，否则线上无法解释为何出现模板简报。
        logger.warning("同业对比简报 LLM 生成失败，回退规则模板: {}", exc)
    except BaseException as exc:
        logger.warning("同业对比简报 LLM 调用被中断，回退规则模板: {}", exc)
        raise
    return {"brief": _template_brief(result), "brief_source": "template"}


__all__ = ["generate_peer_comparison_brief", "_template_brief"]
