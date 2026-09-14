"""单公司分析简报：只消费 ``analyze_company`` 的结构化结果。"""
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


def _record_value(record: dict[str, Any] | None) -> str:
    if record is None:
        return "未提供"
    value = _decimal(record.get("normalized_value"))
    if value is None:
        return "数值为空"
    unit = str(record.get("raw_unit") or "").strip()
    return f"{_number(value)} {unit}".strip()


def _template_brief(result: dict[str, Any]) -> str:
    """从既有分析结果确定性地生成简报，不调用外部服务。"""
    company = result.get("company") or {}
    lines = [
        "# 单公司分析简报",
        "",
        "## 样本边界",
        "",
        f"- 公司：{company.get('name') or '未提供'}",
        f"- 报告期：{result.get('report_period') or '未提供'}",
        "- 本简报仅复述接口返回的指标事实与来源定位，不包含预测、评分或投资建议。",
        "",
        "## 核心指标",
        "",
    ]
    for metric in result.get("metrics") or []:
        name = metric.get("metric_name") or metric.get("metric_code") or "未命名指标"
        current = metric.get("current")
        status = str(current.get("extraction_status") or "") if current else ""
        scope = str(current.get("statement_scope") or "") if current else ""
        lines.append(
            f"- **{name}**：{_record_value(current)}"
            + (f"（{scope} · {status}）" if scope or status else "")
        )
    cash = result.get("profit_cash_flow") or {}
    lines.extend(["", "## 利润与现金流对照", ""])
    if cash.get("comparable"):
        lines.append(
            f"- 经营活动现金流量净额 {cash.get('operating_cash_flow_value')} 元，"
            f"归母净利润 {cash.get('net_profit_value')} 元，"
            f"差额 {cash.get('difference')} 元：{cash.get('note') or ''}"
        )
    else:
        lines.append(f"- {cash.get('note') or '利润与现金流不可比或缺失'}。")
    lines.extend(["", "## 待核实或缺失事项", ""])
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
                "你生成中文 Markdown 单公司财务分析简报。只能使用用户给出的结构化指标数值、"
                "来源定位和待核实事项；禁止使用外部知识、补充数字、推断原因、做预测或给出"
                "投资建议。必须包含：样本边界声明、五项核心指标逐项事实（数值/口径/核验状态）、"
                "利润与经营现金流对照（不可比时明确说明）、待核实事项清单。"
            ),
        },
        {"role": "user", "content": f"给定事实（唯一来源）：\n{facts}"},
    ]


def generate_company_analysis_brief(
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
        # 简报是增强功能，调用失败不能影响已核验的结构化分析结果；
        # 但降级原因必须留痕，否则线上无法解释为何出现模板简报。
        logger.warning("单公司分析简报 LLM 生成失败，回退规则模板: {}", exc)
    except BaseException as exc:
        logger.warning("单公司分析简报 LLM 调用被中断，回退规则模板: {}", exc)
        raise
    return {"brief": _template_brief(result), "brief_source": "template"}


__all__ = ["generate_company_analysis_brief", "_template_brief"]
