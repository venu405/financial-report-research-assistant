"""LangGraph 问答编排：检索增强生成（RAG）工作流。

为什么这里用 LangGraph？
  1. 状态图原生表达"分支/条件边"：检索→生成→评估→(不满意→重试)
  2. checkpointer：对话历史持久化 + 断点续跑（LangGraph 内置）
  3. 每个节点可单独测试、可视化（LangGraph Studio）

图结构：
  rewrite(查询改写) → retrieve(检索) → generate(生成) → evaluate(评估)
                                                     └─(不达标)→ 回到 generate 重试
                                                     └─(达标)→ END
"""
from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
import uuid
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from services.kb import diagnostic_trace
from services.kb.embeddings import EmbeddingClient
from services.kb.vector_store import VectorStore

logger = logging.getLogger(__name__)

MAX_RETRY = 1  # 生成后评估不达标，最多重试 1 次

# 这些事实若缺少目标公司，检索结果很容易把不同公司的材料拼在一起。
# 规则门禁必须在 FAQ、查询改写和检索之前执行，且不新增 LLM 调用。
_COMPANY_FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("上市交易所", re.compile(r"上市交易所")),
    ("上市时间", re.compile(r"上市时间|(?:何时|什么时候|哪年|何年).{0,4}上市")),
    ("上市日期", re.compile(r"上市日期")),
    ("交易所", re.compile(r"交易所")),
    ("股票代码", re.compile(r"股票代码|股票代号")),
    ("证券代码", re.compile(r"证券代码|证券代号")),
    ("固定电话", re.compile(r"固定电话")),
    ("联系电话", re.compile(r"联系电话")),
    ("注册地址", re.compile(r"注册地址|注册地")),
    ("公司地址", re.compile(r"公司地址")),
    ("法定代表人", re.compile(r"法定代表人")),
    ("董事长", re.compile(r"董事长")),
)

_COMPANY_FULL_NAME_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9（）()·&.\-]{2,50}?(?:股份有限公司|有限责任公司|有限公司|集团公司|集团)"
)
_COMPANY_ST_ALIAS_RE = re.compile(r"\*?ST[\u4e00-\u9fffA-Za-z0-9]{1,12}", re.IGNORECASE)
_STOCK_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
_SHORT_NAME_BEFORE_FACT_RE = re.compile(
    r"(?P<name>[\u4e00-\u9fffA-Za-z0-9]{2,20})\s*(?:的)?\s*"
    r"(?:上市交易所|上市时间|上市日期|交易所|股票代码|股票代号|证券代码|证券代号|"
    r"固定电话|联系电话|注册地址|注册地|公司地址|法定代表人|董事长)"
)
_QUESTION_PREFIX_RE = re.compile(
    r"^(?:请问|请查询|查询|查一下|查查|请查|帮我(?:查(?:一下)?|查询)?|想知道|关于)"
)
_GENERIC_COMPANY_SUBJECTS = {
    "公司", "本公司", "该公司", "这家公司", "企业", "本企业", "该企业", "它",
    "股票", "证券", "哪家公司", "哪家企业",
}

# 高置信度的企业材料问句直接进入知识库，避免外部模型把公司简称、财务指标误判为
# “与业务无关”。只覆盖明确的问句；转人工、投诉和提示词攻击仍交给护栏分类。
_KB_MATERIAL_RE = re.compile(
    r"年报|半年报|半年度报告|年度报告|报告披露|营业收入|净利润|现金(?:流量|净流量)|"
    r"研发投入|经营计划|董事|监事|股东|同比|环比"
)
_QUESTION_MARKER_RE = re.compile(r"[？?]|多少|什么|为何|为什么|是否|哪(?:个|些|名|家)?|如何|怎么")
_NON_KB_INTENT_RE = re.compile(
    r"转人工|人工客服|投诉|报障|建工单|创建工单|忽略.{0,12}(?:指令|规则)|"
    r"系统提示词|开发者消息|越狱|扮演.{0,8}(?:系统|管理员)"
)
_FINANCIAL_SCOPE_RE = re.compile(
    r"营业总收入|营业收入|研发投入|研发费用|净利润|现金(?:流量|净流量)|资产总额|负债|每股收益|财务指标"
)
_CAUSE_EVIDENCE_RE = re.compile(
    r"主要(?:原因)?(?:系|是)|由于|因为|受[^。！？\n]{1,18}影响|导致|带动|推动|得益于|源于|所致"
)
_CAUSE_PROTECTION_CHANGE_RE = re.compile(
    r"同比|环比|增减|变化|变动|增加|减少|增长|下降|上升|下滑|降低|提升|"
    r"增幅|降幅|升幅|较上年|较同期|较去年|比上年|比同期|比去年"
)
_CAUSE_PROTECTION_DECREASE_RE = re.compile(
    r"下降|下滑|减少|降低|下跌|负增长|降幅|下降原因|减少原因"
)
_CAUSE_PROTECTION_INCREASE_RE = re.compile(
    r"增长|上升|增加|提升|上涨|正增长|增幅|增长原因|增加原因"
)
_CAUSE_PROTECTION_EVIDENCE_RE = re.compile(
    r"主要(?:原因)?(?:系|是)|原因在于|由于|因为|受[^。！？\n]{1,24}影响|"
    r"导致|带动|推动|得益于|源于|所致"
)
_PLAN_COMMITMENT_QUESTION_RE = re.compile(
    r"经营计划|年度计划|计划(?:营业收入|收入|金额)|经营目标"
)
_COMMITMENT_QUESTION_RE = re.compile(r"业绩承诺")
_COMMITMENT_JUDGEMENT_RE = re.compile(
    r"是否|是不是|会不会|能否|能不能|算不算|构成|不构成|属于|不属于|形成|不形成"
)
_PLAN_AMOUNT_ANSWER_RE = re.compile(
    r"经营计划|年度计划|计划(?:营业收入|收入|金额|目标)|经营目标"
)
_COMMITMENT_NEGATIVE_RE = re.compile(
    r"(?:并不|不|未|不会|并非|不是)(?:构成|属于|形成|视为|认定为)?"
    r"[^。！？；\n]{0,32}业绩承诺"
)
_COMMITMENT_POSITIVE_RE = re.compile(
    r"(?:构成|属于|形成|视为|认定为|是)[^。！？；\n]{0,32}业绩承诺"
)
_FINANCIAL_ITEM_PATTERNS: tuple[str, ...] = (
    "经营活动现金净流量",
    "经营活动现金流量净额",
    "现金净流量",
    "研发投入合计",
    "研发投入总额",
    "营业收入",
    "营业总收入",
    "研发费用",
    "归属于上市公司股东的净利润",
    "归母净利润",
    "净利润",
    "资产总额",
    "负债总额",
    "研发投入",
    "每股收益",
    "毛利率",
    "净利率",
)
_EXPLICIT_SUBJECT_RE = re.compile(
    r"^\s*(?:请问|查询|请查询|查一下|查查|请查|帮我(?:查(?:一下)?|查询)?|想知道|关于)?"
    r"(?P<subject>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9（）()·&.\-]{1,20}?)"
    r"(?=(?:\d{4}年|半年度|年度|经营|营业|净利润|现金|资产|负债|研发|每股|毛利率|净利率))"
)
_GENERIC_SUBJECT_WORDS = _GENERIC_COMPANY_SUBJECTS | {
    "年报", "半年报", "年度报告", "半年度报告", "营业收入", "净利润", "现金流量",
    "资产总额", "负债总额", "财务指标", "经营活动",
}


@dataclass(frozen=True)
class NumericClaim:
    """答案或证据中的数字声明；使用 Decimal，避免浮点换算误差。"""

    raw: str
    number: Decimal
    unit: str
    canonical_value: Decimal
    is_percent: bool
    metric: str = ""
    report_period: str = ""
    statement_scope: str = ""
    start: int = -1
    end: int = -1
    # 宽松行绑定：表格行标题与数值被 PDF 拆到不同行时，metric（盲提取）
    # 会因邻域出现多个指标而放弃绑定。line_metric 取“最近的前置指标名”，
    # 供定向核验使用——核验时已知要找哪个指标，只需确认归属，无需唯一确定。
    line_metric: str = field(default="", compare=False)
    adjustment: str = ""


@dataclass(frozen=True)
class FinancialFact:
    """索引侧结构化财务事实；字段不完整时仍保留可用部分。"""

    metric: str
    raw_value: str
    canonical_value: Decimal | None
    unit: str = ""
    statement_scope: str = ""
    report_period: str = ""


# 指标别名是有意收敛的：检索可以扩大同义词，核验不能把两个披露口径
# 无条件合并。尤其是营业收入/营业总收入、研发投入合计/研发费用。
_FINANCIAL_METRIC_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "rd_revenue_ratio",
        (
            "研发投入占营业收入的比例",
            "研发投入占营业收入比例",
            # 「比率」「比重」与「比例」是同一占比口径的等义后缀，不是放宽：
            # 三者都指向“研发投入 ÷ 营业收入”这一个披露项。缺了它们，问句
            # 写成“……比率”时问题侧提取不到指标键，结构化事实无法绑定，
            # 核验只能拒答（shenlian_rnd_ratio 一类题的直接原因）。
            "研发投入占营业收入的比率",
            "研发投入占营业收入比率",
            "研发投入占营业收入比重",
            "研发投入占比",
        ),
    ),
    (
        "rd_investment_total",
        ("研发投入合计", "研发投入总额", "研发投入金额", "研发投入"),
    ),
    (
        "rd_expense",
        ("研发费用", "研究与开发费用"),
    ),
    (
        "net_profit_attributable",
        (
            "归属于上市公司股东的净利润",
            "归属于上市公司股东净利润",
            "归母净利润",
        ),
    ),
    ("total_revenue", ("营业总收入",)),
    ("revenue", ("营业收入", "主营业务收入")),
    ("profit_total", ("利润总额",)),
    ("net_profit", ("净利润",)),
    ("operating_cost", ("营业成本",)),
    ("selling_expense", ("销售费用",)),
    ("administrative_expense", ("管理费用",)),
    ("financial_expense", ("财务费用",)),
    ("investment_income", ("投资收益",)),
    (
        "operating_cashflow",
        (
            "经营活动产生的现金流量净额",
            "经营活动现金流量净额",
            "经营活动现金净流量",
            "现金净流量",
        ),
    ),
    ("total_assets", ("资产总额", "资产合计")),
    ("total_liabilities", ("负债总额", "负债合计")),
    ("eps", ("每股收益", "基本每股收益")),
)
_METRIC_DISPLAY_NAMES = {
    "rd_revenue_ratio": "研发投入占营业收入的比例",
    "rd_investment_total": "研发投入合计",
    "rd_expense": "研发费用",
    "net_profit_attributable": "归母净利润",
    "total_revenue": "营业总收入",
    "revenue": "营业收入",
    "profit_total": "利润总额",
    "net_profit": "净利润",
    "operating_cost": "营业成本",
    "selling_expense": "销售费用",
    "administrative_expense": "管理费用",
    "financial_expense": "财务费用",
    "investment_income": "投资收益",
    "operating_cashflow": "经营活动现金净流量",
    "total_assets": "资产总额",
    "total_liabilities": "负债总额",
    "eps": "每股收益",
}
_METRIC_QUERY_SYNONYMS = {
    "rd_revenue_ratio": (
        "研发投入占营业收入的比例",
        "研发投入占营业收入的比率",
        "研发投入占营业收入比率",
        "研发投入占比",
    ),
    "revenue": ("营业收入", "主营业务收入"),
    "total_revenue": ("营业总收入", "营业收入"),
    "rd_investment_total": ("研发投入合计", "研发投入总额", "研发投入"),
    "rd_expense": ("研发费用",),
    "net_profit_attributable": ("归母净利润", "归属于上市公司股东的净利润"),
}
_MAX_RETRIEVAL_QUERIES = 8


_NUMBER_CLAIM_RE = re.compile(
    r"(?<!\d)[+-]?(?:\d{1,3}(?:[, \t]\d{3})+|\d+)(?:\.\d+)?"
    r"[ \t]*(万亿|千亿|百亿|十亿|百万元|亿元|万元|千万|百万|十万|千元|港元|美元|亿|万|千|百|元|%|％)?"
)
_UNIT_SCALE: dict[str, Decimal] = {
    "": Decimal("1"),
    "元": Decimal("1"),
    "百": Decimal("100"),
    "千": Decimal("1000"),
    "万": Decimal("10000"),
    "万元": Decimal("10000"),
    "十万": Decimal("100000"),
    "百万": Decimal("1000000"),
    "千万": Decimal("10000000"),
    "亿": Decimal("100000000"),
    "亿元": Decimal("100000000"),
    "百万元": Decimal("1000000"),
    "十亿": Decimal("1000000000"),
    "百亿": Decimal("10000000000"),
    "千亿": Decimal("100000000000"),
    "万亿": Decimal("1000000000000"),
    "千元": Decimal("1000"),
    "港元": Decimal("1"),
    "美元": Decimal("1"),
    "%": Decimal("1"),
    "％": Decimal("1"),
}
_TABLE_UNIT_WHITELIST = frozenset(
    unit for unit in _UNIT_SCALE if unit not in ("", "%", "％")
)
_SCOPE_LABELS = {
    "consolidated": "合并",
    "parent": "母公司",
    "subsidiary": "子公司",
    "unknown": "未知",
}


def _numeric_text_for_matching(text: str) -> str:
    """移除引用/列表编号，并统一全角标点；不改变数字值。"""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = re.sub(r"\[\s*\d+\s*\]", " ", normalized)
    # 生成答案常用“1. … 2. …”列点，列点编号不是事实声明。
    normalized = re.sub(
        r"(?m)^\s*\d{1,3}\s*(?:[)、]\s*|\.(?:\s+|(?=\D)))",
        " ",
        normalized,
    )
    return normalized.replace("\u00a0", " ").replace("\u3000", " ")


def _normalized_metric_text(value: Any) -> str:
    """归一化指标标签，但不做可能改变口径的模糊替换。"""
    return re.sub(r"[\s_\-/:：；;（）()]+", "", unicodedata.normalize("NFKC", str(value or "")).casefold())


def _canonical_metric(value: Any) -> str:
    """把中文/稳定英文指标名归一为有限的内部指标键。"""
    normalized = _normalized_metric_text(value)
    if not normalized:
        return ""
    for metric, aliases in _FINANCIAL_METRIC_ALIASES:
        if normalized == _normalized_metric_text(metric):
            return metric
        for alias in aliases:
            if normalized == _normalized_metric_text(alias):
                return metric
    # 允许 Luna A 后续提供稳定的 snake_case 指标键，但不猜测陌生中文标签。
    if re.fullmatch(r"[a-z][a-z0-9_]*", normalized):
        return normalized
    return normalized


def _metric_alias_mentions(text: str) -> list[tuple[int, int, str, str]]:
    """返回 (start, end, canonical_metric, raw_alias)，长别名优先。"""
    normalized = _numeric_text_for_matching(text)
    found: list[tuple[int, int, str, str]] = []
    aliases: list[tuple[str, str]] = [
        (alias, metric)
        for metric, metric_aliases in _FINANCIAL_METRIC_ALIASES
        for alias in metric_aliases
    ]
    aliases.sort(key=lambda item: len(item[0]), reverse=True)
    for alias, metric in aliases:
        # PDF 抽取常把长指标名拆到相邻行；允许指标字符之间出现空白，
        # 但保留原字符串位置，后续数字邻近/行绑定仍可精确使用。
        alias_pattern = r"\s*".join(re.escape(char) for char in alias)
        for match in re.finditer(alias_pattern, normalized):
            # 长别名已经先登记，短别名若落在其内部则会造成
            # “营业总收入”同时被识别为“营业收入”。
            if any(
                start <= match.start() and match.end() <= end
                for start, end, _, _ in found
            ):
                continue
            found.append((match.start(), match.end(), metric, alias))
    return sorted(found, key=lambda item: (item[0], -(item[1] - item[0])))


def _metric_mentions(text: str) -> list[tuple[int, int, str]]:
    """返回文本中的 (start, end, canonical_metric)，长别名优先。"""
    return [
        (start, end, metric)
        for start, end, metric, _alias in _metric_alias_mentions(text)
    ]


def _metric_alias_for_numeric_claim(text: str, claim: NumericClaim) -> str:
    """返回数字附近的原始指标别名，无法唯一定位时返回空。"""
    if claim.start < 0:
        return ""
    normalized = _numeric_text_for_matching(text)
    start, end = claim.start, claim.end
    mentions = _metric_alias_mentions(normalized)

    line_start = normalized.rfind("\n", 0, start) + 1
    line_end = normalized.find("\n", end)
    if line_end < 0:
        line_end = len(normalized)
    preceding = [
        item
        for item in mentions
        if item[1] <= start
        and item[0] >= line_start
        and item[2] == claim.metric
    ]
    if preceding:
        return min(
            preceding,
            key=lambda item: abs((item[0] + item[1]) / 2 - (start + end) / 2),
        )[3]

    window_start = max(0, start - 80)
    nearby = [
        item
        for item in mentions
        if window_start <= item[0]
        and item[1] <= start
        and item[2] == claim.metric
    ]
    if len({item[3] for item in nearby}) == 1:
        return nearby[0][3]
    return ""


def _raw_revenue_aliases_from_metadata(metadata: dict[str, Any]) -> set[str]:
    """读取未被 canonicalize 的收入字段标签；canonical revenue 视为未知。"""
    labels: list[Any] = []
    raw_metrics = metadata.get("financial_metrics")
    if isinstance(raw_metrics, str):
        labels.extend(raw_metrics.split("|"))
    elif isinstance(raw_metrics, (list, tuple, set)):
        labels.extend(raw_metrics)

    raw_facts = metadata.get("financial_facts_json")
    if isinstance(raw_facts, str):
        try:
            raw_facts = json.loads(raw_facts)
        except (TypeError, ValueError):
            raw_facts = []
    if isinstance(raw_facts, dict):
        raw_facts = raw_facts.get("facts") or raw_facts.get("items") or []
    if isinstance(raw_facts, list):
        labels.extend(
            item.get("metric")
            for item in raw_facts
            if isinstance(item, dict)
        )

    revenue_aliases = next(
        aliases
        for metric, aliases in _FINANCIAL_METRIC_ALIASES
        if metric == "revenue"
    )
    normalized_aliases = {
        _normalized_metric_text(alias): alias for alias in revenue_aliases
    }
    return {
        normalized_aliases[_normalized_metric_text(label)]
        for label in labels
        if _normalized_metric_text(label) in normalized_aliases
    }


def _raw_financial_fact_items_from_metadata(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return raw fact objects so a canonical fact retains its source metric label."""
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("financial_facts_json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if isinstance(raw, dict):
        raw = raw.get("facts", raw.get("financial_facts", raw.get("items", [])))
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _raw_revenue_alias_for_fact(
    metadata: dict[str, Any], fact: FinancialFact
) -> str:
    """Bind one parsed fact back to its raw revenue field, never to the whole record."""
    revenue_aliases = next(
        aliases
        for metric, aliases in _FINANCIAL_METRIC_ALIASES
        if metric == "revenue"
    )
    normalized_aliases = {
        _normalized_metric_text(alias): alias for alias in revenue_aliases
    }
    raw_items = _raw_financial_fact_items_from_metadata(metadata)
    for item in raw_items:
        raw_alias = normalized_aliases.get(
            _normalized_metric_text(item.get("metric"))
        )
        if not raw_alias:
            continue
        parsed = _parse_financial_facts_json([item])
        if parsed and parsed[0] == fact:
            return raw_alias

    # Legacy payloads may only retain a single financial_metrics label. Treat
    # one unambiguous raw alias as compatible; a mixed block-wide alias set is
    # deliberately not enough to bind a specific fact.
    aliases = _raw_revenue_aliases_from_metadata(metadata)
    return next(iter(aliases)) if len(aliases) == 1 else ""


def _strict_revenue_field(question: str) -> str:
    """只为单独询问“营业收入”返回严格字段；其他问法保持兼容。"""
    aliases = {
        alias
        for _start, _end, metric, alias in _metric_alias_mentions(question)
        if metric == "revenue"
    }
    if aliases == {"营业收入"}:
        return "营业收入"
    if aliases == {"主营业务收入"}:
        return "主营业务收入"
    return ""


def _strict_revenue_record_allowed(
    required_field: str,
    record: dict[str, Any],
    claim: NumericClaim | None = None,
) -> bool:
    """收紧后置字段绑定，但保留无原始字段标签旧索引的兼容性。"""
    if required_field != "营业收入":
        return True
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    aliases = _raw_revenue_aliases_from_metadata(metadata)
    text = str(record.get("text") or "")
    if claim is not None:
        text_alias = _metric_alias_for_numeric_claim(text, claim)
        if text_alias:
            # The alias adjacent to this exact numeric claim wins. A second
            # alias elsewhere in the same block must not make a wrong field
            # look valid.
            return text_alias != "\u4e3b\u8425\u4e1a\u52a1\u6536\u5165"
    # 只有明确出现主营业务收入、且没有同一记录的营业收入标签时才拒绝；
    # 无标签的旧 canonical revenue 数据继续走原有兼容路径。
    return not ("主营业务收入" in aliases and "营业收入" not in aliases)


def _strict_revenue_fact_allowed(
    required_field: str, record: dict[str, Any], fact: FinancialFact
) -> bool:
    """Apply strict revenue binding to the particular structured fact."""
    if required_field != "\u8425\u4e1a\u6536\u5165":
        return True
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    raw_alias = _raw_revenue_alias_for_fact(metadata, fact)
    if raw_alias:
        return raw_alias != "\u4e3b\u8425\u4e1a\u52a1\u6536\u5165"

    # If structured metadata has no raw label, use only the text claim that
    # numerically matches this fact. No local label keeps old canonical
    # revenue payloads compatible.
    for evidence_text in _table_body_evidence(
        {"text": str(record.get("text") or ""), "metadata": metadata}
    ):
        for raw_claim in _extract_numeric_claims(evidence_text):
            # 与正文核验保持同一套优先级：metadata 单指标兜垫只能最后参与。
            claim = _claim_with_context(
                raw_claim,
                evidence_text,
                metadata=metadata,
                fallback_metrics=list(
                    _financial_metric_keys_from_metadata(metadata)
                ),
            )
            if _fact_supports_claim(claim, fact):
                alias = _metric_alias_for_numeric_claim(evidence_text, claim)
                if alias:
                    return alias != "\u4e3b\u8425\u4e1a\u52a1\u6536\u5165"
    return True


def _strict_revenue_answer_allowed(
    required_field: str, answer: str, claim: NumericClaim
) -> bool:
    if required_field != "营业收入":
        return True
    return _metric_alias_for_numeric_claim(answer, claim) != "主营业务收入"


def _has_strict_revenue_field_mismatch(
    question: str, answer: str, evidence_texts: list[Any]
) -> bool:
    """用于把收入字段错配写成可诊断的核验原因。"""
    required_field = _strict_revenue_field(question)
    if required_field != "营业收入":
        return False
    for claim in _extract_numeric_claims(answer):
        if not _is_unbound_format_number(claim, answer) and not _strict_revenue_answer_allowed(
            required_field, answer, claim
        ):
            return True
    for record in _evidence_records(evidence_texts):
        for claim in _extract_numeric_claims(record["text"]):
            if _metric_alias_for_numeric_claim(record["text"], claim) == "主营业务收入":
                if not _strict_revenue_record_allowed(required_field, record, claim):
                    return True
        metadata = record.get("metadata") or {}
        aliases = _raw_revenue_aliases_from_metadata(metadata)
        if "主营业务收入" in aliases and "营业收入" not in aliases:
            return True
    return False


# 同一披露项的等义表述分组。只有当证据池把它们当作**两个独立字段分别披露**
# 时，分组内的别名差异才是真的字段错配；否则只是措辞不同。
_EQUIVALENT_METRIC_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"revenue", "total_revenue"}),
)


def _equivalent_metric_group(metric: str) -> frozenset[str]:
    """返回该指标所属的等义组；不属于任何组时返回空集合。"""
    if not metric:
        return frozenset()
    for group in _EQUIVALENT_METRIC_GROUPS:
        if metric in group:
            return group
    return frozenset()


def _evidence_declares_metric(
    records: list[dict[str, Any]],
    fact_records: list[tuple[Any, dict[str, Any]]],
    text_claims: list[tuple[Any, dict[str, Any], str]],
    metric: str,
) -> bool:
    """证据池是否把 metric 当作独立字段披露过。

    财报只披露了“营业收入”时，答案写“营业总收入”并没有第二个数值可串，
    判成字段错配会让正确数字被无辜拒绝（huawei 一类的直接原因）。只有证据
    侧真的存在该字段（结构化事实、正文指标绑定或 metadata 指标键）时，
    答案换成另一个等义别名才构成错配。
    """
    if not metric:
        return False
    for fact, _record in fact_records:
        if getattr(fact, "metric", "") == metric:
            return True
    for bound_claim, _record, _alias in text_claims:
        if getattr(bound_claim, "metric", "") == metric:
            return True
    for record in records:
        metadata = record.get("metadata") if isinstance(record, dict) else None
        if metric in _financial_metric_keys_from_metadata(metadata or {}):
            return True
    return False


def _metric_for_numeric_match(text: str, match: re.Match[str]) -> str:
    """按同一行、邻近标签把一个数字绑定到指标；无法确定时返回空。"""
    start, end = match.span()
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end < 0:
        line_end = len(text)
    line_mentions = _metric_mentions(text[line_start:line_end])
    if line_mentions:
        absolute = [
            (line_start + item_start, line_start + item_end, metric)
            for item_start, item_end, metric in line_mentions
        ]
        preceding = [item for item in absolute if item[1] <= start]
        if preceding:
            return min(
                preceding,
                key=lambda item: abs((item[0] + item[1]) / 2 - (start + end) / 2),
            )[2]
        return ""

    # 表格抽取有时把行标题和数值拆到相邻行。只看有限窗口，且若窗口里
    # 出现多个不同指标则放弃绑定，避免退化成数字袋子。
    window_start = max(0, start - 80)
    window_end = min(len(text), end + 80)
    nearby = _metric_mentions(text[window_start:window_end])
    preceding = [item for item in nearby if window_start + item[1] <= start]
    metrics = {metric for _s, _e, metric in preceding}
    if len(metrics) == 1:
        return next(iter(metrics))
    return ""


# PDF 表格常把行标题与数值拆到相邻行（“营业收入”一行，“1,234,567.89”下一行）。
# 严格绑定要求邻域内指标唯一，多列表格因此大量放弃绑定 —— 核验阶段表现为
# “证据缺 metric 标签”，最终把本来正确的数字判为无证据而拒答。
# 核验是定向的：已知目标指标，只需确认数字归属，取最近的前置指标名即可。
# 默认开启，KB_METRIC_LOOSE_BINDING=0 可回到严格语义做 A/B。
_loose_metric_binding = os.getenv("KB_METRIC_LOOSE_BINDING", "").strip() not in (
    "0",
    "false",
    "no",
)

_LOOSE_BINDING_MAX_LINES = 4
_LOOSE_BINDING_MAX_CHARS = 200


def _metric_before_position(
    text: str, offset: int, line_start: int
) -> str:
    """在 [line_start, offset) 内取离数字最近的指标名。"""
    mentions = _metric_mentions(text[line_start:offset])
    if not mentions:
        return ""
    absolute = [
        (line_start + start, line_start + end, metric)
        for start, end, metric in mentions
    ]
    return min(
        absolute,
        key=lambda item: abs((item[0] + item[1]) / 2 - offset),
    )[2]


def _metric_for_numeric_loose(text: str, match: re.Match[str]) -> str:
    """宽松绑定：行内优先，行内无指标名时向上回溯最近的行标题。

    与 _metric_for_numeric_match 只差“多指标不放弃”这一点。多列表格中，
    同行其他指标不会比紧邻的行标题更近，因此最近优先即可正确归属。
    """
    start, _end = match.span()
    line_start = text.rfind("\n", 0, start) + 1

    # 1) 行内：数字之前最近的指标名
    inline = _metric_before_position(text, start, line_start)
    if inline:
        return inline

    # 2) 行内无指标名：先把界线上推（最多 4 行 / 200 字符），再在扩展范围内
    #    取离数字最近的前置指标名。
    bound = line_start
    for _ in range(_LOOSE_BINDING_MAX_LINES):
        if bound <= 0:
            break
        prev = text.rfind("\n", 0, bound - 1) + 1
        if start - prev > _LOOSE_BINDING_MAX_CHARS:
            break
        bound = prev
        if bound == 0:
            break
    if bound < line_start:
        return _metric_before_position(text, start, bound)
    return ""


def _parse_decimal(value: Any) -> Decimal | None:
    """宽容解析结构化数字；失败返回 None，不让旧索引请求报错。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None
    unit_match = re.search(
        r"(万亿|千亿|百亿|十亿|百万元|亿元|万元|千万|百万|十万|千元|港元|美元|亿|万|千|百|元|%|％)$",
        text,
    )
    unit = unit_match.group(1) if unit_match else ""
    number_text = text[: unit_match.start()] if unit_match else text
    try:
        return Decimal(re.sub(r"[,\s]", "", number_text)) * _UNIT_SCALE[unit]
    except (InvalidOperation, ValueError):
        pass
    claims = _extract_numeric_claims(text) if "_NUMBER_CLAIM_RE" in globals() else []
    if claims:
        return claims[0].canonical_value
    try:
        return Decimal(re.sub(r"[,\s]", "", text))
    except (InvalidOperation, ValueError):
        return None


def _normalize_fact_unit(value: Any) -> str:
    unit = unicodedata.normalize("NFKC", str(value or "")).strip()
    return unit if unit in _UNIT_SCALE else ""


def _normalize_scope(value: Any) -> str:
    text = _normalized_metric_text(value)
    if text in {"consolidated", "合并", "合并口径", "合并报表"}:
        return "consolidated"
    if text in {"parent", "母公司", "母公司口径", "单体"}:
        return "parent"
    if text in {"subsidiary", "子公司"}:
        return "subsidiary"
    return text


def _normalize_report_period(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = re.sub(r"\s+", "", text)
    if not text:
        return ""
    suffix_aliases = (
        ("半年度", "上半年"),
        ("第一季度", "第一季度"),
        ("第二季度", "第二季度"),
        ("第三季度", "第三季度"),
        ("第四季度", "第四季度"),
        ("一季度", "第一季度"),
        ("二季度", "第二季度"),
        ("三季度", "第三季度"),
        ("四季度", "第四季度"),
    )
    for source, target in suffix_aliases:
        if text.endswith(source):
            text = text[: -len(source)] + target
            break
    if text.endswith("年度"):
        text = text[:-2] + "年"
    elif re.fullmatch(r"(?:19|20)\d{2}", text):
        # PDF 抽取可能把“2024年”拆成“2024”和下一行的“年”。
        # 裸四位年份与明确年度在绑定比较时应视为同一期间。
        text += "年"
    return text


# 页眉年份污染：财报每页页眉都写着“XX公司2025年半年度报告”，
# 该页所有数字因此被标注成粗粒度期间“2025年”，而问题问的是“2025年上半年”，
# 严格相等比较会把正确数字判为无证据，最终返回安全拒答。
# KB_LENIENT_PERIOD_MATCH=1 时放宽：同一年份下，一方是纯年粒度即视为兼容。
# 默认关闭，避免影响既有基线（窗口一 Chroma）。
_lenient_period_match = os.getenv("KB_LENIENT_PERIOD_MATCH", "").strip() in (
    "1",
    "true",
    "yes",
)

_PERIOD_SUBPERIOD_RE = re.compile(
    r"(上半年|下半年|第一季度|第二季度|第三季度|第四季度|前三季度)$"
)


# 入库侧用 "unknown" 作为「未识别出口径」的哨兵值（ingest.py 多处 or "unknown"）。
# 它表示"不知道"，不是"另一种口径"，因此不能参与互斥判定，
# 否则证据里明明精确匹配的数字会因证据未标注口径而被否掉。
_UNKNOWN_SCOPES = frozenset({"unknown", "未知", "未标注", "none", "null"})


def _scope_conflicts(claim_scope: Any, evidence_scope: Any) -> bool:
    """两个口径是否真实冲突；任一方为未知/空时视为不冲突。"""
    left = _normalize_scope(claim_scope)
    right = _normalize_scope(evidence_scope)
    if not left or not right:
        return False
    if left in _UNKNOWN_SCOPES or right in _UNKNOWN_SCOPES:
        return False
    return left != right


def _period_year(value: str) -> str:
    match = re.match(r"((?:19|20)\d{2})", str(value or ""))
    return match.group(1) if match else ""


def _periods_compatible(claim_period: Any, evidence_period: Any) -> bool:
    """期间是否互相兼容（用于数字核验的期间比较）。

    归一化后相等即兼容；任一方归一化后为空则视为不兼容——与原有严格比较
    （缺失/无法归一化的期间不能证明同一期间）保持一致，调用方的
    “双方都非空才比较”守卫只保证原始值非空，不保证归一化后非空。
    放宽开关打开时，同一年份下只要一方是纯年粒度（无半年/季度后缀）即兼容——
    粗粒度证据不否定细粒度答案，反之亦然。但“上半年 vs 下半年”“Q1 vs Q2”
    这类同年级别冲突仍然互斥。
    """
    left = _normalize_report_period(claim_period)
    right = _normalize_report_period(evidence_period)
    if not left or not right:
        return False
    if left == right:
        return True
    if not _lenient_period_match:
        return False
    year_left, year_right = _period_year(left), _period_year(right)
    if not year_left or not year_right or year_left != year_right:
        return False
    # 双方都带子期间（如 上半年 vs 下半年）→ 真实冲突，不兼容。
    if _PERIOD_SUBPERIOD_RE.search(left) and _PERIOD_SUBPERIOD_RE.search(right):
        return False
    return True


def _parse_financial_facts_json(value: Any) -> list[FinancialFact]:
    """解析 Luna A 的可选 financial_facts_json 契约，坏值按旧索引处理。"""
    raw = value
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
    if isinstance(raw, dict):
        raw = raw.get("facts", raw.get("financial_facts", []))
    if not isinstance(raw, list):
        return []

    facts: list[FinancialFact] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        metric = _canonical_metric(item.get("metric"))
        if not metric:
            continue
        raw_value = str(item.get("raw_value") or "").strip()
        unit = _normalize_fact_unit(item.get("unit"))
        raw_claims = _extract_numeric_claims(raw_value) if raw_value else []
        if not unit and raw_claims:
            unit = raw_claims[0].unit
        canonical = _parse_decimal(item.get("canonical_value"))
        if canonical is None and raw_claims:
            number = raw_claims[0].number
            canonical = number * _UNIT_SCALE.get(unit, Decimal("1"))
        if not raw_value and canonical is None:
            continue
        facts.append(
            FinancialFact(
                metric=metric,
                raw_value=raw_value,
                canonical_value=canonical,
                unit=unit,
                statement_scope=_normalize_scope(item.get("statement_scope")),
                report_period=_normalize_report_period(item.get("report_period")),
            )
        )
    return facts


def _financial_facts_from_metadata(metadata: dict[str, Any] | None) -> list[FinancialFact]:
    if not isinstance(metadata, dict):
        return []
    return _parse_financial_facts_json(metadata.get("financial_facts_json"))


def _extract_numeric_claims(text: str) -> list[NumericClaim]:
    """提取数字声明，支持逗号/空格/全角格式和常见中文单位。"""
    claims: list[NumericClaim] = []
    # 同一数值在“调整后/调整前”两列中合法重复；位置必须保留，才能
    # 计算它属于哪一个期间/口径列。
    seen: set[tuple[Decimal, str, bool, int]] = set()
    normalized = _numeric_text_for_matching(text)
    for match in _NUMBER_CLAIM_RE.finditer(normalized):
        # P：答案里的年份（2024/2023/2025 等无单位整数）不是财务数字，
        # 不能拿去证据核验，否则必然误判"数字未在证据中找到"。
        if _is_year_like_claim(match):
            continue
        raw = match.group(0).strip()
        matched_unit = (match.group(1) or "").strip()
        unit = matched_unit
        if not matched_unit and _looks_like_percentage_claim(normalized, match):
            unit = "%"
        number_text = re.sub(
            r"[,\s]", "", raw[: raw.rfind(matched_unit)] if matched_unit else raw
        )
        try:
            number = Decimal(number_text)
        except InvalidOperation:
            continue
        is_percent = unit in ("%", "％")
        if is_percent and number > 0 and _looks_like_decline_claim(normalized, match):
            # “下降22.55%”与表格中的“-22.55”是同一个变化事实；方向由
            # 文字表达，数值核验仍要求绝对值精确一致。
            number = -number
        key = (number, unit, is_percent, match.start())
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            NumericClaim(
                raw=raw,
                number=number,
                unit=unit,
                canonical_value=number * _UNIT_SCALE[unit],
                is_percent=is_percent,
                metric=_metric_for_numeric_match(normalized, match),
                start=match.start(),
                end=match.end(),
                line_metric=(
                    _metric_for_numeric_loose(normalized, match)
                    if _loose_metric_binding
                    else ""
                ),
            )
        )
    return claims


def _is_unbound_format_number(claim: NumericClaim, text: str = "") -> bool:
    """忽略答案叙述中的小型格式编号，保留真正绑定到事实的数字。

    例如“连续 3 年触及 2 项风险”不是财务事实。若把这类数字送入
    数值核验，严格模式会因为它们没有出现在财务证据里而丢弃本来正确的答案。
    有单位、百分比或指标标签的数字仍然必须核验。
    """
    if claim.unit or claim.is_percent or abs(claim.number) > Decimal("20"):
        return False
    if not claim.metric or not text or claim.start < 0:
        return not claim.metric

    # The extractor may inherit a metric from the preceding part of a long
    # sentence. Keep a small number only when the metric is immediately bound
    # to it; otherwise it is usually a narrative count/year/section number.
    nearby = text[max(0, claim.start - 24) : claim.start]
    return not bool(
        re.search(
            r"(?:收入|利润|费用|资产|负债|现金流|每股收益|毛利率|净利率)"
            r"\s*(?:为|是|达|至|约|同比|环比|增长|下降|增加|减少|达到)?\s*$",
            nearby,
        )
    )


_PERCENT_TAIL_RE = re.compile(r"[\s| ]*[%％]")
_PERCENT_HINT_RE = re.compile(r"同比|增减|增幅|增长率|下降率|百分比|百分点")


def _looks_like_percentage_claim(text: str, match: re.Match[str]) -> bool:
    """识别表格中“增减(%)”列省略百分号的数字。

    修复（重要）：原实现在数字 ±120 字符的邻域里找 % 或“增减/同比”字样。
    会计数据表的一行通常是：

        | 归属于上市公司股东的净利润(元) | -310,302,902.32 | ... | 73.24% | ...

    行尾“增减比例”列的 % 会让**同一行所有金额数字**都被误标成 unit='%'，
    核验时与答案里的“-310,302,902.32元”因 is_percent 不一致而被拒绝 ——
    数字完全相同却判为“无证据”，这是 zhongcheng 等题拒答的直接原因。

    百分号只应归属紧邻其前的那一个数字，因此改为：
      1) 数字后方紧邻出现 %（允许空格与表格分隔符）；或
      2) 数字前方紧邻范围内有“同比/增减”等比例语义词，且该词与数字之间
         没有其他数字（连续文本里的“同比增减18.87”）。
    """
    start, end = match.span()
    if _PERCENT_TAIL_RE.match(text[end : end + 8]):
        return True
    # 百分号也可能写在**指标名**里，数字换行紧随其后，例如：
    #   研发投入占营业收入的比例(%)\n22.32
    # 判据：% 紧邻数字之前，且 % 之前还有中文（说明它是指标名的一部分，
    # 而不是上一个单元格里的“73.24%”）。
    before = text[max(0, start - 8) : start]
    percent_mark = re.search(r"[%％]", before)
    if percent_mark and re.search(
        r"[\u4e00-\u9fff]", before[: percent_mark.start()]
    ):
        return True
    head = text[max(0, start - 20) : start]
    hint = None
    for candidate in _PERCENT_HINT_RE.finditer(head):
        hint = candidate
    if hint is None:
        return False
    # 关键词必须紧邻数字：会计数据表里“增减(%)”是**列标题**，它和真正的
    # 金额数字之间还隔着指标名（“增减(%)\n营业收入\n807,388,662.39”）。
    # 只检查“中间有没有别的数字”会把它也算成百分比。
    gap = head[hint.end() :]
    if len(gap) > 8 or re.search(r"\d", gap):
        return False
    return True


_DECLINE_RE = re.compile(r"下降|减少|下滑|降低|减幅|同比降|同比减")


def _looks_like_decline_claim(text: str, match: re.Match[str]) -> bool:
    """仅当下降语义词紧邻该数字**之前**、且中间没有别的数字时才取负。

    原实现在数字 ±24 字符的窗口里找“下降/减少”，会跨过表格单元格把
    **同行另一列**的下降语义错误归属过来。真实年报行：

        | 研发投入占营业收入的比例（%） | 22.32 | 37.00 | 减少14.68个百分点 |

    “减少”描述的是增减列（14.68 个百分点），不是当期值 22.32。旧逻辑给
    22.32 也取负成 -22.32，而答案写的是 22.32%，数字因此不匹配 —— 这是
    shenlian 一类题「有证据却被核验拒答」的直接原因。

    判据收紧为：下降词必须在数字**之前**（中文里“同比下降42.89%”都是
    词在前），且它与数字之间不能夹着别的数字（否则那是另一列/另一句的
    数值）。不再向后搜索：向后搜索会把“37.00 | 减少14.68”里的 37.00
    也误判为下降。
    """
    start, _end = match.span()
    before = text[max(0, start - 24) : start]
    hint = None
    for candidate in _DECLINE_RE.finditer(before):
        hint = candidate
    if hint is None:
        return False
    return not re.search(r"\d", before[hint.end() :])


def _looks_like_table_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return bool(
        ("|" in text and "\n" in text)
        or re.search(r"主要会计数据|主要财务指标|本报告期.*上年同期|单位[:：]", compact)
    )


def _table_unit_for_hit(hit: dict[str, Any]) -> str:
    """只为明确表格形态或单位标记的候选启用单位上下文。"""
    metadata = hit.get("metadata") or {}
    is_table = metadata.get("is_table")
    text = str(hit.get("text") or "")
    if not (
        is_table is True
        or str(is_table or "").strip().casefold() in {"true", "1", "yes"}
    ) and not _looks_like_table_text(text):
        return ""
    unit = unicodedata.normalize("NFKC", str(metadata.get("unit") or "")).strip()
    if unit in _TABLE_UNIT_WHITELIST:
        return unit
    unit_match = re.search(
        r"(?:单位|金额单位|货币单位)\s*[:：]?\s*(万亿|千亿|百亿|十亿|百万元|亿元|万元|千万|百万|十万|千元|港元|美元|亿|万|千|百|元)",
        text[:2000],
    )
    return unit_match.group(1) if unit_match else ""


def _is_year_like_claim(match: re.Match[str]) -> bool:
    """表格年份列保留原始无单位声明，不附加金额单位。"""
    if match.group(1):
        return False
    number_text = re.sub(r"[,\s]", "", match.group(0))
    try:
        number = Decimal(number_text)
    except InvalidOperation:
        return False
    return (
        number == number.to_integral_value()
        and Decimal("1900") <= number <= Decimal("2100")
    )


def _annotate_table_body_unit(text: str, unit: str) -> str:
    """生成仅供核验的表格正文副本；原文始终由调用方一并保留。"""
    normalized = _numeric_text_for_matching(text)

    def add_unit(match: re.Match[str]) -> str:
        if (
            match.group(1)
            or _is_year_like_claim(match)
            or _looks_like_percentage_claim(normalized, match)
        ):
            return match.group(0)
        return f"{match.group(0)}{unit}"

    return _NUMBER_CLAIM_RE.sub(add_unit, normalized)


def _table_body_evidence(hit: dict[str, Any]) -> list[str]:
    """返回正文原文及可选单位副本，不包含任何 metadata 行。"""
    text = str(hit.get("text") or "")
    if not text:
        return []
    evidence = [text]
    unit = _table_unit_for_hit(hit)
    if unit:
        annotated = _annotate_table_body_unit(text, unit)
        if annotated != text:
            evidence.append(annotated)
    return evidence


def _unit_family(unit: str) -> str:
    """返回可参与精确换算的单位族；不同币种不能只凭数值相等互换。"""
    if unit in ("%", "％"):
        return "percent"
    if unit in ("港元",):
        return "hkd"
    if unit in ("美元",):
        return "usd"
    return "rmb"


def _units_compatible(claim_unit: str, evidence_unit: str) -> bool:
    if not claim_unit or not evidence_unit:
        return not claim_unit and not evidence_unit
    if claim_unit in ("%", "％") or evidence_unit in ("%", "％"):
        return claim_unit in ("%", "％") and evidence_unit in ("%", "％")
    return _unit_family(claim_unit) == _unit_family(evidence_unit)


def _numeric_claim_supported(claim: NumericClaim, evidence_claims: list[NumericClaim]) -> bool:
    """判断数字是否被证据支持；单位换算只有 Decimal 精确相等才通过。"""
    for evidence in evidence_claims:
        if claim.is_percent != evidence.is_percent:
            continue
        if claim.metric and evidence.metric and claim.metric != evidence.metric:
            continue
        # 一旦答案已明确指标，而正文数字没有同一行/邻行指标标签，不能
        # 把它当作该指标的证据；这正是旧实现“数字袋子”会放过的错误。
        # 宽松行绑定：表格行标题被 PDF 拆到上一行时 metric 为空，此时退而
        # 用 line_metric（数字之前最近的行标题）定向确认归属，仍要求与
        # claim.metric 完全一致，不会退化成无归属的数字袋子。
        if claim.metric and not evidence.metric:
            if evidence.line_metric != claim.metric:
                continue
        if claim.adjustment and evidence.adjustment != claim.adjustment:
            continue
        if claim.report_period and evidence.report_period:
            if not _periods_compatible(claim.report_period, evidence.report_period):
                continue
        # 显式母公司/合并问题不能由另一个明确口径的正文数字替代；
        # 旧索引未保存口径时保持 unknown，不能因此否定正文中的精确数字。
        if _scope_conflicts(claim.statement_scope, evidence.statement_scope):
            continue
        # 旧索引可能只有 financial_metrics 和裸数字，没有保存单位字段。
        # 缺失单位不是“单位错误”：在没有显式相反单位、且数字已绑定唯一指标时，
        # 可沿用答案的金额单位；百分比仍必须在证据中明确出现。
        if claim.unit and not evidence.unit and not claim.is_percent:
            if claim.canonical_value == evidence.canonical_value:
                return True
        # DeepSeek occasionally omits the base currency unit even when the
        # evidence writes the exact amount in yuan. Accept only the same raw
        # number with explicit 元 evidence; scaled units remain ambiguous.
        if not claim.unit and evidence.unit == "元" and claim.number == evidence.number:
            return True
        if claim.number == evidence.number and claim.unit == evidence.unit:
            return True
        # 不同展示单位允许确定性换算，例如 1.2 亿 == 12000 万。
        if _units_compatible(claim.unit, evidence.unit):
            if claim.canonical_value == evidence.canonical_value:
                return True
    return False


def _requested_metric_keys(question: str) -> list[str]:
    """按问题中出现的顺序提取指标键，长别名优先且稳定去重。"""
    keys: list[str] = []
    for _start, _end, metric in _metric_mentions(str(question or "")):
        if metric not in keys:
            keys.append(metric)
    return keys


def _question_period_tokens(question: str) -> list[str]:
    text = unicodedata.normalize("NFKC", str(question or ""))
    tokens = re.findall(
        r"(?:19|20)\d{2}(?:年(?:度|上半年|下半年|半年度|第一季度|第二季度|第三季度|第四季度|一季度|二季度|三季度|四季度|前三季度)?|[-/]\d{1,2})?",
        text,
    )
    return [_normalize_report_period(token) for token in tokens if token]


def _question_scope(question: str) -> str:
    text = str(question or "")
    if "母公司" in text:
        return "parent"
    if "子公司" in text:
        return "subsidiary"
    if "合并" in text:
        return "consolidated"
    return ""


def _period_compatible(fact_period: str, question: str) -> bool:
    if not fact_period:
        return True
    question_periods = _question_period_tokens(question)
    if not question_periods:
        return True
    fact = _normalize_report_period(fact_period)
    # 年份是硬约束；半年度/季度等后缀在问题明确时也必须一致。
    if not any(period[:4] == fact[:4] for period in question_periods):
        return False
    for period in question_periods:
        suffix = period[4:]
        if suffix and suffix not in {"年", "年度"} and suffix not in fact:
            return False
    return True


def _fact_matches_question(fact: FinancialFact, question: str) -> bool:
    requested_scope = _question_scope(question)
    if requested_scope and fact.statement_scope and fact.statement_scope != requested_scope:
        return False
    return _period_compatible(fact.report_period, question)


def _facts_for_question(facts: list[FinancialFact], question: str) -> list[FinancialFact]:
    """先按期间/口径过滤；未写母公司时优先唯一的合并口径事实。"""
    compatible = [fact for fact in facts if _fact_matches_question(fact, question)]
    if _question_period_tokens(question):
        period_bound = [fact for fact in compatible if fact.report_period]
        if period_bound:
            # 同一证据里既有精确年度事实又有旧索引/残缺表头产生的空期间事实时，
            # 只使用精确年度事实，避免其他列的数字借“空期间”冒充目标年度。
            compatible = period_bound
    if _question_scope(question):
        return compatible
    if _prefers_consolidated_scope(question):
        consolidated = [fact for fact in compatible if fact.statement_scope == "consolidated"]
        if consolidated:
            return consolidated
    return compatible


def _fact_supports_claim(claim: NumericClaim, fact: FinancialFact) -> bool:
    if fact.canonical_value is None:
        return False
    if claim.is_percent != (fact.unit in ("%", "％")):
        return False
    if not _units_compatible(claim.unit, fact.unit):
        return False
    if claim.report_period and (
        not fact.report_period
        or _normalize_report_period(claim.report_period)
        != _normalize_report_period(fact.report_period)
    ):
        return False
    if claim.statement_scope and (
        not fact.statement_scope or claim.statement_scope != fact.statement_scope
    ):
        return False
    return claim.canonical_value == fact.canonical_value


def _evidence_records(evidence_entries: list[Any] | None) -> list[dict[str, Any]]:
    """把旧的字符串证据和带 metadata 的新证据统一成内部记录。"""
    records: list[dict[str, Any]] = []
    for entry in evidence_entries or []:
        if isinstance(entry, dict):
            text = str(entry.get("text") or "")
            metadata = entry.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
        else:
            text = str(entry or "")
            metadata = {}
        if text:
            records.append({"text": text, "metadata": metadata})
    return records


def _claim_with_metric(claim: NumericClaim, requested_metrics: list[str]) -> NumericClaim:
    if claim.metric or claim.line_metric or len(requested_metrics) != 1:
        return claim
    # 年份/页码等非财务数字不能被强行贴上唯一指标。
    if _is_metric_fallback_number(claim):
        return claim
    return NumericClaim(
        raw=claim.raw,
        number=claim.number,
        unit=claim.unit,
        canonical_value=claim.canonical_value,
        is_percent=claim.is_percent,
        metric=requested_metrics[0],
        report_period=claim.report_period,
        statement_scope=claim.statement_scope,
        start=claim.start,
        end=claim.end,
        line_metric=claim.line_metric,
        adjustment=claim.adjustment,
    )


def _is_metric_fallback_number(claim: NumericClaim) -> bool:
    """年份/页码等非财务数字不允许被"唯一指标兜底"强行贴标签。"""
    return (
        not claim.unit
        and claim.number == claim.number.to_integral_value()
        and Decimal("1900") <= claim.number <= Decimal("2100")
    )


def _resolve_claim_metric(
    claim: NumericClaim,
    text: str,
    fallback_metrics: list[str] | tuple[str, ...] | set[str] = (),
) -> str:
    """按固定优先级绑定指标；低优先级来源不得覆盖已识别的高优先级结果。

    优先级（高 → 低）：
      1. ``claim.metric``      —— 同一行/紧邻标签已明确绑定
      2. ``claim.line_metric`` —— 宽松行绑定（PDF 把行标题拆到上一行）
      3. 表格行推断            —— ``_table_row_metric_for_claim``
      4. 单指标兜底            —— 只剩一个候选指标时的最后手段

    回归点（格力 2023 年报第 7 页）：该页 metadata 的 financial_metrics
    只有"营业收入"，但正文把"经营活动产生的现金\\n流量净额（元）"拆成两行。
    若第 4 级兜底先于第 3 级执行，已识别为 operating_cashflow 的数字会被
    覆盖成 revenue，触发 metric mismatch，最终由 partial fallback 把
    "经营活动现金流量净额"整字段遮蔽成"无法确定"。
    """
    if claim.metric:
        return claim.metric
    if claim.line_metric:
        return claim.line_metric
    table_row_metric = _table_row_metric_for_claim(text, claim)
    if table_row_metric:
        return table_row_metric
    # 调用方可能传入 set（_financial_metric_keys_from_metadata 的返回类型），
    # 这里统一成序列后再取唯一值，避免对 set 做下标访问。
    fallbacks = tuple(fallback_metrics or ())
    if len(fallbacks) == 1 and not _is_metric_fallback_number(claim):
        return fallbacks[0]
    return ""


def _adjustment_for_question(question: str) -> str:
    compact = re.sub(r"\s+", "", str(question or ""))
    has_before = "调整前" in compact
    has_after = "调整后" in compact
    if has_before and not has_after:
        return "before"
    if has_after and not has_before:
        return "after"
    return ""


def _table_row_adjustment_for_claim(text: str, claim: NumericClaim) -> str:
    """从多列表头的子列位置绑定调整前/调整后，不按数字值猜测。"""
    if not claim.metric or claim.start < 0:
        return ""
    mentions = _metric_mentions(text)
    before = [item for item in mentions if item[1] <= claim.start and item[2] == claim.metric]
    if not before:
        return ""
    _metric_start, metric_end, _metric = before[-1]
    following = [item for item in mentions if item[0] > claim.start]
    row_end = following[0][0] if following else len(text)
    if claim.start >= row_end or claim.start - metric_end > 180:
        return ""
    row_claims = _extract_numeric_claims(text[metric_end:row_end])
    row_index = next(
        (index for index, item in enumerate(row_claims)
         if metric_end + item.start == claim.start),
        None,
    )
    if row_index is None:
        return ""
    header_lines = [
        line for line in text[:metric_end].splitlines()
        if "|" in line and ("调整前" in line or "调整后" in line)
    ]
    for line in header_lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        for column_index, cell in enumerate(cells):
            adjustment = (
                "before" if "调整前" in cell
                else "after" if "调整后" in cell
                else ""
            )
            if adjustment and column_index > 0 and row_index == column_index - 1:
                return adjustment
    return ""


def _claim_with_context(
    claim: NumericClaim,
    text: str,
    *,
    question: str = "",
    metadata: dict[str, Any] | None = None,
    fallback_metrics: list[str] | tuple[str, ...] = (),
) -> NumericClaim:
    """为数字补充仅能从邻近文字明确推出的期间/口径。

    ``fallback_metrics`` 是最低优先级的"单指标兜底"（旧索引 metadata 只留下
    一个指标标签）。它必须最后参与，否则会把表格行推断出的正确指标覆盖掉。
    """
    normalized = _numeric_text_for_matching(text)
    metric = _resolve_claim_metric(claim, normalized, fallback_metrics)
    start = claim.start
    if start < 0:
        return claim
    local_start = max(0, start - 120)
    local = normalized[local_start:start]
    is_table_context = _looks_like_table_text(normalized) or bool(
        isinstance(metadata, dict)
        and (
            metadata.get("is_table") is True
            or str(metadata.get("is_table") or "").strip().casefold()
            in {"true", "1", "yes"}
        )
    )
    contextual_claim = claim
    if metric != claim.metric:
        contextual_claim = NumericClaim(
            raw=claim.raw,
            number=claim.number,
            unit=claim.unit,
            canonical_value=claim.canonical_value,
            is_percent=claim.is_percent,
            metric=metric,
            report_period=claim.report_period,
            statement_scope=claim.statement_scope,
            start=claim.start,
            end=claim.end,
            line_metric=contextual_claim.line_metric,
            adjustment=contextual_claim.adjustment,
        )
    if is_table_context and not contextual_claim.is_percent and _table_row_claim_is_percentage(
        normalized, contextual_claim
    ):
        contextual_claim = NumericClaim(
            raw=contextual_claim.raw,
            number=contextual_claim.number,
            unit="%",
            canonical_value=contextual_claim.number,
            is_percent=True,
            metric=contextual_claim.metric,
            report_period=contextual_claim.report_period,
            statement_scope=contextual_claim.statement_scope,
            start=contextual_claim.start,
            end=contextual_claim.end,
            line_metric=contextual_claim.line_metric,
            adjustment=contextual_claim.adjustment,
        )
    table_period = (
        _table_row_period_for_claim(normalized, contextual_claim, metadata=metadata)
        if is_table_context
        else ""
    )
    near_periods = _question_period_tokens(normalized[max(0, start - 20) : start])
    period = table_period or (near_periods[-1] if near_periods else "")
    metadata_period = (
        _normalize_report_period(metadata.get("report_period"))
        if isinstance(metadata, dict)
        else ""
    )
    if (
        metadata_period
        and period
        and metadata_period[:4] == period[:4]
        and len(metadata_period) > len(period)
    ):
        period = metadata_period
    question_periods = _question_period_tokens(question)
    if contextual_claim.is_percent and len(question_periods) > 1:
        # 同比比例描述的是题目中的当前期与上一期之间的变化，不能被
        # 答案中紧邻的“2023年为……”文字误绑定到上一期。
        period = question_periods[0]
    if is_table_context and contextual_claim.is_percent and _table_row_claim_is_percentage(
        normalized, contextual_claim
    ):
        period = question_periods[0] if question_periods else ""
    adjustment = contextual_claim.adjustment or _adjustment_for_question(question)
    if is_table_context:
        adjustment = _table_row_adjustment_for_claim(normalized, contextual_claim) or adjustment
    periods = _question_period_tokens(local)
    if not period:
        period = periods[-1] if periods else ""
    if not period:
        question_periods = _question_period_tokens(question)
        if len(question_periods) == 1:
            period = question_periods[0]
    if not period and isinstance(metadata, dict):
        period = _normalize_report_period(metadata.get("report_period"))

    scope = ""
    local_compact = re.sub(r"\s+", "", local)
    if re.search(r"母公司", local_compact):
        scope = "parent"
    elif re.search(r"子公司", local_compact):
        scope = "subsidiary"
    elif re.search(r"合并", local_compact):
        scope = "consolidated"
    if not scope and isinstance(metadata, dict):
        scope = _normalize_scope(metadata.get("statement_scope"))
    if not scope and metadata is None:
        scope = _question_scope(question)
    return NumericClaim(
        raw=contextual_claim.raw,
        number=contextual_claim.number,
        unit=contextual_claim.unit,
        canonical_value=contextual_claim.canonical_value,
        is_percent=contextual_claim.is_percent,
        metric=contextual_claim.metric,
        report_period=period,
        statement_scope=scope,
        start=contextual_claim.start,
        end=contextual_claim.end,
        # 仅在明确表格上下文中传递受控的最近行指标；普通叙述不使用
        # “最近指标”推断，且始终受 KB_METRIC_LOOSE_BINDING 开关控制。
        line_metric=(
            contextual_claim.line_metric
            if _loose_metric_binding and is_table_context
            else ""
        ),
        adjustment=adjustment,
    )


def _table_row_bounds(text: str, claim: NumericClaim) -> tuple[int, int] | None:
    """识别纵向 PDF 表格中当前数字所属的标签行与数值行范围。"""
    if claim.start < 0:
        return None
    lines = list(re.finditer(r"[^\n]*(?:\n|$)", text))
    current_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.start() <= claim.start < line.end()
        ),
        None,
    )
    if current_index is None:
        return None
    current_prefix = text[lines[current_index].start() : claim.start]
    if re.search(r"\d", current_prefix):
        return None
    label_index = current_index - 1
    while label_index >= 0:
        label = text[lines[label_index].start() : lines[label_index].end()].strip()
        if label and not re.search(r"\d", label):
            break
        label_index -= 1
    if label_index < 0:
        return None
    row_end = len(text)
    for line in lines[current_index + 1 :]:
        value = text[line.start() : line.end()].strip()
        if value and not re.search(r"\d", value):
            row_end = line.start()
            break
    return lines[label_index].end(), row_end


def _table_row_claim_is_percentage(text: str, claim: NumericClaim) -> bool:
    if not claim.metric or claim.start < 0:
        return False
    mentions = _metric_mentions(text)
    before = [item for item in mentions if item[1] <= claim.start]
    if not before:
        return False
    _start, metric_end, metric = before[-1]
    if metric != claim.metric:
        return False
    following = [item for item in mentions if item[0] > claim.start]
    row_end = following[0][0] if following else len(text)
    if claim.start - metric_end > 180 or claim.start >= row_end:
        return False
    row_start = metric_end
    row_bounds = _table_row_bounds(text, claim)
    if row_bounds:
        row_start, row_end = row_bounds
    row_claims = _extract_numeric_claims(text[row_start:row_end])
    row_index = next(
        (
            index
            for index, item in enumerate(row_claims)
            if row_start + item.start == claim.start
        ),
        None,
    )
    if row_index is None:
        return False
    header = re.sub(r"\s+", "", text[:metric_end])
    if not re.search(r"同比|增减|增幅|增长率|下降率|百分比|百分点|\(%|％", header):
        return False
    if "调整后" in header and "调整前" in header and len(row_claims) >= 4:
        # 部分年报把本期、上期的“调整后/调整前”列展开，变化率列位于
        # 第四个数值；不能按“年份数量 - 1”猜列位。
        return row_index == 3
    periods = _question_period_tokens(text[:metric_end])
    if len(periods) >= 2:
        return row_index == max(len(periods[-4:]) - 1, 0)
    return row_index == len(row_claims) - 1


def _previous_report_period(period: str) -> str:
    match = re.match(r"((?:19|20)\d{2})(.*)$", _normalize_report_period(period))
    if not match:
        return ""
    return f"{int(match.group(1)) - 1}{match.group(2)}"


def _table_row_period_for_claim(
    text: str,
    claim: NumericClaim,
    metadata: dict[str, Any] | None = None,
) -> str:
    """按表头列顺序给跨行抽取的表格数字补期间。

    PDF 文本层常把“营业收入”单独放在数值列之后，不能只看同一行。这里
    只在已识别指标行、且表头有明确年份时建立有限的列绑定；普通叙述文本
    不会因为附近出现一个年份而被强行改写。
    """
    if not claim.metric or claim.start < 0:
        return ""
    mentions = _metric_mentions(text)
    before = [item for item in mentions if item[1] <= claim.start and item[2] == claim.metric]
    if not before:
        return ""
    _metric_start, metric_end, _metric = before[-1]
    following = [item for item in mentions if item[0] > claim.start]
    row_end = following[0][0] if following else len(text)
    if claim.start >= row_end or claim.start - metric_end > 180:
        return ""
    row_start = metric_end
    row_bounds = _table_row_bounds(text, claim)
    if row_bounds:
        row_start, row_end = row_bounds
    row_claims = [
        item
        for item in _extract_numeric_claims(text[row_start:row_end])
        if item.start >= 0
    ]
    if not row_claims:
        return ""
    row_claim_index = next(
        (index for index, item in enumerate(row_claims) if row_start + item.start == claim.start),
        None,
    )
    if row_claim_index is None:
        return ""
    metric_prefix = text[:metric_end]
    header_anchor = max(
        metric_prefix.rfind("主要会计数据"),
        metric_prefix.rfind("主要财务指标"),
    )
    header_text = metric_prefix[header_anchor:] if header_anchor >= 0 else metric_prefix
    header_periods = _question_period_tokens(header_text)[-4:]
    preceding_periods = _question_period_tokens(
        metric_prefix[:header_anchor] if header_anchor >= 0 else ""
    )
    percentage_claim = NumericClaim(
        raw=claim.raw,
        number=claim.number,
        unit=claim.unit,
        canonical_value=claim.canonical_value,
        is_percent=claim.is_percent,
        metric=claim.metric,
        report_period=claim.report_period,
        statement_scope=claim.statement_scope,
        start=claim.start,
        end=claim.end,
    )
    if _table_row_claim_is_percentage(text, percentage_claim):
        return ""
    # 取距离当前行最近的一组年份；通常就是 2024/2023/2022，且不把
    # 页眉中的报告年份混入当前表头。
    amount_index = 0
    for item in row_claims[:row_claim_index]:
        candidate = NumericClaim(
            raw=item.raw,
            number=item.number,
            unit=item.unit,
            canonical_value=item.canonical_value,
            is_percent=item.is_percent,
            metric=claim.metric,
            start=metric_end + item.start,
            end=metric_end + item.end,
        )
        if not _table_row_claim_is_percentage(text, candidate):
            amount_index += 1
    compact_header = re.sub(r"\s+", "", header_text)
    adjustment_layout = "调整后" in compact_header and "调整前" in compact_header
    if adjustment_layout and len(row_claims) >= 4:
        if amount_index == 0 and header_periods:
            header_period = header_periods[0]
        elif amount_index in (1, 2) and len(header_periods) >= 2:
            header_period = header_periods[1]
        elif amount_index >= 3:
            header_period = preceding_periods[-1] if preceding_periods else ""
            if not header_period and len(header_periods) >= 2:
                header_period = _previous_report_period(header_periods[1])
        else:
            header_period = ""
    elif amount_index >= len(header_periods):
        header_period = ""
    else:
        header_period = header_periods[amount_index]
    metadata_period = ""
    if isinstance(metadata, dict):
        metadata_period = _normalize_report_period(metadata.get("report_period"))
    if header_period:
        # “2025 年半年度”常只在元数据里保留，正文表头只写“本报告期
        # (1-6月)”。保留表头年份，同时用元数据补齐报告期间粒度。
        if (
            metadata_period
            and metadata_period[:4] == header_period[:4]
            and len(metadata_period) > len(header_period)
        ):
            return metadata_period
        return header_period
    if not metadata_period:
        return ""
    if amount_index == 0:
        return metadata_period
    if amount_index == 1 and re.search(r"上年同期|上年度|上期", text[:metric_end]):
        return _previous_report_period(metadata_period)
    return ""


def _table_row_metric_for_claim(text: str, claim: NumericClaim) -> str:
    if claim.start < 0:
        return ""
    mentions = _metric_mentions(text)
    before = [item for item in mentions if item[1] <= claim.start]
    if not before:
        return ""
    start, end, metric = before[-1]
    following = [item for item in mentions if item[0] > claim.start]
    row_end = following[0][0] if following else len(text)
    if claim.start - end > 180 or claim.start >= row_end:
        return ""
    return metric


def _table_evidence_has_column_context(records: list[dict[str, Any]]) -> bool:
    """判断同一证据是否明显包含本期/上期等多列，避免把它当成同一事实冲突。"""
    markers = re.compile(
        r"本报告期|上年同期|本期比上年|本年比上年|调整后|调整前|"
        r"(?:19|20)\d{2}年[^\n]{0,30}(?:19|20)\d{2}年"
    )
    for record in records:
        text = record["text"]
        if markers.search(text):
            return True
        compact = re.sub(r"\s+", "", text)
        if (
            re.search(r"本报告期|上年同期|本期比上年|本年比上年|调整后|调整前", compact)
            or len(_question_period_tokens(compact)) >= 2
        ):
            return True
    return False


def _structured_fact_conflict(facts: list[FinancialFact]) -> bool:
    """只在完整四元绑定键内检测互不相容值。"""
    values: dict[tuple[str, str, str, str], set[Decimal]] = {}
    for fact in facts:
        if (
            not fact.metric
            or fact.canonical_value is None
            or not fact.unit
            or not fact.report_period
            or not fact.statement_scope
        ):
            continue
        key = (
            fact.metric,
            _normalize_report_period(fact.report_period),
            fact.statement_scope,
            fact.unit,
        )
        values.setdefault(key, set()).add(fact.canonical_value)
    return any(len(items) > 1 for items in values.values())


def _claim_rejection_reasons(
    claim: Any, candidates: list[tuple[Any, Any, Any]]
) -> list[str]:
    """诊断用：逐条候选证据给出被拒绝的原因。只做记录，不参与任何判定。"""
    reasons: list[str] = []
    for candidate_claim, _record, _alias in candidates[:20]:
        reason = _match_rejection_reason(claim, candidate_claim)
        if reason:
            reasons.append(reason)
    unique: list[str] = []
    seen: set[str] = set()
    for reason in reasons:
        if reason not in seen:
            seen.add(reason)
            unique.append(reason)
    return unique[:8]


def _verify_numeric_claims(
    answer: str,
    evidence_texts: list[Any],
    question: str = "",
    trace_stage: str = "",
) -> tuple[bool, list[NumericClaim]]:
    """返回 (是否全部有证据, 未被支持的数字声明)。

    若候选带 financial_facts_json，先按 metric/期间/口径筛选结构化事实；
    只有没有可用结构化事实时才回退到带局部指标标签的正文核验。

    trace_stage：诊断追踪开启时，传入阶段名会把每条 claim 的绑定与拒绝原因
    记入追踪文件；为空则不追踪。默认空，生产行为完全不变。
    """
    requested_metrics = _requested_metric_keys(question)
    question_periods = _question_period_tokens(question)
    required_revenue_field = _strict_revenue_field(question)
    answer_claims: list[tuple[NumericClaim, str]] = []
    for raw_claim in _extract_numeric_claims(answer):
        if _is_unbound_format_number(raw_claim, answer):
            continue
        # 问句请求的指标同样只是最低优先级兜底；答案里已有明确行/邻近标签时
        # 不得被它覆盖。
        claim = _claim_with_context(
            raw_claim,
            answer,
            question=question,
            fallback_metrics=requested_metrics,
        )
        answer_claims.append(
            (claim, _metric_alias_for_numeric_claim(answer, claim))
        )
    records = _evidence_records(evidence_texts)
    text_claims: list[tuple[NumericClaim, dict[str, Any], str]] = []
    for record in records:
        metadata_metrics = list(
            _financial_metric_keys_from_metadata(record["metadata"])
        )
        table_record = {"text": record["text"], "metadata": record["metadata"]}
        evidence_text_variants = _table_body_evidence(table_record)
        for evidence_text in evidence_text_variants:
            for claim in _extract_numeric_claims(evidence_text):
                # metadata 单指标兜底只能作为最低优先级，走 fallback_metrics
                # 参与；在它之前先让行内绑定 / 宽松行绑定 / 表格行推断生效，
                # 避免旧索引只留下 revenue 时把现金流等指标覆盖掉。
                bound_claim = _claim_with_context(
                    claim,
                    evidence_text,
                    question=question,
                    metadata=record["metadata"],
                    fallback_metrics=metadata_metrics,
                )
                text_claims.append(
                    (
                        bound_claim,
                        record,
                        _metric_alias_for_numeric_claim(evidence_text, bound_claim),
                    )
                )
    fact_records = [
        (fact, record)
        for record in records
        for fact in _financial_facts_from_metadata(record["metadata"])
    ]
    facts = [fact for fact, _record in fact_records]
    unsupported: list[NumericClaim] = []
    # 只在诊断开启时收集每条 claim 的绑定/拒绝细节，关闭时是空列表追加开销。
    _trace_claims = bool(trace_stage) and diagnostic_trace.is_enabled()
    claim_diagnostics: list[dict[str, Any]] = []
    for claim, answer_alias in answer_claims:
        metric = claim.metric or (requested_metrics[0] if len(requested_metrics) == 1 else "")
        # 答案措辞把指标写成了问句没要求的等义别名（例如问“营业收入”、
        # 答“营业总收入”）时，只有证据池确实把该别名当作独立字段披露过，
        # 才算真的字段错配——那时两个字段各有各的值，借值才是错误。
        # 财报只披露了一个收入字段时，措辞差异不是借值，按问句请求的指标
        # 继续核对，避免把正确数字拒掉。
        if (
            metric
            and requested_metrics
            and metric != requested_metrics[0]
            and requested_metrics[0] in _equivalent_metric_group(metric)
            and not _evidence_declares_metric(
                records, fact_records, text_claims, metric
            )
        ):
            metric = requested_metrics[0]
            claim = replace(claim, metric=metric)
        relevant_facts = [fact for fact in facts if metric and fact.metric == metric]
        structured_supported = False
        structured_conflict = False
        if relevant_facts:
            matching_facts = _facts_for_question(relevant_facts, question)
            if matching_facts:
                # 期间/口径字段缺失或同一绑定键出现多个值时，结构化事实不能
                # 独占裁决；交给正文表格行继续核对。多列表中的本期/上期值
                # 不是同一绑定键的冲突，即使旧索引暂时把它们写成同一期间。
                conflict = _structured_fact_conflict(matching_facts)
                structured_conflict = conflict
                if not conflict:
                    complete = [
                        fact
                        for fact in matching_facts
                        if fact.unit
                        and fact.report_period
                        and fact.statement_scope
                    ]
                    structured_supported = any(
                        _fact_supports_claim(claim, fact)
                        and any(
                            id(candidate_fact) == id(fact)
                            and _strict_revenue_fact_allowed(
                                required_revenue_field, candidate_record, fact
                            )
                            for candidate_fact, candidate_record in fact_records
                        )
                        for fact in complete
                    )

        evidence_claim = claim
        if metric and not claim.metric:
            evidence_claim = _claim_with_metric(claim, [metric])
        candidate_text_claims = text_claims
        if len(question_periods) > 1 and claim.report_period:
            # 多期间问题中，未带期间的证据数字不能替代已明确标注的目标列。
            candidate_text_claims = [
                item
                for item in text_claims
                if item[0].report_period
                and _normalize_report_period(item[0].report_period)
                == _normalize_report_period(claim.report_period)
            ]
        if required_revenue_field == "营业收入":
            text_supported = any(
                _numeric_claim_supported(evidence_claim, [item[0]])
                and _strict_revenue_record_allowed(
                    required_revenue_field,
                    item[1],
                    item[0],
                )
                for item in candidate_text_claims
            )
        else:
            text_supported = _numeric_claim_supported(
                evidence_claim, [item[0] for item in candidate_text_claims]
            )
        if (
            required_revenue_field == "营业收入"
            and answer_alias == "主营业务收入"
        ):
            structured_supported = False
            text_supported = False
        if _trace_claims:
            claim_diagnostics.append(
                {
                    "claim": diagnostic_trace.claim_record(claim),
                    "answer_alias": answer_alias,
                    "metric_used": metric,
                    "structured_supported": structured_supported,
                    "structured_conflict": structured_conflict,
                    "text_supported": text_supported,
                    "relevant_fact_count": len(relevant_facts),
                    "candidate_evidence_count": len(candidate_text_claims),
                    "rejection_reasons": _claim_rejection_reasons(
                        evidence_claim, candidate_text_claims
                    ),
                }
            )
        if structured_conflict and not _table_evidence_has_column_context(records):
            unsupported.append(claim)
            continue
        if not structured_supported and not text_supported:
            unsupported.append(claim)
    if _trace_claims:
        diagnostic_trace.add_stage(
            trace_stage,
            {
                "supported": not unsupported,
                "evidence_record_count": len(records),
                "fact_count": len(facts),
                "requested_metrics": list(requested_metrics),
                "required_revenue_field": required_revenue_field,
                "answer_claims": diagnostic_trace.claim_snapshot(
                    [item[0] for item in answer_claims]
                ),
                "unsupported_claims": diagnostic_trace.claim_snapshot(unsupported),
                "claim_diagnostics": claim_diagnostics,
            },
        )
    return not unsupported, unsupported


def _has_explicit_company_subject(question: str) -> bool:
    """识别财务问句中的公司全称、简称或股票代码，避免无谓再次澄清。"""
    text = str(question or "").strip()
    if _has_explicit_company(text):
        return True
    match = _EXPLICIT_SUBJECT_RE.search(text)
    if not match:
        return False
    return match.group("subject") not in _GENERIC_SUBJECT_WORDS


def _requested_financial_items(question: str) -> list[str]:
    """提取问题中明确要求的多个财务项目，用于保守的漏答检测。"""
    text = str(question or "")
    found: list[str] = []
    for item in sorted(_FINANCIAL_ITEM_PATTERNS, key=len, reverse=True):
        if item in text and not any(item in existing for existing in found):
            found.append(item)
    return sorted(found, key=text.find)


def _cause_terms_from_evidence(evidence_texts: list[Any]) -> list[str]:
    """提取证据中明确的原因短语，供原因题做保守完整性检查。"""
    terms: list[str] = []
    for record in _evidence_records(evidence_texts):
        for line in re.split(r"[\n。！？；]", record["text"]):
            line = line.strip()
            if not line:
                continue
            match = re.search(
                r"(?:主要(?:原因)?(?:系|是)|(?:变动|变化)原因(?:是|为)|原因(?:是|为)|由于)"
                r"\s*[：:，,]?\s*(?P<term>[^，,。；\n]{3,80})",
                line,
            )
            if not match:
                match = re.search(
                    r"受(?P<term>[^，,。；\n]{2,40})影响",
                    line,
                )
            if not match:
                continue
            term = match.group("term").strip(" ：:，,")
            term = re.sub(r"^(?:见|详见|参见|未提供|未说明|暂无|没有)\s*", "", term)
            term = re.sub(r"所致$", "", term).strip(" ：:，,")
            if not term or term in {"正文", "相关说明", "说明"}:
                continue
            if term and term not in terms:
                terms.append(term)
    return terms[:5]


def _is_refusal_answer(answer: str) -> bool:
    """识别模型明确拒答，供无关证据时转人工。"""
    return bool(
        re.search(
            r"未检索到|未找到|无法(?:确认|确定|回答)|暂时不能给出|"
            r"请提供[^。！？\n]*(?:全称|名称|代码)|不确定",
            str(answer or ""),
        )
    )


def _answer_mentions_unresolved_item(item: str, answer: str) -> bool:
    """判断答案是否明确把某个字段标为无法确定，而非悄悄漏答。"""
    unresolved = (
        r"无法(?:准确)?(?:确定|确认|回答|可靠(?:对应|确认))|"
        r"不能(?:给出|确定|确认)|暂时不能(?:给出|确定)|"
        r"未(?:检索到|找到|提供)|缺少|暂无"
    )
    escaped_item = re.escape(item)
    return bool(
        re.search(
            rf"(?:{escaped_item}[^。！？；;，,\n]{{0,48}}(?:{unresolved})|"
            rf"(?:{unresolved})[^。！？；;，,\n]{{0,48}}{escaped_item})",
            str(answer or ""),
        )
    )


# 调试开关：KB_DEBUG_VERIFY=1 时，把核验失败的 claim 与候选证据逐字段打到日志，
# 用于定位“数字明明在证据里却判失败”的具体维度。默认关闭。
_debug_verify = os.getenv("KB_DEBUG_VERIFY", "").strip() in ("1", "true", "yes")


def _log_verification_miss(
    question: str,
    answer: str,
    evidence_texts: list[Any],
    unsupported: list[NumericClaim],
) -> None:
    """把未通过核验的数字声明与候选证据逐字段对比后记录，便于定位失配维度。"""
    requested_metrics = _requested_metric_keys(question)
    records = _evidence_records(evidence_texts)
    facts = [
        fact
        for record in records
        for fact in _financial_facts_from_metadata(record["metadata"])
    ]
    lines: list[str] = [f"[verify-miss] question={question!r}"]
    lines.append(f"[verify-miss] requested_metrics={requested_metrics}")
    for claim in unsupported:
        lines.append(
            f"[verify-miss] claim raw={claim.raw!r} number={claim.number} unit={claim.unit!r} "
            f"metric={claim.metric!r} period={claim.report_period!r} scope={claim.statement_scope!r}"
        )
    lines.append(f"[verify-miss] 结构化 facts 共 {len(facts)} 条：")
    for fact in facts[:20]:
        lines.append(
            f"[verify-miss]   fact metric={fact.metric!r} raw={fact.raw_value!r} unit={fact.unit!r} "
            f"period={fact.report_period!r} scope={fact.statement_scope!r}"
        )
    lines.append(f"[verify-miss] 证据记录共 {len(records)} 条：")
    for record in records[:8]:
        text_claims = _extract_numeric_claims(str(record.get("text") or ""))
        lines.append(
            f"[verify-miss]   record page={record['metadata'].get('page')} "
            f"数字={[c.raw for c in text_claims][:8]}"
        )
    # 逐候选打印字段，并说明每个未通过声明被拒的具体维度
    candidates: list[NumericClaim] = []
    for record in records:
        metadata_metrics = list(_financial_metric_keys_from_metadata(record["metadata"]))
        for evidence_text in _table_body_evidence(
            {"text": record["text"], "metadata": record["metadata"]}
        ):
            for raw_claim in _extract_numeric_claims(evidence_text):
                candidates.append(
                    _claim_with_context(
                        raw_claim,
                        evidence_text,
                        question=question,
                        metadata=record["metadata"],
                        fallback_metrics=metadata_metrics,
                    )
                )
    lines.append(f"[verify-miss] 候选证据数字共 {len(candidates)} 个（前 24 个）：")
    for cand in candidates[:24]:
        lines.append(
            f"[verify-miss]   cand raw={cand.raw!r} number={cand.number} unit={cand.unit!r} "
            f"metric={cand.metric!r} period={cand.report_period!r} scope={cand.statement_scope!r}"
        )
    for claim in unsupported[:3]:
        for cand in candidates[:24]:
            reason = _match_rejection_reason(claim, cand)
            lines.append(
                f"[verify-miss] 判定 {claim.raw!r} vs {cand.raw!r} -> "
                f"{'通过' if reason is None else reason}"
            )
    logger.info("\n".join(lines))


def _is_plan_commitment_question(question: str) -> bool:
    """识别同时要求经营计划金额和业绩承诺判断的窄问题契约。"""
    text = unicodedata.normalize("NFKC", str(question or ""))
    return bool(
        _PLAN_COMMITMENT_QUESTION_RE.search(text)
        and _COMMITMENT_QUESTION_RE.search(text)
        and _COMMITMENT_JUDGEMENT_RE.search(text)
    )


def _commitment_polarities(text: str) -> set[str]:
    """提取业绩承诺结论的极性；询问式“是否构成”不算答案结论。"""
    polarities: set[str] = set()
    for sentence in re.split(r"[\n。！？；]", unicodedata.normalize("NFKC", str(text or ""))):
        if not _COMMITMENT_QUESTION_RE.search(sentence):
            continue
        if _COMMITMENT_NEGATIVE_RE.search(sentence):
            polarities.add("negative")
            continue
        if _COMMITMENT_POSITIVE_RE.search(sentence) and not re.search(
            r"是否|是不是|会不会|能否|能不能|算不算", sentence
        ):
            polarities.add("positive")
    return polarities


def _has_supported_plan_amount(answer: str, evidence_texts: list[Any]) -> bool:
    """要求计划金额既出现在答案，也由带计划语义的当前证据精确支持。"""
    if not _PLAN_AMOUNT_ANSWER_RE.search(answer):
        return False
    answer_claims = [
        claim
        for claim in _extract_numeric_claims(answer)
        if not _is_unbound_format_number(claim, answer)
    ]
    if not answer_claims:
        return False
    for record in _evidence_records(evidence_texts):
        text = record["text"]
        if not _PLAN_AMOUNT_ANSWER_RE.search(text):
            continue
        evidence_claims = _extract_numeric_claims(text)
        if any(
            _numeric_claim_supported(claim, evidence_claims)
            for claim in answer_claims
        ):
            return True
    return False


def _plan_commitment_verification_reasons(
    question: str, answer: str, evidence_texts: list[Any]
) -> list[str]:
    """对计划金额/业绩承诺双字段做证据绑定，不建立第二套评分系统。"""
    if not _is_plan_commitment_question(question):
        return []

    reasons: list[str] = []
    if not _has_supported_plan_amount(answer, evidence_texts):
        reasons.append("经营计划金额缺少当前证据支持或答案未给出计划金额结论")

    answer_polarities = _commitment_polarities(answer)
    evidence_polarities = {
        polarity
        for record in _evidence_records(evidence_texts)
        for polarity in _commitment_polarities(record["text"])
    }
    if len(answer_polarities) != 1:
        reasons.append("答案缺少明确的构成/不构成业绩承诺结论")
    elif answer_polarities != evidence_polarities:
        reasons.append("业绩承诺结论缺少当前证据中的同语义支持")
    return reasons


def _match_rejection_reason(claim: NumericClaim, evidence: NumericClaim) -> str | None:
    """返回候选证据被拒绝的原因；None 表示这一对可以判通过。"""
    if claim.is_percent != evidence.is_percent:
        return "is_percent 不一致"
    if claim.metric and evidence.metric and claim.metric != evidence.metric:
        return f"metric 不一致({claim.metric} vs {evidence.metric})"
    if claim.metric and not evidence.metric:
        return "证据缺 metric 标签"
    if claim.report_period and evidence.report_period and not _periods_compatible(
        claim.report_period, evidence.report_period
    ):
        return f"期间不兼容({claim.report_period} vs {evidence.report_period})"
    if _scope_conflicts(claim.statement_scope, evidence.statement_scope):
        return f"口径不一致({claim.statement_scope} vs {evidence.statement_scope})"
    if claim.unit and not evidence.unit and not claim.is_percent:
        if claim.canonical_value == evidence.canonical_value:
            return None
    if not claim.unit and evidence.unit == "元" and claim.number == evidence.number:
        return None
    if claim.number == evidence.number and claim.unit == evidence.unit:
        return None
    if _units_compatible(claim.unit, evidence.unit):
        if claim.canonical_value == evidence.canonical_value:
            return None
    return "数值/单位不相等"


def _answer_verification_reasons(
    question: str,
    answer: str,
    evidence_texts: list[Any],
) -> list[str]:
    """纯规则生成核验失败原因；无证据时不把诚实拒答误判为失败。"""
    if not evidence_texts:
        return []
    reasons: list[str] = []
    supported, unsupported = _verify_numeric_claims(
        answer,
        evidence_texts,
        question=question,
        trace_stage="numeric_verification",
    )
    if unsupported and _debug_verify:
        _log_verification_miss(question, answer, evidence_texts, unsupported)
    if not supported:
        if _has_strict_revenue_field_mismatch(question, answer, evidence_texts):
            reasons.append(
                "问题要求营业收入，答案或证据仅绑定到主营业务收入，不能视为同一字段。"
            )
        else:
            values = ", ".join(claim.raw for claim in unsupported[:5])
            reasons.append(
                "以下字段的数字未在证据中找到或无法精确换算；保留其他有证据字段，"
                f"仅将这些字段标为无法确定：{values}"
            )
    reasons.extend(
        _plan_commitment_verification_reasons(question, answer, evidence_texts)
    )

    requested_items = _requested_financial_items(question)
    answer_items = [item for item in requested_items if item in answer]
    answer_numbers = _extract_numeric_claims(answer)
    unresolved_items = [
        item
        for item in requested_items
        if _answer_mentions_unresolved_item(item, answer)
    ]
    answered_items = [
        item for item in answer_items if item not in unresolved_items
    ]
    if len(requested_items) >= 2 and (
        len(answer_items) + len(unresolved_items) < len(requested_items)
        and len(answer_numbers) < len(requested_items)
    ):
        missing = "、".join(
            item
            for item in requested_items
            if item not in answer and item not in unresolved_items
        )
        reasons.append(f"问题要求多个项目，但答案可能遗漏：{missing}")

    if _is_reason_question(question):
        cause_terms = _cause_terms_from_evidence(evidence_texts)
        refusal = _is_refusal_answer(answer)
        has_cause_marker = bool(
            re.search(
                r"主要(?:原因)?|由于|因为|受[^。！？\n]{1,18}影响|导致|带动|推动|"
                r"得益于|源于|所致",
                answer,
            )
        )
        if cause_terms and (refusal or not has_cause_marker):
            reasons.append(
                "问题要求说明变动原因，证据已包含明确原因，但答案未完整引用："
                + "、".join(cause_terms[:3])
            )

    # 多指标答案允许“已支持字段 + 其他字段无法确定”。这种局部保留不是
    # 对整题的拒答；只有没有任何已回答字段时，才按显式主体拒答处理。
    partial_field_response = bool(answer_numbers and answered_items)
    refusal = _is_refusal_answer(answer)
    if (
        refusal
        and _has_explicit_company_subject(question)
        and not partial_field_response
    ):
        reasons.append("问题已有明确公司主体且证据非空，不应再次拒答或索要全称")
    return reasons


def _partial_numeric_fallback_answer(
    question: str,
    answer: str,
    evidence_texts: list[Any],
) -> str | None:
    """混合数字答案最终重试失败时，仅遮蔽未被支持的数字。

    只对至少包含一个财务项目、且同时存在已支持和未支持数字的答案生效。
    单字段问题也可能被模型附带一个未被当前证据支持的同比百分比；这种情况
    应只遮蔽该附带数字，不能因为它不是题目要求的字段就抹掉已支持的主值。
    字符位置来自同一套确定性核验结果，避免用字符串替换误伤重复的合法数字；
    若替换后仍有漏答/绑定问题，则交回原有整题安全说明。
    """
    if not _requested_financial_items(question):
        return None
    supported, unsupported = _verify_numeric_claims(
        answer, evidence_texts, question=question
    )
    if supported or not unsupported:
        return None
    normalized_answer_claims = [
        claim
        for claim in _extract_numeric_claims(answer)
        if not _is_unbound_format_number(claim, answer)
    ]
    if len(unsupported) >= len(normalized_answer_claims):
        return None
    unsupported_claims = [claim for claim in unsupported if claim.start >= 0]
    if not unsupported_claims:
        return None
    safe_answer = unicodedata.normalize("NFKC", str(answer or ""))
    replacements: list[tuple[int, int]] = []
    for claim in sorted(unsupported_claims, key=lambda item: item.start, reverse=True):
        start = safe_answer.find(claim.raw, max(claim.start, 0))
        if start < 0:
            return None
        replacements.append((start, start + len(claim.raw)))
        safe_answer = f"{safe_answer[:start]}无法确定{safe_answer[start + len(claim.raw):]}"
    if _answer_verification_reasons(question, safe_answer, evidence_texts):
        return None
    return safe_answer


def _is_obvious_kb_question(question: str) -> bool:
    """识别高置信度的企业报告问句，作为不依赖 LLM 的稳定快速路径。"""
    text = str(question or "").strip()
    return bool(
        text
        and _KB_MATERIAL_RE.search(text)
        and _QUESTION_MARKER_RE.search(text)
        and not _NON_KB_INTENT_RE.search(text)
    )


def _prefers_consolidated_scope(question: str) -> bool:
    """财务问题未明确写母公司时，默认按上市公司合并口径理解。"""
    text = str(question or "")
    return bool(_FINANCIAL_SCOPE_RE.search(text) and "母公司" not in text)


def _question_subject_hints(question: str) -> list[str]:
    """提取第二次检索可复用的公司主体提示，不对主体做开放式猜测。"""
    text = str(question or "")
    hints: list[str] = []
    for pattern in (_COMPANY_FULL_NAME_RE, _COMPANY_ST_ALIAS_RE):
        for match in pattern.finditer(text):
            value = match.group(0).strip()
            if value and value not in hints:
                hints.append(value)
    code = _STOCK_CODE_RE.search(text)
    if code and code.group(0) not in hints:
        hints.append(code.group(0))
    metric_pattern = "|".join(
        re.escape(alias)
        for _metric, aliases in _FINANCIAL_METRIC_ALIASES
        for alias in aliases
    )
    subject_match = re.search(
        rf"(?P<subject>[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9（）()·&.\-]{{1,20}}?)"
        rf"(?=(?:19|20)\d{{2}}年|{metric_pattern})",
        text,
    )
    if subject_match:
        value = subject_match.group("subject").strip(" 的")
        if value and value not in _GENERIC_SUBJECT_WORDS and value not in hints:
            hints.append(value)
    return hints[:3]


def _is_reason_question(question: str) -> bool:
    return bool(re.search(r"原因|为什么|为何|主要系|所致|变动原因|变化原因|下降原因|增长原因", str(question or "")))


def _chapter_hints(question: str) -> list[str]:
    text = str(question or "")
    if re.search(r"经营情况概述|经营情况|经营回顾|管理层讨论与分析", text):
        return ["经营情况概述", "经营情况", "管理层讨论与分析"]
    return []


def _retrieval_queries(
    question: str,
    rewritten: str = "",
    retrieval_attempt: int = 0,
) -> list[str]:
    """保序构造有限检索表达，原问题永远保留。

    多指标问题才拆出独立指标查询；原因/章节查询只追加有限同义表达，
    第二次门槛降级会把公司、年份、指标和章节组合成精确查询。
    """
    original = str(question or "").strip()
    rewritten = str(rewritten or "").strip()
    queries = [original, rewritten]
    metric_keys = _requested_metric_keys(original)
    if len(metric_keys) >= 2:
        for metric in metric_keys:
            display = _METRIC_DISPLAY_NAMES.get(metric, metric)
            queries.append(f"{original} 指标：{display}")

    chapter_hints = _chapter_hints(original)
    if chapter_hints and metric_keys:
        metric_words: list[str] = []
        for metric in metric_keys:
            metric_words.extend(_METRIC_QUERY_SYNONYMS.get(metric, (_METRIC_DISPLAY_NAMES.get(metric, metric),)))
        queries.append(f"{' '.join(chapter_hints[:2])} {'/'.join(dict.fromkeys(metric_words))}")

    if _is_reason_question(original):
        queries.append(f"{original} 主要系 原因 所致 变动原因")

    if _prefers_consolidated_scope(original):
        queries.append(f"{original} 合并财务报表 合并口径")

    if retrieval_attempt > 0:
        subject = " ".join(_question_subject_hints(original))
        periods = " ".join(_question_period_tokens(original))
        metrics = " ".join(
            _METRIC_QUERY_SYNONYMS.get(metric, (_METRIC_DISPLAY_NAMES.get(metric, metric),))[0]
            for metric in metric_keys
        )
        chapters = " ".join(chapter_hints[:2])
        exact_terms = " ".join(part for part in (subject, periods, metrics, chapters) if part)
        if exact_terms:
            queries.append(f"{original} 精确匹配 {exact_terms}")

    return list(dict.fromkeys(query for query in queries if query))[:_MAX_RETRIEVAL_QUERIES]


def _company_fact_label(question: str) -> str | None:
    """返回需要明确公司的敏感事实类型；普通问题返回 None。"""
    text = question or ""
    for label, pattern in _COMPANY_FACT_PATTERNS:
        if pattern.search(text):
            return label
    return None


def _has_explicit_company(text: str) -> bool:
    """判断文本中是否有可解释的公司主体标识。"""
    text = text or ""
    if _COMPANY_FULL_NAME_RE.search(text):
        return True
    if _COMPANY_ST_ALIAS_RE.search(text):
        return True
    if _STOCK_CODE_RE.search(text):
        return True

    # 支持“广道的上市时间”“广道上市时间”等常见简称，同时排除“公司/本公司”等泛称。
    for match in _SHORT_NAME_BEFORE_FACT_RE.finditer(text):
        candidate = _QUESTION_PREFIX_RE.sub("", match.group("name")).strip()
        if not candidate:
            continue
        if candidate in _GENERIC_COMPANY_SUBJECTS:
            continue
        if any(word in candidate for word in ("股票", "证券", "这家", "本公司", "该公司")):
            continue
        return True
    return False


def _company_clarification(question: str, history: list[dict[str, str]]) -> str | None:
    """敏感企业事实缺少主体时生成确定性的澄清话术。

    只读取 role=user 的最近历史；助手曾经猜测或回答过的公司不能确认主体。
    """
    fact = _company_fact_label(question)
    if not fact or _has_explicit_company(question):
        return None
    recent_user_history = [
        str(message.get("content", ""))
        for message in (history or [])[-8:]
        if message.get("role") == "user"
    ]
    if any(_has_explicit_company(message) for message in recent_user_history):
        return None
    return f"请问您想查询哪家公司的{fact}？请提供公司名称或股票代码。"

# LLM 生成调用超时（秒）：Ollama/网络 hang 时不拖死请求，及时报错让上层处理。
# 可用 LLM_TIMEOUT 环境变量覆盖，默认 60 秒。
_LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60") or 60)


def _merge_llm_extra_body(
    extra_body: dict[str, Any] | None,
    reasoning_effort: str | None,
) -> dict[str, Any] | None:
    """Merge optional OpenAI-compatible fields without changing the default request.

    ``reasoning_effort`` is intentionally nested in ``extra_body``.  This works
    with the older OpenAI SDK range supported by the project and is accepted by
    Ollama's OpenAI-compatible endpoint.  A caller-provided body is copied so
    unrelated provider-specific fields remain intact.
    """
    merged = dict(extra_body or {})
    if reasoning_effort is not None and str(reasoning_effort).strip():
        merged["reasoning_effort"] = reasoning_effort
    return merged or None


def _llm_invoke(
    llm,
    messages: list[dict[str, str]],
    model: str = "deepseek-chat",
    *,
    reasoning_effort: str | None = None,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """统一 LLM 调用：兼容 OpenAI 客户端与 LangChain 风格 LLM。

    - OpenAI 客户端（本项目实际使用）：llm.chat.completions.create(model=model, ...)
    - LangChain 风格（mock/其他）：llm.invoke(messages)

    model 显式传参（P1 修复：去掉 llm._model 私有属性 hack，openai 升级不失效）。
    timeout（P0）：防 LLM/网络 hang 拖死请求。
    """
    if hasattr(llm, "invoke"):
        return str(llm.invoke(messages)).strip()
    # OpenAI 兼容客户端
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1024,
        "timeout": _LLM_TIMEOUT,
    }
    merged_extra_body = _merge_llm_extra_body(extra_body, reasoning_effort)
    if merged_extra_body is not None:
        request_kwargs["extra_body"] = merged_extra_body
    resp = llm.chat.completions.create(**request_kwargs)
    return (resp.choices[0].message.content or "").strip()


def _llm_stream(
    llm,
    messages: list[dict[str, str]],
    model: str = "deepseek-chat",
    *,
    reasoning_effort: str | None = None,
    extra_body: dict[str, Any] | None = None,
):
    """流式 LLM 调用（P1-2）：逐 token yield。兼容 OpenAI 客户端与 LangChain 风格。"""
    if hasattr(llm, "stream_invoke"):
        # LangChain 风格（FakeLLM 等）
        for chunk in llm.stream_invoke(messages):
            yield chunk
    else:
        # OpenAI 兼容客户端：stream=True 逐 token
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1024,
            "timeout": _LLM_TIMEOUT,
            "stream": True,
        }
        merged_extra_body = _merge_llm_extra_body(extra_body, reasoning_effort)
        if merged_extra_body is not None:
            request_kwargs["extra_body"] = merged_extra_body
        stream = llm.chat.completions.create(**request_kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta


def _answerability_threshold() -> float:
    """可回答性门槛阈值：top-1 证据分低于它视为证据不足。

    P2：默认 0.5（客服场景门槛应默认开启，防止低分检索硬答胡说 → 触发转人工）。
    设 KB_ANSWERABILITY_SCORE=0 可显式关闭门槛。
    """
    raw = os.getenv("KB_ANSWERABILITY_SCORE", "")
    try:
        return float(raw) if raw else 0.5
    except ValueError:
        return 0.5


# P2-3：可选降本开关——top-1 证据分高时跳过 LLM 评估。默认关闭（见 node_evaluate）。
_skip_eval_on_high_score = bool(
    os.getenv("KB_SKIP_EVAL_ON_HIGH_SCORE", "").strip() in ("1", "true", "yes")
)


def _evidence_score(hit: dict[str, Any]) -> float:
    """从命中里提取证据分，统一归一到 0-1。

    优先级：rerank_score（LLM/crossencoder 都是 0-10 量纲，除以 10）> 余弦 score（0-1）。
    BM25-only 命中的 chunk 无向量分（score 为 None 或 0），但有 rerank_score 就用 rerank 分；
    两者都没有时给中性分 0.5——关键词精确命中不能因缺余弦分而被误判"答不了"去转人工。
    """
    rs = hit.get("rerank_score")
    if rs is not None:
        try:
            return max(0.0, min(float(rs) / 10.0, 1.0))
        except (TypeError, ValueError):
            pass
    score = hit.get("score")
    if score is not None:
        try:
            s = float(score)
            if s > 0:
                return min(s, 1.0)
        except (TypeError, ValueError):
            pass
    return 0.5  # BM25-only 命中：中性分，不误判


def _format_candidate_metadata(hit: dict[str, Any]) -> str:
    """把候选 metadata 展平成模型可读的一行；缺字段也保持兼容。"""
    metadata = hit.get("metadata") or {}
    title = metadata.get("doc_title") or hit.get("doc_title") or "未提供"
    page_start = metadata.get("page_start") or metadata.get("page")
    page_end = metadata.get("page_end")
    if page_start is None:
        page = "未提供"
    elif page_end is not None and page_end != page_start:
        page = f"{page_start}-{page_end}"
    else:
        page = str(page_start)
    def shown(key: str) -> Any:
        value = metadata.get(key)
        return "未提供" if value is None or value == "" else value

    scope = shown("statement_scope")
    scope_key = str(scope).strip().casefold()
    scope_label = _SCOPE_LABELS.get(scope_key)
    if scope_label is None:
        scope_label = "未知" if scope == "未提供" else str(scope)

    facts = _financial_facts_from_metadata(metadata)
    if facts:
        facts_text = " | ".join(
            ", ".join(
                part
                for part in (
                    f"metric={fact.metric}",
                    f"raw_value={fact.raw_value or '未提供'}",
                    f"unit={fact.unit or '未提供'}",
                    f"statement_scope={fact.statement_scope or '未提供'}",
                    f"report_period={fact.report_period or '未提供'}",
                )
                if part
            )
            for fact in facts
        )
    else:
        facts_text = "未提供"

    fields = (
        ("公司标题", title),
        ("页码", page),
        ("章节", shown("section_path")),
        ("table_name", shown("table_name")),
        ("statement_scope", scope),
        ("口径", scope_label),
        ("report_period", shown("report_period")),
        ("unit", shown("unit")),
        ("table_id", shown("table_id")),
        ("is_table", shown("is_table")),
        ("financial_metrics", shown("financial_metrics")),
        ("financial_facts", facts_text),
    )
    return "；".join(f"{key}={value}" for key, value in fields)


def _format_context_block(
    contexts: list[dict[str, Any]], passages: list[str], *, limit: int | None = None
) -> str:
    """按引用编号同时呈现候选 metadata 和扩展后的正文。"""
    count = min(len(contexts), len(passages))
    if limit is not None:
        count = min(count, max(limit, 0))
    blocks = []
    for index in range(count):
        blocks.append(
            f"[{index + 1}] {_format_candidate_metadata(contexts[index])}\n"
            f"内容：{passages[index]}"
        )
    return "\n\n".join(blocks)


_MAX_VERIFICATION_EVIDENCE_BLOCKS = 5
_MAX_NEIGHBOR_HOPS = 2


def _passage_record_from_chunk(
    chunk: dict[str, Any], fallback_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    if not metadata and fallback_metadata:
        metadata = fallback_metadata
    return {
        "chunk_id": str(chunk.get("chunk_id") or metadata.get("chunk_id") or ""),
        "text": str(chunk.get("text") or ""),
        "metadata": metadata,
    }


def _record_scope(record: dict[str, Any]) -> tuple[str, str]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    return (
        str(metadata.get("doc_id") or record.get("doc_id") or ""),
        str(metadata.get("kb_id") or record.get("kb_id") or ""),
    )


def _record_chunk_index(record: dict[str, Any]) -> int | None:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    value = metadata.get("chunk_index", record.get("chunk_index"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _chunk_matches_candidate_scope(
    candidate: dict[str, Any], chunk: dict[str, Any], kb_id: str
) -> bool:
    candidate_metadata = (
        candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    )
    chunk_metadata = (
        chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    )
    candidate_doc_id = str(
        candidate_metadata.get("doc_id") or candidate.get("doc_id") or ""
    )
    candidate_kb_id = str(
        candidate_metadata.get("kb_id") or candidate.get("kb_id") or kb_id or ""
    )
    chunk_doc_id = str(chunk_metadata.get("doc_id") or chunk.get("doc_id") or "")
    chunk_kb_id = str(
        chunk_metadata.get("kb_id") or chunk.get("kb_id") or kb_id or ""
    )
    return bool(
        candidate_doc_id
        and candidate_kb_id
        and chunk_doc_id == candidate_doc_id
        and chunk_kb_id == candidate_kb_id
    )


def _bounded_passage_records(
    candidate: dict[str, Any],
    records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """限制为候选自己的同 doc/kb 证据窗口，最多中心前后各两跳。"""
    if not records:
        return []

    candidate_metadata = (
        candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    )
    candidate_chunk_id = str(
        candidate.get("chunk_id") or candidate_metadata.get("chunk_id") or ""
    )
    candidate_text = str(candidate.get("text") or "")
    candidate_doc_id = str(
        candidate_metadata.get("doc_id") or candidate.get("doc_id") or ""
    )
    candidate_kb_id = str(
        candidate_metadata.get("kb_id") or candidate.get("kb_id") or ""
    )

    center_position = 0
    matched_by_chunk_id = False
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        record_metadata = (
            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        )
        record_chunk_id = str(
            record.get("chunk_id") or record_metadata.get("chunk_id") or ""
        )
        if candidate_chunk_id and record_chunk_id == candidate_chunk_id:
            center_position = position
            matched_by_chunk_id = True
            break

    if not matched_by_chunk_id:
        for position, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            if candidate_text and str(record.get("text") or "") == candidate_text:
                center_position = position
                break

    if not candidate_doc_id or not candidate_kb_id:
        center_record = records[center_position]
        center_doc_id, center_kb_id = _record_scope(center_record)
        candidate_doc_id = candidate_doc_id or center_doc_id
        candidate_kb_id = candidate_kb_id or center_kb_id

    selected: list[tuple[int, dict[str, Any]]] = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        if abs(position - center_position) > _MAX_NEIGHBOR_HOPS:
            continue
        record_doc_id, record_kb_id = _record_scope(record)
        if candidate_doc_id and record_doc_id != candidate_doc_id:
            continue
        if candidate_kb_id and record_kb_id != candidate_kb_id:
            continue
        selected.append((position, record))

    indexed = all(_record_chunk_index(record) is not None for _, record in selected)
    if indexed:
        selected.sort(key=lambda item: (_record_chunk_index(item[1]), item[0]))
    else:
        selected.sort(key=lambda item: item[0])

    bounded: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for _, record in selected:
        doc_id, kb_id = _record_scope(record)
        chunk_id = str(record.get("chunk_id") or "")
        key = (doc_id, kb_id, chunk_id, str(record.get("text") or ""))
        if key in seen:
            continue
        seen.add(key)
        bounded.append(record)
        if len(bounded) >= _MAX_VERIFICATION_EVIDENCE_BLOCKS:
            break
    return bounded


def _verification_window_metadata(
    candidate: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """为受控拼接窗口补齐一致的旧索引 metadata，不合并冲突字段。"""
    candidate_metadata = (
        candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    )
    merged = dict(candidate_metadata)
    for key in ("doc_id", "kb_id", "report_period", "unit", "statement_scope"):
        if str(merged.get(key) or "").strip():
            continue
        values: list[str] = []
        for record in records:
            metadata = (
                record.get("metadata")
                if isinstance(record.get("metadata"), dict)
                else {}
            )
            value = str(metadata.get(key) or "").strip()
            if value and value not in values:
                values.append(value)
        if len(values) == 1:
            merged[key] = values[0]
    if not merged.get("is_table"):
        if any(
            isinstance(record.get("metadata"), dict)
            and record["metadata"].get("is_table") is True
            for record in records
        ):
            merged["is_table"] = True
    return merged


def _verification_evidence_items(
    answer: str,
    contexts: list[dict[str, Any]],
    passages: list[str],
    passage_records: list[list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """取答案引用的候选及其完整相邻正文，保留各段 metadata 供核验。

    ``passages`` 是喂给模型的拼接正文；旧 Chroma 候选经常把表头、指标和
    数值拆到相邻分块，单独用引用候选的正文会丢掉可绑定的字段。节点可额外
    传入 ``passage_records`` 保存每个相邻分块的正文和 metadata；没有该值时
    仍兼容旧调用方，使用拼接后的 passage。仅正文/结构化事实能支持数字，
    空 hit 本身不会被当作证据。
    """
    cited = []
    for value in re.findall(r"\[(\d+)\]", answer or ""):
        index = int(value) - 1
        if 0 <= index < len(contexts) and index not in cited:
            cited.append(index)
    indices = cited or list(range(min(len(contexts), len(passages))))
    evidence: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def append_evidence(text: Any, metadata: Any) -> None:
        normalized_text = str(text or "")
        if not normalized_text.strip():
            return
        normalized_metadata = metadata if isinstance(metadata, dict) else {}
        key = (
            normalized_text,
            json.dumps(normalized_metadata, ensure_ascii=False, sort_keys=True, default=str),
        )
        if key in seen:
            return
        seen.add(key)
        evidence.append({"text": normalized_text, "metadata": normalized_metadata})

    for index in indices:
        hit = contexts[index]
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        if passage_records is not None:
            if index >= len(passage_records):
                continue
            bounded_records = _bounded_passage_records(hit, passage_records[index])
            for record in bounded_records:
                if not isinstance(record, dict):
                    continue
                record_metadata = (
                    record.get("metadata")
                    if isinstance(record.get("metadata"), dict)
                    else metadata
                )
                for text in _table_body_evidence(
                    {"text": record.get("text"), "metadata": record_metadata}
                ):
                    append_evidence(text, record_metadata)
            bounded_window_text = "\n\n".join(
                str(record.get("text") or "")
                for record in bounded_records
                if isinstance(record, dict)
            ).strip()
            if bounded_window_text:
                append_evidence(
                    bounded_window_text,
                    _verification_window_metadata(hit, bounded_records),
                )
            continue

        for text in _table_body_evidence(hit):
            append_evidence(text, metadata)
        if index < len(passages):
            # 保留拼接正文作为一个整体，允许跨分块表头/指标/数值绑定；
            # 相邻分块本身仍以各自 metadata 单独加入，避免丢失期间/单位。
            for text in _table_body_evidence(
                {"text": str(passages[index] or ""), "metadata": metadata}
            ):
                append_evidence(text, metadata)
    return evidence


def _single_field_candidate_supports_explicit_claim(
    sentence: str,
    question: str,
    candidate_evidence: list[Any],
) -> bool:
    """Check the explicitly labelled requested value for single-field relocation."""
    requested_metrics = _requested_metric_keys(question)
    if len(requested_metrics) != 1:
        return False
    claims = [
        claim
        for claim in _extract_numeric_claims(sentence)
        if not _is_unbound_format_number(claim, sentence)
    ]
    for target in claims:
        alias = _metric_alias_for_numeric_claim(sentence, target)
        if _canonical_metric(alias) != requested_metrics[0]:
            continue
        single_claim_answer = sentence
        for other in reversed(claims):
            if other.start == target.start and other.end == target.end:
                continue
            single_claim_answer = (
                single_claim_answer[: other.start]
                + single_claim_answer[other.end :]
            )
        if _verify_numeric_claims(
            single_claim_answer,
            candidate_evidence,
            question=question,
        )[0]:
            return True
    return False


def _relocate_numeric_citations(
    answer: str,
    question: str,
    contexts: list[dict[str, Any]],
    passages: list[str],
    passage_records: list[list[dict[str, Any]]] | None = None,
) -> str:
    """将唯一严格匹配数字的句子引用归位到同 doc/kb 的召回块。

    这是引用修正，不是数字核验放宽：当前引用必须确实无法支持该句，
    备用候选仍完整经过 `_verify_numeric_claims`，且多个候选时保持原样。
    """
    if not answer or not contexts:
        return answer
    records = passage_records if passage_records else None
    citation_groups: list[tuple[int, int]] = []
    matches = list(re.finditer(r"\[\d+\]", answer))
    position = 0
    while position < len(matches):
        first = matches[position]
        last = first
        position += 1
        while position < len(matches) and not answer[last.end() : matches[position].start()].strip():
            last = matches[position]
            position += 1
        citation_groups.append((first.start(), last.end()))

    replacements: list[tuple[int, int, str]] = []
    boundaries = "。！？\n"
    def citation_indices(group_start: int, group_end: int) -> list[int]:
        return list(
            dict.fromkeys(
                int(value) - 1
                for value in re.findall(r"\[(\d+)\]", answer[group_start:group_end])
                if 0 <= int(value) - 1 < len(contexts)
            )
        )

    for group_position, (group_start, group_end) in enumerate(citation_groups):
        cited = citation_indices(group_start, group_end)
        if not cited:
            continue
        cited_scopes = {
            _record_scope(contexts[index])
            for index in cited
            if index < len(contexts)
        }
        if len(cited_scopes) != 1:
            continue
        cited_scope = next(iter(cited_scopes))
        if not all(cited_scope):
            continue

        previous_boundary = max(
            (answer.rfind(boundary, 0, group_start) for boundary in boundaries),
            default=-1,
        )
        preceding = answer[previous_boundary + 1 : group_start]
        citation_after_sentence = not preceding.strip() or preceding.rstrip()[-1:] in boundaries
        if citation_after_sentence and previous_boundary >= 0:
            previous_boundary = max(
                (answer.rfind(boundary, 0, previous_boundary) for boundary in boundaries),
                default=-1,
            )
        sentence_start = previous_boundary + 1
        next_boundary = min(
            (answer.find(boundary, group_end) for boundary in boundaries
             if answer.find(boundary, group_end) >= 0),
            default=len(answer),
        )
        sentence_end = group_end if citation_after_sentence else next_boundary + 1
        sentence = answer[sentence_start:sentence_end]
        if not _extract_numeric_claims(_numeric_text_for_matching(sentence)):
            continue

        current_evidence = _verification_evidence_items(
            sentence, contexts, passages, records
        )
        current_supported, _ = _verify_numeric_claims(
            sentence, current_evidence, question
        )
        if current_supported:
            continue

        matching_candidates: list[int] = []
        for candidate_index, candidate in enumerate(contexts):
            if candidate_index in cited or _record_scope(candidate) != cited_scope:
                continue
            candidate_evidence = _verification_evidence_items(
                f"[{candidate_index + 1}]", contexts, passages, records
            )
            candidate_supported, _ = _verify_numeric_claims(
                sentence, candidate_evidence, question
            )
            if not candidate_supported:
                candidate_supported = _single_field_candidate_supports_explicit_claim(
                    sentence,
                    question,
                    candidate_evidence,
                )
            if candidate_supported:
                matching_candidates.append(candidate_index)
        if len(matching_candidates) != 1:
            continue
        replacement = f"[{matching_candidates[0] + 1}]"
        replacements.append((group_start, group_end, replacement))

        # 模型有时会在数字句后追加“上述数值见资料[1]”一类指代句。
        # 只同步紧邻的同一旧引用组；新数字、段落边界、不同引用或无明确
        # 指代词时立即停止，绝不做全文替换。
        follow_position = group_position + 1
        previous_end = group_end
        while follow_position < len(citation_groups):
            follow_start, follow_end = citation_groups[follow_position]
            bridge = answer[previous_end:follow_start]
            follow_cited = citation_indices(follow_start, follow_end)
            if (
                follow_cited != cited
                or re.search(r"\n\s*\n", bridge)
                or re.search(r"\[\d+\]", bridge)
                or _extract_numeric_claims(_numeric_text_for_matching(bridge))
                or not re.search(r"该数值|此数值|上述数值|该字段", bridge)
            ):
                break
            bridge_boundaries = [char for char in bridge if char in boundaries]
            if len(bridge_boundaries) > 1:
                break
            replacements.append((follow_start, follow_end, replacement))
            previous_end = follow_end
            follow_position += 1

    for start, end, replacement in reversed(replacements):
        answer = answer[:start] + replacement + answer[end:]
    return answer


def _verification_evidence_texts(
    answer: str,
    contexts: list[dict[str, Any]],
    passages: list[str],
    passage_records: list[list[dict[str, Any]]] | None = None,
) -> list[str]:
    """兼容旧测试/调用方的纯正文证据接口。"""
    return [
        item["text"]
        for item in _verification_evidence_items(
            answer, contexts, passages, passage_records
        )
    ]


def _financial_metric_keys_from_metadata(metadata: dict[str, Any]) -> set[str]:
    value = metadata.get("financial_metrics") if isinstance(metadata, dict) else None
    if isinstance(value, str):
        values = value.split("|")
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = []
    return {_canonical_metric(item) for item in values if _canonical_metric(item)}


def _period_kind(text: str) -> str:
    """识别年度、半年度和季度材料，避免同一年不同报告互相冒充。"""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    if re.search(r"半年度|上半年|下半年", normalized):
        return "half_year"
    if re.search(r"第[一二三四1234]季度|[一二三四1234]季度", normalized):
        return "quarter"
    if re.search(r"年度报告|年报|全年|年度", normalized):
        return "annual"
    return ""


def _period_matches_question(question: str, evidence_text: str) -> bool:
    """按问题要求的报告类型和年份检查候选证据。"""
    question_kind = _period_kind(question)
    evidence_kind = _period_kind(evidence_text)
    if question_kind and evidence_kind != question_kind:
        return False
    question_periods = _question_period_tokens(question)
    if not question_periods:
        return True
    evidence_periods = _question_period_tokens(evidence_text)
    if not evidence_periods:
        return False
    return any(_period_compatible(period, question) for period in evidence_periods)


def _candidate_authoritative_doc_id(
    question: str, hit: dict[str, Any]
) -> str:
    """仅从权威标题/主体元数据锁定问题主体对应的 doc_id。"""
    if not isinstance(hit, dict):
        return ""
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    subjects = _question_subject_hints(question)
    authority_text = " ".join(
        str(value or "")
        for value in (
            metadata.get("company"),
            metadata.get("doc_title"),
            hit.get("doc_title"),
        )
        if str(value or "").strip()
    )
    normalized_authority = unicodedata.normalize("NFKC", authority_text).casefold()
    if not subjects or not normalized_authority:
        return ""
    if not any(
        unicodedata.normalize("NFKC", subject).casefold() in normalized_authority
        for subject in subjects
    ):
        return ""
    return str(metadata.get("doc_id") or hit.get("doc_id") or "").strip()


def _candidate_period_matches_question(
    question: str, hit: dict[str, Any]
) -> bool:
    """优先以候选 metadata 的期间判定，避免错误期间被标题文本掩盖。"""
    if not _question_period_tokens(question):
        return False
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    report_period = str(metadata.get("report_period") or "").strip()
    if report_period:
        return _period_matches_question(question, report_period)
    title = str(metadata.get("doc_title") or hit.get("doc_title") or "").strip()
    if title:
        return _period_matches_question(question, title)
    return _period_matches_question(question, str(hit.get("text") or ""))


def _change_direction_near_metric(text: str, metric: str) -> str:
    """返回指标邻域内的增减方向；同时出现两种方向时不作猜测。"""
    normalized = _numeric_text_for_matching(text)
    windows = []
    for start, end, found_metric in _metric_mentions(normalized):
        if found_metric != metric:
            continue
        windows.append(normalized[max(0, start - 96) : min(len(normalized), end + 160)])
    directions = {
        direction
        for direction, pattern in (
            ("decrease", _CAUSE_PROTECTION_DECREASE_RE),
            ("increase", _CAUSE_PROTECTION_INCREASE_RE),
        )
        if any(pattern.search(window) for window in windows)
    }
    return next(iter(directions)) if len(directions) == 1 else ""


def _reason_question_protection_metrics(
    question: str, hit: dict[str, Any]
) -> tuple[str, ...]:
    """只返回同时具备主体、期间、指标变化和因果表达的原因题指标。"""
    requested_metrics = _requested_metric_keys(question)
    if (
        not requested_metrics
        or not _candidate_authoritative_doc_id(question, hit)
        or not _candidate_period_matches_question(question, hit)
    ):
        return ()

    text = _numeric_text_for_matching(str(hit.get("text") or ""))
    matched: list[str] = []
    for start, end, metric in _metric_mentions(text):
        if metric not in requested_metrics or metric in matched:
            continue
        window = text[max(0, start - 96) : min(len(text), end + 160)]
        if not _CAUSE_PROTECTION_CHANGE_RE.search(window):
            continue
        if not _CAUSE_PROTECTION_EVIDENCE_RE.search(window):
            continue
        question_direction = _change_direction_near_metric(question, metric)
        evidence_direction = _change_direction_near_metric(text, metric)
        if question_direction and evidence_direction != question_direction:
            continue
        matched.append(metric)
    return tuple(metric for metric in requested_metrics if metric in matched)


def _has_reliable_question_evidence(
    question: str,
    contexts: list[dict[str, Any]],
    passages: list[str],
) -> bool:
    """识别已有精确证据的低分候选，避免单一重排分抖动误转人工。"""
    metrics = _requested_metric_keys(question)
    periods = _question_period_tokens(question)
    subjects = _question_subject_hints(question)
    for index, hit in enumerate(contexts):
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        title = str(metadata.get("doc_title") or hit.get("doc_title") or "")
        text = "\n".join(
            part for part in (
                str(hit.get("text") or ""),
                str(passages[index] if index < len(passages) else ""),
            ) if part
        )
        subject_ok = not subjects or any(subject in title or subject in text for subject in subjects)
        if not subject_ok:
            continue
        evidence_period_text = " ".join(
            part
            for part in (
                str(metadata.get("report_period") or ""),
                title,
                text,
            )
            if part
        )
        if not _period_matches_question(question, evidence_period_text):
            continue
        facts = _financial_facts_from_metadata(metadata)
        if metrics and facts:
            if any(
                fact.metric in metrics
                for fact in _facts_for_question(facts, question)
            ):
                return True
        if metrics:
            mentioned = {metric for _s, _e, metric in _metric_mentions(text)}
            period_ok = not periods or _period_matches_question(question, evidence_period_text)
            if mentioned.intersection(metrics) and period_ok:
                return True
        elif text.strip():
            return True
    return False


def _deterministic_financial_match_metrics(
    question: str, hit: dict[str, Any]
) -> tuple[str, ...]:
    """返回候选中被问题明确绑定、且有精确证据的财务指标。

    这是重排后的保底判定，不参与相关性打分。主体和期间先过硬约束，
    再接受完整的 financial_facts 或带指标绑定的正文数字；原因题则要求
    正文同时包含明确的原因表达，避免用财务表格顶掉原因证据。
    """
    requested_metrics = _requested_metric_keys(question)
    if not requested_metrics or not isinstance(hit, dict):
        return ()

    if _is_reason_question(question):
        return _reason_question_protection_metrics(question, hit)

    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    title = str(metadata.get("doc_title") or hit.get("doc_title") or "")
    text = str(hit.get("text") or "")
    subjects = _question_subject_hints(question)
    if subjects and not any(subject in title or subject in text for subject in subjects):
        return ()

    question_periods = _question_period_tokens(question)
    period_text = " ".join(
        part
        for part in (
            str(metadata.get("report_period") or ""),
            title,
            text,
        )
        if part
    )
    if question_periods and not _period_matches_question(question, period_text):
        return ()

    requested = set(requested_metrics)
    matched: set[str] = set()
    facts = _facts_for_question(
        _financial_facts_from_metadata(metadata), question
    )
    for fact in facts:
        if (
            fact.metric in requested
            and fact.raw_value
            and fact.canonical_value is not None
            and fact.unit
        ):
            matched.add(fact.metric)

    for evidence_text in _table_body_evidence(hit):
        evidence_period_text = " ".join(
            part
            for part in (
                str(metadata.get("report_period") or ""),
                title,
                evidence_text,
            )
            if part
        )
        if question_periods and not _period_matches_question(
            question, evidence_period_text
        ):
            continue
        numeric_metrics = {
            claim.metric
            for claim in _extract_numeric_claims(evidence_text)
            if claim.metric in requested and claim.canonical_value is not None
        }
        matched.update(numeric_metrics)

    return tuple(metric for metric in requested_metrics if metric in matched)


def _rerank_candidate_key(hit: dict[str, Any]) -> str:
    """返回用于保护候选去重的稳定键，不改变检索器的排序契约。"""
    chunk_id = str(hit.get("chunk_id") or "")
    if chunk_id:
        return f"chunk:{chunk_id}"
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    return f"text:{metadata.get('doc_id', '')}\0{hit.get('text', '')}"


def _protect_reranked_financial_candidates(
    question: str,
    candidates: list[dict[str, Any]],
    reranked: list[dict[str, Any]],
    top_n: int,
) -> list[dict[str, Any]]:
    """保留一个原始候选中的确定性财务证据，其余位置维持重排顺序。"""
    if top_n <= 0:
        return []
    selected = list(reranked or [])[:top_n]
    protected = next(
        (
            candidate
            for candidate in candidates
            if _deterministic_financial_match_metrics(question, candidate)
        ),
        None,
    )
    if protected is None:
        return selected

    protected_key = _rerank_candidate_key(protected)
    if any(_rerank_candidate_key(item) == protected_key for item in selected):
        return selected

    remaining = [
        item
        for item in selected
        if _rerank_candidate_key(item) != protected_key
    ]
    metadata = protected.get("metadata") if isinstance(protected.get("metadata"), dict) else {}
    logger.info(
        "重排保护确定性财务候选：chunk=%s doc=%s metrics=%s",
        protected.get("chunk_id", ""),
        metadata.get("doc_id", ""),
        ",".join(_deterministic_financial_match_metrics(question, protected)),
    )
    # 把保护候选置于首位，让门槛和生成都能看到精确证据；其他候选保持
    # reranker 的相对顺序，且总数仍不超过 top_n。
    return [protected, *remaining][:top_n]


def _normalize_citations(
    answer: str, all_citations: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """把模型引用重新编号为连续的 1..N，并清除悬空编号。

    模型可能只引用候选 1、2、5。接口只返回实际使用的三条来源时，必须把
    正文同步改为 [1][2][3]，否则前端的第 5 条来源无法展开。引用按正文首次
    出现顺序排列；超出候选范围的数字标记去掉方括号，避免伪装成有效来源。
    """
    raw_indices = [int(value) for value in re.findall(r"\[(\d+)\]", answer or "")]
    ordered_valid: list[int] = []
    for index in raw_indices:
        if 1 <= index <= len(all_citations) and index not in ordered_valid:
            ordered_valid.append(index)

    mapping = {old: new for new, old in enumerate(ordered_valid, start=1)}

    def replace_marker(match: re.Match[str]) -> str:
        old = int(match.group(1))
        return f"[{mapping[old]}]" if old in mapping else match.group(1)

    normalized_answer = re.sub(r"\[(\d+)\]", replace_marker, answer or "")
    citations: list[dict[str, Any]] = []
    for new_index, old_index in enumerate(ordered_valid, start=1):
        citation = dict(all_citations[old_index - 1])
        citation["index"] = new_index
        citations.append(citation)
    return normalized_answer, citations


def _normalize_category(raw: str) -> str:
    """护栏分类标签归一化——抗 LLM 输出抖动（大小写/中文/多余词）。"""
    r = (raw or "").strip().lower()
    mapping = [
        ("kb", "kb_question"), ("知识库", "kb_question"), ("检索", "kb_question"),
        ("faq", "faq"), ("常见", "faq"),
        ("smalltalk", "smalltalk"), ("闲聊", "smalltalk"), ("寒暄", "smalltalk"),
        ("ticket", "ticket_intent"), ("工单", "ticket_intent"), ("报障", "ticket_intent"),
        ("human", "human_request"), ("转人工", "human_request"), ("人工", "human_request"),
        ("out", "out_of_scope"), ("越狱", "out_of_scope"), ("无关", "out_of_scope"),
    ]
    for key, val in mapping:
        if key in r:
            return val
    return "kb_question"  # 默认进知识库检索


class QaState(TypedDict):
    """图状态：节点间传递的共享数据（LangGraph 的 State）。"""

    question: str                    # 原始问题
    kb_id: str                       # 所属知识库（检索范围限定，P3）
    intent: str                      # 护栏分类结果（kb_question/faq/smalltalk/...）
    faq_hit: bool                    # FAQ 是否命中（命中直答，跳过文档 RAG）
    rewritten: str                   # 改写后问题（多轮上下文）
    history: list[dict[str, str]]    # 对话历史
    recall_raw: list[dict[str, Any]]  # 召回原始结果（rerank 前，客服改造第1项）
    contexts: list[dict[str, Any]]   # 检索到的分块（rerank 后）
    answer: str                      # 生成的答案
    citations: list[dict[str, Any]]  # 引用（chunk 元数据）
    retries: int                     # 已重试次数（生成质量）
    retry_requested: bool            # 当前评估是否请求回到 generate
    retrieval_attempts: int          # 检索降级重试次数（可回答性门槛）
    escalate: bool                   # 是否需转人工/建工单（证据不足两次）
    needs_clarification: bool        # 主体不明确时先澄清，不检索、不转人工
    gate_action: str                 # gate 路由决定：pass / retry / escalate
    low_quality: bool                # 评估结果：是否不达标（条件边读它）
    score: int                       # P4：忠实度评分 0-10（评估节点写入，前端展示）
    passages: list[str]              # 上下文文本（喂给 LLM）
    passage_records: list[list[dict[str, Any]]]  # 拼接上下文的逐分块正文/metadata
    metadata_filters: dict[str, Any] | None  # 向量/BM25 共用的 metadata 过滤
    verification_reasons: list[str]  # 生成后确定性核验失败原因
    answer_degraded: bool             # 重试仍失败时已替换为安全说明
    trace_id: str                     # 诊断追踪的请求令牌（默认关闭时为空串，无副作用）


def build_qa_graph(
    *,
    llm,
    embeddings: EmbeddingClient,
    vector_store: VectorStore,
    top_k: int = 5,
    hybrid: bool = True,
    checkpointer: Any = None,
    model: str = "deepseek-chat",
    reasoning_effort: str | None = None,
    min_score: float = 0.0,
    faq_store: Any = None,
    faq_threshold: float = 0.8,
    reranker: Any = None,
    persona_store: Any = None,
    recall_k: int | None = None,
    max_candidates_per_doc: int | None = 3,
    metadata_filters: dict[str, Any] | None = None,
    neighbor_expansion: int = 0,
) -> Any:
    """构建 LangGraph 问答图。llm 为 OpenAI 兼容客户端（DeepSeek）。

    hybrid=True 时用混合检索（向量 + BM25 + RRF），False 时退回纯向量。
    checkpointer（P4）：传入 LangGraph checkpointer（如 SqliteSaver）后，
      对话状态按 thread_id 持久化，支持跨请求恢复与断点续跑；None 时不持久化。
    min_score（P0）：向量相关性阈值，余弦相似度低于它视为未命中（0=不过滤）。
    reasoning_effort：可选的 OpenAI 兼容思考强度；None 时不向请求添加额外字段。
    faq_store（客服改造第5项）：FAQ 存储，faq 意图先走 FAQ 直答；None 则跳过。
    reranker（客服改造第1项）：重排器（reranker.py），None 时用 NoopReranker。
    persona_store（客服改造第6项）：客服人设/话术，generate 拼 system prompt。
    recall_k：重排前召回数量；未指定时为 top_k 的 4 倍，检索器本身不再隐式放大。
    max_candidates_per_doc：召回候选的单文档上限；None 表示不限制，默认 3。
    metadata_filters：可选 metadata 过滤，同时透传向量和 BM25。
    neighbor_expansion：没有父块时，命中块向前/向后补充的相邻块层数。
    """
    from services.kb.reranker import NoopReranker
    from services.kb.retriever import HybridRetriever, VectorOnlyRetriever

    if reranker is None:
        reranker = NoopReranker()

    final_top_k = max(int(top_k), 0)
    effective_recall_k = (
        max(int(recall_k), final_top_k)
        if recall_k is not None
        else max(final_top_k * 4, final_top_k)
    )

    def _trace_config() -> dict[str, Any]:
        """诊断追踪的运行配置快照。

        只取这一份白名单（题目要求的元数据），不遍历 os.environ——避免把任何
        凭据类环境变量写进追踪文件；键名命中敏感词的值仍会被追踪层再次脱敏。
        """
        return {
            "vector_backend": str(os.getenv("KB_VECTOR_BACKEND", "") or "").strip(),
            "qdrant_collection": str(os.getenv("KB_QDRANT_COLLECTION", "") or "").strip(),
            "top_k": final_top_k,
            "recall_k": effective_recall_k,
            "max_hits_per_doc": max_candidates_per_doc,
            "answerability_score": _answerability_threshold(),
            "structured_fin_route": str(os.getenv("KB_STRUCTURED_FIN_ROUTE", "") or "").strip(),
            "fin_route_top_n": str(os.getenv("KB_FIN_ROUTE_TOP_N", "") or "").strip(),
            "rerank_mode": str(os.getenv("KB_RERANK_MODE", "") or "").strip(),
            "embedding_mode": str(os.getenv("KB_EMBEDDING_MODE", "") or "").strip(),
            "embedding_model": str(os.getenv("KB_EMBEDDING_MODEL", "") or "").strip(),
        }

    if hybrid:
        retriever: Any = HybridRetriever(
            vector_store,
            embeddings=embeddings,
            top_k=effective_recall_k,
            min_score=min_score,
            max_candidates_per_doc=max_candidates_per_doc,
        )
    else:
        retriever = VectorOnlyRetriever(
            vector_store,
            embeddings=embeddings,
            top_k=effective_recall_k,
            min_score=min_score,
            max_candidates_per_doc=max_candidates_per_doc,
        )

    def node_guardrail(state: QaState) -> dict[str, Any]:
        """护栏前置：入口意图分类，挡越狱/套提示词/闲聊/明确转人工。

        分类：kb_question / faq / smalltalk / ticket_intent / human_request / out_of_scope。
        非知识库类意图不进检索（省钱 + 防注入），由 direct_reply 走对应话术。
        """
        question = state["question"]
        clarification = _company_clarification(question, state.get("history", []))
        if clarification:
            logger.info("主体不明确，跳过 FAQ/改写/检索并请求澄清：%s", question[:80])
            return {
                "intent": "clarification",
                "needs_clarification": True,
                "answer": clarification,
            }
        if _is_obvious_kb_question(question):
            logger.info("高置信度企业材料问句，直接进入知识库：%s", question[:80])
            return {"intent": "kb_question"}
        prompt = (
            "你是客服意图分类器。把用户消息归到以下类别之一：\n"
            "kb_question（可被企业知识库回答的问题）\n"
            "faq（常见问题咨询）\n"
            "smalltalk（闲聊/寒暄/打招呼）\n"
            "ticket_intent（报障/投诉/要建工单）\n"
            "human_request（明确要求转人工客服）\n"
            "out_of_scope（恶意/套话/越狱/与业务无关）\n"
            f"用户消息：{question}\n"
            "只输出类别名，不要解释。"
        )
        try:
            raw = _llm_invoke(
                llm,
                [{"role": "user", "content": prompt}],
                model=model,
                reasoning_effort=reasoning_effort,
            )
            category = _normalize_category(raw)
        except Exception as exc:
            logger.warning("护栏分类失败，默认进知识库: %s", exc)
            category = "kb_question"
        logger.info("护栏分类：%s → %s", question[:40], category)
        return {"intent": category}

    def node_direct_reply(state: QaState) -> dict[str, Any]:
        """非检索类意图的轻量回复（闲聊/转人工/拒绝），不进检索、不调生成。"""
        intent = state.get("intent", "kb_question")
        kb_id = state.get("kb_id", "default")
        if state.get("needs_clarification"):
            return {
                "answer": state.get("answer") or "请提供公司名称或股票代码。",
                "citations": [],
                "escalate": False,
            }
        persona = persona_store.get_persona(kb_id) if persona_store else None
        if intent == "smalltalk":
            company = (persona or {}).get("company_name", "本公司")
            return {
                "answer": f"您好！我是{company}的智能客服，可以为您解答产品、流程等业务问题，请问有什么可以帮您？",
                "citations": [],
            }
        if intent == "human_request":
            transfer = (persona or {}).get("transfer_message", "已为您转接人工客服，请稍候。")
            return {
                "answer": transfer,
                "citations": [],
                "escalate": True,
            }
        if intent == "out_of_scope":
            refuse = (persona or {}).get("refuse_message", "抱歉，我只能回答与本公司业务相关的问题。")
            return {
                "answer": refuse,
                "citations": [],
            }
        # ticket_intent / faq：走后续链路兜底
        return {}

    def node_faq_lookup(state: QaState) -> dict[str, Any]:
        """FAQ 匹配（客服改造第5项）：faq 意图先走 FAQ 向量直答，命中跳过文档 RAG。"""
        query = state.get("rewritten") or state["question"]
        kb_id = state.get("kb_id", "default")
        if faq_store is None:
            return {"faq_hit": False}
        hit = faq_store.search(query, kb_id, threshold=faq_threshold)
        if not hit:
            return {"faq_hit": False}
        logger.info("FAQ 命中（score=%.3f）：%s", hit["score"], hit["question"][:40])
        return {
            "faq_hit": True,
            "answer": hit["answer"],
            "citations": [
                {
                    "index": 1,
                    "text": hit["question"],
                    "doc_title": "FAQ",
                    "source_type": "faq",
                    "metadata": {"source_type": "faq", "doc_title": "FAQ"},
                }
            ],
        }

    def route_after_faq(state: QaState) -> str:
        """FAQ 条件边：命中 → END（直答）；未命中 → rewrite（文档 RAG 兜底）。"""
        return END if state.get("faq_hit") else "rewrite"

    def route_after_guardrail(state: QaState) -> str:
        """护栏条件边：闲聊/转人工/越狱 → direct_reply；faq → faq_lookup；其余 → rewrite。"""
        intent = state.get("intent", "kb_question")
        if intent in ("clarification", "smalltalk", "human_request", "out_of_scope"):
            return "direct_reply"
        if intent == "faq":
            return "faq_lookup"
        return "rewrite"

    def node_rewrite(state: QaState) -> dict[str, Any]:
        """查询改写：结合对话历史，把当前问题改写成自包含的检索查询。

        场景：用户问"它支持 PDF 吗？"（"它"指代上文的文档系统）
        → 改写为"知识库系统是否支持 PDF 文档导入"
        """
        question = state["question"]
        history = state.get("history", [])
        if not history:
            return {"rewritten": question}  # 无历史不改写

        history_text = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:100]}"
            for m in history[-4:]  # 只看最近 4 轮
        )
        prompt = (
            "基于对话历史，把当前问题改写成独立可检索的查询。\n"
            f"对话历史：\n{history_text}\n"
            f"当前问题：{question}\n"
            "只输出改写后的查询，不要解释。"
        )
        # P1：改写 LLM 调用无降级会 500，这里 try/except 回退原问题（同 guardrail 降级）
        try:
            rewritten = _llm_invoke(
                llm,
                [{"role": "user", "content": prompt}],
                model=model,
                reasoning_effort=reasoning_effort,
            )
        except Exception as exc:
            logger.warning("查询改写失败，回退原问题: %s", exc)
            return {"rewritten": question}
        # 🟡15：改写结果校验——LLM 返回寒暄/解释类垃圾文本时回退原问题，
        # 避免垃圾文本被当检索词用（召回质量崩塌）。
        # 合法改写应为单行短查询：含换行（解释/多段）或超长（跑题）都判为无效。
        rewritten = rewritten.strip()
        if not rewritten or "\n" in rewritten or len(rewritten) > 120:
            logger.warning("查询改写结果异常（%.0f 字符），回退原问题", len(rewritten))
            return {"rewritten": question}
        return {"rewritten": rewritten}

    def _trace_bind(state: QaState) -> str:
        """把注册表里的追踪记录挂到当前节点上下文。

        LangGraph 每个节点都在独立的上下文副本里跑，ContextVar 跨节点必然失效，
        所以每个埋点节点入口都要用图状态里的 trace_id 重新 bind 一次。
        """
        if not diagnostic_trace.is_enabled():
            return ""
        token = str(state.get("trace_id") or "").strip()
        if not token:
            return ""
        diagnostic_trace.bind(token)
        return token

    def node_retrieve(state: QaState) -> dict[str, Any]:
        """原问题必检索，安全改写只作为补充；两者结果融合后进入 recall_raw。"""
        # 门槛降级重试时保留原问题作为唯一查询，避免改写结果继续放大偏差。
        rewritten = ""
        if state.get("retrieval_attempts", 0) == 0:
            rewritten = (state.get("rewritten") or "").strip()
        queries = _retrieval_queries(
            state["question"],
            rewritten,
            retrieval_attempt=state.get("retrieval_attempts", 0),
        )
        request_filters = state.get("metadata_filters")
        if metadata_filters and request_filters:
            filters = {"$and": [metadata_filters, request_filters]}
        else:
            filters = request_filters or metadata_filters
        # 诊断追踪：必须在 search_queries 之前开启，否则检索器内部的
        # raw_recall / structured_fin_route / post_merge 阶段会挂不上本次请求。
        if diagnostic_trace.is_enabled():
            diagnostic_trace.ensure_started(
                {
                    "question": str(state["question"] or "")[
                        : diagnostic_trace.MAX_QUESTION_TEXT
                    ],
                    "kb_id": state.get("kb_id", "default"),
                    "retrieval_attempt": state.get("retrieval_attempts", 0),
                    "queries": [str(item) for item in queries],
                    "config": _trace_config(),
                },
                token=str(state.get("trace_id") or ""),
            )
        hits = retriever.search_queries(
            queries,
            top_k=effective_recall_k,
            kb_id=state.get("kb_id", "default"),
            metadata_filters=filters,
        )
        if diagnostic_trace.is_enabled():
            diagnostic_trace.add_stage(
                "node_retrieve",
                {
                    "recall_raw_count": len(hits),
                    "candidates": diagnostic_trace.candidate_snapshot(hits, "recall_raw"),
                },
            )
        return {"recall_raw": hits}

    def node_rerank(state: QaState) -> dict[str, Any]:
        """Rerank：召回 top-N → 精排 → top-K 进生成。

        用注入的 reranker 对象（reranker.py：llm / crossencoder / off 三模式）。
        召回阶段多召回（top_k*2），rerank 精排筛掉弱相关，只把 top_k 喂给生成。
        """
        _trace_bind(state)
        query = "\n补充检索表达：".join(
            _retrieval_queries(
                state["question"],
                (state.get("rewritten") or "").strip()
                if state.get("retrieval_attempts", 0) == 0
                else "",
                retrieval_attempt=state.get("retrieval_attempts", 0),
            )
        )
        if _prefers_consolidated_scope(state["question"]):
            query = f"{query}\n口径要求：问题未明确写母公司，优先合并财务报表口径"
        candidates = state.get("recall_raw", [])
        if not candidates:
            if diagnostic_trace.is_enabled():
                diagnostic_trace.add_stage(
                    "node_rerank", {"outcome": "empty_candidates"}
                )
            return {"contexts": [], "passages": [], "passage_records": []}
        reranked = reranker.rerank(query, candidates, final_top_k)
        if diagnostic_trace.is_enabled():
            kept_ids = {id(hit) for hit in reranked}
            diagnostic_trace.add_stage(
                "rerank_after_model",
                {
                    "final_top_k": final_top_k,
                    "rerank_query": diagnostic_trace.clip_text(query, 300),
                    "before": diagnostic_trace.ranking_ids(candidates),
                    "after": diagnostic_trace.ranking_ids(reranked),
                    "dropped_by_rerank": diagnostic_trace.ranking_ids(
                        [hit for hit in candidates if id(hit) not in kept_ids]
                    ),
                    "candidates": diagnostic_trace.candidate_snapshot(reranked, "reranked"),
                },
            )
        reranked = _protect_reranked_financial_candidates(
            state["question"], candidates, reranked, final_top_k
        )
        if diagnostic_trace.is_enabled():
            diagnostic_trace.add_stage(
                "post_protect",
                {
                    "final_order": diagnostic_trace.ranking_ids(reranked),
                    "candidates": diagnostic_trace.candidate_snapshot(
                        reranked, "final_context"
                    ),
                },
            )
        kb_id = state.get("kb_id", "default")
        passages: list[str] = []
        passage_records: list[list[dict[str, Any]]] = []
        for candidate in reranked:
            metadata = candidate.get("metadata") or {}
            parent_id = str(metadata.get("parent_id") or "")
            if parent_id:
                parent = vector_store.get_chunk_by_id(parent_id, kb_id=kb_id)
                if (
                    parent
                    and _chunk_matches_candidate_scope(candidate, parent, kb_id)
                    and (parent.get("text") or "").strip()
                ):
                    parent_text = str(parent["text"])
                    candidate_text = str(candidate.get("text") or "")
                    parent_metadata = (
                        parent.get("metadata")
                        if isinstance(parent.get("metadata"), dict)
                        else {}
                    )
                    parent_record = _passage_record_from_chunk(parent, parent_metadata)
                    candidate_record = _passage_record_from_chunk(candidate, metadata)
                    if candidate_text and candidate_text not in parent_text:
                        passages.append(
                            f"{parent_text}\n\n[匹配分块]\n{candidate_text}"
                        )
                        passage_records.append(
                            _bounded_passage_records(
                                candidate, [parent_record, candidate_record]
                            )
                        )
                    else:
                        passages.append(parent_text)
                        passage_records.append(
                            _bounded_passage_records(candidate, [parent_record])
                        )
                    continue

            before: list[dict[str, Any]] = []
            after: list[dict[str, Any]] = []
            previous_id = str(metadata.get("previous_chunk_id") or "")
            next_id = str(metadata.get("next_chunk_id") or "")
            neighbor_hops = min(max(int(neighbor_expansion), 0), _MAX_NEIGHBOR_HOPS)
            for _ in range(neighbor_hops):
                if previous_id:
                    previous = vector_store.get_chunk_by_id(previous_id, kb_id=kb_id)
                    if previous and _chunk_matches_candidate_scope(
                        candidate, previous, kb_id
                    ):
                        before.insert(
                            0,
                            _passage_record_from_chunk(previous),
                        )
                        previous_id = str(
                            (previous.get("metadata") or {}).get("previous_chunk_id") or ""
                        )
                    else:
                        previous_id = ""
                if next_id:
                    following = vector_store.get_chunk_by_id(next_id, kb_id=kb_id)
                    if following and _chunk_matches_candidate_scope(
                        candidate, following, kb_id
                    ):
                        after.append(
                            _passage_record_from_chunk(following),
                        )
                        next_id = str(
                            (following.get("metadata") or {}).get("next_chunk_id") or ""
                        )
                    else:
                        next_id = ""
            candidate_record = {
                "text": str(candidate.get("text") or ""),
                "metadata": metadata,
            }
            window_records = _bounded_passage_records(
                candidate, [*before, candidate_record, *after]
            )
            passage_records.append(window_records)
            passages.append(
                "\n\n".join(
                    record["text"] for record in window_records
                )
            )
        return {
            "contexts": reranked,
            "passages": passages,
            "passage_records": passage_records,
        }

    def node_gate(state: QaState) -> dict[str, Any]:
        """可回答性门槛：rerank 后 top-1 证据分不足时不硬答，走降级链。

        降级链：证据不足 → 换原问题重检索一次 → 仍不足 → 标记 escalate 转人工。
        阈值 KB_ANSWERABILITY_SCORE（默认 0=不启用门槛）。
        """
        contexts = state.get("contexts", [])
        threshold = _answerability_threshold()
        # P1：门槛关闭（阈值 <= 0）时直接 pass，空检索也走 generate 的"未检索到"话术
        if threshold <= 0:
            return {"gate_action": "pass", "escalate": False}
        top_score = _evidence_score(contexts[0]) if contexts else 0.0
        if contexts and (
            top_score >= threshold
            or _has_reliable_question_evidence(
                state.get("question", ""), contexts, state.get("passages", [])
            )
        ):
            return {"gate_action": "pass", "escalate": False}
        attempts = state.get("retrieval_attempts", 0)
        if attempts < 1:
            # 第一次降级：换原问题重检索（不清空 rewritten，由 retrieve 按 attempts 判断）
            logger.info("检索证据不足（top_score=%.3f），换原问题重检索一次", top_score)
            return {
                "gate_action": "retry",
                "retrieval_attempts": attempts + 1,
                "escalate": False,
            }
        # 第二次仍不足：如实告知 + 转人工
        logger.warning("检索证据不足（两次），触发转人工/建工单")
        return {
            "gate_action": "escalate",
            "retrieval_attempts": attempts + 1,
            "escalate": True,
        }

    def route_after_gate(state: QaState) -> str:
        """gate 条件边：pass/escalate → generate；retry → retrieve。"""
        return "retrieve" if state.get("gate_action") == "retry" else "generate"

    def node_generate(state: QaState) -> dict[str, Any]:
        """生成：把检索到的分块作为上下文，LLM 生成带引用的答案。"""
        _trace_bind(state)
        passages = state.get("passages", [])
        kb_id = state.get("kb_id", "default")
        persona = persona_store.get_persona(kb_id) if persona_store else None
        # 可回答性门槛两次不过：如实告知 + 转人工（客服防胡说的命门）
        if state.get("escalate"):
            transfer = (persona or {}).get("transfer_message", "已为您转接人工客服，请稍候。")
            return {
                "answer": f"抱歉，知识库中未检索到能准确回答您问题的相关信息，{transfer}",
                "citations": [],
                "passages": [],  # P2：清空 passages，避免 evaluate 误判低质量触发无效 generate 重试
                "verification_reasons": [],
            }
        if not passages:
            refuse = (persona or {}).get("refuse_message", "知识库中未检索到相关信息，请尝试换个问法。")
            return {"answer": refuse, "citations": [], "verification_reasons": []}

        # 构造上下文：编号分块，让 LLM 用 [1][2] 标注引用
        context_block = _format_context_block(state.get("contexts", []), passages)
        correction_block = ""
        if state.get("retries", 0) > 0:
            verification_reasons = state.get("verification_reasons") or []
            if verification_reasons:
                correction_block = (
                    "这是一次定向纠错重试。上一版回答未通过确定性核验："
                    + "；".join(verification_reasons)
                    + "。请只依据下列资料修正，不要补造资料、改写数字单位或省略口径标签。\n"
                )
            else:
                correction_block = (
                    "这是一次定向纠错重试。上一版回答未通过完整性/相关性质量评估，"
                    "请重新核对问题并补齐资料明确支持的要点；原因题尤其不要只写变化结果或单一表层原因，"
                    "应覆盖证据明确写出的原因并逐项引用。仍然只依据下列资料作答，不要补造资料。\n"
                )
        prompt = (
            "仅基于以下资料回答用户问题。\n"
            "要求：\n"
            "1. 若资料不足，明确说明缺少相关信息\n"
            "2. 回答末尾用 [编号] 标注引用来源（如 [1][2]）\n"
            "3. 问题询问某一家公司的具体事实，但当前问题和用户历史都无法确认公司时，先澄清公司名称或股票代码，禁止依据检索排名猜测公司；问题已包含公司全称、常用简称、ST简称或股票代码时视为主体明确，不得再次索要全称或代码\n"
            "4. 问题询问行业通用内容、常见结构或共性时，可以综合多家公司资料归纳，并明确这是共性总结；只有询问单一公司事实时才禁止混合不同公司\n"
            "5. 财务资料同时出现合并口径与母公司口径时，问题未明确写“母公司”的，优先回答合并口径；无法确认时同时列出并标明口径，禁止用母公司数值替代公司整体数值\n"
            "6. 默认用 3-5 个要点简洁回答，总长度不超过 180 个汉字；用户明确要求详细说明时除外\n"
            "7. 数字、百分比、年份和单位必须逐字依据资料；不同单位只有在精确换算后才可转换，不能四舍五入或擅自改单位\n"
            "8. 若候选 metadata 含 financial_facts，按相同 metric、report_period、statement_scope 绑定数值；简单财务事实必须原样复制 raw_value 和 unit，不能从同一证据的其他指标取数字\n"
            "9. 数字按字段逐项核对：有精确证据的字段照常回答并引用；缺证据或无法把字段与数字绑定的字段只写“该字段无法确定”，不得为它补数字，也不得因此否定同一回答中其他已被证据支持的字段；营业收入/营业总收入、研发投入合计/研发费用等近似概念不得互换\n"
            "10. 若问题询问原因、为何或变动原因，先回答变化结论，再覆盖证据中明确写出的主要原因，并为每个原因逐项引用；不要编造证据未明确写出的原因，也不要只写变化结果或单一表层原因\n"
            f"{correction_block}"
            f"资料（每条含候选 metadata，口径字段缺失时按未提供处理）：\n{context_block}\n\n"
            f"问题：{state['question']}\n"
            "回答："
        )
        messages: list[dict[str, str]] = []
        if persona_store:
            messages.append({"role": "system", "content": persona_store.build_system_prompt(kb_id)})
        messages.append({"role": "user", "content": prompt})
        # P1-2：真 token 级流式——用 get_stream_writer 把每个 token 透传给外层 run_qa_stream。
        # 非流式（invoke）时 get_stream_writer 返回 no-op writer，_llm_stream 仍累积完整答案。
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
        chunks: list[str] = []
        for token in _llm_stream(
            llm,
            messages,
            model=model,
            reasoning_effort=reasoning_effort,
        ):
            chunks.append(token)
            writer({"type": "token", "text": token})
        answer = "".join(chunks)

        if diagnostic_trace.is_enabled():
            diagnostic_trace.add_stage(
                "initial_answer",
                {
                    "retries": state.get("retries", 0),
                    "answer": diagnostic_trace.clip_text(
                        answer, diagnostic_trace.MAX_ANSWER_TEXT
                    ),
                    "prompt_context_block": diagnostic_trace.clip_text(
                        context_block, 1200
                    ),
                },
            )
        # 模型对报告类型不匹配的近似材料诚实拒答时，直接转人工；否则
        # evaluate 会把“无答案”当成低质量答案重试，且可能错误地保留引用。
        if _is_refusal_answer(answer) and not _has_reliable_question_evidence(
            state["question"], state.get("contexts", []), passages
        ):
            if diagnostic_trace.is_enabled():
                diagnostic_trace.add_stage(
                    "refusal_escalation",
                    {
                        "reason": "refusal_answer_without_reliable_evidence",
                        "answer": diagnostic_trace.clip_text(
                            answer, diagnostic_trace.MAX_ANSWER_TEXT
                        ),
                    },
                )
            transfer = (persona or {}).get("transfer_message", "已为您转接人工客服，请稍候。")
            return {
                "answer": f"抱歉，知识库中未检索到能准确回答您问题的相关信息，{transfer}",
                "citations": [],
                "passages": [],
                "verification_reasons": [],
                "escalate": True,
            }

        # 引用元数据：与答案里的 [n] 对应。
        # P2：平铺定位字段（doc_title/chunk_index/page），前端可直接跳转原文，不用钻 metadata。
        all_citations = []
        for i, h in enumerate(state.get("contexts", [])):
            meta = h.get("metadata", {}) or {}
            all_citations.append(
                {
                    "index": i + 1,
                    "chunk_id": h.get("chunk_id", ""),
                    "text": h.get("text", "")[:120],
                    "doc_id": meta.get("doc_id", ""),
                    "doc_title": meta.get("doc_title", ""),
                    "chunk_index": meta.get("chunk_index"),
                    "page": meta.get("page") or meta.get("page_start") or None,
                    "page_start": meta.get("page_start") or None,
                    "page_end": meta.get("page_end") or None,
                    "section_path": meta.get("section_path", ""),
                    "source_type": meta.get("source_type", ""),
                    "metadata": meta,
                }
            )
        answer = _relocate_numeric_citations(
            answer,
            state["question"],
            state.get("contexts", []),
            passages,
            state.get("passage_records", []),
        )
        if diagnostic_trace.is_enabled():
            diagnostic_trace.add_stage(
                "citation_relocation",
                {
                    "answer_after": diagnostic_trace.clip_text(
                        answer, diagnostic_trace.MAX_ANSWER_TEXT
                    ),
                    "citation_candidates": [
                        {
                            "index": item.get("index"),
                            "chunk_id": item.get("chunk_id", ""),
                            "doc_id": (item.get("metadata") or {}).get("doc_id", ""),
                            "page": item.get("page"),
                            "section_path": diagnostic_trace.clip_text(
                                item.get("section_path"), 120
                            ),
                        }
                        for item in all_citations[:12]
                    ],
                },
            )
        verification_evidence = _verification_evidence_items(
            answer,
            state.get("contexts", []),
            passages,
            state.get("passage_records", []),
        )
        verification_reasons = _answer_verification_reasons(
            state["question"], answer, verification_evidence
        )
        if diagnostic_trace.is_enabled():
            diagnostic_trace.add_stage(
                "generation_verification",
                {
                    "verification_reasons": [
                        diagnostic_trace.clip_text(item, diagnostic_trace.MAX_REASON_TEXT)
                        for item in verification_reasons
                    ],
                    "evidence_item_count": len(verification_evidence),
                    "evidence_items": [
                        diagnostic_trace.clip_text(item, 200)
                        for item in verification_evidence[:12]
                    ],
                    "answer_before_relocation": diagnostic_trace.clip_text(
                        answer, diagnostic_trace.MAX_ANSWER_TEXT
                    ),
                },
            )
        answer, citations = _normalize_citations(answer, all_citations)
        if diagnostic_trace.is_enabled():
            diagnostic_trace.add_stage(
                "final_answer",
                {
                    "answer": diagnostic_trace.clip_text(
                        answer, diagnostic_trace.MAX_ANSWER_TEXT
                    ),
                    "citation_count": len(citations),
                    "citations": [
                        {
                            "index": item.get("index"),
                            "chunk_id": item.get("chunk_id", ""),
                            "doc_id": (item.get("metadata") or {}).get("doc_id", ""),
                            "page": item.get("page"),
                        }
                        for item in citations[:12]
                    ],
                    "verification_reasons": [
                        diagnostic_trace.clip_text(item, diagnostic_trace.MAX_REASON_TEXT)
                        for item in verification_reasons
                    ],
                },
            )
        return {
            "answer": answer,
            "citations": citations,
            "verification_reasons": verification_reasons,
        }

    def _evaluate_faithfulness(state: QaState) -> tuple[int, bool]:
        """忠实度评估（LLM 打分版，P2 升级；P4 返回分数供展示）。

        让 LLM 判断"答案是否基于给定资料"（0-10 分，<6 视为不达标）。
        返回 (score, passed)：score 给前端展示，passed 给条件边判定。
        LLM 评估比简单规则更准：能识别"答案没引用资料却胡编"的情况。
        评估失败（LLM 抖动/解析失败）→ 退回简单规则检查（保守收敛）。
        """
        answer = (state.get("answer") or "").strip()
        passages = state.get("passages", [])
        if not answer:
            return 0, False  # 空答案必不达标
        if not passages:
            return 10, True  # 无资料时诚实声明即可，不重试

        # 与重排阶段保持同等可见范围。关键事实常在年报分块后半段；只看前 300 字
        # 会出现“生成看到了证据、评估却没看到”的假低分。
        context_block = _format_context_block(
            state.get("contexts", []),
            [p[:900] for p in passages],
            limit=4,
        )
        user_history = "\n".join(
            str(message.get("content", ""))[:200]
            for message in state.get("history", [])[-8:]
            if message.get("role") == "user"
        )
        history_block = f"【用户历史】\n{user_history}\n\n" if user_history else ""
        prompt = (
            "你是 RAG 质量评估员。从两个维度评估下面的【答案】：\n"
            "1. 忠实度（是否严格基于资料，无编造）：10=完全基于，0=无关/编造\n"
            "2. 相关性（是否回答了用户问题）：10=切题，0=答非所问\n\n"
            f"{history_block}【用户问题】{state.get('question', '')}\n"
            f"【资料】\n{context_block}\n\n"
            f"【答案】\n{answer}\n\n"
            "只输出两个 0-10 的整数，格式：忠实度 相关性（空格分隔），不要解释。"
        )
        try:
            raw = _llm_invoke(
                llm,
                [{"role": "user", "content": prompt}],
                model=model,
                reasoning_effort=reasoning_effort,
            )
            import re as _re

            nums = _re.findall(r"\d+", raw)
            faith = int(nums[0]) if nums else 0
            relev = int(nums[1]) if len(nums) > 1 else faith
            score = min(faith, relev)  # 取低分（任一维度差都算不达标）
            logger.info("RAG 评估：忠实度 %d/10，相关性 %d/10，取低 %d/10", faith, relev, score)
            return score, score >= 6
        except Exception:
            logger.warning("忠实度评估失败，退回规则检查")
            passed = bool(answer) and ("未检索到" not in answer)
            return (10 if passed else 0), passed

    def node_evaluate(state: QaState) -> dict[str, Any]:
        """评估：LLM 忠实度打分（P2 升级版），失败退回规则检查。

        结果写入 state.low_quality，供条件边（route_after_evaluate）读取——
        保证"评估判定"和"路由决策"用同一份结论，不会出现一边判重试一边放行。
        """
        _trace_bind(state)
        answer = state.get("answer", "").strip()
        has_passages = bool(state.get("passages"))
        retries = state.get("retries", 0)

        # 简单规则前置检查（零成本快速拦截明显问题）
        low_quality = not answer or ("未检索到" in answer and has_passages)
        score = 0
        verification_reasons = state.get("verification_reasons", [])
        if verification_reasons:
            # 确定性证据核验优先于 LLM 评分；不增加一次独立调用，也不能被高分捷径绕过。
            low_quality = True
            score = 0
            logger.info("生成后证据核验未通过：%s", "；".join(verification_reasons))
        elif not low_quality:
            # P2-3（可选降本）：KB_SKIP_EVAL_ON_HIGH_SCORE=1 时，top-1 证据分很高则跳过
            # LLM 评估直接判通过，省一次调用。默认关闭——忠实度评估是"防胡说"的防线，
            # 不该因检索分高就跳过（检索强 ≠ 生成忠实）。
            if _skip_eval_on_high_score:
                contexts = state.get("contexts", [])
                top_score = _evidence_score(contexts[0]) if contexts else 0.0
                if top_score >= 0.8:
                    score = 10
                    low_quality = False
                    return {
                        "retries": retries,
                        "retry_requested": False,
                        "low_quality": low_quality,
                        "score": score,
                    }
            # 通过规则检查后，再用 LLM 深度评估忠实度
            score, passed = _evaluate_faithfulness(state)
            low_quality = not passed

        if low_quality and retries < MAX_RETRY:
            return {
                "retries": retries + 1,
                "retry_requested": True,
                "low_quality": low_quality,
                "score": score,
            }
        if low_quality and retries >= MAX_RETRY:
            partial_answer = None
            if verification_reasons:
                evidence = _verification_evidence_items(
                    answer,
                    state.get("contexts", []),
                    state.get("passages", []),
                    state.get("passage_records", []),
                )
                partial_answer = _partial_numeric_fallback_answer(
                    state.get("question", ""), answer, evidence
                )
            if partial_answer is not None:
                safe_answer = partial_answer
                verification_reasons = []
                logger.warning("生成核验重试仍失败，保留已核验字段并遮蔽不支持数字")
            elif verification_reasons:
                safe_answer = "已找到相关资料，但字段与数字无法可靠对应，暂时不能给出确定数值。"
                logger.warning("生成核验重试仍失败，返回安全说明而非原答案")
            else:
                safe_answer = "已找到相关资料，但当前答案未通过可靠性核验，暂时不能给出确定回答。"
                logger.warning("生成质量重试仍失败，返回安全说明而非原答案")
            if diagnostic_trace.is_enabled():
                diagnostic_trace.add_stage(
                    "safe_fallback",
                    {
                        "branch": "partial_answer" if partial_answer is not None else (
                            "field_binding_safe" if verification_reasons else "quality_safe"
                        ),
                        "retries": retries,
                        "verification_reasons": [
                            diagnostic_trace.clip_text(
                                item, diagnostic_trace.MAX_REASON_TEXT
                            )
                            for item in verification_reasons
                        ],
                        "answer": diagnostic_trace.clip_text(
                            safe_answer, diagnostic_trace.MAX_ANSWER_TEXT
                        ),
                    },
                )
            return {
                "answer": safe_answer,
                "retries": retries,
                "retry_requested": False,
                "low_quality": low_quality,
                "score": score,
                "verification_reasons": verification_reasons,
                "answer_degraded": True,
            }
        return {
            "retries": retries,
            "retry_requested": False,
            "low_quality": low_quality,
            "score": score,
        }

    def route_after_evaluate(state: QaState) -> str:
        """条件边：不达标且有重试额度 → 回 generate；否则结束。"""
        if state.get("low_quality") and state.get("retry_requested", False):
            return "generate"  # 回到生成节点重试
        return END

    # ---- 组装图 ----
    builder = StateGraph(QaState)
    builder.add_node("guardrail", node_guardrail)
    builder.add_node("direct_reply", node_direct_reply)
    builder.add_node("faq_lookup", node_faq_lookup)
    builder.add_node("rewrite", node_rewrite)
    builder.add_node("retrieve", node_retrieve)
    builder.add_node("rerank", node_rerank)
    builder.add_node("gate", node_gate)
    builder.add_node("generate", node_generate)
    builder.add_node("evaluate", node_evaluate)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {"direct_reply": "direct_reply", "faq_lookup": "faq_lookup", "rewrite": "rewrite"},
    )
    builder.add_edge("direct_reply", END)  # 闲聊/转人工/拒绝直接结束，不评估
    builder.add_conditional_edges(
        "faq_lookup",
        route_after_faq,
        {END: END, "rewrite": "rewrite"},
    )
    builder.add_edge("rewrite", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "gate")
    builder.add_conditional_edges(
        "gate",
        route_after_gate,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {"generate": "generate", END: END},
    )

    return builder.compile(checkpointer=checkpointer)


def run_qa(
    graph: Any,
    *,
    question: str,
    history: list[dict[str, str]] | None = None,
    kb_id: str = "default",
    metadata_filters: dict[str, Any] | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """执行问答图，返回 {answer, citations, contexts, score}。

    kb_id：限定检索的知识库（多库隔离，P3）。
    metadata_filters：可选 metadata 过滤，同时应用于向量和 BM25 检索。
    thread_id（P4）：对话线程 ID。传同一 ID 时 LangGraph checkpointer 持久化对话
      状态（SQLite），支持跨请求恢复与断点续跑；不传则每次自动开新线程。
    """
    initial: QaState = {
        "question": question,
        "kb_id": kb_id,
        "intent": "",
        "faq_hit": False,
        "rewritten": "",
        "history": history or [],
        "recall_raw": [],
        "contexts": [],
        "passages": [],
        "passage_records": [],
        "answer": "",
        "citations": [],
        "retries": 0,
        "retry_requested": False,
        "retrieval_attempts": 0,
        "escalate": False,
        "needs_clarification": False,
        "gate_action": "",
        "low_quality": False,
        "score": 0,
        "metadata_filters": metadata_filters,
        "verification_reasons": [],
        "answer_degraded": False,
        # 诊断追踪令牌：随图状态流转，节点靠它从注册表取回同一条记录
        # （关闭时只是个空串，不参与任何逻辑）。
        "trace_id": uuid.uuid4().hex[:12] if diagnostic_trace.is_enabled() else "",
    }
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    # 诊断追踪：图执行无论成功还是抛异常都必须收尾，否则注册表里会留下永不
    # 落盘的残留记录（越积越多直到触发容量清理）。异常分支写完错误摘要后
    # 必须重新抛出原异常——追踪是旁路，不吞异常、不改接口行为。
    _trace_token = str(initial.get("trace_id") or "")
    _traced = diagnostic_trace.is_enabled()
    try:
        result = graph.invoke(initial, config=config)
    except BaseException as exc:  # noqa: BLE001 —— 收尾后原样抛出
        if _traced:
            diagnostic_trace.finish_request(
                diagnostic_trace.error_outcome(exc), token=_trace_token
            )
        raise
    # 诊断追踪收尾：放在这里而不是图节点里，保证 guardrail 直答、gate 转人工等
    # 任何提前结束的分支也一定会落盘，不会把记录漏到下一个请求。
    if _traced:
        diagnostic_trace.finish_request(
            {
                "status": "ok",
                "answer": diagnostic_trace.clip_text(
                    result.get("answer", ""), diagnostic_trace.MAX_ANSWER_TEXT
                ),
                "escalate": result.get("escalate", False),
                "retries": result.get("retries", 0),
                "score": result.get("score", 0),
                "intent": result.get("intent", ""),
                "answer_degraded": result.get("answer_degraded", False),
                "context_count": len(result.get("contexts", []) or []),
                "recall_raw_count": len(result.get("recall_raw", []) or []),
                "verification_reasons": [
                    diagnostic_trace.clip_text(item, diagnostic_trace.MAX_REASON_TEXT)
                    for item in (result.get("verification_reasons", []) or [])
                ],
            },
            token=_trace_token,
        )
    contexts = result.get("contexts", [])
    intent = result.get("intent", "kb_question")
    faq_hit = result.get("faq_hit", False)
    needs_clarification = result.get("needs_clarification", False)
    attempts = 0 if needs_clarification else result.get("retrieval_attempts", 0) + 1
    # P2：FAQ 直答/闲聊等非检索路径未经过 evaluate，score 是 initial 的 0，
    # 直接返回会污染忠实度统计——这里统一置 None 表示"未评估"。
    evaluated = (
        intent not in ("smalltalk", "human_request", "out_of_scope")
        and not faq_hit
        and not needs_clarification
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
        "contexts": contexts,
        "retries": result.get("retries", 0),
        "score": result.get("score", 0) if evaluated else None,
        "needs_clarification": needs_clarification,
        "escalate": result.get("escalate", False),
        # 检索元信息（客服改造第3项：检索日志留痕，main.py 落库）
        "search_meta": {
            "rewritten": result.get("rewritten", ""),
            "intent": intent,
            "faq_hit": faq_hit,
            "recall_raw": result.get("recall_raw", []),  # rerank 前候选
            "contexts": contexts,  # rerank 后 top-k
            "top_score": contexts[0].get("score", 0.0) if contexts else 0.0,
            "attempts": attempts,
            "escalate": result.get("escalate", False),
            "needs_clarification": needs_clarification,
        },
    }


def _run_qa_stream_events(
    graph: Any,
    *,
    question: str,
    history: list[dict[str, str]] | None = None,
    kb_id: str = "default",
    metadata_filters: dict[str, Any] | None = None,
    thread_id: str | None = None,
    trace_box: dict[str, Any] | None = None,
) -> Any:
    """run_qa_stream 的事件主体（内部）。

    trace_box：把 trace token 与最终 outcome 回传给外层包装，由外层统一收尾。
    主体自己不落盘，是为了让"客户端中途断开""图抛异常""正常跑完"三种退出
    路径只有一个收尾点，避免重复落盘或漏落盘。
    """
    initial: QaState = {
        "question": question,
        "kb_id": kb_id,
        "intent": "",
        "faq_hit": False,
        "rewritten": "",
        "history": history or [],
        "recall_raw": [],
        "contexts": [],
        "passages": [],
        "passage_records": [],
        "answer": "",
        "citations": [],
        "retries": 0,
        "retry_requested": False,
        "retrieval_attempts": 0,
        "escalate": False,
        "needs_clarification": False,
        "gate_action": "",
        "low_quality": False,
        "score": 0,
        "metadata_filters": metadata_filters,
        "verification_reasons": [],
        "answer_degraded": False,
        # 与 run_qa 一致：诊断追踪令牌随图状态流转
        "trace_id": uuid.uuid4().hex[:12] if diagnostic_trace.is_enabled() else "",
    }
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    final_state: dict[str, Any] = dict(initial)
    pending_custom: list[Any] = []
    token_streamed = False
    evaluation_seen = False
    # P1-2：stream_mode=["updates", "custom"]——updates 是节点进度，custom 是 generate
    # 节点用 get_stream_writer 透传的 LLM token（真 token 级流式）。
    for event in graph.stream(initial, config=config, stream_mode=["updates", "custom"]):
        kind, data = event[0], event[1]
        if kind == "updates":
            node = list(data.keys())[0]
            final_state.update(data[node])
            yield {"type": "node", "node": node, "answer": final_state.get("answer", "")}
            if node == "evaluate":
                evaluation_seen = True
                if final_state.get("retry_requested", False):
                    # 本轮证据核验失败：丢弃已经生成但尚未提交的 token，等待下一轮。
                    pending_custom.clear()
                elif final_state.get("answer_degraded", False):
                    # 最后一轮也未通过核验：生成 token 仍是已知错误数字，全部丢弃。
                    pending_custom.clear()
                elif pending_custom:
                    # 只有 evaluate 确认当前答案不再回 generate 后，才对外提交 token。
                    for token_event in pending_custom:
                        yield token_event
                    pending_custom.clear()
                    token_streamed = True
        elif kind == "custom":
            # token 事件（{"type": "token", "text": ...}）
            pending_custom.append(data)
    if pending_custom and not evaluation_seen:
        # 兼容没有 evaluate 节点的自定义图；标准问答图会在 evaluate 处提交或丢弃。
        for token_event in pending_custom:
            yield token_event
        token_streamed = True
    # 兜底：若 LLM 走非流式分支（无 custom token，如 escalate/refuse 直答），
    # 把最终 answer 分块吐出，保证前端始终能收到打字机增量。
    answer = final_state.get("answer", "")
    if not token_streamed and answer:
        step = 3
        for i in range(0, len(answer), step):
            yield {"type": "token", "text": answer[i : i + step]}
    contexts = final_state.get("contexts", [])
    needs_clarification = final_state.get("needs_clarification", False)
    attempts = 0 if needs_clarification else final_state.get("retrieval_attempts", 0) + 1
    # 诊断追踪：与 run_qa 对齐，但这里只把 outcome 交给外层包装统一落盘
    # （原因见 _run_qa_stream_events 文档字符串）。
    if diagnostic_trace.is_enabled() and trace_box is not None:
        trace_box["token"] = str(initial.get("trace_id") or "")
        trace_box["outcome"] = {
            "status": "ok",
            "answer": diagnostic_trace.clip_text(
                answer, diagnostic_trace.MAX_ANSWER_TEXT
            ),
            "escalate": final_state.get("escalate", False),
            "retries": final_state.get("retries", 0),
            "score": final_state.get("score", 0),
            "intent": final_state.get("intent", ""),
            "answer_degraded": final_state.get("answer_degraded", False),
            "context_count": len(contexts),
            "recall_raw_count": len(final_state.get("recall_raw", []) or []),
            "verification_reasons": [
                diagnostic_trace.clip_text(item, diagnostic_trace.MAX_REASON_TEXT)
                for item in (final_state.get("verification_reasons", []) or [])
            ],
        }
    yield {
        "type": "final",
        "answer": answer,
        "citations": final_state.get("citations", []),
        "contexts": contexts,
        "retries": final_state.get("retries", 0),
        "score": None if needs_clarification else final_state.get("score", 0),
        "escalate": final_state.get("escalate", False),
        "needs_clarification": needs_clarification,
        "answer_degraded": final_state.get("answer_degraded", False),
        # P1：final 补 search_meta，与同步 run_qa 对齐（流式路径也要能落检索日志）
        "search_meta": {
            "rewritten": final_state.get("rewritten", ""),
            "intent": final_state.get("intent", "kb_question"),
            "faq_hit": final_state.get("faq_hit", False),
            "recall_raw": final_state.get("recall_raw", []),
            "contexts": contexts,
            "top_score": contexts[0].get("score", 0.0) if contexts else 0.0,
            "attempts": attempts,
            "escalate": final_state.get("escalate", False),
            "needs_clarification": needs_clarification,
        },
    }


def run_qa_stream(
    graph: Any,
    *,
    question: str,
    history: list[dict[str, str]] | None = None,
    kb_id: str = "default",
    metadata_filters: dict[str, Any] | None = None,
    thread_id: str | None = None,
) -> Any:
    """流式执行问答图：yield 节点进度事件 + 最终结果。

    P2：/kb/ask/stream 用——同步 ask 用户干等 10s+ 无反馈，这里按节点
    （rewrite→retrieve→generate→evaluate）推送进度，SSE 前端可实时展示。

    诊断追踪收尾由本包装层负责：流式生成器可能在**任意一次 yield** 处被
    close()（客户端断开 → GeneratorExit），也可能在图执行中途抛异常，只靠
    主体末尾那一次 finish_request 会漏掉这两种路径，注册表里留下永不落盘的
    残留。这里对三种退出路径分别处理：
      - 正常跑完：落 trace_box 里的正常 outcome；
      - 客户端断开：图已跑完则落正常 outcome，中途断开则只记
        `client_disconnected`，**绝不把未完成请求写成正常成功**；
      - 图异常：落受限长度的错误类型/摘要，然后原样重新抛出，不吞异常。
    收尾一律按 trace_box 里的 token 定位，不会碰其他并发请求的记录。
    """
    trace_box: dict[str, Any] = {}
    try:
        for event in _run_qa_stream_events(
            graph,
            question=question,
            history=history,
            kb_id=kb_id,
            metadata_filters=metadata_filters,
            thread_id=thread_id,
            trace_box=trace_box,
        ):
            yield event
    except GeneratorExit:
        # 客户端提前断开：trace_box 有 outcome 才说明图确实跑完了（只是 final
        # 事件没被接收）；否则只能记为未完成。
        diagnostic_trace.finish_request(
            trace_box.get("outcome") or {"status": "client_disconnected"},
            token=str(trace_box.get("token") or ""),
        )
        raise
    except BaseException as exc:  # noqa: BLE001 —— 收尾后原样抛出
        diagnostic_trace.finish_request(
            diagnostic_trace.error_outcome(exc, status="stream_error"),
            token=str(trace_box.get("token") or ""),
        )
        raise
    else:
        diagnostic_trace.finish_request(
            trace_box.get("outcome") or {"status": "completed_no_outcome"},
            token=str(trace_box.get("token") or ""),
        )
