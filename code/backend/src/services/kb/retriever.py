"""混合检索：向量检索 + BM25 关键词检索 → RRF 融合。

为什么需要混合检索？（面试必讲）
  - 纯向量检索的盲区：关键词精确匹配。比如查"5万元"这种数字/专有名词，
    向量空间里可能找不到精确匹配的 chunk，而 BM25 能直接命中。
  - 纯 BM25 的盲区：同义改写（"招投标" vs "公开招标"）命中不了，向量能兜住。
  - 两者互补 → RRF（Reciprocal Rank Fusion）按排名融合，不依赖分数尺度对齐。

RRF 公式：score(d) = Σ_retriever 1 / (k + rank_retriever(d))，k=60 是常见默认值
  - 只比较"排名"，不比较原始分数 → 向量分数(0-1)和 BM25 分数(无界)可以公平融合

多知识库隔离（P3）：
  - 向量检索带 where={"kb_id":xxx} 过滤
  - BM25 索引按 kb_id 分别构建与缓存——绝不把别的库的 chunk 建进本库索引
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from rank_bm25 import BM25Okapi

from services.kb import diagnostic_trace
from services.kb.embeddings import EmbeddingClient
from services.kb.vector_store import VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # RRF 常数（论文推荐 60）
# BM25 通道在 RRF 融合中的权重。财报语料里"营业收入""净利润"等词极其常见，
# 2-gram BM25 会把大量同行业其他公司的分块排到前面；权重 1.0 时 BM25 的
# top-1（1/61≈0.0164）会压过向量的第 8 名（0.0145），噪声块因此挤占 top-N。
# 默认 0.5，让 BM25 只做补充（提升已召回块）而不主导排序。
# KB_BM25_RRF_WEIGHT=1 回到等权融合。
def _bm25_rrf_weight() -> float:
    raw = os.getenv("KB_BM25_RRF_WEIGHT", "").strip()
    if not raw:
        return 0.5
    try:
        return float(raw)
    except ValueError:
        return 0.5
# ---------------------------------------------------------------------------
# 财务结构化路由（KB_STRUCTURED_FIN_ROUTE=1 开启，默认关闭）
#
# 为什么要这条通道：财务数字问答失败的真正原因不是"文档没召回"，而是
# **块级定位失效** —— 正确文档稳定排在 top-10，但装着正确数字的那一块连
# top-100 都进不去。因为表格数字块在语义向量空间里与问句并不相似（它是一
# 堆数字，不是"营业收入是多少"的语义），而 BM25 也捞不到（问句里根本没有
# 那串数字）。纯语义+关键词检索天然定位不到"某年报第 7 页主要会计数据表
# 里的某一行"。
#
# 但索引里已经写好了 section_path / is_table / report_period / doc_id 这些
# 结构化元数据。两阶段做法：
#   1) 常规检索确定"问的是哪几家公司"（取 top-K 涉及的 doc_id）
#   2) 用 where 过滤 is_table=True + report_period + section_path（会计数据
#      章节）+ doc_id，在这个小池子里重新排序
# 实测（scripts/proto_structured_route.py）：候选池从 7.3 万块收敛到 4~13 块，
# 6 道原本连 top-100 都进不去的题，正确块全部落到 #1~#3。
# ---------------------------------------------------------------------------
_structured_fin_route = os.getenv("KB_STRUCTURED_FIN_ROUTE", "").strip() in (
    "1",
    "true",
    "yes",
)
# 财报里"主要会计数据"表的章节标题写法（与 section_path 做包含匹配）
_FIN_KEY_SECTIONS = ("主要会计数据", "主要财务指标")
# 只有明确问到财务指标才走路由，避免影响普通问答
_FIN_METRIC_HINT_RE = re.compile(
    r"营业收入|营业总收入|净利润|归母净利润|归属于上市公司股东的净利润|"
    r"研发投入|研发费用|现金流量净额|资产总额|负债总额|每股收益|毛利率|净利率"
)
_FIN_PERIOD_RE = re.compile(r"(20\d{2})\s*年")


def _financial_route_periods(query: str) -> tuple[str, ...]:
    """返回问句明确要求的全部报告期间，保持出现顺序且去重。"""
    if not _structured_fin_route:
        return ()
    text = str(query or "")
    if not _FIN_METRIC_HINT_RE.search(text):
        return ()
    periods: list[str] = []
    half_year = "半年" in text or "中期" in text or "1-6" in text or "半年度" in text
    for match in _FIN_PERIOD_RE.finditer(text):
        period = f"{match.group(1)}年半年度" if half_year else f"{match.group(1)}年度"
        if period not in periods:
            periods.append(period)
    return tuple(periods)
# 结构化通道最多插入多少个块。过滤后的池子本身只有 4~14 块，取得太少会漏掉
# 装着目标行的那个切片（同一张会计数据表常被切成多块，只有部分含"营业收入"行）。
_FIN_ROUTE_TOP_N = int(os.getenv("KB_FIN_ROUTE_TOP_N", "8") or 8)
# 元数据不完整或严格命中疑似非核心块时，按单文档扩大候选池；扩大始终受上限约束。
_FIN_ROUTE_FALLBACK_TOP_K = max(_FIN_ROUTE_TOP_N * 4, 32)
# Same-document metadata supplementation is deliberately bounded.  It can
# also run after a strict hit, because the first matching block may be a
# non-canonical table for the same metric.
_FIN_ROUTE_ENUM_MAX = max(_FIN_ROUTE_FALLBACK_TOP_K, 32)
_FIN_ROUTE_NARRATIVE_MAX = max(_FIN_ROUTE_TOP_N, 8)
# Keep only a small, deterministic prefix of valid narrative hits from the
# ordinary first-pass recall ahead of structured injections.  This protects
# page/section evidence without turning raw recall into an unbounded bypass.
_FIN_ROUTE_RAW_NARRATIVE_MAX = 2
# 会计数据表常被切成多块：有的切片是表头/公司名称/单位说明，**不含任何数字**。
# 这类块注入进去只会白占 top-K 名额（实测会把原本 #1 的正确数字块挤到 #4，
# 直接造成一道题退步），因此注入前一律丢弃。
_FIN_NUMBER_RE = re.compile(
    r"(?<!\d)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)"
)
_COMPANY_SUFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "集团公司",
    "集团",
)
_COMPANY_FULL_NAME_RE = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9（）()·&.\-*]{2,80}?(?:股份有限公司|有限责任公司|有限公司|集团公司|集团)"
)
# 股票简称中的 ST 后面是简称本身，不应把紧随其后的年份/问题文本吞进去。
# 允许空格是为了兼容 OCR 常见的“*ST 某企”形式；规范化时会去掉空格。
_COMPANY_ST_RE = re.compile(
    r"\*?S\s*T\s*[\u4e00-\u9fffA-Za-z]{1,12}?"
    r"(?=(?:19|20)\d{2}\s*年|年报|年度|半年度|上半年|下半年|"
    r"营业|收入|利润|研发|现金流|资产|负债|每股|毛利率|净利率|"
    r"[\s\-—_/|：:，,。！？?]|$)",
    re.IGNORECASE,
)
_COMPANY_CODE_RE = re.compile(r"(?<!\d)\d{6}(?!\d)")
_COMPANY_QUERY_PREFIX_RE = re.compile(
    r"^(?:请问|查询|请查询|查一下|查查|请查|帮我(?:查(?:一下)?|查询)?|想知道|关于)"
)
_COMPANY_CONTEXT_RE = re.compile(
    r"营业收入|营业总收入|净利润|归母净利润|研发投入|研发费用|现金流|资产总额|负债总额|"
    r"每股收益|毛利率|净利率|单位\s*[:：]|本报告期|上年同期|项目|合计|万元|亿元|元|%"
)


def _financial_route_period(query: str) -> str:
    """从问句推断目标 report_period；非财务指标问句或无法判断期间时返回空串。

    返回值要与 ingest 写入的 report_period 格式一致（"2024年度"/"2025年半年度"）。
    """
    if not _structured_fin_route:
        return ""
    text = str(query or "")
    if not _FIN_METRIC_HINT_RE.search(text):
        return ""
    match = _FIN_PERIOD_RE.search(text)
    if not match:
        return ""
    year = match.group(1)
    if "半年" in text or "中期" in text or "1-6" in text or "半年度" in text:
        return f"{year}年半年度"
    return f"{year}年度"


def _normalize_company_text(value: Any) -> str:
    return re.sub(r"[\s_\-/:：；;（）()（）]", "", str(value or "").casefold())


def _company_base_name(value: str) -> str:
    normalized = _normalize_company_text(value)
    for suffix in _COMPANY_SUFFIXES:
        suffix_normalized = _normalize_company_text(suffix)
        if normalized.endswith(suffix_normalized):
            return normalized[: -len(suffix_normalized)]
    return normalized


def _company_name_base(value: Any) -> str:
    """返回可用于身份比较的公司主体，不包含公司类型后缀。"""
    base = _company_base_name(str(value or ""))
    # “股份有限公司”等孤立后缀不是公司主体，不能作为身份标识。
    if len(base) < 2 or base in {
        _normalize_company_text(item) for item in _COMPANY_SUFFIXES
    }:
        return ""
    return base


def _add_company_name_markers(
    markers: set[tuple[str, str]], value: Any, *, include_st_alias: bool = True
) -> None:
    """从一个明确的公司/标题字段提取规范化身份标记。"""
    raw = str(value or "")
    if not raw:
        return

    for match in _COMPANY_FULL_NAME_RE.finditer(raw):
        full_name = match.group(0)
        # 标题常把“证券简称-公司全称”连在一起。剥掉前面的简称，
        # 否则同一公司会被误识别为两个不同全称。
        st_match = _COMPANY_ST_RE.match(full_name)
        if st_match:
            full_name = full_name[st_match.end() :].lstrip(" -—_/:：")
        base = _company_name_base(full_name)
        if base:
            markers.add(("name", base))
            markers.add(("alias", base))

    for match in _COMPANY_ST_RE.finditer(raw):
        st_value = _normalize_company_text(match.group(0))
        if not st_value:
            continue
        markers.add(("st", st_value))
        if include_st_alias:
            markers.add(("alias", st_value))
            bare_alias = st_value[2:] if st_value.startswith("st") else st_value
            if len(bare_alias) >= 2:
                markers.add(("alias", bare_alias))

    for match in _COMPANY_CODE_RE.finditer(raw):
        markers.add(("code", match.group(0)))


def _company_title_alias_markers(markers: set[tuple[str, str]], title: Any) -> None:
    """从报告标题的主体段提取证券简称；不扫描报告正文。"""
    raw = str(title or "")
    if not raw:
        return
    before_period = re.split(r"(?:19|20)\d{2}\s*年", raw, maxsplit=1)[0]
    for segment in re.split(r"[-—_/|：:]", before_period):
        normalized = _normalize_company_text(segment)
        if len(normalized) < 2 or len(normalized) > 40:
            continue
        if normalized.startswith("st") and len(normalized) > 2:
            markers.add(("alias", normalized))
            markers.add(("alias", normalized[2:]))
            continue
        if not any(
            normalized.endswith(_normalize_company_text(suffix))
            for suffix in _COMPANY_SUFFIXES
        ):
            markers.add(("alias", normalized))


def _company_markers(source_text: str, *, title: bool = False) -> set[tuple[str, str]]:
    """从一个权威字段提取身份；不会把任意正文中的公司引用聚合为身份。"""
    markers: set[tuple[str, str]] = set()
    _add_company_name_markers(markers, source_text)
    if title:
        _company_title_alias_markers(markers, source_text)
    return markers


def _company_field_markers(
    markers: set[tuple[str, str]], key: str, value: Any
) -> None:
    if value is None or value == "":
        return
    if key in {"stock_code", "ticker", "code"}:
        for match in _COMPANY_CODE_RE.finditer(str(value)):
            markers.add(("code", match.group(0)))
        return
    if key in {"company_aliases", "aliases"}:
        for alias in re.split(r"[|,，;；/、]+", str(value)):
            normalized = _normalize_company_text(alias)
            if len(normalized) >= 2:
                markers.add(("alias", normalized))
                base = _company_name_base(alias)
                if base:
                    markers.add(("alias", base))
        return
    _add_company_name_markers(markers, value)
    normalized = _normalize_company_text(value)
    if key in {"company", "company_name", "issuer", "issuer_name"} and len(normalized) >= 2:
        if not any(normalized.endswith(_normalize_company_text(suffix)) for suffix in _COMPANY_SUFFIXES):
            markers.add(("alias", normalized))


def _company_document_markers(item: dict[str, Any]) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """返回(文档级身份, 受控正文兜底身份)。"""
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    authoritative: set[tuple[str, str]] = set()
    title_keys = {"doc_title", "title"}
    field_keys = (
        "company", "company_name", "issuer", "issuer_name", "company_aliases",
        "aliases", "stock_code", "ticker", "code",
        "source", "file_name", "file_path", "doc_title", "title",
    )
    for key in field_keys:
        value = metadata.get(key)
        if value in (None, ""):
            value = item.get(key)
        if value in (None, ""):
            continue
        if key in title_keys:
            authoritative.update(_company_markers(str(value), title=True))
        else:
            _company_field_markers(authoritative, key, value)

    fallback: set[tuple[str, str]] = set()
    text = str(item.get("text") or "")
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first_line:
        # 只保存首行作为兜底证据；是否使用由路由在文档级权威字段未命中时决定。
        fallback.update(_company_markers(first_line))
    # 财报首个分块有时同时给出“公司简称为 X”。这类显式标签是受控的
    # 文本兜底；普通正文里提到其他公司不会被当作主体。
    for match in re.finditer(
        r"(?:公司简称|证券简称|简称)\s*(?:为|是|[:：])\s*"
        r"([\u4e00-\u9fffA-Za-z0-9*]{2,20})",
        text[:240],
    ):
        alias = _normalize_company_text(match.group(1))
        if len(alias) >= 2:
            fallback.add(("alias", alias))
    return authoritative, fallback


def _company_query_clues(query: str) -> list[tuple[str, str]]:
    """提取问题中的公司线索；没有明确主体时返回空，不猜测召回文档。"""
    text = str(query or "")
    clues: list[tuple[str, str]] = []

    def add(kind: str, value: str) -> None:
        normalized = _normalize_company_text(value)
        if normalized and (kind, normalized) not in clues:
            clues.append((kind, normalized))

    for match in _COMPANY_FULL_NAME_RE.finditer(text):
        full_name = _normalize_company_text(match.group(0))
        add("name", full_name)
        base = _company_base_name(full_name)
        if len(base) >= 2:
            add("alias", base)
    for match in _COMPANY_ST_RE.finditer(text):
        st_value = _normalize_company_text(match.group(0))
        add("st", st_value)
        add("alias", st_value)
        if st_value.startswith("st") and len(st_value[2:]) >= 2:
            add("alias", st_value[2:])
    for match in _COMPANY_CODE_RE.finditer(text):
        add("code", match.group(0))

    # 简称通常位于年份/财务指标之前，例如“格力电器2023年营业收入”。
    prefix = _COMPANY_QUERY_PREFIX_RE.sub("", text.strip(), count=1)
    period_start = re.search(r"(?:19|20)\d{2}\s*年", prefix)
    if period_start:
        prefix = prefix[: period_start.start()]
    else:
        metric_start = _FIN_METRIC_HINT_RE.search(prefix)
        if metric_start:
            prefix = prefix[: metric_start.start()]
    prefix = prefix.strip(" 的，,：:？?\t\r\n")
    if prefix and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9（）()·&.\-*]{2,40}", prefix):
        base = _company_base_name(prefix)
        if len(base) >= 2:
            add("alias", base)
    return clues


def _company_source_text(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    parts: list[str] = []
    for key in ("doc_title", "title", "source", "file_name", "file_path", "company_name", "stock_code"):
        value = metadata.get(key)
        if value:
            parts.append(str(value))
        value = item.get(key)
        if value:
            parts.append(str(value))
    # 页眉通常在正文前部；只取有限前缀，避免把正文中别的公司引用当作主体。
    text = str(item.get("text") or "")
    if text:
        parts.append(text[:240])
    return " ".join(parts)


def _financial_route_doc_ids(
    query: str, corpus: list[dict[str, Any]]
) -> tuple[str, ...]:
    """把明确主体映射到当前 kb 的有限 doc_id；歧义一律不启用路由。"""
    clues = _company_query_clues(query)
    if not clues:
        return ()

    documents: dict[
        str, tuple[set[tuple[str, str]], set[tuple[str, str]]]
    ] = {}
    for item in corpus:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        doc_id = str(metadata.get("doc_id") or item.get("doc_id") or "").strip()
        if not doc_id:
            continue
        authoritative, fallback = _company_document_markers(item)
        existing = documents.get(doc_id)
        if existing:
            documents[doc_id] = (
                existing[0] | authoritative,
                existing[1] | fallback,
            )
        else:
            documents[doc_id] = (authoritative, fallback)

    matches_by_clue: dict[tuple[str, str], set[str]] = {}
    for clue_kind, clue_value in clues:
        def _matches(markers: set[tuple[str, str]]) -> bool:
            if clue_kind == "code":
                return ("code", clue_value) in markers
            elif clue_kind == "st":
                return ("st", clue_value) in markers
            elif clue_kind == "name":
                name_base = _company_name_base(clue_value)
                return bool(name_base) and (
                    ("name", name_base) in markers or ("alias", name_base) in markers
                )
            else:
                return ("alias", clue_value) in markers

        # 优先在文档级字段/标题中匹配；只有该线索完全没有权威命中时，
        # 才使用每个文档首行的受控兜底，避免正文提及别家公司扩大范围。
        matched = {
            doc_id
            for doc_id, (authoritative, _fallback) in documents.items()
            if _matches(authoritative)
        }
        if not matched:
            # A sufficiently long query alias may be a unique prefix of the
            # authoritative company name.  Never use fallback/body markers
            # for this expansion, and reject ambiguous prefixes outright.
            if clue_kind == "alias" and len(clue_value) >= 4:
                prefix_matches = {
                    doc_id
                    for doc_id, (authoritative, _fallback) in documents.items()
                    if any(
                        marker_kind == "name"
                        and marker_value.startswith(clue_value)
                        for marker_kind, marker_value in authoritative
                    )
                }
                if len(prefix_matches) > 1:
                    return ()
                matched = prefix_matches
        if not matched:
            matched = {
                doc_id
                for doc_id, (_authoritative, fallback) in documents.items()
                if _matches(fallback)
            }
        if matched:
            matches_by_clue[(clue_kind, clue_value)] = matched

    if not matches_by_clue:
        return ()
    # 强线索优先；普通简称仅在没有全称/ST/股票代码命中时使用。
    selected: set[str] = set()
    for kind in ("code", "st", "name", "alias"):
        candidates = set().union(
            *(ids for (clue_kind, _), ids in matches_by_clue.items() if clue_kind == kind)
        )
        if candidates:
            selected = candidates
            break
    if not selected:
        return ()

    # 明确线索不能互相指向不同文档集合；否则不依据排名猜主体。
    for candidates in matches_by_clue.values():
        if not candidates.intersection(selected):
            return ()

    # 同一简称若命中带有不同全称、ST 标识或股票代码的文档，视为主体歧义。
    identity_values: set[tuple[str, str]] = set()
    query_identity_values = {
        (kind, value)
        for kind, value in clues
        if kind in {"name", "st", "code", "alias"}
    }
    for doc_id in selected:
        markers = documents[doc_id][0] or documents[doc_id][1]
        for kind, value in markers:
            if kind in {"name", "st", "code"}:
                identity_values.add((kind, value))
    for kind in ("name", "st", "code"):
        values = {
            value for marker_kind, value in identity_values if marker_kind == kind
        }
        relevant_query_kinds = {kind}
        if kind == "name":
            relevant_query_kinds.add("alias")
        if kind == "st":
            relevant_query_kinds.add("alias")
        exact_query_values = {
            value
            for query_kind, value in query_identity_values
            if query_kind in relevant_query_kinds and value in values
        }
        # Prefer an exact identity clue from the question when a title has a
        # duplicated/concatenated alias.  This does not broaden the match:
        # conflicting exact markers remain ambiguous and are rejected.
        if exact_query_values:
            if len(exact_query_values) > 1:
                return ()
            continue
        prefix_query_values = {
            value
            for query_kind, value in query_identity_values
            if query_kind == "alias"
            and len(value) >= 4
            and any(identity.startswith(value) for identity in values)
        }
        if prefix_query_values:
            canonical_values: set[str] = set()
            for identity in values:
                canonical = identity
                for prefix in prefix_query_values:
                    while canonical.startswith(prefix + prefix):
                        canonical = prefix + canonical[len(prefix) * 2 :]
                canonical_values.add(canonical)
            if len(canonical_values) == 1:
                continue
        if len(values) > 1:
            return ()
    return tuple(sorted(selected))


def _has_financial_number(text: str) -> bool:
    """允许普通整数金额，但排除孤立年份、页码和股票代码块。"""
    values = list(_FIN_NUMBER_RE.finditer(str(text or "")))
    if not values:
        return False
    context = bool(_FIN_METRIC_HINT_RE.search(text) or _COMPANY_CONTEXT_RE.search(text) or "|" in text)
    meaningful = []
    for match in values:
        raw = match.group(0).replace(",", "")
        try:
            number = int(raw) if "." not in raw else float(raw)
        except ValueError:
            continue
        if isinstance(number, int) and 1900 <= number <= 2100:
            continue
        if re.search(rf"(?:第\s*|页码\s*){re.escape(match.group(0))}\s*页?", text):
            continue
        if isinstance(number, int) and len(raw) == 6 and not context:
            continue
        meaningful.append(match)
    if not meaningful:
        return False
    return context or any("." in item.group(0) or "," in item.group(0) for item in meaningful)


# 中文分词简化：按非字母数字切分 + 过滤单字符/停用词
_STOPWORDS = {
    "的", "了", "和", "与", "或", "在", "是", "有", "为", "对", "把", "被",
    "这", "那", "个", "等", "及", "并", "而", "从", "到", "于", "之", "其",
    "公司", "我们", "你们", "他们", "以及", "关于", "进行", "一个", "如何",
}


def _tokenize(text: str) -> list[str]:
    """简易中文分词：中文按 2-gram 拆，数字/字母段独立成词。

    修复（重要）：原实现用 re.split(r"[^\\w\\u4e00-\\u9fff]+")，而 Python 的
    \\w 默认已包含中文和数字，含年份的中文问句（如"裕太微2025年上半年营业
    收入是多少"）整句不会被切分；又因为它含数字，re.fullmatch(r"[\\u4e00-\\u9fff]+")
    不成立，走不到 2-gram 分支，最终退化成 **一个超长 token**。语料里不可能
    出现这个整句，BM25 分数全为 0，被 `scores[idx] > 0` 全部丢弃 —— 混合检索
    实际退化成纯向量检索，财务问答（几乎都带年份）的关键词召回能力完全失效。
    改为先按"中文段 / 数字字母段"切开，再各自成词。
    """
    lowered = str(text or "").lower()
    tokens: list[str] = []
    for segment in re.findall(r"[\u4e00-\u9fff]+|[0-9a-z]+", lowered):
        if not segment:
            continue
        # 中文按 2-gram 拆（弥补未用分词器）：如 "招投标" → ["招投","投标"]
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment) and len(segment) > 1:
            tokens.extend(segment[i : i + 2] for i in range(len(segment) - 1))
            tokens.append(segment)  # 整词也保留（长词匹配更精准）
        else:
            tokens.append(segment)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# 纯页眉/页脚噪声块：PDF 抽取常把每页页眉（"XX公司2025年半年度报告"）单独切
# 成一个分块。这类块不含任何实质内容，却同时命中查询里的公司名、年份和报告
# 类型 —— 向量与 BM25 两个通道都给高分，把真正带数字的分块挤出 top-N。
# 判据收紧为"极短 + 完全匹配页眉模式"，避免误伤有内容的短块。
# 实测（36 题离线检索 A/B）：开启后证据召回 ev@5 完全不变（12/36），但页码
# 召回 page@10 反而从 45.5% 掉到 27.3% —— 页眉块虽无实质文字，却携带该页的
# page 元数据，删掉它们等于丢掉这些页在 top-K 里的占位。因此**默认关闭**，
# KB_FILTER_LOW_INFO_CHUNKS=1 可显式开启做对照。
_filter_low_info_chunks = os.getenv("KB_FILTER_LOW_INFO_CHUNKS", "").strip() in (
    "1",
    "true",
    "yes",
)

_LOW_INFO_HEADER_RE = re.compile(
    r".{2,40}(?:19|20)\d{2}年(?:半年度|年度)?报告(?:全文|摘要)?"
)


def _is_low_information_chunk(text: str) -> bool:
    """仅含页眉/页脚的分块：公司名 + 年份 + 报告类型，无实质内容。"""
    if not _filter_low_info_chunks:
        return False
    compact = re.sub(r"\s+", "", str(text or ""))
    if not compact or len(compact) > 45:
        return False
    return bool(_LOW_INFO_HEADER_RE.fullmatch(compact))


_FINANCIAL_QUERY_GROUPS: dict[str, frozenset[str]] = {
    "revenue": frozenset(("营业收入", "营业总收入")),
    "listed_net_profit": frozenset(("归属于上市公司股东的净利润", "归属于母公司股东的净利润", "归属于母公司所有者的净利润", "归母净利润")),
    "rd_total": frozenset(("研发投入合计", "研发投入总额", "研发投入")),
    "rd_expense": frozenset(("研发费用",)),
    "operating_cash_flow": frozenset(("经营活动产生的现金流量净额", "经营活动现金流量净额", "经营活动产生现金流量净额")),
}
_FINANCIAL_QUERY_ALIASES = tuple(
    sorted(
        (
            (alias, group)
            for group, aliases in _FINANCIAL_QUERY_GROUPS.items()
            for alias in aliases
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)
_CAUSE_QUERY_RE = re.compile(r"为什么|原因|主要系|所致|变动原因|受[^，。；！？]{0,24}影响|由于|导致")
_CAUSE_EVIDENCE_RE = re.compile(r"主要系|主要原因|原因在于|所致|受[^，。；！？]{0,24}影响|由于|导致|因素")
_FINANCIAL_QUERY_RE = re.compile(r"营业|收入|利润|研发|现金流|资产|负债|财务|每股|毛利率|净利率")
_NUMERIC_QUERY_RE = re.compile(r"多少|数额|金额|合计|同比|增长率|数值|是多少|百分比|占比|万元|亿元|元")
_SECTION_QUERY_TERMS = ("经营情况概述", "经营情况", "管理层讨论与分析", "财务报表", "利润表", "现金流量表")


def _query_profile(query: str) -> dict[str, Any]:
    """提取可解释的轻量查询意图，不改变原始查询文本。"""
    compact = re.sub(r"\s+", "", str(query or ""))
    metric_groups = {
        group
        for alias, group in _FINANCIAL_QUERY_ALIASES
        if alias in compact
    }
    explicit_scope = ""
    for marker, scope in (("母公司", "parent"), ("子公司", "subsidiary"), ("合并", "consolidated")):
        if marker in compact:
            explicit_scope = scope
            break
    is_cause = bool(_CAUSE_QUERY_RE.search(compact))
    section_terms = tuple(term for term in _SECTION_QUERY_TERMS if term in compact)
    is_financial = bool(metric_groups or _FINANCIAL_QUERY_RE.search(compact))
    return {
        "metric_groups": frozenset(metric_groups),
        "explicit_scope": explicit_scope,
        "is_cause": is_cause,
        "is_financial": is_financial,
        "is_numeric": bool(metric_groups and (_NUMERIC_QUERY_RE.search(compact) or not is_cause)),
        "section_terms": section_terms,
    }


def _merge_query_profiles(queries: list[str]) -> dict[str, Any]:
    """合并多查询意图；原始问题决定口径，补充查询只扩充指标/章节信号。"""
    if not queries:
        return _query_profile("")
    profile = _query_profile(queries[0])
    metric_groups = set(profile["metric_groups"])
    section_terms = set(profile["section_terms"])
    is_cause = profile["is_cause"]
    for query in queries[1:]:
        extra = _query_profile(query)
        metric_groups.update(extra["metric_groups"])
        section_terms.update(extra["section_terms"])
        is_cause = is_cause or extra["is_cause"]
    profile["metric_groups"] = frozenset(metric_groups)
    profile["section_terms"] = tuple(term for term in _SECTION_QUERY_TERMS if term in section_terms)
    profile["is_cause"] = is_cause
    profile["is_financial"] = bool(profile["is_financial"] or metric_groups)
    profile["is_numeric"] = bool(metric_groups and (_NUMERIC_QUERY_RE.search(queries[0]) or not is_cause))
    return profile


def _financial_route_queries(
    query: str, profile: dict[str, Any]
) -> tuple[tuple[str, str], ...]:
    """为每个请求指标生成独立的结构化检索表达。"""
    groups = tuple(sorted(profile["metric_groups"]))
    if not groups:
        return ()
    section_terms = profile["section_terms"] or _FIN_KEY_SECTIONS
    section_text = " ".join(section_terms)
    return tuple(
        (
            group,
            "\n".join(
                (
                    str(query),
                    "财务指标：" + " ".join(
                        sorted(_FINANCIAL_QUERY_GROUPS[group], key=len, reverse=True)
                    ),
                    "章节：" + section_text,
                )
            ),
        )
        for group in groups
    )


def _financial_route_hit_matches_group(
    hit: dict[str, Any], group: str
) -> bool:
    """只保留当前指标的表格块，避免多指标查询互相借用数字。"""
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    aliases = _FINANCIAL_QUERY_GROUPS[group]
    stored = _stored_financial_metrics(metadata)
    if group in stored or stored.intersection(aliases):
        return True
    searchable = re.sub(
        r"\s+",
        "",
        "".join(
            (
                str(metadata.get("table_name") or ""),
                str(hit.get("text") or ""),
            )
        ),
    )
    return any(alias in searchable for alias in aliases)


def _financial_route_hit_matches_period(
    hit: dict[str, Any],
    periods: tuple[str, ...],
    *,
    inherited_periods: set[str] | None = None,
) -> bool:
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    stored_period = str(metadata.get("report_period") or "").strip()
    if stored_period:
        return stored_period in periods
    if inherited_periods is not None:
        # A blank chunk period may inherit only an unambiguous period from the
        # already locked document.  Do not infer it from arbitrary body text.
        authoritative = " ".join(
            str(metadata.get(key) or "")
            for key in ("doc_title", "title", "source", "file_name", "file_path")
        )
        years = set(re.findall(r"(?:19|20)\d{2}", authoritative))
        if years:
            half_markers = (
                "\u534a\u5e74",
                "\u4e0a\u534a\u5e74",
                "\u4e0b\u534a\u5e74",
                "\u4e2d\u671f",
            )
            is_half_year = any(marker in authoritative for marker in half_markers)
            title_periods = {
                f"{year}\u5e74\u534a\u5e74\u5ea6" if is_half_year else f"{year}\u5e74\u5ea6"
                for year in years
            }
            return title_periods.issubset(set(periods))
        return bool(inherited_periods) and inherited_periods.issubset(set(periods))
    searchable = " ".join(
        str(metadata.get(key) or "")
        for key in ("doc_title", "title", "source", "file_name", "file_path")
    )
    searchable += " " + str(hit.get("text") or "")[:240]
    for period in periods:
        year = period[:4]
        if year not in searchable:
            continue
        if "半年度" in period and not any(
            marker in searchable for marker in ("半年度", "上半年", "下半年", "中期")
        ):
            continue
        return True
    return False


def _financial_route_document_periods(
    corpus: list[dict[str, Any]], doc_ids: set[str]
) -> dict[str, set[str]]:
    """Collect authoritative periods for already locked documents only."""
    periods_by_doc = {doc_id: set() for doc_id in doc_ids}
    fields = ("doc_title", "title", "source", "file_name", "file_path")
    for item in corpus:
        doc_id = _doc_key(item)
        if doc_id not in periods_by_doc:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        stored_period = str(metadata.get("report_period") or "").strip()
        if stored_period:
            periods_by_doc[doc_id].add(stored_period)

    # If ingestion did not put report_period on any chunk, derive it only from
    # document-level identity fields, never from arbitrary chunk body text.
    for doc_id, known_periods in periods_by_doc.items():
        if known_periods:
            continue
        authoritative = " ".join(
            str((item.get("metadata") or {}).get(field) or "")
            for item in corpus
            if _doc_key(item) == doc_id
            for field in fields
        )
        years = set(re.findall(r"(?:19|20)\d{2}", authoritative))
        half_markers = ("\u534a\u5e74", "\u4e0a\u534a\u5e74", "\u4e0b\u534a\u5e74", "\u4e2d\u671f")
        is_half_year = any(marker in authoritative for marker in half_markers)
        for year in years:
            known_periods.add(
                f"{year}\u5e74\u534a\u5e74\u5ea6" if is_half_year else f"{year}\u5e74\u5ea6"
            )
    return periods_by_doc


def _financial_route_hit_matches_section(
    hit: dict[str, Any], section_terms: tuple[str, ...]
) -> bool:
    if not section_terms:
        return False
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    section_path = str(metadata.get("section_path") or "")
    heading_text = str(hit.get("text") or "")[:240]
    return any(term in section_path or term in heading_text for term in section_terms)


def _metadata_is_table(metadata: dict[str, Any]) -> bool:
    value = metadata.get("is_table")
    return value is True or str(value or "").strip().casefold() in {"true", "1", "yes"}


def _stored_financial_metrics(metadata: dict[str, Any]) -> set[str]:
    return {item.strip() for item in str(metadata.get("financial_metrics") or "").split("|") if item.strip()}


def _metric_match_kind(hit: dict[str, Any], profile: dict[str, Any]) -> str:
    if not profile["metric_groups"]:
        return ""
    metadata = hit.get("metadata") or {}
    stored = _stored_financial_metrics(metadata)
    for group in profile["metric_groups"]:
        if stored & _FINANCIAL_QUERY_GROUPS[group]:
            return "fact"
    table_or_text = "".join(
        (
            str(metadata.get("table_name") or ""),
            str(hit.get("text") or ""),
        )
    )
    compact = re.sub(r"\s+", "", table_or_text)
    if any(alias in compact for group in profile["metric_groups"] for alias in _FINANCIAL_QUERY_GROUPS[group]):
        return "header"
    return ""


def _candidate_rank_adjustment(hit: dict[str, Any], profile: dict[str, Any]) -> float:
    """返回小幅确定性加权；数值用于排序，不替代原始 RRF 分。"""
    metadata = hit.get("metadata") or {}
    text = str(hit.get("text") or "")
    compact_text = re.sub(r"\s+", "", text)
    adjustment = 0.0

    metric_match = _metric_match_kind(hit, profile)
    if profile["is_numeric"] and metric_match == "fact":
        adjustment += 0.018
    elif profile["is_numeric"] and metric_match == "header":
        adjustment += 0.008

    if profile["is_numeric"]:
        scope = str(metadata.get("statement_scope") or "unknown")
        requested_scope = profile["explicit_scope"]
        if requested_scope:
            adjustment += 0.005 if scope == requested_scope else -0.003
        elif scope == "consolidated":
            adjustment += 0.005
        elif scope in {"parent", "subsidiary"}:
            adjustment -= 0.002

    if profile["is_cause"]:
        is_narrative = bool(_CAUSE_EVIDENCE_RE.search(compact_text))
        if is_narrative and not _metadata_is_table(metadata):
            adjustment += 0.012
        elif _metadata_is_table(metadata) and not is_narrative:
            adjustment -= 0.006

    section_path = str(metadata.get("section_path") or "")
    for term in profile["section_terms"]:
        if term in section_path:
            adjustment += 0.008
        elif term in compact_text:
            adjustment += 0.003
    return round(adjustment, 6)


def _cause_candidate_priority(
    hit: dict[str, Any],
    profile: dict[str, Any],
    route_queries: tuple[tuple[str, str], ...],
) -> int:
    """Rank eligible cause narratives ahead of ordinary financial tables."""
    if not profile["is_cause"]:
        return 0
    metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
    if _metadata_is_table(metadata):
        return 0
    compact = re.sub(r"\s+", "", str(hit.get("text") or ""))
    if not _CAUSE_EVIDENCE_RE.search(compact):
        return 0
    aliases = tuple(
        alias
        for group, _route_query in route_queries
        for alias in sorted(_FINANCIAL_QUERY_GROUPS[group], key=len, reverse=True)
    )
    field_bound = any(
        re.search(
            rf"{re.escape(alias)}.{{0,24}}(?:变动原因说明|原因说明)",
            compact,
        )
        for alias in aliases
    )
    return 2 if field_bound else 1


def _metadata_matches(metadata: dict[str, Any] | None, where: dict[str, Any] | None) -> bool:
    """在 BM25 语料上复用 VectorStore where 的常见 Chroma 过滤语义。

    向量库仍负责真正的 where 查询；这里的轻量解释器只保证 BM25 不绕过同一组
    过滤条件。支持简单字段、比较操作以及 $and/$or/$not，足够覆盖 metadata
    过滤的公开接口，也不会把 kb_id 隔离交给调用方。
    """
    if not where:
        return True
    metadata = metadata or {}
    for key, condition in where.items():
        if key == "$and":
            if not all(_metadata_matches(metadata, item) for item in condition):
                return False
            continue
        if key == "$or":
            if not any(_metadata_matches(metadata, item) for item in condition):
                return False
            continue
        if key == "$not":
            if _metadata_matches(metadata, condition):
                return False
            continue

        actual = metadata.get(key)
        if isinstance(condition, dict):
            for operator, expected in condition.items():
                if operator in ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte"):
                    try:
                        if operator == "$eq" and actual != expected:
                            return False
                        if operator == "$ne" and actual == expected:
                            return False
                        if operator == "$gt" and not (actual is not None and actual > expected):
                            return False
                        if operator == "$gte" and not (actual is not None and actual >= expected):
                            return False
                        if operator == "$lt" and not (actual is not None and actual < expected):
                            return False
                        if operator == "$lte" and not (actual is not None and actual <= expected):
                            return False
                    except TypeError:
                        return False
                elif operator == "$in":
                    if actual not in expected:
                        return False
                elif operator == "$nin":
                    if actual in expected:
                        return False
                else:
                    # 未知操作符不能让 BM25 越过过滤条件。
                    return False
        elif actual != condition:
            return False
    return True


def _candidate_key(hit: dict[str, Any]) -> str:
    """返回跨查询去重键；正常数据优先使用稳定的 chunk_id。"""
    chunk_id = str(hit.get("chunk_id") or "")
    if chunk_id:
        return f"chunk:{chunk_id}"
    metadata = hit.get("metadata") or {}
    return "text:{0}\0{1}".format(metadata.get("doc_id", ""), hit.get("text", ""))


def _doc_key(hit: dict[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    return str(metadata.get("doc_id") or hit.get("doc_id") or hit.get("chunk_id") or "")


def _dedupe_and_diversify(
    hits: list[dict[str, Any]], max_per_doc: int | None
) -> list[dict[str, Any]]:
    """稳定去重，并按文档轮转选择，避免一个文档占满候选。"""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        key = _candidate_key(hit)
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)

    if max_per_doc is None or max_per_doc <= 0:
        return unique

    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    # 每轮从各文档取一个，既保持原排序优先级，又让后面的文档有机会进入候选。
    for _ in range(max_per_doc):
        for hit in unique:
            key = _candidate_key(hit)
            if key in selected_keys:
                continue
            doc_key = _doc_key(hit)
            if sum(1 for item in selected if _doc_key(item) == doc_key) >= max_per_doc:
                continue
            selected.append(hit)
            selected_keys.add(key)
    return selected


class HybridRetriever:
    """向量 + BM25 混合检索器。BM25 索引按知识库分别构建与缓存（P3）。"""

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        embeddings: EmbeddingClient,
        top_k: int = 5,
        min_score: float = 0.0,
        max_candidates_per_doc: int | None = None,
        max_per_doc: int | None = None,
    ):
        self._store = vector_store
        self._embeddings = embeddings
        self._top_k = top_k
        # 相关性阈值：余弦相似度 < min_score 的向量结果视为未命中，不参与融合。
        # 0 = 不过滤（向后兼容）。真实语料建议实测后调（BGE-M3 相关文档通常 0.5+）。
        self._min_score = min_score
        self._max_per_doc = (
            max_per_doc if max_per_doc is not None else max_candidates_per_doc
        )
        # 按知识库分别缓存 BM25 索引（多库隔离，避免跨库串数据）
        self._bm25: dict[str, BM25Okapi | None] = {}
        self._corpus: dict[str, list[dict[str, Any]]] = {}
        # P2-1：记录每个库上次构建时的写序号（mutation_seq），替代 count() 全表扫描
        self._seq_snapshot: dict[str, int] = {}

    def _rebuild_if_needed(self, kb_id: str) -> None:
        """该库写序号变化后重建 BM25 索引（内存 seq 比对，跳过 count 全表扫）。

        按 kb 维护的 seq（P1 复核）：只有本库有写入才重建，其它库写入不影响。
        多 worker 下以本地写为准（进程内缓存各自独立）。
        """
        current_seq = self._store.mutation_seq(kb_id)
        if kb_id not in self._bm25 or current_seq != self._seq_snapshot.get(kb_id):
            corpus = self._store.all_items(kb_id=kb_id)
            tokenized = [_tokenize(c["text"]) for c in corpus]
            self._bm25[kb_id] = BM25Okapi(tokenized) if tokenized else None
            self._corpus[kb_id] = corpus
            self._seq_snapshot[kb_id] = current_seq
            logger.info("BM25 索引重建（kb=%s）：%d 个分块", kb_id, len(corpus))

    def _search_one(
        self,
        query: str,
        *,
        top_k: int,
        kb_id: str,
        metadata_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """执行一次查询；top_k 是明确的候选数，不在检索器内隐式放大。"""
        self._rebuild_if_needed(kb_id)

        # ---------- 1. 向量检索 Top-K（带 kb_id 过滤）----------
        qvec = self._embeddings.embed_query(query)
        vec_hits = self._store.search(
            qvec, top_k=top_k, where=metadata_filters, kb_id=kb_id
        )
        # 相关性阈值：低相似度的向量结果视为未命中，不参与融合（避免硬凑 top-N 垃圾）
        if self._min_score > 0:
            vec_hits = [h for h in vec_hits if h.get("score", 0.0) >= self._min_score]
        vec_ranks = {h["chunk_id"]: i for i, h in enumerate(vec_hits)}

        # ---------- 2. BM25 检索 Top-K（在该库的索引上）----------
        bm25_ranks: dict[str, int] = {}
        base_corpus = self._corpus.get(kb_id, [])
        if metadata_filters:
            # 缓存的是完整 kb 语料；带过滤条件时构建本次查询的轻量索引，保证
            # 被过滤的文档既不会得 BM25 分，也不会挤占 BM25 的名次。
            corpus = [
                item
                for item in base_corpus
                if _metadata_matches(item.get("metadata"), metadata_filters)
            ]
            filtered_tokens = [_tokenize(item["text"]) for item in corpus]
            bm25 = BM25Okapi(filtered_tokens) if filtered_tokens else None
        else:
            corpus = base_corpus
            bm25 = self._bm25.get(kb_id)
        if bm25 and corpus:
            tokens = _tokenize(query)
            if tokens:
                scores = bm25.get_scores(tokens)
                # 按分数排序取 Top-N
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                bm25_rank = 0
                for idx in order:
                    if bm25_rank >= top_k:
                        break
                    cid = corpus[idx]["chunk_id"]
                    if scores[idx] > 0:  # 零分（无命中）不参与融合
                        bm25_ranks[cid] = bm25_rank
                        bm25_rank += 1

        # ---------- 3. RRF 融合 ----------
        fused: dict[str, float] = {}
        bm25_weight = _bm25_rrf_weight()
        for cid, rank in vec_ranks.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        for cid, rank in bm25_ranks.items():
            fused[cid] = fused.get(cid, 0.0) + bm25_weight / (RRF_K + rank + 1)

        # 按融合分排序（不在这里截断：低信息量块要在截断前剔除，
        # 否则被过滤的名额无法由后面的有效块递补）
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # 补全完整信息（从向量结果 / corpus 中取）
        vec_by_id = {h["chunk_id"]: h for h in vec_hits}
        corpus_by_id = {c["chunk_id"]: c for c in corpus}
        results = []
        for cid, score in ordered:
            if cid in vec_by_id:
                item = dict(vec_by_id[cid])
            elif cid in corpus_by_id:
                item = {
                    "chunk_id": cid,
                    "text": corpus_by_id[cid]["text"],
                    "metadata": corpus_by_id[cid]["metadata"],
                    "distance": None,
                    "score": 0.0,
                }
            else:
                continue
            # 纯页眉/页脚块不占位：剔除后由后续有效块递补，保证仍返回 top_k 条
            if _is_low_information_chunk(item.get("text")):
                continue
            item["rrf_score"] = round(score, 4)  # 融合分（调试/展示用）
            results.append(item)
            if len(results) >= top_k:
                break
        return results

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        kb_id: str = "default",
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """兼容旧单查询接口；多查询请使用 search_queries。"""
        return self.search_queries(
            [query], top_k=top_k, kb_id=kb_id, metadata_filters=metadata_filters
        )

    def search_queries(
        self,
        queries: list[str] | tuple[str, ...],
        *,
        top_k: int | None = None,
        kb_id: str = "default",
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """检索多条查询并稳定融合，第一条查询的顺序优先。

        调用方传入的第一条查询应为原始问题；后续查询只是补充表达。相同 chunk
        在不同查询中只保留一份，但会累加各查询的 RRF 贡献。
        """
        k = self._top_k if top_k is None else max(int(top_k), 0)
        if k <= 0:
            return []
        unique_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in queries:
            normalized = str(query or "").strip()
            if normalized and normalized not in seen_queries:
                unique_queries.append(normalized)
                seen_queries.add(normalized)
        if not unique_queries:
            return []

        fused: dict[str, tuple[float, int, dict[str, Any]]] = {}
        sequence = 0
        for query in unique_queries:
            for hit in self._search_one(
                query,
                top_k=k,
                kb_id=kb_id,
                metadata_filters=metadata_filters,
            ):
                key = _candidate_key(hit)
                contribution = float(hit.get("rrf_score") or 0.0)
                if key in fused:
                    old_score, first_seen, old_hit = fused[key]
                    fused[key] = (old_score + contribution, first_seen, old_hit)
                else:
                    fused[key] = (contribution, sequence, dict(hit))
                sequence += 1

        profile = _merge_query_profiles(unique_queries)
        ordered = sorted(
            fused.values(),
            key=lambda item: (
                -(item[0] + _candidate_rank_adjustment(item[2], profile)),
                item[1],
                _candidate_key(item[2]),
            ),
        )
        if diagnostic_trace.is_enabled():
            diagnostic_trace.add_stage(
                "raw_recall",
                {
                    "kb_id": kb_id,
                    "queries": list(unique_queries),
                    "fused_count": len(fused),
                    "candidates": diagnostic_trace.candidate_snapshot(
                        [hit for _score, _seq, hit in ordered], "vector_bm25_fusion"
                    ),
                },
            )
        ordered = self._merge_financial_route(
            ordered,
            query=unique_queries[0],
            top_k=k,
            kb_id=kb_id,
            metadata_filters=metadata_filters,
        )
        results: list[dict[str, Any]] = []
        for score, _, hit in ordered:
            hit["rrf_score"] = round(score, 4)
            results.append(hit)
        deduped = _dedupe_and_diversify(results, self._max_per_doc)
        final = deduped[:k]
        if diagnostic_trace.is_enabled():
            # 用 id() 判定归属：去重/多样化返回的是同一批 dict 对象，不做内容比较。
            deduped_ids = {id(hit) for hit in deduped}
            final_ids = {id(hit) for hit in final}
            diagnostic_trace.add_stage(
                "post_merge",
                {
                    "kb_id": kb_id,
                    "top_k": k,
                    "max_per_doc": self._max_per_doc,
                    "merged_order": diagnostic_trace.ranking_ids(
                        [hit for _score, _seq, hit in ordered]
                    ),
                    "after_dedupe": diagnostic_trace.ranking_ids(deduped),
                    "dropped_by_dedupe": diagnostic_trace.ranking_ids(
                        [hit for hit in results if id(hit) not in deduped_ids]
                    ),
                    "dropped_by_topk": diagnostic_trace.ranking_ids(
                        [hit for hit in deduped if id(hit) not in final_ids]
                    ),
                    "final_candidates": diagnostic_trace.candidate_snapshot(
                        final, "final"
                    ),
                },
            )
        return final

    def _merge_financial_route(
        self,
        ordered: list[tuple[float, int, dict[str, Any]]],
        *,
        query: str,
        top_k: int,
        kb_id: str,
        metadata_filters: dict[str, Any] | None,
    ) -> list[tuple[float, int, dict[str, Any]]]:
        """把"结构化定位"命中的块提到候选最前面。

        只在 KB_STRUCTURED_FIN_ROUTE=1 且问句是"财务指标 + 年份"时生效。
        命中的块会被赋予高于常规结果的分值，保证它们既不会被后续截断丢弃，
        也仍然受 max_per_doc / 去重规则的约束。任何异常都静默降级为常规结果，
        绝不能因为这条通道让检索整体失败。
        """
        # 追踪用的路由决策快照。**诊断关闭时必须完全不构造**：不遍历、不裁剪、
        # 不复制任何候选——这些字段除了写进追踪文件没有任何用途，无条件构造
        # 就是在正常检索热路径上白烧 CPU 和内存（候选量按 recall_k=20 计）。
        # route 里剩下的标量（period / 计数 / 布尔开关）是 O(1) 的控制流副产品，
        # 不构成候选遍历，保留它们换取代码可读性。
        _tracing = diagnostic_trace.is_enabled()
        route: dict[str, Any] = {"enabled": _structured_fin_route, "route_top_n": _FIN_ROUTE_TOP_N}

        def _emit_route() -> None:
            if diagnostic_trace.is_enabled():
                diagnostic_trace.add_stage("structured_fin_route", dict(route))

        if not ordered:
            route["outcome"] = "skipped_empty_ordered"
            _emit_route()
            return ordered
        periods = _financial_route_periods(query)
        period = periods[0] if periods else ""
        route["period"] = period
        if _tracing:
            route["periods"] = list(periods)
        if not periods:
            route["outcome"] = "skipped_no_period_in_query"
            _emit_route()
            return ordered
        try:
            # 阶段一：只从问题主体映射当前 kb 语料中的同公司 doc_id；
            # 普通召回结果只用于排序，不能扩大公司范围。
            doc_ids = list(
                _financial_route_doc_ids(query, self._corpus.get(kb_id, []))
            )
            if _tracing:
                route["doc_ids"] = doc_ids[:20]
            if not doc_ids:
                route["outcome"] = "skipped_no_company_doc_ids"
                return ordered
            # 阶段二：限定 会计数据表 + 期间 + 公司
            profile = _query_profile(query)
            route_queries = _financial_route_queries(query, profile)
            if not route_queries:
                route["outcome"] = "skipped_no_metric_in_query"
                _emit_route()
                return ordered
            asks_full_and_summary = "全文" in query and "摘要" in query
            corpus = self._corpus.get(kb_id, [])
            if _tracing:
                route["metric_groups"] = [group for group, _ in route_queries]
                route["section_terms"] = list(
                    profile["section_terms"] or _FIN_KEY_SECTIONS
                )
            doc_id_set = set(doc_ids)
            document_periods = _financial_route_document_periods(corpus, doc_id_set)

            def _route_hit_eligible(
                hit: dict[str, Any], doc_id: str, group: str | None = None
            ) -> bool:
                if _doc_key(hit) != doc_id:
                    return False
                metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
                if metadata_filters and not _metadata_matches(metadata, metadata_filters):
                    return False
                if not _has_financial_number(str(hit.get("text") or "")):
                    return False
                if not _financial_route_hit_matches_period(
                    hit,
                    periods,
                    inherited_periods=document_periods.get(doc_id, set()),
                ):
                    return False
                if not _metadata_is_table(metadata):
                    narrative_allowed = bool(
                        profile["section_terms"]
                        or asks_full_and_summary
                        or profile["is_cause"]
                    )
                    if not narrative_allowed:
                        return False
                    if profile["section_terms"] and not _financial_route_hit_matches_section(
                        hit, profile["section_terms"]
                    ):
                        return False
                    if profile["is_cause"] and not _CAUSE_EVIDENCE_RE.search(
                        re.sub(r"\s+", "", str(hit.get("text") or ""))
                    ):
                        return False
                return group is None or _financial_route_hit_matches_group(hit, group)

            def _raw_narrative_preservation_eligible(
                hit: dict[str, Any],
            ) -> bool:
                """Keep a few valid raw narratives before route injections."""
                if _metadata_is_table(
                    hit.get("metadata")
                    if isinstance(hit.get("metadata"), dict)
                    else {}
                ):
                    return False
                doc_id = _doc_key(hit)
                if doc_id not in doc_id_set:
                    return False
                metadata = (
                    hit.get("metadata")
                    if isinstance(hit.get("metadata"), dict)
                    else {}
                )
                if metadata_filters and not _metadata_matches(metadata, metadata_filters):
                    return False
                if not _has_financial_number(str(hit.get("text") or "")):
                    return False
                if not _financial_route_hit_matches_period(
                    hit,
                    periods,
                    inherited_periods=document_periods.get(doc_id, set()),
                ):
                    return False
                return any(
                    _financial_route_hit_matches_group(hit, group)
                    for group, _route_query in route_queries
                )

            if _tracing:
                route["document_periods"] = {
                    doc_id: sorted(document_periods.get(doc_id, set()))
                    for doc_id in doc_ids[:20]
                }
            sections = sorted(
                {
                    str((item.get("metadata") or {}).get("section_path"))
                    for item in corpus
                    if _doc_key(item) in doc_id_set
                    and str((item.get("metadata") or {}).get("section_path") or "")
                }
            )
            if _tracing:
                route["sections"] = [item[:80] for item in sections[:20]]
            if not sections and not profile["section_terms"]:
                route["outcome"] = "skipped_no_fin_sections"
                return ordered
            # section_path varies between report pagination chunks. Keep an
            # auditable section filter, but derive it only from locked doc_ids;
            # the per-metric query carries the requested section terms.
            clauses: list[dict[str, Any]] = [
                {"is_table": True},
                {
                    "report_period": (
                        periods[0] if len(periods) == 1 else {"$in": list(periods)}
                    )
                },
                {"section_path": {"$in": sections}},
                {"doc_id": {"$in": doc_ids}},
            ]
            if metadata_filters:
                clauses.append(dict(metadata_filters))
            route["filters"] = {
                "is_table": True,
                "report_period": list(periods),
                "section_count": len(sections),
                "doc_id_count": len(doc_ids),
            }
            structured: list[dict[str, Any]] = []
            matched_groups: set[str] = set()
            structured_hits = self._search_one(
                query,
                top_k=_FIN_ROUTE_TOP_N,
                kb_id=kb_id,
                metadata_filters={"$and": clauses},
            )
            structured_found = len(structured_hits)
            for hit in structured_hits:
                if _doc_key(hit) not in doc_id_set:
                    continue
                matched_hit = False
                for group, _route_query in route_queries:
                    if _financial_route_hit_matches_group(hit, group):
                        matched_groups.add(group)
                        matched_hit = True
                if matched_hit:
                    structured.append(hit)
            route["structured_found"] = structured_found
            unique_structured: dict[tuple[Any, ...], dict[str, Any]] = {}
            for hit in structured:
                unique_structured.setdefault(_candidate_key(hit), hit)
            structured = list(unique_structured.values())
            structured_doc_ids = {
                _doc_key(hit) for hit in structured if _doc_key(hit) in set(doc_ids)
            }
            asks_full_and_summary = "全文" in query and "摘要" in query
            missing_doc_ids = [
                doc_id for doc_id in doc_ids if doc_id not in structured_doc_ids
            ]
            missing_groups = [
                group for group, _ in route_queries if group not in matched_groups
            ]
            if structured and missing_groups:
                # 先用当前缺失指标的专门表达补一次严格通道；只有仍未覆盖时，
                # 才进入按 doc_id 隔离的宽兜底。这样多指标问题不会因首轮命中
                # 一个字段就停止寻找另一个字段。
                query_by_group = dict(route_queries)
                for group in missing_groups:
                    group_hits = self._search_one(
                        query_by_group[group],
                        top_k=_FIN_ROUTE_TOP_N,
                        kb_id=kb_id,
                        metadata_filters={"$and": clauses},
                    )
                    structured_found += len(group_hits)
                    for hit in group_hits:
                        if (
                            _doc_key(hit) in doc_id_set
                            and _financial_route_hit_matches_group(hit, group)
                        ):
                            structured.append(hit)
                            matched_groups.add(group)
                route["structured_found"] = structured_found
                unique_structured = {
                    _candidate_key(hit): hit for hit in structured
                }
                structured = list(unique_structured.values())
                missing_groups = [
                    group for group, _ in route_queries if group not in matched_groups
                ]
            # Keep the strict-route gaps.  A broad fallback may find an
            # unrelated block for the same metric, but that must not suppress
            # the bounded same-document supplementation below.
            strict_missing_groups = tuple(missing_groups)
            if _tracing:
                route["missing_metric_groups"] = missing_groups
            fallback_requests: list[tuple[str | None, str]] = []
            if not structured:
                fallback_targets = doc_ids
                fallback_groups: list[str | None] = [None]
            else:
                fallback_targets = missing_doc_ids if asks_full_and_summary else []
                fallback_groups = [None]
            for group in fallback_groups:
                for doc_id in fallback_targets:
                    fallback_requests.append((group, doc_id))
            if missing_groups and structured and not asks_full_and_summary:
                for group in missing_groups:
                    for doc_id in doc_ids:
                        request = (group, doc_id)
                        if request not in fallback_requests:
                            fallback_requests.append(request)
            if profile["section_terms"] and not fallback_requests:
                # A named financial section also warrants one bounded,
                # locked-document pass so narrative evidence is eligible even
                # when the table route already covered the metric group.
                fallback_requests.extend((None, doc_id) for doc_id in doc_ids)
            fallback_used = False
            fallback_candidates: list[dict[str, Any]] = []
            # Even when the strict route already covered every requested
            # metric, keep one bounded same-document pass.  A strict hit can
            # come from an unrelated table carrying the same metric label;
            # enumerating the locked document lets the canonical "主要会计
            # 数据/主要财务指标" block compete for injection as well.
            if fallback_requests or structured:
                # 兜底仍按 doc_id 分开查询；即使底层返回越界结果，也在本地再次锁边界。
                query_by_group = dict(route_queries)
                for group, doc_id in fallback_requests:
                    search_query = query if group is None else query_by_group[group]
                    fallback_clauses: list[dict[str, Any]] = [
                        {"doc_id": {"$in": [doc_id]}},
                    ]
                    if metadata_filters:
                        fallback_clauses.append(dict(metadata_filters))
                    fallback_hits = self._search_one(
                        search_query,
                        top_k=_FIN_ROUTE_FALLBACK_TOP_K,
                        kb_id=kb_id,
                        metadata_filters={"$and": fallback_clauses},
                    )
                    for hit in fallback_hits:
                        if not _route_hit_eligible(hit, doc_id, group):
                            continue
                        matching_groups = [
                            candidate_group
                            for candidate_group, _ in route_queries
                            if _financial_route_hit_matches_group(hit, candidate_group)
                        ]
                        if group is not None:
                            matching_groups = [
                                candidate_group
                                for candidate_group in matching_groups
                                if candidate_group == group
                            ]
                        if not matching_groups:
                            continue
                        matched_groups.update(matching_groups)
                        fallback_candidates.append(hit)
                attempted_requests = set(fallback_requests)
                remaining_groups = [
                    group for group, _ in route_queries if group not in matched_groups
                ]
                for group in remaining_groups:
                    for doc_id in doc_ids:
                        request = (group, doc_id)
                        if request in attempted_requests:
                            continue
                        attempted_requests.add(request)
                        fallback_clauses = [
                            {"doc_id": {"$in": [doc_id]}},
                        ]
                        if metadata_filters:
                            fallback_clauses.append(dict(metadata_filters))
                        fallback_hits = self._search_one(
                            query_by_group[group],
                            top_k=_FIN_ROUTE_FALLBACK_TOP_K,
                            kb_id=kb_id,
                            metadata_filters={"$and": fallback_clauses},
                        )
                        for hit in fallback_hits:
                            if _route_hit_eligible(hit, doc_id, group):
                                matched_groups.add(group)
                                fallback_candidates.append(hit)
                missing_groups = [
                    group for group, _ in route_queries if group not in matched_groups
                ]

                def _bounded_document_candidates(
                    group: str, *, narrative_only: bool
                ) -> list[dict[str, Any]]:
                    candidates: list[tuple[int, dict[str, Any]]] = []
                    for index, item in enumerate(corpus):
                        item_doc_id = _doc_key(item)
                        if item_doc_id not in doc_id_set or not _route_hit_eligible(
                            item, item_doc_id, group
                        ):
                            continue
                        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                        if narrative_only != (not _metadata_is_table(metadata)):
                            continue
                        if (
                            narrative_only
                            and profile["section_terms"]
                            and not _financial_route_hit_matches_section(
                                item, profile["section_terms"]
                            )
                        ):
                            continue
                        candidates.append((index, item))

                    aliases = _FINANCIAL_QUERY_GROUPS[group]

                    def _candidate_sort_key(
                        pair: tuple[int, dict[str, Any]]
                    ) -> tuple[Any, ...]:
                        index, item = pair
                        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
                        stored = _stored_financial_metrics(metadata)
                        compact = re.sub(r"\s+", "", str(item.get("text") or ""))
                        fact_match = int(bool(stored & aliases))
                        header_match = int(any(alias in compact for alias in aliases))
                        preferred_section_terms = tuple(
                            dict.fromkeys(
                                (*_FIN_KEY_SECTIONS, *profile["section_terms"])
                            )
                        )
                        section_match = sum(
                            term in str(metadata.get("section_path") or "")
                            for term in preferred_section_terms
                        )
                        return (
                            -fact_match,
                            -header_match,
                            -section_match,
                            -_candidate_rank_adjustment(item, profile),
                            index,
                            _candidate_key(item),
                        )

                    limit = (
                        _FIN_ROUTE_NARRATIVE_MAX
                        if narrative_only
                        else _FIN_ROUTE_ENUM_MAX
                    )
                    return [
                        dict(item)
                        for _index, item in sorted(candidates, key=_candidate_sort_key)[
                            :limit
                        ]
                    ]

                # The strict route defines which metric groups are genuinely
                # uncovered.  A fallback hit cannot turn that into an
                # unbounded raw-candidate append.
                supplement_groups = tuple(
                    dict.fromkeys((*strict_missing_groups, *missing_groups))
                )
                if structured and not supplement_groups:
                    # The strict route found the requested group(s), but that
                    # is not proof that it found the canonical financial block.
                    # Enumerate each already-covered group within the locked
                    # document/period boundary, subject to the existing cap.
                    supplement_groups = tuple(group for group, _ in route_queries)
                enumerated_candidates: list[dict[str, Any]] = []
                for group in supplement_groups:
                    enumerated_candidates.extend(
                        _bounded_document_candidates(group, narrative_only=False)
                    )
                # Narrative evidence is eligible for a named section or a
                # cause question; it still needs the same locked doc, period,
                # metric, number, and cause-expression gates as table evidence.
                if profile["section_terms"] or profile["is_cause"]:
                    for group, _route_query in route_queries:
                        enumerated_candidates.extend(
                            _bounded_document_candidates(group, narrative_only=True)
                        )
                structured.extend(enumerated_candidates)
                if _tracing:
                    route["enumerated_count"] = len(enumerated_candidates)
                fallback_used = bool(fallback_candidates)
                if _tracing:
                    route["fallback_doc_ids"] = sorted(
                        {doc_id for _, doc_id in fallback_requests}
                    )[:20]
                    route["fallback_found"] = len(fallback_candidates)
            structured.extend(fallback_candidates)
            unique_structured = {
                _candidate_key(hit): hit for hit in structured
            }
            structured = list(unique_structured.values())
            if not structured:
                route["outcome"] = "skipped_no_structured_hits"
                return ordered
            # 同一张会计数据表常被切成多个块，只有部分切片含目标指标行（例如
            # "营业收入" 那一列）。先按"块里出现了几个问句指标词"重排，再按
            # 原向量名次稳定兜底，避免把不含目标行的切片排在最前面。
            metric_terms = _FIN_METRIC_HINT_RE.findall(query)

            def _metric_hits(hit: dict[str, Any]) -> int:
                if not metric_terms:
                    return 0
                text = str(hit.get("text") or "")
                return sum(1 for term in metric_terms if term in text)

            # Keep the canonical financial sections ahead of an equally
            # metric-matching block from an unrelated section.  This priority
            # is applied after metric matching and before the stable insertion
            # order, so it also covers strict hits that precede enumerated
            # same-document candidates.
            preferred_section_terms = tuple(
                dict.fromkeys((*_FIN_KEY_SECTIONS, *profile["section_terms"]))
            )

            def _section_priority(hit: dict[str, Any]) -> int:
                metadata = (
                    hit.get("metadata")
                    if isinstance(hit.get("metadata"), dict)
                    else {}
                )
                section_path = str(metadata.get("section_path") or "")
                return sum(term in section_path for term in preferred_section_terms)

            structured = sorted(
                enumerate(structured),
                key=lambda pair: (
                    -_cause_candidate_priority(pair[1], profile, route_queries),
                    -_metric_hits(pair[1]),
                    -_section_priority(pair[1]),
                    pair[0],
                ),
            )
            structured = [hit for _, hit in structured]
            ceiling = max((score for score, _, _ in ordered), default=0.0) + 1.0
            known = {_candidate_key(hit) for _, _, hit in ordered}
            injected: list[tuple[float, int, dict[str, Any]]] = []
            dropped: list[dict[str, Any]] = []
            for offset, hit in enumerate(structured):
                key = _candidate_key(hit)
                if key in known:
                    if _tracing:
                        dropped.append({"offset": offset, "reason": "already_in_normal_recall"})
                    continue
                # 丢弃不含任何财务数字的切片（表头/公司名称/单位说明）
                if not _has_financial_number(str(hit.get("text") or "")):
                    if _tracing:
                        dropped.append({"offset": offset, "reason": "no_financial_number"})
                    continue
                known.add(key)
                # 递减极小量，保证注入块之间仍保持结构化通道内部的顺序
                injected.append((ceiling - offset * 1e-3, -1, dict(hit)))
            if _tracing:
                route["dropped"] = dropped[:20]
                # 只在诊断开启时才压快照：candidate_snapshot 会遍历候选并逐条
                # 裁剪文本，关闭时在热路径上做这些纯属浪费。
                route["structured_candidates"] = diagnostic_trace.candidate_snapshot(
                    structured, "structured_fin_route_raw"
                )
                route["injected_candidates"] = diagnostic_trace.candidate_snapshot(
                    [hit for _score, _seq, hit in injected], "structured_fin_route"
                )
            route["injected_count"] = len(injected)
            if injected and fallback_used:
                route["outcome"] = "fallback_injected"
            else:
                route["outcome"] = "injected" if injected else "no_injection_after_filter"
            preserved_raw: list[tuple[float, int, dict[str, Any]]] = []
            if injected:
                for item in ordered:
                    if len(preserved_raw) >= _FIN_ROUTE_RAW_NARRATIVE_MAX:
                        break
                    if _raw_narrative_preservation_eligible(item[2]):
                        preserved_raw.append(item)
                if _tracing:
                    route["preserved_raw_count"] = len(preserved_raw)
                    route["preserved_raw_candidates"] = [
                        str(hit.get("chunk_id") or "")
                        for _score, _sequence, hit in preserved_raw
                    ]
            logger.info(
                "财务结构化路由命中（period=%s, docs=%d, 注入=%d）",
                period,
                len(doc_ids),
                len(injected),
            )
            if not preserved_raw:
                return injected + ordered
            preserved_keys = {_candidate_key(hit) for _, _, hit in preserved_raw}
            remaining = [
                item for item in ordered if _candidate_key(item[2]) not in preserved_keys
            ]
            return preserved_raw + injected + remaining
        except Exception:  # noqa: BLE001 —— 通道失败一律降级，不影响主检索
            route["outcome"] = "exception_degraded"
            logger.warning("财务结构化路由失败，已降级为常规检索结果", exc_info=True)
            return ordered
        finally:
            _emit_route()


class VectorOnlyRetriever:
    """纯向量检索（对照组）：接口与 HybridRetriever 一致，用于对比测试。"""

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        embeddings: EmbeddingClient,
        top_k: int = 5,
        min_score: float = 0.0,
        max_candidates_per_doc: int | None = None,
        max_per_doc: int | None = None,
    ):
        self._store = vector_store
        self._embeddings = embeddings
        self._top_k = top_k
        self._min_score = min_score
        self._max_per_doc = (
            max_per_doc if max_per_doc is not None else max_candidates_per_doc
        )

    def _search_one(
        self,
        query: str,
        *,
        top_k: int,
        kb_id: str,
        metadata_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        qvec = self._embeddings.embed_query(query)
        hits = self._store.search(
            qvec, top_k=top_k, where=metadata_filters, kb_id=kb_id
        )
        if self._min_score > 0:
            hits = [h for h in hits if h.get("score", 0.0) >= self._min_score]
        return hits

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        kb_id: str = "default",
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.search_queries(
            [query], top_k=top_k, kb_id=kb_id, metadata_filters=metadata_filters
        )

    def search_queries(
        self,
        queries: list[str] | tuple[str, ...],
        *,
        top_k: int | None = None,
        kb_id: str = "default",
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        k = self._top_k if top_k is None else max(int(top_k), 0)
        if k <= 0:
            return []
        unique_queries: list[str] = []
        seen_queries: set[str] = set()
        for query in queries:
            normalized = str(query or "").strip()
            if normalized and normalized not in seen_queries:
                unique_queries.append(normalized)
                seen_queries.add(normalized)
        if not unique_queries:
            return []

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for query in unique_queries:
            for hit in self._search_one(
                query,
                top_k=k,
                kb_id=kb_id,
                metadata_filters=metadata_filters,
            ):
                key = _candidate_key(hit)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(hit)
        return _dedupe_and_diversify(merged, self._max_per_doc)[:k]
