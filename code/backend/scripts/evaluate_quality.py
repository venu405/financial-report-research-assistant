#!/usr/bin/env python3
"""RAG 与 OCR 可重复评测脚本（v2：检索层指标 + 可机器比较的报告）。

RAG 测试调用既有 /kb/ask；OCR 测试直接调用本地 RapidOCR。

检索层指标（v2 新增，避免"用答案关键词冒充召回提升"）：
  题集可选真值字段：
    expected_sources:  期望命中的文档 doc_id、doc_title 或 source_path
                      （字符串或 {doc_id/doc_title/source_path}）
    expected_pages:    期望命中的页码（整数，或 {doc_id|doc_title, page}）
    expected_evidence: 期望出现在检索文本中的证据片段（子串）
  指标（对 contexts=重排后 与 recall_raw=重排前 两个候选池分别计算）：
    hit@K        分母 = 有 expected_sources 真值的案例数；top-K 中出现任一期望文档的比例
    recall@K     分母同上；top-K 中命中的期望文档数 / 期望文档总数（按 doc 去重）
    evidence_rate 分母 = 有 expected_evidence 真值的案例数；命中的证据片段比例
    page.hit_rate 分母 = 有页码真值 **且** 候选携带页码元数据的案例数。
                  候选无页码元数据时记 not_scoreable_no_metadata（不可计分，不是失败）
  状态区分（每案例 retrieval.status）：
    not_scoreable     无任何真值 -> 不进检索分母
    scored            有真值且检索到候选 -> 进分母
    retrieval_failure 有真值但两个候选池都为空 -> 进分母并按未命中计
    error             请求本身失败（服务/网络错误）-> 不进检索分母（基础设施故障
                      不等于召回差），计入 summary.error_count

报告包含 schema_version/generated_at/label/testset/base_url（脱敏）与样本数，
JSON 结构稳定，可由 scripts/compare_quality_reports.py 做 A/B 比较。
报告不记录 user_id、请求头或任何访问令牌。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.request
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import yaml
from pypdf import PdfReader

from services.kb.ingest import _ocr_pdf_page

SCHEMA_VERSION = 2
DEFAULT_HIT_KS = (1, 3, 5)
# 样本量低于该值时，汇总里标记 small_sample（比较工具据此提示无统计显著性）
SMALL_SAMPLE = 10
_POOL_KEYS = ("contexts", "recall_raw")

_EVAL_NUMBER_RE = re.compile(
    r"(?<!\d)[+-]?(?:\d{1,3}(?:[, \t]\d{3})+|\d+)(?:\.\d+)?"
    r"[ \t]*(万亿|千亿|百亿|十亿|亿元|万元|千万|百万|十万|亿|万|千|百|元|%|％)?"
)
_EVAL_UNIT_SCALE: dict[str, Decimal] = {
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
    "十亿": Decimal("1000000000"),
    "百亿": Decimal("10000000000"),
    "千亿": Decimal("100000000000"),
    "万亿": Decimal("1000000000000"),
    "%": Decimal("1"),
    "％": Decimal("1"),
}
_DECLINE_WORDS = ("减少", "下降", "下滑", "降低", "减幅", "同比降", "同比减")
_GROWTH_WORDS = ("增长", "增加", "上升", "上扬", "提升")


# ==================== 真值归一化 ====================


def normalize_expected_sources(entries: Any) -> list[dict[str, str]]:
    """归一来源真值；字符串仍同时兼容 doc_id、doc_title 旧语义。"""
    normalized: list[dict[str, str]] = []
    for entry in entries or []:
        if isinstance(entry, str):
            normalized.append({"doc_id": entry, "doc_title": entry})
        elif isinstance(entry, dict):
            item = {
                "doc_id": str(entry.get("doc_id") or ""),
                "doc_title": str(entry.get("doc_title") or ""),
            }
            if entry.get("source_path"):
                item["source_path"] = str(entry["source_path"])
            normalized.append(item)
    return [e for e in normalized if e["doc_id"] or e["doc_title"]]


def _canonical_source(value: str) -> str:
    """把路径或标题归一为可比较的文件名；doc_id 仍在外层走精确匹配。"""
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    normalized = normalized.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1].casefold()


_SOURCE_SUFFIXES = frozenset(
    {
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
        ".md",
        ".csv",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
    }
)


def _source_path_keys(value: Any) -> set[str]:
    """生成路径键，兼容项目根前缀但不把任意目录降级为同名文件。"""
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip()
    normalized = normalized.replace("\\", "/").rstrip("/").casefold()
    if not normalized:
        return set()
    keys = {normalized}
    marker = "data_kb_test/"
    marker_index = normalized.find(marker)
    if marker_index >= 0:
        keys.add(normalized[marker_index:])
    return keys


def _source_name_keys(value: Any) -> set[str]:
    """生成文件名和已知文档扩展名 stem，避免任意后缀被静默剥离。"""
    filename = _canonical_source(str(value or ""))
    if not filename:
        return set()
    keys = {filename}
    for suffix in _SOURCE_SUFFIXES:
        if filename.endswith(suffix) and len(filename) > len(suffix):
            keys.add(filename[: -len(suffix)])
            break
    return keys


def _canonical_text(value: str) -> str:
    """证据匹配忽略 PDF 抽取产生的空白差异。"""
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _numeric_tokens(value: Any) -> list[dict[str, Any]]:
    """提取可用于纯函数比较的 Decimal 数值及展示单位。"""
    text = unicodedata.normalize("NFKC", str(value or ""))
    tokens: list[dict[str, Any]] = []
    for match in _EVAL_NUMBER_RE.finditer(text):
        raw = match.group(0).strip()
        unit = (match.group(1) or "").strip()
        number_text = re.sub(
            r"[,\s]", "", raw[: match.group(0).rfind(unit)] if unit else raw
        )
        try:
            number = Decimal(number_text)
        except (InvalidOperation, ValueError):
            continue
        tokens.append(
            {
                "raw": raw,
                "number": number,
                "unit": unit,
                "canonical": number * _EVAL_UNIT_SCALE[unit],
                "start": match.start(),
                "end": match.end(),
            }
        )
    return tokens


def _keyword_text(value: Any) -> str:
    """归一全角字符和可忽略的千分位/空白，不改变数值本身。"""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s,，]", "", normalized)


_COMPARISON_BASIS_RE = re.compile(r"相较于上年同期|较上年同期|比上年同期|同比")
_PERFORMANCE_COMMITMENT_CLAIM_RE = re.compile(
    r"构成(?:(?:公司|对投资者的)*)业绩承诺"
)
_PERFORMANCE_COMMITMENT_UNCERTAIN_MARKERS = (
    "可能",
    "是否",
    "不确定",
    "无法判断",
    "不能确定",
    "难以判断",
    "不能说",
)


def _comparison_keyword_text(value: Any) -> str:
    """仅消除比较基准措辞，保留指标、方向和数值。"""
    return _COMPARISON_BASIS_RE.sub("", _keyword_text(value))


def _performance_commitment_polarity(value: Any) -> str | None:
    """识别窄义的“构成……业绩承诺”结论及其确定性。"""
    text = _keyword_text(value)
    if "业绩承诺" not in text:
        return None

    negative = False
    positive = False
    for match in _PERFORMANCE_COMMITMENT_CLAIM_RE.finditer(text):
        if text[: match.start()].endswith("不"):
            negative = True
        else:
            positive = True

    if not (negative or positive):
        return None
    if positive or any(marker in text for marker in _PERFORMANCE_COMMITMENT_UNCERTAIN_MARKERS):
        return "other"
    return "negative" if negative else "other"


def _performance_commitment_negation_matches(expected: Any, answer: Any) -> bool | None:
    """只在双方都是明确“不构成……业绩承诺”时允许省略限定语。"""
    expected_polarity = _performance_commitment_polarity(expected)
    answer_polarity = _performance_commitment_polarity(answer)
    if expected_polarity is None or answer_polarity is None:
        return None
    return expected_polarity == answer_polarity == "negative"


def _direction_conflicts(expected: str, answer: str) -> bool:
    """比较期措辞可变，但增长与下降方向不可互换。"""
    expected_text = unicodedata.normalize("NFKC", expected)
    answer_text = unicodedata.normalize("NFKC", answer)
    expected_decline = any(word in expected_text for word in _DECLINE_WORDS)
    expected_growth = any(word in expected_text for word in _GROWTH_WORDS)
    answer_decline = any(word in answer_text for word in _DECLINE_WORDS)
    answer_growth = any(word in answer_text for word in _GROWTH_WORDS)
    return (expected_decline and answer_growth and not answer_decline) or (
        expected_growth and answer_decline and not answer_growth
    )


def _decline_near(answer: str, token: dict[str, Any]) -> bool:
    text = unicodedata.normalize("NFKC", str(answer or ""))
    start = max(0, int(token["start"]) - 24)
    end = min(len(text), int(token["end"]) + 24)
    return any(word in text[start:end] for word in _DECLINE_WORDS)


def _numeric_token_equal(
    expected: dict[str, Any], actual: dict[str, Any], *, allow_rounded_units: bool = True
) -> bool:
    expected_unit = expected["unit"]
    actual_unit = actual["unit"]
    if expected_unit in ("%", "％") or actual_unit in ("%", "％"):
        if expected_unit in ("%", "％") and actual_unit in ("%", "％"):
            return expected["number"] == actual["number"]
        # 期望关键词未声明单位、答案补写了百分号：答案比关键词更明确（信息只增不减），
        # 按展示数字相等判定。典型场景是占比类问题：题集写 expect_keyword "22.32"，
        # 标准答案与模型答案都写 "22.32%"，此前会被 canonical（÷100）判为不等，
        # 导致连题集自带的标准答案都过不了自己的关键词检查。
        # 反向（期望带 %、答案省略 %）属于丢失单位信息，仍然拒绝。
        if not expected_unit and actual_unit in ("%", "％"):
            return expected["number"] == actual["number"]
        return False
    if expected_unit == actual_unit and expected["number"] == actual["number"]:
        return True
    if expected["canonical"] == actual["canonical"]:
        return True
    if not allow_rounded_units:
        return False

    # 题集有时使用有限小数位的亿元/万元展示值，而答案给出更精确的元值。
    # 只允许 expected 使用更粗的单位、且 actual 的绝对分辨率确实更细时，
    # 将 actual 换算到 expected 单位后按 expected 的小数位四舍五入比较。
    expected_scale = _EVAL_UNIT_SCALE[expected_unit]
    actual_scale = _EVAL_UNIT_SCALE[actual_unit]
    expected_places = max(0, -expected["number"].as_tuple().exponent)
    expected_quantum = expected_scale.scaleb(-expected_places)
    actual_quantum = actual_scale.scaleb(
        -max(0, -actual["number"].as_tuple().exponent)
    )
    if expected_scale <= actual_scale or actual_quantum >= expected_quantum:
        return False

    expected_display_quantum = Decimal("1").scaleb(-expected_places)
    actual_in_expected_unit = actual["canonical"] / expected_scale
    return (
        actual_in_expected_unit.quantize(
            expected_display_quantum, rounding=ROUND_HALF_UP
        )
        == expected["number"].quantize(
            expected_display_quantum, rounding=ROUND_HALF_UP
        )
    )


def numeric_semantically_matches(expected: Any, answer: Any) -> bool:
    """判断一个期望片段是否出现在答案中，兼容精确单位换算和下降语义。

    这是评测层纯函数：不把“约/大约”的近似值当成精确值；只有明确下降
    语义才把 ``-22.55`` 与“同比减少 22.55%”视为同一数值语义。
    """
    expected_text = str(expected or "")
    answer_text = str(answer or "")
    expected_normalized = unicodedata.normalize("NFKC", expected_text).strip()
    numeric_only = bool(
        re.fullmatch(
            r"[+-]?(?:\d{1,3}(?:[, \t]\d{3})+|\d+)(?:\.\d+)?"
            r"[ \t]*(?:万亿|千亿|百亿|十亿|亿元|万元|千万|百万|十万|亿|万|千|百|元|%|％)?",
            expected_normalized,
        )
    )
    commitment_match = _performance_commitment_negation_matches(
        expected_text, answer_text
    )
    if commitment_match is not None:
        return commitment_match
    if not numeric_only and expected_text in answer_text:
        return True
    if not numeric_only and _keyword_text(expected_text) in _keyword_text(answer_text):
        return True
    if not numeric_only and _direction_conflicts(expected_text, answer_text):
        return False
    if not numeric_only:
        comparison_expected = _comparison_keyword_text(expected_text)
        comparison_answer = _comparison_keyword_text(answer_text)
        if comparison_expected and comparison_expected in comparison_answer:
            return True

    expected_tokens = _numeric_tokens(expected_text)
    answer_tokens = _numeric_tokens(answer_text)
    if not expected_tokens or not answer_tokens:
        return False
    approximation_markers = ("约", "大约")
    allow_rounded_units = not any(
        marker in expected_text or marker in answer_text
        for marker in approximation_markers
    )
    for expected_token in expected_tokens:
        matched = False
        for actual_token in answer_tokens:
            if _numeric_token_equal(
                expected_token,
                actual_token,
                allow_rounded_units=allow_rounded_units,
            ):
                matched = True
                break
            # 测试集用 -22.55 表示下降幅度，而自然语言答案常写成
            # “同比减少22.55%”；必须同时看到下降词，避免增长被误判。
            if (
                expected_token["number"] == -actual_token["number"]
                and expected_token["number"] < 0
                and _decline_near(answer_text, actual_token)
            ):
                matched = True
                break
        if not matched:
            return False
    return True


def semantic_keyword_matches(expected: Any, answer: Any) -> bool:
    """expect_keyword 的兼容匹配：旧字符串优先，数字再做精确语义比较。"""
    return numeric_semantically_matches(expected, answer)


# 便于调用方/单元测试使用的短别名；保持实现只有一份。
_numeric_semantically_matches = numeric_semantically_matches
_semantic_keyword_matches = semantic_keyword_matches


def normalize_expected_pages(entries: Any) -> list[dict[str, Any]]:
    """把 expected_pages 归一为 [{page, doc_id?, doc_title?}]。"""
    normalized: list[dict[str, Any]] = []
    for entry in entries or []:
        if isinstance(entry, bool):
            continue
        if isinstance(entry, int):
            normalized.append({"page": entry, "doc_id": "", "doc_title": ""})
        elif isinstance(entry, dict) and entry.get("page") is not None:
            normalized.append(
                {
                    "page": int(entry["page"]),
                    "doc_id": str(entry.get("doc_id") or ""),
                    "doc_title": str(entry.get("doc_title") or ""),
                }
            )
    return normalized


def _normalize_evidence(entries: Any) -> list[str]:
    return [str(entry) for entry in entries or [] if str(entry)]


def build_case_truth(case: dict[str, Any]) -> dict[str, Any]:
    """从测试案例提取检索真值（缺字段一律为空，旧题集继续可用）。"""
    return {
        "sources": normalize_expected_sources(case.get("expected_sources")),
        "pages": normalize_expected_pages(case.get("expected_pages")),
        "evidence": _normalize_evidence(case.get("expected_evidence")),
    }


# ==================== 候选命中归一化 ====================


def _hit_meta(hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit.get("metadata")
    return meta if isinstance(meta, dict) else {}


def hit_doc_fields(hit: dict[str, Any]) -> tuple[str, str]:
    """稳健提取 (doc_id, doc_title)：兼容 contexts 的 metadata 嵌套与 citations 的平铺。"""
    meta = _hit_meta(hit)
    doc_id = str(hit.get("doc_id") or meta.get("doc_id") or "")
    doc_title = str(hit.get("doc_title") or meta.get("doc_title") or "")
    return doc_id, doc_title


def hit_source_path(hit: dict[str, Any]) -> str:
    """提取新 Qdrant payload 的 source_path，兼容平铺和 metadata 嵌套。"""
    meta = _hit_meta(hit)
    return str(hit.get("source_path") or meta.get("source_path") or "")


def hit_page(hit: dict[str, Any]) -> int | None:
    """稳健提取页码：page / metadata.page / metadata.page_start。"""
    for value in (hit.get("page"), _hit_meta(hit).get("page"), _hit_meta(hit).get("page_start")):
        if value is None or isinstance(value, bool):
            continue
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page > 0:
            return page
    return None


def hit_text(hit: dict[str, Any]) -> str:
    return str(hit.get("text") or "")


def _hit_dedupe_key(hit: dict[str, Any]) -> str:
    chunk_id = str(hit.get("chunk_id") or "")
    if chunk_id:
        return f"chunk:{chunk_id}"
    doc_id, _title = hit_doc_fields(hit)
    chunk_index = _hit_meta(hit).get("chunk_index", hit.get("chunk_index"))
    if doc_id and chunk_index is not None:
        return f"idx:{doc_id}:{chunk_index}"
    return f"text:{hash(hit_text(hit))}"


def dedupe_hits(hits: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """保序去重：同一 chunk 在多查询融合/引用回填中可能重复出现。"""
    seen: set[str] = set()
    ranked: list[dict[str, Any]] = []
    for hit in hits or []:
        if not isinstance(hit, dict):
            continue
        key = _hit_dedupe_key(hit)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(hit)
    return ranked


def _source_matches(hit: dict[str, Any], expected: dict[str, str]) -> bool:
    doc_id, doc_title = hit_doc_fields(hit)
    if expected["doc_id"] and doc_id == expected["doc_id"]:
        return True

    actual_path = hit_source_path(hit)
    expected_paths = [
        value
        for value in (expected.get("source_path"),)
        if value
    ]
    if expected.get("doc_title") and (
        "/" in expected["doc_title"] or "\\" in expected["doc_title"]
    ):
        expected_paths.append(expected["doc_title"])
    if actual_path and expected_paths:
        # Qdrant 已提供路径时优先按路径判断；路径存在但不一致不能再
        # 用同名 doc_title 掩盖目录错配。data_kb_test 前缀允许项目根不同。
        return any(
            _source_path_keys(expected_path) & _source_path_keys(actual_path)
            for expected_path in expected_paths
        )

    expected_names = set()
    for value in (expected.get("source_path"), expected.get("doc_title")):
        expected_names.update(_source_name_keys(value))
    return bool(expected_names & _source_name_keys(doc_title))


# ==================== 检索指标（纯函数） ====================


def evaluate_pool(
    truth: dict[str, Any], hits: list[dict[str, Any]] | None, hit_ks: tuple[int, ...]
) -> dict[str, Any]:
    """对一个候选池计算单案例检索指标。无真值的维度值为 None（不进分母）。"""
    ranked = dedupe_hits(hits)
    record: dict[str, Any] = {}

    if truth["sources"]:
        expected = truth["sources"]
        for k in hit_ks:
            top = ranked[:k]
            record[f"hit@{k}"] = any(
                _source_matches(hit, entry) for hit in top for entry in expected
            )
            found = sum(
                1 for entry in expected if any(_source_matches(hit, entry) for hit in top)
            )
            record[f"recall@{k}"] = round(found / len(expected), 4)
    else:
        for k in hit_ks:
            record[f"hit@{k}"] = None
            record[f"recall@{k}"] = None

    if truth["evidence"]:
        texts = [_canonical_text(hit_text(hit)) for hit in ranked]
        found = sum(
            1
            for evidence in truth["evidence"]
            if any(_canonical_text(evidence) in text for text in texts)
        )
        record["evidence_rate"] = round(found / len(truth["evidence"]), 4)
    else:
        record["evidence_rate"] = None

    if truth["pages"]:
        has_page_meta = any(hit_page(hit) is not None for hit in ranked)
        if not has_page_meta:
            # 候选不携带页码元数据（如旧链路未写 page）：不可计分，不是检索失败
            record["page_hit"] = None
            record["page_recall"] = None
            record["page_no_metadata"] = True
        else:

            def _page_ok(entry: dict[str, Any]) -> bool:
                for hit in ranked:
                    if hit_page(hit) != entry["page"]:
                        continue
                    doc_id, doc_title = hit_doc_fields(hit)
                    if entry["doc_id"] and doc_id != entry["doc_id"]:
                        continue
                    if entry["doc_title"] and doc_title != entry["doc_title"]:
                        continue
                    return True
                return False

            outcomes = [_page_ok(entry) for entry in truth["pages"]]
            record["page_hit"] = any(outcomes)
            record["page_recall"] = round(sum(outcomes) / len(outcomes), 4)
            record["page_no_metadata"] = False
    else:
        record["page_hit"] = None
        record["page_recall"] = None
        record["page_no_metadata"] = False
    return record


def _truth_counts(truth: dict[str, Any]) -> dict[str, int]:
    return {
        "sources": len(truth["sources"]),
        "pages": len(truth["pages"]),
        "evidence": len(truth["evidence"]),
    }


def evaluate_case_retrieval(
    truth: dict[str, Any], response: dict[str, Any] | None, hit_ks: tuple[int, ...]
) -> dict[str, Any]:
    """单案例检索评估。区分 error / not_scoreable / retrieval_failure / scored。"""
    if response is None or not isinstance(response, dict):
        return {
            "status": "error",
            "expected": _truth_counts(truth),
            "contexts": None,
            "recall_raw": None,
        }
    meta = response.get("search_meta")
    meta = meta if isinstance(meta, dict) else {}
    contexts = response.get("contexts")
    if not isinstance(contexts, list) or not contexts:
        # contexts 缺失时退回 citations（平铺结构，dedupe/字段提取均兼容）
        citations = response.get("citations")
        contexts = citations if isinstance(citations, list) else []
    recall_raw = meta.get("recall_raw")
    recall_raw = recall_raw if isinstance(recall_raw, list) else []

    has_truth = bool(truth["sources"] or truth["pages"] or truth["evidence"])
    counts = _truth_counts(truth)
    if not has_truth:
        return {
            "status": "not_scoreable",
            "expected": counts,
            "contexts": None,
            "recall_raw": None,
        }
    status = "scored" if (contexts or recall_raw) else "retrieval_failure"
    return {
        "status": status,
        "expected": counts,
        "contexts": evaluate_pool(truth, contexts, hit_ks),
        "recall_raw": evaluate_pool(truth, recall_raw, hit_ks),
    }


def _mean(values: list[Any]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate_pool(
    records: list[dict[str, Any]], pool: str, hit_ks: tuple[int, ...]
) -> dict[str, Any]:
    """把一组案例的 retrieval 记录按池聚合。每个指标自带分母（None 值不进分母）。"""
    eligible = [
        record[pool]
        for record in records
        if record.get(pool) is not None
        and record.get("status") in ("scored", "retrieval_failure")
    ]
    agg: dict[str, Any] = {}
    for k in hit_ks:
        hits = [item[f"hit@{k}"] for item in eligible if item.get(f"hit@{k}") is not None]
        recalls = [
            item[f"recall@{k}"] for item in eligible if item.get(f"recall@{k}") is not None
        ]
        agg[f"hit@{k}"] = _mean(hits)
        agg[f"hit@{k}_denominator"] = len(hits)
        agg[f"recall@{k}"] = _mean(recalls)
        agg[f"recall@{k}_denominator"] = len(recalls)
    evidence = [
        item["evidence_rate"] for item in eligible if item.get("evidence_rate") is not None
    ]
    agg["evidence_rate"] = _mean(evidence)
    agg["evidence_rate_denominator"] = len(evidence)
    page_hits = [item["page_hit"] for item in eligible if item.get("page_hit") is not None]
    agg["page"] = {
        "hit_rate": _mean(page_hits),
        "denominator": len(page_hits),
        "not_scoreable_no_metadata": sum(
            1 for item in eligible if item.get("page_no_metadata")
        ),
    }
    return agg


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    # nearest-rank：p50([1,2,3,4])=2，p95=4；比 Python 的 bankers rounding
    # 更适合延迟门槛，也不会在偶数样本上跳到上中位数。
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return round(ordered[index], 3)


def build_summary(results: list[dict[str, Any]], hit_ks: tuple[int, ...]) -> dict[str, Any]:
    """全局汇总：答案层 + 可回答性 + 引用 + 延迟 + 检索层（含分母与样本量提示）。"""
    ok = [item for item in results if "error" not in item]
    total = len(results)
    passed = sum(1 for item in results if item.get("passed"))
    answerable = [
        item for item in ok if "answerable" in (item.get("checks") or {})
    ]
    escalates = [bool(item["escalate"]) for item in ok if "escalate" in item]
    citations = [int(item.get("citations") or 0) for item in ok]
    latency = [
        float(item["elapsed_s"]) for item in ok if item.get("elapsed_s") is not None
    ]
    retrieval_records = [item.get("retrieval") or {} for item in results]
    summary = {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "error_count": total - len(ok),
        "small_sample": total < SMALL_SAMPLE,
        "answerable": {
            "denominator": len(answerable),
            "accuracy": _mean([bool(item["checks"]["answerable"]) for item in answerable]),
        },
        "escalate_rate": _mean(escalates),
        "avg_citations": _mean(citations),
        "latency": {
            "count": len(latency),
            "p50_s": _percentile(latency, 0.5),
            "p95_s": _percentile(latency, 0.95),
            "max_s": round(max(latency), 3) if latency else None,
            "avg_s": round(sum(latency) / len(latency), 3) if latency else None,
        },
        "retrieval": {
            pool: aggregate_pool(retrieval_records, pool, hit_ks) for pool in _POOL_KEYS
        },
    }
    return summary


def _aggregate_group(items: list[dict[str, Any]], hit_ks: tuple[int, ...]) -> dict[str, Any]:
    """按 category/tag 切片聚合：保留 total/passed（旧格式兼容）+ pass_rate + 检索层。"""
    total = len(items)
    passed = sum(1 for item in items if item.get("passed"))
    records = [item.get("retrieval") or {} for item in items]
    return {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else None,
        "retrieval": {
            pool: aggregate_pool(records, pool, hit_ks) for pool in _POOL_KEYS
        },
    }


def sanitize_base_url(url: str) -> str:
    """只保留 scheme://host[:port]，剥离 userinfo/query 中可能存在的凭据。"""
    try:
        parsed = urlsplit(url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return f"{parsed.scheme}://{host}{port}"
    except ValueError:
        return ""


# ==================== 运行层 ====================


def ask(
    base_url: str,
    kb_id: str,
    question: str,
    user_id: str | None,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"question": question, "kb_id": kb_id}
    if user_id:
        payload["user_id"] = user_id
    if history:
        payload["history"] = history
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/kb/ask",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _build_rag_report(
    path: Path,
    base_url: str,
    label: str,
    hit_ks: tuple[int, ...],
    case_ids: set[str] | None,
    results: list[dict[str, Any]],
    *,
    planned_total: int,
    run_status: str,
) -> dict[str, Any]:
    """Build the stable report shape from the cases completed so far."""
    by_category: dict[str, dict[str, Any]] = {}
    by_tags: dict[str, dict[str, Any]] = {}
    category_items: dict[str, list[dict[str, Any]]] = {}
    tag_items: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        category_items.setdefault(item["category"], []).append(item)
        for tag in item.get("tags") or []:
            tag_items.setdefault(str(tag), []).append(item)
    for category, items in category_items.items():
        by_category[category] = _aggregate_group(items, hit_ks)
    for tag, items in tag_items.items():
        by_tags[tag] = _aggregate_group(items, hit_ks)

    return {
        "kind": "rag",
        "schema_version": SCHEMA_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "label": label,
        "testset": str(path),
        "base_url": sanitize_base_url(base_url),
        "hit_ks": list(hit_ks),
        "selected_case_ids": sorted(case_ids) if case_ids else None,
        "total": len(results),
        "planned_total": planned_total,
        "completed_total": len(results),
        "run_status": run_status,
        "passed": sum(x["passed"] for x in results),
        "summary": build_summary(results, hit_ks),
        "by_category": by_category,
        "by_tags": by_tags,
        "results": results,
    }


def run_rag(
    path: Path,
    base_url: str,
    user_id: str | None,
    *,
    label: str = "",
    hit_ks: tuple[int, ...] = DEFAULT_HIT_KS,
    case_ids: set[str] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    cases = list(cfg.get("tests", []))
    if case_ids:
        cases = [case for case in cases if str(case.get("id")) in case_ids]
        found = {str(case.get("id")) for case in cases}
        missing = sorted(case_ids - found)
        if missing:
            raise ValueError(f"题集不存在以下 case id：{', '.join(missing)}")
    planned_total = len(cases)
    if progress_callback is not None:
        try:
            progress_callback(
                _build_rag_report(
                    path,
                    base_url,
                    label,
                    hit_ks,
                    case_ids,
                    results,
                    planned_total=planned_total,
                    run_status="in_progress",
                )
            )
        except Exception as exc:
            print(f"progress report write failed: {exc}", file=sys.stderr)
    for case in cases:
        started = time.monotonic()
        kb_id = case.get("kb_id", cfg.get("kb_id", "default"))
        try:
            response = ask(base_url, kb_id, case["question"], user_id, case.get("history"))
            elapsed_s = round(time.monotonic() - started, 2)
            meta = response.get("search_meta", {})
            answer = response.get("answer", "")
            citations = response.get("citations", [])
            escalate = bool(response.get("escalate", meta.get("escalate", False)))
            needs_clarification = bool(meta.get("needs_clarification", False))
            inline_refs = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
            citation_indices = [c.get("index") for c in citations]
            citation_kb_ids = [
                (c.get("metadata") or {}).get("kb_id") or c.get("kb_id") for c in citations
            ]

            checks: dict[str, bool] = {
                "response_shape": isinstance(answer, str)
                and isinstance(citations, list)
                and isinstance(meta, dict),
                "citation_indices": not citations
                or citation_indices == list(range(1, len(citations) + 1)),
                "citation_kb_isolation": all(item == kb_id for item in citation_kb_ids),
                "no_citations_on_escalate": not escalate or not citations,
            }
            if "expect_answerable" in case:
                checks["answerable"] = (not escalate) == case["expect_answerable"]
            if "expect_escalate" in case:
                checks["escalate"] = escalate == case["expect_escalate"]
            if "expect_intent" in case:
                checks["intent"] = meta.get("intent") == case["expect_intent"]
            if "expect_needs_clarification" in case:
                checks["needs_clarification"] = needs_clarification == case["expect_needs_clarification"]
            if "expect_keyword" in case:
                checks["keyword"] = semantic_keyword_matches(case["expect_keyword"], answer)
            if "expect_keywords_all" in case:
                checks["keywords_all"] = all(
                    semantic_keyword_matches(word, answer)
                    for word in case["expect_keywords_all"]
                )
            if "expect_keywords_any" in case:
                checks["keywords_any"] = any(
                    semantic_keyword_matches(word, answer)
                    for word in case["expect_keywords_any"]
                )
            if "expect_forbidden_keywords" in case:
                checks["forbidden_keywords"] = all(
                    word not in answer for word in case["expect_forbidden_keywords"]
                )
            if "expect_min_citations" in case:
                checks["min_citations"] = len(citations) >= case["expect_min_citations"]
            if "expect_max_citations" in case:
                checks["max_citations"] = len(citations) <= case["expect_max_citations"]
            if case.get("expect_inline_citations"):
                checks["inline_citations"] = bool(inline_refs) and all(
                    1 <= ref <= len(citations) for ref in inline_refs
                )
            if "expect_top_score_min" in case:
                checks["top_score"] = float(meta.get("top_score") or 0) >= float(case["expect_top_score_min"])
            if "expect_score_min" in case:
                checks["faithfulness"] = float(response.get("score") or 0) >= float(case["expect_score_min"])
            if "max_elapsed_s" in case:
                checks["latency"] = elapsed_s <= float(case["max_elapsed_s"])

            truth = build_case_truth(case)
            results.append({
                "id": case.get("id"), "category": case.get("category", "uncategorized"),
                "tags": list(case.get("tags") or []),
                "kb_id": kb_id, "question": case["question"], "passed": all(checks.values()),
                "checks": checks, "elapsed_s": elapsed_s, "intent": meta.get("intent"),
                "top_score": meta.get("top_score"), "faithfulness": response.get("score"),
                "citations": len(citations), "citation_kb_ids": citation_kb_ids,
                "escalate": escalate, "needs_clarification": needs_clarification,
                "retrieval": evaluate_case_retrieval(truth, response, hit_ks),
                "answer": answer,
            })
        except Exception as exc:
            truth = build_case_truth(case)
            results.append({
                "id": case.get("id"), "category": case.get("category", "uncategorized"),
                "tags": list(case.get("tags") or []),
                "kb_id": kb_id, "question": case["question"], "passed": False,
                "retrieval": evaluate_case_retrieval(truth, None, hit_ks),
                "error": str(exc),
            })
        if progress_callback is not None:
            try:
                progress_callback(
                    _build_rag_report(
                        path,
                        base_url,
                        label,
                        hit_ks,
                        case_ids,
                        results,
                        planned_total=planned_total,
                        run_status="in_progress",
                    )
                )
            except Exception as exc:
                # A reporting sink must not turn a completed request into a failed case.
                print(f"progress report write failed: {exc}", file=sys.stderr)

    return _build_rag_report(
        path,
        base_url,
        label,
        hit_ks,
        case_ids,
        results,
        planned_total=planned_total,
        run_status="completed",
    )


def run_ocr(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = path.parent.parent
    results: list[dict[str, Any]] = []
    for sample in cfg.get("samples", []):
        pdf = root / sample["path"]
        page_index = int(sample.get("page", 0))
        native = PdfReader(str(pdf)).pages[page_index].extract_text() or ""
        ocr = _ocr_pdf_page(str(pdf), page_index, ocr_mode=sample.get("ocr_mode", "local"))
        missing = [term for term in sample.get("expect_terms", []) if term not in ocr]
        passed = len(ocr.strip()) >= int(sample.get("min_chars", 1)) and not missing
        results.append({"path": sample["path"], "page": page_index + 1, "passed": passed,
                        "native_chars": len(native.strip()), "ocr_chars": len(ocr.strip()), "missing_terms": missing})
    return {"kind": "ocr", "testset": str(path), "total": len(results), "passed": sum(x["passed"] for x in results), "results": results}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> bool:
    """Write a complete JSON document without destroying the previous report."""
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        print(f"report write failed: {exc}", file=sys.stderr)
        try:
            temp_path.unlink()
        except OSError:
            pass
        return False


def _with_rag_status(report: dict[str, Any], status: str) -> dict[str, Any]:
    updated = dict(report)
    updated["run_status"] = status
    return updated


def _parse_hit_ks(raw: str) -> tuple[int, ...]:
    ks: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise argparse.ArgumentTypeError(f"hit K 必须 >= 1：{value}")
        ks.add(value)
    return tuple(sorted(ks))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "RAG/OCR 评测。RAG 报告含检索层指标（hit@K/recall@K/evidence/page，"
            "仅在题集提供 expected_sources/expected_pages/expected_evidence 真值时计分），"
            "可用 scripts/compare_quality_reports.py 比较两份报告。"
            "退出码：0=全部通过；1=存在失败案例；2=参数错误。"
        )
    )
    parser.add_argument("--rag", type=Path, help="RAG YAML 测试集")
    parser.add_argument("--ocr", type=Path, help="OCR YAML 测试集")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default=None, help="鉴权用户 ID（不写入报告）")
    parser.add_argument("--label", default="", help="配置标签（如 rag-v2-hybrid），写入报告供 A/B 比较")
    parser.add_argument(
        "--hit-ks", type=_parse_hit_ks, default=DEFAULT_HIT_KS, metavar="K1,K2",
        help="检索命中率的 K 值列表（默认 1,3,5）",
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="只运行指定案例 ID（可重复）；用于同语料、同题目的受控 A/B",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reports: list[dict[str, Any]] = []
    partial_rag: dict[str, Any] | None = None
    write_failures = 0

    def reports_with_write_diagnostics(
        snapshot_reports: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not write_failures:
            return snapshot_reports
        return [
            {
                **report,
                "report_write_failures": write_failures,
            }
            if report.get("kind") == "rag"
            else report
            for report in snapshot_reports
        ]

    def write_snapshot(
        snapshot_reports: list[dict[str, Any]],
        *,
        final: bool = False,
    ) -> bool:
        nonlocal write_failures
        if not args.output:
            return True
        payload = {"reports": reports_with_write_diagnostics(snapshot_reports)}
        if _write_json_atomic(args.output, payload):
            return True
        write_failures += 1
        phase = "final report" if final else "progress snapshot"
        print(
            f"{phase} write failed (attempt {write_failures}); "
            "completed requests were retained",
            file=sys.stderr,
        )
        return False

    def on_rag_progress(partial: dict[str, Any]) -> None:
        nonlocal partial_rag
        partial_rag = partial
        write_snapshot([*reports, partial])

    try:
        final_write_ok = True
        if args.rag:
            rag_report = run_rag(
                args.rag,
                args.base_url,
                args.user_id,
                label=args.label,
                hit_ks=tuple(args.hit_ks) if args.hit_ks else DEFAULT_HIT_KS,
                case_ids=set(args.case_id) or None,
                progress_callback=on_rag_progress,
            )
            reports.append(rag_report)
            partial_rag = None
            final_write_ok = write_snapshot(reports, final=True) and final_write_ok
        if args.ocr:
            reports.append(run_ocr(args.ocr))
            final_write_ok = write_snapshot(reports, final=True) and final_write_ok
    except KeyboardInterrupt:
        if partial_rag is not None:
            partial_report = _with_rag_status(partial_rag, "interrupted")
            write_snapshot([*reports, partial_report], final=True)
            report_reports = [*reports, partial_report]
        else:
            report_reports = reports
        report = {"reports": reports_with_write_diagnostics(report_reports)}
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        print("evaluation interrupted; partial report preserved", file=sys.stderr)
        return 130
    except Exception as exc:
        if partial_rag is not None:
            partial_report = _with_rag_status(partial_rag, "failed")
            write_snapshot([*reports, partial_report], final=True)
            report_reports = [*reports, partial_report]
        else:
            report_reports = reports
        report = {"reports": reports_with_write_diagnostics(report_reports)}
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        print(f"evaluation failed: {exc}", file=sys.stderr)
        return 1
    if not reports:
        parser.error("至少提供 --rag 或 --ocr")
    report = {"reports": reports_with_write_diagnostics(reports)}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.output and not final_write_ok:
        print("final report could not be written; returning non-zero", file=sys.stderr)
        return 1
    return 0 if all(r["passed"] == r["total"] for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
