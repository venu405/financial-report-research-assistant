"""Replay audited historical-behavior fixtures through current verification code.

The historical evaluate_quality reports do not contain citation text or metadata,
so they cannot be replayed safely. The fixtures below are explicit, reviewed
inputs copied from the fixed-context tests; they do not use a report's ``passed``
field as the expected result and do not access a model, service, or vector DB.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))
if str(BACKEND / "src") not in sys.path:
    sys.path.insert(0, str(BACKEND / "src"))

from services.kb.qa_graph import (  # noqa: E402
    _answer_verification_reasons,
    _verify_numeric_claims,
)


ALLOWED_KB_ID = "cninfo_report"
HISTORICAL_REPLAY_FIXTURES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "real_huawei_2024_revenue",
        "source_basis": "test_rag_fixed_context_huawei_jinzhou.py:huawei_fixed_context",
        "question": "吉林华微电子2024年营业收入是多少？",
        "answer": "2024年营业收入为2,057,608,183.78元。[1]",
        "evidence": [
            {
                "chunk_id": "huawei-revenue-fixed",
                "text": "主要会计数据\n营业收入 2,057,608,183.78",
                "metadata": {
                    "kb_id": ALLOWED_KB_ID,
                    "doc_id": "ST华微-吉林华微电子股份有限公司2024年年报.pdf",
                    "source_type": "cninfo_report",
                    "page": 6,
                    "chunk_index": 0,
                    "report_period": "2024年度",
                    "financial_facts_json": json.dumps(
                        [
                            {
                                "metric": "revenue",
                                "raw_value": "2,057,608,183.78",
                                "canonical_value": "2,057,608,183.78",
                                "unit": "元",
                                "report_period": "2024年度",
                                "statement_scope": "consolidated",
                            }
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            }
        ],
    },
    {
        "case_id": "real_jinzhou_port_2025_plan_not_promise",
        "source_basis": "test_rag_fixed_context_huawei_jinzhou.py:jinzhou_rerank_context",
        "question": "锦州港2024年年报披露的2025年经营计划中，计划营业收入是多少？该计划是否构成绩效承诺？",
        "answer": "计划营业收入为17.21亿元；上述经营计划并不构成公司对投资者的业绩承诺。[1][2]",
        "evidence": [
            {
                "chunk_id": "jinzhou-plan-fixed",
                "text": "公司拟定2025年度计划实现营业收入17.21亿元",
                "metadata": {
                    "kb_id": ALLOWED_KB_ID,
                    "doc_id": "ST锦港-锦州港股份有限公司2024年年度报告.pdf",
                    "page": 25,
                    "report_period": "2024年度",
                },
            },
            {
                "chunk_id": "jinzhou-disclaimer-fixed",
                "text": "上述经营计划并不构成公司对投资者的业绩承诺",
                "metadata": {
                    "kb_id": ALLOWED_KB_ID,
                    "doc_id": "ST锦港-锦州港股份有限公司2024年年度报告.pdf",
                    "page": 25,
                    "report_period": "2024年度",
                },
            },
        ],
    },
    {
        "case_id": "real_longyu_2024_revenue",
        "source_basis": "test_rag_fixed_context_shenlian_longyu.py:longyu_exact_revenue",
        "question": "上海龙宇数据2024年营业收入是多少？",
        "answer": "2024年营业收入为1,404,920,973.02元。[1]",
        "evidence": [
            {
                "chunk_id": "longyu-revenue-fixed",
                "text": "主要会计数据\n营业收入 1,404,920,973.02",
                "metadata": {
                    "kb_id": ALLOWED_KB_ID,
                    "doc_id": "longyu-2024-annual-report",
                    "page": 7,
                    "report_period": "2024年度",
                    "financial_metrics": "营业收入",
                    "financial_facts_json": json.dumps(
                        [
                            {
                                "metric": "营业收入",
                                "raw_value": "1,404,920,973.02",
                                "canonical_value": "1404920973.02",
                                "unit": "元",
                                "report_period": "2024年度",
                                "statement_scope": "consolidated",
                            }
                        ],
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    },
    {
        "case_id": "real_shenlian_2025_h1_operating_cash",
        "source_basis": "test_rag_fixed_context_shenlian_longyu.py:shenlian_cash_complete",
        "question": "申联生物2025年上半年经营活动现金净流量是多少？",
        "answer": "申联生物2025年上半年经营活动产生的现金流量净额为-39,389,053.48元。[1]",
        "evidence": [
            {
                "chunk_id": "shenlian-cash-fixed",
                "text": "主要会计数据\n经营活动产生的现金流量净额 -39,389,053.48",
                "metadata": {
                    "kb_id": ALLOWED_KB_ID,
                    "doc_id": "shenlian-2025-h1-report",
                    "page": 7,
                    "report_period": "2025年半年度",
                    "financial_metrics": "经营活动产生的现金流量净额",
                    "financial_facts_json": json.dumps(
                        [
                            {
                                "metric": "经营活动产生的现金流量净额",
                                "raw_value": "-39,389,053.48",
                                "canonical_value": "-39389053.48",
                                "unit": "元",
                                "report_period": "2025年半年度",
                                "statement_scope": "consolidated",
                            }
                        ],
                        ensure_ascii=False,
                    ),
                },
            }
        ],
    },
)


def replay_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    """Replay one fixture and return an auditable, case-specific result."""
    case_id = str(fixture.get("case_id") or "<missing-case-id>")
    evidence = fixture.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return {"case_id": case_id, "ok": False, "category": "invalid_fixture"}
    for item in evidence:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        if not isinstance(metadata, dict):
            return {"case_id": case_id, "ok": False, "category": "invalid_metadata"}
        foreign = metadata.get("kb_id") != ALLOWED_KB_ID
        if foreign:
            return {"case_id": case_id, "ok": False, "category": "cross_kb"}

    supported, unsupported = _verify_numeric_claims(
        str(fixture["answer"]), evidence, question=str(fixture["question"])
    )
    if not supported:
        return {
            "case_id": case_id,
            "ok": False,
            "category": "numeric_verification",
            "unsupported_count": len(unsupported),
        }
    reasons = _answer_verification_reasons(
        str(fixture["question"]), str(fixture["answer"]), evidence
    )
    if reasons:
        return {
            "case_id": case_id,
            "ok": False,
            "category": "answer_verification_reasons",
            "reasons": reasons,
        }
    return {"case_id": case_id, "ok": True}


def main() -> int:
    failures = 0
    for fixture in HISTORICAL_REPLAY_FIXTURES:
        result = replay_fixture(fixture)
        if result["ok"]:
            print(f"REPLAY {result['case_id']} PASS")
        else:
            failures += 1
            print(
                f"REPLAY {result['case_id']} FAIL "
                f"category={result.get('category', 'unknown')}"
            )
    print(
        "REPLAY coverage=4; skipped=other historical-report cases "
        "(reports lack evidence text/metadata; not counted as passed)"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
