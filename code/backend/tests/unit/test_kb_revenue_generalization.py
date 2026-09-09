"""收入字段核验的通用回归：保留明确行绑定，同时局部遮蔽未支持附加数字。"""

from __future__ import annotations

import json

import pytest

from services.kb.qa_graph import (
    _answer_verification_reasons,
    _partial_numeric_fallback_answer,
    _verify_numeric_claims,
)


def _record(
    text: str,
    *,
    report_period: str = "2024年度",
    unit: str = "万元",
    statement_scope: str = "unknown",
) -> dict:
    return {
        "text": text,
        "metadata": {
            "is_table": True,
            "financial_metrics": "revenue",
            "report_period": report_period,
            "unit": unit,
            "statement_scope": statement_scope,
        },
    }


@pytest.mark.parametrize(
    ("company", "question", "evidence_text"),
    [
        (
            "甲星科技",
            "甲星科技2024年营业收入是多少？",
            "甲星科技2024年度主要会计数据：营业收入 100万元",
        ),
        (
            "乙辰电子",
            "乙辰电子2024年年报主要会计数据中的营业收入为多少？",
            "乙辰电子2024年度主要会计数据\n营业收入\n100万元",
        ),
    ],
)
def test_revenue_only_metadata_supports_same_and_cross_line_answers(
    company: str, question: str, evidence_text: str
):
    evidence = [_record(evidence_text)]
    answer = f"{company}2024年营业收入为100万元，同时同比增长12.34%。"

    supported, unsupported = _verify_numeric_claims(
        answer, evidence, question=question
    )

    assert supported is False
    assert [claim.raw for claim in unsupported] == ["12.34%"]

    safe_answer = _partial_numeric_fallback_answer(question, answer, evidence)
    assert safe_answer is not None
    assert "营业收入为100万元" in safe_answer
    assert "12.34%" not in safe_answer
    assert _answer_verification_reasons(question, safe_answer, evidence) == []


def test_revenue_only_metadata_without_any_row_label_remains_a_fallback():
    evidence = [_record("甲星科技2024年度财务数据：100万元")]

    assert _verify_numeric_claims(
        "营业收入为100万元。",
        evidence,
        question="甲星科技2024年营业收入是多少？",
    )[0] is True


def test_principal_business_revenue_cannot_satisfy_plain_revenue():
    evidence = [_record("甲星科技2024年度主要会计数据\n主营业务收入\n100万元")]

    supported, unsupported = _verify_numeric_claims(
        "营业收入为100万元。",
        evidence,
        question="甲星科技2024年营业收入是多少？",
    )

    assert supported is False
    assert [claim.raw for claim in unsupported] == ["100万元"]


@pytest.mark.parametrize(
    ("metadata", "question", "answer"),
    [
        (
            {"report_period": "2023年度", "unit": "万元", "statement_scope": "unknown"},
            "甲星科技2024年营业收入是多少？",
            "营业收入为100万元。",
        ),
        (
            {"report_period": "2024年度", "unit": "万元", "statement_scope": "unknown"},
            "甲星科技2024年营业收入是多少？",
            "营业收入为100元。",
        ),
        (
            {"report_period": "2024年度", "unit": "万元", "statement_scope": "parent"},
            "甲星科技2024年合并口径营业收入是多少？",
            "合并口径营业收入为100万元。",
        ),
    ],
)
def test_revenue_period_unit_and_scope_mismatches_still_reject(
    metadata: dict, question: str, answer: str
):
    record_metadata = {
        "is_table": True,
        "financial_metrics": "revenue",
        **metadata,
    }
    evidence = [
        {
            "text": f"甲星科技{record_metadata['report_period']}\n营业收入\n100万元",
            "metadata": record_metadata,
        }
    ]

    assert _verify_numeric_claims(answer, evidence, question=question)[0] is False


def test_incomplete_financial_facts_do_not_override_explicit_revenue_row():
    evidence = [
        {
            "text": "乙辰电子2024年度主要会计数据\n营业收入\n100万元",
            "metadata": {
                "is_table": True,
                "financial_metrics": "revenue",
                "financial_facts_json": json.dumps(
                    [
                        {
                            "metric": "revenue",
                            "raw_value": "100",
                            "canonical_value": "1000000",
                            "report_period": "",
                            "statement_scope": "",
                            "unit": "万元",
                        }
                    ],
                    ensure_ascii=False,
                ),
                "report_period": "2024年度",
                "unit": "万元",
                "statement_scope": "unknown",
            },
        }
    ]

    assert _verify_numeric_claims(
        "营业收入为100万元。",
        evidence,
        question="乙辰电子2024年年报主要会计数据中的营业收入为多少？",
    )[0] is True
