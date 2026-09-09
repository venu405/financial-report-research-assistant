"""安全核验升级后，财务主字段不应被附带数字或缺证据误伤。"""

from __future__ import annotations

import json

import pytest

from services.kb.qa_graph import (
    _answer_verification_reasons,
    _has_reliable_question_evidence,
    _partial_numeric_fallback_answer,
    _relocate_numeric_citations,
    _single_field_candidate_supports_explicit_claim,
    _verify_numeric_claims,
)

_CASES = (
    {
        "company": "澄星智造股份有限公司",
        "metric": "revenue",
        "label": "营业收入",
        "answer_label": "营业收入",
        "value": "123.45",
        "question_form": "营业收入是多少？",
    },
    {
        "company": "霁川电子有限公司",
        "metric": "net_profit_attributable",
        "label": "归属于上市公司股东的净利润",
        "answer_label": "归母净利润",
        "value": "-45.67",
        "question_form": "年报主要会计数据中的归母净利润为多少？",
    },
)


def _fact(case: dict[str, str], *, period: str = "2025年度") -> dict[str, str]:
    value = case["value"]
    canonical = str(float(value) * 10000)
    return {
        "metric": case["metric"],
        "raw_value": f"{value}万元",
        "canonical_value": canonical,
        "unit": "万元",
        "report_period": period,
        "statement_scope": "consolidated",
    }


def _record(
    case: dict[str, str],
    *,
    period: str = "2025年度",
    evidence_label: str | None = None,
    fact_metric: str | None = None,
    fact_value: str | None = None,
) -> dict[str, object]:
    label = evidence_label or case["label"]
    value = fact_value if fact_value is not None else case["value"]
    fact = _fact(case, period=period)
    if fact_metric is not None:
        fact["metric"] = fact_metric
    if fact_value is not None:
        fact["raw_value"] = f"{fact_value}万元"
        fact["canonical_value"] = str(float(fact_value) * 10000)
    text = f"{case['company']}{period}主要会计数据\n{label}\n{value}万元"
    return {
        "text": text,
        "metadata": {
            "doc_title": f"{case['company']}{period}年度报告",
            "is_table": True,
            "financial_metrics": fact["metric"],
            "financial_facts_json": json.dumps([fact], ensure_ascii=False),
            "report_period": period,
            "unit": "万元",
            "statement_scope": "consolidated",
        },
    }


def _scoped_record(
    case: dict[str, str],
    *,
    value: str,
    doc_id: str = "fictional-report",
    kb_id: str = "fictional-kb",
) -> dict[str, object]:
    record = _record(case, fact_value=value)
    metadata = record["metadata"]
    assert isinstance(metadata, dict)
    metadata.update({"doc_id": doc_id, "kb_id": kb_id})
    return record


@pytest.mark.parametrize("case", _CASES, ids=lambda item: item["metric"])
def test_supported_primary_financial_field_is_preserved(case: dict[str, str]):
    question = f"{case['company']}2025年{case['question_form']}"
    answer = f"{case['company']}2025年{case['answer_label']}为{case['value']}万元。[1]"
    evidence = [_record(case)]

    supported, unsupported = _verify_numeric_claims(
        answer, evidence, question=question
    )

    assert supported is True
    assert unsupported == []
    assert _answer_verification_reasons(question, answer, evidence) == []


@pytest.mark.parametrize("case", _CASES, ids=lambda item: item["metric"])
def test_unsupported_secondary_percentage_is_locally_degraded(
    case: dict[str, str],
):
    question = f"{case['company']}2025年{case['question_form']}"
    answer = (
        f"{case['company']}2025年{case['answer_label']}为{case['value']}万元，"
        "同比增长12.5%。[1]"
    )
    evidence = [_record(case)]

    supported, unsupported = _verify_numeric_claims(
        answer, evidence, question=question
    )

    assert supported is False
    assert [claim.raw for claim in unsupported] == ["12.5%"]

    safe_answer = _partial_numeric_fallback_answer(question, answer, evidence)

    assert safe_answer is not None
    assert f"{case['answer_label']}为{case['value']}万元" in safe_answer
    assert "12.5%" not in safe_answer
    assert _answer_verification_reasons(question, safe_answer, evidence) == []


@pytest.mark.parametrize(
    ("case_index", "mismatch"),
    [
        (0, "subject"),
        (0, "period"),
        (0, "metric"),
        (1, "value"),
    ],
)
def test_subject_period_metric_or_value_mismatch_keeps_refusal_boundary(
    case_index: int, mismatch: str
):
    case = _CASES[case_index]
    other = _CASES[1 - case_index]
    question = f"{case['company']}2025年{case['question_form']}"
    answer = f"{case['company']}2025年{case['answer_label']}为{case['value']}万元。"

    if mismatch == "subject":
        evidence = [_record(other, evidence_label=case["label"], fact_metric=case["metric"])]
    elif mismatch == "period":
        evidence = [_record(case, period="2024年度")]
    elif mismatch == "metric":
        evidence = [
            _record(
                case,
                evidence_label=other["label"],
                fact_metric=other["metric"],
            )
        ]
    else:
        evidence = [_record(case, fact_value="999.99")]

    contexts = [
        {
            "text": evidence[0]["text"],
            "metadata": evidence[0]["metadata"],
        }
    ]
    reliable = _has_reliable_question_evidence(
        question, contexts, [str(evidence[0]["text"])]
    )
    numerically_supported, _ = _verify_numeric_claims(
        answer, evidence, question=question
    )

    assert not (reliable and numerically_supported), mismatch
    assert _partial_numeric_fallback_answer(question, answer, evidence) is None


def test_absent_evidence_cannot_become_answerable_via_partial_fallback():
    case = _CASES[0]
    question = f"{case['company']}2025年{case['question_form']}"
    answer = f"{case['company']}2025年{case['answer_label']}为{case['value']}万元。"

    assert _has_reliable_question_evidence(question, [], []) is False
    assert _verify_numeric_claims(answer, [], question=question)[0] is False
    assert _partial_numeric_fallback_answer(question, answer, []) is None


@pytest.mark.parametrize("case", _CASES, ids=lambda item: item["metric"])
def test_relocates_single_field_citation_then_partially_degrades_extra_percentage(
    case: dict[str, str],
):
    question = f"{case['company']}2025年{case['question_form']}"
    answer = (
        f"{case['company']}2025年{case['answer_label']}为{case['value']}万元，"
        "同比增长12.5%。[1]"
    )
    current = _scoped_record(case, value="11.11")
    target = _scoped_record(case, value=case["value"])
    contexts = [current, target]
    passages = [str(current["text"]), str(target["text"])]

    assert _single_field_candidate_supports_explicit_claim(
        answer, question, [target]
    ) is True
    relocated = _relocate_numeric_citations(
        answer, question, contexts, passages
    )

    assert relocated.endswith("[2]")
    assert "[1]" not in relocated
    safe_answer = _partial_numeric_fallback_answer(
        question, relocated, [target]
    )
    assert safe_answer is not None
    assert f"{case['answer_label']}为{case['value']}万元" in safe_answer
    assert "12.5%" not in safe_answer


@pytest.mark.parametrize("scope_key", ["doc_id", "kb_id"])
def test_relocation_never_crosses_doc_or_kb_scope(scope_key: str):
    case = _CASES[0]
    question = f"{case['company']}2025年{case['question_form']}"
    answer = (
        f"{case['company']}2025年{case['answer_label']}为{case['value']}万元，"
        "同比增长12.5%。[1]"
    )
    current = _scoped_record(case, value="11.11")
    target_scope = {"doc_id": "fictional-report", "kb_id": "fictional-kb"}
    target_scope[scope_key] = f"other-{scope_key}"
    target = _scoped_record(case, value=case["value"], **target_scope)

    relocated = _relocate_numeric_citations(
        answer,
        question,
        [current, target],
        [str(current["text"]), str(target["text"])],
    )

    assert relocated == answer


def test_ambiguous_same_scope_candidates_do_not_relocate():
    case = _CASES[0]
    question = f"{case['company']}2025年{case['question_form']}"
    answer = f"{case['company']}2025年{case['answer_label']}为{case['value']}万元。[1]"
    current = _scoped_record(case, value="11.11")
    target_one = _scoped_record(case, value=case["value"])
    target_two = _scoped_record(case, value=case["value"])

    relocated = _relocate_numeric_citations(
        answer,
        question,
        [current, target_one, target_two],
        [
            str(current["text"]),
            str(target_one["text"]),
            str(target_two["text"]),
        ],
    )

    assert relocated == answer


def test_multi_metric_question_does_not_use_single_field_relocation():
    revenue = _CASES[0]
    profit = _CASES[1]
    question = (
        f"{revenue['company']}2025年营业收入和归母净利润分别是多少？"
    )
    answer = (
        f"{revenue['company']}2025年营业收入为{revenue['value']}万元，"
        f"归母净利润为{profit['value']}万元，同比增长12.5%。[1]"
    )
    current = _scoped_record(revenue, value="11.11")
    target = {
        "text": (
            f"{revenue['company']}2025年度主要会计数据\n"
            f"营业收入\n{revenue['value']}万元\n"
            f"{profit['label']}\n{profit['value']}万元"
        ),
        "metadata": {
            **_scoped_record(revenue, value=revenue["value"])["metadata"],
            "financial_metrics": "revenue|net_profit_attributable",
            "financial_facts_json": json.dumps(
                [_fact(revenue), _fact(profit)], ensure_ascii=False
            ),
        },
    }

    relocated = _relocate_numeric_citations(
        answer,
        question,
        [current, target],
        [str(current["text"]), str(target["text"])],
    )

    assert relocated == answer
