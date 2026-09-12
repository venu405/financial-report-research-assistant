"""申联现金流 / 龙宇收入的固定上下文离线归因单测。

本文件只调用确定性核验、证据上下文和评测数字契约函数；不构造真实
embedding、向量库、LLM 或网络客户端。每个快照把“路由保留的上下文、
预设初始回答、后置核验结果”分开记录，便于对应线上报告的三个阶段。
"""
from __future__ import annotations

import json

from scripts.evaluate_quality import numeric_semantically_matches
from services.kb.qa_graph import (
    _answer_verification_reasons,
    _has_reliable_question_evidence,
    _verification_evidence_items,
    _verify_numeric_claims,
)

SHENLIAN_CASH_QUESTION = "申联生物2025年上半年经营活动现金净流量是多少？"
SHENLIAN_CASH_ANSWER = (
    "申联生物2025年上半年经营活动产生的现金流量净额为-39,389,053.48元。[1]"
)
SHENLIAN_DOC_ID = "shenlian-2025-h1-report"
SHENLIAN_KB_ID = "cninfo_report"

LONGYU_QUESTION = "上海龙宇数据2024年营业收入是多少？"
LONGYU_EXACT_ANSWER = "2024年营业收入为1,404,920,973.02元。[1]"
LONGYU_DISTRACTOR_ANSWER = "2024年主营业务收入为14.03亿元，同比减少55.07%。[1]"
LONGYU_DOC_ID = "longyu-2024-annual-report"


def _facts_json(
    metric: str,
    raw_value: str,
    *,
    canonical_value: str,
    unit: str,
    report_period: str,
) -> str:
    return json.dumps(
        [
            {
                "metric": metric,
                "raw_value": raw_value,
                "canonical_value": canonical_value,
                "unit": unit,
                "statement_scope": "consolidated",
                "report_period": report_period,
            }
        ],
        ensure_ascii=False,
    )


def _hit(
    *,
    chunk_id: str,
    doc_id: str,
    doc_title: str,
    text: str,
    metric: str,
    page: int,
    report_period: str,
    raw_value: str,
    canonical_value: str,
    unit: str = "元",
    is_table: bool = True,
) -> dict:
    metadata = {
        "doc_id": doc_id,
        "kb_id": SHENLIAN_KB_ID,
        "doc_title": doc_title,
        "page": page,
        "financial_metrics": metric,
        "report_period": report_period,
        "unit": unit,
        "statement_scope": "consolidated",
        "is_table": is_table,
        "financial_facts_json": _facts_json(
            metric,
            raw_value,
            canonical_value=canonical_value,
            unit=unit,
            report_period=report_period,
        ),
    }
    return {"chunk_id": chunk_id, "text": text, "metadata": metadata, "score": 0.75}


def _verification_snapshot(question: str, answer: str, contexts: list[dict]) -> dict:
    """固定输入下，显式分开上下文保留、初答和后置核验。"""
    passages = [str(hit["text"]) for hit in contexts]
    final_evidence = _verification_evidence_items(answer, contexts, passages)
    verification, unsupported = _verify_numeric_claims(
        answer, final_evidence, question=question
    )
    reasons = _answer_verification_reasons(question, answer, final_evidence)
    return {
        "final_context": contexts,
        "final_evidence": final_evidence,
        "initial_answer": answer,
        "verification": verification,
        "unsupported": unsupported,
        "reasons": reasons,
        "reliable_evidence": _has_reliable_question_evidence(
            question, contexts, passages
        ),
    }


def test_shenlian_cash_complete_five_field_context_verifies_exact_answer():
    hit = _hit(
        chunk_id="shenlian-cash-page-7",
        doc_id=SHENLIAN_DOC_ID,
        doc_title="申联生物 2025年半年度报告",
        text="主要会计数据\n经营活动产生的现金流量净额 -39,389,053.48",
        metric="经营活动产生的现金流量净额",
        page=7,
        report_period="2025年半年度",
        raw_value="-39,389,053.48",
        canonical_value="-39389053.48",
    )

    snapshot = _verification_snapshot(
        SHENLIAN_CASH_QUESTION, SHENLIAN_CASH_ANSWER, [hit]
    )
    metadata = snapshot["final_context"][0]["metadata"]

    # 路由/最终上下文：五项绑定字段均仍在证据记录中。
    assert metadata["doc_id"] == SHENLIAN_DOC_ID
    assert metadata["page"] == 7
    assert metadata["financial_metrics"] == "经营活动产生的现金流量净额"
    assert "-39,389,053.48" in snapshot["final_evidence"][0]["text"]
    assert metadata["report_period"] == "2025年半年度"
    assert "-39,389,053.48" in "\n".join(
        item["text"] for item in snapshot["final_evidence"]
    )

    # 初始回答与后置核验：固定答案可被同一证据精确支持。
    assert snapshot["initial_answer"] == SHENLIAN_CASH_ANSWER
    assert snapshot["reliable_evidence"] is True
    assert snapshot["verification"] is True
    assert snapshot["unsupported"] == []
    assert snapshot["reasons"] == []


def test_shenlian_cash_missing_report_period_rejects_reliable_gate_but_records_actual_verification():
    hit = _hit(
        chunk_id="shenlian-cash-page-7-no-period",
        doc_id=SHENLIAN_DOC_ID,
        doc_title="申联生物",
        text="经营活动产生的现金流量净额 -39,389,053.48",
        metric="经营活动产生的现金流量净额",
        page=7,
        report_period="",
        raw_value="-39,389,053.48",
        canonical_value="-39389053.48",
    )

    snapshot = _verification_snapshot(
        SHENLIAN_CASH_QUESTION, SHENLIAN_CASH_ANSWER, [hit]
    )

    # 证据仍进入最终上下文，但缺少期间标签后，可靠证据门不能确认题目期间。
    assert snapshot["final_context"][0]["metadata"]["page"] == 7
    assert snapshot["final_context"][0]["metadata"]["report_period"] == ""
    assert "-39,389,053.48" in snapshot["final_evidence"][0]["text"]
    assert snapshot["reliable_evidence"] is False

    # 当前后置数字核验会从同一精确正文行完成核验；这不是把缺期间元数据
    # 判成“检索失败”，而是一个可单独观察的路由/可靠性缺口。
    assert snapshot["initial_answer"] == SHENLIAN_CASH_ANSWER
    assert snapshot["verification"] is True
    assert snapshot["unsupported"] == []
    assert snapshot["reasons"] == []


def test_shenlian_rd_ratio_is_same_document_control_sample():
    hit = _hit(
        chunk_id="shenlian-rd-ratio-page-7",
        doc_id=SHENLIAN_DOC_ID,
        doc_title="申联生物 2025年半年度报告",
        text="主要会计数据\n研发投入占营业收入的比例（%） 22.32",
        metric="研发投入占营业收入的比例",
        page=7,
        report_period="2025年半年度",
        raw_value="22.32",
        canonical_value="22.32",
        unit="%",
    )
    question = "申联生物2025年上半年研发投入占营业收入的比例是多少？"
    answer = "2025年上半年研发投入占营业收入的比例为22.32%。[1]"

    snapshot = _verification_snapshot(question, answer, [hit])
    assert snapshot["final_context"][0]["metadata"]["doc_id"] == SHENLIAN_DOC_ID
    assert snapshot["final_context"][0]["metadata"]["page"] == 7
    assert "22.32" in snapshot["final_evidence"][0]["text"]
    assert snapshot["final_context"][0]["metadata"]["report_period"] == "2025年半年度"
    assert snapshot["reliable_evidence"] is True
    assert snapshot["verification"] is True
    assert snapshot["reasons"] == []


def test_longyu_exact_revenue_field_wins_over_same_context_distractor():
    exact_hit = _hit(
        chunk_id="longyu-revenue-page-7",
        doc_id=LONGYU_DOC_ID,
        doc_title="上海龙宇数据 2024年年度报告",
        text="主要会计数据\n营业收入 1,404,920,973.02",
        metric="营业收入",
        page=7,
        report_period="2024年度",
        raw_value="1,404,920,973.02",
        canonical_value="1404920973.02",
    )
    distractor_hit = _hit(
        chunk_id="longyu-narrative-page-16",
        doc_id=LONGYU_DOC_ID,
        doc_title="上海龙宇数据 2024年年度报告",
        text="公司实现主营业务收入14.03亿元，较去年同期减少55.07%",
        metric="主营业务收入",
        page=16,
        report_period="2024年度",
        raw_value="14.03",
        canonical_value="1403000000",
        unit="亿元",
        is_table=False,
    )

    snapshot = _verification_snapshot(
        LONGYU_QUESTION,
        LONGYU_EXACT_ANSWER,
        [exact_hit, distractor_hit],
    )
    evidence_text = "\n".join(item["text"] for item in snapshot["final_evidence"])

    assert "营业收入 1,404,920,973.02" in evidence_text
    assert "主营业务收入14.03亿元" not in evidence_text
    assert snapshot["initial_answer"] == LONGYU_EXACT_ANSWER
    assert snapshot["verification"] is True
    assert snapshot["unsupported"] == []
    assert snapshot["reasons"] == []
    assert numeric_semantically_matches("1,404,920,973.02", LONGYU_EXACT_ANSWER)
    assert not numeric_semantically_matches("1,404,920,973.02", LONGYU_DISTRACTOR_ANSWER)
    assert not numeric_semantically_matches("1,404,920,973.02", "14.05亿元")


def test_longyu_distractor_only_cannot_fully_pass_field_binding():
    distractor_hit = _hit(
        chunk_id="longyu-narrative-page-16-only",
        doc_id=LONGYU_DOC_ID,
        doc_title="上海龙宇数据 2024年年度报告",
        text="公司实现主营业务收入14.03亿元，较去年同期减少55.07%",
        metric="主营业务收入",
        page=16,
        report_period="2024年度",
        raw_value="14.03",
        canonical_value="1403000000",
        unit="亿元",
        is_table=False,
    )

    snapshot = _verification_snapshot(
        LONGYU_QUESTION, LONGYU_DISTRACTOR_ANSWER, [distractor_hit]
    )

    assert snapshot["reliable_evidence"] is True
    assert snapshot["verification"] is False
    assert snapshot["unsupported"]
    assert snapshot["reasons"] == [
        "问题要求营业收入，答案或证据仅绑定到主营业务收入，不能视为同一字段。"
    ]


def test_longyu_explicit_primary_business_revenue_remains_supported():
    hit = _hit(
        chunk_id="longyu-primary-revenue-page-16",
        doc_id=LONGYU_DOC_ID,
        doc_title="上海龙宇数据 2024年年度报告",
        text="公司实现主营业务收入14.03亿元，较去年同期减少55.07%",
        metric="主营业务收入",
        page=16,
        report_period="2024年度",
        raw_value="14.03",
        canonical_value="1403000000",
        unit="亿元",
        is_table=False,
    )
    question = "上海龙宇数据2024年主营业务收入是多少？"
    answer = "2024年主营业务收入为14.03亿元。[1]"

    snapshot = _verification_snapshot(question, answer, [hit])
    assert snapshot["verification"] is True
    assert snapshot["unsupported"] == []
    assert snapshot["reasons"] == []


def test_longyu_primary_answer_is_rejected_against_exact_revenue_evidence():
    hit = _hit(
        chunk_id="longyu-revenue-page-7-answer-mismatch",
        doc_id=LONGYU_DOC_ID,
        doc_title="上海龙宇数据 2024年年度报告",
        text="主要会计数据\n营业收入 1,404,920,973.02",
        metric="营业收入",
        page=7,
        report_period="2024年度",
        raw_value="1,404,920,973.02",
        canonical_value="1404920973.02",
    )
    answer = "2024年主营业务收入为1,404,920,973.02元。[1]"

    snapshot = _verification_snapshot(LONGYU_QUESTION, answer, [hit])
    assert snapshot["verification"] is False
    assert snapshot["unsupported"]
    assert snapshot["reasons"] == [
        "问题要求营业收入，答案或证据仅绑定到主营业务收入，不能视为同一字段。"
    ]


def test_longyu_total_revenue_does_not_cross_bind_to_revenue():
    hit = _hit(
        chunk_id="longyu-total-revenue-page-7",
        doc_id=LONGYU_DOC_ID,
        doc_title="上海龙宇数据 2024年年度报告",
        text="主要会计数据\n营业总收入 1,404,920,973.02",
        metric="营业总收入",
        page=7,
        report_period="2024年度",
        raw_value="1,404,920,973.02",
        canonical_value="1404920973.02",
    )

    revenue_snapshot = _verification_snapshot(
        LONGYU_QUESTION, LONGYU_EXACT_ANSWER, [hit]
    )
    assert revenue_snapshot["verification"] is False
    assert revenue_snapshot["unsupported"]

    total_snapshot = _verification_snapshot(
        "上海龙宇数据2024年营业总收入是多少？",
        "2024年营业总收入为1,404,920,973.02元。[1]",
        [hit],
    )
    assert total_snapshot["verification"] is True
    assert total_snapshot["reasons"] == []


def test_longyu_legacy_canonical_revenue_without_conflicting_label_remains_supported():
    hit = _hit(
        chunk_id="longyu-legacy-revenue-number-only",
        doc_id=LONGYU_DOC_ID,
        doc_title="上海龙宇数据 2024年年度报告",
        text="1,404,920,973.02",
        metric="revenue",
        page=7,
        report_period="2024年度",
        raw_value="1,404,920,973.02",
        canonical_value="1404920973.02",
    )

    snapshot = _verification_snapshot(
        LONGYU_QUESTION, LONGYU_EXACT_ANSWER, [hit]
    )
    assert snapshot["verification"] is True
    assert snapshot["unsupported"] == []
    assert snapshot["reasons"] == []


def test_longyu_same_block_binds_each_revenue_alias_to_its_own_fact():
    metadata = {
        "doc_id": LONGYU_DOC_ID,
        "kb_id": SHENLIAN_KB_ID,
        "doc_title": "上海龙宇数据 2024 年度报告",
        "page": 7,
        "financial_metrics": "营业收入|主营业务收入",
        "report_period": "2024年度",
        "unit": "元",
        "statement_scope": "consolidated",
        "is_table": True,
        "financial_facts_json": json.dumps(
            [
                {
                    "metric": "营业收入",
                    "raw_value": "1,404,920,973.02",
                    "canonical_value": "1404920973.02",
                    "unit": "元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
                {
                    "metric": "主营业务收入",
                    "raw_value": "14.03",
                    "canonical_value": "1403000000",
                    "unit": "亿元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
            ],
            ensure_ascii=False,
        ),
    }
    hit = {
        "chunk_id": "longyu-same-block-two-revenue-fields",
        "text": "主要会计数据\n营业收入 1,404,920,973.02元；主营业务收入14.03亿元",
        "metadata": metadata,
        "score": 0.75,
    }
    cases = (
        (LONGYU_QUESTION, LONGYU_EXACT_ANSWER, True),
        (
            LONGYU_QUESTION,
            "2024年营业收入为14.03亿元。[1]",
            False,
        ),
        (
            "上海龙宇数据2024年主营业务收入是多少？",
            "2024年主营业务收入为14.03亿元。[1]",
            True,
        ),
    )

    for question, answer, expected in cases:
        snapshot = _verification_snapshot(question, answer, [hit])
        assert snapshot["verification"] is expected
        if expected:
            assert snapshot["unsupported"] == []
            assert snapshot["reasons"] == []
        else:
            assert snapshot["unsupported"]
            assert "营业收入" in snapshot["reasons"][0]
            assert "主营业务收入" in snapshot["reasons"][0]
