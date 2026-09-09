"""原因题候选保护的通用离线契约测试。"""

from __future__ import annotations

import pytest

from services.kb.qa_graph import _protect_reranked_financial_candidates

COMPANY = "星澜工业股份有限公司"
OTHER_COMPANY = "远岑材料股份有限公司"
KB_ID = "synthetic-kb"


def _cause_candidate(
    *,
    doc_id: str,
    company: str = COMPANY,
    period: str = "2024年度",
    text: str,
    kb_id: str = KB_ID,
) -> dict:
    return {
        "chunk_id": f"{doc_id}-cause-chunk",
        "text": text,
        "metadata": {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "doc_title": f"{company}2024年年度报告",
            "report_period": period,
            "source": "synthetic-annual-report.pdf",
            "page": "synthetic-cause-page",
            "section_path": "经营情况概述",
            "is_table": False,
        },
    }


@pytest.mark.parametrize(
    ("metric", "question_text"),
    [
        ("营业收入", "营业收入下降的原因是什么"),
        ("归母净利润", "归母净利润下降的原因是什么"),
    ],
)
def test_protects_same_company_period_metric_change_cause_candidate(
    metric: str, question_text: str
):
    question = f"{COMPANY}2024年{question_text}"
    cause = _cause_candidate(
        doc_id="fictional-cause-target",
        text=f"{metric}同比下降，主要系产品结构调整所致。",
    )
    weak = _cause_candidate(
        doc_id="fictional-cause-weak",
        text="公司治理情况及业务安排。",
    )

    result = _protect_reranked_financial_candidates(
        question,
        [weak, cause],
        [weak],
        top_n=2,
    )

    assert [item["chunk_id"] for item in result] == [
        cause["chunk_id"],
        weak["chunk_id"],
    ]
    assert result[0]["metadata"] == cause["metadata"]
    assert result[0]["metadata"]["doc_id"] == "fictional-cause-target"
    assert result[0]["metadata"]["source"] == "synthetic-annual-report.pdf"
    assert result[0]["metadata"]["page"] == "synthetic-cause-page"


def test_protection_does_not_expand_to_other_company_mentioned_in_body():
    question = f"{COMPANY}2024年营业收入下降的原因是什么"
    other_company_cause = _cause_candidate(
        doc_id="fictional-other-company",
        company=OTHER_COMPANY,
        text=(
            f"正文提到{COMPANY}，但本报告主体为{OTHER_COMPANY}；"
            "营业收入同比下降，主要系产品结构调整所致。"
        ),
    )
    reranked = [_cause_candidate(doc_id="fictional-weak", text="组织架构说明。")]

    result = _protect_reranked_financial_candidates(
        question,
        [reranked[0], other_company_cause],
        reranked,
        top_n=2,
    )

    assert result == reranked


def test_protection_does_not_use_wrong_period_candidate():
    question = f"{COMPANY}2024年营业收入下降的原因是什么"
    wrong_period = _cause_candidate(
        doc_id="fictional-wrong-period",
        period="2023年度",
        text="营业收入同比下降，主要系产品结构调整所致。",
    )
    reranked = [_cause_candidate(doc_id="fictional-weak", text="组织架构说明。")]

    result = _protect_reranked_financial_candidates(
        question,
        [reranked[0], wrong_period],
        reranked,
        top_n=2,
    )

    assert result == reranked


@pytest.mark.parametrize(
    "text",
    [
        "营业收入已经披露，主要系产品结构调整所致。",
        "主要系产品结构调整所致。",
        "营业收入同比下降。",
    ],
)
def test_protection_requires_change_signal_metric_and_cause(text: str):
    question = f"{COMPANY}2024年营业收入下降的原因是什么"
    candidate = _cause_candidate(doc_id="fictional-incomplete-cause", text=text)
    reranked = [_cause_candidate(doc_id="fictional-weak", text="组织架构说明。")]

    result = _protect_reranked_financial_candidates(
        question,
        [reranked[0], candidate],
        reranked,
        top_n=2,
    )

    assert result == reranked


def test_non_reason_question_does_not_trigger_cause_protection():
    question = f"{COMPANY}2024年营业收入是多少"
    candidate = _cause_candidate(
        doc_id="fictional-non-reason",
        text="营业收入同比下降，主要系产品结构调整所致。",
    )
    reranked = [_cause_candidate(doc_id="fictional-weak", text="组织架构说明。")]

    result = _protect_reranked_financial_candidates(
        question,
        [reranked[0], candidate],
        reranked,
        top_n=2,
    )

    assert result == reranked


def test_protection_is_bounded_by_top_k_and_does_not_return_all_raw_candidates():
    question = f"{COMPANY}2024年营业收入下降的原因是什么"
    cause = _cause_candidate(
        doc_id="fictional-cause-target",
        text="营业收入同比下降，主要系产品结构调整所致。",
    )
    first_ranked = _cause_candidate(doc_id="fictional-first-ranked", text="公司概况。")
    second_ranked = _cause_candidate(doc_id="fictional-second-ranked", text="风险提示。")
    raw_only = _cause_candidate(
        doc_id="fictional-raw-only",
        text="营业收入同比上升，主要系市场需求改善所致。",
    )

    result = _protect_reranked_financial_candidates(
        question,
        [first_ranked, raw_only, cause, second_ranked],
        [first_ranked, second_ranked],
        top_n=2,
    )

    assert len(result) == 2
    assert result[0]["metadata"]["doc_id"] == "fictional-cause-target"
    assert {item["metadata"]["doc_id"] for item in result} == {
        "fictional-cause-target",
        "fictional-first-ranked",
    }
    assert "fictional-raw-only" not in {
        item["metadata"]["doc_id"] for item in result
    }
