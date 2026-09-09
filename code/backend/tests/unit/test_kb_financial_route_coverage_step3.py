"""第 3 步财务路由契约：合成候选、跨公司和跨期间安全边界。"""

from __future__ import annotations

import pytest

import services.kb.retriever as retriever_module
from services.kb.qa_graph import _facts_for_question, _question_period_tokens
from services.kb.retriever import (
    HybridRetriever,
    _company_query_clues,
    _financial_route_doc_ids,
    _metric_match_kind,
    _query_profile,
)

COMPANY_A = {
    "full": "晨星智造股份有限公司",
    "short": "晨星智造",
    "st": "ST晨星",
    "code": "681234",
}
COMPANY_B = {
    "full": "远岑材料股份有限公司",
    "short": "远岑材料",
    "st": "ST远岑",
    "code": "682345",
}


def _route_metadata(
    company: dict[str, str],
    doc_id: str,
    *,
    period: str = "2024年度",
    identity_field: str = "company_name",
    identity_value: str | None = None,
    family: str = "annual-report",
    metric: str = "营业收入",
) -> dict[str, object]:
    value = identity_value or company["full"]
    metadata: dict[str, object] = {
        "doc_id": doc_id,
        "report_period": period,
        "section_path": "主要会计数据",
        "is_table": True,
        "financial_metrics": metric,
        "report_family": family,
    }
    if identity_field:
        metadata[identity_field] = value
    return metadata


def _candidate(
    company: dict[str, str],
    doc_id: str,
    text: str,
    *,
    period: str = "2024年度",
    identity_field: str = "company_name",
    identity_value: str | None = None,
    family: str = "annual-report",
    metric: str = "营业收入",
) -> dict[str, object]:
    return {
        "chunk_id": f"{doc_id}-chunk",
        "text": text,
        "metadata": _route_metadata(
            company,
            doc_id,
            period=period,
            identity_field=identity_field,
            identity_value=identity_value,
            family=family,
            metric=metric,
        ),
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "doc_title",
            "ST晨星-晨星智造股份有限公司2024年年度报告全文",
        ),
        ("source", "晨星智造股份有限公司_2024_annual.pdf"),
        ("file_name", "晨星智造股份有限公司-年度报告-2024.pdf"),
    ],
)
def test_authoritative_identity_field_routes_only_the_matching_document(
    field: str, value: str
):
    target = _candidate(
        COMPANY_A,
        "fictional-a-authoritative",
        "主要会计数据\n营业收入 111.1 万元",
        identity_field=field,
        identity_value=value,
    )
    other = _candidate(
        COMPANY_B,
        "fictional-b-noise",
        "主要会计数据\n营业收入 222.2 万元\n正文提到晨星智造股份有限公司",
    )

    query = "晨星智造2024年营业总收入是多少"
    assert _company_query_clues(query)
    assert _financial_route_doc_ids(query, [target, other]) == (
        "fictional-a-authoritative",
    )


def test_total_revenue_query_is_revenue_route_but_main_business_revenue_is_not(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    revenue = _candidate(
        COMPANY_A,
        "fictional-a-revenue",
        "营业收入\n111.1 万元",
        metric="营业收入",
    )
    main_business = _candidate(
        COMPANY_A,
        "fictional-a-main-business",
        "主营业务收入\n222.2 万元",
        metric="主营业务收入",
    )
    corpus = [revenue, main_business]
    profile = _query_profile("晨星智造2024年营业总收入是多少")

    assert profile["metric_groups"] == frozenset({"revenue"})
    assert _metric_match_kind(main_business, profile) == ""

    retriever = HybridRetriever(object(), embeddings=object(), top_k=5)
    retriever._corpus["synthetic-kb"] = corpus
    monkeypatch.setattr(
        retriever,
        "_search_one",
        lambda query, *, top_k, kb_id, metadata_filters: [
            main_business,
            revenue,
        ],
    )

    merged = retriever._merge_financial_route(
        [(0.1, 0, {"chunk_id": "unrelated", "text": "普通文本", "metadata": {}})],
        query="晨星智造2024年营业总收入是多少",
        top_k=5,
        kb_id="synthetic-kb",
        metadata_filters=None,
    )

    injected_ids = {
        hit["metadata"]["doc_id"]
        for _score, _sequence, hit in merged
        if hit["metadata"].get("doc_id", "").startswith("fictional-a-")
    }
    assert injected_ids == {"fictional-a-revenue"}


def test_same_report_can_supply_revenue_and_rd_total_across_pages_without_rd_expense(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    revenue = _candidate(
        COMPANY_A,
        "fictional-a-revenue-page",
        "营业收入\n111.1 万元",
        metric="营业收入",
    )
    rd_total = _candidate(
        COMPANY_A,
        "fictional-a-rd-total-page",
        "研发投入合计\n333.3 万元",
        metric="研发投入合计",
    )
    rd_expense = _candidate(
        COMPANY_A,
        "fictional-a-rd-expense-page",
        "研发费用\n444.4 万元",
        metric="研发费用",
    )
    corpus = [revenue, rd_total, rd_expense]
    query = "晨星智造2024年年报主要会计数据中的营业收入和研发投入合计是多少"
    retriever = HybridRetriever(object(), embeddings=object(), top_k=8)
    retriever._corpus["synthetic-kb"] = corpus
    monkeypatch.setattr(
        retriever,
        "_search_one",
        lambda query, *, top_k, kb_id, metadata_filters: [
            revenue,
            rd_expense,
            rd_total,
        ],
    )

    merged = retriever._merge_financial_route(
        [(0.1, 0, {"chunk_id": "unrelated", "text": "普通文本", "metadata": {}})],
        query=query,
        top_k=8,
        kb_id="synthetic-kb",
        metadata_filters=None,
    )

    ids = {
        hit["metadata"]["doc_id"]
        for _score, _sequence, hit in merged
        if hit["metadata"].get("doc_id", "").startswith("fictional-a-")
    }
    assert ids == {
        "fictional-a-revenue-page",
        "fictional-a-rd-total-page",
    }
    assert "fictional-a-rd-expense-page" not in ids


def _periods_in_filter(value: object) -> set[str]:
    if isinstance(value, list):
        periods: set[str] = set()
        for item in value:
            periods.update(_periods_in_filter(item))
        return periods
    if not isinstance(value, dict):
        return set()
    periods: set[str] = set()
    if "report_period" in value:
        condition = value["report_period"]
        if isinstance(condition, str):
            periods.add(condition)
        elif isinstance(condition, dict):
            values = condition.get("$in", [])
            if isinstance(values, list):
                periods.update(item for item in values if isinstance(item, str))
    for key in ("$and", "$or"):
        periods.update(_periods_in_filter(value.get(key)))
    return periods


def test_two_requested_periods_stay_in_one_company_report_family(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    target_2024 = _candidate(
        COMPANY_A,
        "fictional-a-2024",
        "营业收入\n111.1 万元",
        period="2024年度",
    )
    target_2023 = _candidate(
        COMPANY_A,
        "fictional-a-2023",
        "营业收入\n112.2 万元",
        period="2023年度",
    )
    wrong_period = _candidate(
        COMPANY_A,
        "fictional-a-wrong-period",
        "营业收入\n113.3 万元",
        period="2022年度",
    )
    other_company = _candidate(
        COMPANY_B,
        "fictional-b-2024",
        "营业收入\n221.1 万元",
        period="2024年度",
    )
    body_only = {
        "chunk_id": "fictional-body-only",
        "text": "财务数据\n正文提到晨星智造股份有限公司，但不是该文档主体\n营业收入 999.9 万元",
        "metadata": {
            "doc_id": "fictional-body-only",
            "report_period": "2024年度",
            "section_path": "主要会计数据",
            "is_table": True,
            "financial_metrics": "营业收入",
        },
    }
    query = "晨星智造2024年和2023年年报营业收入分别是多少"
    assert _question_period_tokens(query) == ["2024年", "2023年"]
    assert set(
        _financial_route_doc_ids(
            query,
            [target_2024, target_2023, wrong_period, other_company, body_only],
        )
    ) == {"fictional-a-2024", "fictional-a-2023", "fictional-a-wrong-period"}

    candidates = [target_2024, target_2023, wrong_period, other_company]
    retriever = HybridRetriever(object(), embeddings=object(), top_k=8)
    retriever._corpus["synthetic-kb"] = candidates
    calls: list[object] = []

    def search_by_filter(query, *, top_k, kb_id, metadata_filters):
        calls.append(metadata_filters)
        requested_periods = _periods_in_filter(metadata_filters)
        return [
            hit
            for hit in candidates
            if hit["metadata"]["report_period"] in requested_periods
        ]

    monkeypatch.setattr(retriever, "_search_one", search_by_filter)
    merged = retriever._merge_financial_route(
        [(0.1, 0, {"chunk_id": "unrelated", "text": "普通文本", "metadata": {}})],
        query=query,
        top_k=8,
        kb_id="synthetic-kb",
        metadata_filters=None,
    )

    selected = {
        hit["metadata"]["doc_id"]
        for _score, _sequence, hit in merged
        if hit["metadata"].get("doc_id", "").startswith("fictional-")
    }
    assert selected == {"fictional-a-2024", "fictional-a-2023"}
    assert calls


def test_financial_facts_keep_both_periods_and_drop_wrong_period():
    facts = [
        {
            "metric": "revenue",
            "raw_value": "111.1 万元",
            "canonical_value": "1111000",
            "unit": "万元",
            "report_period": "2024年度",
        },
        {
            "metric": "revenue",
            "raw_value": "112.2 万元",
            "canonical_value": "1122000",
            "unit": "万元",
            "report_period": "2023年度",
        },
        {
            "metric": "revenue",
            "raw_value": "113.3 万元",
            "canonical_value": "1133000",
            "unit": "万元",
            "report_period": "2022年度",
        },
    ]
    from services.kb.qa_graph import _parse_financial_facts_json

    selected = _facts_for_question(
        _parse_financial_facts_json(facts),
        "晨星智造2024年和2023年营业收入分别是多少",
    )

    assert {fact.report_period for fact in selected} == {"2024年", "2023年"}


def test_body_name_fallback_requires_controlled_document_identity_not_rank():
    uncontrolled = {
        "chunk_id": "fictional-uncontrolled-body",
        "text": "财务数据\n正文提到晨星智造股份有限公司\n营业收入 111.1 万元",
        "metadata": {"doc_id": "fictional-uncontrolled-body"},
        "score": 999.0,
    }
    controlled = {
        "chunk_id": "fictional-controlled-header",
        "text": "证券简称：晨星智造\n营业收入 112.2 万元",
        "metadata": {"doc_id": "fictional-controlled-header"},
        "score": 0.01,
    }
    query = "晨星智造2024年营业收入是多少"

    assert _financial_route_doc_ids(query, [uncontrolled]) == ()
    assert _financial_route_doc_ids(query, [controlled]) == (
        "fictional-controlled-header",
    )
