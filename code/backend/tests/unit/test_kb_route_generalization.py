"""跨公司、跨问法的财务文档路由泛化契约测试。"""

from __future__ import annotations

import pytest

import services.kb.retriever as retriever_module
from services.kb.retriever import (
    HybridRetriever,
    _company_query_clues,
    _financial_route_doc_ids,
)
from tests.mocks import FakeEmbedding


def _company_spec(
    full_name: str,
    short_name: str,
    st_name: str,
    stock_code: str,
    prefix: str,
) -> dict[str, str]:
    return {
        "full_name": full_name,
        "short_name": short_name,
        "st_name": st_name,
        "stock_code": stock_code,
        "full_doc": f"{prefix}-full-doc",
        "summary_doc": f"{prefix}-summary-doc",
    }


COMPANIES = (
    _company_spec(
        "星河智造股份有限公司",
        "星河智造",
        "ST星河",
        "688901",
        "fictional-star",
    ),
    _company_spec(
        "远景新材股份有限公司",
        "远景新材",
        "ST远景",
        "300902",
        "fictional-view",
    ),
)


def _route_item(
    company: dict[str, str],
    doc_id: str,
    *,
    summary: bool = False,
    complete_metadata: bool = True,
    body: str = "营业收入 234,567,890.12 元。",
) -> dict:
    title_suffix = "摘要" if summary else "年度报告"
    metadata = {
        "doc_id": doc_id,
        "doc_title": (
            f"{company['st_name']}-{company['full_name']}2024年{title_suffix}"
        ),
        "company_name": company["full_name"],
        "company_aliases": f"{company['short_name']}|{company['st_name']}",
        "stock_code": company["stock_code"],
    }
    if complete_metadata:
        metadata.update(
            {
                "report_period": "2024年度",
                "section_path": "主要会计数据",
                "is_table": True,
                "financial_metrics": "营业收入",
            }
        )
    return {
        "chunk_id": f"{doc_id}-chunk",
        "text": f"{company['full_name']} 2024年度\n{body}",
        "metadata": metadata,
    }


def _route_corpus() -> list[dict]:
    first, second = COMPANIES
    return [
        _route_item(first, first["full_doc"]),
        _route_item(first, first["summary_doc"], summary=True),
        _route_item(second, second["full_doc"]),
        _route_item(second, second["summary_doc"], summary=True),
    ]


@pytest.mark.parametrize(
    ("company_index", "query_builder"),
    [
        (0, lambda c: f"{c['full_name']}2024年营业收入是多少？"),
        (0, lambda c: f"{c['short_name']}2024年年报主要会计数据中的营业收入为多少？"),
        (0, lambda c: f"{c['st_name']}2024年全文和摘要都列出的营业收入是多少？"),
        (0, lambda c: f"股票代码{c['stock_code']}，2024年营业收入是多少？"),
        (1, lambda c: f"{c['full_name']}2024年营业收入是多少？"),
        (1, lambda c: f"{c['short_name']}2024年年报主要会计数据中的营业收入为多少？"),
        (1, lambda c: f"{c['st_name']}2024年全文和摘要都列出的营业收入是多少？"),
        (1, lambda c: f"股票代码{c['stock_code']}，2024年营业收入是多少？"),
    ],
)
def test_route_generalizes_identity_and_question_variants(
    company_index: int, query_builder
):
    company = COMPANIES[company_index]
    selected = set(_financial_route_doc_ids(query_builder(company), _route_corpus()))
    expected = {company["full_doc"], company["summary_doc"]}
    other = COMPANIES[1 - company_index]

    assert selected == expected
    assert not selected.intersection({other["full_doc"], other["summary_doc"]})


def test_route_does_not_guess_without_subject_or_on_conflicting_strong_clues():
    corpus = _route_corpus()
    first, second = COMPANIES

    assert _company_query_clues("2024年营业收入是多少？") == []
    assert _financial_route_doc_ids("2024年营业收入是多少？", corpus) == ()

    conflict = (
        f"{first['short_name']} {second['stock_code']}2024年营业收入是多少？"
    )
    assert _financial_route_doc_ids(conflict, corpus) == ()


def test_body_mention_of_other_company_does_not_change_document_subject():
    first, second = COMPANIES
    target = _route_item(
        first,
        first["full_doc"],
        body=(
            f"营业收入 234,567,890.12 元；正文比较了{second['full_name']}的行业数据。"
        ),
    )
    corpus = [
        target,
        _route_item(first, first["summary_doc"], summary=True),
        _route_item(second, second["full_doc"]),
        _route_item(second, second["summary_doc"], summary=True),
    ]

    selected = set(
        _financial_route_doc_ids(
            f"{first['short_name']}2024年营业收入是多少？", corpus
        )
    )

    assert selected == {first["full_doc"], first["summary_doc"]}
    assert second["full_doc"] not in selected
    assert second["summary_doc"] not in selected


def test_strict_route_is_used_without_fallback_when_all_sources_are_found(monkeypatch):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    company = COMPANIES[0]
    full = _route_item(company, company["full_doc"])
    summary = _route_item(company, company["summary_doc"], summary=True)
    noise = _route_item(COMPANIES[1], COMPANIES[1]["full_doc"])
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=5)
    retriever._corpus["fictional-kb"] = [full, summary, noise]
    calls: list[dict] = []

    def strict_search(query, *, top_k, kb_id, metadata_filters):
        calls.append(metadata_filters)
        return [full, summary]

    monkeypatch.setattr(retriever, "_search_one", strict_search)
    merged = retriever._merge_financial_route(
        [(0.9, 0, noise)],
        query=f"{company['st_name']}2024年全文和摘要都列出的营业收入是多少？",
        top_k=5,
        kb_id="fictional-kb",
        metadata_filters=None,
    )

    assert len(calls) == 1
    injected_ids = {hit["metadata"]["doc_id"] for _, _, hit in merged[:2]}
    assert injected_ids == {company["full_doc"], company["summary_doc"]}
    assert all(hit["metadata"]["doc_id"] != COMPANIES[1]["full_doc"] for _, _, hit in merged[:2])


def test_fallback_is_doc_bound_and_only_fills_missing_summary_metadata(monkeypatch):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    company = COMPANIES[0]
    full = _route_item(company, company["full_doc"])
    summary = _route_item(
        company,
        company["summary_doc"],
        summary=True,
        complete_metadata=False,
        body="营业收入 234,567,890.12 元。",
    )
    other_company = COMPANIES[1]
    other = _route_item(other_company, other_company["full_doc"])
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=5)
    retriever._corpus["fictional-kb"] = [full, summary, other]
    calls: list[dict] = []

    def search_by_doc(query, *, top_k, kb_id, metadata_filters):
        calls.append(metadata_filters)
        clauses = metadata_filters["$and"]
        doc_clause = next(clause["doc_id"]["$in"] for clause in clauses if "doc_id" in clause)
        if doc_clause == [summary["metadata"]["doc_id"]]:
            return [summary]
        return [full]

    monkeypatch.setattr(retriever, "_search_one", search_by_doc)
    merged = retriever._merge_financial_route(
        [(0.9, 0, other)],
        query=f"{company['short_name']}2024年全文和摘要都列出的营业收入是多少？",
        top_k=5,
        kb_id="fictional-kb",
        metadata_filters=None,
    )

    assert len(calls) == 2
    strict_doc_ids = next(
        clause["doc_id"]["$in"]
        for clause in calls[0]["$and"]
        if "doc_id" in clause
    )
    fallback_doc_ids = next(
        clause["doc_id"]["$in"]
        for clause in calls[1]["$and"]
        if "doc_id" in clause
    )
    assert set(strict_doc_ids) == {company["full_doc"], company["summary_doc"]}
    assert fallback_doc_ids == [company["summary_doc"]]
    injected_ids = {hit["metadata"]["doc_id"] for _, _, hit in merged[:2]}
    assert injected_ids == {company["full_doc"], company["summary_doc"]}
    assert all(
        hit["metadata"]["doc_id"] != other_company["full_doc"]
        for _, _, hit in merged[:2]
    )
