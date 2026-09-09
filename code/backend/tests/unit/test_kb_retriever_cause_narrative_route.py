"""Offline regression tests for the financial cause-narrative route.

The fixtures intentionally use fictional companies, reports, identifiers, and
amounts.  ``_search_one`` is replaced in route tests so this module never
needs a vector database, an embedding service, or a language model.
"""

from __future__ import annotations

import services.kb.retriever as retriever_module
from services.kb.retriever import HybridRetriever


class _NoopEmbeddings:
    def embed_query(self, _query: str) -> list[float]:
        return [0.0]


KB_ID = "fictional-offline-finance-kb"


def _hit(
    chunk_id: str,
    *,
    doc_id: str,
    alias: str,
    company_name: str,
    period: str = "2026年半年度",
    body: str,
    is_table: bool,
    section_path: str = "经营情况概述",
    financial_metrics: str = "营业收入",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"{company_name} {period}\n{body}",
        "metadata": {
            "doc_id": doc_id,
            # Keep the full company name as the authoritative document field;
            # the query exercises matching its fictional short-name prefix.
            "doc_title": f"{company_name}{period}报告",
            "company_name": company_name,
            "report_period": period,
            "is_table": is_table,
            "section_path": section_path,
            "financial_metrics": financial_metrics,
        },
        "rrf_score": 0.01,
    }


def _doc_ids_for(metadata_corpus: list[dict]) -> tuple[str, ...]:
    return retriever_module._financial_route_doc_ids(
        "海岛星辰2026年上半年营业收入下降原因", metadata_corpus
    )


def test_company_prefix_locks_one_authoritative_doc_and_rejects_ambiguous_prefix():
    """A short-name prefix may select one report, but never guess between two."""

    target = _hit(
        "island-authority-page",
        doc_id="island-authority-report",
        alias="海岛星辰",
        company_name="海岛星辰医药集团股份有限公司",
        body="正文提及远港云杉制造有限公司，仅作为行业对比噪声。",
        is_table=True,
        section_path="主要会计数据",
    )
    target_second_chunk = _hit(
        "island-body-page",
        doc_id="island-authority-report",
        alias="海岛星辰",
        company_name="海岛星辰医药集团股份有限公司",
        body="正文再次提到远港云杉制造有限公司，但不改变报告主体。",
        is_table=False,
    )
    unrelated_body_mention = _hit(
        "harbor-body-page",
        doc_id="harbor-report",
        alias="远港云杉",
        company_name="远港云杉制造有限公司",
        body="正文提到海岛星辰，但本报告主体仍是远港云杉制造有限公司。",
        is_table=True,
        section_path="主要会计数据",
    )

    assert _doc_ids_for([target, target_second_chunk, unrelated_body_mention]) == (
        "island-authority-report",
    )

    ambiguous_a = _hit(
        "misty-a-page",
        doc_id="misty-tech-report",
        alias="雾桥星火",
        company_name="雾桥星火科技股份有限公司",
        body="营业收入 4,321.09 元。",
        is_table=True,
        section_path="主要会计数据",
    )
    ambiguous_b = _hit(
        "misty-b-page",
        doc_id="misty-medicine-report",
        alias="雾桥星火",
        company_name="雾桥星火医药股份有限公司",
        body="营业收入 5,432.10 元。",
        is_table=True,
        section_path="主要会计数据",
    )
    ambiguous_query = "雾桥星火2026年上半年营业收入下降原因"

    assert retriever_module._financial_route_doc_ids(
        ambiguous_query, [ambiguous_a, ambiguous_b]
    ) == ()


def test_cause_narrative_is_injected_after_table_first_recall_and_noise_is_rejected(
    monkeypatch,
):
    """Cause evidence is bounded by the locked doc, period, metric, number, and cause gates."""

    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(
        object(), embeddings=_NoopEmbeddings(), top_k=8
    )
    company = "海岛星辰医药集团股份有限公司"
    doc_id = "island-authority-report"
    query = "海岛星辰2026年上半年营业收入下降原因"

    strict_table = _hit(
        "revenue-table-first",
        doc_id=doc_id,
        alias="海岛星辰",
        company_name=company,
        body=(
            "| 项目 | 营业收入 | 本期 |\n"
            "| --- | --- | --- |\n"
            "| 金额 | 8,765,432.10 |"
        ),
        is_table=True,
        section_path="主要会计数据",
    )
    cause_narrative = _hit(
        "revenue-cause-narrative",
        doc_id=doc_id,
        alias="海岛星辰",
        company_name=company,
        body="营业收入较上年同期下降37.50%，主要系核心产品销售额减少所致。",
        is_table=False,
    )
    wrong_period = _hit(
        "wrong-period-cause",
        doc_id=doc_id,
        alias="海岛星辰",
        company_name=company,
        period="2025年半年度",
        body="营业收入较上年同期下降21.00%，主要系渠道调整所致。",
        is_table=False,
    )
    other_company = _hit(
        "other-company-cause",
        doc_id="harbor-report",
        alias="远港云杉",
        company_name="远港云杉制造有限公司",
        body="营业收入较上年同期下降19.00%，主要系订单减少所致。",
        is_table=False,
    )
    no_causal_expression = _hit(
        "no-causal-expression",
        doc_id=doc_id,
        alias="海岛星辰",
        company_name=company,
        body="营业收入同比下降18.00%，销售额减少。",
        is_table=False,
    )
    no_requested_metric = _hit(
        "other-metric-cause",
        doc_id=doc_id,
        alias="海岛星辰",
        company_name=company,
        body="归母净利润较上年同期下降12.00%，主要系费用增加所致。",
        is_table=False,
        financial_metrics="归属于上市公司股东的净利润",
    )
    no_financial_number = _hit(
        "no-financial-number",
        doc_id=doc_id,
        alias="海岛星辰",
        company_name=company,
        body="营业收入下降，主要系核心产品销售额减少所致。",
        is_table=False,
    )
    retriever._corpus[KB_ID] = [
        strict_table,
        cause_narrative,
        wrong_period,
        other_company,
        no_causal_expression,
        no_requested_metric,
        no_financial_number,
    ]

    calls: list[dict] = []

    def fake_search_one(query_text, *, top_k, kb_id, metadata_filters):
        calls.append(
            {
                "query": query_text,
                "top_k": top_k,
                "kb_id": kb_id,
                "metadata_filters": metadata_filters,
            }
        )
        # The strict structured pass sees only the table.  The correct
        # narrative is deliberately available only through ``_corpus``.
        return [strict_table]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)

    ordinary = _hit(
        "ordinary-recall",
        doc_id=doc_id,
        alias="海岛星辰",
        company_name=company,
        body="公司概况与经营范围。",
        is_table=False,
        section_path="公司概况",
        financial_metrics="",
    )
    merged = retriever._merge_financial_route(
        [(0.8, 0, ordinary)],
        query=query,
        top_k=8,
        kb_id=KB_ID,
        metadata_filters=None,
    )

    assert calls, "the offline strict structured search should run"
    assert calls[0]["kb_id"] == KB_ID
    assert any(
        clause.get("is_table") is True
        for clause in calls[0]["metadata_filters"]["$and"]
    )

    merged_by_id = {hit[2]["chunk_id"]: hit[2] for hit in merged}
    assert "revenue-table-first" in merged_by_id
    assert "revenue-cause-narrative" in merged_by_id
    assert merged_by_id["revenue-cause-narrative"]["metadata"]["is_table"] is False
    assert (
        "营业收入较上年同期下降37.50%，主要系核心产品销售额减少所致"
        in merged_by_id["revenue-cause-narrative"]["text"]
    )
    assert {
        "wrong-period-cause",
        "other-company-cause",
        "no-causal-expression",
        "other-metric-cause",
        "no-financial-number",
    }.isdisjoint(merged_by_id)


def test_field_bound_cause_narrative_precedes_crowded_tables_within_doc_cap(
    monkeypatch,
):
    """A field-bound cause block must survive a same-document table crowd."""

    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(
        object(),
        embeddings=_NoopEmbeddings(),
        top_k=20,
        max_candidates_per_doc=12,
    )
    company = "松岚曜石健康科技有限公司"
    doc_id = "fictional-cause-crowd-report"
    query = "松岚曜石2026年上半年营业收入下降原因"

    crowded_tables = [
        _hit(
            f"fictional-table-{index:02d}",
            doc_id=doc_id,
            alias="松岚曜石",
            company_name=company,
            body=(
                "| 项目 | 营业收入 | 本期金额 | 上期金额 |\n"
                "| --- | --- | --- | --- |\n"
                f"| 分块{index:02d} | 营业收入 | {120000 + index * 101.25:,.2f} | "
                f"{240000 + index * 101.25:,.2f} | -{10 + index / 100:.2f}% |"
            ),
            is_table=True,
            section_path="主要会计数据",
        )
        for index in range(13)
    ]
    generic_cause = _hit(
        "fictional-generic-cause",
        doc_id=doc_id,
        alias="松岚曜石",
        company_name=company,
        body=(
            "营业收入下降主要源于市场需求变化，主要原因在于需求阶段性调整，"
            "下降幅度为13.50%。"
        ),
        is_table=False,
        section_path="经营情况概述",
    )
    field_bound_cause = _hit(
        "fictional-field-bound-cause",
        doc_id=doc_id,
        alias="松岚曜石",
        company_name=company,
        body=(
            "营业收入变动原因说明：主要系核心产品销售额较上年同期减少所致，"
            "下降幅度为13.50%。"
        ),
        is_table=False,
        section_path="经营情况概述",
    )
    ordinary = _hit(
        "fictional-ordinary-recall",
        doc_id=doc_id,
        alias="松岚曜石",
        company_name=company,
        body="公司治理与组织架构说明。",
        is_table=False,
        section_path="公司概况",
        financial_metrics="",
    )
    retriever._corpus[KB_ID] = [
        *crowded_tables,
        generic_cause,
        field_bound_cause,
        ordinary,
    ]

    calls: list[dict] = []

    def fake_search_one(query_text, *, top_k, kb_id, metadata_filters):
        calls.append(
            {
                "query": query_text,
                "top_k": top_k,
                "kb_id": kb_id,
                "metadata_filters": metadata_filters,
            }
        )
        if metadata_filters is None:
            return [ordinary]
        # The strict route sees one table first; the narrative blocks are
        # deliberately available only through the locked in-memory corpus.
        return [crowded_tables[0]]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)

    merged = retriever.search_queries([query], top_k=20, kb_id=KB_ID)
    merged_ids = [hit["chunk_id"] for hit in merged]

    assert calls, "the offline structured route should call the mocked search"
    assert any(
        clause.get("is_table") is True
        for clause in calls[1]["metadata_filters"]["$and"]
    )
    assert len(crowded_tables) >= 13
    assert merged_ids[:12].count(field_bound_cause["chunk_id"]) == 1
    assert merged_ids.index(field_bound_cause["chunk_id"]) < 12
    assert merged_ids.index(field_bound_cause["chunk_id"]) < merged_ids.index(
        generic_cause["chunk_id"]
    )
    first_table_position = min(
        merged_ids.index(table["chunk_id"]) for table in crowded_tables if table["chunk_id"] in merged_ids
    )
    assert merged_ids.index(field_bound_cause["chunk_id"]) < first_table_position
