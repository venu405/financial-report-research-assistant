"""Offline regression coverage for preserving a financial risk narrative.

The corpus is entirely fictional.  The mocked search path keeps this test
independent from vector stores, embeddings, containers, and language models.
"""

from __future__ import annotations

from typing import Any

import services.kb.retriever as retriever_module
from services.kb.retriever import HybridRetriever


class _NoopEmbeddings:
    def embed_query(self, _query: str) -> list[float]:
        return [0.0]


KB_ID = "offline-fictional-risk-kb"
TARGET_COMPANY = "岚河星辉智能装备股份有限公司"
TARGET_ALIAS = "岚河星辉"
FULL_DOC = "fictional-full-report"
SUMMARY_DOC = "fictional-summary-report"
OTHER_DOC = "fictional-other-company-report"
TARGET_PERIOD = "2024年度"
OLD_PERIOD = "2023年度"


def _hit(
    chunk_key: str,
    *,
    doc_id: str,
    company_name: str,
    period: str = TARGET_PERIOD,
    body: str,
    is_table: bool,
    section_path: str,
    financial_metrics: str = "营业收入|归属于母公司股东的净利润",
) -> dict[str, Any]:
    return {
        "chunk_id": f"synthetic-{chunk_key}",
        "text": f"{company_name} {period}\n{body}",
        "metadata": {
            "doc_id": doc_id,
            "doc_title": f"{company_name}{period}年度报告",
            "company_name": company_name,
            "report_period": period,
            "is_table": is_table,
            "section_path": section_path,
            "financial_metrics": financial_metrics,
        },
        "rrf_score": 0.01,
    }


def _table_chunks(doc_id: str, company_name: str, prefix: str) -> list[dict[str, Any]]:
    return [
        _hit(
            f"{prefix}-table-{index:02d}",
            doc_id=doc_id,
            company_name=company_name,
            body=(
                "| 项目 | 营业收入 | 净利润 |\n"
                "| --- | ---: | ---: |\n"
                f"| 分块{index:02d} | {180000 + index * 137.25:,.2f} | "
                f"{-23000 - index * 17.5:,.2f} |"
            ),
            is_table=True,
            section_path="主要会计数据",
        )
        for index in range(16)
    ]


def test_raw_risk_narrative_survives_full_summary_table_crowd_and_boundaries(
    monkeypatch,
):
    """A high-ranked narrative must not be squeezed out by route injection."""

    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(
        object(),
        embeddings=_NoopEmbeddings(),
        top_k=40,
        max_candidates_per_doc=12,
    )
    query = (
        f"{TARGET_ALIAS}2024年年度报告披露的营业收入和净利润分别是多少？"
        "这触发了什么风险？"
    )

    full_tables = _table_chunks(FULL_DOC, TARGET_COMPANY, "full")
    summary_tables = _table_chunks(SUMMARY_DOC, TARGET_COMPANY, "summary")
    assert len(full_tables) + len(summary_tables) >= 32

    risk_narrative = _hit(
        "target-risk-narrative",
        doc_id=FULL_DOC,
        company_name=TARGET_COMPANY,
        body=(
            "重大风险提示：因持续亏损且营业收入触及上市规则相关条件，"
            "公司股票将被实施退市风险警示，股票简称前冠以*ST。"
            "本期营业收入为176,543,210.00元，归属于母公司股东的净利润为-23,456,789.00元。"
        ),
        is_table=False,
        section_path="重大风险提示",
    )
    wrong_period_risk = _hit(
        "wrong-period-risk-narrative",
        doc_id=FULL_DOC,
        company_name=TARGET_COMPANY,
        period=OLD_PERIOD,
        body=(
            "重大风险提示：因亏损触及相关条件，公司将被实施退市风险警示。"
            "营业收入为120,000,000.00元，净利润为-18,000,000.00元。"
        ),
        is_table=False,
        section_path="重大风险提示",
    )
    other_company_risk = _hit(
        "other-company-risk-narrative",
        doc_id=OTHER_DOC,
        company_name="澄湾远见新材料股份有限公司",
        body=(
            "重大风险提示：公司股票将被实施退市风险警示。"
            "营业收入为999,000,000.00元，净利润为-88,000,000.00元。"
        ),
        is_table=False,
        section_path="重大风险提示",
    )
    ordinary_recall = _hit(
        "ordinary-recall",
        doc_id=FULL_DOC,
        company_name=TARGET_COMPANY,
        body="公司治理与组织架构说明。",
        is_table=False,
        section_path="公司概况",
        financial_metrics="",
    )

    # The two report forms share the same authoritative company and period;
    # the risk block exists only in the full-report corpus as raw evidence.
    retriever._corpus[KB_ID] = [
        *full_tables,
        *summary_tables,
        risk_narrative,
        wrong_period_risk,
        other_company_risk,
        ordinary_recall,
    ]

    def fake_search_one(
        _query_text: str,
        *,
        top_k: int,
        kb_id: str,
        metadata_filters: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        assert kb_id == KB_ID
        if metadata_filters is None:
            # The correct narrative is intentionally a high-ranked raw hit.
            return [risk_narrative, ordinary_recall]
        # Structured calls are table-only; narrative evidence must be
        # preserved from the locked in-memory corpus rather than from search.
        return [*full_tables, *summary_tables]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)

    observed: dict[str, list[dict[str, Any]]] = {}
    original_merge = retriever._merge_financial_route

    def recording_merge(*args: Any, **kwargs: Any) -> list[tuple[float, int, dict[str, Any]]]:
        merged = original_merge(*args, **kwargs)
        observed["merged"] = [hit for _score, _sequence, hit in merged]
        return merged

    monkeypatch.setattr(retriever, "_merge_financial_route", recording_merge)
    original_dedupe = retriever_module._dedupe_and_diversify

    def recording_dedupe(
        hits: list[dict[str, Any]], max_per_doc: int | None
    ) -> list[dict[str, Any]]:
        observed["before_dedupe"] = list(hits)
        return original_dedupe(hits, max_per_doc)

    monkeypatch.setattr(retriever_module, "_dedupe_and_diversify", recording_dedupe)

    results = retriever.search_queries([query], top_k=40, kb_id=KB_ID)
    merged_ids = {hit["chunk_id"] for hit in observed["merged"]}
    before_dedupe_ids = {hit["chunk_id"] for hit in observed["before_dedupe"]}
    result_ids = {hit["chunk_id"] for hit in results}

    target_id = risk_narrative["chunk_id"]
    assert target_id in merged_ids
    assert target_id in before_dedupe_ids
    assert target_id in result_ids

    same_doc_results = [
        hit
        for hit in results
        if hit["metadata"].get("doc_id") == FULL_DOC
    ]
    assert same_doc_results.index(risk_narrative) < 12
    assert all(
        hit["metadata"].get("report_period") == TARGET_PERIOD
        for hit in results
        if hit["metadata"].get("doc_id") in {FULL_DOC, SUMMARY_DOC}
    )
    assert wrong_period_risk["chunk_id"] not in merged_ids
    assert other_company_risk["chunk_id"] not in merged_ids
    assert wrong_period_risk["chunk_id"] not in result_ids
    assert other_company_risk["chunk_id"] not in result_ids
