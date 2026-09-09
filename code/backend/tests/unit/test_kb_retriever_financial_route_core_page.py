"""Regression coverage for same-document financial core-table supplementation."""

from __future__ import annotations

import services.kb.retriever as retriever_module
from services.kb.retriever import HybridRetriever


class _NoopEmbeddings:
    def embed_query(self, _query: str) -> list[float]:
        return [0.0]


def _financial_hit(
    chunk_id: str,
    *,
    doc_id: str,
    company_name: str,
    period: str,
    section_path: str,
    amount: str,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": (
            f"{company_name} {period} {section_path} "
            f"营业收入 {amount} 元"
        ),
        "metadata": {
            "doc_id": doc_id,
            "kb_id": "fictional-finance-kb",
            "company_name": company_name,
            "company_aliases": f"{company_name}|{company_name[:2]}",
            "report_period": period,
            "is_table": True,
            "section_path": section_path,
            "financial_metrics": "营业收入",
        },
        "rrf_score": 0.01,
    }


def test_financial_route_supplements_core_table_after_non_core_revenue_hit(
    monkeypatch,
):
    """A same-doc core revenue table must survive an early non-core hit.

    The strict search returns only the non-core block.  The core block exists
    solely in the retriever corpus, so this fails if the route short-circuits
    as soon as the revenue group is marked covered.
    """

    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=_NoopEmbeddings(), top_k=5)
    kb_id = "fictional-finance-kb"
    company = "甲方星河科技有限公司"
    doc_id = "fictional-half-year-report"

    non_core = _financial_hit(
        "non-core-revenue",
        doc_id=doc_id,
        company_name=company,
        period="2026年半年度",
        section_path="经营情况讨论与分析",
        amount="811,111.11",
    )
    core = _financial_hit(
        "core-revenue",
        doc_id=doc_id,
        company_name=company,
        period="2026年半年度",
        section_path="主要会计数据",
        amount="9,222,333.44",
    )
    wrong_period = _financial_hit(
        "wrong-period-revenue",
        doc_id=doc_id,
        company_name=company,
        period="2025年半年度",
        section_path="主要会计数据",
        amount="7,000,000.00",
    )
    other_company = _financial_hit(
        "other-company-revenue",
        doc_id="other-company-report",
        company_name="乙方远山制造有限公司",
        period="2026年半年度",
        section_path="主要会计数据",
        amount="88,888,888.88",
    )
    ordinary_recall = {
        "chunk_id": "ordinary-recall-placeholder",
        "text": f"{company} 2026年半年度 公司概况与经营范围。",
        "metadata": {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "company_name": company,
            "company_aliases": f"{company}|{company[:2]}",
            "report_period": "2026年半年度",
            "is_table": False,
            "section_path": "公司概况",
        },
        "rrf_score": 0.8,
    }
    retriever._corpus[kb_id] = [
        non_core,
        core,
        wrong_period,
        other_company,
        ordinary_recall,
    ]

    calls: list[dict] = []

    def fake_search_one(query, *, top_k, kb_id, metadata_filters):
        calls.append(
            {
                "query": query,
                "top_k": top_k,
                "kb_id": kb_id,
                "metadata_filters": metadata_filters,
            }
        )
        # Strict structured recall sees the wrong/non-core table first.
        return [non_core]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)

    query = f"{company} 2026年半年度营业收入是多少"
    merged = retriever._merge_financial_route(
        [(0.8, 0, ordinary_recall)],
        query=query,
        top_k=5,
        kb_id=kb_id,
        metadata_filters=None,
    )

    merged_ids = [hit[2]["chunk_id"] for hit in merged]
    assert "core-revenue" in merged_ids
    assert merged_ids.index("core-revenue") < merged_ids.index("non-core-revenue")
    assert "wrong-period-revenue" not in merged_ids
    assert "other-company-revenue" not in merged_ids
    assert calls, "the strict structured search should be exercised"
