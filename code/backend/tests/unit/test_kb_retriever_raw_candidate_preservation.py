"""Tests for preserving high-ranked raw financial narratives during route merge."""

from __future__ import annotations

import services.kb.retriever as retriever_module
from services.kb.retriever import HybridRetriever, _dedupe_and_diversify
from tests.mocks import FakeEmbedding

KB_ID = "raw-preservation-kb"
DOC_ID = "doc-alpha"
OTHER_DOC_ID = "doc-beta"
PERIOD = "2024年度"
REVENUE = "营业收入"


def _hit(
    chunk_id: str,
    doc_id: str,
    text: str,
    *,
    is_table: bool,
    section_path: str = "七、主要会计数据",
    financial_metrics: str = REVENUE,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "doc_id": doc_id,
            "doc_title": f"{doc_id} {PERIOD}报告",
            "report_period": PERIOD,
            "is_table": is_table,
            "section_path": section_path,
            "financial_metrics": financial_metrics,
        },
    }


def test_raw_narrative_survives_structured_injection_and_doc_cap(monkeypatch):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=5, max_per_doc=12)

    raw_plan = _hit(
        "raw-plan",
        DOC_ID,
        "经营计划：计划营业收入17.21亿元。",
        is_table=False,
        section_path="六、公司关于未来发展的讨论与分析",
        financial_metrics="",
    )
    structured_target = _hit(
        "structured-target",
        DOC_ID,
        "主要会计数据 营业收入 123.45亿元",
        is_table=True,
    )
    structured_decoys = [
        _hit(
            f"structured-decoy-{index}",
            DOC_ID,
            f"主要会计数据 营业收入 {index + 1}.00亿元",
            is_table=True,
        )
        for index in range(31)
    ]
    structured_hits = [structured_target, *structured_decoys]
    other_company = _hit(
        "other-company-table",
        OTHER_DOC_ID,
        "主要会计数据 营业收入999.00亿元",
        is_table=True,
    )
    retriever._corpus[KB_ID] = [raw_plan, *structured_hits, other_company]

    monkeypatch.setattr(
        retriever_module,
        "_financial_route_doc_ids",
        lambda _query, _corpus: (DOC_ID,),
    )

    def fake_search_one(query, *, top_k, kb_id, metadata_filters):
        assert kb_id == KB_ID
        assert top_k == retriever_module._FIN_ROUTE_TOP_N
        return [*structured_hits, other_company]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)
    query = "某公司2024年经营计划营业收入是多少"
    ordered = [(0.9, 0, raw_plan)]

    merged = retriever._merge_financial_route(
        ordered,
        query=query,
        top_k=5,
        kb_id=KB_ID,
        metadata_filters=None,
    )
    merged_hits = [hit for _score, _sequence, hit in merged]
    assert len(structured_hits) == 32
    assert merged_hits[0]["chunk_id"] == "raw-plan"
    assert merged_hits[1]["chunk_id"] == "structured-target"

    deduped = _dedupe_and_diversify(merged_hits, max_per_doc=12)
    deduped_ids = [hit["chunk_id"] for hit in deduped]
    assert "raw-plan" in deduped_ids
    assert "structured-target" in deduped_ids
    assert "other-company-table" not in deduped_ids
    assert sum(hit["metadata"]["doc_id"] == DOC_ID for hit in deduped) <= 12

    repeated = _dedupe_and_diversify(merged_hits, max_per_doc=12)
    assert deduped_ids == [hit["chunk_id"] for hit in repeated]
