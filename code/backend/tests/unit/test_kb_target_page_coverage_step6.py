"""第 6 步：跨页指标候选覆盖与安全边界的离线契约。"""

from __future__ import annotations

from typing import Any

import pytest

import services.kb.retriever as retriever_module
from services.kb.retriever import HybridRetriever, _financial_route_hit_matches_group

KB_ID = "synthetic-target-coverage-kb"
COMPANY_NAME = "星岚智造股份有限公司"
DOC_ID = "synthetic-xinglan-2024-report"
OTHER_DOC_ID = "synthetic-other-2024-report"


class EchoEmbedding:
    """让 fake vector store 可以按 query 文本记录调用，但不调用模型。"""

    def embed_query(self, query: str) -> str:
        return query


class RankedMemoryVectorStore:
    """按夹具顺序截断的只读 fake store，模拟真实 Top-K 漏掉低排名块。"""

    def __init__(self, items: list[dict[str, Any]]):
        self.items = items
        self.calls: list[dict[str, Any]] = []

    def mutation_seq(self, kb_id: str) -> int:
        return 0

    def all_items(self, *, kb_id: str) -> list[dict[str, Any]]:
        assert kb_id == KB_ID
        return self.items

    def search(
        self,
        query_embedding: str,
        *,
        top_k: int = 5,
        where: dict[str, Any] | None = None,
        kb_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {"query": query_embedding, "top_k": top_k, "where": where, "kb_id": kb_id}
        )
        candidates = [
            item
            for item in self.items
            if (not kb_id or item["metadata"].get("kb_id") == kb_id)
            and retriever_module._metadata_matches(item.get("metadata"), where)
        ]
        return [
            {**item, "score": 1.0 - index * 0.001}
            for index, item in enumerate(candidates[:top_k])
        ]


def _hit(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = DOC_ID,
    period: str = "2024年度",
    section: str = "主要会计数据",
    is_table: bool = True,
    metrics: str | None = "营业收入",
    company: str = COMPANY_NAME,
    title: str | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "kb_id": KB_ID,
            "doc_id": doc_id,
            "doc_title": title
            or f"星岚智造-{COMPANY_NAME}2024年年度报告全文",
            "company_name": company,
            "report_period": period,
            "section_path": section,
            "is_table": is_table,
            "financial_metrics": metrics,
        },
    }


def _retriever(
    monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]], *, top_k: int = 5
) -> tuple[HybridRetriever, RankedMemoryVectorStore]:
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    store = RankedMemoryVectorStore(items)
    retriever = HybridRetriever(store, embeddings=EchoEmbedding(), top_k=top_k)
    # 保留真实 search_queries/_merge_financial_route 路径，只关闭 fake corpus 的 BM25
    # 贡献，使夹具中的“低排名”只由 fake vector store 的 Top-K 控制。
    retriever._corpus[KB_ID] = items
    retriever._bm25[KB_ID] = None
    retriever._seq_snapshot[KB_ID] = store.mutation_seq(KB_ID)
    return retriever, store


def _result_ids(result: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("chunk_id")) for item in result}


def test_low_rank_same_report_metric_block_is_recovered_without_cross_document_pollution(
    monkeypatch: pytest.MonkeyPatch,
):
    decoys = [
        _hit(
            f"synthetic-decoy-{index}",
            f"{COMPANY_NAME} 2024年度主要会计数据的其他披露块 {index}",
            metrics="",
        )
        for index in range(40)
    ]
    target = _hit(
        "synthetic-revenue-target",
        f"{COMPANY_NAME} 2024年度营业收入为123.45元。",
        metrics="营业收入",
    )
    wrong_period = _hit(
        "synthetic-revenue-wrong-period",
        f"{COMPANY_NAME} 2023年度营业收入为98.76元。",
        period="2023年度",
        title=f"星岚智造-{COMPANY_NAME}2023年年度报告全文",
    )
    other_company = _hit(
        "synthetic-other-revenue",
        "另一虚构公司2024年度营业收入为777.77元。",
        doc_id=OTHER_DOC_ID,
        company="另一虚构公司股份有限公司",
        title="另一虚构公司股份有限公司2024年年度报告全文",
    )
    retriever, store = _retriever(monkeypatch, decoys + [target, wrong_period, other_company])

    result = retriever.search_queries(
        ["星岚智造2024年营业收入是多少"], top_k=5, kb_id=KB_ID
    )

    # 目标块位于原始/严格 Top-K 之外；修复必须在已锁定 doc_id 内有界补回。
    assert target["chunk_id"] in _result_ids(result)
    assert any(call["top_k"] > retriever_module._FIN_ROUTE_TOP_N for call in store.calls)
    assert wrong_period["chunk_id"] not in _result_ids(result)
    assert other_company["chunk_id"] not in _result_ids(result)


def test_explicit_discussion_section_allows_period_inheritance_for_narrative_metric_block(
    monkeypatch: pytest.MonkeyPatch,
):
    table_noise = [
        _hit(
            f"synthetic-section-noise-{index}",
            f"{COMPANY_NAME} 2024年度表格说明块 {index}",
            metrics="",
        )
        for index in range(5)
    ]
    narrative_target = _hit(
        "synthetic-narrative-revenue",
        f"{COMPANY_NAME} 2024年经营情况讨论与分析：营业总收入为234.56元。",
        period="",
        section="经营情况讨论与分析 > 经营成果",
        is_table=False,
        metrics=None,
    )
    wrong_period = _hit(
        "synthetic-narrative-wrong-period",
        f"{COMPANY_NAME} 2023年经营情况讨论与分析：营业总收入为111.11元。",
        period="2023年度",
        section="经营情况讨论与分析 > 经营成果",
        is_table=False,
        metrics=None,
        title=f"星岚智造-{COMPANY_NAME}2023年年度报告全文",
    )
    other_company = _hit(
        "synthetic-narrative-other-company",
        "另一虚构公司2024年经营情况讨论与分析：营业总收入为888.88元。",
        doc_id=OTHER_DOC_ID,
        company="另一虚构公司股份有限公司",
        period="",
        section="经营情况讨论与分析",
        is_table=False,
        metrics=None,
        title="另一虚构公司股份有限公司2024年年度报告全文",
    )
    retriever, _store = _retriever(
        monkeypatch, table_noise + [narrative_target, wrong_period, other_company]
    )

    result = retriever.search_queries(
        ["星岚智造2024年经营情况讨论与分析章节中的营业总收入是多少"],
        top_k=5,
        kb_id=KB_ID,
    )

    assert narrative_target["chunk_id"] in _result_ids(result)
    assert wrong_period["chunk_id"] not in _result_ids(result)
    assert other_company["chunk_id"] not in _result_ids(result)


def test_without_section_intent_table_only_route_does_not_inject_narrative_block(
    monkeypatch: pytest.MonkeyPatch,
):
    table_noise = [
        _hit(
            f"synthetic-no-section-noise-{index}",
            f"{COMPANY_NAME} 2024年度普通财务说明块 {index}",
            metrics="",
        )
        for index in range(5)
    ]
    narrative_target = _hit(
        "synthetic-narrative-without-intent",
        f"{COMPANY_NAME} 2024年经营情况讨论与分析：营业总收入为345.67元。",
        period="",
        section="经营情况讨论与分析 > 经营成果",
        is_table=False,
        metrics=None,
    )
    retriever, _store = _retriever(monkeypatch, table_noise + [narrative_target])

    result = retriever.search_queries(
        ["星岚智造2024年营业总收入是多少"], top_k=5, kb_id=KB_ID
    )

    assert narrative_target["chunk_id"] not in _result_ids(result)


def test_missing_rd_total_is_recovered_but_rd_expense_and_other_documents_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    revenue = _hit(
        "synthetic-revenue-for-multi-metric",
        f"{COMPANY_NAME} 2024年度营业收入为456.78元。",
        metrics="营业收入",
    )
    rd_expense_decoys = [
        _hit(
            f"synthetic-rd-expense-{index}",
            f"{COMPANY_NAME} 2024年度研发费用为{200 + index}.00元。",
            section="研发投入",
            metrics="研发费用",
        )
        for index in range(40)
    ]
    rd_total = _hit(
        "synthetic-rd-total-target",
        f"{COMPANY_NAME} 2024年度研发投入合计为567.89元。",
        section="研发投入",
        metrics="研发投入合计",
    )
    other_rd_total = _hit(
        "synthetic-other-rd-total",
        "另一虚构公司2024年度研发投入合计为999.99元。",
        doc_id=OTHER_DOC_ID,
        company="另一虚构公司股份有限公司",
        title="另一虚构公司股份有限公司2024年年度报告全文",
        section="研发投入",
        metrics="研发投入合计",
    )
    retriever, _store = _retriever(
        monkeypatch, [revenue, *rd_expense_decoys, rd_total, other_rd_total], top_k=5
    )

    result = retriever.search_queries(
        ["星岚智造2024年营业收入和研发投入合计分别是多少"],
        top_k=5,
        kb_id=KB_ID,
    )
    ids = _result_ids(result)

    assert revenue["chunk_id"] in ids
    assert rd_total["chunk_id"] in ids
    assert _financial_route_hit_matches_group(rd_expense_decoys[0], "rd_total") is False
    assert other_rd_total["chunk_id"] not in ids
