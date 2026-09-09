"""混合检索测试：中文分词 + kb_id 隔离 + hybrid vs vector 对比。"""
from __future__ import annotations

import pytest

import services.kb.retriever as retriever_module
from services.kb.retriever import (
    HybridRetriever,
    VectorOnlyRetriever,
    _candidate_rank_adjustment,
    _query_profile,
    _tokenize,
)
from services.kb.retriever import (
    _company_query_clues as production_company_query_clues,
)
from services.kb.retriever import (
    _financial_route_doc_ids as production_financial_route_doc_ids,
)
from services.kb.vector_store import VectorStore
from tests.mocks import FakeEmbedding


def test_tokenize_chinese_2gram():
    toks = _tokenize("采购超过五万元必须招投标")
    assert "招投" in toks and "投标" in toks  # 2-gram 拆分
    assert "采购" in toks  # 整词保留


def test_hybrid_retriever_kb_isolation(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(["公司采购超过5万元必须招投标"]),
        texts=["公司采购超过5万元必须招投标"],
        doc_id="d1", doc_title="采购制度", source_type="md",
        chunk_indices=[0], kb_id="default",
    )
    store.add_chunks(
        embeddings=emb.embed_texts(["产品保修期为2年"]),
        texts=["产品保修期为2年"],
        doc_id="d2", doc_title="产品手册", source_type="md",
        chunk_indices=[0], kb_id="product",
    )

    retriever = HybridRetriever(store, embeddings=emb, top_k=3)

    hits_default = retriever.search("招投标", kb_id="default")
    assert hits_default
    assert all(h["metadata"]["kb_id"] == "default" for h in hits_default)

    # 跨库查：BM25 只在该库索引上跑，向量检索也带 kb_id 过滤
    hits_product = retriever.search("招投标", kb_id="product")
    assert all(h["metadata"]["kb_id"] == "product" for h in hits_product)
    assert len(hits_product) <= 1  # product 库只有保修期内容，最多 1 个候选


def test_hybrid_vs_vector_precise_match(tmp_path):
    """hybrid（BM25+向量+RRF）对精确数字/专有名词命中率 >= 纯向量。

    场景：语料含"5万元"精确词，hybrid 的 BM25 能精确命中；
    纯向量靠语义近似，FakeEmbedding 下未必召回。证明 RRF 融合的价值。
    """
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(["单笔采购金额超过5万元必须公开招投标"]),
        texts=["单笔采购金额超过5万元必须公开招投标"],
        doc_id="d1", doc_title="采购制度", source_type="md",
        chunk_indices=[0], kb_id="default",
    )

    hybrid = HybridRetriever(store, embeddings=emb, top_k=3)
    vec_only = VectorOnlyRetriever(store, embeddings=emb, top_k=3)

    # 精确数字查询：hybrid 必命中（BM25 精确匹配"5万元"）
    h_hits = hybrid.search("5万元", kb_id="default")
    assert h_hits, "hybrid 应命中 5万元"
    assert "5万元" in h_hits[0]["text"]

    # 纯向量也至少能召回（单库只有一条，必然命中）
    v_hits = vec_only.search("5万元", kb_id="default")
    assert v_hits, "vector 也应召回（单条语料）"

    # 关键断言：hybrid 命中数 >= vector（RRF 融合不会比纯向量差）
    assert len(h_hits) >= len(v_hits)


def test_multi_query_keeps_original_deduplicates_and_diversifies(monkeypatch):
    """多查询的第一条原问题不能丢失，重复 chunk 只保留一份并轮转文档。"""
    retriever = HybridRetriever(
        object(), embeddings=FakeEmbedding(), top_k=4, max_per_doc=1
    )
    calls = []
    hits_by_query = {
        "原始问题 2026": [
            {"chunk_id": "a1", "text": "原始证据", "metadata": {"doc_id": "doc-a"}, "rrf_score": 1.0},
            {"chunk_id": "a2", "text": "同文档第二块", "metadata": {"doc_id": "doc-a"}, "rrf_score": 0.9},
        ],
        "安全改写": [
            {"chunk_id": "a1", "text": "原始证据", "metadata": {"doc_id": "doc-a"}, "rrf_score": 1.0},
            {"chunk_id": "b1", "text": "另一文档证据", "metadata": {"doc_id": "doc-b"}, "rrf_score": 0.8},
        ],
    }

    def fake_search_one(query, **kwargs):
        calls.append(query)
        return hits_by_query[query]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)
    results = retriever.search_queries(["原始问题 2026", "安全改写"], top_k=4)

    assert calls == ["原始问题 2026", "安全改写"]
    assert [hit["chunk_id"] for hit in results] == ["a1", "b1"]
    assert len({hit["chunk_id"] for hit in results}) == len(results)
    assert all(sum(1 for hit in results if hit["metadata"]["doc_id"] == doc) <= 1 for doc in {"doc-a", "doc-b"})


def test_metadata_filter_applies_to_vector_and_bm25(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(["采购制度 PDF 2026"]),
        texts=["采购制度 PDF 2026"],
        doc_id="pdf-doc", doc_title="PDF 制度", source_type="pdf",
        chunk_indices=[0], kb_id="default",
    )
    store.add_chunks(
        embeddings=emb.embed_texts(["采购制度 Markdown 2026"]),
        texts=["采购制度 Markdown 2026"],
        doc_id="md-doc", doc_title="Markdown 制度", source_type="md",
        chunk_indices=[0], kb_id="default",
    )

    retriever = HybridRetriever(store, embeddings=emb, top_k=5)
    hits = retriever.search(
        "采购制度 2026", kb_id="default", metadata_filters={"source_type": "pdf"}
    )

    assert hits
    assert all(hit["metadata"]["source_type"] == "pdf" for hit in hits)
    assert all(hit["metadata"]["kb_id"] == "default" for hit in hits)


def test_min_score_does_not_fill_empty_vector_results_but_keeps_bm25(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    store.add_chunks(
        embeddings=emb.embed_texts(["精确编号 ABC-2026"]),
        texts=["精确编号 ABC-2026"],
        doc_id="d1", doc_title="编号", source_type="md",
        chunk_indices=[0], kb_id="default",
    )
    store.add_chunks(
        embeddings=emb.embed_texts(["普通制度说明"]),
        texts=["普通制度说明"],
        doc_id="d2", doc_title="制度", source_type="md",
        chunk_indices=[0], kb_id="default",
    )
    store.add_chunks(
        embeddings=emb.embed_texts(["另一份流程说明"]),
        texts=["另一份流程说明"],
        doc_id="d3", doc_title="流程", source_type="md",
        chunk_indices=[0], kb_id="default",
    )

    # 极高门槛会移除向量命中，但精确关键词仍应由 BM25 送入重排。
    retriever = HybridRetriever(store, embeddings=emb, top_k=3, min_score=1.1)
    hits = retriever.search("ABC-2026", kb_id="default")
    assert len(hits) == 1
    assert hits[0]["score"] == 0.0

    empty = retriever.search("完全不存在的词", kb_id="default")
    assert empty == []


def test_financial_rerank_prefers_exact_fact_and_consolidated_scope(monkeypatch):
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=3)
    hits = [
        {
            "chunk_id": "parent-table",
            "text": "| 项目 | 营业收入 |\n| --- | --- |\n| 本期 | 100 |",
            "metadata": {
                "doc_id": "d1",
                "financial_metrics": "营业收入",
                "statement_scope": "parent",
                "is_table": True,
            },
            "rrf_score": 0.0200,
        },
        {
            "chunk_id": "consolidated-fact",
            "text": "| 项目 | 营业收入 |\n| --- | --- |\n| 本期 | 100 |",
            "metadata": {
                "doc_id": "d1",
                "financial_metrics": "营业收入",
                "statement_scope": "consolidated",
                "is_table": True,
            },
            "rrf_score": 0.0190,
        },
        {
            "chunk_id": "unknown-text",
            "text": "营业收入情况说明。",
            "metadata": {"doc_id": "d1", "statement_scope": "unknown", "is_table": False},
            "rrf_score": 0.0205,
        },
    ]
    monkeypatch.setattr(retriever, "_search_one", lambda *_args, **_kwargs: hits)

    results = retriever.search("2024年营业收入", kb_id="default")

    assert [hit["chunk_id"] for hit in results] == [
        "consolidated-fact",
        "parent-table",
        "unknown-text",
    ]
    profile = _query_profile("2024年营业收入")
    assert _candidate_rank_adjustment(hits[1], profile) > _candidate_rank_adjustment(hits[0], profile)
    assert _candidate_rank_adjustment(hits[0], profile) > _candidate_rank_adjustment(hits[2], profile)


def test_reason_query_prefers_narrative_over_numeric_table(monkeypatch):
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=2)
    hits = [
        {
            "chunk_id": "numeric-table",
            "text": "| 营业收入 | 100 | 90 |",
            "metadata": {"doc_id": "d1", "is_table": True, "financial_metrics": "营业收入"},
            "rrf_score": 0.0220,
        },
        {
            "chunk_id": "reason-body",
            "text": "营业收入同比下降，主要系呼吸系统用药销售额减少。",
            "metadata": {"doc_id": "d1", "is_table": False, "section_path": "经营情况概述"},
            "rrf_score": 0.0200,
        },
    ]
    monkeypatch.setattr(retriever, "_search_one", lambda *_args, **_kwargs: hits)

    results = retriever.search("营业收入下降的原因", kb_id="default")

    assert [hit["chunk_id"] for hit in results] == ["reason-body", "numeric-table"]


def test_section_path_signal_is_deterministic_and_limited(monkeypatch):
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=2)
    hits = [
        {
            "chunk_id": "other-section",
            "text": "营业收入 100",
            "metadata": {"doc_id": "d1", "section_path": "财务摘要", "is_table": True},
            "rrf_score": 0.0210,
        },
        {
            "chunk_id": "target-section",
            "text": "营业收入 100",
            "metadata": {"doc_id": "d1", "section_path": "经营情况概述", "is_table": True},
            "rrf_score": 0.0200,
        },
    ]
    monkeypatch.setattr(retriever, "_search_one", lambda *_args, **_kwargs: hits)

    results = retriever.search("经营情况概述 营业收入", kb_id="default")

    assert [hit["chunk_id"] for hit in results] == ["target-section", "other-section"]


def _financial_route_test_hit(
    chunk_id: str,
    doc_id: str,
    company_name: str,
    *,
    section_path: str = "主要会计数据",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"{company_name} 2024年度 {section_path} 营业收入 123,456.78元",
        "metadata": {
            "doc_id": doc_id,
            "kb_id": "fictional-finance-kb",
            "company_name": company_name,
            "company_aliases": f"{company_name}|{company_name[:2]}",
            "report_period": "2024年度",
            "is_table": True,
            "section_path": section_path,
        },
        "rrf_score": 0.01,
    }


def _financial_route_doc_ids(metadata_filters: dict) -> set[str]:
    clauses = metadata_filters["$and"]
    doc_clause = next(clause for clause in clauses if "doc_id" in clause)
    return set(doc_clause["doc_id"]["$in"])


def test_financial_route_filters_explicit_company_doc_ids(monkeypatch):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=8)
    kb_id = "fictional-finance-kb"
    alpha_name = "甲方科技股份有限公司"
    beta_name = "乙方制造有限公司"
    alpha = _financial_route_test_hit("alpha-main", "alpha-doc-main", alpha_name)
    alpha_extra = _financial_route_test_hit(
        "alpha-extra", "alpha-doc-extra", alpha_name, section_path="主要财务指标"
    )
    beta = _financial_route_test_hit("beta-main", "beta-doc-main", beta_name)
    retriever._corpus[kb_id] = [alpha, alpha_extra, beta]
    ordered = [(0.8, 0, alpha), (0.7, 1, beta)]
    calls = []

    def fake_search_one(query, *, top_k, kb_id, metadata_filters):
        calls.append(
            {
                "query": query,
                "top_k": top_k,
                "kb_id": kb_id,
                "metadata_filters": metadata_filters,
            }
        )
        return [alpha]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)
    retriever._merge_financial_route(
        ordered,
        query=f"{alpha_name} 2024年营业收入是多少",
        top_k=8,
        kb_id=kb_id,
        metadata_filters={"kb_id": kb_id},
    )

    assert len(calls) == 1
    assert calls[0]["top_k"] == 8
    assert calls[0]["kb_id"] == kb_id
    filters = calls[0]["metadata_filters"]
    assert filters["$and"]
    assert {clause["report_period"] for clause in filters["$and"] if "report_period" in clause} == {
        "2024年度"
    }
    assert {clause["is_table"] for clause in filters["$and"] if "is_table" in clause} == {True}
    assert {
        section
        for clause in filters["$and"]
        if "section_path" in clause
        for section in clause["section_path"]["$in"]
    } == {"主要会计数据", "主要财务指标"}
    assert _financial_route_doc_ids(filters) <= {"alpha-doc-main", "alpha-doc-extra"}
    assert "alpha-doc-main" in _financial_route_doc_ids(filters)
    assert {"kb_id": kb_id} in filters["$and"]


def test_financial_route_excludes_other_company_from_structured_injection(monkeypatch):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=8)
    kb_id = "fictional-finance-kb"
    alpha_name = "甲方科技股份有限公司"
    beta_name = "乙方制造有限公司"
    alpha_ranked = _financial_route_test_hit("alpha-ranked", "alpha-doc", alpha_name)
    beta_ranked = _financial_route_test_hit("beta-ranked", "beta-doc", beta_name)
    alpha_structured = _financial_route_test_hit("alpha-structured", "alpha-doc", alpha_name)
    beta_structured = _financial_route_test_hit("beta-structured", "beta-doc", beta_name)
    retriever._corpus[kb_id] = [alpha_ranked, beta_ranked]
    ordered = [(0.99, 0, beta_ranked), (0.10, 1, alpha_ranked)]
    calls = []

    def fake_search_one(query, *, top_k, kb_id, metadata_filters):
        calls.append(metadata_filters)
        allowed_doc_ids = _financial_route_doc_ids(metadata_filters)
        return [
            hit
            for hit in (alpha_structured, beta_structured)
            if hit["metadata"]["doc_id"] in allowed_doc_ids
        ]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)
    merged = retriever._merge_financial_route(
        ordered,
        query=f"{alpha_name} 2024年营业收入是多少",
        top_k=8,
        kb_id=kb_id,
        metadata_filters=None,
    )

    assert len(calls) == 1
    assert _financial_route_doc_ids(calls[0]) == {"alpha-doc"}
    original_chunk_ids = {hit["chunk_id"] for _, _, hit in ordered}
    injected = [
        hit
        for _, _, hit in merged
        if hit["chunk_id"] not in original_chunk_ids
    ]
    assert injected
    assert {hit["metadata"]["doc_id"] for hit in injected} == {"alpha-doc"}
    assert all(hit["metadata"]["doc_id"] != "beta-doc" for hit in injected)


def test_financial_route_ambiguous_subject_returns_ordered_without_structured_search(monkeypatch):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=8)
    kb_id = "fictional-finance-kb"
    alpha = _financial_route_test_hit("alpha", "alpha-doc", "甲方科技股份有限公司")
    beta = _financial_route_test_hit("beta", "beta-doc", "乙方制造有限公司")
    retriever._corpus[kb_id] = [alpha, beta]
    ordered = [(0.9, 0, beta), (0.8, 1, alpha)]
    calls = []

    def fail_if_called(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("ambiguous company must not call structured _search_one")

    monkeypatch.setattr(retriever, "_search_one", fail_if_called)
    result = retriever._merge_financial_route(
        ordered,
        query="2024年营业收入是多少",
        top_k=8,
        kb_id=kb_id,
        metadata_filters=None,
    )

    assert calls == []
    assert result is ordered


def test_financial_route_exception_returns_ordered(monkeypatch):
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=8)
    kb_id = "fictional-finance-kb"
    alpha = _financial_route_test_hit("alpha", "alpha-doc", "甲方科技股份有限公司")
    beta = _financial_route_test_hit("beta", "beta-doc", "乙方制造有限公司")
    retriever._corpus[kb_id] = [alpha, beta]
    ordered = [(0.9, 0, beta), (0.8, 1, alpha)]
    calls = []

    def raise_search_error(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("synthetic structured-channel failure")

    monkeypatch.setattr(retriever, "_search_one", raise_search_error)
    result = retriever._merge_financial_route(
        ordered,
        query="甲方科技 2024年营业收入是多少",
        top_k=8,
        kb_id=kb_id,
        metadata_filters=None,
    )

    assert len(calls) == 1
    assert result is ordered


def _real_financial_route_item(
    doc_id: str,
    doc_title: str,
    company_name: str,
    *,
    body: str = "主要会计数据：营业收入 123,456.78 元。",
    report_period: str = "2024年度",
    stock_code: str = "",
) -> dict:
    """构造接近真实财报分块的 all_items 项；doc_id 使用测试专用稳定值。"""
    metadata = {
        "doc_id": doc_id,
        "doc_title": doc_title,
        "file_path": f"data_kb_test/cninfo/{doc_title}.pdf",
        "company_name": company_name,
        "report_period": report_period,
    }
    if stock_code:
        metadata["stock_code"] = stock_code
    return {
        "text": f"{company_name} {report_period}\n{body}",
        "metadata": metadata,
    }


def _real_financial_route_corpus() -> list[dict]:
    """五道真实题对应的全文/摘要，加上会造成污染的无关公司文档。"""
    return [
        _real_financial_route_item(
            "fixture-huawei-full",
            "ST华微-吉林华微电子股份有限公司2024年年度报告",
            "吉林华微电子股份有限公司",
            stock_code="600360",
        ),
        _real_financial_route_item(
            "fixture-huawei-summary",
            "ST华微-吉林华微电子股份有限公司2024年年度报告摘要",
            "吉林华微电子股份有限公司",
            stock_code="600360",
        ),
        _real_financial_route_item(
            "fixture-zhongcheng-full",
            "ST中程-2024年年度报告",
            "青岛中资中程股份有限公司",
            body="公司简称为青岛中程；主要会计数据：归母净利润 -310,302,902.32 元。",
            stock_code="300208",
        ),
        _real_financial_route_item(
            "fixture-zhongcheng-summary",
            "ST中程-2024年年度报告摘要",
            "青岛中资中程股份有限公司",
            body="公司简称为青岛中程；摘要列示归母净利润 -310,302,902.32 元。",
            stock_code="300208",
        ),
        _real_financial_route_item(
            "fixture-gree-full",
            "格力电器-2023年年度报告",
            "珠海格力电器股份有限公司",
            body="2023 年主要会计数据：营业收入 203,979,266,387.09 元。",
            report_period="2023年度",
            stock_code="000651",
        ),
        _real_financial_route_item(
            "fixture-gree-summary",
            "格力电器-2023年年度报告摘要",
            "珠海格力电器股份有限公司",
            body="摘要列示 2023 年营业收入 203,979,266,387.09 元。",
            report_period="2023年度",
            stock_code="000651",
        ),
        _real_financial_route_item(
            "fixture-furun-full",
            "ST富润-2024年年度报告",
            "浙江富润数字科技股份有限公司",
        ),
        _real_financial_route_item(
            "fixture-furun-summary",
            "ST富润-2024年年度报告摘要",
            "浙江富润数字科技股份有限公司",
        ),
        _real_financial_route_item(
            "fixture-guangdao-full",
            "ST广道-2025年半年度报告",
            "深圳市广道数字技术股份有限公司",
            body=(
                "本报告期营业收入 23,357,748.41 元。正文比较了 ST龙宇、凯利泰等公司的披露。"
            ),
            report_period="2025年半年度",
        ),
        _real_financial_route_item(
            "fixture-guangdao-summary",
            "ST广道-2025年半年度报告摘要",
            "深圳市广道数字技术股份有限公司",
            body=(
                "摘要列示营业收入 23,357,748.41 元；行业比较部分提到上海龙宇数据股份有限公司。"
            ),
            report_period="2025年半年度",
        ),
        _real_financial_route_item(
            "fixture-longyu-noise",
            "ST龙宇-上海龙宇数据股份有限公司2024年年度报告",
            "上海龙宇数据股份有限公司",
        ),
        _real_financial_route_item(
            "fixture-kailitai-noise",
            "凯利泰-2024年年度报告",
            "上海凯利泰医疗科技股份有限公司",
        ),
    ]


@pytest.mark.parametrize(
    ("query", "expected_doc_ids"),
    [
        (
            "吉林华微电子2024年营业收入是多少？",
            {"fixture-huawei-full", "fixture-huawei-summary"},
        ),
        (
            "青岛中程2024年归属于上市公司股东的净利润是多少？",
            {"fixture-zhongcheng-full", "fixture-zhongcheng-summary"},
        ),
        (
            "青岛中资中程2024年归属于上市公司股东的净利润是多少？",
            {"fixture-zhongcheng-full", "fixture-zhongcheng-summary"},
        ),
        (
            "格力电器2023年年报主要会计数据中，营业收入是多少？",
            {"fixture-gree-full", "fixture-gree-summary"},
        ),
        (
            "ST富润2024年营业收入是多少？",
            {"fixture-furun-full", "fixture-furun-summary"},
        ),
        (
            "ST广道2025年半年度报告全文和摘要都列出的本报告期营业收入与归母净利润分别是多少？",
            {"fixture-guangdao-full", "fixture-guangdao-summary"},
        ),
    ],
)
def test_financial_route_real_report_names_select_only_matching_doc_ids(
    query: str, expected_doc_ids: set[str]
):
    selected = set(production_financial_route_doc_ids(query, _real_financial_route_corpus()))

    assert selected == expected_doc_ids


def test_financial_route_does_not_treat_other_company_body_mentions_as_subject_ambiguity():
    query = "ST广道2025年半年度报告全文和摘要都列出的本报告期营业收入与归母净利润分别是多少？"
    corpus = [
        item
        for item in _real_financial_route_corpus()
        if item["metadata"]["doc_id"]
        in {"fixture-guangdao-full", "fixture-guangdao-summary", "fixture-longyu-noise", "fixture-kailitai-noise"}
    ]

    selected = set(production_financial_route_doc_ids(query, corpus))

    assert selected == {"fixture-guangdao-full", "fixture-guangdao-summary"}
    assert "fixture-longyu-noise" not in selected
    assert "fixture-kailitai-noise" not in selected


def test_financial_route_without_company_subject_returns_empty():
    query = "2024年营业收入是多少？"

    assert production_company_query_clues(query) == []
    assert production_financial_route_doc_ids(query, _real_financial_route_corpus()) == ()


def test_financial_route_conflicting_strong_company_clues_returns_empty():
    query = "ST华微 300208 2024年营业收入是多少？"
    corpus = _real_financial_route_corpus()

    assert production_financial_route_doc_ids(query, corpus) == ()


def test_financial_route_fallback_keeps_doc_boundary_and_covers_incomplete_sources(
    monkeypatch,
):
    """严格过滤为空时，元数据缺失的全文/摘要仍按 doc_id 逐个兜底。"""
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=5)
    kb_id = "fictional-guangdao-kb"
    full_doc_id = "fixture-guangdao-full"
    summary_doc_id = "fixture-guangdao-summary"

    full_cover = {
        "chunk_id": "guangdao-full-cover",
        "text": "ST广道-2025年半年度报告",
        "metadata": {
            "doc_id": full_doc_id,
            "doc_title": "ST广道-2025年半年度报告",
            "company_name": "深圳市广道数字技术股份有限公司",
            "report_period": "2025年半年度",
            "section_path": "董监高异议及声明",
            "is_table": False,
            "financial_metrics": "",
        },
    }
    full_target = {
        "chunk_id": "guangdao-full-target",
        "text": (
            "ST广道 2025年半年度报告\n"
            "营业收入 23,357,748.41 元；"
            "归属于上市公司股东的净利润 -20,631,376.11 元。"
        ),
        "metadata": {
            "doc_id": full_doc_id,
            "doc_title": "ST广道-2025年半年度报告",
            "company_name": "深圳市广道数字技术股份有限公司",
            "report_period": "2025年半年度",
            "section_path": "董监高异议及声明",
            "is_table": True,
            "financial_metrics": "营业收入|归属于上市公司股东的净利润",
        },
    }
    summary_cover = {
        "chunk_id": "guangdao-summary-cover",
        "text": "ST广道-2025年半年度报告摘要",
        "metadata": {
            "doc_id": summary_doc_id,
            "doc_title": "ST广道-2025年半年度报告摘要",
            "report_period": "",
            "section_path": "",
            "is_table": False,
            "financial_metrics": None,
        },
    }
    summary_target = {
        "chunk_id": "guangdao-summary-target",
        "text": (
            "ST广道 2025年半年度报告摘要\n"
            "营业收入 23,357,748.41 元；归母净利润 -20,631,376.11 元。"
        ),
        "metadata": {
            "doc_id": summary_doc_id,
            "doc_title": "ST广道-2025年半年度报告摘要",
            "report_period": "",
            "section_path": "",
            "is_table": False,
            "financial_metrics": None,
        },
    }
    other_company = {
        "chunk_id": "longyu-high-score",
        "text": "ST龙宇 2024年年度报告 主要会计数据 营业收入 999,999,999.99 元。",
        "metadata": {
            "doc_id": "fixture-longyu-noise",
            "doc_title": "ST龙宇-上海龙宇数据股份有限公司2024年年度报告",
            "company_name": "上海龙宇数据股份有限公司",
            "section_path": "主要会计数据",
        },
    }
    retriever._corpus[kb_id] = [
        full_cover,
        full_target,
        summary_cover,
        summary_target,
        other_company,
    ]
    ordered = [(0.99, 0, other_company)]
    calls = []

    def fake_search_one(query, *, top_k, kb_id, metadata_filters):
        calls.append({"top_k": top_k, "metadata_filters": metadata_filters})
        clauses = metadata_filters["$and"]
        if any("section_path" in clause for clause in clauses):
            return []
        requested_doc_id = next(
            clause["doc_id"]["$in"][0]
            for clause in clauses
            if "doc_id" in clause
        )
        # 模拟底层返回高分噪声和无数字封面；实现必须二次锁定 doc_id 并做内容门禁。
        return [
            other_company,
            full_cover,
            full_target,
            summary_cover,
            summary_target,
        ] if requested_doc_id in {full_doc_id, summary_doc_id} else []

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)
    query = "ST广道2025年半年度报告全文和摘要都列出的本报告期营业收入与归母净利润分别是多少？"
    merged = retriever._merge_financial_route(
        ordered,
        query=query,
        top_k=5,
        kb_id=kb_id,
        metadata_filters=None,
    )

    original_chunk_ids = {hit["chunk_id"] for _, _, hit in ordered}
    injected = [
        hit
        for _, _, hit in merged
        if hit["chunk_id"] not in original_chunk_ids
    ]
    assert len(calls) == 3  # 一次严格检索 + 全文/摘要各一次兜底
    assert calls[1]["top_k"] > calls[0]["top_k"]
    assert {
        clause["doc_id"]["$in"][0]
        for call in calls[1:]
        for clause in call["metadata_filters"]["$and"]
        if "doc_id" in clause
    } == {full_doc_id, summary_doc_id}
    assert {hit["metadata"]["doc_id"] for hit in injected} == {
        full_doc_id,
        summary_doc_id,
    }
    assert {hit["chunk_id"] for hit in injected} == {
        "guangdao-full-target",
        "guangdao-summary-target",
    }
    assert "guangdao-full-cover" not in {hit["chunk_id"] for hit in injected}
    assert "guangdao-summary-cover" not in {hit["chunk_id"] for hit in injected}
    assert "longyu-high-score" not in {hit["chunk_id"] for hit in injected}


# ---------------------------------------------------------------------------
# 定向6/6 与全量36题不一致归因（2026-09-03）配套防护
#
# 归因结论是 A 类（全量链路漏传 KB_STRUCTURED_FIN_ROUTE），B 类（连续运行的
# 缓存/状态/顺序污染）已被最小实验排除。下面三项测试把"排除"固化成断言：
# 一旦将来真的引入跨请求状态，这里会立刻亮红灯，而不用再等 36 题跑完。
# ---------------------------------------------------------------------------


def test_repeated_search_returns_identical_results_across_requests(tmp_path):
    """同一问题单独查与"被别的查询插在中间"时，结果必须逐项一致。"""
    store = VectorStore(persist_dir=str(tmp_path))
    emb = FakeEmbedding()
    text = "柯利达2024年归属于上市公司股东的净利润为8,583,076.40元"
    store.add_chunks(
        embeddings=emb.embed_texts([text]),
        texts=[text],
        doc_id="d1",
        doc_title="柯利达2024年年度报告",
        source_type="pdf",
        chunk_indices=[0],
        kb_id="default",
    )
    retriever = HybridRetriever(store, embeddings=emb, top_k=3)
    question = "柯利达2024年净利润是多少"

    solo = retriever.search(question, kb_id="default")
    # 中间夹一次完全不同的查询，模拟全量评测里的"前序题"
    retriever.search("员工差旅费报销标准", kb_id="default")
    again = retriever.search(question, kb_id="default")

    assert solo, "单独运行应能召回候选"
    assert [hit["chunk_id"] for hit in solo] == [hit["chunk_id"] for hit in again]
    assert [round(hit.get("score", 0.0), 9) for hit in solo] == [
        round(hit.get("score", 0.0), 9) for hit in again
    ]


def test_prior_queries_do_not_mutate_corpus_or_candidates(monkeypatch):
    """连续跑多道题后，语料快照与候选结果都不能被前一个请求改动。"""
    import copy

    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=8)
    kb_id = "fictional-finance-kb"
    alpha_name = "甲方科技股份有限公司"
    beta_name = "乙方制造有限公司"
    alpha = _financial_route_test_hit("alpha-main", "alpha-doc", alpha_name)
    beta = _financial_route_test_hit("beta-main", "beta-doc", beta_name)
    retriever._corpus[kb_id] = [alpha, beta]
    frozen = copy.deepcopy(retriever._corpus)

    monkeypatch.setattr(
        retriever, "_search_one", lambda *_a, **_k: [(0.9, 0, alpha)]
    )

    ordered = [(0.8, 0, alpha), (0.7, 1, beta)]
    question = f"{alpha_name} 2024年营业收入是多少"

    first = retriever._merge_financial_route(
        list(ordered),
        query=question,
        top_k=8,
        kb_id=kb_id,
        metadata_filters={"kb_id": kb_id},
    )
    # 插入另两家公司的查询，模拟全量评测中前面的题
    for other in (beta_name, "丙方贸易有限公司"):
        retriever._merge_financial_route(
            list(ordered),
            query=f"{other} 2023年净利润是多少",
            top_k=8,
            kb_id=kb_id,
            metadata_filters={"kb_id": kb_id},
        )
    second = retriever._merge_financial_route(
        list(ordered),
        query=question,
        top_k=8,
        kb_id=kb_id,
        metadata_filters={"kb_id": kb_id},
    )

    assert [hit[2]["chunk_id"] for hit in first] == [
        hit[2]["chunk_id"] for hit in second
    ]
    # 语料被前序请求写入是最典型的污染方式，必须与快照逐一相等
    assert retriever._corpus == frozen


def test_structured_route_keeps_period_and_subject_filters(monkeypatch):
    """结构化路由只提升"块级定位"，不得绕过期间/主体/口径过滤。

    开启路由是为了让装着正确数字的块进入候选；它绝不能让期间不符或
    主体不符的块混进来 —— 否则就是把安全门禁打开了。
    """
    monkeypatch.setattr(retriever_module, "_structured_fin_route", True)
    retriever = HybridRetriever(object(), embeddings=FakeEmbedding(), top_k=8)
    kb_id = "fictional-finance-kb"
    target_name = "甲方科技股份有限公司"
    other_name = "乙方制造有限公司"
    right_period = _financial_route_test_hit("alpha-2024", "alpha-doc", target_name)
    wrong_period = _financial_route_test_hit("alpha-2023", "alpha-doc", target_name)
    wrong_period["metadata"]["report_period"] = "2023年度"
    other_company = _financial_route_test_hit("beta-2024", "beta-doc", other_name)
    retriever._corpus[kb_id] = [right_period, wrong_period, other_company]

    captured: list[dict] = []

    def fake_search_one(query, *, top_k, kb_id, metadata_filters):
        captured.append(metadata_filters)
        return [(0.9, 0, right_period)]

    monkeypatch.setattr(retriever, "_search_one", fake_search_one)

    retriever._merge_financial_route(
        [(0.8, 0, right_period)],
        query=f"{target_name} 2024年营业收入是多少",
        top_k=8,
        kb_id=kb_id,
        metadata_filters={"kb_id": kb_id},
    )

    assert captured, "结构化路由应发起一次过滤检索"
    clauses = captured[0]["$and"]
    # 期间必须是问句里的 2024年度，2023 的块不得入选
    assert {
        clause["report_period"] for clause in clauses if "report_period" in clause
    } == {"2024年度"}
    # 主体只限甲方，乙方的块不得入选
    assert _financial_route_doc_ids(captured[0]) == {"alpha-doc"}
