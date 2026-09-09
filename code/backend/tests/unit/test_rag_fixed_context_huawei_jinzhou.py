"""Huawei/Jinzhou 固定上下文离线归因单测。

本文件只使用固定候选、FakeLLM 和 FakeEmbedding；不会访问真实模型、向量库、
服务、网络或外部数据库。测试意图是把召回/精排、生成、后置核验三个阶段
拆开，记录当前生产图在固定输入下的确定性行为。
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from services.kb.qa_graph import (
    _answer_verification_reasons,
    _verify_numeric_claims,
    build_qa_graph,
    run_qa,
)
from services.kb.reranker import NoopReranker
from services.kb.retriever import HybridRetriever
from tests.mocks import FakeEmbedding, FakeLLM


HUAWEI_QUESTION = "吉林华微电子2024年营业收入是多少？"
JINZHOU_QUESTION = (
    "锦州港2024年年报披露的2025年经营计划中，计划营业收入是多少？"
    "该计划是否构成业绩承诺？"
)

HUAWEI_REVENUE = "2,057,608,183.78"
JINZHOU_PLAN = "公司拟定2025年度计划实现营业收入17.21亿元"
JINZHOU_DISCLAIMER = "上述经营计划并不构成公司对投资者的业绩承诺"


class _FixedVectorStore:
    """只为 qa_graph 提供类型兼容；固定候选不经过 vector_store。"""

    def get_chunk_by_id(self, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        raise AssertionError("fixed-context test unexpectedly requested a neighbor/parent")


def _candidate(
    *,
    chunk_id: str,
    text: str,
    score: float,
    page: int,
    facts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "kb_id": "cninfo_report",
        "doc_id": "ST华微-吉林华微电子股份有限公司2024年年度报告.pdf"
        if chunk_id.startswith("huawei")
        else "ST锦港-锦州港股份有限公司2024年年度报告.pdf",
        "doc_title": "ST华微-吉林华微电子股份有限公司2024年年度报告.pdf"
        if chunk_id.startswith("huawei")
        else "ST锦港-锦州港股份有限公司2024年年度报告.pdf",
        "source_type": "cninfo_report",
        "page": page,
        "chunk_index": 0,
        "report_period": "2024年度",
    }
    if facts is not None:
        metadata["financial_facts_json"] = json.dumps(
            facts, ensure_ascii=False, separators=(",", ":")
        )
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": score,
        "metadata": metadata,
    }


def _install_fixed_retrieval(monkeypatch: pytest.MonkeyPatch, candidates: list[dict[str, Any]]) -> None:
    """把生产图的检索入口固定为给定候选，避免触发 embedding/vector_store。"""

    def fixed_search_queries(
        _self: Any,
        _queries: list[str] | tuple[str, ...],
        *,
        top_k: int | None = None,
        kb_id: str = "default",
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        assert kb_id == "cninfo_report"
        assert metadata_filters is None
        limit = len(candidates) if top_k is None else min(top_k, len(candidates))
        return [
            {
                **item,
                "metadata": dict(item.get("metadata") or {}),
            }
            for item in candidates[:limit]
        ]

    monkeypatch.setattr(HybridRetriever, "search_queries", fixed_search_queries)


def _run_fixed_graph(
    monkeypatch: pytest.MonkeyPatch,
    *,
    question: str,
    candidates: list[dict[str, Any]],
    initial_answer: str,
    top_k: int,
) -> tuple[dict[str, Any], FakeLLM]:
    _install_fixed_retrieval(monkeypatch, candidates)
    llm = FakeLLM(
        route={
            "客服意图分类器": "kb_question",
            "仅基于以下资料": initial_answer,
            "RAG 质量评估员": "10 10",
        }
    )
    graph = build_qa_graph(
        llm=llm,
        embeddings=FakeEmbedding(),
        vector_store=_FixedVectorStore(),
        top_k=top_k,
        recall_k=max(top_k, len(candidates)),
        reranker=NoopReranker(),
    )
    result = run_qa(graph, question=question, kb_id="cninfo_report")
    return result, llm


def _huawei_candidate() -> dict[str, Any]:
    text = f"主要会计数据\n营业收入 {HUAWEI_REVENUE}"
    return _candidate(
        chunk_id="huawei-revenue-fixed",
        text=text,
        score=0.95,
        page=6,
        facts=[
            {
                "metric": "revenue",
                "raw_value": HUAWEI_REVENUE,
                "canonical_value": HUAWEI_REVENUE,
                "unit": "元",
                "report_period": "2024年度",
                "statement_scope": "consolidated",
            }
        ],
    )


def test_huawei_fixed_context_binds_revenue_and_graph_passes(monkeypatch: pytest.MonkeyPatch):
    """同一固定证据下，营业收入和精确值可绑定，生产图稳定放行。"""

    initial_answer = f"2024年营业收入为{HUAWEI_REVENUE}元。[1]"
    evidence = [_huawei_candidate()]
    supported, unsupported = _verify_numeric_claims(
        initial_answer,
        evidence,
        question=HUAWEI_QUESTION,
    )
    reasons = _answer_verification_reasons(
        HUAWEI_QUESTION,
        initial_answer,
        evidence,
    )

    assert supported is True
    assert unsupported == []
    assert reasons == []

    result, llm = _run_fixed_graph(
        monkeypatch,
        question=HUAWEI_QUESTION,
        candidates=evidence,
        initial_answer=initial_answer,
        top_k=1,
    )

    assert HUAWEI_REVENUE in result["answer"]
    assert "营业收入" in result["answer"]
    assert result["retries"] == 0
    assert result["score"] == 10
    assert len(result["citations"]) == 1
    generation_prompt = next(prompt for prompt in llm.prompts if "仅基于以下资料" in prompt)
    assert "营业收入" in generation_prompt
    assert HUAWEI_REVENUE in generation_prompt


def test_huawei_wrong_initial_answer_is_deterministically_refused_after_verification(
    monkeypatch: pytest.MonkeyPatch,
):
    """固定证据不变时，错误数值由后置核验确定性拦截并最终拒答。"""

    initial_answer = "2024年营业收入为2,057,608,183.79元。[1]"
    evidence = [_huawei_candidate()]
    supported, unsupported = _verify_numeric_claims(
        initial_answer,
        evidence,
        question=HUAWEI_QUESTION,
    )
    reasons = _answer_verification_reasons(
        HUAWEI_QUESTION,
        initial_answer,
        evidence,
    )

    assert supported is False
    assert [claim.raw for claim in unsupported] == ["2,057,608,183.79元"]
    assert reasons

    result, _llm = _run_fixed_graph(
        monkeypatch,
        question=HUAWEI_QUESTION,
        candidates=evidence,
        initial_answer=initial_answer,
        top_k=1,
    )

    assert result["retries"] == 1
    assert result["score"] == 0
    assert "字段与数字无法可靠对应" in result["answer"]
    assert HUAWEI_REVENUE not in result["answer"]


def _jinzhou_candidates() -> list[dict[str, Any]]:
    return [
        _candidate(
            chunk_id="jinzhou-plan-fixed",
            text=JINZHOU_PLAN,
            score=0.86,
            page=25,
        ),
        _candidate(
            chunk_id="jinzhou-disclaimer-fixed",
            text=JINZHOU_DISCLAIMER,
            score=0.80,
            page=25,
        ),
    ]


def test_jinzhou_rerank_context_retains_plan_and_non_promise_fields(
    monkeypatch: pytest.MonkeyPatch,
):
    """固定双段上下文经过生产精排入口后，两个字段都进入生成上下文。"""

    initial_answer = (
        "计划营业收入为17.21亿元；上述经营计划并不构成公司对投资者的业绩承诺。[1][2]"
    )
    result, llm = _run_fixed_graph(
        monkeypatch,
        question=JINZHOU_QUESTION,
        candidates=_jinzhou_candidates(),
        initial_answer=initial_answer,
        top_k=2,
    )

    selected_text = "\n".join(item["text"] for item in result["contexts"])
    assert JINZHOU_PLAN in selected_text
    assert JINZHOU_DISCLAIMER in selected_text
    assert "17.21亿元" in result["answer"]
    assert JINZHOU_DISCLAIMER in result["answer"]
    assert result["retries"] == 0
    assert result["score"] == 10
    assert len(result["citations"]) == 2
    generation_prompt = next(prompt for prompt in llm.prompts if "仅基于以下资料" in prompt)
    assert JINZHOU_PLAN in generation_prompt
    assert JINZHOU_DISCLAIMER in generation_prompt


def test_jinzhou_missing_plan_is_not_accepted_as_complete(monkeypatch: pytest.MonkeyPatch):
    """没有经营计划数字时，不能仅凭免责声明把整题当作完整通过。"""

    disclaimer_only = [_jinzhou_candidates()[1]]
    initial_answer = "该计划不构成对投资者的业绩承诺。[1]"
    result, _llm = _run_fixed_graph(
        monkeypatch,
        question=JINZHOU_QUESTION,
        candidates=disclaimer_only,
        initial_answer=initial_answer,
        top_k=1,
    )

    assert result["retries"] == 1
    assert result["score"] == 0
    assert "字段与数字无法可靠对应" in result["answer"]


def test_jinzhou_missing_non_promise_field_cannot_pass(
    monkeypatch: pytest.MonkeyPatch,
):
    """缺少免责声明时，即使数字在场，也不应把单字段答案当完整通过。"""

    plan_only = [_jinzhou_candidates()[0]]
    initial_answer = "计划营业收入为17.21亿元。[1]"
    result, _llm = _run_fixed_graph(
        monkeypatch,
        question=JINZHOU_QUESTION,
        candidates=plan_only,
        initial_answer=initial_answer,
        top_k=1,
    )

    assert result["retries"] == 1
    assert result["score"] == 0
    assert "字段与数字无法可靠对应" in result["answer"]


def test_single_field_revenue_does_not_trigger_plan_commitment_gate():
    """普通单字段营业收入问题不受计划/业绩承诺窄规则影响。"""

    answer = f"2024年营业收入为{HUAWEI_REVENUE}元。[1]"
    assert _answer_verification_reasons(
        HUAWEI_QUESTION,
        answer,
        [_huawei_candidate()],
    ) == []


def test_jinzhou_gate_accepts_natural_synonyms_but_not_actual_performance_as_commitment():
    """接受“经营目标/不会形成”等同义表达，不把“实际业绩”当承诺结论。"""

    evidence = _jinzhou_candidates()
    natural_answer = "2025年经营目标为17.21亿元；该计划不会形成对投资者的业绩承诺。[1][2]"
    assert _answer_verification_reasons(
        JINZHOU_QUESTION,
        natural_answer,
        evidence,
    ) == []

    actual_performance_answer = "2025年经营目标为17.21亿元；实际业绩为17.21亿元。[1]"
    assert _answer_verification_reasons(
        JINZHOU_QUESTION,
        actual_performance_answer,
        evidence,
    )


def test_jinzhou_gate_rejects_commitment_polarity_conflict():
    """答案称“构成”而证据称“不构成”时，不能通过双字段门禁。"""

    conflicting_answer = "计划营业收入为17.21亿元；该经营计划构成业绩承诺。[1][2]"
    reasons = _answer_verification_reasons(
        JINZHOU_QUESTION,
        conflicting_answer,
        _jinzhou_candidates(),
    )
    assert "同语义支持" in "；".join(reasons)


def test_huawei_total_revenue_wording_passes_when_evidence_declares_only_revenue(
    monkeypatch: pytest.MonkeyPatch,
):
    """问“营业收入”而答案写“营业总收入”时，只要证据没把营业总收入当作
    独立字段披露，就不应判成字段错配。

    回归背景：8-31 的 24/36 基线里该题答案就是“营业总收入”并通过；Longyu
    严格字段绑定合入后，答案措辞引入 `total_revenue` 指标键，与证据侧
    `revenue` 严格不匹配，正确数字被拒（targeted4 0/4 的确证回归）。
    """

    initial_answer = f"2024年度营业总收入为{HUAWEI_REVENUE}元。[1]"
    evidence = [_huawei_candidate()]
    supported, unsupported = _verify_numeric_claims(
        initial_answer,
        evidence,
        question=HUAWEI_QUESTION,
    )
    assert supported is True
    assert unsupported == []
    assert _answer_verification_reasons(
        HUAWEI_QUESTION, initial_answer, evidence
    ) == []

    result, _llm = _run_fixed_graph(
        monkeypatch,
        question=HUAWEI_QUESTION,
        candidates=evidence,
        initial_answer=initial_answer,
        top_k=1,
    )

    assert HUAWEI_REVENUE in result["answer"]
    assert result["retries"] == 0
    assert result["score"] == 10


def test_total_revenue_wording_with_wrong_number_is_still_refused():
    """措辞回落不等于放宽：写成营业总收入但数字错误仍然拒绝。"""

    wrong = "2,057,608,183.79"
    answer = f"2024年度营业总收入为{wrong}元。[1]"
    supported, unsupported = _verify_numeric_claims(
        answer,
        [_huawei_candidate()],
        question=HUAWEI_QUESTION,
    )

    assert supported is False
    assert [claim.raw for claim in unsupported] == [f"{wrong}元"]


def test_revenue_question_still_rejects_main_business_revenue_value():
    """Longyu 门禁保持：证据同时披露两个收入字段时，借主营业务收入的值仍被拒。"""

    main_business = "1,111,111,111.11"
    evidence = [
        _candidate(
            chunk_id="huawei-two-revenue-fields",
            text=f"主要会计数据\n营业收入 {HUAWEI_REVENUE}\n主营业务收入 {main_business}",
            score=0.9,
            page=6,
        )
    ]

    borrowed = f"2024年营业收入为{main_business}元。[1]"
    assert _verify_numeric_claims(borrowed, evidence, question=HUAWEI_QUESTION)[
        0
    ] is False

    correct = f"2024年营业收入为{HUAWEI_REVENUE}元。[1]"
    assert _verify_numeric_claims(correct, evidence, question=HUAWEI_QUESTION)[
        0
    ] is True
