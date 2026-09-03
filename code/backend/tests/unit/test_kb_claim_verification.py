"""生成后数字/口径核验测试：纯函数优先，图链路只使用本地 fake。"""
from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal

import pytest

from services.kb import ingest
from services.kb.qa_graph import (
    _answer_verification_reasons,
    _cause_terms_from_evidence,
    _claim_with_context,
    _claim_with_metric,
    _extract_numeric_claims,
    _format_candidate_metadata,
    _format_context_block,
    _has_reliable_question_evidence,
    _metric_mentions,
    _normalize_report_period,
    _partial_numeric_fallback_answer,
    _prefers_consolidated_scope,
    _resolve_claim_metric,
    _retrieval_queries,
    _verification_evidence_items,
    _verification_evidence_texts,
    _verify_numeric_claims,
    build_qa_graph,
    run_qa,
    run_qa_stream,
)
from services.kb.retriever import HybridRetriever
from services.kb.vector_store import VectorStore
from tests.mocks import FakeEmbedding, FakeLLM


def _financial_graph(tmp_path, responses: list[str], evidence: str):
    store = VectorStore(persist_dir=str(tmp_path))
    embedding = FakeEmbedding()
    store.add_chunks(
        embeddings=embedding.embed_texts([evidence]),
        texts=[evidence],
        doc_id="guangdao-report",
        doc_title="广道集团 2025 年报",
        source_type="pdf",
        chunk_indices=[0],
        kb_id="default",
        extra_metadata=[
            {
                "page": 7,
                "page_start": 7,
                "section_path": "财务报告 > 主要财务数据",
                "table_name": "主要会计数据",
                "statement_scope": "合并",
                "report_period": "2025年度",
                "unit": "万元",
                "is_table": True,
            }
        ],
    )
    llm = FakeLLM(responses=responses, route={"RAG 质量评估员": "10 10"})
    graph = build_qa_graph(
        llm=llm,
        embeddings=embedding,
        vector_store=store,
        top_k=1,
        recall_k=1,
        metadata_filters={"source_type": "pdf"},
    )
    return graph, llm


def _legacy_window_record(
    text: str,
    chunk_index: int,
    *,
    doc_id: str = "legacy-doc",
    kb_id: str = "default",
    report_period: str = "2024年度",
    unit: str = "万元",
    is_table: bool = True,
) -> dict:
    """构造旧 Chroma 相邻块；测试不依赖 Chroma 客户端或真实库。"""
    return {
        "text": text,
        "metadata": {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "chunk_index": chunk_index,
            "previous_chunk_id": f"{doc_id}-{chunk_index - 1}",
            "next_chunk_id": f"{doc_id}-{chunk_index + 1}",
            "report_period": report_period,
            "unit": unit,
            "is_table": is_table,
        },
    }


def _legacy_window_evidence(
    answer: str,
    records: list[dict],
) -> list[dict]:
    """按候选中心块调用现有证据入口，模拟旧库窗口输出。"""
    return _verification_evidence_items(
        answer,
        [records[0]],
        ["\n".join(record["text"] for record in records)],
        [records],
    )


def test_numeric_claims_support_exact_formatting_and_decimal_units():
    ok, unsupported = _verify_numeric_claims(
        "金额为39,389,053.48元",
        ["金额为39 389 053.48元"],
    )
    assert ok is True
    assert unsupported == []
    claim = _extract_numeric_claims("金额为３９，３８９，０５３．４８元")[0]
    assert claim.number == Decimal("39389053.48")

    converted, _ = _verify_numeric_claims("规模为1.2亿", ["规模为12000万"])
    not_converted, unsupported = _verify_numeric_claims(
        "金额为39,389,053.48",
        ["金额为3,938.91万"],
    )
    assert converted is True
    assert not_converted is False
    assert unsupported[0].raw == "39,389,053.48"


def test_numeric_claims_do_not_float_round_or_cross_percent_units():
    assert _verify_numeric_claims("增长5%", ["增长5％"])[0] is True
    assert _verify_numeric_claims("增长5%", ["增长0.05"])[0] is False
    assert _verify_numeric_claims("金额为100.00万元", ["金额为100万元"])[0] is True
    assert _verify_numeric_claims("金额为100.01万元", ["金额为100万元"])[0] is False


def test_multi_metric_answer_keeps_operating_cashflow_on_its_own_line():
    facts = [
        {
            "metric": "revenue",
            "raw_value": "203,979,266,387.09",
            "canonical_value": "203979266387.09",
            "unit": "元",
            "statement_scope": "consolidated",
            "report_period": "2023年度",
        },
    ]
    answer = (
        "- 营业收入：203,979,266,387.09元\n"
        "- 归母净利润：29,017,387,604.18元\n"
        "- 经营活动现金流量净额：56,398,426,354.17元\n"
        "- 基本每股收益：5.22元"
    )
    question = (
        "格力电器2023年营业收入、归母净利润、经营活动现金流量净额和"
        "基本每股收益分别是多少？"
    )
    evidence = [
        {
            "text": (
                "项目\n2023年\n2022年\n本年比上年同期增减\n"
                "营业收入（元）\n203,979,266,387.09\n"
                "188,988,382,706.68\n7.93%\n"
                "归属于上市公司股东\n的净利润（元）\n"
                "29,017,387,604.18\n24,506,623,782.46\n18.41%\n"
                "经营活动产生的现金\n流量净额（元）\n"
                "56,398,426,354.17\n28,668,435,921.27\n96.73%\n"
                "基本每股收益（元/\n股）\n5.22\n4.43\n17.83%"
            ),
            "metadata": {
                "is_table": True,
                "financial_metrics": "revenue",
                "report_period": "2023年度",
                "statement_scope": "unknown",
                "financial_facts_json": json.dumps(facts, ensure_ascii=False),
            },
        }
    ]

    claims = _extract_numeric_claims(answer)
    assert [claim.metric for claim in claims] == [
        "revenue",
        "net_profit_attributable",
        "operating_cashflow",
        "eps",
    ]
    assert [claim.line_metric for claim in claims] == [
        "revenue",
        "net_profit_attributable",
        "operating_cashflow",
        "eps",
    ]
    assert claims[2].metric != "net_profit_attributable"
    evidence_claim = next(
        claim
        for claim in _extract_numeric_claims(evidence[0]["text"])
        if claim.canonical_value == claims[2].canonical_value
    )
    assert evidence_claim.metric == ""
    assert evidence_claim.line_metric == "operating_cashflow"
    bound_evidence_claim = _claim_with_context(
        _claim_with_metric(evidence_claim, ["revenue"]),
        evidence[0]["text"],
        question=question,
        metadata=evidence[0]["metadata"],
    )
    assert bound_evidence_claim.metric == "operating_cashflow"
    assert bound_evidence_claim.line_metric == "operating_cashflow"
    assert _verify_numeric_claims(answer, evidence, question=question)[0] is True
    assert _partial_numeric_fallback_answer(question, answer, evidence) is None


def test_metadata_metric_is_fallback_when_no_line_metric_exists():
    claim = _extract_numeric_claims("1,234元")[0]

    assert claim.metric == ""
    assert claim.line_metric == ""
    assert _claim_with_metric(claim, ["revenue"]).metric == "revenue"


def test_numeric_claim_without_same_line_label_keeps_neighbor_fallback():
    claims = _extract_numeric_claims("营业收入\n100万元")

    assert len(claims) == 1
    assert claims[0].metric == "revenue"
    assert claims[0].line_metric == "revenue"


def test_legacy_chroma_payload_uses_exact_text_without_financial_facts():
    evidence = [
        {
            "text": "单位：元\n报告期：2024年\n营业收入 948,565,984.05",
            "metadata": {
                "is_table": True,
                "report_period": "2024年",
                "unit": "元",
            },
        }
    ]

    assert _verify_numeric_claims(
        "营业收入为948,565,984.05元[1]",
        evidence,
        question="帕瓦2024年营业收入是多少？",
    )[0] is True


def test_legacy_chroma_adjacent_segments_keep_metric_period_and_unit_binding():
    contexts = [{"text": "扣除与主营业务无关的", "metadata": {}}]
    passages = ["扣除与主营业务无关的\n\n收入后的营业收入 110,682,912.05"]
    passage_records = [
        [
            {"text": "扣除与主营业务无关的", "metadata": {}},
            {
                "text": "收入后的营业收入 110,682,912.05",
                "metadata": {
                    "is_table": True,
                    "report_period": "2024年",
                    "unit": "元",
                },
            },
        ]
    ]

    evidence = _verification_evidence_items(
        "营业收入为110,682,912.05元[1]",
        contexts,
        passages,
        passage_records,
    )
    assert _verify_numeric_claims(
        "营业收入为110,682,912.05元[1]",
        evidence,
        question="ST富润2024年扣除与主营业务无关的收入后的营业收入是多少？",
    )[0] is True


def test_legacy_window_two_hops_still_binds_split_metric_and_value():
    records = [
        _legacy_window_record("营业收入", 10),
        _legacy_window_record("主要会计数据表格说明", 11),
        _legacy_window_record("100万元", 12),
    ]
    answer = "营业收入为100万元[1]"

    assert _verify_numeric_claims(
        answer,
        _legacy_window_evidence(answer, records),
        question="示例公司2024年营业收入是多少？",
    )[0] is True


def test_legacy_window_binds_with_consistent_neighbor_metadata_when_center_is_sparse():
    records = [
        _legacy_window_record("营业收入", 10, report_period="", unit=""),
        _legacy_window_record("主要会计数据表格说明", 11),
        _legacy_window_record("100万元", 12),
    ]
    for record in records[1:]:
        record["metadata"]["statement_scope"] = "consolidated"

    evidence = _legacy_window_evidence("营业收入为100万元[1]", records)

    assert _verify_numeric_claims(
        "营业收入为100万元[1]",
        evidence,
        question="示例公司2024年合并营业收入是多少？",
    )[0] is True


def test_partial_numeric_fallback_masks_only_the_unsupported_field_in_window():
    records = [
        _legacy_window_record("营业收入", 10, report_period="", unit=""),
        _legacy_window_record("100万元", 11),
    ]
    answer = "营业收入为100万元[1]，净利润为999万元[1]"
    question = "示例公司2024年营业收入和净利润分别是多少？"
    evidence = _legacy_window_evidence(answer, records)

    assert _partial_numeric_fallback_answer(question, answer, evidence) == (
        "营业收入为100万元[1],净利润为无法确定[1]"
    )


def test_legacy_window_does_not_mix_chunks_beyond_hard_hop_limit():
    # 两跳是允许的边界；第三跳的数字即使存在于旧库，也不能被窗口带入。
    records = [
        _legacy_window_record("营业收入", 10),
        _legacy_window_record("表头说明", 11),
        _legacy_window_record("无关列说明", 12),
        _legacy_window_record("100万元", 13),
    ]
    answer = "营业收入为100万元[1]"

    assert _verify_numeric_claims(
        answer,
        _legacy_window_evidence(answer, records),
        question="示例公司2024年营业收入是多少？",
    )[0] is False


def test_legacy_window_never_splices_different_doc_or_kb():
    answer = "营业收入为100万元[1]"
    question = "示例公司2024年营业收入是多少？"

    for overrides in (
        {"doc_id": "other-doc", "kb_id": "default"},
        {"doc_id": "legacy-doc", "kb_id": "other-kb"},
    ):
        records = [
            _legacy_window_record("营业收入", 10),
            _legacy_window_record("100万元", 11, **overrides),
        ]
        assert _verify_numeric_claims(
            answer,
            _legacy_window_evidence(answer, records),
            question=question,
        )[0] is False


def test_legacy_adjacent_number_without_metric_label_is_not_evidence():
    records = [
        _legacy_window_record("主要会计数据", 10),
        _legacy_window_record("100万元", 11),
    ]
    answer = "营业收入为100万元[1]"

    assert _verify_numeric_claims(
        answer,
        _legacy_window_evidence(answer, records),
        question="示例公司2024年营业收入是多少？",
    )[0] is False


@pytest.mark.xfail(
    strict=False,
    reason="策略记录：多期间字段为空时的旧窗口行为暂不作为阶段阻塞门槛",
)
def test_legacy_multi_period_window_does_not_guess_unqualified_period():
    records = [
        _legacy_window_record("2024年度 营业收入", 10, report_period=""),
        _legacy_window_record("100万元", 11, report_period=""),
        _legacy_window_record("2023年度 营业收入", 12, report_period=""),
        _legacy_window_record("90万元", 13, report_period=""),
    ]
    answer = "营业收入为100万元[1]"

    assert _verify_numeric_claims(
        answer,
        _legacy_window_evidence(answer, records),
        question="示例公司营业收入是多少？",
    )[0] is False


@pytest.mark.xfail(
    strict=False,
    reason="策略记录：营业收入/营业总收入互认边界暂不作为阶段阻塞门槛",
)
def test_legacy_window_keeps_units_and_nearby_metrics_strict():
    evidence = [
        {
            "text": "营业收入100万元；营业总收入100万元；营业收入增长5%",
            "metadata": {
                "doc_id": "legacy-doc",
                "kb_id": "default",
                "report_period": "2024年度",
                "unit": "万元",
                "is_table": True,
            },
        }
    ]

    assert not _verify_numeric_claims(
        "营业收入为100元[1]",
        evidence,
        question="示例公司2024年营业收入是多少？",
    )[0]
    assert not _verify_numeric_claims(
        "营业收入增长0.05[1]",
        evidence,
        question="示例公司2024年营业收入增长率是多少？",
    )[0]
    assert not _verify_numeric_claims(
        "营业总收入为100万元[1]",
        evidence,
        question="示例公司2024年营业总收入是多少？",
    )[0]


def test_rounded_coarse_unit_is_not_exact_legacy_evidence():
    evidence = [
        {
            "text": "营业收入 94,856.60万元",
            "metadata": {
                "is_table": True,
                "report_period": "2024年",
                "unit": "万元",
            },
        }
    ]

    assert not _verify_numeric_claims(
        "营业收入为948,565,984.05元[1]",
        evidence,
        question="帕瓦2024年营业收入是多少？",
    )[0]


def test_legacy_exact_text_does_not_allow_fabricated_number():
    evidence = [
        {
            "text": "营业收入 100万元",
            "metadata": {"is_table": True, "report_period": "2024年", "unit": "万元"},
        }
    ]

    supported, unsupported = _verify_numeric_claims(
        "营业收入为999万元[1]",
        evidence,
        question="示例公司2024年营业收入是多少？",
    )
    assert supported is False
    assert [claim.raw for claim in unsupported] == ["999万元"]


def test_structured_facts_bind_metric_and_do_not_use_number_bag():
    metadata = {
        "doc_title": "鹏博士 2024 年报",
        "financial_facts_json": json.dumps(
            [
                {
                    "metric": "revenue",
                    "raw_value": "1,876,694,452.50",
                    "canonical_value": "1876694452.50",
                    "unit": "元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
                {
                    "metric": "rd_expense",
                    "raw_value": "74,381,245.99",
                    "canonical_value": "74381245.99",
                    "unit": "元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
                {
                    "metric": "revenue",
                    "raw_value": "2,606,047,231.66",
                    "canonical_value": "2606047231.66",
                    "unit": "元",
                    "statement_scope": "consolidated",
                    "report_period": "",
                },
            ],
            ensure_ascii=False,
        ),
    }
    evidence = [{"text": "营业收入 1,876,694,452.50；研发费用 74,381,245.99", "metadata": metadata}]
    question = "鹏博士2024年营业收入是多少？"

    assert _verify_numeric_claims(
        "营业收入为1,876,694,452.50元[1]", evidence, question=question
    )[0] is True
    assert _verify_numeric_claims(
        "营业收入为74,381,245.99元[1]", evidence, question=question
    )[0] is False
    assert _verify_numeric_claims(
        "营业收入为2,606,047,231.66元[1]", evidence, question=question
    )[0] is False

    # 旧索引没有结构化字段时，也只接受同一局部指标标签下的数字。
    plain_evidence = ["营业收入 1,876,694,452.50；研发费用 74,381,245.99"]
    assert _verify_numeric_claims(
        "营业收入为74,381,245.99元[1]", plain_evidence, question=question
    )[0] is False


def test_ingest_financial_metadata_contract_is_consumed_without_translation():
    table_text = (
        "合并利润表\n单位：万元\n报告期：2024年度\n"
        "项目 | 2024年 | 2023年\n"
        "营业收入 | 100.25 | 90.00\n"
        "研发费用 | 25.50 | 20.00"
    )
    facts = ingest._extract_financial_facts(
        table_text,
        is_table=True,
        statement_scope="consolidated",
        report_period="2024年度",
        unit="万元",
    )
    metadata = ingest._financial_metadata_fields(facts)
    evidence = [{"text": table_text, "metadata": metadata}]
    question = "示例公司2024年营业收入是多少？"

    assert _verify_numeric_claims(
        "营业收入为100.25万元[1]", evidence, question=question
    )[0] is True
    assert _verify_numeric_claims(
        "营业收入为25.50万元[1]", evidence, question=question
    )[0] is False


def test_pengbo_multiyear_ingest_facts_pass_period_aware_verification():
    table_text = (
        "| 主要会计数据 | 2024年 | 2023年 |  | 本期比上年同期增减(%) | 2022年 |  |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "|  |  | 调整后 | 调整前 |  | 调整后 | 调整前 |\n"
        "| 营业收入 | 1,876,694,452.50 | 2,606,047,231.66 | 2,606,047,231.66 | -27.99 | 3,704,914,217.80 | 3,704,914,217.80 |"
    )
    facts = ingest._extract_financial_facts(
        table_text,
        is_table=True,
        statement_scope="consolidated",
        report_period="2024年度",
        unit="元",
    )
    evidence = [
        {
            "text": table_text,
            "metadata": ingest._financial_metadata_fields(facts),
        }
    ]

    assert _verify_numeric_claims(
        "鹏博士2024年营业收入为1,876,694,452.50元[1]",
        evidence,
        question="鹏博士2024年营业收入是多少？",
    )[0] is True
    assert _verify_numeric_claims(
        "鹏博士2024年营业收入为2,606,047,231.66元[1]",
        evidence,
        question="鹏博士2024年营业收入是多少？",
    )[0] is False


def test_period_missing_structured_fact_falls_back_to_strict_metric_text():
    """真实候选的期间字段为空时，不能因有一条不完整事实就直接安全降级。"""
    evidence = [
        {
            "text": "营业收入 100万元；研发费用 20万元",
            "metadata": {
                "financial_facts_json": json.dumps(
                    [
                        {
                            "metric": "营业收入",
                            "raw_value": "100",
                            "canonical_value": "1000000",
                            "unit": "万元",
                            "statement_scope": "consolidated",
                            "report_period": "",
                        }
                    ],
                    ensure_ascii=False,
                )
            },
        }
    ]
    question = "示例公司2024年营业收入是多少？"

    assert _verify_numeric_claims("营业收入为100万元[1]", evidence, question=question)[0]
    assert not _verify_numeric_claims("营业收入为100.01万元[1]", evidence, question=question)[0]
    # 仍然不能借同一证据块中的另一个指标数字通过。
    assert not _verify_numeric_claims("营业收入为20万元[1]", evidence, question=question)[0]


def test_same_metric_period_scope_unit_binding_rejects_wrong_variants():
    facts = [
        {
            "metric": "营业收入",
            "raw_value": "100",
            "canonical_value": "1000000",
            "unit": "万元",
            "statement_scope": "consolidated",
            "report_period": "2024年度",
        },
        {
            "metric": "营业收入",
            "raw_value": "90",
            "canonical_value": "900000",
            "unit": "万元",
            "statement_scope": "consolidated",
            "report_period": "2023年度",
        },
        {
            "metric": "营业收入",
            "raw_value": "80",
            "canonical_value": "800000",
            "unit": "万元",
            "statement_scope": "parent",
            "report_period": "2024年度",
        },
    ]
    evidence = [
        {
            "text": "2024年营业收入100万元；2023年营业收入90万元；母公司2024年营业收入80万元",
            "metadata": {"financial_facts_json": json.dumps(facts, ensure_ascii=False)},
        }
    ]

    assert _verify_numeric_claims(
        "2024年营业收入为100万元[1]", evidence, question="公司2024年营业收入是多少？"
    )[0]
    assert not _verify_numeric_claims(
        "2024年营业收入为90万元[1]", evidence, question="公司2024年营业收入是多少？"
    )[0]
    assert _verify_numeric_claims(
        "母公司2024年营业收入为80万元[1]",
        evidence,
        question="公司母公司2024年营业收入是多少？",
    )[0]
    assert not _verify_numeric_claims(
        "母公司2024年营业收入为100万元[1]",
        evidence,
        question="公司母公司2024年营业收入是多少？",
    )[0]
    assert not _verify_numeric_claims(
        "2024年营业收入为100.00元[1]",
        evidence,
        question="公司2024年营业收入是多少？",
    )[0]


def test_dongshi_table_column_periods_do_not_turn_prior_year_into_current_year():
    """东时真实失败形态：结构化层暂把多列数字都写成同一期间。"""
    text = (
        "主要会计数据\n2024年\n2023年\n2022年\n本期比上年同期增减(%)\n"
        "营业收入\n"
        "807,388,662.39\n1,042,430,987.88\n-22.55\n1,000,176,579.15"
    )
    facts = [
        {
            "metric": "营业收入",
            "raw_value": value,
            "canonical_value": value.replace(",", ""),
            "unit": "元",
            "statement_scope": "consolidated",
            # 旧索引将本期/上期/变化列错误压成同一期间。
            "report_period": "2024年",
        }
        for value in ("807,388,662.39", "1,042,430,987.88", "-22.55")
    ]
    evidence = [{"text": text, "metadata": {"financial_facts_json": json.dumps(facts, ensure_ascii=False), "is_table": True, "unit": "元"}}]
    question = "ST东时2024年营业收入是多少？"

    assert _verify_numeric_claims("2024年营业收入为807,388,662.39元[1]", evidence, question=question)[0]
    assert not _verify_numeric_claims("2024年营业收入为1,042,430,987.88元[1]", evidence, question=question)[0]


def test_yitai_half_year_table_uses_metadata_period_and_change_column():
    """伊泰真实抽取形态：当前期/上年同期/变化列均跨行，结构化事实为空。"""
    text = (
        "内蒙古伊泰煤炭股份有限公司2025年半年度报告\n主要会计数据\n"
        "本报告期（1－6月）\n上年同期\n本报告期比上年同期增减(%)\n"
        "营业收入\n20,774,218,376.59\n24,940,919,250.91\n-16.71\n"
        "利润总额\n3,638,853,363.64\n5,346,541,657.75\n-31.94"
    )
    evidence = [{
        "text": text,
        "metadata": {
            "is_table": True,
            "unit": "元",
            "report_period": "2025年半年度",
            "statement_scope": "consolidated",
        },
    }]
    question = "伊泰2025年半年度营业收入是多少？"

    assert _verify_numeric_claims(
        "营业收入为20,774,218,376.59元[1]", evidence, question=question
    )[0]
    assert not _verify_numeric_claims(
        "营业收入为24,940,919,250.91元[1]", evidence, question=question
    )[0]
    assert _verify_numeric_claims(
        "营业收入同比下降16.71%[1]", evidence,
        question="伊泰2025年半年度营业收入同比变化原因？",
    )[0]
    assert not _verify_numeric_claims(
        "营业收入为20,774,218,376.59万元[1]", evidence, question=question
    )[0]


def test_multi_metric_and_cross_page_evidence_bind_each_number_to_its_metric():
    facts = [
        {
            "metric": "营业收入",
            "raw_value": "1,482,763,229.80",
            "canonical_value": "1482763229.80",
            "unit": "元",
            "statement_scope": "consolidated",
            "report_period": "2024年",
        },
        {
            "metric": "研发投入合计",
            "raw_value": "10,639,426.95",
            "canonical_value": "10639426.95",
            "unit": "元",
            "statement_scope": "consolidated",
            "report_period": "2024年",
        },
    ]
    evidence = [
        {"text": "营业收入 1,482,763,229.80元", "metadata": {"financial_facts_json": json.dumps(facts[:1], ensure_ascii=False)}},
        {"text": "研发投入合计 10,639,426.95元", "metadata": {"financial_facts_json": json.dumps(facts[1:], ensure_ascii=False)}},
    ]
    question = "海越能源2024年营业收入和研发投入合计分别是多少？"

    assert _verify_numeric_claims(
        "营业收入为1,482,763,229.80元，研发投入合计为10,639,426.95元[1][2]",
        evidence,
        question=question,
    )[0]
    assert not _verify_numeric_claims(
        "营业收入为10,639,426.95元，研发投入合计为1,482,763,229.80元[1][2]",
        evidence,
        question=question,
    )[0]


def test_multi_metric_partial_answer_keeps_supported_field_and_blocks_unknown_number():
    evidence = [
        {
            "text": "营业收入 100万元",
            "metadata": {
                "financial_facts_json": json.dumps(
                    [
                        {
                            "metric": "revenue",
                            "raw_value": "100",
                            "canonical_value": "1000000",
                            "unit": "万元",
                            "statement_scope": "consolidated",
                            "report_period": "2024年度",
                        }
                    ],
                    ensure_ascii=False,
                )
            },
        }
    ]
    question = "海越能源2024年营业收入和净利润分别是多少？"
    partial = "营业收入为100万元[1]，净利润无法确定。"

    assert _verify_numeric_claims(partial, evidence, question=question)[0] is True
    assert _answer_verification_reasons(question, partial, evidence) == []

    unsupported = "营业收入为100万元[1]，净利润为999万元[1]"
    supported, claims = _verify_numeric_claims(
        unsupported, evidence, question=question
    )
    assert supported is False
    assert [claim.raw for claim in claims] == ["999万元"]
    assert _answer_verification_reasons(question, unsupported, evidence)


def test_multi_metric_partial_answer_is_not_degraded_after_graph_evaluation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, _llm = _financial_graph(
        tmp_path,
        ["营业收入为100万元。[1]，净利润无法确定。"],
        "广道2025年营业收入为100万元",
    )

    result = run_qa(
        graph,
        question="广道2025年营业收入和净利润分别是多少？",
        kb_id="default",
    )

    assert result["retries"] == 0
    assert "营业收入为100万元" in result["answer"]
    assert "净利润无法确定" in result["answer"]


def test_multi_metric_retry_keeps_supported_field_after_unknown_field_is_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, _llm = _financial_graph(
        tmp_path,
        [
            "营业收入为100万元[1]，净利润为999万元[1]",
            "营业收入为100万元[1]，净利润无法确定。",
        ],
        "广道2025年营业收入为100万元",
    )

    result = run_qa(
        graph,
        question="广道2025年营业收入和净利润分别是多少？",
        kb_id="default",
    )

    assert result["retries"] == 1
    assert "营业收入为100万元" in result["answer"]
    assert "净利润无法确定" in result["answer"]
    assert "999万元" not in result["answer"]


def test_multi_metric_final_fallback_masks_only_unsupported_number(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, _llm = _financial_graph(
        tmp_path,
        [
            "营业收入为100万元[1]，净利润为999万元[1]",
            "营业收入为100万元[1]，净利润为999万元[1]",
        ],
        "广道2025年营业收入为100万元",
    )

    result = run_qa(
        graph,
        question="广道2025年营业收入和净利润分别是多少？",
        kb_id="default",
    )

    assert result["retries"] == 1
    assert "营业收入为100万元" in result["answer"]
    assert "净利润为无法确定" in result["answer"]
    assert "999万元" not in result["answer"]


def test_reason_question_uses_text_fallback_for_percent_and_keeps_wrong_units_rejected():
    evidence = [
        {
            "text": "营业收入较上年同期下降42.89%，主要系呼吸系统用药销售额减少所致。",
            "metadata": {
                "financial_facts_json": json.dumps(
                    [
                        {
                            "metric": "营业收入",
                            "raw_value": "-42.89",
                            "canonical_value": "-42.89",
                            # 结构化字段暂把增减率继承成了金额单位。
                            "unit": "元",
                            "statement_scope": "consolidated",
                            "report_period": "2025年半年度",
                        }
                    ],
                    ensure_ascii=False,
                )
            },
        }
    ]
    question = "葫芦娃2025年上半年营业收入同比下降42.89%的主要原因是什么？"
    answer = "营业收入较上年同期下降42.89%，主要系呼吸系统用药销售额减少所致。[1]"

    assert _verify_numeric_claims(answer, evidence, question=question)[0]
    assert _answer_verification_reasons(question, answer, evidence) == []
    assert not _verify_numeric_claims(
        "营业收入较上年同期下降42.90%，主要系呼吸系统用药销售额减少所致。[1]",
        evidence,
        question=question,
    )[0]
    assert not _verify_numeric_claims(
        "营业收入为42.89元[1]", evidence, question="葫芦娃2025年上半年营业收入是多少？"
    )[0]


def test_structured_facts_keep_rd_investment_and_rd_expense_distinct():
    metadata = {
        "financial_facts_json": json.dumps(
            [
                {
                    "metric": "研发费用",
                    "raw_value": "6,456,606.95",
                    "canonical_value": "6456606.95",
                    "unit": "元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
                {
                    "metric": "研发投入合计",
                    "raw_value": "10,639,426.95",
                    "canonical_value": "10639426.95",
                    "unit": "元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
            ],
            ensure_ascii=False,
        )
    }
    evidence = [{"text": "研发费用 6,456,606.95；研发投入合计 10,639,426.95", "metadata": metadata}]
    question = "海越2024年研发投入合计是多少？"

    assert _verify_numeric_claims(
        "研发投入合计为10,639,426.95元[1]", evidence, question=question
    )[0] is True
    assert _verify_numeric_claims(
        "研发投入合计为6,456,606.95元[1]", evidence, question=question
    )[0] is False


def test_financial_metrics_only_metadata_is_a_safe_old_index_fallback():
    evidence = [
        {
            "text": "1,876,694,452.50",
            "metadata": {"financial_metrics": "revenue"},
        }
    ]
    assert _verify_numeric_claims(
        "营业收入为1,876,694,452.50元[1]",
        evidence,
        question="鹏博士2024年营业收入是多少？",
    )[0] is True


def test_report_period_quarter_aliases_normalize_without_double_replacement():
    assert _normalize_report_period("2024年第一季度") == "2024年第一季度"
    assert _normalize_report_period("2024年一季度") == "2024年第一季度"


def test_retrieval_queries_split_metrics_reason_and_chapter_with_bound():
    question = "海越2024年经营情况概述中营业收入和研发投入合计分别是多少，变动原因是什么？"
    queries = _retrieval_queries(question)

    assert queries[0] == question
    assert len(queries) <= 8
    assert len(queries) == len(set(queries))
    assert any("指标：营业收入" in query for query in queries)
    assert any("指标：研发投入合计" in query for query in queries)
    assert any("主要系" in query and "变动原因" in query for query in queries)
    assert any("经营情况概述" in query and "营业收入" in query for query in queries)

    retry_queries = _retrieval_queries(question, retrieval_attempt=1)
    assert any("精确匹配" in query and "海越" in query and "2024" in query for query in retry_queries)


def test_low_score_exact_evidence_does_not_escalate_gate():
    contexts = [
        {
            "score": 0.01,
            "text": "华微2024年营业收入为100万元",
            "metadata": {"doc_title": "华微2024年报"},
        }
    ]
    assert _has_reliable_question_evidence(
        "华微2024年营业收入是多少？", contexts, [contexts[0]["text"]]
    ) is True


def test_low_score_exact_document_is_not_sent_to_human(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0.9")
    store = VectorStore(persist_dir=str(tmp_path))
    embedding = FakeEmbedding()
    llm = FakeLLM(
        route={
            "仅基于以下资料": "营业收入为100万元。[1]",
            "RAG 质量评估员": "10 10",
        }
    )

    def fake_search_queries(self, queries, **kwargs):
        return [
            {
                "chunk_id": "huawei-revenue",
                "text": "华微2024年营业收入为100万元",
                "metadata": {
                    "doc_id": "huawei-2024",
                    "doc_title": "华微2024年报",
                    "kb_id": "default",
                    "report_period": "2024年度",
                },
                "score": 0.01,
            }
        ]

    monkeypatch.setattr(HybridRetriever, "search_queries", fake_search_queries)
    graph = build_qa_graph(
        llm=llm,
        embeddings=embedding,
        vector_store=store,
        top_k=1,
        recall_k=1,
    )

    result = run_qa(graph, question="华微2024年营业收入是多少？", kb_id="default")

    assert result["escalate"] is False
    assert "100万元" in result["answer"]


def test_parent_scope_is_not_forced_to_consolidated():
    parent_question = "广道母公司2025年营业收入是多少？"
    assert _prefers_consolidated_scope(parent_question) is False
    assert _retrieval_queries(parent_question) == [parent_question]

    consolidated_question = "广道2025年营业收入是多少？"
    assert _prefers_consolidated_scope(consolidated_question) is True
    assert _retrieval_queries(consolidated_question)[-1].endswith("合并财务报表 合并口径")


def test_generation_context_exposes_optional_report_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, llm = _financial_graph(
        tmp_path,
        ["营业收入为100万元。[1]"],
        "广道2025年营业收入为100万元",
    )
    result = run_qa(graph, question="广道2025年营业收入是多少？", kb_id="default")

    assert result["retries"] == 0
    prompt = next(prompt for prompt in llm.prompts if "仅基于以下资料" in prompt)
    for value in (
        "公司标题=广道集团 2025 年报",
        "页码=7",
        "章节=财务报告 > 主要财务数据",
        "table_name=主要会计数据",
        "statement_scope=合并",
        "report_period=2025年度",
        "unit=万元",
    ):
        assert value in prompt


def test_generation_context_tolerates_missing_optional_metadata():
    block = _format_context_block(
        [{"chunk_id": "c1", "text": "正文", "metadata": {}}],
        ["正文"],
    )
    assert "公司标题=未提供" in block
    assert "table_name=未提供" in block
    assert "is_table=未提供" in block


def test_table_unit_context_is_body_only_and_keeps_years_unqualified():
    hit = {
        "text": "营业收入 100 90\n2024 2025",
        "metadata": {
            "is_table": True,
            "unit": "万元",
            "page": 7,
            "table_id": "2024",
        },
    }
    passages = [hit["text"]]

    evidence = _verification_evidence_texts("营业收入100万元[1]", [hit], passages)
    assert _verify_numeric_claims("营业收入100万元[1]", evidence)[0] is True
    assert _verify_numeric_claims("营业收入100.01万元[1]", evidence)[0] is False
    assert _verify_numeric_claims("报告年份2024[1]", evidence)[0] is True
    assert _verify_numeric_claims("报告年份2024万元[1]", evidence)[0] is False
    assert _verify_numeric_claims("页码7万元[1]", evidence)[0] is False
    assert "table_id=" not in "\n".join(evidence)

    not_a_table = {
        "text": "营业收入 100",
        "metadata": {"is_table": False, "unit": "万元"},
    }
    no_inference = _verification_evidence_texts(
        "营业收入100万元[1]", [not_a_table], [not_a_table["text"]]
    )
    assert _verify_numeric_claims("营业收入100万元[1]", no_inference)[0] is False


def test_machine_scope_values_include_clear_chinese_label():
    expected = {
        "consolidated": "合并",
        "parent": "母公司",
        "subsidiary": "子公司",
        "unknown": "未知",
    }
    for machine_value, chinese_label in expected.items():
        metadata_line = _format_candidate_metadata(
            {"metadata": {"statement_scope": machine_value}}
        )
        assert f"statement_scope={machine_value}" in metadata_line
        assert f"口径={chinese_label}" in metadata_line


def test_explicit_company_refusal_with_evidence_triggers_one_targeted_retry(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, llm = _financial_graph(
        tmp_path,
        ["无法确认，请提供公司全称。[1]", "营业收入为100万元。[1]"],
        "广道2025年营业收入为100万元",
    )
    result = run_qa(graph, question="广道2025年营业收入是多少？", kb_id="default")

    generation_prompts = [prompt for prompt in llm.prompts if "仅基于以下资料" in prompt]
    assert result["retries"] == 1
    assert len(generation_prompts) == 2
    assert "定向纠错重试" in generation_prompts[1]
    assert "不应再次拒答" in generation_prompts[1]
    assert "100万元" in result["answer"]


def test_multiple_numeric_items_omitted_triggers_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, llm = _financial_graph(
        tmp_path,
        [
            "营业收入为100万元。[1]",
            "营业收入为100万元，净利润为20万元。[1]",
        ],
        "广道2025年营业收入为100万元，净利润为20万元",
    )
    result = run_qa(
        graph,
        question="广道2025年营业收入和净利润分别是多少？",
        kb_id="default",
    )

    assert result["retries"] == 1
    assert "净利润为20万元" in result["answer"]
    retry_prompt = [prompt for prompt in llm.prompts if "仅基于以下资料" in prompt][1]
    assert "遗漏" in retry_prompt


def test_non_numeric_answer_does_not_trigger_numeric_retry():
    reasons = _answer_verification_reasons(
        "广道的经营范围是什么？",
        "经营范围包括技术服务和软件开发。",
        ["经营范围包括技术服务和软件开发。"],
    )
    assert reasons == []


def test_reason_question_with_explicit_cause_evidence_triggers_retry():
    reasons = _answer_verification_reasons(
        "葫芦娃2025年上半年营业收入同比下降42.89%的主要原因是什么？",
        "资料中未提供该变动的具体原因。",
        ["营业收入较上年同期下降42.89%，主要系呼吸系统用药销售额减少。"],
    )

    assert any("呼吸系统用药销售额减少" in reason for reason in reasons)


def test_numeric_verification_allows_only_one_retry_then_stops(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, llm = _financial_graph(
        tmp_path,
        ["营业收入为999万元。[1]", "营业收入为999万元。[1]"],
        "广道2025年营业收入为100万元",
    )
    result = run_qa(graph, question="广道2025年营业收入是多少？", kb_id="default")

    generation_prompts = [prompt for prompt in llm.prompts if "仅基于以下资料" in prompt]
    assert result["retries"] == 1
    assert len(generation_prompts) == 2
    assert "字段与数字无法可靠对应" in result["answer"]
    assert "999万元" not in result["answer"]


def test_stream_buffers_failed_generation_until_final_evaluation(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, _ = _financial_graph(
        tmp_path,
        ["营业收入为999万元。[1]", "营业收入为100万元。[1]"],
        "广道2025年营业收入 100 90",
    )

    events = list(run_qa_stream(graph, question="广道2025年营业收入是多少？"))
    final = next(event for event in events if event.get("type") == "final")
    streamed = "".join(
        event["text"] for event in events if event.get("type") == "token"
    )

    assert final["retries"] == 1
    assert streamed == final["answer"]
    assert "999" not in streamed


def test_stream_returns_safe_answer_when_numeric_retry_also_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    graph, _ = _financial_graph(
        tmp_path,
        ["营业收入为999万元。[1]", "营业收入为999万元。[1]"],
        "广道2025年营业收入为100万元",
    )

    events = list(run_qa_stream(graph, question="广道2025年营业收入是多少？"))
    final = next(event for event in events if event.get("type") == "final")
    streamed = "".join(
        event["text"] for event in events if event.get("type") == "token"
    )

    assert final["retries"] == 1
    assert "字段与数字无法可靠对应" in final["answer"]
    assert "999万元" not in streamed
    assert streamed == final["answer"]


def test_numeric_verification_ignores_unbound_small_narrative_numbers():
    supported, unsupported = _verify_numeric_claims(
        "元成环境2024年归属于母公司净利润为-325,030,861.71元，营业收入为145,839,584.29元；连续3年触及2项风险。[1][2]",
        [
            "元成环境2024年归属于母公司净利润为-325,030,861.71元，营业收入为145,839,584.29元。"
        ],
        question="元成环境2024年营业收入和归属于母公司净利润分别是多少？",
    )

    assert supported is True
    assert unsupported == []


def test_reason_answer_with_cause_marker_need_not_repeat_full_evidence_phrase():
    reasons = _answer_verification_reasons(
        "裕太微2025年上半年营业收入同比增长43.41%的主要原因是什么？",
        "主要由于半导体市场回暖和新产品持续放量，带动营业收入增长，营业收入为43.41%。[1]",
        [
            "裕太微2025年上半年营业收入为43.41%，主要系半导体市场延续增长态势，以及2.5G网通芯片等新产品持续销售放量。"
        ],
    )

    assert reasons == []


def test_cause_extractor_ignores_unresolved_reference_label():
    assert _cause_terms_from_evidence(["营业收入变动原因：见正文。"])[0:1] == []


def test_annual_question_does_not_treat_half_year_evidence_as_reliable():
    contexts = [
        {
            "text": "ST广道2025年半年度报告营业收入为100万元。",
            "metadata": {
                "doc_title": "ST广道 2025年半年度报告",
                "report_period": "2025年半年度",
            },
        }
    ]

    assert (
        _has_reliable_question_evidence(
            "ST广道2025年年度报告披露的全年营业收入是多少？",
            contexts,
            [contexts[0]["text"]],
        )
        is False
    )


def test_refusal_with_wrong_report_period_escalates_to_human(tmp_path, monkeypatch):
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    store = VectorStore(persist_dir=str(tmp_path))
    embedding = FakeEmbedding()
    evidence = "ST广道2025年半年度报告营业收入为100万元。"
    store.add_chunks(
        embeddings=embedding.embed_texts([evidence]),
        texts=[evidence],
        doc_id="guangdao-half-year-report",
        doc_title="ST广道 2025年半年度报告",
        source_type="pdf",
        chunk_indices=[0],
        kb_id="default",
        extra_metadata=[
            {
                "report_period": "2025年半年度",
                "unit": "万元",
                "is_table": True,
            }
        ],
    )
    llm = FakeLLM(
        responses=["无法确认，请提供公司全称。"],
        route={"RAG 质量评估员": "10 10"},
    )
    graph = build_qa_graph(
        llm=llm,
        embeddings=embedding,
        vector_store=store,
        top_k=1,
        recall_k=1,
        metadata_filters={"source_type": "pdf"},
    )

    result = run_qa(
        graph,
        question="ST广道2025年年度报告披露的全年营业收入是多少？",
        kb_id="default",
    )

    assert result["escalate"] is True
    assert result["citations"] == []
    assert "转接人工" in result["answer"]


def test_financial_ratio_synonym_maps_only_to_ratio_fact_not_amount_facts():
    metadata = {
        "financial_facts_json": json.dumps(
            [
                {
                    "metric": "研发投入占营业收入比例",
                    "raw_value": "12.34",
                    "canonical_value": "12.34",
                    "unit": "%",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
                {
                    "metric": "研发投入合计",
                    "raw_value": "123.45",
                    "canonical_value": "1234500",
                    "unit": "万元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
                {
                    "metric": "研发费用",
                    "raw_value": "67.89",
                    "canonical_value": "678900",
                    "unit": "万元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                },
            ],
            ensure_ascii=False,
        )
    }
    evidence = [
        {
            "text": (
                "研发投入占营业收入比例（%） 12.34；"
                "研发投入合计 123.45万元；研发费用 67.89万元"
            ),
            "metadata": metadata,
        }
    ]
    question = "虚构企业2024年研发投入占营业收入比率是多少？"

    assert _verify_numeric_claims(
        "研发投入占营业收入比率为12.34%[1]", evidence, question=question
    )[0] is True
    assert _verify_numeric_claims(
        "研发投入占营业收入比率为123.45%[1]", evidence, question=question
    )[0] is False
    assert _verify_numeric_claims(
        "研发投入占营业收入比率为67.89%[1]", evidence, question=question
    )[0] is False
    assert _verify_numeric_claims(
        "研发投入占营业收入比率为123.45万元[1]", evidence, question=question
    )[0] is False


def test_financial_decimal_unit_conversion_is_exact_and_rejects_near_values():
    evidence = [
        {
            "text": "资产总额 1,234,500元",
            "metadata": {
                "financial_facts_json": json.dumps(
                    [
                        {
                            "metric": "资产总额",
                            "raw_value": "1,234,500",
                            "canonical_value": "1234500",
                            "unit": "元",
                            "statement_scope": "consolidated",
                            "report_period": "2024年度",
                        }
                    ],
                    ensure_ascii=False,
                )
            },
        }
    ]
    question = "虚构企业2024年资产总额是多少？"

    assert _verify_numeric_claims(
        "资产总额为123.45万元[1]", evidence, question=question
    )[0] is True
    assert _verify_numeric_claims(
        "资产总额为1,234,500.01元[1]", evidence, question=question
    )[0] is False
    assert _verify_numeric_claims(
        "资产总额为123.45元[1]", evidence, question=question
    )[0] is False
    assert _verify_numeric_claims(
        "资产总额为123.4501万元[1]", evidence, question=question
    )[0] is False


def test_financial_table_binds_current_prior_and_adjustment_columns():
    """多列表格必须按表头列位绑定期间，不能把上期值或调整列当成本期值。

    表结构照搬真实年报「主要会计数据」表（ST富润 2024 年报 page 5 即为此布局）：

        主要会计数据 | 2024年 | 2023年 | 本期比上年同期增减(%) | 2022年

    发生追溯调整时，年报的真实写法是把「调整前/调整后」作为**上年同期的子列**
    展开，而不是排在 2023 年之后的两个独立列：

        主要会计数据 | 2024年 | 2023年        |        | 增减(%) | 2022年
                     |        | 调整前 | 调整后 |        |

    因此“2024 年调整前营业收入”在真实结构里并不存在——调整列属于上年同期。
    旧版测试把调整列写成 2023 年之后的独立列、又断言“2024 年调整前”成立，
    与真实财报结构不符，这里按真实列布局重写。
    """
    metadata = {
        "is_table": True,
        "unit": "万元",
        "report_period": "2024年度",
        "statement_scope": "consolidated",
    }
    flat_table = (
        "主要会计数据\n"
        "| 主要会计数据 | 2024年度 | 2023年度 | 本期比上年同期增减(%) | 2022年度 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 营业收入 | 120.00万元 | 110.00万元 | 8.26 | 125.00万元 |\n"
    )
    adjusted_table = (
        "主要会计数据\n"
        "| 主要会计数据 | 2024年度 | 2023年度 | | 本期比上年同期增减(%) | 2022年度 |\n"
        "| --- | --- | 调整前 | 调整后 | --- | --- |\n"
        "| 营业收入 | 120.00万元 | 110.00万元 | 115.00万元 | 8.26 | 125.00万元 |\n"
    )

    flat_evidence = [{"text": flat_table, "metadata": metadata}]
    adjusted_evidence = [{"text": adjusted_table, "metadata": metadata}]

    # 本报告期只能取本期列
    assert _verify_numeric_claims(
        "营业收入为120.00万元[1]",
        flat_evidence,
        question="赤羽虚构企业2024年营业收入是多少？",
    )[0] is True
    assert _verify_numeric_claims(
        "营业收入为110.00万元[1]",
        flat_evidence,
        question="赤羽虚构企业2024年营业收入是多少？",
    )[0] is False
    assert _verify_numeric_claims(
        "营业收入为125.00万元[1]",
        flat_evidence,
        question="赤羽虚构企业2024年营业收入是多少？",
    )[0] is False

    # 上年同期取上年列，第三年取 2022 列；增减列是百分比，不当作金额
    assert _verify_numeric_claims(
        "营业收入为110.00万元[1]",
        flat_evidence,
        question="赤羽虚构企业2023年营业收入是多少？",
    )[0] is True
    assert _verify_numeric_claims(
        "营业收入为120.00万元[1]",
        flat_evidence,
        question="赤羽虚构企业2023年营业收入是多少？",
    )[0] is False
    assert _verify_numeric_claims(
        "营业收入为125.00万元[1]",
        flat_evidence,
        question="赤羽虚构企业2022年营业收入是多少？",
    )[0] is True

    # 调整前/调整后挂在 2023 年之下：问 2023 年调整列成立，问 2024 年不成立
    assert _verify_numeric_claims(
        "营业收入为110.00万元[1]",
        adjusted_evidence,
        question="赤羽虚构企业2023年调整前营业收入是多少？",
    )[0] is True
    assert _verify_numeric_claims(
        "营业收入为115.00万元[1]",
        adjusted_evidence,
        question="赤羽虚构企业2023年调整后营业收入是多少？",
    )[0] is True
    assert _verify_numeric_claims(
        "营业收入为110.00万元[1]",
        adjusted_evidence,
        question="赤羽虚构企业2024年调整前营业收入是多少？",
    )[0] is False
    assert _verify_numeric_claims(
        "营业收入为115.00万元[1]",
        adjusted_evidence,
        question="赤羽虚构企业2024年调整后营业收入是多少？",
    )[0] is False


def test_partial_numeric_fallback_keeps_unique_field_and_blocks_cross_scope_sources():
    target_doc = "fictional-target-doc"
    target_kb = "fictional-target-kb"
    records = [
        _legacy_window_record(
            "营业收入 123.45万元",
            10,
            doc_id=target_doc,
            kb_id=target_kb,
        ),
        _legacy_window_record(
            "母公司口径归母净利润 222.22万元",
            11,
            doc_id=target_doc,
            kb_id=target_kb,
        ),
        _legacy_window_record(
            "合并口径归母净利润 222.22万元",
            12,
            doc_id="fictional-other-doc",
            kb_id=target_kb,
        ),
        _legacy_window_record(
            "合并口径归母净利润 222.22万元",
            13,
            doc_id=target_doc,
            kb_id="fictional-other-kb",
        ),
    ]
    records[0]["metadata"]["financial_facts_json"] = json.dumps(
        [
            {
                "metric": "营业收入",
                "raw_value": "123.45",
                "canonical_value": "1234500",
                "unit": "万元",
                "statement_scope": "consolidated",
                "report_period": "2024年度",
            }
        ],
        ensure_ascii=False,
    )
    records[1]["metadata"]["financial_facts_json"] = json.dumps(
        [
            {
                "metric": "归母净利润",
                "raw_value": "222.22",
                "canonical_value": "2222200",
                "unit": "万元",
                "statement_scope": "parent",
                "report_period": "2024年度",
            }
        ],
        ensure_ascii=False,
    )
    for record in records[2:]:
        record["metadata"]["financial_facts_json"] = json.dumps(
            [
                {
                    "metric": "归母净利润",
                    "raw_value": "222.22",
                    "canonical_value": "2222200",
                    "unit": "万元",
                    "statement_scope": "consolidated",
                    "report_period": "2024年度",
                }
            ],
            ensure_ascii=False,
        )

    question = "赤羽虚构企业2024年合并口径营业收入和归母净利润分别是多少？"
    answer = "营业收入为123.45万元[1]，归母净利润为222.22万元[1]"
    evidence = _legacy_window_evidence(answer, records)

    assert all(
        item["metadata"].get("doc_id") == target_doc
        and item["metadata"].get("kb_id") == target_kb
        for item in evidence
    )
    safe_answer = _partial_numeric_fallback_answer(question, answer, evidence)
    assert safe_answer is not None
    assert "营业收入为123.45万元" in safe_answer
    assert "归母净利润为222.22万元" not in safe_answer
    assert "归母净利润为无法确定" in safe_answer
    assert _answer_verification_reasons(question, safe_answer, evidence) == []


def _citation_relocation_hit(
    chunk_id: str,
    text: str,
    *,
    doc_id: str = "fictional-revenue-doc",
    kb_id: str = "fictional-kb",
    report_period: str = "2024年度",
    metric: str = "营业收入",
    value: str = "110,682,912.05",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "score": 0.9,
        "metadata": {
            "doc_id": doc_id,
            "kb_id": kb_id,
            "chunk_index": int(chunk_id.rsplit("-", 1)[-1]),
            "page": int(chunk_id.rsplit("-", 1)[-1]) + 1,
            "report_period": report_period,
            "unit": "元",
            "statement_scope": "consolidated",
            "is_table": True,
            "financial_facts_json": json.dumps(
                [
                    {
                        "metric": metric,
                        "raw_value": value,
                        "canonical_value": value.replace(",", ""),
                        "unit": "元",
                        "statement_scope": "consolidated",
                        "report_period": report_period,
                    }
                ],
                ensure_ascii=False,
            ),
        },
    }


def _run_citation_relocation_graph(monkeypatch, hits: list[dict]) -> dict:
    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    llm = FakeLLM(
        route={
            "仅基于以下资料": "营业收入为110,682,912.05元。[1]",
            "RAG 质量评估员": "10 10",
        }
    )

    def fake_search_queries(self, queries, **kwargs):
        return [dict(hit) for hit in hits]

    monkeypatch.setattr(HybridRetriever, "search_queries", fake_search_queries)
    graph = build_qa_graph(
        llm=llm,
        embeddings=FakeEmbedding(),
        vector_store=object(),
        top_k=len(hits),
        recall_k=len(hits),
        max_candidates_per_doc=None,
    )
    return run_qa(
        graph,
        question="甲方科技股份有限公司2024年度合并口径营业收入是多少？",
        kb_id="fictional-kb",
    )


def test_citation_relocation_uses_unique_exact_same_doc_kb_evidence(monkeypatch):
    hits = [
        _citation_relocation_hit(
            "fictional-revenue-10",
            "甲方科技股份有限公司 2024年度 合并口径 营业收入 110,682,912.06元",
            value="110,682,912.06",
        ),
        _citation_relocation_hit(
            "fictional-revenue-11",
            "甲方科技股份有限公司 2024年度 合并口径 营业收入 110,682,912.05元",
        ),
    ]

    result = _run_citation_relocation_graph(monkeypatch, hits)

    assert "110,682,912.05元" in result["answer"]
    assert result["citations"]
    assert result["citations"][0]["chunk_id"] == "fictional-revenue-11"
    assert result["citations"][0]["doc_id"] == "fictional-revenue-doc"
    assert result["citations"][0]["metadata"]["kb_id"] == "fictional-kb"
    assert result["retries"] == 0


@pytest.mark.parametrize(
    ("case", "invalid_hit_factory"),
    [
        (
            "cross_doc",
            lambda: _citation_relocation_hit(
                "fictional-revenue-11",
                "乙方科技股份有限公司 2024年度 合并口径 营业收入 110,682,912.05元",
                doc_id="fictional-other-doc",
            ),
        ),
        (
            "cross_kb",
            lambda: _citation_relocation_hit(
                "fictional-revenue-11",
                "甲方科技股份有限公司 2024年度 合并口径 营业收入 110,682,912.05元",
                kb_id="fictional-other-kb",
            ),
        ),
        (
            "wrong_period",
            lambda: _citation_relocation_hit(
                "fictional-revenue-11",
                "甲方科技股份有限公司 2023年度 合并口径 营业收入 110,682,912.05元",
                report_period="2023年度",
            ),
        ),
        (
            "wrong_metric",
            lambda: _citation_relocation_hit(
                "fictional-revenue-11",
                "甲方科技股份有限公司 2024年度 合并口径 研发投入 110,682,912.05元",
                metric="研发投入",
            ),
        ),
        (
            "ambiguous_duplicate",
            lambda: _citation_relocation_hit(
                "fictional-revenue-12",
                "甲方科技股份有限公司 2024年度 合并口径 营业收入 110,682,912.05元",
            ),
        ),
    ],
)
def test_citation_relocation_rejects_scope_period_metric_or_ambiguity(
    monkeypatch, case, invalid_hit_factory
):
    hits = [
        _citation_relocation_hit(
            "fictional-revenue-10",
            "甲方科技股份有限公司 2024年度 合并口径 营业收入 110,682,912.06元",
            value="110,682,912.06",
        ),
        invalid_hit_factory(),
    ]
    if case == "ambiguous_duplicate":
        hits.append(
            _citation_relocation_hit(
                "fictional-revenue-13",
                "甲方科技股份有限公司 2024年度 合并口径 营业收入 110,682,912.05元",
            )
        )

    result = _run_citation_relocation_graph(monkeypatch, hits)

    invalid_ids = {hit["chunk_id"] for hit in hits[1:]}
    assert all(
        citation.get("chunk_id") not in invalid_ids
        for citation in result["citations"]
    )


class _CitationWindowStore:
    """只提供邻块读取，模拟旧 Chroma 的 get_chunk_by_id 纯内存契约。"""

    def __init__(self, chunks: list[dict]):
        self._chunks = {chunk["chunk_id"]: chunk for chunk in chunks}

    def get_chunk_by_id(self, chunk_id: str, *, kb_id: str):
        chunk = self._chunks.get(chunk_id)
        if not chunk:
            return None
        metadata = chunk.get("metadata") or {}
        return chunk if metadata.get("kb_id") == kb_id else None


def _furun_trace_chunk(
    chunk_id: str,
    chunk_index: int,
    text: str,
    *,
    report_period: str = "",
    unit: str = "",
    section_path: str = "七、近三年主要会计数据和财务指标",
    previous_chunk_id: str = "",
    next_chunk_id: str = "",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "metadata": {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "doc_id": "fictional-furun-report",
            "kb_id": "fictional-cninfo",
            "doc_title": "ST富润-2024年年度报告.pdf",
            "page": 5 if chunk_index in (9, 10, 11) else 145,
            "page_start": 5 if chunk_index in (9, 10, 11) else 145,
            "page_end": 5 if chunk_index in (9, 10, 11) else 145,
            "section_path": section_path,
            "table_id": "table-2" if chunk_index in (9, 10, 11) else "table-105",
            "table_name": "",
            "source_type": "pdf",
            "chunk_type": "child",
            "statement_scope": "unknown",
            "report_period": report_period,
            "unit": unit,
            "is_table": True,
            "parent_id": "",
            "previous_chunk_id": previous_chunk_id,
            "next_chunk_id": next_chunk_id,
        },
    }


def _run_furun_trace_shape(monkeypatch, *, exact_value: bool):
    doc_id = "fictional-furun-report"
    kb_id = "fictional-cninfo"
    wrong_410 = _furun_trace_chunk(
        f"{doc_id}-410",
        410,
        "浙江富润数字科技股份有限公司2024 年年度报告\n不具备商业实质的收入小计",
        section_path="二、不具备商业实质的收入",
        next_chunk_id=f"{doc_id}-411",
    )
    wrong_411 = _furun_trace_chunk(
        f"{doc_id}-411",
        411,
        "三、与主营业务无关或不具备商业实质的其他收入\n"
        "营业收入扣除后金额\n11,068.29\n7,289.77",
        section_path="三、与主营业务无关或不具备商业实质的其他收入",
        previous_chunk_id=f"{doc_id}-410",
        next_chunk_id=f"{doc_id}-412",
    )
    wrong_412 = _furun_trace_chunk(
        f"{doc_id}-412",
        412,
        "营业收入分解信息\n营业收入\n110,682,912.06\n97,416,388.81",
        section_path="三、与主营业务无关或不具备商业实质的其他收入",
        next_chunk_id=f"{doc_id}-413",
        previous_chunk_id=f"{doc_id}-411",
    )
    correct_value = "110,682,912.05" if exact_value else "110,682,912.06"
    correct_9 = _furun_trace_chunk(
        f"{doc_id}-9",
        9,
        "七、近三年主要会计数据和财务指标\n(一) 主要会计数据\n"
        "单位：元  币种：人民币\n主要会计数据\n2024年\n2023年",
        report_period="2024年",
        unit="元",
        previous_chunk_id=f"{doc_id}-8",
        next_chunk_id=f"{doc_id}-10",
    )
    correct_10 = _furun_trace_chunk(
        f"{doc_id}-10",
        10,
        "营业收入\n134,241,812.75\n93,231,677.42\n"
        "扣除与主营业务无关的业\n务收入和不具备商业实质的收入后的营业收入\n"
        f"{correct_value}\n72,897,668.51",
        report_period="2024年",
        unit="元",
        previous_chunk_id=f"{doc_id}-9",
        next_chunk_id=f"{doc_id}-11",
    )
    correct_11 = _furun_trace_chunk(
        f"{doc_id}-11",
        11,
        "本期比上年同期增减(%)\n51.83\n2022年\n175,609,918.35",
        report_period="2024年",
        unit="元",
        previous_chunk_id=f"{doc_id}-10",
        next_chunk_id=f"{doc_id}-12",
    )
    unrelated_14 = _furun_trace_chunk(
        f"{doc_id}-14",
        14,
        "八、境内外会计准则下会计数据差异\n适用：否",
        section_path="八、境内外会计准则下会计数据差异",
        next_chunk_id=f"{doc_id}-15",
    )
    unrelated_15 = _furun_trace_chunk(
        f"{doc_id}-15",
        15,
        "归属于上市公司股东的净利润\n-360,794,503.43\n"
        "经营活动产生的现金流量净额\n-52,553,814.81",
        section_path="八、境内外会计准则下会计数据差异",
        previous_chunk_id=f"{doc_id}-14",
        next_chunk_id=f"{doc_id}-16",
    )
    unrelated_16 = _furun_trace_chunk(
        f"{doc_id}-16",
        16,
        "主要财务指标\n基本每股收益（元／股）\n-0.71",
        section_path="八、境内外会计准则下会计数据差异",
        previous_chunk_id=f"{doc_id}-15",
    )
    chunks = [
        wrong_410,
        wrong_412,
        correct_9,
        correct_11,
        unrelated_14,
        unrelated_15,
        unrelated_16,
    ]
    hits = [wrong_411, correct_10, unrelated_15]
    store = _CitationWindowStore(chunks)
    llm = FakeLLM(
        route={
            "仅基于以下资料": (
                "根据资料，ST富润2024年扣除与主营业务无关的业务收入和不具备商业实质的收入"
                "后的营业收入为110,682,912.05元 [1]。上述数值见资料[1]。"
            ),
            "RAG 质量评估员": "10 10",
        }
    )

    def fake_search_queries(self, queries, **kwargs):
        return [dict(hit) for hit in hits]

    monkeypatch.setenv("KB_ANSWERABILITY_SCORE", "0")
    monkeypatch.setattr(HybridRetriever, "search_queries", fake_search_queries)
    graph = build_qa_graph(
        llm=llm,
        embeddings=FakeEmbedding(),
        vector_store=store,
        top_k=3,
        recall_k=3,
        max_candidates_per_doc=None,
        neighbor_expansion=2,
    )
    result = run_qa(
        graph,
        question="ST富润2024年扣除与主营业务无关及不具备商业实质的收入后，营业收入是多少？",
        kb_id=kb_id,
    )
    return result, llm


def test_real_furun_trace_shape_relocates_repeated_citation_to_exact_window(
    monkeypatch,
):
    result, llm = _run_furun_trace_shape(monkeypatch, exact_value=True)

    generation_prompt = next(
        prompt for prompt in llm.prompts if "仅基于以下资料" in prompt
    )
    assert "110,682,912.06" in generation_prompt
    assert "110,682,912.05" in generation_prompt
    assert "110,682,912.05元" in result["answer"]
    assert result["citations"]
    assert all(
        citation["chunk_id"] == "fictional-furun-report-10"
        for citation in result["citations"]
    )
    assert result["retries"] == 0


def test_real_furun_trace_shape_does_not_relocate_001_near_value(monkeypatch):
    result, _ = _run_furun_trace_shape(monkeypatch, exact_value=False)

    assert all(
        citation["chunk_id"] != "fictional-furun-report-10"
        for citation in result["citations"]
    )


# ---------------------------------------------------------------------------
# 格力 2023 年报第 7 页定向回归。
# 正文与 metadata 均取自 Qdrant collection kb_full_codex_20260830_24c7407e
# 中 chunk_id = doc-341b4598ee5c5c9892bc83a3ea2a7bc4-13 的真实 payload：
# 该页 financial_metrics 只有"营业收入"，而正文把"经营活动产生的现金
# \n流量净额（元）"拆成两行。这正是"单指标兜底覆盖已识别指标"的触发场景。
# ---------------------------------------------------------------------------

_GREE_P7_TEXT = (
    "六、主要会计数据和财务指标\n公司是否需追溯调整或重述以前年度会计数据\n"
    "□是 \uf052否\n项目\n2023 年\n2022 年\n本年比上年增减\n2021 年\n"
    "营业收入（元）\n203,979,266,387.09\n188,988,382,706.68\n7.93%\n"
    "187,868,874,892.71\n归属于上市公司股东\n的净利润（元）\n"
    "29,017,387,604.18\n24,506,623,782.46\n18.41%\n23,063,732,372.62\n"
    "27,565,461,117.79\n23,986,248,264.15\n14.92%\n21,850,050,895.31\n"
    "归属于上市公司股东\n的扣除非经常性损益\n的净利润（元）\n"
    "经营活动产生的现金\n流量净额（元）\n56,398,426,354.17\n"
    "28,668,435,921.27\n96.73%\n1,894,363,258.72\n基本每股收益（元/\n股）\n"
    "5.22\n4.43\n17.83%\n4.04\n稀释每股收益（元/\n股）\n5.22\n4.43\n17.83%\n"
    "4.04\n加权平均净资产收益\n率\n26.53%\n24.19%\n2.34%\n21.34%\n项目\n"
    "2023 年末\n2022 年末\n本年末比上年末增减\n2021 年末\n总资产（元）\n"
    "368,053,902,576.37\n355,024,758,878.82\n3.67%\n319,598,183,780.38\n"
    "归属于上市公司股东\n的净资产（元）\n116,793,716,103.39\n"
    "96,758,734,892.25\n20.71%\n103,651,654,599.87"
)

# 真实 payload 里 financial_facts_json 只保留了三个"营业收入"事实。
_GREE_P7_FACTS_JSON = json.dumps(
    [
        {
            "canonical_value": "187868874892.71",
            "metric": "营业收入",
            "raw_value": "187,868,874,892.71",
            "report_period": "2023年度",
            "statement_scope": "unknown",
            "unit": "",
        },
        {
            "canonical_value": "188988382706.68",
            "metric": "营业收入",
            "raw_value": "188,988,382,706.68",
            "report_period": "2023年度",
            "statement_scope": "unknown",
            "unit": "",
        },
        {
            "canonical_value": "203979266387.09",
            "metric": "营业收入",
            "raw_value": "203,979,266,387.09",
            "report_period": "2023年度",
            "statement_scope": "unknown",
            "unit": "",
        },
    ],
    ensure_ascii=False,
)

_GREE_P7_METADATA = {
    "chunk_id": "doc-341b4598ee5c5c9892bc83a3ea2a7bc4-13",
    "section_path": "六、主要会计数据和财务指标",
    "table_name": "主要会计数据",
    "statement_scope": "unknown",
    "report_period": "2023年度",
    "unit": "",
    "is_table": True,
    "financial_facts_json": _GREE_P7_FACTS_JSON,
    "financial_metrics": "营业收入",
    "page": 7,
    "page_start": 7,
    "page_end": 7,
    "doc_id": "doc-341b4598ee5c5c9892bc83a3ea2a7bc4",
    "doc_title": "格力电器-2023年年度报告",
    "source_type": "pdf",
    "kb_id": "cninfo_report",
}

_GREE_QUESTION = (
    "格力电器2023年年报主要会计数据中，营业收入、归母净利润、"
    "经营活动现金流量净额和基本每股收益分别是多少？"
)

_GREE_ANSWER = (
    "格力电器2023年年报主要数据如下：\n"
    "- 营业收入：203,979,266,387.09元\n"
    "- 归母净利润：29,017,387,604.18元\n"
    "- 经营活动现金流量净额：56,398,426,354.17元\n"
    "- 基本每股收益：5.22元/股\n"
    "\n[1][2]"
)

_GREE_EVIDENCE = [{"text": _GREE_P7_TEXT, "metadata": _GREE_P7_METADATA}]


def _gree_claim(raw_value: str) -> object:
    return next(
        claim
        for claim in _extract_numeric_claims(_GREE_P7_TEXT)
        if claim.raw == raw_value
    )


def _gree_bound(raw_value: str):
    return _claim_with_context(
        _gree_claim(raw_value),
        _GREE_P7_TEXT,
        question=_GREE_QUESTION,
        metadata=_GREE_P7_METADATA,
        fallback_metrics=["revenue"],
    )


def test_gree_pdf_wrapped_cashflow_label_binds_value_on_following_line():
    """PDF 把"经营活动产生的现金\\n流量净额（元）"拆行，数字在下一行。"""
    # 1) 跨换行的指标名必须能被识别成一个整体标签。
    assert "operating_cashflow" in {
        metric for _s, _e, metric in _metric_mentions("经营活动产生的现金\n流量净额（元）")
    }

    raw_claim = _gree_claim("56,398,426,354.17")
    assert raw_claim.metric == ""
    # 2) 宽松行绑定把上一行的行标题带给这个数字。
    assert raw_claim.line_metric == "operating_cashflow"
    # 3) 该数字紧邻的前一个标签是"扣除非经常性损益的净利润"，不能被串绑。
    assert raw_claim.line_metric != "net_profit_attributable"

    # 4) 同一事实不换行时也必须同样绑定，避免只覆盖了一种版式。
    flat_claims = _extract_numeric_claims("经营活动产生的现金流量净额（元）\n56,398,426,354.17")
    assert flat_claims[0].metric == "operating_cashflow"

    # 5) 带着这份"只有 revenue"的 metadata 走完上下文绑定后，换行版式识别
    #    出来的行标签仍然是最终结果，不能被单指标兜底改写成 revenue。
    assert _gree_bound("56,398,426,354.17").metric == "operating_cashflow"


def test_gree_revenue_only_metadata_does_not_recover_cashflow_into_revenue():
    """metadata 只有 revenue 时，已识别的 operating_cashflow 不得被覆盖。"""
    bound = _gree_bound("56,398,426,354.17")

    assert bound.metric == "operating_cashflow"
    assert _resolve_claim_metric(
        _gree_claim("56,398,426,354.17"), _GREE_P7_TEXT, ["revenue"]
    ) == "operating_cashflow"
    # 反向断言：若兜底抢先生效，这里会变成 revenue 并触发 metric mismatch。
    assert bound.metric != "revenue"


def test_gree_multi_year_rows_do_not_cross_bind_metrics():
    """同一张多列表里的 2023/2022 数字各归各指标，不跨指标串绑。"""
    expected = {
        "203,979,266,387.09": ("revenue", "2023年"),
        "188,988,382,706.68": ("revenue", "2022年"),
        "29,017,387,604.18": ("net_profit_attributable", "2023年"),
        "56,398,426,354.17": ("operating_cashflow", "2023年"),
        "5.22": ("eps", "2023年"),
    }
    for raw_value, (metric, period) in expected.items():
        bound = _gree_bound(raw_value)
        assert bound.metric == metric, raw_value
        assert bound.report_period == period, raw_value

    # 现金流数值不允许出现在 revenue / 归母净利润 的绑定上。
    assert _gree_bound("56,398,426,354.17").metric not in {
        "revenue",
        "net_profit_attributable",
    }
    # 把现金流值错填到"归母净利润"位置必须被判为无证据。
    swapped = _GREE_ANSWER.replace(
        "归母净利润：29,017,387,604.18元",
        "归母净利润：56,398,426,354.17元",
    )
    assert _verify_numeric_claims(swapped, _GREE_EVIDENCE, question=_GREE_QUESTION)[0] is False


def test_gree_single_metric_fallback_still_works_without_line_metric():
    """没有明确 line_metric / 表格行线索时，metadata 单指标兜底仍然生效。"""
    claim = _extract_numeric_claims("单位：元\n1,234.56")[0]
    assert claim.metric == ""
    assert claim.line_metric == ""

    bound = _claim_with_context(
        claim,
        "单位：元\n1,234.56",
        question="营业收入是多少？",
        metadata=_GREE_P7_METADATA,
        fallback_metrics=["revenue"],
    )
    assert bound.metric == "revenue"
    # 兜底对年份/页码这类非财务数字仍然无效。
    year_claim = replace(
        claim,
        number=Decimal("2023"),
        canonical_value=Decimal("2023"),
        unit="",
        is_percent=False,
    )
    assert _resolve_claim_metric(year_claim, "单位：元\n2023", ["revenue"]) == ""


def test_gree_four_field_answer_passes_without_partial_fallback():
    """四字段完整答案必须整体通过核验，不触发 partial fallback 遮蔽。"""
    supported, unsupported = _verify_numeric_claims(
        _GREE_ANSWER, _GREE_EVIDENCE, question=_GREE_QUESTION
    )

    assert supported is True
    assert unsupported == []
    assert _partial_numeric_fallback_answer(
        _GREE_QUESTION, _GREE_ANSWER, _GREE_EVIDENCE
    ) is None
    assert _answer_verification_reasons(
        _GREE_QUESTION, _GREE_ANSWER, _GREE_EVIDENCE
    ) == []
    for value in ("203,979,266,387.09", "29,017,387,604.18", "56,398,426,354.17", "5.22"):
        assert value in _GREE_ANSWER
