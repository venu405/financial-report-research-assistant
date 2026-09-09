"""结构化财务表格上下文与跨页继承测试。"""
from __future__ import annotations

import json

from services.kb import ingest


def _build_from_blocks(monkeypatch, tmp_path, blocks, **kwargs):
    path = tmp_path / "financial.pdf"
    path.write_bytes(b"synthetic pdf placeholder")
    monkeypatch.setattr(ingest, "parse_document_structured", lambda *_args, **_kwargs: blocks)
    return ingest.build_chunks(path, doc_id="financial-1", **kwargs)


def test_financial_scope_period_unit_and_scalar_metadata(monkeypatch, tmp_path):
    blocks = ingest._structured_blocks_from_pages(
        [
            (
                1,
                "合并利润表\n单位：元\n2024年年度报告\n"
                "项目 2024年 2023年\n营业收入 100 90",
            ),
            (
                2,
                "母公司利润表\n单位：万元\n2024年年度报告\n"
                "项目 2024年 2023年\n营业收入 10 9",
            ),
        ]
    )

    assert blocks[0].table_name == "合并利润表"
    assert blocks[0].statement_scope == "consolidated"
    assert blocks[0].report_period == "2024年度"
    assert blocks[0].unit == "元"
    assert blocks[1].table_name == "母公司利润表"
    assert blocks[1].statement_scope == "parent"
    assert blocks[1].unit == "万元"
    assert blocks[0].table_id != blocks[1].table_id

    chunks = _build_from_blocks(monkeypatch, tmp_path, blocks, chunk_size=500)
    assert len(chunks) == 2
    for chunk in chunks:
        assert chunk.metadata["is_table"] is True
        assert isinstance(chunk.metadata["table_id"], str)
        assert all(isinstance(value, (str, int, float, bool)) for value in chunk.metadata.values())
    assert chunks[0].metadata["statement_scope"] == "consolidated"
    assert chunks[1].metadata["statement_scope"] == "parent"


def test_financial_context_inherits_across_consecutive_pages(monkeypatch, tmp_path):
    blocks = ingest._structured_blocks_from_pages(
        [
            (
                1,
                "合并现金流量表\n单位：万元\n报告期：2024年度\n"
                "项目 2024年 2023年\n经营活动现金流 100 90",
            ),
            (
                2,
                "续表\n销售商品收到的现金 80 70\n"
                "购买商品支付的现金 60 50",
            ),
        ]
    )

    assert blocks[1].is_table is True
    assert blocks[1].table_name == "合并现金流量表"
    assert blocks[1].statement_scope == "consolidated"
    assert blocks[1].report_period == "2024年度"
    assert blocks[1].unit == "万元"
    assert blocks[1].table_id == blocks[0].table_id

    chunks = _build_from_blocks(monkeypatch, tmp_path, blocks, chunk_size=500)
    assert {chunk.metadata["table_id"] for chunk in chunks} == {blocks[0].table_id}
    assert all(chunk.metadata["unit"] == "万元" for chunk in chunks)


def test_statement_scope_requires_explicit_table_or_context_label():
    blocks = ingest._structured_blocks_from_pages(
        [
            (
                1,
                "利润表\n单位：万元\n报告期：2024年度\n"
                "项目 2024年 2023年\n营业收入 100 90",
            ),
            (
                2,
                "母公司利润表\n单位：万元\n报告期：2024年度\n"
                "项目 2024年 2023年\n营业收入 10 9",
            ),
            (
                3,
                "子公司口径\n利润表\n单位：万元\n报告期：2024年度\n"
                "项目 2024年 2023年\n营业收入 8 7",
            ),
        ]
    )

    assert [block.statement_scope for block in blocks] == [
        "unknown",
        "parent",
        "subsidiary",
    ]


def test_conflicting_explicit_scope_breaks_continuation():
    blocks = ingest._structured_blocks_from_pages(
        [
            (
                1,
                "合并现金流量表\n单位：万元\n报告期：2024年度\n"
                "项目 2024年 2023年\n经营活动产生的现金流量净额 100 90",
            ),
            (
                2,
                "续表\n母公司口径\n经营活动产生的现金流量净额 80 70\n"
                "购买商品支付的现金 60 50",
            ),
        ]
    )

    assert blocks[0].statement_scope == "consolidated"
    assert blocks[1].is_table is False
    assert blocks[1].table_id == ""
    assert blocks[1].statement_scope == "unknown"


def test_financial_facts_are_traceable_decimal_strings_and_stable(monkeypatch, tmp_path):
    blocks = ingest._structured_blocks_from_pages(
        [
            (
                1,
                "合并利润表\n单位：万元\n报告期：2024年度\n"
                "项目 | 2024年 | 2023年\n"
                "| --- | --- | --- |\n"
                "营业收入 | 100.25 | 90.00\n"
                "归母净利润 | (3,000.00) | 2,000.00\n"
                "研发费用 | 25.50 | 20.00\n"
                "营业收入同比 | 2.5% | 1.5%",
            )
        ]
    )
    chunks = _build_from_blocks(monkeypatch, tmp_path, blocks, chunk_size=800)
    assert len(chunks) == 1

    metadata = chunks[0].metadata
    assert metadata["financial_metrics"] == "营业收入|归属于上市公司股东的净利润|研发费用"
    facts = json.loads(metadata["financial_facts_json"])
    assert facts == ingest._stable_financial_facts(facts)
    assert all(
        set(fact) == {
            "metric",
            "raw_value",
            "canonical_value",
            "unit",
            "statement_scope",
            "report_period",
        }
        for fact in facts
    )
    revenue = next(fact for fact in facts if fact["metric"] == "营业收入" and fact["raw_value"] == "100.25")
    assert revenue["canonical_value"] == "1002500.00"
    assert revenue["unit"] == "万元"
    assert revenue["statement_scope"] == "consolidated"
    assert revenue["report_period"] == "2024年度"
    assert any(
        fact["metric"] == "归属于上市公司股东的净利润" and fact["canonical_value"] == "-30000000.00"
        for fact in facts
    )
    assert not any("2.5" in fact["raw_value"] or fact["raw_value"] in {"2024", "12"} for fact in facts)

    facts_again = ingest._extract_financial_facts(
        blocks[0].text,
        is_table=blocks[0].is_table,
        statement_scope=blocks[0].statement_scope,
        report_period=blocks[0].report_period,
        unit=blocks[0].unit,
    )
    assert ingest._financial_metadata_fields(facts_again) == {
        "financial_facts_json": metadata["financial_facts_json"],
        "financial_metrics": metadata["financial_metrics"],
    }


def test_multiyear_table_binds_periods_by_column_and_skips_change_and_note_columns():
    text = (
        "七、近三年主要会计数据和财务指标\n"
        "2022年\n主要会计数据\n2024年\n2023年\n本期比上年同期增减(%)\n"
        "营业收入\n1,876,694,452.50\n2,606,047,231.66\n-27.99\n3,704,914,217.80\n\n"
        "| 主要会计数据 | 附注 | 2024年 | 2023年 |  | 本期比上年同期增减(%) | 2022年 |  |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "|  |  |  | 调整后 | 调整前 |  | 调整后 | 调整前 |\n"
        "| 营业收入 | 61 | 1,876,694,452.50 | 2,606,047,231.66 | 2,606,047,231.66 | -27.99 | 3,704,914,217.80 | 3,704,914,217.80 |"
    )

    facts = ingest._extract_financial_facts(
        text,
        is_table=True,
        statement_scope="consolidated",
        report_period="2024年度",
        unit="元",
    )
    revenue = [fact for fact in facts if fact["metric"] == "营业收入"]

    assert {
        (fact["raw_value"], fact["report_period"])
        for fact in revenue
    } == {
        ("1,876,694,452.50", "2024年度"),
        ("2,606,047,231.66", "2023年"),
        ("3,704,914,217.80", "2022年"),
    }
    assert not any(fact["raw_value"] in {"61", "-27.99"} for fact in facts)


def test_relative_period_headers_map_current_and_prior_year_without_change_rate():
    text = (
        "| 项目 | 本期数 | 上年同期数 | 变动比例（%） |\n"
        "| --- | --- | --- | --- |\n"
        "| 营业收入 | 1,876,694,452.50 | 2,606,047,231.66 | -27.99 |"
    )

    facts = ingest._extract_financial_facts(
        text,
        is_table=True,
        statement_scope="consolidated",
        report_period="2024年度",
        unit="元",
    )

    assert {
        (fact["raw_value"], fact["report_period"])
        for fact in facts
    } == {
        ("1,876,694,452.50", "2024年度"),
        ("2,606,047,231.66", "2023年度"),
    }


def test_inline_units_and_parentheses_keep_value_and_period_binding():
    text = (
        "| 项目 | 本期数 | 上期数 |\n"
        "| --- | --- | --- |\n"
        "| 营业收入 | 1.5万元 | (2,000)元 |"
    )

    facts = ingest._extract_financial_facts(
        text,
        is_table=True,
        statement_scope="consolidated",
        report_period="2024年度",
        unit="",
    )

    assert {
        (fact["raw_value"], fact["canonical_value"], fact["unit"], fact["report_period"])
        for fact in facts
    } == {
        ("1.5", "15000.0", "万元", "2024年度"),
        ("(2,000)", "-2000", "元", "2023年度"),
    }


def test_new_table_or_plain_body_stops_inheritance():
    blocks = ingest._structured_blocks_from_pages(
        [
            (
                1,
                "合并利润表\n单位：元\n2024年度\n"
                "项目 2024年 2023年\n营业收入 100 90",
            ),
            (2, "本页为管理层讨论与分析，收入变化原因见正文。"),
            (3, "净利润 10 9\n归属于股东的净利润 8 7"),
            (
                4,
                "合并现金流量表\n单位：元\n2024年度\n"
                "项目 2024年 2023年\n经营活动现金流 30 20",
            ),
        ]
    )

    assert blocks[1].is_table is False
    assert blocks[1].table_id == ""
    assert blocks[2].is_table is False
    assert blocks[2].table_id == ""
    assert blocks[3].table_name == "合并现金流量表"
    assert blocks[3].table_id != blocks[0].table_id


def test_financial_metadata_reaches_parent_and_child_chunks(monkeypatch, tmp_path):
    blocks = ingest._structured_blocks_from_pages(
        [
            (
                1,
                "合并资产负债表\n单位：亿元\n2024年年度报告\n"
                "项目 2024年 2023年\n"
                + "\n".join(f"资产项目{index} {index} {index - 1}" for index in range(1, 12)),
            )
        ]
    )

    chunks = _build_from_blocks(
        monkeypatch,
        tmp_path,
        blocks,
        parent_chunk_size=80,
        child_chunk_size=35,
    )
    parents = [chunk for chunk in chunks if chunk.chunk_type == "parent"]
    children = [chunk for chunk in chunks if chunk.chunk_type == "child"]

    assert parents and children
    assert all(chunk.metadata["is_table"] is True for chunk in chunks)
    assert {chunk.metadata["table_name"] for chunk in chunks} == {"合并资产负债表"}
    assert {chunk.metadata["statement_scope"] for chunk in chunks} == {"consolidated"}
    assert {chunk.metadata["unit"] for chunk in chunks} == {"亿元"}
    assert {chunk.metadata["report_period"] for chunk in chunks} == {"2024年度"}
    assert all(child.parent_id for child in children)


def test_non_financial_structured_and_legacy_chunks_remain_safe(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# 采购制度\n\n审批流程和留痕要求。", encoding="utf-8")

    structured = ingest.build_chunks(path, doc_id="policy-1")
    assert structured
    assert all(chunk.metadata["is_table"] is False for chunk in structured)
    assert all(chunk.metadata["table_id"] == "" for chunk in structured)
    assert all(chunk.metadata["statement_scope"] == "unknown" for chunk in structured)

    legacy = ingest.build_legacy_chunks(path, doc_id="policy-legacy")
    assert legacy
    assert all(chunk.metadata == {} for chunk in legacy)


def test_document_title_is_reliable_report_period_fallback(tmp_path):
    path = tmp_path / "示例公司-2024年年度报告.md"
    path.write_text(
        "合并利润表\n"
        "单位：万元\n"
        "项目 | 2024年 | 2023年\n"
        "| --- | --- | --- |\n"
        "营业收入 | 100 | 90\n",
        encoding="utf-8",
    )

    chunks = ingest.build_chunks(path, doc_id="period-title")

    assert chunks[0].metadata["report_period"] == "2024年度"


def test_explicit_document_title_is_report_period_fallback_for_unstamped_filename(tmp_path):
    path = tmp_path / "report.md"
    path.write_text(
        "半年度报告\n"
        "合并利润表\n"
        "单位：万元\n"
        "项目 | 2025年 | 2024年\n"
        "| --- | --- | --- |\n"
        "营业收入 | 100 | 90\n",
        encoding="utf-8",
    )

    chunks = ingest.build_chunks(
        path,
        doc_id="explicit-period-title",
        document_title="某公司2025年半年度报告",
    )

    assert chunks
    assert {chunk.metadata["report_period"] for chunk in chunks} == {"2025年半年度"}


def test_unit_row_and_data_split_into_adjacent_blocks_keep_table_context():
    blocks = ingest._annotate_financial_blocks(
        [
            ingest.StructuredBlock(
                text="合并利润表\n项目 2024年 2023年",
                page_start=10,
                page_end=10,
                section_path="财务报告",
            ),
            ingest.StructuredBlock(
                text="单位：万元",
                page_start=11,
                page_end=11,
                section_path="财务报告",
            ),
            ingest.StructuredBlock(
                text="续表\n营业收入 100 90",
                page_start=12,
                page_end=12,
                section_path="财务报告",
            ),
        ],
        doc_period="2024年度",
    )

    assert all(block.is_table for block in blocks)
    assert all(block.table_name == "合并利润表" for block in blocks)
    assert all(block.unit == "万元" for block in blocks)
    assert all(block.report_period == "2024年度" for block in blocks)
    assert len({block.table_id for block in blocks}) == 1


def test_major_accounting_data_does_not_guess_consolidated_scope():
    blocks = ingest._annotate_financial_blocks(
        [
            ingest.StructuredBlock(
                text=(
                    "主要会计数据\n单位：元\n"
                    "项目 | 2024年 | 2023年\n"
                    "| --- | --- | --- |\n"
                    "营业收入 | 100 | 90"
                ),
                page_start=1,
                page_end=1,
                section_path="财务报告",
            )
        ],
        doc_period="2024年度",
    )

    assert blocks[0].statement_scope == "unknown"


def test_complex_balance_columns_without_periods_are_not_guessed():
    text = (
        "| 项目 | 期初余额 | 本期增加 | 本期减少 | 期末余额 |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| 营业收入 | 100 | 20 | 5 | 115 |"
    )

    facts = ingest._extract_financial_facts(
        text,
        is_table=True,
        statement_scope="consolidated",
        report_period="2024年度",
        unit="元",
    )

    assert facts == []


def test_conflicting_report_period_stops_cross_page_inheritance():
    blocks = ingest._annotate_financial_blocks(
        [
            ingest.StructuredBlock(
                text=(
                    "合并利润表\n单位：元\n报告期：2024年度\n"
                    "项目 2024年 2023年\n营业收入 100 90"
                ),
                page_start=1,
                page_end=1,
                section_path="财务报告",
            ),
            ingest.StructuredBlock(
                text="续表\n报告期：2023年度\n营业收入 80 70",
                page_start=2,
                page_end=2,
                section_path="财务报告",
            ),
        ]
    )

    assert blocks[0].is_table is True
    assert blocks[1].is_table is False
    assert blocks[1].table_id == ""


def test_numbered_new_financial_heading_stops_previous_table_context():
    blocks = ingest._annotate_financial_blocks(
        [
            ingest.StructuredBlock(
                text="合并利润表\n单位：元\n营业收入 100 90",
                page_start=1,
                page_end=1,
                section_path="财务报告",
            ),
            ingest.StructuredBlock(
                text="2、利润构成\n单位：万元\n营业收入 8 7",
                page_start=2,
                page_end=2,
                section_path="财务报告",
            ),
        ]
    )

    assert blocks[0].is_table is True
    assert blocks[1].is_table is False
    assert blocks[1].table_id == ""
