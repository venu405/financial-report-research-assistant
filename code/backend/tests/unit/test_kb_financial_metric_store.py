from __future__ import annotations

import pytest

from services.kb.financial_metric_store import FinancialMetricStore


def _metric(**overrides):
    data = {
        "kb_id": "default",
        "company_name": "测试公司",
        "company_code": "600000",
        "report_period": "2024",
        "period_type": "annual",
        "metric_code": "revenue",
        "metric_name": "营业收入",
        "raw_value": "1.25",
        "raw_unit": "亿元",
        "statement_scope": "consolidated",
        "source_doc_id": "doc-1",
        "source_title": "2024 年报",
        "source_page": 6,
        "source_page_end": 7,
        "source_chunk_id": "chunk-1",
        "source_text": "营业收入 1.25 亿元",
        "extraction_status": "verified",
        "created_by": "tester",
    }
    data.update(overrides)
    return data


def test_units_and_decimal_strings_are_normalized_without_float_loss(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    yuan = store.create(_metric(raw_value="1", raw_unit="亿元"))
    wan = store.create(_metric(metric_code="total_assets", raw_value="12,345.60", raw_unit="万元"))
    negative = store.create(_metric(metric_code="net_profit_parent", raw_value="-1,234.50", raw_unit="元"))

    assert yuan["raw_value"] == "1"
    assert yuan["normalized_value"] == "100000000"
    assert yuan["normalized_unit"] == "元"
    assert wan["raw_value"] == "12345.60"
    assert wan["normalized_value"] == "123456000"
    assert negative["normalized_value"] == "-1234.5"


def test_missing_stays_empty_and_invalid_units_are_rejected(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    missing = store.create(_metric(extraction_status="missing", raw_value=None, raw_unit=""))

    assert missing["raw_value"] is None
    assert missing["normalized_value"] is None
    assert missing["normalized_unit"] == ""

    with pytest.raises(ValueError, match="只支持"):
        store.create(_metric(raw_unit="万亿"))
    with pytest.raises(ValueError, match="missing"):
        store.create(_metric(extraction_status="missing", raw_value="0", raw_unit="元"))
    with pytest.raises(ValueError, match="合法十进制"):
        store.create(_metric(raw_value="not-a-number", raw_unit="元"))


def test_filters_by_kb_company_period_and_metric(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    store.create(_metric())
    store.create(_metric(company_name="另一公司", report_period="2023", metric_code="total_assets"))
    store.create(_metric(kb_id="internal", company_name="测试公司"))

    items, total = store.list(
        kb_id="default", company_name="另一公司", report_period="2023", metric_code="total_assets"
    )
    assert total == 1
    assert len(items) == 1
    assert items[0]["kb_id"] == "default"

    items, total = store.list(kb_id="internal")
    assert total == 1
    assert items[0]["company_name"] == "测试公司"


def test_revision_keeps_old_new_snapshot_actor_and_reason(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    created = store.create(_metric())
    updated = store.update(
        created["id"],
        {"raw_value": "2", "raw_unit": "万元", "statement_scope": "parent"},
        actor="reviewer",
        reason="人工核对母公司报表并修正单位",
    )

    assert updated is not None
    assert updated["raw_value"] == "2"
    assert updated["normalized_value"] == "20000"
    assert updated["statement_scope"] == "parent"
    revisions = store.revisions(created["id"])
    assert len(revisions) == 1
    assert revisions[0]["actor"] == "reviewer"
    assert revisions[0]["reason"] == "人工核对母公司报表并修正单位"
    assert revisions[0]["old_snapshot"]["normalized_value"] == "125000000"
    assert revisions[0]["new_snapshot"]["normalized_value"] == "20000"


def test_not_found_get_and_update_are_explicit(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    assert store.get(9999) is None
    assert store.update(9999, {"raw_value": "1"}, actor="tester", reason="校验不存在记录") is None
