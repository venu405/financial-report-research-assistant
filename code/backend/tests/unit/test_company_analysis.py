from __future__ import annotations

import importlib.util
from pathlib import Path

from services.kb.company_analysis import analyze_company
from services.kb.financial_metric_store import FinancialMetricStore


METRICS = (
    "revenue",
    "net_profit_parent",
    "operating_cash_flow",
    "total_assets",
    "total_liabilities",
)


def _record(*, period="2024", code="revenue", value="100", company="测试公司", scope="consolidated", updated_by="tester"):
    return {
        "kb_id": "default",
        "company_name": company,
        "company_code": "",
        "report_period": period,
        "period_type": "annual",
        "metric_code": code,
        "metric_name": code,
        "raw_value": value,
        "raw_unit": "元",
        "statement_scope": scope,
        "source_doc_id": "doc-test",
        "source_title": "测试年报",
        "source_page": 6,
        "source_page_end": 7,
        "source_chunk_id": "chunk-test",
        "source_text": "原始证据片段",
        "extraction_status": "verified",
        "created_by": updated_by,
    }


def test_fixed_order_decimal_comparison_duplicate_and_cash_flow_relation(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    for code in METRICS:
        store.create(_record(period="2023", code=code, value="100"))
    store.create(_record(period="2024", code="revenue", value="110"))
    store.create(_record(period="2024", code="revenue", value="120"))
    store.create(_record(period="2024", code="net_profit_parent", value="30"))
    store.create(_record(period="2024", code="operating_cash_flow", value="20"))
    store.create(_record(period="2024", code="total_assets", value="1000"))
    store.create(_record(period="2024", code="total_liabilities", value="400"))
    store.update(1, {"source_text": "旧期间修订"}, actor="reviewer", reason="测试最新记录排序")

    result = analyze_company(
        store, kb_id="default", company_name="测试公司", report_period="2024", comparison_period="2023"
    )

    assert [item["metric_code"] for item in result["metrics"]] == list(METRICS)
    revenue = result["metrics"][0]
    assert revenue["current"]["raw_value"] == "120"
    assert revenue["change_amount"] == "20"
    assert revenue["change_rate_percent"] == "20"
    assert revenue["comparable"] is True
    assert any(item["code"] == "duplicate_metric" and item["metric_code"] == "revenue" for item in result["pending_items"])
    assert result["profit_cash_flow"] == {
        "net_profit_value": "30",
        "operating_cash_flow_value": "20",
        "difference": "-10",
        "relation": "lower",
        "comparable": True,
        "note": "经营活动现金流量净额低于归母净利润",
    }
    assert result["disclosures"][0]["source_text"] == "原始证据片段"


def test_zero_base_and_scope_mismatch_are_explicitly_unavailable(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    store.create(_record(period="2024", code="revenue", value="10"))
    store.create(_record(period="2023", code="revenue", value="0"))
    store.create(_record(period="2024", code="net_profit_parent", value="30", scope="consolidated"))
    store.create(_record(period="2024", code="operating_cash_flow", value="20", scope="parent"))
    store.create(_record(period="2024", code="total_assets", value="1000", scope="parent"))
    store.create(_record(period="2023", code="total_assets", value="900", scope="consolidated"))

    result = analyze_company(store, kb_id="default", company_name="测试公司", report_period="2024", comparison_period="2023")
    revenue = result["metrics"][0]
    assert revenue["change_amount"] == "10"
    assert revenue["change_rate_percent"] is None
    assert any(item["code"] == "zero_comparison_base" for item in result["pending_items"])
    assert result["profit_cash_flow"]["relation"] == "unavailable"
    assert result["profit_cash_flow"]["comparable"] is False
    assert any(item["code"] == "profit_cash_flow_not_comparable" for item in result["pending_items"])
    assert any(item["code"] == "statement_scope_mismatch" and item["metric_code"] == "total_assets" for item in result["pending_items"])


def test_derived_period_preserves_annual_suffix_and_unrecognized_period_stays_null(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    store.create(_record(period="2024年度", code="revenue", value="10"))
    derived = analyze_company(store, kb_id="default", company_name="测试公司", report_period="2024年度")
    assert derived["comparison_period"] == "2023年度"
    assert any(item["code"] == "comparison_period_not_found" for item in derived["pending_items"])

    store.create(_record(period="H1-2024", code="revenue", value="10"))
    unrecognized = analyze_company(store, kb_id="default", company_name="测试公司", report_period="H1-2024")
    assert unrecognized["comparison_period"] is None
    assert any(item["code"] == "comparison_period_unavailable" for item in unrecognized["pending_items"])


def test_seed_json_is_idempotent_in_a_temporary_database(tmp_path):
    backend = Path(__file__).resolve().parents[2]
    script_path = backend / "scripts" / "seed_financial_metrics.py"
    spec = importlib.util.spec_from_file_location("seed_financial_metrics", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    db_path = tmp_path / "seed.db"
    seed_path = backend / "data" / "financial_metrics_seed_2024.json"
    first = module.seed_metrics(seed_path, db_path, "default")
    second = module.seed_metrics(seed_path, db_path, "default")
    store = FinancialMetricStore(db_path)
    items, total = store.list(kb_id="default", limit=200)

    assert first == {"inserted": 15, "skipped": 0, "total": 15}
    assert second == {"inserted": 0, "skipped": 15, "total": 15}
    assert total == 15
    assert len(items) == 15
