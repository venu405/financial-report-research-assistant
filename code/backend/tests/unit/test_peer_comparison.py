from __future__ import annotations

import pytest

from services.kb.financial_metric_store import FinancialMetricStore
from services.kb.peer_comparison import (
    METRIC_ORDER,
    compare_companies,
    normalize_company_names,
    normalize_metric_codes,
)


def _metric(
    company_name: str,
    metric_code: str,
    value: str | None,
    *,
    company_code: str = "",
    scope: str = "consolidated",
    period_type: str = "annual",
    status: str = "verified",
) -> dict:
    return {
        "kb_id": "default",
        "company_name": company_name,
        "company_code": company_code,
        "report_period": "2024",
        "period_type": period_type,
        "metric_code": metric_code,
        "metric_name": metric_code,
        "raw_value": value,
        "raw_unit": "元" if value is not None else "",
        "statement_scope": scope,
        "source_doc_id": f"doc-{company_name}",
        "source_title": "2024 annual report",
        "source_page": 10,
        "source_page_end": 11,
        "source_chunk_id": f"chunk-{company_name}",
        "source_text": "",
        "extraction_status": status,
        "created_by": "test",
    }


def test_two_company_comparison_keeps_order_negative_values_and_zero_bars(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    for company, code, value in (
        ("Alpha", "revenue", "-100"),
        ("Beta", "revenue", "50"),
        ("Alpha", "net_profit_parent", "0"),
        ("Beta", "net_profit_parent", "0"),
    ):
        store.create(_metric(company, code, value, company_code=f"{company}-code"))

    result = compare_companies(
        store,
        kb_id="default",
        company_names=[" Alpha ", "Beta"],
        report_period="2024",
        metric_codes=["total_assets", "net_profit_parent", "revenue", "revenue"],
    )

    assert [item["name"] for item in result["companies"]] == ["Alpha", "Beta"]
    assert [item["metric_code"] for item in result["metrics"]] == [
        "revenue",
        "net_profit_parent",
        "total_assets",
    ]
    revenue = result["metrics"][0]
    assert revenue["comparable"] is True
    assert [row["value"] for row in revenue["rows"]] == ["-100", "50"]
    assert [row["bar_percent"] for row in revenue["rows"]] == ["100", "50"]

    zero_profit = result["metrics"][1]
    assert zero_profit["comparable"] is True
    assert [row["bar_percent"] for row in zero_profit["rows"]] == ["0", "0"]


def test_three_company_missing_company_and_metric_are_explicit(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    store.create(_metric("Alpha", "revenue", "100"))
    store.create(_metric("Beta", "revenue", "200"))

    result = compare_companies(
        store,
        kb_id="default",
        company_names=["Alpha", "Beta", "Gamma"],
        report_period="2024",
        metric_codes=["revenue"],
    )

    assert len(result["companies"]) == 3
    assert result["companies"][2] == {"name": "Gamma", "code": None, "found": False}
    metric = result["metrics"][0]
    assert metric["comparable"] is False
    assert metric["rows"][-1]["value"] is None
    assert metric["rows"][-1]["comparable"] is False
    assert {item["code"] for item in result["pending_items"]} >= {
        "missing_company",
        "missing_metric",
    }


def test_duplicate_latest_record_and_scope_mismatch_are_reported(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    store.create(_metric("Alpha", "revenue", "10", scope="consolidated"))
    store.create(_metric("Alpha", "revenue", "20", scope="consolidated"))
    store.create(_metric("Beta", "revenue", "30", scope="parent"))

    result = compare_companies(
        store,
        kb_id="default",
        company_names=["Alpha", "Beta"],
        report_period="2024",
        metric_codes=["revenue"],
    )

    metric = result["metrics"][0]
    assert metric["comparable"] is False
    assert metric["rows"][0]["value"] == "20"
    assert metric["rows"][0]["statement_scope"] == "consolidated"
    assert metric["rows"][1]["statement_scope"] == "parent"
    assert any(item["code"] == "duplicate_metric" for item in result["pending_items"])
    assert any(item["code"] == "statement_scope_mismatch" for item in result["pending_items"])
    assert all(row["bar_percent"] is None for row in metric["rows"])


def test_parameter_validation_and_default_metric_order():
    with pytest.raises(ValueError):
        normalize_company_names(["Only one"])
    with pytest.raises(ValueError):
        normalize_company_names(["Alpha", " Alpha "])
    with pytest.raises(ValueError):
        normalize_company_names(["Alpha", ""])
    with pytest.raises(ValueError):
        normalize_metric_codes(["revenue", "unknown"])
    with pytest.raises(ValueError):
        normalize_metric_codes(["revenue", ""])

    assert normalize_metric_codes(None) == list(METRIC_ORDER)
    assert normalize_metric_codes(["total_assets", "revenue", "total_assets"]) == [
        "revenue",
        "total_assets",
    ]
