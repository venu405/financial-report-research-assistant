from __future__ import annotations

from fastapi.testclient import TestClient

from services.kb.financial_metric_store import FinancialMetricStore
from tests.test_api_kb import _make_client


def _metric(company_name: str, metric_code: str, value: str) -> dict:
    return {
        "kb_id": "default",
        "company_name": company_name,
        "company_code": f"{company_name}-code",
        "report_period": "2024",
        "period_type": "annual",
        "metric_code": metric_code,
        "metric_name": metric_code,
        "raw_value": value,
        "raw_unit": "元",
        "statement_scope": "consolidated",
        "source_doc_id": f"doc-{company_name}",
        "source_title": "2024 annual report",
        "source_page": 8,
        "source_page_end": 9,
        "source_chunk_id": f"chunk-{company_name}",
        "source_text": "",
        "extraction_status": "verified",
        "created_by": "test",
    }


def _authorized_client(monkeypatch, tmp_path) -> tuple[TestClient, str, str]:
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("comparison-admin", role="admin")
    auth.grant_access(user_id, "default")
    return _make_client(monkeypatch, tmp_path), user_id, token


def test_peer_comparison_http_contract_supports_repeated_filters(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("Alpha", "revenue", "-100"))
    store.create(_metric("Beta", "revenue", "50"))
    store.create(_metric("Alpha", "net_profit_parent", "0"))
    store.create(_metric("Beta", "net_profit_parent", "0"))

    client, _, token = _authorized_client(monkeypatch, tmp_path)
    response = client.get(
        "/kb/peer-comparison",
        params=[
            ("company_names", " Alpha "),
            ("company_names", "Beta"),
            ("report_period", "2024"),
            ("metric_codes", "total_assets"),
            ("metric_codes", "revenue"),
            ("metric_codes", "revenue"),
        ],
        headers={"X-Api-Token": token},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["report_period"] == "2024年度"
    assert body["companies"] == [
        {"name": "Alpha", "queried_name": "Alpha", "code": "Alpha-code", "found": True},
        {"name": "Beta", "queried_name": "Beta", "code": "Beta-code", "found": True},
    ]
    assert [metric["metric_code"] for metric in body["metrics"]] == [
        "revenue",
        "total_assets",
    ]
    revenue = body["metrics"][0]
    assert revenue["unit"] == "元"
    assert revenue["comparable"] is True
    assert [row["value"] for row in revenue["rows"]] == ["-100", "50"]
    assert [row["bar_percent"] for row in revenue["rows"]] == ["100", "50"]
    assert body["metrics"][1]["comparable"] is False
    assert "不代表整个行业" in body["selection_note"]
    assert "排名" not in body["selection_note"]


def test_peer_comparison_validates_selection_and_kb_access(monkeypatch, tmp_path):
    client, _, token = _authorized_client(monkeypatch, tmp_path)
    headers = {"X-Api-Token": token}

    missing = client.get(
        "/kb/peer-comparison",
        params={"company_names": "Alpha", "report_period": "2024"},
        headers=headers,
    )
    assert missing.status_code == 422

    one_company = client.get(
        "/kb/peer-comparison",
        params=[("company_names", "Alpha"), ("report_period", "2024")],
        headers=headers,
    )
    assert one_company.status_code == 422

    duplicate_company = client.get(
        "/kb/peer-comparison",
        params=[
            ("company_names", "Alpha"),
            ("company_names", " Alpha "),
            ("report_period", "2024"),
        ],
        headers=headers,
    )
    assert duplicate_company.status_code == 422

    invalid_metric = client.get(
        "/kb/peer-comparison",
        params=[
            ("company_names", "Alpha"),
            ("company_names", "Beta"),
            ("report_period", "2024"),
            ("metric_codes", "not-a-metric"),
        ],
        headers=headers,
    )
    assert invalid_metric.status_code == 422

    anonymous = client.get(
        "/kb/peer-comparison",
        params=[
            ("company_names", "Alpha"),
            ("company_names", "Beta"),
            ("report_period", "2024"),
        ],
    )
    assert anonymous.status_code == 200

    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    outsider, outsider_token = auth.create_user("comparison-outsider")
    assert outsider
    forbidden = client.get(
        "/kb/peer-comparison",
        params={
            "company_names": ["Alpha", "Beta"],
            "report_period": "2024",
            "user_id": outsider,
        },
        headers={"X-Api-Token": outsider_token},
    )
    assert forbidden.status_code == 403
