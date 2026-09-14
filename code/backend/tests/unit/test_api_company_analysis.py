from __future__ import annotations

from fastapi.testclient import TestClient

from tests.test_api_kb import _make_client


def _metric_payload(*, metric_code: str, value: str, period: str, user_id: str) -> dict:
    return {
        "kb_id": "default",
        "company_name": "API分析公司",
        "company_code": "",
        "report_period": period,
        "period_type": "annual",
        "metric_code": metric_code,
        "metric_name": metric_code,
        "raw_value": value,
        "raw_unit": "元",
        "statement_scope": "consolidated",
        "source_doc_id": "doc-api-analysis",
        "source_title": "API测试年报",
        "source_page": 6,
        "source_page_end": 7,
        "source_chunk_id": "chunk-api-analysis",
        "source_text": "原始证据",
        "extraction_status": "verified",
        "user_id": user_id,
    }


def _authorized_client(monkeypatch, tmp_path) -> tuple[TestClient, str, str]:
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("分析管理员", role="admin")
    auth.grant_access(user_id, "default")
    return _make_client(monkeypatch, tmp_path), user_id, token


def test_company_analysis_http_contract_and_permission(monkeypatch, tmp_path):
    client, user_id, token = _authorized_client(monkeypatch, tmp_path)
    headers = {"X-Api-Token": token}
    for period, value in (("2023", "100"), ("2024", "120")):
        created = client.post(
            "/kb/financial-metrics",
            json=_metric_payload(metric_code="revenue", value=value, period=period, user_id=user_id),
            headers=headers,
        )
        assert created.status_code == 200

    response = client.get(
        "/kb/company-analysis",
        params={"company_name": "API分析公司", "report_period": "2024", "comparison_period": "2023"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["company"] == {"name": "API分析公司", "queried_name": "API分析公司", "code": None}
    assert body["report_period"] == "2024年度"
    assert body["comparison_period"] == "2023"
    assert [item["metric_code"] for item in body["metrics"]] == [
        "revenue", "net_profit_parent", "operating_cash_flow", "total_assets", "total_liabilities"
    ]
    assert body["metrics"][0]["change_amount"] == "20"
    assert body["metrics"][0]["change_rate_percent"] == "20"
    assert body["metrics"][0]["comparable"] is True
    assert body["profit_cash_flow"]["relation"] == "unavailable"
    assert any(item["code"] == "missing_current_metric" for item in body["pending_items"])

    anonymous = client.get(
        "/kb/company-analysis",
        params={"company_name": "API分析公司", "report_period": "2024"},
    )
    assert anonymous.status_code == 200


def test_company_analysis_requires_query_parameters_and_respects_kb_access(monkeypatch, tmp_path):
    client, user_id, token = _authorized_client(monkeypatch, tmp_path)
    missing_query = client.get("/kb/company-analysis", headers={"X-Api-Token": token})
    assert missing_query.status_code == 422

    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    outsider, outsider_token = auth.create_user("分析无权用户")
    assert outsider
    forbidden = client.get(
        "/kb/company-analysis",
        params={"company_name": "API分析公司", "report_period": "2024", "user_id": outsider},
        headers={"X-Api-Token": outsider_token},
    )
    assert forbidden.status_code == 403
