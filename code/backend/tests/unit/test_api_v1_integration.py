from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from services.kb.auth import AuthStore
from services.kb.financial_metric_store import FinancialMetricStore
from tests.test_api_kb import _make_client


@pytest.fixture(autouse=True)
def _isolate_legacy_token_flag(monkeypatch):
    """v1 必须独立强制 token；测试显式隔离项目 .env 的旧开关。"""
    monkeypatch.setenv("KB_REQUIRE_TOKEN", "0")


def _metric(company_name: str, code: str = "revenue", value: str = "100") -> dict:
    return {
        "kb_id": "default",
        "company_name": company_name,
        "company_code": company_name,
        "report_period": "2024",
        "period_type": "annual",
        "metric_code": code,
        "metric_name": code,
        "raw_value": value,
        "raw_unit": "元",
        "statement_scope": "consolidated",
        "source_doc_id": f"doc-{company_name}",
        "source_title": f"{company_name}年报",
        "source_page": 1,
        "source_text": "来源原文",
        "extraction_status": "verified",
        "created_by": "test",
    }


def _authorized_client(monkeypatch, tmp_path) -> tuple[TestClient, str]:
    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("v1-admin", role="admin")
    auth.grant_access(user_id, "default")
    return _make_client(monkeypatch, tmp_path), token


def test_v1_query_supports_bearer_and_request_id(monkeypatch, tmp_path):
    client, token = _authorized_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/v1/query",
        headers={"Authorization": f"Bearer {token}", "X-Request-Id": "client-req-1"},
        json={"knowledge_base_id": "default", "question": "收入是多少？"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "测试回答"
    assert "contexts" not in response.json()
    assert "search_meta" not in response.json()
    assert "score" not in response.json()
    assert response.json()["request_id"] == "client-req-1"
    assert response.headers["X-Request-Id"] == "client-req-1"

    alias = client.post(
        "/api/v1/queries",
        headers={"X-Api-Token": token},
        json={"knowledge_base_id": "default", "question": "别名"},
    )
    assert alias.status_code == 200
    assert alias.json()["knowledge_base_id"] == "default"


def test_v1_query_preserves_needs_clarification_business_flag(monkeypatch, tmp_path):
    import tests.test_api_kb as kb_tests
    original_result = kb_tests._result

    def clarified_result(escalate: bool = False) -> dict:
        result = original_result(escalate)
        result["needs_clarification"] = True
        return result

    monkeypatch.setattr(kb_tests, "_result", clarified_result)
    client, token = _authorized_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/v1/query",
        headers={"X-Api-Token": token},
        json={"knowledge_base_id": "default", "question": "需要澄清吗？"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["needs_clarification"] is True
    assert "search_meta" not in response.json()


def test_v1_financial_and_analysis_enforce_kb_permission(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("甲公司"))
    store.close()
    client, token = _authorized_client(monkeypatch, tmp_path)

    metrics = client.get(
        "/api/v1/financial-metrics",
        headers={"X-Api-Token": token},
        params={"knowledge_base_id": "default", "company_name": "甲公司"},
    )
    assert metrics.status_code == 200, metrics.text
    assert metrics.json()["items"][0]["company_name"] == "甲公司"
    assert metrics.json()["request_id"]

    outsider_auth = AuthStore(tmp_path / "kb_users.db")
    _, outsider_token = outsider_auth.create_user("outsider")
    denied = client.post(
        "/api/v1/company-analyses",
        headers={"Authorization": f"Bearer {outsider_token}"},
        json={
            "knowledge_base_id": "default",
            "company_name": "甲公司",
            "report_period": "2024",
        },
    )
    assert denied.status_code == 403


def test_v1_stream_and_export_have_request_id(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("甲公司"))
    store.create(_metric("乙公司", value="200"))
    store.close()
    client, token = _authorized_client(monkeypatch, tmp_path)
    headers = {"X-Api-Token": token, "X-Request-Id": "stream-req"}

    stream = client.post(
        "/api/v1/query/stream",
        headers=headers,
        json={"knowledge_base_id": "default", "question": "请回答"},
    )
    assert stream.status_code == 200, stream.text
    assert stream.headers["X-Request-Id"] == "stream-req"
    assert '"request_id": "stream-req"' in stream.text

    exported = client.post(
        "/api/v1/reports/export",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "report_type": "peer_comparison",
            "knowledge_base_id": "default",
            "company_names": ["甲公司", "乙公司"],
            "report_period": "2024",
            "metric_codes": ["revenue"],
        },
    )
    assert exported.status_code == 200, exported.text
    assert exported.json()["filename"] == "peer-comparison-export.md"
    assert exported.json()["request_id"]


def test_v1_company_analysis_success_and_peer_validation(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("甲公司"))
    store.create(_metric("乙公司", value="200"))
    store.close()
    client, token = _authorized_client(monkeypatch, tmp_path)
    headers = {"Authorization": f"Bearer {token}"}

    company = client.post(
        "/api/v1/company-analyses",
        headers=headers,
        json={"knowledge_base_id": "default", "company_name": "甲公司", "report_period": "2024"},
    )
    assert company.status_code == 200, company.text
    assert company.json()["knowledge_base_id"] == "default"
    assert company.json()["request_id"]

    duplicate = client.post(
        "/api/v1/peer-comparisons",
        headers=headers,
        json={
            "knowledge_base_id": "default",
            "company_names": ["甲公司", "甲公司"],
            "report_period": "2024",
        },
    )
    assert duplicate.status_code == 422


def test_v1_credentials_and_input_limits(monkeypatch, tmp_path):
    client, token = _authorized_client(monkeypatch, tmp_path)
    other_auth = AuthStore(tmp_path / "kb_users.db")
    _, other_token = other_auth.create_user("other", role="admin")
    conflicting = client.post(
        "/api/v1/query",
        headers={"X-Api-Token": token, "Authorization": f"Bearer {other_token}"},
        json={"knowledge_base_id": "default", "question": "x"},
    )
    assert conflicting.status_code == 401

    too_long_id = client.post(
        "/api/v1/query",
        headers={"X-Api-Token": token, "X-Request-Id": "!" * 129},
        json={"knowledge_base_id": "default", "question": "x"},
    )
    assert too_long_id.status_code == 200
    generated_id = too_long_id.headers["X-Request-Id"]
    assert generated_id != "!" * 129
    assert len(generated_id) == 32
    assert too_long_id.json()["request_id"] == generated_id

    too_long_question = client.post(
        "/api/v1/query",
        headers={"X-Api-Token": token},
        json={"knowledge_base_id": "default", "question": "x" * 8001},
    )
    assert too_long_question.status_code == 422
    bad_filter = client.post(
        "/api/v1/query",
        headers={"X-Api-Token": token},
        json={"knowledge_base_id": "default", "question": "x", "metadata_filters": {"unsafe": 1}},
    )
    assert bad_filter.status_code == 422


def test_v1_sse_error_is_sanitized_and_request_id_is_consistent(monkeypatch, tmp_path):
    from tests.test_api_kb import MockGraph

    original_stream = MockGraph.stream

    def fail_stream(self, *args, **kwargs):
        raise RuntimeError("secret backend address")

    monkeypatch.setattr(MockGraph, "stream", fail_stream)
    client, token = _authorized_client(monkeypatch, tmp_path)
    response = client.post(
        "/api/v1/query/stream",
        headers={"X-Api-Token": token, "X-Request-Id": "sse-err"},
        json={"knowledge_base_id": "default", "question": "x"},
    )
    assert response.status_code == 200
    assert "secret backend address" not in response.text
    events = [json.loads(part[5:]) for part in response.text.split("\n\n") if part.startswith("data:")]
    assert events and all(event["request_id"] == "sse-err" for event in events)
    assert events[-1]["code"] == "INTERNAL_ERROR"
    monkeypatch.setattr(MockGraph, "stream", original_stream)


def test_v1_rejects_malformed_authorization(monkeypatch, tmp_path):
    client, _ = _authorized_client(monkeypatch, tmp_path)
    response = client.get(
        "/api/v1/financial-metrics",
        headers={"Authorization": "Basic not-bearer"},
        params={"knowledge_base_id": "default"},
    )
    assert response.status_code == 401


def test_v1_metrics_query_limits_and_openapi_contract(monkeypatch, tmp_path):
    client, token = _authorized_client(monkeypatch, tmp_path)
    headers = {"X-Api-Token": token}
    for key, value in {
        "knowledge_base_id": " " * 2,
        "company_name": " " * 2,
        "report_period": " " * 2,
        "metric_code": " " * 2,
        "user_id": " " * 2,
    }.items():
        response = client.get(
            "/api/v1/financial-metrics",
            headers=headers,
            params={"knowledge_base_id": "default", key: value},
        )
        assert response.status_code == 422, (key, response.text)

    too_long = client.get(
        "/api/v1/financial-metrics",
        headers=headers,
        params={"knowledge_base_id": "default", "company_name": "x" * 10000},
    )
    assert too_long.status_code == 422

    schema = client.get("/openapi.json").json()
    schemes = schema["components"]["securitySchemes"]
    assert "HTTPBearer" in schemes
    assert "APIKeyHeader" in schemes
    stream_schema = schema["paths"]["/api/v1/query/stream"]["post"]
    assert stream_schema["responses"]["200"]["content"]["text/event-stream"]
    assert "401" in stream_schema["responses"]
    assert "403" in stream_schema["responses"]
