from __future__ import annotations

from fastapi.testclient import TestClient

from services.kb.financial_metric_store import FinancialMetricStore
from services.kb.peer_comparison import compare_companies
from services.kb.peer_comparison_brief import _template_brief
from tests.test_api_kb import _make_client


def _metric(company_name: str, value: str, *, scope: str = "consolidated") -> dict:
    return {
        "kb_id": "default", "company_name": company_name, "company_code": f"{company_name}-code",
        "report_period": "2024", "period_type": "annual", "metric_code": "revenue",
        "metric_name": "revenue", "raw_value": value, "raw_unit": "元",
        "statement_scope": scope, "source_doc_id": f"doc-{company_name}",
        "source_title": "2024 annual report", "source_page": 8, "source_page_end": 9,
        "source_chunk_id": f"chunk-{company_name}", "source_text": "",
        "extraction_status": "verified", "created_by": "test",
    }


def _authorized_client(monkeypatch, tmp_path) -> tuple[TestClient, str]:
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("brief-admin", role="admin")
    auth.grant_access(user_id, "default")
    return _make_client(monkeypatch, tmp_path), token


def _payload() -> dict:
    return {
        "kb_id": "default", "company_names": ["Alpha", "Beta"],
        "report_period": "2024", "metric_codes": ["revenue"],
    }


def test_peer_comparison_brief_llm_success(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("Alpha", "120"))
    store.create(_metric("Beta", "100"))
    client, token = _authorized_client(monkeypatch, tmp_path)

    response = client.post("/kb/peer-comparison/brief", json=_payload(), headers={"X-Api-Token": token})

    assert response.status_code == 200, response.text
    assert response.json() == {"brief": "ok", "brief_source": "llm"}


def test_peer_comparison_brief_llm_failure_falls_back_to_template(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("Alpha", "120"))
    store.create(_metric("Beta", "100", scope="parent"))
    monkeypatch.setattr(
        "services.kb.peer_comparison_brief._llm_invoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider down")),
    )
    client, token = _authorized_client(monkeypatch, tmp_path)

    response = client.post("/kb/peer-comparison/brief", json=_payload(), headers={"X-Api-Token": token})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["brief_source"] == "template"
    assert "Alpha：120" in body["brief"]
    assert "Beta：100" in body["brief"]
    assert "报表口径不一致" in body["brief"]
    assert "不构成投资建议" in body["brief"]


def test_peer_comparison_brief_timeout_falls_back_to_template(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("Alpha", "120"))
    store.create(_metric("Beta", "100"))
    monkeypatch.setattr(
        "services.kb.peer_comparison_brief._llm_invoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")),
    )
    client, token = _authorized_client(monkeypatch, tmp_path)

    response = client.post("/kb/peer-comparison/brief", json=_payload(), headers={"X-Api-Token": token})

    assert response.status_code == 200, response.text
    assert response.json()["brief_source"] == "template"


def test_peer_comparison_brief_missing_auth_is_401_for_private_kb(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_ENFORCE_KB_VISIBILITY", "1")
    client = _make_client(monkeypatch, tmp_path)

    response = client.post("/kb/peer-comparison/brief", json=_payload())

    assert response.status_code == 401


def test_template_brief_reports_difference_and_non_comparable_items(tmp_path):
    store = FinancialMetricStore(tmp_path / "metrics.db")
    store.create(_metric("Alpha", "120"))
    store.create(_metric("Beta", "100"))
    comparable = compare_companies(
        store, kb_id="default", company_names=["Alpha", "Beta"], report_period="2024", metric_codes=["revenue"]
    )
    brief = _template_brief(comparable)

    assert "领先方：Alpha，直接数值高于 Beta，差额为 20 元。" in brief

    store = FinancialMetricStore(tmp_path / "mismatch.db")
    store.create(_metric("Alpha", "120"))
    store.create(_metric("Beta", "100", scope="parent"))
    incomparable = compare_companies(
        store, kb_id="default", company_names=["Alpha", "Beta"], report_period="2024", metric_codes=["revenue"]
    )
    assert "不可比项说明" in _template_brief(incomparable)
