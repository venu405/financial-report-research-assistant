from __future__ import annotations

from fastapi.testclient import TestClient

from services.kb.financial_metric_store import FinancialMetricStore
from tests.test_api_kb import _make_client


def _metric(company_name: str, metric_code: str, value: str) -> dict:
    return {
        "kb_id": "default", "company_name": company_name, "company_code": f"{company_name}-code",
        "report_period": "2024年度", "period_type": "annual", "metric_code": metric_code,
        "metric_name": metric_code, "raw_value": value, "raw_unit": "元",
        "statement_scope": "consolidated", "source_doc_id": f"doc-{company_name}",
        "source_title": "2024 annual report", "source_page": 8, "source_page_end": 9,
        "source_chunk_id": f"chunk-{company_name}", "source_text": "",
        "extraction_status": "verified", "created_by": "test",
    }


def _authorized_client(monkeypatch, tmp_path) -> tuple[TestClient, str]:
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("analysis-brief-admin", role="admin")
    auth.grant_access(user_id, "default")
    return _make_client(monkeypatch, tmp_path), token


def _payload() -> dict:
    return {"kb_id": "default", "company_name": "Alpha", "report_period": "2024"}


def test_company_analysis_brief_llm_success(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    for code in ("revenue", "net_profit_parent", "operating_cash_flow", "total_assets", "total_liabilities"):
        store.create(_metric("Alpha", code, "100"))
    client, token = _authorized_client(monkeypatch, tmp_path)

    response = client.post("/kb/company-analysis/brief", json=_payload(), headers={"X-Api-Token": token})

    assert response.status_code == 200, response.text
    assert response.json() == {"brief": "ok", "brief_source": "llm"}


def test_company_analysis_brief_llm_failure_falls_back_to_template(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("Alpha", "revenue", "120"))
    monkeypatch.setattr(
        "services.kb.company_analysis_brief._llm_invoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("provider down")),
    )
    client, token = _authorized_client(monkeypatch, tmp_path)

    response = client.post("/kb/company-analysis/brief", json=_payload(), headers={"X-Api-Token": token})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["brief_source"] == "template"
    assert "单公司分析简报" in body["brief"]
    assert "不包含预测、评分或投资建议" in body["brief"]
    assert "缺少" in body["brief"]  # 其余四项缺失列入待核实


def test_company_analysis_brief_resolves_short_name_and_period(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    full = "苏州长光华芯光电技术股份有限公司"
    for code in ("revenue", "net_profit_parent", "operating_cash_flow", "total_assets", "total_liabilities"):
        store.create(_metric(full, code, "100"))
    client, token = _authorized_client(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "services.kb.company_analysis_brief._llm_invoke",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("force template")),
    )

    response = client.post(
        "/kb/company-analysis/brief",
        json={"kb_id": "default", "company_name": "长光华芯", "report_period": "2024年"},
        headers={"X-Api-Token": token},
    )

    assert response.status_code == 200, response.text
    assert full in response.json()["brief"]


def test_company_analysis_brief_missing_auth_is_401_for_private_kb(monkeypatch, tmp_path):
    monkeypatch.setenv("KB_ENFORCE_KB_VISIBILITY", "1")
    client = _make_client(monkeypatch, tmp_path)

    response = client.post("/kb/company-analysis/brief", json=_payload())

    assert response.status_code == 401
