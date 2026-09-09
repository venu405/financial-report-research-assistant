from __future__ import annotations

import re

from fastapi.testclient import TestClient

from services.kb.financial_metric_store import FinancialMetricStore
from tests.test_api_kb import _make_client


def _metric(company_name: str, metric_code: str, value: str, *, period: str = "2024") -> dict:
    return {
        "kb_id": "default",
        "company_name": company_name,
        "company_code": f"code-{company_name}",
        "report_period": period,
        "period_type": "annual",
        "metric_code": metric_code,
        "metric_name": metric_code,
        "raw_value": value,
        "raw_unit": "元",
        "statement_scope": "consolidated",
        "source_doc_id": f"doc-{company_name}",
        "source_title": f"{company_name} 2024年报",
        "source_page": 12,
        "source_page_end": 13,
        "source_chunk_id": f"chunk-{company_name}",
        "source_text": f"{company_name}营业收入原文",
        "extraction_status": "verified",
        "created_by": "test",
    }


def _authorized_client(monkeypatch, tmp_path) -> tuple[TestClient, str, str]:
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("export-admin", role="admin")
    auth.grant_access(user_id, "default")
    return _make_client(monkeypatch, tmp_path), user_id, token


def test_company_analysis_export_returns_utf8_markdown_and_missing_values(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("华微公司", "revenue", "-100"))
    store.close()
    client, _, token = _authorized_client(monkeypatch, tmp_path)

    response = client.get(
        "/kb/company-analysis/export",
        params={
            "kb_id": "default",
            "company_name": "华微公司",
            "report_period": "2024",
            "comparison_period": "2023",
        },
        headers={"X-Api-Token": token},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"] == 'attachment; filename="company-analysis-export.md"'
    assert re.search(r"UTC生成时间：\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", response.text)
    assert "华微公司" in response.text
    assert "-100" in response.text
    assert "2024年报" in response.text
    assert "未提供/不可计算" in response.text
    assert "不构成投资建议" in response.text
    assert not list(tmp_path.glob("*.md"))


def test_peer_comparison_export_supports_repeated_query_and_permission(monkeypatch, tmp_path):
    store = FinancialMetricStore(tmp_path / "kb_financial_metrics.db")
    store.create(_metric("甲公司", "revenue", "-100"))
    store.create(_metric("乙公司", "revenue", "50"))
    store.close()
    client, _, token = _authorized_client(monkeypatch, tmp_path)

    response = client.get(
        "/kb/peer-comparison/export",
        params=[
            ("company_names", "甲公司"),
            ("company_names", "乙公司"),
            ("report_period", "2024"),
            ("metric_codes", "total_assets"),
            ("metric_codes", "revenue"),
            ("metric_codes", "revenue"),
        ],
        headers={"X-Api-Token": token},
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.headers["content-disposition"] == 'attachment; filename="peer-comparison-export.md"'
    assert "甲公司、乙公司" in response.text
    assert "-100" in response.text
    assert "不代表整个行业" in response.text
    assert "未提供/不可计算" in response.text
    assert "排名" not in response.text

    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    outsider, outsider_token = auth.create_user("export-outsider")
    assert outsider
    forbidden = client.get(
        "/kb/peer-comparison/export",
        params=[
            ("company_names", "甲公司"),
            ("company_names", "乙公司"),
            ("report_period", "2024"),
            ("user_id", outsider),
        ],
        headers={"X-Api-Token": outsider_token},
    )
    assert forbidden.status_code == 403

    company_missing_query = client.get(
        "/kb/company-analysis/export",
        headers={"X-Api-Token": token},
    )
    assert company_missing_query.status_code == 422
