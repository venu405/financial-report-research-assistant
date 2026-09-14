from __future__ import annotations

from fastapi.testclient import TestClient

from tests.test_api_kb import _make_client


def _payload(**overrides):
    data = {
        "kb_id": "default",
        "company_name": "接口测试公司",
        "company_code": "600001",
        "report_period": "2024",
        "period_type": "annual",
        "metric_code": "revenue",
        "metric_name": "营业收入",
        "raw_value": "1.2",
        "raw_unit": "亿元",
        "statement_scope": "consolidated",
        "source_doc_id": "doc-api",
        "source_title": "接口测试年报",
        "source_page": 6,
        "source_page_end": 6,
        "source_chunk_id": "chunk-api",
        "source_text": "营业收入 1.2 亿元",
        "extraction_status": "verified",
    }
    data.update(overrides)
    return data


def _authorized_client(monkeypatch, tmp_path) -> tuple[TestClient, str, str]:
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("财务指标管理员", role="admin")
    auth.grant_access(user_id, "default")
    return _make_client(monkeypatch, tmp_path), user_id, token


def test_anonymous_read_and_unauthorized_write(monkeypatch, tmp_path):
    client = _make_client(monkeypatch, tmp_path)
    read = client.get("/kb/financial-metrics")
    write = client.post("/kb/financial-metrics", json=_payload())

    assert read.status_code == 200
    assert read.json() == {"items": [], "total": 0, "limit": 50, "offset": 0}
    assert write.status_code == 401


def test_authorized_create_filter_patch_and_revision_history(monkeypatch, tmp_path):
    client, user_id, token = _authorized_client(monkeypatch, tmp_path)
    headers = {"X-Api-Token": token}

    created = client.post("/kb/financial-metrics", json={**_payload(), "user_id": user_id}, headers=headers)
    assert created.status_code == 200
    item = created.json()["item"]
    assert item["created_by"] == user_id
    assert item["normalized_value"] == "120000000"

    listed = client.get(
        "/kb/financial-metrics",
        params={"company_name": "接口测试公司", "report_period": "2024", "metric_code": "revenue"},
        headers=headers,
    )
    assert listed.status_code == 200
    assert listed.json()["total"] == 1

    patched = client.patch(
        f"/kb/financial-metrics/{item['id']}",
        json={"raw_value": "2", "raw_unit": "万元", "reason": "人工复核单位", "user_id": user_id},
        headers=headers,
    )
    assert patched.status_code == 200
    assert patched.json()["item"]["normalized_value"] == "20000"

    revisions = client.get(f"/kb/financial-metrics/{item['id']}/revisions", headers=headers)
    assert revisions.status_code == 200
    assert revisions.json()["revisions"][0]["actor"] == user_id
    assert revisions.json()["revisions"][0]["reason"] == "人工复核单位"
    assert revisions.json()["revisions"][0]["old_snapshot"]["normalized_value"] == "120000000"


def test_forbidden_user_and_missing_record_semantics(monkeypatch, tmp_path):
    client, owner_id, owner_token = _authorized_client(monkeypatch, tmp_path)
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    outsider_id, outsider_token = auth.create_user("无权用户")
    created = client.post(
        "/kb/financial-metrics", json={**_payload(), "user_id": owner_id}, headers={"X-Api-Token": owner_token}
    )
    metric_id = created.json()["item"]["id"]

    forbidden = client.patch(
        f"/kb/financial-metrics/{metric_id}",
        json={"raw_value": "3", "raw_unit": "元", "reason": "越权测试", "user_id": outsider_id},
        headers={"X-Api-Token": outsider_token},
    )
    assert forbidden.status_code == 403

    missing = client.get(
        "/kb/financial-metrics/999999/revisions", headers={"X-Api-Token": owner_token}
    )
    assert missing.status_code == 404


def test_api_rejects_invalid_unit_and_missing_reason(monkeypatch, tmp_path):
    client, user_id, token = _authorized_client(monkeypatch, tmp_path)
    headers = {"X-Api-Token": token}
    invalid = client.post(
        "/kb/financial-metrics", json={**_payload(), "raw_unit": "亿", "user_id": user_id}, headers=headers
    )
    assert invalid.status_code == 422

    created = client.post("/kb/financial-metrics", json={**_payload(), "user_id": user_id}, headers=headers)
    metric_id = created.json()["item"]["id"]
    no_reason = client.patch(
        f"/kb/financial-metrics/{metric_id}",
        json={"raw_value": "3", "raw_unit": "元", "user_id": user_id},
        headers=headers,
    )
    assert no_reason.status_code == 422


def test_api_rejects_invalid_enums_and_preserves_missing_as_null(monkeypatch, tmp_path):
    client, user_id, token = _authorized_client(monkeypatch, tmp_path)
    headers = {"X-Api-Token": token}

    invalid_metric = client.post(
        "/kb/financial-metrics",
        json={**_payload(), "metric_code": "monthly_revenue", "user_id": user_id},
        headers=headers,
    )
    invalid_period = client.post(
        "/kb/financial-metrics",
        json={**_payload(), "period_type": "monthly", "user_id": user_id},
        headers=headers,
    )
    assert invalid_metric.status_code == 422
    assert invalid_period.status_code == 422

    missing = client.post(
        "/kb/financial-metrics",
        json={
            **_payload(),
            "metric_code": "operating_cash_flow",
            "extraction_status": "missing",
            "raw_value": None,
            "raw_unit": "",
            "user_id": user_id,
        },
        headers=headers,
    )
    assert missing.status_code == 200
    assert missing.json()["item"]["raw_value"] is None
    assert missing.json()["item"]["normalized_value"] is None


def test_list_supports_short_name_and_annual_period_shortcut(monkeypatch, tmp_path):
    from services.kb.auth import AuthStore

    auth = AuthStore(tmp_path / "kb_users.db")
    user_id, token = auth.create_user("metrics-viewer", role="admin")
    auth.grant_access(user_id, "default")
    client = _make_client(monkeypatch, tmp_path)
    headers = {"X-Api-Token": token}

    full = "苏州长光华芯光电技术股份有限公司"
    create = client.post("/kb/financial-metrics", json=_payload(company_name=full), headers=headers)
    assert create.status_code == 200, create.text

    response = client.get(
        "/kb/financial-metrics",
        params={"company_name": "长光华芯", "report_period": "2024年"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["company_name"] == full
