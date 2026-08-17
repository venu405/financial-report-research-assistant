"""百度 OCR 后备通道测试：mock requests.post，不触发真实网络。

覆盖：未配置 key 静默关闭 / 结果解析 / 接口业务错误 / 网络异常降级 / token 缓存。
"""
from __future__ import annotations

import pytest

from services.kb import baidu_ocr


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """每个用例隔离环境变量、token 缓存与 QPS 计数。"""
    monkeypatch.delenv("BAIDU_OCR_API_KEY", raising=False)
    monkeypatch.delenv("BAIDU_OCR_SECRET_KEY", raising=False)
    baidu_ocr._token_cache["token"] = ""
    baidu_ocr._token_cache["expires_at"] = 0.0
    baidu_ocr._qps_hits.clear()  # P2-2：清理 QPS 计数，避免测试间限流状态泄漏
    yield


def _fake_post(token_payload, ocr_payload):
    def fake_post(url, **kwargs):
        class Resp:
            def raise_for_status(self):
                return None

            def json(self):
                if "oauth/2.0/token" in url:
                    return token_payload
                return ocr_payload

        return Resp()

    return fake_post


def test_baidu_unconfigured_returns_empty(monkeypatch):
    assert baidu_ocr.is_configured() is False
    assert baidu_ocr.ocr_image(b"x") == ""


def test_baidu_parse_words_result(monkeypatch):
    monkeypatch.setenv("BAIDU_OCR_API_KEY", "ak")
    monkeypatch.setenv("BAIDU_OCR_SECRET_KEY", "sk")
    monkeypatch.setattr(
        baidu_ocr.requests,
        "post",
        _fake_post(
            {"access_token": "tok", "expires_in": 3600},
            {"words_result": [{"words": "月复一月"}, {"words": "年复一年"}]},
        ),
    )
    assert baidu_ocr.ocr_image(b"fake-image") == "月复一月\n年复一年"


def test_baidu_api_error_returns_empty(monkeypatch):
    """接口返回业务错误码 → 返回空串（降级），不抛异常。"""
    monkeypatch.setenv("BAIDU_OCR_API_KEY", "ak")
    monkeypatch.setenv("BAIDU_OCR_SECRET_KEY", "sk")
    monkeypatch.setattr(
        baidu_ocr.requests,
        "post",
        _fake_post(
            {"access_token": "tok", "expires_in": 3600},
            {"error_code": 17, "error_msg": "Open api daily request limit reached"},
        ),
    )
    assert baidu_ocr.ocr_image(b"x") == ""


def test_baidu_network_error_degrades(monkeypatch):
    """网络异常 → 返回空串（静默降级），不抛异常。"""
    monkeypatch.setenv("BAIDU_OCR_API_KEY", "ak")
    monkeypatch.setenv("BAIDU_OCR_SECRET_KEY", "sk")

    def boom(url, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(baidu_ocr.requests, "post", boom)
    assert baidu_ocr.ocr_image(b"x") == ""


def test_access_token_cached(monkeypatch):
    """同一进程内重复调用只取一次 token（过期前走缓存）。"""
    monkeypatch.setenv("BAIDU_OCR_API_KEY", "ak")
    monkeypatch.setenv("BAIDU_OCR_SECRET_KEY", "sk")
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)

        class Resp:
            def raise_for_status(self):
                return None

            def json(self):
                if "oauth/2.0/token" in url:
                    return {"access_token": "tok", "expires_in": 3600}
                return {"words_result": [{"words": "hi"}]}

        return Resp()

    monkeypatch.setattr(baidu_ocr.requests, "post", fake_post)
    assert baidu_ocr.ocr_image(b"x") == "hi"
    assert baidu_ocr.ocr_image(b"x") == "hi"
    token_url_calls = [u for u in calls if "oauth/2.0/token" in u]
    assert len(token_url_calls) == 1
