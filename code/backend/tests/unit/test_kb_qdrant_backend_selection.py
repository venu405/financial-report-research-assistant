"""KB vector backend selection tests without external services."""
from __future__ import annotations

import io

import pytest
from loguru import logger
from pydantic import ValidationError

import main as main_mod
from config import Configuration
from services.kb.qdrant_vector_store import QdrantVectorStore


class _FakeChromaStore:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.migrate_calls = 0

    def migrate_default_kb_id(self):
        self.migrate_calls += 1


class _FakeQdrantStore:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _clear_backend_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "KB_VECTOR_BACKEND",
        "KB_QDRANT_URL",
        "KB_QDRANT_COLLECTION",
        "KB_QDRANT_API_KEY",
        "KB_QDRANT_VECTOR_SIZE",
        "KB_QDRANT_TIMEOUT",
        "KB_QDRANT_CREATE_IF_MISSING",
    ):
        monkeypatch.delenv(key, raising=False)


def test_default_backend_uses_chroma_and_migrates(monkeypatch: pytest.MonkeyPatch):
    _clear_backend_env(monkeypatch)
    config = Configuration.from_env()
    fake = _FakeChromaStore
    monkeypatch.setattr("services.kb.vector_store.VectorStore", fake)

    store = main_mod._build_kb_vector_store(config)

    assert isinstance(store, fake)
    assert store.kwargs == {
        "persist_dir": "./chroma_data",
        "collection_name": "enterprise_kb",
        "embedding_model": "bge-m3",
    }
    assert store.migrate_calls == 1


def test_explicit_qdrant_passes_all_parameters_and_does_not_create(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("KB_VECTOR_BACKEND", "qdrant")
    monkeypatch.setenv("KB_QDRANT_URL", "http://qdrant.test:6333")
    monkeypatch.setenv("KB_QDRANT_COLLECTION", "kb_staging_20260831")
    monkeypatch.setenv("KB_QDRANT_API_KEY", "qdrant-secret")
    monkeypatch.setenv("KB_QDRANT_VECTOR_SIZE", "1024")
    monkeypatch.setenv("KB_QDRANT_TIMEOUT", "12.5")
    monkeypatch.setenv("KB_QDRANT_CREATE_IF_MISSING", "false")
    config = Configuration.from_env()
    monkeypatch.setattr(
        "services.kb.qdrant_vector_store.QdrantVectorStore", _FakeQdrantStore
    )

    store = main_mod._build_kb_vector_store(config)

    assert store.kwargs == {
        "url": "http://qdrant.test:6333",
        "collection_name": "kb_staging_20260831",
        "vector_size": 1024,
        "api_key": "qdrant-secret",
        "timeout": 12.5,
        "create_if_missing": False,
    }


def test_qdrant_test_environment_forces_create_if_missing_false(
    monkeypatch: pytest.MonkeyPatch,
):
    config = Configuration(
        kb_vector_backend="qdrant",
        kb_qdrant_create_if_missing=True,
        app_env="testing",
    )
    monkeypatch.setattr(
        "services.kb.qdrant_vector_store.QdrantVectorStore", _FakeQdrantStore
    )

    store = main_mod._build_kb_vector_store(config)

    assert store.kwargs["create_if_missing"] is False


def test_invalid_backend_is_rejected(monkeypatch: pytest.MonkeyPatch):
    _clear_backend_env(monkeypatch)
    monkeypatch.setenv("KB_VECTOR_BACKEND", "weaviate")

    with pytest.raises(ValidationError):
        Configuration.from_env()


def test_qdrant_api_key_is_not_logged(monkeypatch: pytest.MonkeyPatch):
    secret = "qdrant-secret-not-in-log"
    config = Configuration(kb_vector_backend="qdrant", kb_qdrant_api_key=secret)
    monkeypatch.setattr(
        "services.kb.qdrant_vector_store.QdrantVectorStore", _FakeQdrantStore
    )
    output = io.StringIO()
    sink_id = logger.add(output, format="{message}")
    try:
        main_mod._build_kb_vector_store(config)
    finally:
        logger.remove(sink_id)

    assert secret not in output.getvalue()


def test_qdrant_healthcheck_uses_one_collection_info_request(
    monkeypatch: pytest.MonkeyPatch,
):
    store = object.__new__(QdrantVectorStore)
    calls = 0

    def collection_info():
        nonlocal calls
        calls += 1
        return {"config": {"params": {"vectors": {"size": 1024, "distance": "Cosine"}}}}

    monkeypatch.setattr(store, "_get_collection_info", collection_info)

    assert store.healthcheck() is True
    assert calls == 1


def test_ready_check_prefers_lightweight_healthcheck():
    class HealthcheckedStore:
        def healthcheck(self):
            return True

        def list_kbs(self):
            raise AssertionError("readyz must not scan Qdrant points")

    main_mod._check_kb_vector_store_ready(HealthcheckedStore())


def test_ready_check_falls_back_to_list_kbs_for_chroma():
    class LegacyStore:
        def __init__(self):
            self.calls = 0

        def list_kbs(self):
            self.calls += 1
            return ["default"]

    store = LegacyStore()
    main_mod._check_kb_vector_store_ready(store)

    assert store.calls == 1
