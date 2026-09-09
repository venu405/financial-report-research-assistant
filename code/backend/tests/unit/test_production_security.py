import pytest

from config import Configuration
from main import _validate_production_security


def test_development_allows_local_passwordless_setup(monkeypatch):
    monkeypatch.delenv("KB_REQUIRE_TOKEN", raising=False)
    _validate_production_security(Configuration(app_env="development"))


def test_production_requires_admin_key_and_tokens(monkeypatch):
    monkeypatch.delenv("KB_REQUIRE_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ADMIN_API_KEY.*KB_REQUIRE_TOKEN"):
        _validate_production_security(Configuration(app_env="production"))


def test_production_accepts_complete_security_configuration(monkeypatch):
    monkeypatch.setenv("KB_REQUIRE_TOKEN", "1")
    monkeypatch.setenv("KB_BACKUP_SIGNING_KEY", "x" * 32)
    monkeypatch.setenv("KB_REQUIRE_BACKUP_SIGNATURE", "1")
    config = Configuration(app_env="production", admin_api_key="a-real-random-secret")
    _validate_production_security(config)


def test_rag_v2_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("KB_RECALL_K", "24")
    monkeypatch.setenv("KB_MAX_HITS_PER_DOC", "2")
    monkeypatch.setenv("KB_MIN_SIMILARITY", "0.42")
    monkeypatch.setenv("KB_PARENT_CHILD_ENABLED", "0")
    monkeypatch.setenv("KB_PARENT_CHUNK_SIZE", "1800")
    monkeypatch.setenv("KB_NEIGHBOR_EXPANSION", "2")

    config = Configuration.from_env()

    assert config.kb_recall_k == 24
    assert config.kb_max_hits_per_doc == 2
    assert config.kb_min_similarity == pytest.approx(0.42)
    assert config.kb_parent_child_enabled is False
    assert config.kb_parent_chunk_size == 1800
    assert config.kb_neighbor_expansion == 2
