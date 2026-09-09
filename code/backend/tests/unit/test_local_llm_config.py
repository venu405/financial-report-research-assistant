"""本地 OpenAI 兼容 LLM 配置测试。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import Configuration


def test_reasoning_effort_is_unset_by_default(monkeypatch):
    monkeypatch.delenv("LLM_REASONING_EFFORT", raising=False)

    config = Configuration.from_env()

    assert config.llm_reasoning_effort is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(" none ", "none"), ("LOW", "low"), ("medium", "medium"), ("high", "high")],
)
def test_reasoning_effort_is_normalized(monkeypatch, raw, expected):
    monkeypatch.setenv("LLM_REASONING_EFFORT", raw)

    config = Configuration.from_env()

    assert config.llm_reasoning_effort == expected


def test_empty_reasoning_effort_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "  ")

    config = Configuration.from_env()

    assert config.llm_reasoning_effort is None


def test_invalid_reasoning_effort_fails_with_clear_message(monkeypatch):
    monkeypatch.setenv("LLM_REASONING_EFFORT", "verbose")

    with pytest.raises(ValidationError, match="LLM_REASONING_EFFORT"):
        Configuration.from_env()
