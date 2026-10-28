"""Tests for ai_memory.env.get_env (process env with user-scope fallback)."""
from __future__ import annotations

from ai_memory.env import get_env


def test_process_env_wins(monkeypatch):
    monkeypatch.setenv("AIMEM_TEST_KEY", "from-process")
    assert get_env("AIMEM_TEST_KEY", user_lookup=lambda _: "from-user") == "from-process"


def test_falls_back_to_user_scope(monkeypatch):
    monkeypatch.delenv("AIMEM_TEST_KEY", raising=False)
    assert get_env("AIMEM_TEST_KEY", user_lookup=lambda _: "from-user") == "from-user"


def test_empty_process_value_falls_back(monkeypatch):
    monkeypatch.setenv("AIMEM_TEST_KEY", "")
    assert get_env("AIMEM_TEST_KEY", user_lookup=lambda _: "from-user") == "from-user"


def test_missing_everywhere_returns_none(monkeypatch):
    monkeypatch.delenv("AIMEM_TEST_KEY", raising=False)
    assert get_env("AIMEM_TEST_KEY", user_lookup=lambda _: None) is None
    assert get_env("AIMEM_TEST_KEY", user_lookup=lambda _: "") is None


def test_default_lookup_handles_unknown_name(monkeypatch):
    monkeypatch.delenv("AIMEM_DEFINITELY_UNSET_9F2C", raising=False)
    assert get_env("AIMEM_DEFINITELY_UNSET_9F2C") is None
