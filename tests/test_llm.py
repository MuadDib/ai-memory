"""Tests for LLM provider helpers — rate-limit classification.

These are pure-function tests over the 429 body classifier. The retry/backoff
loop itself talks to the network and is exercised manually, not here.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIStatusError

from ai_memory import daemon
from ai_memory.llm.interface import LlmQuotaError, LlmRateLimitError
from ai_memory.llm.openai_llm import OpenAILlm, _is_permanent_rate_limit, _is_quota_exhausted

_QUOTA_BODY = (
    '{"error": {"message": "You exceeded your current quota, please check your plan '
    'and billing details.", "type": "insufficient_quota", "code": "insufficient_quota"}}'
)


def test_quota_exhausted_bodies_are_recognised() -> None:
    assert _is_quota_exhausted(_QUOTA_BODY)
    assert _is_quota_exhausted('{"error": {"code": "credit_balance_exhausted"}}')
    assert not _is_quota_exhausted('{"error": {"code": "rate_limit_exceeded"}}')


def test_openai_quota_429_raises_at_once_without_retrying() -> None:
    llm = OpenAILlm(api_key="test-key", max_retries=5, backoff_seconds=0)
    response = httpx.Response(
        429, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        text=_QUOTA_BODY,
    )
    llm._client = MagicMock()
    llm._client.chat.completions.create.side_effect = APIStatusError(
        "quota", response=response, body=None
    )
    with pytest.raises(LlmQuotaError):
        llm.complete(system="s", messages=[{"role": "user", "content": "hi"}])
    assert llm._client.chat.completions.create.call_count == 1


def test_daemon_pauses_for_hours_on_quota_error(tmp_path: Path) -> None:
    service = SimpleNamespace(dream=MagicMock(side_effect=LlmQuotaError("out of credit")))
    state = daemon.DaemonRunState()
    log_path = tmp_path / "daemon.jsonl"

    daemon._run_pass(service, state, "scheduled", log_path)

    assert state.circuit_open_until - daemon.time.time() > 5 * 3600
    assert state.consecutive_failures == 0  # not a "stuck" pass
    events = [json.loads(line)["event"] for line in log_path.read_text().splitlines()]
    assert events == ["dream_start", "quota_exhausted"]


def test_permanent_rate_limit_request_too_large() -> None:
    """OpenAI's TPM 'request too large' is permanent — must be reshaped, not retried."""
    body = (
        '{"error": {"message": "Request too large for gpt-4o in organization '
        "org-xyz on tokens per min (TPM): Limit 30000, Requested 142133. The "
        'input or output tokens must be reduced.", "code": "rate_limit_exceeded"}}'
    )
    assert _is_permanent_rate_limit(body) is True


def test_transient_rate_limit_requests_per_min() -> None:
    """Ordinary throttling (requests per minute) is transient — retry with backoff."""
    body = (
        '{"error": {"message": "Rate limit reached for gpt-4o-mini ... '
        'Limit 500, please try again later.", "code": "rate_limit_exceeded"}}'
    )
    assert _is_permanent_rate_limit(body) is False


def test_transient_token_throttle_with_requested_is_not_permanent() -> None:
    """The dangerous case: a transient TOKEN throttle whose body also contains
    'tokens per min' and 'Requested' must NOT be misread as permanent. The
    request fits the per-minute ceiling; it just needs to wait for the window to
    refill, so the adapter must retry rather than abandon the episode."""
    body = (
        '{"error": {"message": "Rate limit reached for gpt-4o in organization '
        "org-xyz on tokens per min (TPM): Limit 30000, Used 28000, Requested 9000. "
        'Please try again in 3.2s.", "code": "rate_limit_exceeded"}}'
    )
    assert _is_permanent_rate_limit(body) is False


def test_rate_limit_error_carries_permanent_flag() -> None:
    err = LlmRateLimitError("too big", permanent=True, retry_after=12.0)
    assert err.permanent is True
    assert err.retry_after == 12.0
    assert isinstance(err, RuntimeError)
