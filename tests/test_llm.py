"""Tests for LLM provider helpers — rate-limit classification.

These are pure-function tests over the 429 body classifier. The retry/backoff
loop itself talks to the network and is exercised manually, not here.
"""
from __future__ import annotations

from ai_memory.llm.interface import LlmRateLimitError
from ai_memory.llm.openai_llm import _is_permanent_rate_limit


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
