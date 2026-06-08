"""
OpenAI LLM provider for the dreaming module.

Uses the Responses-style chat completion API. Default model is `gpt-4o-mini`
which is cheap enough that running the dream cycle a few times a day costs
pennies. Swap to `gpt-4o` or `o4-mini` via config when better judgment is
worth more tokens.

Rate-limit handling: ordinary throttling (HTTP 429, "too many requests") is
retried with exponential backoff. A *permanent* 429 — OpenAI's
"Request too large ... tokens per min (TPM)", where a single request exceeds
the per-minute ceiling — is surfaced as a non-retryable LlmRateLimitError so
the caller reshapes the request (chunk / map-reduce) instead of hammering it.
"""
from __future__ import annotations

import logging
import os
import time

from openai import APIStatusError, OpenAI

from ai_memory.llm.interface import CompletionResult, LlmRateLimitError, Message

logger = logging.getLogger(__name__)


class OpenAILlm:
    """OpenAI-backed LLM for summarisation and fact extraction."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        *,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not set. Provide one via the OPENAI_API_KEY "
                "environment variable or the api_key argument."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_retries = max(0, max_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}"

    def complete(
        self, system: str, messages: list[Message], max_tokens: int = 4096
    ) -> CompletionResult:
        # OpenAI's chat API takes the system prompt as a leading message rather
        # than a top-level field, unlike Anthropic. Pre-pend it.
        api_messages = [{"role": "system", "content": system}]
        api_messages.extend({"role": m["role"], "content": m["content"]} for m in messages)

        attempt = 0
        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=api_messages,
                    max_tokens=max_tokens,
                )
                break
            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                body = _response_body(exc)
                if status == 429:
                    permanent = _is_permanent_rate_limit(body)
                    if permanent:
                        # Reshaping is the only way out — never retry as-is.
                        raise LlmRateLimitError(
                            f"OpenAI request too large for the per-minute token "
                            f"limit (model {self._model}): {body}",
                            permanent=True,
                        ) from exc
                    if attempt < self._max_retries:
                        delay = _retry_after_seconds(exc) or (
                            self._backoff_seconds * (2 ** attempt)
                        )
                        logger.warning(
                            "OpenAI 429 (transient); backing off %.1fs "
                            "(attempt %d/%d).",
                            delay, attempt + 1, self._max_retries,
                        )
                        time.sleep(delay)
                        attempt += 1
                        continue
                    raise LlmRateLimitError(
                        f"OpenAI rate limit not cleared after {self._max_retries} "
                        f"retries: {body}",
                        permanent=False,
                    ) from exc
                raise RuntimeError(
                    f"OpenAI chat completion failed with status {status}: {body}"
                ) from exc

        choice = response.choices[0]
        text = choice.message.content or ""
        # The OpenAI SDK exposes a `refusal` attribute on the message when the
        # model declines via the structured-refusal channel (separate from
        # content). It can be missing on older SDK versions, hence getattr.
        refusal = getattr(choice.message, "refusal", None) or ""
        finish_reason = choice.finish_reason or "stop"
        usage = response.usage
        return CompletionResult(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model_id=self.model_id,
            finish_reason=finish_reason,
            refusal=refusal,
        )


def _response_body(exc: APIStatusError) -> str:
    try:
        return exc.response.text  # type: ignore[union-attr]
    except Exception:
        return str(exc)


def _is_permanent_rate_limit(body: str) -> bool:
    """True only when a SINGLE request is too large to ever succeed.

    OpenAI phrases this specific, non-retryable condition as
    "Request too large for <model> ... on tokens per min (TPM)". We key ONLY on
    that distinctive phrase.

    The ordinary, transient throttle — "Rate limit reached ... please try again
    in Ns" — ALSO contains "tokens per min" and "Requested N", so a broader
    match would wrongly classify a recoverable throttle as permanent and abandon
    a request (and its whole episode) that would have succeeded after a short
    wait. When unsure we treat the 429 as transient and retry: chunking already
    guarantees we never *send* an oversized request, so a genuine permanent 429
    is nearly impossible at runtime anyway — erring toward retry is safe.
    """
    return "request too large" in body.lower()


def _retry_after_seconds(exc: APIStatusError) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")  # type: ignore[union-attr]
        return float(value) if value is not None else None
    except Exception:
        return None
