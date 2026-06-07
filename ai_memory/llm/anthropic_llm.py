"""Anthropic LLM provider."""
from __future__ import annotations

import logging
import os
import time

from anthropic import Anthropic, APIStatusError

from ai_memory.llm.interface import CompletionResult, LlmRateLimitError, Message

logger = logging.getLogger(__name__)


class AnthropicLlm:
    """Anthropic-backed LLM for summarisation and fact extraction."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        *,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
    ) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Anthropic API key not set. Provide one via the ANTHROPIC_API_KEY "
                "environment variable or the api_key argument."
            )
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_retries = max(0, max_retries)
        self._backoff_seconds = max(0.0, backoff_seconds)

    @property
    def model_id(self) -> str:
        return f"anthropic:{self._model}"

    def complete(
        self, system: str, messages: list[Message], max_tokens: int = 4096
    ) -> CompletionResult:
        attempt = 0
        while True:
            try:
                response = self._client.messages.create(
                    model=self._model,
                    system=system,
                    messages=list(messages),
                    max_tokens=max_tokens,
                )
                break
            except APIStatusError as exc:
                status = getattr(exc, "status_code", None)
                # 429 (rate limit) and 529 (overloaded) are transient — back off.
                if status in (429, 529) and attempt < self._max_retries:
                    delay = _retry_after_seconds(exc) or (
                        self._backoff_seconds * (2 ** attempt)
                    )
                    logger.warning(
                        "Anthropic %s (transient); backing off %.1fs (attempt %d/%d).",
                        status, delay, attempt + 1, self._max_retries,
                    )
                    time.sleep(delay)
                    attempt += 1
                    continue
                if status in (429, 529):
                    raise LlmRateLimitError(
                        f"Anthropic rate limit not cleared after {self._max_retries} "
                        f"retries (status {status}).",
                        permanent=False,
                    ) from exc
                raise

        # Anthropic returns content as a list of blocks; we only consume text.
        text_parts = [block.text for block in response.content if getattr(block, "type", "") == "text"]
        # Map Anthropic's stop_reason vocabulary to the same finish_reason
        # buckets the OpenAI provider uses, so dreaming code can compare
        # without per-provider branches.
        anth_stop = getattr(response, "stop_reason", None) or "end_turn"
        finish_reason = {
            "end_turn": "stop",
            "stop_sequence": "stop",
            "max_tokens": "length",
            "tool_use": "tool_use",
        }.get(anth_stop, anth_stop)
        return CompletionResult(
            text="".join(text_parts),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model_id=self.model_id,
            finish_reason=finish_reason,
            refusal="",
        )


def _retry_after_seconds(exc: APIStatusError) -> float | None:
    try:
        value = exc.response.headers.get("retry-after")  # type: ignore[union-attr]
        return float(value) if value is not None else None
    except Exception:
        return None
