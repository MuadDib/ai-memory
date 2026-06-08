"""
LLM interface — used by the dreaming module to summarise episodes and
extract atomic facts.

Kept minimal:
    - One method: complete(system, messages) -> CompletionResult
    - CompletionResult carries text + token counts so we can attribute
      cost to dream-log entries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class LlmError(RuntimeError):
    """Base class for provider errors surfaced to the dreaming pipeline."""


class LlmRateLimitError(LlmError):
    """A 429 / rate-limit response from the provider.

    `permanent` is True when the request can never succeed in its current
    shape — e.g. OpenAI's "Request too large ... tokens per min (TPM)" where
    the single request exceeds the *per-minute* token ceiling. Such requests
    must be reshaped (chunked / map-reduced), never blindly retried.

    `permanent` is False for ordinary throttling (too many requests for now);
    the provider adapter retries those with backoff before giving up.
    `retry_after` carries the provider's hint in seconds when present.
    """

    def __init__(
        self, message: str, *, permanent: bool = False, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.permanent = permanent
        self.retry_after = retry_after


@dataclass(frozen=True)
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    model_id: str
    # Why the LLM stopped — "stop" (clean), "length" (hit max_tokens),
    # "content_filter" (provider safety), or provider-specific values.
    # Surfaced so the dreaming module can distinguish genuine empty output
    # from truncation/refusal.
    finish_reason: str = "stop"
    # If the provider produced a refusal in its dedicated refusal channel
    # (rather than normal content), the refusal text lives here. Empty
    # string means no refusal.
    refusal: str = ""


@runtime_checkable
class Llm(Protocol):
    """Chat-style LLM provider used by the dreaming module."""

    @property
    def model_id(self) -> str:
        ...

    def complete(
        self, system: str, messages: list[Message], max_tokens: int = 4096
    ) -> CompletionResult:
        ...
