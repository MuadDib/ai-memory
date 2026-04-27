"""Anthropic LLM provider."""
from __future__ import annotations

import os

from anthropic import Anthropic

from ai_memory.llm.interface import CompletionResult, Message


class AnthropicLlm:
    """Anthropic-backed LLM for summarisation and fact extraction."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Anthropic API key not set. Provide one via the ANTHROPIC_API_KEY "
                "environment variable or the api_key argument."
            )
        self._client = Anthropic(api_key=api_key)
        self._model = model

    @property
    def model_id(self) -> str:
        return f"anthropic:{self._model}"

    def complete(
        self, system: str, messages: list[Message], max_tokens: int = 4096
    ) -> CompletionResult:
        response = self._client.messages.create(
            model=self._model,
            system=system,
            messages=list(messages),
            max_tokens=max_tokens,
        )
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
