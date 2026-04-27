"""
OpenAI LLM provider for the dreaming module.

Uses the Responses-style chat completion API. Default model is `gpt-4o-mini`
which is cheap enough that running the dream cycle a few times a day costs
pennies. Swap to `gpt-4o` or `o4-mini` via config when better judgment is
worth more tokens.
"""
from __future__ import annotations

import os

from openai import APIStatusError, OpenAI

from ai_memory.llm.interface import CompletionResult, Message


class OpenAILlm:
    """OpenAI-backed LLM for summarisation and fact extraction."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not set. Provide one via the OPENAI_API_KEY "
                "environment variable or the api_key argument."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model

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

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=api_messages,
                max_tokens=max_tokens,
            )
        except APIStatusError as exc:
            body: str
            try:
                body = exc.response.text  # type: ignore[union-attr]
            except Exception:
                body = str(exc)
            raise RuntimeError(
                f"OpenAI chat completion failed with status "
                f"{getattr(exc, 'status_code', '?')}: {body}"
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
