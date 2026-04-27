"""
OpenAI-backed embedder.

Default model: `text-embedding-3-small` (1536 dimensions native;
Matryoshka-truncatable down to e.g. 512 if you want smaller indexes).

Cost reference (April 2026): roughly $0.02 per 1M input tokens.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from openai import APIStatusError, OpenAI

logger = logging.getLogger(__name__)


class OpenAIEmbedder:
    """OpenAI embeddings provider."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        api_key: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not set. Provide one via the OPENAI_API_KEY "
                "environment variable or the api_key argument."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"openai:{self._model}:{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed texts. Empty input returns empty output."""
        if not texts:
            return []

        kwargs: dict[str, Any] = {"model": self._model, "input": texts}
        # text-embedding-3-* support Matryoshka truncation; older models do not.
        if self._model.startswith("text-embedding-3"):
            kwargs["dimensions"] = self._dimensions

        try:
            response = self._client.embeddings.create(**kwargs)
        except APIStatusError as exc:
            # Surface the API's actual error body so we can see *why* it 4xx'd.
            body: str
            try:
                body = exc.response.text  # type: ignore[union-attr]
            except Exception:
                body = str(exc)
            logger.error(
                "OpenAI embeddings call failed (status=%s, model=%s, dims=%d, n_inputs=%d): %s",
                getattr(exc, "status_code", "?"),
                self._model,
                self._dimensions,
                len(texts),
                body,
            )
            raise RuntimeError(
                f"OpenAI embeddings request failed with status "
                f"{getattr(exc, 'status_code', '?')}: {body}"
            ) from exc

        # Preserve input order.
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
