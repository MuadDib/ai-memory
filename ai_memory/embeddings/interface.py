"""
Embedder interface — every provider implements one method: `embed(texts) -> vectors`.

Deliberately kept narrow:
    - Batched input, batched output. Callers can pass single-text lists.
    - Returned vectors are plain lists of floats (no numpy dependency in
      the interface — keep portability surface flat).
    - The `model_id` property identifies which model produced the vectors;
      we store this per row so future migrations between models are safe.

A future C# port maps this to:
    interface IEmbedder {
        string ModelId { get; }
        int Dimensions { get; }
        Task<IReadOnlyList<float[]>> EmbedAsync(IReadOnlyList<string> texts, CancellationToken ct);
    }
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Embedding provider contract."""

    @property
    def model_id(self) -> str:
        """Stable identifier for the model used (e.g. 'openai:text-embedding-3-small')."""
        ...

    @property
    def dimensions(self) -> int:
        """The dimensionality of vectors produced."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input, in the same order."""
        ...
