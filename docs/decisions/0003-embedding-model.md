# ADR-0003: Embedding Model — text-embedding-3-small at 1536 dim

**Status:** Accepted  
**Date:** 2026-04-27

## Context

Vector search requires a fixed embedding model and dimension. Changing the model after ingestion invalidates every stored vector, requiring a full re-embed of all notes and episodes. The choice affects cost, quality, and portability.

## Decision

Use OpenAI `text-embedding-3-small` at 1536 dimensions (`float32`).

- Dimension is hard-coded in the schema (`vec_size = 1536`). Any change requires a schema migration and full re-embed.
- Every note and episode row carries an `embedding_model` column so that, if the model ever changes, rows can be identified and lazily or batch re-embedded rather than requiring a single blocking migration.
- The embedder is behind the `Embedder` interface (`storage/interface.py`). Swapping to a local model requires only a new adapter — no call-site changes.

## Consequences

- `text-embedding-3-small` costs ~$0.02 / 1M tokens — negligible for a local-first single-user system.
- Switching models (e.g. to a local sentence-transformers model) requires: (1) schema migration to change `vec_size`, (2) re-embed all rows identified by `embedding_model != new_model`, (3) update the embedder adapter. The per-row `embedding_model` field makes step 2 scriptable.
- `RecallConfig.vector_distance_floor = 1.1` is empirically tuned for this model on this corpus. Re-tuning is required after any model switch.

## Alternatives considered

- **`text-embedding-ada-002`**: Older OpenAI model, 1536 dim, higher cost per token, lower benchmark quality. Replaced by `text-embedding-3-small` which is cheaper and better.
- **`text-embedding-3-large` (3072 dim)**: Higher quality but doubles storage and compute. Not justified for single-user local use at current corpus sizes.
- **Local sentence-transformers** (e.g. `all-MiniLM-L6-v2`, 384 dim): Zero API cost and no data leaving the machine. Deferred — needs a local inference runtime and quality benchmarking against the current eval suite. The interface design keeps this path open (ADR-0009).
- **Cohere / Voyage embeddings**: Competitive quality, but adds a second API dependency. Rejected for Phase 1 simplicity.
