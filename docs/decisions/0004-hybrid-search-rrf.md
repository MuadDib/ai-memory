# ADR-0004: Hybrid Search — BM25 + Vector via Reciprocal Rank Fusion

**Status:** Accepted  
**Date:** 2026-04-27

## Context

Recall quality depends on both keyword precision (exact name/tool matches) and semantic similarity (paraphrase, related concepts). Neither BM25 nor vector search alone is sufficient. The two signals need to be fused into a single ranked list without requiring score normalisation.

## Decision

Run BM25 (FTS5) and vector ANN (sqlite-vec) in parallel, then merge with **Reciprocal Rank Fusion** (k=60):

```
rrf_score(doc, rank) = 1 / (k + rank)
final_score(doc) = rrf_score_bm25(doc) + rrf_score_vector(doc)
```

Apply a **recency boost** at merge time:

```
boosted = final_score * (1 + recency_weight * decay(age_days))
```

Both note-level and episode-level searches are run independently; results are interleaved by boosted RRF score. The `vector_distance_floor` config parameter gates out low-quality vector matches before ranking.

Implementation: `ai_memory/core/recall.py`. Every call logs distance distributions (min/median/max) at INFO level to support future threshold tuning.

## Consequences

- RRF is parameter-free (k=60 is the standard default) and robust to score-scale differences between BM25 and cosine distance — no normalisation required.
- Keyword-heavy queries (exact names, tool names) are rescued by BM25 even when vector similarity is low.
- Semantic/paraphrase queries are rescued by vector even when exact tokens don't match.
- Recency boost favours recent notes/episodes without making old ones unreachable.
- No cross-encoder rerank: top-k precision is left on the table. Adding a lightweight cross-encoder as a final step is the highest-value pending improvement (see open follow-ups).
- `vector_distance_floor` must be re-tuned if the embedding model changes (ADR-0003).

## Alternatives considered

- **Vector-only**: Fast, simple. Misses keyword-exact queries (e.g. "AWS Lambda" when the embedding is noisy). Rejected.
- **BM25-only**: Good for known terms, fails on paraphrases and concept drift. Rejected.
- **Weighted score fusion**: Requires normalising incompatible score scales. RRF avoids this entirely. Rejected.
- **Cross-encoder rerank (final step)**: Would improve top-k precision but adds latency and a model dependency. Deferred — tracked as an open follow-up.
- **Dense-only with query expansion**: More complex, harder to debug. Rejected for Phase 1.
