# ADR-0001: Four-Tier Memory Model

**Status:** Accepted  
**Date:** 2026-04-27

## Context

An AI memory system needs to serve two opposing requirements: fast recall on the hot path (no LLM latency), and high-quality consolidation that produces clean, queryable knowledge from noisy conversation turns. These can't be served by a single storage layer.

## Decision

Memory is organised into four distinct tiers, each with different mutability and lifecycle:

| Tier | Name | Contents | Mutability |
|------|------|----------|------------|
| 0 | Profile | Durable identity + preference facts | Upsert (key/value) |
| 1 | Notes | Atomic bi-temporal facts | Insert + invalidate (never delete) |
| 2 | Episodes | Session summaries + metadata | Insert, then update summary once |
| 3 | Verbatim | Raw conversation turns on disk (JSONL) | Append-only, never modified |

**Tier 3 is the source of truth.** Everything else is derived and can be rebuilt from it. The on-disk raw files are immutable; the database indexes are re-generatable.

**Bi-temporal notes (Tier 1):** Every note carries `valid_from`, `valid_to`, `ingested_at`. Updates never delete — they invalidate old notes and insert new ones. This preserves the history of contradictions and enables rollback.

## Consequences

- `memory.db` loss is recoverable: re-run `ai-memory import-cowork` + `ai-memory dream`.
- Recall is always served from Tier 1 + Tier 0 — no LLM on the hot path.
- The dream cycle (ADR-0005) is the only writer to Tier 1 from live data; bootstrap and `memory_remember` are the two other writers.
- Changing tier semantics (e.g. making notes mutable) silently breaks rebuilds and bi-temporal queries.

## Alternatives considered

- **Single vector store** (e.g. Chroma): No tier separation, no rebuild path, no profile concept. Rejected.
- **Three tiers (no verbatim)**: Loses the rebuild guarantee. Rejected.
- **Graph DB for notes**: Portability discipline (ADR-0009) rules out graph dependencies in core. Deferred to Phase 5.
