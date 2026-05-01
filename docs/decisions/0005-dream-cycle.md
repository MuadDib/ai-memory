# ADR-0005: Dream Cycle — Seven-Phase Consolidation Pipeline

**Status:** Accepted  
**Date:** 2026-04-27

## Context

Raw conversation turns (Tier 3) contain noise, repetition, and implicit knowledge. Structured recall (Tier 1) requires clean, atomic, queryable facts. The transformation from raw to structured must happen asynchronously — never on the recall hot path — and must be auditable and re-runnable.

## Decision

Consolidation runs as an offline **dream cycle** triggered by `ai-memory dream`. It processes each unconsolidated episode through seven phases in order:

| Phase | Name | Description |
|-------|------|-------------|
| 1 | Load | Fetch raw JSONL turns for the episode from Tier 3 |
| 2 | Summary | LLM single-shot: generate a 2–4 sentence episode summary stored on the episode row |
| 3 | Extract | LLM chunked: extract atomic facts in three categories (user facts / system facts / problems-and-fixes). 50-turn windows with 5-turn overlap; model auto-upgrades to `quality_model` for episodes ≥ `long_episode_turns` turns |
| 4 | Deduplicate | Check each extracted fact against existing notes via vector search; DUPLICATE / CONTRADICTS / NEW verdict; mid-band LLM check for ambiguous pairs |
| 5 | Promote | Decide which NEW/CONTRADICTS facts to write to Tier 0 (profile) vs keep in Tier 1 (notes). Explicit PROMOTE / DO NOT PROMOTE criteria guard against polluting the profile with ephemeral or system facts |
| 6 | Integrate | Persist surviving notes: embed, insert into notes + notes_vec + notes_fts; invalidate contradicted notes |
| 7 | Epilogue | Mark episode `consolidated_at`, append structured entry to dream-log including token counts, model IDs, and all system prompts used |

The dream-log (queryable via `ai-memory dream-log`) is the audit trail. System prompts are quoted into log entries so prompt regressions are detectable.

Episodes can be reset and re-dreamed via `ai-memory redream` (ADR-0012).

## Consequences

- Heavy LLM work is isolated to dream cycles — zero LLM latency on the recall hot path.
- Each phase is a pure function over its inputs; phases can be tested independently.
- The chunked extraction (Phase 3) with overlap prevents facts spanning the 50-turn boundary from being dropped.
- `long_episode_turns` auto-upgrade ensures quality on large sessions without paying gpt-4o rates for small ones.
- Phase 4 contradiction detection (CONTRADICTS verdict) currently fires rarely — the mid-band verdict prompt needs stronger signal for genuine contradictions. Tracked as an open follow-up.
- Changing phase order or the three-category extract structure silently re-biases what gets extracted — treat the phase sequence as a stable contract.

## Alternatives considered

- **Inline consolidation (on every `memory_remember`)**: Adds LLM latency to every write. Rejected.
- **Streaming consolidation in background thread**: Race conditions between writer and reader. Deferred.
- **Single-shot full-episode extraction**: Works for short episodes; degrades badly for 200+ turn sessions. Chunked extraction with overlap is more reliable at scale.
- **No summary phase**: Summary drives Phase 5 promotion context. Removing it degrades promotion accuracy. Rejected.
