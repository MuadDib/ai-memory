# ADR-0007: Chunked Extraction in Phase 3

**Status:** Accepted  
**Date:** 2026-04-27

## Context

Phase 3 (Extract) sends conversation transcripts to an LLM to extract atomic facts. Long episodes (100–500+ turns) exceed practical context windows for single-shot extraction, and LLM quality degrades on very long inputs. Facts near chunk boundaries can be lost if chunks don't overlap.

## Decision

Phase 3 chunks the extract step into **50-turn windows with 5-turn overlap**:

- Turns are split into windows: `[0:50]`, `[45:95]`, `[90:140]`, …
- Each window is rendered using the standard transcript format (ADR-0006) and sent as a separate LLM call.
- Extracted facts from all windows are concatenated before Phase 4 deduplication — cross-window duplicates are handled by Phase 4, not here.
- The **summary** (Phase 2) remains single-shot over the full episode. It uses the full transcript or a condensed version; it is not chunked.

The `quality_model` auto-upgrade (config: `long_episode_turns`) applies to the Phase 3 extract LLM, not to summary.

## Consequences

- Episodes of any length can be processed without hitting context limits.
- The 5-turn overlap ensures facts that span a chunk boundary appear in at least one window's output.
- Overlap introduces potential cross-window duplicate facts; Phase 4 deduplication handles these.
- Chunking adds multiple LLM calls per episode (linear in episode length). For a 200-turn episode: 4–5 extract calls vs 1 summary call.
- Changing window size or overlap changes which facts get extracted — treat as a stable contract for reproducibility. Re-dreaming with different parameters produces different notes.

## Alternatives considered

- **Single-shot full-episode extraction**: Works for short episodes (< 50 turns). Quality degrades on long sessions; some LLMs silently truncate. Rejected for general use.
- **Sliding window without overlap**: Boundary facts are dropped. Rejected — the overlap cost is minimal.
- **Recursive summarise-then-extract**: Adds a summarisation step before extraction, potentially losing detail. Rejected.
- **Map-reduce extraction**: Extract per-chunk, then LLM merge. Adds a merge step with its own error surface. Deferred as a potential improvement for very long sessions.
