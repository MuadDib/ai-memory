# ADR-0011: Bi-temporal Notes — Invalidate, Never Delete

**Status:** Accepted  
**Date:** 2026-04-27

## Context

Facts in memory become outdated. A user's job title changes; a preference reverses; a bug gets fixed. The naive approach (update in place) loses history. We need to know both *when something was true in the world* and *when we learned it*, and we need to be able to roll back if a consolidation run produces bad notes.

## Decision

Notes are **bi-temporal** and **append-only**:

- `valid_from` / `valid_to` — when the fact was true in the world (`valid_to = NULL` means currently valid)
- `ingested_at` — when this note row was written to the database

**Updating a fact means:**
1. Setting `valid_to = NOW` on the existing note (invalidation)
2. Inserting a new note with the updated fact and `valid_from = NOW`

**Notes are never physically deleted.** The `invalidated` flag (`valid_to IS NOT NULL`) filters them from normal recall.

Dream Phase 6 (Integrate) handles invalidation when Phase 4 returns a CONTRADICTS verdict.

The `reset_episode_for_redream()` path is the only exception: it physically deletes notes derived from the target episode before re-running the dream cycle. This is acceptable because the source of truth (Tier 3 JSONL) is intact and the notes are immediately regenerated.

## Consequences

- Full contradiction history is preserved and queryable (`SELECT * FROM notes WHERE valid_to IS NOT NULL`).
- Rollback after a bad dream run: mark the episode unconsolidated, delete its derived notes, re-dream. The `redream` CLI automates this.
- Storage grows monotonically. For a local single-user system at current corpus sizes this is acceptable.
- Phase 4 CONTRADICTS detection currently fires rarely — the verdict prompt needs stronger signal for genuine contradictions (open follow-up).
- Queries that intend "current facts only" must always filter `WHERE valid_to IS NULL`. Missing this filter silently returns stale facts.

## Alternatives considered

- **Mutable in-place update**: Simple but destroys history. Rollback is impossible. Rejected.
- **Event sourcing (append-only log, no invalidation flag)**: Pure event sourcing but requires replaying to get current state. More complex than needed for this scale. Rejected.
- **Soft-delete with `deleted_at`**: Simpler than bi-temporal, but loses *when* the fact was valid, making the timeline queryable only as "deleted" vs "not deleted". Rejected in favour of full bi-temporal.
- **Versioned notes** (version counter, current boolean): Equivalent to bi-temporal but less expressive for temporal queries. Rejected.
