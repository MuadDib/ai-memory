# ADR-0019: Incremental Extraction for Growing Episodes

**Status:** Accepted
**Date:** 2026-11-23
**Related:** ADR-0013 (dream resilience, map-reduce summary)
**Implementation:** `Episode.extracted_turns` (schema v5), `dreaming._extraction_start`,
`dreaming._merge_summaries`, `DreamConfig.incremental_extraction` (default `True`).
Tests: `tests/test_incremental_extraction.py`.

## Context

The Cowork importer re-marks an episode pending (`consolidated_at = NULL`)
whenever its session gets new turns. Phase 3 then read **every** turn of the
episode again: full summary, full extraction, and the quality-model upgrade
decided by the episode's total turn count.

A session kept open for weeks is therefore re-read in full on every pass that
follows new activity. Cost rises with each pass, and almost all of it goes on
turns that were already extracted. On the live corpus, one such session reached
1,138 turns (637k transcript characters, 26 extract chunks on gpt-4o) to pick up
a few dozen new turns. It was re-dreamed about 14 times in 6 days, and those
passes account for most of the ~18M tokens that used up an API credit top-up.
Phase 4 then had to dedup hundreds of re-extracted facts per pass, and slightly
reworded repeats showed up as false contradictions.

## Decision

Each episode stores a watermark, `extracted_turns`: how many of its turns, in
turn order (`ORDER BY ts, rowid`), Phase 3 has already extracted.

When a pending episode has a stored summary and a watermark below its turn count:

- **Extract** only from 5 turns (`EXTRACT_CHUNK_OVERLAP`) before the watermark.
  The overlap gives a fact on the old boundary its context; Phase 4 dedups
  what it repeats.
- **Summarise** those turns, then merge the summary with the stored one using
  the existing map-reduce merge prompt (stored summary as Part 1).
- **Pick the model** from the number of turns read, so a 20-turn extension of a
  1,000-turn session uses the base model.
- **Set the watermark** to the turn count when the episode is consolidated.

Everything else falls back to a full extraction: a first dream, no stored summary,
a watermark beyond the turn count, or `incremental_extraction: false`. If nothing
is past the watermark, the episode is closed without any LLM calls.
`reset_episode_for_redream` resets the watermark, so `redream` stays a full redo.

Turns imported while a pass is running used to be stranded. The importer only
re-marks an episode that is already consolidated, and the pass then marked it
consolidated without reading them. The pass now counts the episode's turns again
before closing it, and leaves it pending if new turns arrived.

### Migration (schema v4 → v5)

The migration adds the column, then backfills it:

- **Consolidated episodes:** the watermark is the full turn count.
- **Pending episodes with a summary** (dreamed before, then grew): the watermark
  is the number of turns stamped no later than the newest note sourced from the
  episode. Notes carry the pass start time.
- **Everything else:** 0, which means a full extraction, the old behaviour.

The estimate can be a turn or two off, and the overlap absorbs that. On the live
corpus the 1,138-turn session got a watermark of 1,139.

## Alternatives rejected

- **A re-dream cooldown (no re-mark until the session has been idle for N hours).**
  It reduces how often the full re-read happens but doesn't change what each one
  costs, so cost still grows with the session's length.
- **A turn-id watermark instead of a count.** It would survive reordering, but
  turns are only ever appended, and a count needs no join and is easier to
  backfill.
- **Rebuilding the summary from scratch on a cheaper model.** It still reads the
  whole transcript, and the summary is the cheaper half of Phase 3 anyway.

## Consequences

- A growing episode costs roughly the size of its new turns per pass instead of
  its whole length. The first pass after upgrading still reads the backlog since
  the last dream (201 turns for the session above).
- The summary is now a running merge, so very old detail can fade over many
  merges. Extraction isn't affected: facts from earlier turns are already notes.
- Recent turns are extracted without the full early context. The 5-turn overlap
  and the stored summary (which extraction doesn't see) are the trade-off.
  Revisit if eval or note-quality reviews show grounding loss on extended episodes.
