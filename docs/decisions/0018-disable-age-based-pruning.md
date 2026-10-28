# ADR-0018: Disable Age-Based Pruning by Default

**Status:** Accepted
**Date:** 2026-10-28
**Related:** ADR-0014 (contradiction mechanism), ADR-0016 (bounded recency boost)
**Implementation:** `DreamConfig.prune_enabled` (default `False`),
`dreaming._phase6_prune`. Tests: `tests/test_dreaming_integration.py`.

## Context

Phase 6 soft-deletes a note when
`exp(-age / 90 days) * (1 + ln(1 + access_count)) < 0.05`, unless it is promoted
to the profile, younger than 30 days, or re-accessed in the last 90 days.

The rule assumed `access_count` measured use. It doesn't. Only Phase 4 dedup
and consolidation hits bump it, and recall never does. So the rule is an age
rule in practice. With `access_count = 0` (5,532 of 6,744 valid notes) a note
crosses the threshold at 90 · ln 20 ≈ 270 days.

It had never fired, because the oldest notes were ~185 days old. A projection
over the live corpus on 2026-10-28 showed what it was about to do:

| By | Soft-deleted | Share of valid notes |
|---|---|---|
| 2027-01 | 530 | 8% |
| 2027-03 | 1,851 | 27% |
| 2027-07 | 5,872 | 87% |
| 2027-11 | 6,719 | 100% |

24 of the 25 notes that answer passing eval cases fell between 2027-01-25 and
2027-07-09. The one survivor is promoted to the profile.

## Decision

Turn age-based pruning **off by default** (`dream.prune_enabled: false`). The
code stays, and `prune_enabled: true` restores the old behaviour.

Age isn't evidence that a fact is stale. In a personal memory most durable
facts are stated once and rarely asked about. The two things pruning was
meant to handle each have a better owner:

- **A fact that changed** is superseded by a newer contradicting note
  (Phase 4, ADR-0014). Recency at recall only breaks ties (ADR-0016).
- **Junk** (transient state, narration, context-free fragments; ~48% of a
  recent sample) should be rejected at extraction, or removed by a sweep that
  judges quality, not age.

## Alternatives rejected

- **Bump `access_count` on recall.** It would protect only notes that happen
  to be recalled. It adds writes to the recall path from several MCP server
  processes, makes eval runs change the corpus, and compounds with the
  conviction boost (recalled notes rank higher, so get recalled more). True
  but rarely asked facts would still be deleted.
- **A floor on the prune decay** (as ADR-0016 does for recall). Any floor at or
  above the threshold means nothing ever prunes, and anything below it only
  moves the cliff. That's this decision, but harder to read.

## Consequences

- No behaviour change today: Phase 6 had pruned 0 notes ever.
- The corpus grows by about 800 notes a week until extraction gets stricter.
  The costs of that (Phase 5 clustering time, recall noise) are now
  extraction's to fix, not pruning's.
- Follow-up: a quality-targeted junk sweep (LLM-judged, opt-in, dry-run report
  first) if extraction fixes aren't enough.
