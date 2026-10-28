# ADR-0016: Bounded Recency Boost and BM25 Stopwords

**Status:** Accepted
**Date:** 2026-10-28
**Related:** ADR-0014 (contradiction mechanism), the
the 2026-10-28 weekly health review, finding 1 (local-only, in `docs/reviews/private/`)
**Implementation:** `ai_memory/core/recall.py` (`_recency_boost`),
`RecallConfig.note_recency_floor` / `episode_recency_floor`,
`ai_memory/storage/sqlite_store.py` (`_fts5_escape`, `_FTS_STOPWORDS`). Tests:
`tests/test_recall.py`, `tests/test_storage_sqlite.py`. Measured with
`scripts/eval_recall_variants.py`.

## Context

The recall eval fell from 22/27 (2026-06-08) to 4/27 (2026-10-28) with no code
change. The corpus aged and recall got worse on its own.

1. **Recency outweighed relevance.** The merge stage multiplied the RRF score
   by `exp(-age / 90 days)`. That's an e-folding constant, so the effective
   half-life was about 62 days, not 90. RRF scores only span about 3.6x across
   the candidate pool (rank 1 in both rankers vs rank 50 in one), while the
   decay spanned about 7.7x between the 6-month-old bootstrap facts and today's
   notes. A fresh note found by one ranker beat an old note found by both.
2. **BM25 matched on grammar.** `_fts5_escape` ORs every query token, so filler
   words matched most of the corpus ("the" is in 4,838 of 6,705 valid notes).
   They padded the 50-slot BM25 list, and RRF credits rank, not strength.

The durable facts were bootstrapped together (all ~180 days old), while recent
notes are disproportionately transient or junk. So any recency preference
between comparably relevant notes systematically picks junk.

## Decision

1. **True half-life, bounded below by a per-tier floor:**
   `boost = floor + (1 - floor) * 0.5 ** (age / half_life)`.
   - Notes: `note_recency_floor = 0.95`. Age can cost a note at most 5%. That's
     enough to break near-ties between two equally relevant restatements of a
     fact (newer wins), and never enough to override relevance.
   - Episodes: `episode_recency_floor = 0.0`. Episodes are time-bound events,
     so they keep the full decay.
2. **Drop English function words from BM25 queries** (`_FTS_STOPWORDS`). The
   list is static and holds function words only. Content words are left to
   BM25's IDF.

Measured on the live corpus (27 cases; pass = expected text in the top k):

| Variant | Pass | MRR@20 |
|---|---|---|
| Old: `exp`, no floor, no stopwords | 4/27 | — |
| True half-life, no floor (+ stopwords) | 6/27 | — |
| Note floor 0.65–0.90 (+ stopwords) | 23/27 | 0.677 |
| **Note floor 0.95 (+ stopwords), shipped** | **25/27** | **0.700** |
| Note floor 1.0, no decay (+ stopwords) | 26/27 | 0.708 |

## Alternatives rejected

- **Half-life 365 (config only).** 20/27, and it only slows the decline: the
  same failure returns as the facts age.
- **No note decay at all (floor 1.0).** It scores best by one case, which sits
  at rank 7 vs 9 of k=8. But with no decay there's nothing to break ties
  between restatements.
- **Relative BM25 cut** (keep only hits within X of the best BM25 score). It
  was tried at 0.3/0.5/0.7, gave no MRR gain and was worse at 0.7. Not shipped.
- **Query-time stop terms by document frequency** (> 20% of notes). Same result
  as the static list on this suite, but it costs a count query per token and
  ports worse.

## Consequences

- **A changed fact is supersession's job, not recency's.** A corroborated old
  fact's conviction boost (about 1.28x at 3 sources) beats any 5% recency
  nudge. So "I moved to X" only wins once Phase 4 invalidates the old note,
  and Phase 4 has invalidated 0 notes to date (ADR-0014 follow-up).
- **Margins are thin.** Several cases pass at rank 5–8 of k=8. Top hits are
  often transient notes ("Igor is using a scratch workspace for this
  session"), so extraction quality is now the main recall lever.
- **Phase 6 prune used the same unbounded decay.** It is now off by default;
  see ADR-0018.
