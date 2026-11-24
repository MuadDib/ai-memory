# ADR-0021: Cache Declined Phase 5 Promotion Verdicts

**Status:** Accepted
**Date:** 2026-11-23
**Related:** ADR-0013 (promotion via quality model), ADR-0020 (vectorised clustering)
**Implementation:** `promotion_verdicts` table (schema v6),
`dreaming._promotion_cluster_key`, `dreaming._promotion_verdict_version`,
`DreamConfig.promotion_verdict_ttl_days` (default 30, 0 = off).
Tests: `tests/test_promotion_verdict_cache.py`.

## Context

Phase 5 sends every eligible cluster to the quality model (gpt-4o) and asks
whether it should become a profile entry. An eligible cluster has at least 3
notes from at least 2 episodes. Phase 5 runs on every pass that adds notes.

Most eligible clusters are declined, and a declined cluster stays eligible, so
the next pass sends it again unchanged and gets the same answer.

On the live corpus, one pass judged 53 clusters and promoted 2. Those 53
gpt-4o calls cost about $0.15, roughly 90% of the pass. With 3–4 passes a day
that came to about $0.50–0.70 a day, almost all of it repeating earlier
verdicts.

## Decision

Cache **declined** verdicts in a `promotion_verdicts` table.

- **Key.** SHA-256 of the cluster's sorted member note ids plus a verdict
  version. The verdict version is a hash of the promotion model id and the
  `PROMOTION_SYSTEM` prompt text.
  - If a note joins or leaves the cluster, the key changes and the cluster is
    judged again. A changed cluster is new evidence.
  - Editing the prompt or switching the promotion model changes every key, so
    nothing stale is reused.
- **What is cached.** Only real declines: a valid response with a null
  key/value. If the response still fails validation after the retry, the code
  falls back to the default "no promotion", which looks like a decline. Those
  fallbacks are **not** cached, so a bad response is retried on the next pass.
  Promoted clusters need no cache entry: their notes are flagged
  `promoted_to_profile` and leave the clustering input.
- **Expiry.** Entries expire after `promotion_verdict_ttl_days` (default 30) and
  are pruned at the start of Phase 5. This lets a cluster be judged again
  occasionally even if its membership never changes. Setting the value to `0`
  turns the cache off and restores the old behaviour.
- **Journal.** The dream journal records eligible, skipped and to-judge counts
  on every pass.

## Consequences

- The first pass after deploying still judges every eligible cluster, because
  the cache starts empty. After that, only new or changed clusters reach the
  model. The ongoing cost of Phase 5 should fall by most of its about $0.15 per
  pass. This is still to be confirmed from the journal counts once it is live.
- A cluster that was declined wrongly stays declined until its membership
  changes or 30 days pass. That's acceptable: the same model gave the same
  answer every pass anyway.
- Schema v6 only adds a table (`CREATE TABLE IF NOT EXISTS`), so no data
  migration is needed. The table is a derived cache and can be safely emptied
  at any time.
