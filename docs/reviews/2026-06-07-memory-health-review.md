# Memory Health Review — 2026-06-07

*Read-only audit of the live corpus at `%LOCALAPPDATA%\ai-memory\` against the
goals in `shared-ai-memory-proposal.md` (v4). No data was modified.*

**Headline:** the read path is healthy, but the **write/consolidation half has
been dead-looping since 2026-05-05**. A single rate-limit poison-pill freezes the
entire dream cycle; every downstream goal (profile promotion, dedup,
contradiction detection, pruning) is starved as a consequence and cannot be
fairly evaluated until the freeze is lifted.

Decision recorded in **[ADR-0013](../decisions/0013-dream-cycle-resilience-and-rate-limit-safety.md)**.
Remediation steps in **[the remediation plan](../plans/2026-06-07-dream-deadlock-remediation.md)**.

---

## 1. Corpus snapshot (2026-06-07)

| Metric | Value |
|---|---|
| Turns (Tier 3) | 13,570 |
| Episodes (Tier 2) | 985 |
| Episodes **pending dream** | **841 (85%)** |
| Notes (Tier 1, valid) | 765 |
| Profile facts (Tier 0) | **4** |
| Notes flagged `promoted_to_profile` | 25 |
| Notes ever invalidated (Phase 4) | 3 (1 currently invalid) |
| Notes ever pruned (Phase 6) | 0 |
| Notes with entities | 333 / 766 |
| Dream passes logged | 50,367 |
| ...that ever completed (`ended_at` set) | **53** |
| ...orphaned mid-pass | **50,318 (99.9%)** |
| Last real consolidation | **2026-05-05** |
| Last *completed* dream pass | 2026-05-01 |
| Episode source mix | 912 chatgpt (bootstrap), 47 claude-chat, 11 cowork, 1 claude-code, rest test/debug |

---

## 2. Root cause: the dream cycle is in an infinite 429 death-loop

Every dream pass for the last ~5 weeks fails on the **identical** error, ~once
every 63 seconds (600 occurrences today alone):

```
OpenAI chat completion failed with status 429:
Request too large for gpt-4o ... on tokens per min (TPM):
Limit 30000, Requested 142133.
```

### The failure chain

1. **One giant pass.** `dream()` pulls *all* unconsolidated episodes into a
   single pass (`dreaming.py` ~L337) — currently 841 of them.
2. **gpt-4o auto-upgrade.** `config.yaml` routes episodes ≥ `long_episode_turns`
   (100) to the `quality_model` (gpt-4o) for Phase 3. There are **10 pending
   episodes ≥ 100 turns** (largest 869 and 814 turns).
3. **Un-chunked summary.** Phase 3 chunks the *extract* step but the **summary
   is single-shot over the whole transcript** (`dreaming.py` ~L379, codified in
   ADR-0007). An 800-turn episode → a ~142k-token request to gpt-4o, whose org
   limit is **30k TPM**. Rejected outright (`429 request too large`) — it can
   never succeed, no matter how many retries.
4. **No per-episode isolation.** One episode throwing aborts the **entire pass**.
   The poison pill therefore blocks all 841 pending episodes, not just itself.
5. **No circuit breaker.** The daemon's `try/except` (`daemon.py` ~L85) swallows
   the exception and retries the *exact same* doomed pass 63 s later, forever.
6. **Orphaned journal rows.** `dream()` inserts the `dream_log` row *before*
   doing work and only sets `ended_at`/`journal` on success (`dreaming.py` L319).
   Every failure orphans the row — hence 50,318 empty rows that have rendered the
   `dream-log` introspection feature unusable.

The error always reports the same `Requested 142133` tokens, confirming it is the
same episode poison-pilling the pass deterministically on every wake-up.

---

## 3. Scorecard — goals vs. reality

| Goal (proposal v4) | Reality | Status |
|---|---|---|
| "Gets smarter overnight" via dreaming | Zero notes produced since 2026-05-05 | 🔴 Broken |
| Tier 0 = "what every client thinks about Igor" | 4 facts (name, location, role, company). Tea, PostgreSQL, no-abbreviations, comms style all sit in notes, never promoted | 🔴 Falling short |
| Phase 4 contradiction detection | 3 invalidations lifetime | 🔴 Dead |
| Phase 6 decay / prune | 0 notes pruned, ever | 🔴 Never runs |
| Phase 4 dedup | 18 "tea" notes, tennis ×3, Belgrade ×2, Italian ×2 — bootstrap vs extracted phrasings never merged | 🟠 Too strict (known) |
| Shared *cross-client* memory | 912/985 episodes are the one-time ChatGPT bootstrap; live capture tiny | 🟠 Barely fed |
| p95 recall < 200 ms | 2,828 ms measured (query embed = 2,797 ms — remote round-trip) | 🟠 14× over |
| Hybrid recall ranks best answer on top | Episode hits score ~0.5, exact-answer notes ~0.01 — notes buried regardless of relevance | 🟠 Scale mismatch |
| Introspectable dream log | 50k orphan rows make it unreadable | 🟠 Polluted |
| Recall read path | Works; correctly returns the PostgreSQL preference | 🟢 Healthy |
| Eval harness (recall@k) | Exists, 44 results, all passing — but last run 2026-04-30, not wired into the loop | 🟢/🟠 Stale |

---

## 4. Secondary findings (real, but downstream of the freeze)

- **Noise & hallucination in notes.** Long-term memory contains junk:
  *"The promotion is open to UK residents aged 18+ who are existing British Gas
  customers"* (tagged `project`), *"end-to-end tests related to serrated
  knives"*, and a flat-wrong *"AWS Durable Functions function at the same cold
  start rate…"* (not a real product — extraction hallucinated it). This is
  proposal risk §12.1 materialising.
- **Tag taxonomy leaks provenance.** Source names (`cowork`, `claude-chat`,
  `bootstrap`, `chatgpt-export`, `e2e-test`) are mixed into the *semantic* tag
  space alongside `preference`/`project`/`technical`. ADR-0008's redesign is not
  clean in the data.
- **Promotion writes the flag but not the profile.** 25 notes carry
  `promoted_to_profile=1` yet only 4 profile keys exist, and `profile.md` has
  not changed since Apr 30. Promotion is mostly a no-op even when it "fires".
- **Timestamp drift.** Schema sketch / CLAUDE.md say `INTEGER` epoch; storage is
  actually ISO-8601 **text**. The `company` profile row has
  `updated_at = 1970-01-01T00:33:46Z` — a real parse bug. Recency-boost math
  depends on correct parsing.

---

## 5. What is *not* broken

- Recall (BM25 + vector + RRF + recency) returns correct, well-provenanced hits.
- The schema, raw JSONL store, importer cursor, and eval harness are all intact.
- The freeze is **fully recoverable**: no data was lost. Once the poison-pill is
  defused, the 841-episode backlog can be drained by re-running dream.

---

## 6. Priority of fixes

See **[the remediation plan](../plans/2026-06-07-dream-deadlock-remediation.md)**
for detail. In short:

- **P0 — break the deadlock:** per-episode isolation, chunk/cap the summary call,
  token-budget the model choice, daemon circuit breaker, journal-on-complete,
  then backfill the 841-episode backlog.
- **P1 — once writes flow:** re-baseline the eval harness; tune Phase 4 dedup;
  fix promotion → Tier 0; add an extraction quality gate.
- **P2 — known/deferred:** local embedder (recall latency), recall score
  normalisation, tag/provenance separation.
