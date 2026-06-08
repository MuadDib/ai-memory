# ADR-0013: Dream-Cycle Resilience and Rate-Limit Safety

**Status:** Accepted
**Date:** 2026-06-07
**Related:** ADR-0005 (dream cycle), ADR-0007 (chunked extraction — amended here)
**Source:** [Memory Health Review 2026-06-07](../reviews/2026-06-07-memory-health-review.md)
**Implementation:** code landed 2026-06-07 (per-episode isolation, map-reduce
summary, `max_request_tokens`, 429 classification + backoff, daemon circuit
breaker, journal-on-complete; 109 tests passing). Remaining ops steps (stop
service, run orphan cleanup, backfill) tracked in
[the remediation plan](../plans/2026-06-07-dream-deadlock-remediation.md).

## Context

A health audit on 2026-06-07 found the dream cycle had been frozen since
2026-05-05. Root cause: a single oversized episode poison-pills every pass.

The failure chain (full detail in the review):

1. `dream()` processes **all** unconsolidated episodes in **one pass**.
2. Episodes ≥ `long_episode_turns` (100) are auto-upgraded to `gpt-4o`.
3. The Phase 3 **summary call is single-shot over the full transcript**
   (ADR-0007 deliberately left summary un-chunked). An ~800-turn episode becomes
   a ~142k-token request.
4. The org's gpt-4o limit is **30k TPM**, so the request is rejected with
   `429 request too large` — permanently, not transiently.
5. There is **no per-episode error isolation**, so one episode aborts the whole
   pass and blocks the other 840.
6. There is **no circuit breaker**, so the daemon retries the identical doomed
   pass every 63 s indefinitely (50,318 failed attempts at audit time).
7. The `dream_log` row is inserted **before** work and only finalised on success,
   so every failure orphans a row — destroying the introspection feature.

ADR-0007's "summary stays single-shot" assumption held for the corpus sizes seen
in April but does not hold for 800-turn imported ChatGPT sessions on a low-TPM
account. The architecture optimised for cost ("quality model for long episodes")
and in doing so created its own deadlock.

## Decision

Make the dream cycle **fault-isolating and rate-limit-aware**. Six changes:

1. **Per-episode failure isolation.** Wrap each episode's Phase 3 in a try/except.
   A failing episode is recorded (structured failure entry in the journal, like
   the existing per-chunk `EXTRACT_FAILURE` log) and **skipped**, leaving
   `consolidated_at = NULL` for a later retry. One bad episode must never abort
   the pass.

2. **Bound every LLM call by tokens, not just turns.** The summary step must be
   chunked or token-capped the same way extract is (this revises ADR-0007). No
   single Phase 3 request may exceed a configurable `max_request_tokens` derived
   from the selected model's TPM ceiling. Oversized transcripts are summarised
   map-reduce style (summarise chunks → merge) rather than in one shot.

3. **Token-budget the model choice.** The `quality_model` upgrade must consider
   the request size against the model's rate limit. A 142k-token job must not be
   routed to a 30k-TPM model. If the budget is exceeded, either downshift to a
   chunked path on the cheaper model or split the work — never emit a request
   that cannot physically succeed.

4. **429 handling with backoff.** Distinguish *transient* 429 (retry with
   exponential backoff + jitter, honouring `Retry-After`) from *permanent* 429
   ("request too large" — never retry the same payload; reshape it instead).

5. **Daemon circuit breaker.** After N consecutive failed passes the daemon stops
   re-firing the same work, logs loudly (and surfaces via `stats`), and backs off
   to a long interval instead of hammering every 63 s.

6. **Journal-on-complete.** Persist the `dream_log` row only when a pass finishes
   (or reconcile/garbage-collect orphans on the next start), so `dream-log`
   remains a usable audit trail.

## Consequences

- The pipeline degrades gracefully: a poison-pill episode is isolated and
  retried, never fatal to the batch.
- Cost/latency of long episodes rises slightly (map-reduce summary = a few extra
  calls) in exchange for never producing an impossible request.
- **ADR-0007 is amended:** "summary is single-shot" becomes "summary is
  single-shot *only when it fits the token budget*; otherwise map-reduce."
- `dream_log` regains meaning; one row ≈ one real pass.
- A config knob (`max_request_tokens` / per-model TPM hint) becomes part of the
  stable surface and must be documented in the README.
- Backlog drain is safe to run unattended after the change.

## Alternatives considered

- **Just raise the OpenAI tier.** Removes *this* 429 but not the class of bug —
  a future larger episode or a different account hits the same wall. Necessary
  hygiene, not a fix. Rejected as the sole remedy.
- **Cap episode size at import.** Splitting 800-turn imports into sub-episodes
  avoids large transcripts, but loses episode coherence and doesn't protect
  against legitimately long live sessions. Deferred as a complementary option.
- **Drop the gpt-4o upgrade entirely.** Cheapest fix, but sacrifices the
  extraction-quality gains on long episodes that motivated ADR-0007. Rejected;
  token-budgeting keeps the quality path where it fits.
- **Move dreaming fully synchronous/manual.** Removes the runaway loop but
  abandons the "smarter overnight" goal. Rejected.
