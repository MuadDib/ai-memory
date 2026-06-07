# Plan: Dream-Cycle Deadlock Remediation

**Created:** 2026-06-07
**Owner:** Igor
**Driver:** [Memory Health Review 2026-06-07](../reviews/2026-06-07-memory-health-review.md)
**Decision:** [ADR-0013](../decisions/0013-dream-cycle-resilience-and-rate-limit-safety.md)

Goal: lift the 5-week consolidation freeze, drain the 841-episode backlog, and
make the dream cycle unable to deadlock itself again. Then address the downstream
quality shortfalls the freeze had been masking.

---

## P0 — Break the deadlock (do first)

Until this is done the memory is read-only and growing staler every day.

Code changes landed 2026-06-07 (109 tests passing). Ops steps that need
elevation / a live API budget are still open and marked below.

- [x] **Stop the bleeding.** `AiMemoryDream` service stopped (verified
      `Status = Stopped`). No more 63 s failure passes.
- [x] **Single-flight lock on the dream pass.** *Gap found during the 2026-06-07
      follow-up review:* the backlog was actually being drained by
      `dream --trigger idle` processes spawned by the **PreCompact Claude Code
      hook**, and **two were running concurrently**. ADR-0013's pid-lock + circuit
      breaker only guarded the **daemon** path; the `service.dream()` → CLI/hook
      path took no lock, and `dream()` selects all pending episodes up front with
      no per-row claim — so concurrent passes double-process the same episodes
      (wasted LLM spend on the throttled account + near-duplicate notes). Fix:
      a `dream.pid` single-flight lock in `service.dream()` (new
      `ai_memory/process_lock.py`, shared with the daemon); overlapping passes
      now return a "skipped" report instead of piling on. The exe-name recycle
      guard is **off** for this lock because the real holder is base python (the
      `ai-memory.exe`/venv launcher spawns it), whose path lacks "ai-memory" — a
      name check would silently defeat the lock. 8 new tests
      (`test_process_lock.py`, `test_dream_singleflight.py`).
- [x] **Per-episode isolation** (`dreaming.py`): each episode's Phase 3 is wrapped
      in try/except; a failure logs `EPISODE_FAILURE` to the journal, increments
      `episodes_failed`, and leaves `consolidated_at = NULL` for retry. One bad
      episode no longer aborts the pass.
- [x] **Chunk / cap the summary call** (`dreaming.py` `_summarise_episode`):
      single-shot when the transcript fits `max_request_tokens`, otherwise
      map-reduce (summarise chunks → merge). Amends ADR-0007.
- [x] **Token-budget the model choice**: `DreamConfig.max_request_tokens`
      (default 25000) caps every Phase 3 request, so nothing exceeds a low TPM
      ceiling. Added to `config.py` + documented in README.
- [x] **429 backoff**: `LlmRateLimitError` classifies permanent ("request too
      large", reshape — never retry) vs transient (exponential backoff honouring
      `Retry-After`) in both the OpenAI and Anthropic adapters; retry knobs in
      `DreamConfig`.
- [x] **Daemon circuit breaker** (`daemon.py`): after 3 consecutive failed/stuck
      passes the breaker opens for 1 h (`circuit_open` event), instead of
      hammering every 60 s. Any forward progress resets it.
- [x] **Journal-on-complete**: the `dream_log` row is now written exactly once,
      fully populated, at the END of the pass. No more placeholder/orphan rows.
- [x] **Orphan cleanup tool**: `scripts/cleanup_orphan_dream_logs.py` (dry-run by
      default, `--apply` to delete; test in `tests/test_cleanup_orphan_dream_logs.py`).
      Dry run reports ~50,331 orphan rows.
- [x] **Run the cleanup.** Done — `dream_log` went **50,367 → 54 rows, 0
      orphaned**. The `dream-log` introspection feature is usable again.
- [ ] **Raise / confirm the OpenAI rate tier** (hygiene — speeds the backfill; the
      code now survives a low tier but throttles heavily on the ~10 big episodes).
- 🔄 **Backfill the 841 backlog — IN PROGRESS (2026-06-07 afternoon).** Draining
      via `drain.ps1` (a `dream --max-episodes 25` loop with stall detection); the
      new single-flight lock keeps hook/daemon dreams from overlapping it. Down
      from 841 → ~600 pending and falling. When `ai-memory stats` shows 0 pending,
      `Start-Service AiMemoryDream`.
- [ ] **Backfill the 841 backlog.** Prefer the **batched, resumable** form now that
      `dream` supports a cap (each episode commits as it consolidates, so a batch
      can be re-run safely):
      `ai-memory dream --max-episodes 50` — repeat until `ai-memory stats` shows
      0 pending. A single uncapped `ai-memory dream` also works but is one long
      pass. Confirm the 869-/814-turn episodes consolidate (journal shows
      `mode=map-reduce`, `finish=stop`, no permanent 429). Then **restart the
      service** (`Start-Service AiMemoryDream`) so the daemon keeps it current.

  > **Smoke-tested 2026-06-07 — PASS.** `ai-memory dream --max-episodes 1` on the
  > 814-turn poison-pill episode: consolidated in ~12 min, `notes_added=353`,
  > journal `mode=map-reduce` + `finish=stop`, no permanent 429 (transient TPM
  > 429s absorbed by retry/backoff), one finalized dream_log row. Phase 5
  > promoted a clean `hobbies` fact via gpt-4o — no Tier-0 junk.
  >
  > Two follow-on bugs found + fixed during the smoke test:
  > 1. **`_is_permanent_rate_limit` over-broad** — its `"tokens per min" AND
  >    "requested"` clause also matched the *transient* "Rate limit reached"
  >    body, which would wrongly abandon a retryable throttle. Now keys only on
  >    "request too large". (regression test added)
  > 2. **Windows cp1252 console crash** — `click.echo` of a journal containing
  >    `→` raised `UnicodeEncodeError`, exiting 1 *after* a successful pass.
  >    `cli.py` now reconfigures stdout/stderr to UTF-8.

**Exit criteria:** a manual `dream` pass completes with `ended_at` set; pending
episode count drops toward 0; no new orphan rows; the 869- and 814-turn episodes
consolidate without a 429.

---

## P1 — Quality

Split into **pre-backfill** (safe to do blind; shapes how the 841 episodes
consolidate) and **post-backfill** (needs real distance data — tuning these blind
is exactly what proposal §12 warns against).

### Done pre-backfill (2026-06-07)

- [x] **Eval baseline captured**: `ai-memory eval` → 26/27, recall@k 96.3% on the
      current (pre-backfill) corpus. Persisted to `eval_results` as the "before".
- [x] **Promotion → Tier 0 hardened** (Phase 5). Root cause was worse than a thin
      profile: Phase 5 had promoted **external-entity junk** into the permanent
      profile (`current_company='British Gas Trading Limited'`, AWS revenue %,
      project file paths), later deleted but leaving 25 stale `promoted_to_profile`
      flags that black-holed good facts. Fix: (a) hardened `PROMOTION_SYSTEM` with
      the exact leaked examples + "decline when in doubt"; (b) routed the rare,
      high-stakes promotion verdict through the **quality model** (gpt-4o); (c)
      reset the 25 stale flags so good facts re-promote. 2 new tests.
- [x] **Fixed the `1970-01-01` `company` timestamp** (legacy data artifact; live
      path already uses `now_iso()`, so it won't recur).

### Deferred to post-backfill (need real distance data)

- [ ] **Tune Phase 4 dedup**: `DUPLICATE_DIST_BELOW` is currently 0.10 — and was
      *deliberately* lowered from 0.20 so value-changes (port 8080→9000) reach the
      LLM as CONTRADICTS. Raising it to ~0.35 to collapse the "tea ×18" near-dups
      trades off against contradiction detection, so it **must** be eval-driven on
      the drained corpus, not blind.
- [ ] **Extraction grounding gate**: embed each extracted fact, drop those far from
      their source chunk (keeps British-Gas-promo / "AWS Durable Functions"
      hallucinations out of Tier 1). Threshold must be read off the real
      good-vs-junk distance distribution — same data-driven approach as the recall
      floor tuning.
- [ ] **Revisit Phase 4 contradiction detection** with live data.
- [ ] **Recall score normalisation** (also P2): episode hits (~0.5) bury note hits
      (~0.01) regardless of relevance.

---

## P2 — Known / deferred

- [ ] **Local embedder** (sentence-transformers via ONNX) to kill the 2.8 s
      recall latency (currently 14× over the <200 ms goal; remote embed dominates).
- [ ] **Recall score normalisation**: episode hits (~0.5) bury note hits (~0.01);
      put them on a comparable scale before the merge-sort.
- [ ] **Separate provenance from semantic tags**: stop `cowork`/`claude-chat`/
      `bootstrap`/`e2e-test` leaking into the semantic tag space.
- [ ] **Grow live cross-client capture**: 912/985 episodes are the one-time
      ChatGPT bootstrap; the Claude Code / Desktop Stop-hook path is barely
      feeding the system.

---

## Notes

- No data was lost in the freeze; the backlog is fully recoverable.
- Recommended sequence: P0 in one branch (the deadlock is the only thing blocking
  a working memory), verify with a manual backfill, *then* open P1 against the
  freshly-drained corpus so eval numbers are meaningful.
