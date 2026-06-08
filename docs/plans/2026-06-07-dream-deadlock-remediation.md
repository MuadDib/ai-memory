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

- [x] **Recall score normalisation** (also P2 — done 2026-06-07 evening):
      episode hits were scored via `1/(1+distance)` (~0.4–0.9) while notes scored
      via `RRF*recency` (~0.01–0.03) — a ~30x scale gap that let weakly-related
      episodes mathematically bury well-corroborated notes once the drained
      backfill tripled the note corpus (eval recall@k 96.3% → 70.4%, 26/27 → 19/27).
      Fix: folded episodes into the *same* `reciprocal_rank_fusion` as a third
      ranker (`recall.py`), so an item's TYPE never decides the outcome — only its
      RANK across the signals that found it does. Regression test added
      (`test_recall_merge.py::test_doubly_corroborated_note_outranks_weakly_matched_episode`,
      proven to fail against the old scoring and pass against the new).
- [x] **BM25 was silently dead on every natural-language query** (found while
      re-baselining the above — not in the original list, but the same
      "recall ranking quality" bucket). `_fts5_escape` joined every token with
      FTS5's implicit AND, so a query like "what operating system and shell
      environment does Igor use" required notes to contain literally every word
      including "what"/"does"/"use" — zero notes ever matched, leaving vector
      search as the *only* ranker (`bm25_candidates=0` on nearly every recall).
      Fix: join with `OR` instead — true bag-of-words, relying on `bm25()`'s
      IDF weighting to rank rare-term hits ("Murphy", "gpt-4o-mini") above
      filler-word-only hits. Regression test added
      (`test_storage_sqlite.py::test_bm25_search_natural_language_query_matches_on_shared_terms`).
      Also added `PRAGMA busy_timeout = 5000` to `SqliteStore.initialise()` —
      concurrent CLI + `serve` access was throwing "database is locked"
      immediately with no retry window.
  > **Eval recovery, measured 2026-06-07 evening** (`AI_MEMORY_HOME` pointed at
  > the live, fully-drained corpus): post-backfill regressed baseline **19/27
  > (70.4%)** → **21/27 (77.8%)** after both fixes — a real recovery, though still
  > short of the pre-backfill **26/27 (96.3%)**. Case-level diff: net +2 (gained
  > `llm-provider`, `murphy-bed`, `primary-language`, `profile-company` lost
  > nothing from the BM25 fix; the RRF-unification step alone was actually net
  > -2 in isolation — see "Remaining gap" below for why that's not a red flag).
  >
  > **Remaining gap is corpus density, not scoring** — confirmed by direct
  > inspection: the correct note for every still-failing case (e.g. "Igor works
  > as a Senior Software Developer, Team Lead, Architect at Citywire", "Igor's
  > working environment includes Windows with WSL Ubuntu") IS retrieved and
  > ranks just *below* the eval's k cutoff (rank 9–13 vs k=5/8), buried under a
  > pile of near-duplicate notes about the same general topic ("Igor lives and
  > works in London", "Igor is a programmer who leads a team…", "Igor uses a
  > Windows environment for development"). This is precisely what the next item
  > below is for.
- [x] **Tune Phase 4 dedup + retroactive consolidation — DONE 2026-06-08.**
      Two parts: (a) retuned the live thresholds; (b) built and ran a retroactive
      pass over the already-stored corpus (the freeze had let near-dups pile up
      faster than the live path could ever catch them, since live dedup only sees
      a note vs *prior* notes at ingestion).

      **What landed:**
      - `scripts/consolidate_duplicate_notes.py` — walks valid notes oldest-first,
        re-runs the Phase 4 verdict on each near-neighbour pair, merges DUPLICATEs
        and resolves CONTRADICTS. Dry-run by default; `--apply` to write; `--limit`,
        `--use-mini`, `--pace-seconds`, `--no-second-opinion` knobs. Soft-delete
        only (reversible via `valid_to`). 15 tests.
      - **Preference-protection gate** (`dreaming._blocks_preference_override`,
        wired into BOTH the retro script AND live `_phase4_integrate`): a structural,
        zero-LLM rule — a non-`preference` candidate may never DUPLICATE/CONTRADICTS
        a `preference`-tagged note. Found via a 2026-06-07 measurement: episodic
        notes (`problem`/`workflow`/`fix`) were *systematically* (4/4) judged to
        override standing preferences, and the bias **survived an adversarial
        same-model second opinion** — only structural/tag signal or *model
        diversity* catches a systematic bias. Gate fired 18× on the live run, 0
        false blocks.
      - **Provenance parity**: retro CONTRADICTS now records the `contradicts` link
        on the surviving note, matching the live Phase 4 path.
      - Two real bugs fixed en route: a `StopIteration` on hallucinated `existing_id`
        (now logged + skipped, regression-tested in both paths); a `UnicodeEncodeError`
        from cp1252 stdout on Unicode note text (stdout reconfigured to UTF-8).

      **Live-corpus result (2026-06-08):** dry-run on 1980 valid notes (gpt-4o-mini
      judge, adversarial double-check) flagged 230 actions. Applied the **182
      DUPLICATEs** directly from the report (deterministic; no re-walk). The **48
      CONTRADICTS** were re-judged via a **cross-model vote (gpt-4o + Claude
      sonnet-4-6)** — only pairs *both* models confirmed survived: **32 killed as
      false positives (67%)**, 9 model-split (held), **7 confirmed and applied**.
      Corpus **1980 → 1791 valid notes (−189)**. Two timestamped DB backups taken;
      all changes reversible.

      **Key finding for the live path:** cross-model agreement is *necessary but
      not sufficient* — among the 7 confirmed, ≥1 was a shared-blind-spot false
      positive (two different dream-pass token counts read as a contradiction) and
      others were *temporal evolution* (count grew 116→124) or *partial-information
      loss* (superseding a stale timeout also destroyed a still-true WAL-mode fact).
      Voting cannot fix these; they need conviction/recency context and
      non-destructive resolution. → see the live-path design below.
      - [x] *(follow-up, 2026-06-08)* Re-baselined `ai-memory eval` on the deduped
        1791-note corpus: **20/27 (74.1%)** — essentially flat vs the pre-dedup
        **77.8% (21/27)**, i.e. **dedup did NOT deliver the hypothesised recall
        recovery; if anything it cost one case.** The hypothesis ("near-dup clusters
        bury the specific Citywire/WSL/backend facts") was wrong: those cases still
        fail after the clusters were collapsed. Two effects roughly cancel —
        removing redundant notes is good hygiene (−189 notes, less noise fed to the
        agent), but collapsing the corroborating near-dups also *reduces the RRF
        mass* behind the general fact, so the buried specific note doesn't rise. The
        residual gap is a **retrieval-ranking** problem (the correct note embeds far
        from the natural-language query and ranks just below k) — addressable by the
        cross-encoder rerank / grounding items, **not** by dedup. Net: dedup was the
        right call for corpus hygiene and contradiction cleanup, but it is not a
        recall lever.
- [x] **Recall-quality levers #1–#4 (2026-06-08)** — since dedup wasn't the lever,
      attacked the ranking problem directly. Measured on the live corpus via
      `scripts/eval_recall_variants.py` (runs the suite under each config variant):

      | variant | recall@k | Δ |
      |---|---|---|
      | post-dedup, pre-changes | 20/27 (74.1%) | — |
      | **#3 conviction boost (on by default)** | **22/27 (81.5%)** | **+2** |
      | + #1 LLM rerank (opt-in) | 23/27 (85.2%) | +1 (fixes `murphy-bed`) |
      | + #2 HyDE (opt-in) | 22/27 (81.5%) | **+0 (no help on this suite)** |

      - **#3 conviction-aware ranking** — corroboration (source episodes) +
        reinforcement (access_count), log-scaled, applied at merge. The big, free
        win; **on by default** (`recall.conviction_weight=0.2`). Recovered past the
        pre-dedup 77.8% baseline.
      - **#1 LLM cross-encoder rerank** — re-scores the top-20 fused pool. Adds one
        case but costs an LLM call per recall, so **opt-in** (`recall.rerank_enabled`)
        per the hot-path rule.
      - **#2 HyDE** — embed a hypothetical answer instead of the question. Implemented
        + fail-safe, but **no measurable lift on this suite**; left **opt-in**
        (`recall.hyde_enabled`), off by default. (May help other query shapes; no
        evidence yet.)
      - **#4 floor tuning** — checked the live distance distribution: notes cluster at
        L2 0.56–0.87, well under the 1.1 floor, and the remaining failures are
        *retrieved but low-ranked / not in the fused pool*, not filtered out. So the
        floor is **not** the lever; left at 1.1. (`final_score_floor` stays 0.)
      - Still failing (4): `backend-language`, `frontend-framework`, `user-origin`,
        `user-role` — the right note isn't in the top-20 fused pool, so rerank can't
        reach it. This is a *retrieval/embedding* gap (a better embedder or the
        grounding/data work below), not a ranking one.
- [ ] **Extraction grounding gate**: embed each extracted fact, drop those far from
      their source chunk (keeps British-Gas-promo / "AWS Durable Functions"
      hallucinations out of Tier 1). Threshold must be read off the real
      good-vs-junk distance distribution — same data-driven approach as the recall
      floor tuning.
- [ ] **Revisit Phase 4 contradiction detection** with live data — **now
      designed**, see [ADR-0014 — Conviction-Gated, Non-Destructive Contradiction
      Resolution](../decisions/0014-conviction-gated-contradiction-resolution.md).
      The 2026-06-08 retro pass surfaced the concrete failure modes (systematic
      over-flagging; temporal-evolution-vs-contradiction; partial-information loss
      on supersede) that the ADR's quarantine + cross-model + conviction design
      addresses for the *unattended* live path.

---

## P2 — Known / deferred

- [ ] **Local embedder** (sentence-transformers via ONNX) to kill the 2.8 s
      recall latency (currently 14× over the <200 ms goal; remote embed dominates).
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
