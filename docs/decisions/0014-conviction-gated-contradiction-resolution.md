# ADR-0014: Conviction-Gated, Non-Destructive Contradiction Resolution

**Status:** Partially accepted — §1–§5 landed 2026-06-08; §6 (review digest) pending
**Date:** 2026-06-08
**Related:** ADR-0011 (bi-temporal notes), ADR-0008 (tag taxonomy), ADR-0013
(dream-cycle resilience); open follow-up #6 (conviction-aware promotion)
**Source:** the 2026-06-08 retroactive consolidation pass
([remediation plan](../plans/2026-06-07-dream-deadlock-remediation.md), "Tune
Phase 4 dedup")
**Implementation:**
- **§1 same-scope guard** — `INTEGRATE_VERDICT_SYSTEM` now requires same
  subject/attribute/scope/timeframe/object, with the observed false classes
  (Q4-vs-FY, Pro-vs-non-Pro, Roald-vs-Robert, general-vs-specific, separate
  measurements) spelled out as NOT-contradictions. Guard test added.
- **§4 quarantine default** — live `_phase4_integrate` CONTRADICTS branch now
  inserts the new fact linked via `contradicts` and **keeps both notes valid**
  (no `invalidate_note`); new `quarantined` counter + journal line.
- **§2 cross-model confirm** — a CONTRADICTS verdict supersedes the existing note
  ONLY if an optional `confirm_llm` (a second, ideally different-family model)
  also returns CONTRADICTS for it; otherwise quarantine. New
  `_confirm_contradiction` helper (fail-safe: any error/parse-failure/non-match →
  not confirmed, never destroys). Threaded `dream() → _phase4_integrate`; built in
  `MemoryService.build` from new `LlmConfig.contradiction_confirm_model` +
  `confirm_provider` (defaults empty → `confirm_llm=None` → **quarantine all**,
  the safe default). A missing provider key degrades to quarantine, never crashes.
  149 tests passing.

  **To enable cross-model supersede** (once the daemon runs): set
  `llm.contradiction_confirm_model: claude-sonnet-4-6` and
  `llm.confirm_provider: anthropic` in `config.yaml`, and make `ANTHROPIC_API_KEY`
  available **to the service process** (the NSSM service env, not just a user
  `setx` — services don't inherit interactive user vars).
- **§3 partial-information guard** — even a cross-model-confirmed contradiction is
  downgraded to quarantine if the new fact does not fully cover the old one's
  still-valid content. New `SUPERSEDE_COVERAGE_SYSTEM` prompt + `_CoverageResult`
  model + `_new_supersedes_cleanly` helper (one extra call on the already-tiny
  confirmed subset, reusing the confirmer). Fail-safe: any error/validation
  failure → not clean → quarantine (never destroy).
- **§5 conviction-gated resolution** — new Phase 4b (`_phase4b_resolve_contradictions`,
  off by default via `DreamConfig.resolve_contradictions`) walks quarantined pairs
  (valid notes linked via `contradicts`) and resolves one only when ALL of: (a) a
  re-check confirms it is still a genuine contradiction — this filters out the
  false-contradictions that were quarantined, so conviction can never destroy a
  different-scope fact; (b) the conviction gap exceeds
  `contradiction_resolution_min_gap` (default 3.0); (c) the winner fully covers the
  loser (§3 reused). Conviction = distinct source episodes + access_count +
  promoted bonus + recency term (`_note_conviction`, pure/tested). Reuses the
  cross-model confirmer when available, else the base llm. 155 tests passing.
- Pending: §6 review digest (the human-facing surface for pairs conviction never
  separates).

## Context

Phase 4 integration decides, for each incoming fact, whether it DUPLICATEs,
CONTRADICTS, COMPLEMENTS, or is UNRELATED-to its nearest stored notes. On a
CONTRADICTS verdict the **existing note is invalidated** (`valid_to` set,
`superseded_by` pointed at the new note). This runs **unattended, overnight, in
the daemon** — there is no human in the loop.

The 2026-06-08 retro pass over the live corpus gave us the first real data on
how good those CONTRADICTS verdicts are, and the answer is: **not good enough to
destroy data unattended.**

### What the data showed

- **gpt-4o-mini over-flags massively.** Of 48 mini-flagged contradictions,
  re-judging with a **cross-model vote (gpt-4o + Claude sonnet-4-6)** rejected
  **32 (67%)** as false. Mini treats any same-topic pair with different wording
  as a contradiction.
- **Same-model adversarial retry does not catch this.** The bias is *systematic*,
  not random — a second call to the same model repeats it. Only **model
  diversity** (a different model family) or a **structural rule** catches a
  systematic bias. (This is the same lesson that produced the
  preference-protection gate, `_blocks_preference_override`.)
- **Cross-model agreement is necessary but NOT sufficient.** Among the 7 pairs
  *both* gpt-4o and Claude confirmed, at least three were still wrong in ways
  voting cannot detect:
  1. **Shared blind spot** — two *different* dream-pass token counts (127k vs
     150022) read as a contradiction by both models. Two measurements at
     different times do not conflict.
  2. **Temporal evolution ≠ contradiction** — "116 tests green" → "124 tests
     green" is a *progression*; "Strong .NET preference" → "prefers not to focus
     on .NET" is a *changed stance*. At the verdict layer these are
     indistinguishable from a contradiction; only conviction/recency context
     separates them.
  3. **Partial-information loss** — superseding "WAL mode + 15s busy_timeout"
     with "added busy_timeout=5000" destroys the still-true WAL-mode fact bundled
     into the old note.

### The core mistake

Treating a **single contradicting mention as sufficient to destroy a standing
fact.** Human memory does not work this way: one conflicting data point lowers
confidence; *repeated corroboration* changes the belief. Our bi-temporal schema
(ADR-0011) already supports keeping both and resolving later — we just weren't
using it for contradictions.

## Decision

Replace "CONTRADICTS → invalidate immediately" with a **layered, default-safe**
pipeline. Destruction becomes the rare, high-confidence, reversible exception;
**quarantine is the default.**

### 1. Same-scope guard (verdict prompt) — kills the largest false class

Extend `INTEGRATE_VERDICT_SYSTEM`: a contradiction requires **same subject, same
attribute, same scope, same timeframe, same modality.** Different
time period (Q4 vs FY), different object (OpenRun **Pro** vs OpenRun), different
entity (Roald vs Robert Dahl), or general-principle vs specific-instance is
**COMPLEMENTS / UNRELATED, not CONTRADICTS.** Add these as eval cases.

### 2. Cross-model confirmation for any destructive action

A CONTRADICTS verdict that would invalidate an existing note must be confirmed
by a **second, different-family model** (gpt-4o + Claude). Disagreement →
quarantine (§4), never destroy. Bounded cost: only the mid-band CONTRADICTS
subset is escalated, not every candidate. Reuses the provider interface; no new
dependency.

### 3. Partial-information guard

Before superseding, check the old note carries facts not present in the new one
(cheap: the same verdict model, or an embedding-coverage check). If the old note
is only *partially* contradicted, **quarantine** — never destroy a note that is
partly true.

### 4. Quarantine as the default (non-destructive)

When a contradiction is *not* cleared for destruction by §2+§3, **keep both notes
valid** and record the tension via the existing `contradicts` column (already on
every note, already written by the live path). Recall surfaces both; the conflict
is metadata, not a delete.

### 5. Conviction-gated resolution

A quarantined pair resolves over time, evidence-driven, not on first sight:
- the new fact **supersedes** the old once it accrues conviction above it —
  corroboration count (independent re-statements), recency, source quality
  (turn count / cross-window agreement, per follow-up #6); **or**
- the old fact **decays out** via the existing Phase 6 prune if it stops being
  reinforced while the new one is.

This unifies contradiction resolution with the conviction-aware promotion work
already on the roadmap — same signals, same machinery.

### 6. Human digest for the residual — delivery only, never decider

Pairs that stay contested (cross-model split, or conviction never separates them)
are surfaced in a **periodic digest** ("N contested facts, here's each side") via
the existing `dream_log` / a CLI `review` subcommand — *push*, not a queue to
visit. The human decides; the system never auto-destroys on a contested pair.

**This digest job, if scheduled, MUST honour ADR-0013:** heartbeat / dead-man's
switch and **journal-on-success** (not just on error) — a silent scheduled job is
exactly the failure that froze the dream cycle for five weeks.

### Confidence-tiered routing (summary)

```
verdict = CONTRADICTS
  ├─ §1 same-scope?            no → COMPLEMENTS/UNRELATED (no action)
  ├─ §2 cross-model agree?     no → §4 quarantine
  ├─ §3 fully contradicted?    no → §4 quarantine
  └─ yes to all → supersede (reversible) ; else
         §5 conviction gate decides over time ; else
         §6 human digest
```

## Consequences

**Positive**
- No unattended destruction of a standing fact on a single, possibly-wrong
  contradiction. The failure modes the retro pass found are structurally
  prevented, not just made rarer.
- Reuses what exists: `contradicts` column (ADR-0011), the provider interface,
  Phase 6 prune, the conviction-aware-promotion signals (#6). Little net-new
  surface.
- Everything stays reversible (soft-delete); quarantine is reversible by
  construction (nothing is deleted).

**Negative / costs**
- Recall must gracefully present quarantined conflicts (return both, flag the
  tension) — a recall-layer change, and a ranking question (which side ranks
  first before resolution? lean recency + conviction).
- Cross-model confirmation adds latency + a second provider on the CONTRADICTS
  subset (bounded, dream-time only — never the recall hot path).
- A contested pair can linger in quarantine until conviction or a human resolves
  it. Acceptable: a visible, queryable conflict beats a silent wrong delete.

## Alternatives considered

- **Manual review UI / alert queue.** Rejected as the default: a standing
  *pull* queue rots in a personal system, and it cannot cover the unattended
  daemon path at all. Retained only as the §6 *push* digest for the residual.
- **Smarter single model (o-series / Opus) instead of cross-model.** Lowers the
  error rate but does not break correlation with a *systematic* single-model
  bias; the retro data shows model *diversity* is what catches it.
- **Scheduled agent that auto-resolves.** Rejected as a *decider* (ADR-0013: a
  silent scheduled job is what froze the system). Allowed only as digest delivery
  with the ADR-0013 reliability guards.

## Implementation sketch (phased)

1. **§1 same-scope prompt + eval cases** — cheapest, highest leverage, helps both
   live and retro. (prompt edit + eval rows)
2. **§4 quarantine default** — change live `_phase4_integrate` CONTRADICTS branch
   to record the `contradicts` link and keep both, *not* invalidate, unless
   §2+§3 clear it. (dreaming.py + tests)
3. **§2 cross-model confirm** — factor the retro `rejudge` cross-model logic into
   a reusable `confirm_contradiction(a, b, gpt, claude)` helper. (new helper +
   config for the second judge)
4. **§5 conviction gate** — fold into the conviction-aware promotion work (#6);
   shared scoring.
5. **§6 review digest** — CLI `review` subcommand over quarantined pairs; optional
   scheduled push with ADR-0013 guards.

Ship 1–2 first (they remove most risk on their own); 3–5 incrementally.
