"""Retroactively merge near-duplicate / contradicting notes already in storage.

Phase 4 integration only ever compares a *new* candidate fact against existing
notes at dream time — it never revisits notes that are already sitting side by
side in the corpus. The 2026-06-07 corpus health review found that an
under-tuned ``UNRELATED_DIST_ABOVE`` (see ``ai_memory.core.dreaming``) let
real near-duplicate facts ("Igor lives and works in London" / "Igor is
currently working in central London...") and even contradictions slip past the
LLM verdict and get inserted side by side. Retuning the threshold only fixes
*future* ingestion; the clusters already in storage keep burying the specific,
well-corroborated facts that recall is supposed to surface.

This script walks every valid note oldest-first and, for each one, asks the
*same* Phase 4 LLM verdict ("DUPLICATE / CONTRADICTS / COMPLEMENTS /
UNRELATED") used during dreaming — but against notes already in the corpus
rather than freshly-extracted candidates:

  * DUPLICATE   -> invalidate the newer note, bump the older (canonical) one's
                   access count, and record the supersession link for audit.
  * CONTRADICTS -> invalidate the OLDER note (it's been superseded by the
                   newer fact), exactly like a live Phase 4 contradiction.
  * otherwise   -> leave both alone.

Dry-run by default; pass --apply to actually write. Every decision is logged
so the report can be reviewed before committing to it (LLM verdicts are not
infallible, and merging is much harder to undo than re-running this script).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from ai_memory.config import load_config
from ai_memory.core import dreaming
from ai_memory.core.service import MemoryService
from ai_memory.timestamps import now_iso

# Windows consoles default to legacy cp1252; note text routinely contains
# Unicode (subscripts, em dashes, smart quotes) that crashes a bare print()
# mid-run. Mirrors the same fix in ai_memory/cli.py.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass


@dataclass
class ConsolidationStats:
    candidates_checked: int = 0
    llm_calls: int = 0
    llm_tokens: int = 0
    auto_duplicates: int = 0
    llm_duplicates: int = 0
    contradictions: int = 0
    disputed: int = 0
    gated: int = 0


def _l2_distance(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _verdict_for(verdicts, existing_id: str):
    """First actionable (DUPLICATE/CONTRADICTS) verdict for a given existing_id, else None."""
    for v in verdicts:
        if v.existing_id == existing_id and v.verdict in ("DUPLICATE", "CONTRADICTS"):
            return v
    return None


def consolidate(
    service: MemoryService,
    *,
    duplicate_dist_below: float,
    unrelated_dist_above: float,
    neighbours_k: int,
    limit: int | None,
    apply: bool,
    log,
    judge_llm=None,
    second_opinion: bool = True,
    pace_seconds: float = 0.0,
) -> ConsolidationStats:
    """Walk valid notes oldest-first and merge near-dup / contradiction pairs.

    Returns aggregate stats. ``log`` is a callable taking a single string —
    used so tests can capture the report instead of printing it.

    ``judge_llm`` defaults to ``service.quality_llm`` (falling back to
    ``service.llm``) — this operation is irreversible and bulk, unlike a live
    Phase 4 candidate where a wrong verdict just produces one redundant note,
    so it warrants the higher-quality model (mirrors the Phase 5 promotion
    pattern: route rare, high-stakes verdicts through the quality model).

    ``second_opinion``: when an action-worthy verdict (DUPLICATE/CONTRADICTS)
    comes back, ask again independently and only act if BOTH calls agree on
    the same verdict AND the same existing_id. Disagreements are logged as
    "disputed" and left untouched — the conservative default for an op that
    can permanently destroy a true standing fact. A 2026-06-07 sample run
    against gpt-4o-mini found roughly half its DUPLICATE/CONTRADICTS verdicts
    on already-stored notes were wrong (e.g. ruling a one-off incident note
    "contradicts" a core standing preference) — far too high a rate for an
    irreversible bulk operation without a confirmation gate.
    """
    store = service.store
    llm = judge_llm or service.quality_llm or service.llm
    stats = ConsolidationStats()

    notes = store.list_valid_notes()  # ordered by ingested_at ascending
    embeddings = {n.id: store.get_note_embedding(n.id) for n in notes}
    invalidated: set[str] = set()
    now = now_iso()

    for idx, note in enumerate(notes):
        if note.id in invalidated:
            continue
        if limit is not None and stats.candidates_checked >= limit:
            log(f"--limit {limit} reached; stopping.")
            break

        emb = embeddings.get(note.id)
        if emb is None:
            continue

        # Compare only against OLDER, still-valid notes — mirrors Phase 4,
        # where the "existing" side is whatever was already in storage when
        # the candidate arrived. This also gives a stable canonical survivor:
        # the earlier note always wins a DUPLICATE verdict.
        older = [
            (other, _l2_distance(emb, embeddings[other.id]))
            for other in notes[:idx]
            if other.id not in invalidated and embeddings.get(other.id) is not None
        ]
        if not older:
            continue
        older.sort(key=lambda pair: pair[1])
        neighbours = [
            pair for pair in older[:neighbours_k] if pair[1] <= unrelated_dist_above
        ]
        if not neighbours:
            continue

        stats.candidates_checked += 1

        nearest_note, nearest_dist = neighbours[0]
        if nearest_dist < duplicate_dist_below:
            stats.auto_duplicates += 1
            log(
                f"[auto-DUPLICATE dist={nearest_dist:.4f}] "
                f"{note.id} {note.text!r}\n"
                f"    -> kept {nearest_note.id} {nearest_note.text!r}"
            )
            invalidated.add(note.id)
            if apply:
                store.invalidate_note(note.id, when=now, superseded_by=nearest_note.id)
                store.bump_note_access(nearest_note.id, when=now)
            continue

        # Pace the LLM calls to stay under the provider's per-minute token
        # budget. gpt-4o on a low TPM tier 429s under a continuous full-corpus
        # walk (the same throttle behind the 2026-06-07 dream deadlock); a
        # small sleep before each verdict keeps a long run from dying. 0 = off
        # (library default, so tests are unaffected).
        if pace_seconds:
            time.sleep(pace_seconds)

        verdicts, tokens = dreaming._classify_candidate_vs_neighbours(
            llm=llm, candidate_text=note.text, neighbours=neighbours
        )
        stats.llm_calls += 1
        stats.llm_tokens += tokens

        # The LLM occasionally returns an existing_id that doesn't match any of
        # the neighbours it was shown (hallucination / copy error) — skip those
        # verdicts rather than crashing on a lookup miss.
        action = None
        existing = None
        for v in verdicts:
            if v.verdict not in ("DUPLICATE", "CONTRADICTS"):
                continue
            match = next((n for n, _ in neighbours if n.id == v.existing_id), None)
            if match is None:
                log(f"[WARN] verdict referenced unknown existing_id {v.existing_id!r} for "
                    f"{note.id} — ignoring that verdict")
                continue
            action, existing = v, match
            break
        if action is None:
            continue

        if dreaming._blocks_preference_override(existing.tags, note.tags):
            stats.gated += 1
            log(
                f"[GATED — preference protected] {note.id} {note.text!r}\n"
                f"    vs {existing.id} {existing.text!r}\n"
                f"    1st verdict: {action.verdict} ({action.reason})\n"
                f"    blocked: existing is tagged 'preference' but candidate is not "
                f"(candidate tags={note.tags}) -> left untouched"
            )
            continue

        if second_opinion:
            verdicts2, tokens2 = dreaming._classify_candidate_vs_neighbours(
                llm=llm, candidate_text=note.text, neighbours=neighbours
            )
            stats.llm_calls += 1
            stats.llm_tokens += tokens2
            confirm = _verdict_for(verdicts2, action.existing_id)
            if confirm is None or confirm.verdict != action.verdict:
                stats.disputed += 1
                log(
                    f"[DISPUTED — skipped] {note.id} {note.text!r}\n"
                    f"    vs {existing.id} {existing.text!r}\n"
                    f"    1st: {action.verdict} ({action.reason})\n"
                    f"    2nd: {confirm.verdict if confirm else 'no action'} "
                    f"({confirm.reason if confirm else 'n/a'})"
                )
                continue

        if action.verdict == "DUPLICATE":
            stats.llm_duplicates += 1
            log(
                f"[LLM-DUPLICATE] {note.id} {note.text!r}\n"
                f"    -> kept {existing.id} {existing.text!r}\n"
                f"    reason: {action.reason}"
            )
            invalidated.add(note.id)
            if apply:
                store.invalidate_note(note.id, when=now, superseded_by=existing.id)
                store.bump_note_access(existing.id, when=now)
        else:  # CONTRADICTS
            stats.contradictions += 1
            log(
                f"[LLM-CONTRADICTS] {note.id} {note.text!r}\n"
                f"    -> supersedes {existing.id} {existing.text!r}\n"
                f"    reason: {action.reason}"
            )
            invalidated.add(existing.id)
            if apply:
                store.invalidate_note(existing.id, when=now, superseded_by=note.id)
                # Record the conflict on the surviving note, mirroring the live
                # Phase 4 path (dreaming.py: _insert_candidate(..., contradicts=[...]))
                # so a retro-applied contradiction keeps the same provenance link
                # the live ingestion path would have written.
                if existing.id not in note.contradicts:
                    note.contradicts.append(existing.id)
                    store.update_note(note)

    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Retroactively merge near-duplicate / contradicting notes already "
            "in storage, using the Phase 4 LLM verdict. Dry-run by default."
        )
    )
    parser.add_argument(
        "--apply", action="store_true", default=False,
        help="Actually write invalidations/access bumps. Without this flag, "
             "performs a dry run (still makes LLM calls — that's the cost of "
             "getting a real report).",
    )
    parser.add_argument(
        "--duplicate-dist-below", type=float, default=dreaming.DUPLICATE_DIST_BELOW,
        help=f"Auto-merge without an LLM call below this L2 distance "
             f"(default: dreaming.DUPLICATE_DIST_BELOW = {dreaming.DUPLICATE_DIST_BELOW}).",
    )
    parser.add_argument(
        "--unrelated-dist-above", type=float, default=dreaming.UNRELATED_DIST_ABOVE,
        help=f"Skip the LLM call above this L2 distance — treated as unrelated "
             f"(default: dreaming.UNRELATED_DIST_ABOVE = {dreaming.UNRELATED_DIST_ABOVE}).",
    )
    parser.add_argument(
        "--neighbours", type=int, default=dreaming.INTEGRATE_NEIGHBOURS,
        help="How many older neighbours to compare each note against "
             f"(default: {dreaming.INTEGRATE_NEIGHBOURS}).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after checking this many candidates (LLM-cost control / "
             "iterative testing). Omit to walk the whole corpus.",
    )
    parser.add_argument(
        "--no-second-opinion", action="store_true", default=False,
        help="Skip the adversarial confirmation call — act on the first verdict "
             "alone. NOT recommended: a 2026-06-07 sample found roughly half of "
             "single-pass DUPLICATE/CONTRADICTS verdicts on stored notes were "
             "wrong. Only useful for fast iteration while testing thresholds.",
    )
    parser.add_argument(
        "--pace-seconds", type=float, default=0.0,
        help="Sleep this many seconds before each LLM verdict call to stay "
             "under the provider's per-minute token budget. Use on a "
             "full-corpus walk against a low-TPM model (gpt-4o) to avoid 429 "
             "death; e.g. --pace-seconds 8. Default 0 (no pacing).",
    )
    parser.add_argument(
        "--use-mini", action="store_true", default=False,
        help="Judge with service.llm (gpt-4o-mini) instead of service.quality_llm "
             "(gpt-4o). NOT recommended for the same reason as above — this is "
             "an irreversible bulk operation, not a live Phase 4 candidate where "
             "a wrong call just costs one redundant note.",
    )
    args = parser.parse_args(argv)

    config = load_config()
    service = MemoryService.build(config)
    service.start()
    try:
        judge_llm = service.llm if args.use_mini else (service.quality_llm or service.llm)
        print(f"Judging with: {judge_llm.model_id}"
              f"{' (no second opinion)' if args.no_second_opinion else ' + adversarial confirmation'}\n")
        stats = consolidate(
            service,
            duplicate_dist_below=args.duplicate_dist_below,
            unrelated_dist_above=args.unrelated_dist_above,
            neighbours_k=args.neighbours,
            limit=args.limit,
            apply=args.apply,
            log=print,
            judge_llm=judge_llm,
            second_opinion=not args.no_second_opinion,
            pace_seconds=args.pace_seconds,
        )
    finally:
        service.stop()

    mode = "APPLIED" if args.apply else "DRY RUN — no changes written"
    print(f"\n--- {mode} ---")
    print(f"candidates checked (had a near neighbour): {stats.candidates_checked}")
    print(f"auto-duplicates (distance pre-filter):      {stats.auto_duplicates}")
    print(f"LLM calls:                                   {stats.llm_calls} "
          f"({stats.llm_tokens} tokens)")
    print(f"LLM-verdict duplicates merged:               {stats.llm_duplicates}")
    print(f"contradictions resolved:                     {stats.contradictions}")
    print(f"disputed (1st/2nd disagreed — skipped):      {stats.disputed}")
    print(f"gated (preference-protection override):      {stats.gated}")
    total_merged = stats.auto_duplicates + stats.llm_duplicates
    print(f"total notes invalidated:                     {total_merged + stats.contradictions}")
    if not args.apply and total_merged + stats.contradictions:
        print("\nRe-run with --apply to write these changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
