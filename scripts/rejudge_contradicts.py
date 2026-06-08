"""Re-judge the CONTRADICTS verdicts from a mini dry-run via a CROSS-MODEL vote.

The gpt-4o-mini pass over-flags contradictions (~half are false: different
scope/timeframe/object, or a specific instance "contradicting" a general fact).
A same-model second opinion can't catch a *systematic* model bias — only model
diversity can. So this re-runs ONLY the flagged contradiction pairs through TWO
independent judges (gpt-4o + Claude) and confirms a contradiction only when
BOTH agree on CONTRADICTS for the same existing note, and the preference gate
doesn't veto it.

Read-only: writes nothing to the corpus. Produces a filtered shortlist
(rejudge_results.txt) of genuine contradictions for a final human eyeball.

Needs ANTHROPIC_API_KEY in the environment (plus the usual OPENAI_API_KEY).

Usage:
    python rejudge_contradicts.py <clean_log.txt>
"""
from __future__ import annotations

import io
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ai_memory.config import load_config
from ai_memory.core import dreaming
from ai_memory.core.service import MemoryService
from ai_memory.llm.anthropic_llm import AnthropicLlm

ID = r"([0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|[0-9A-HJKMNP-TV-Za-z]{26})"
HEAD = re.compile(r"^\[LLM-CONTRADICTS\]\s+" + ID)
SUP = re.compile(r"^\s*->\s*supersedes\s+" + ID)


def parse_pairs(path: str) -> list[tuple[str, str]]:
    """(candidate_id, existing_id): candidate supersedes existing in the mini run."""
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    pairs, i = [], 0
    while i < len(lines):
        m = HEAD.match(lines[i])
        if m:
            cand = m.group(1)
            existing = None
            for j in range(i + 1, min(i + 4, len(lines))):
                s = SUP.match(lines[j])
                if s:
                    existing = s.group(1)
                    break
            if existing:
                pairs.append((cand, existing))
        i += 1
    return pairs


def _l2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _verdict_from(llm, cand_text, neighbours, existing_id):
    """Return the actionable verdict object for existing_id from one model, or None."""
    verdicts, _ = dreaming._classify_candidate_vs_neighbours(
        llm=llm, candidate_text=cand_text, neighbours=neighbours
    )
    return next((v for v in verdicts if v.existing_id == existing_id
                 and v.verdict in ("DUPLICATE", "CONTRADICTS")), None), verdicts


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[ABORT] ANTHROPIC_API_KEY not set in environment — needed for the "
              "cross-model vote. Set it (setx) and re-run from a fresh shell.")
        return 1

    path = sys.argv[1] if len(sys.argv) > 1 else "consolidate_full_dryrun5_clean.txt"
    pairs = parse_pairs(path)
    print(f"parsed CONTRADICTS pairs: {len(pairs)}")

    config = load_config()
    service = MemoryService.build(config)
    service.start()
    gpt = service.quality_llm or service.llm          # gpt-4o
    claude = AnthropicLlm(model="claude-sonnet-4-6")    # independent second family
    print(f"judge A: {gpt.model_id}")
    print(f"judge B: {claude.model_id}")
    print("rule: CONFIRMED only if BOTH say CONTRADICTS (same existing) and not gated\n")

    confirmed, downgraded, split, gated, moot = [], [], [], [], []
    try:
        store = service.store
        for idx, (cand_id, exist_id) in enumerate(pairs, 1):
            cand = store.get_note(cand_id)
            exist = store.get_note(exist_id)
            if cand is None or exist is None:
                moot.append((cand_id, exist_id, "note missing"))
                continue
            if cand.valid_to is not None or exist.valid_to is not None:
                moot.append((cand_id, exist_id, "already invalidated (dup pass / prior)"))
                continue

            cemb = store.get_note_embedding(cand_id)
            eemb = store.get_note_embedding(exist_id)
            dist = _l2(cemb, eemb) if cemb and eemb else 0.0
            neighbours = [(exist, dist)]

            if dreaming._blocks_preference_override(exist.tags, cand.tags):
                gated.append((cand, exist, f"existing 'preference', candidate {cand.tags}"))
                print(f"  [{idx}/{len(pairs)}] GATED: {cand_id[:8]} vs {exist_id[:8]}")
                continue

            time.sleep(1)
            va, _ = _verdict_from(gpt, cand.text, neighbours, exist_id)
            time.sleep(1)
            vb, _ = _verdict_from(claude, cand.text, neighbours, exist_id)

            a = va.verdict if va else "none"
            b = vb.verdict if vb else "none"
            if a == "CONTRADICTS" and b == "CONTRADICTS":
                confirmed.append((cand, exist, va.reason, vb.reason))
                label = "CONFIRMED"
            elif a in ("none", "DUPLICATE") and b in ("none", "DUPLICATE"):
                downgraded.append((cand, exist, f"gpt-4o={a}, claude={b}"))
                label = "DOWNGRADED"
            else:
                split.append((cand, exist, a, b))
                label = "SPLIT"
            print(f"  [{idx}/{len(pairs)}] {label} (gpt-4o={a}, claude={b}): "
                  f"{cand_id[:8]} vs {exist_id[:8]}")
    finally:
        service.stop()

    print("\n" + "=" * 72)
    print(f"CONFIRMED  (both models agree CONTRADICTS):   {len(confirmed)}")
    print(f"DOWNGRADED (both say not-a-contradiction):    {len(downgraded)}")
    print(f"SPLIT      (models disagree — needs a human): {len(split)}")
    print(f"GATED      (preference-protected):            {len(gated)}")
    print(f"MOOT       (note already invalid/missing):    {len(moot)}")
    print("=" * 72)

    with io.open("rejudge_results.txt", "w", encoding="utf-8") as f:
        f.write(f"=== CONFIRMED — both gpt-4o AND claude say CONTRADICTS ({len(confirmed)}) ===\n")
        f.write("These are the genuine, same-scope contradictions. Human review before apply.\n\n")
        for n, (cand, exist, ra, rb) in enumerate(confirmed, 1):
            f.write(f"### {n}\n[NEW/supersedes] {cand.id} {cand.text!r}  tags={cand.tags}\n"
                    f"[OLD/superseded] {exist.id} {exist.text!r}  tags={exist.tags}\n"
                    f"  gpt-4o: {ra}\n  claude: {rb}\n\n")
        f.write(f"\n=== SPLIT — models disagree ({len(split)}) ===\n\n")
        for n, (cand, exist, a, b) in enumerate(split, 1):
            f.write(f"### {n}\n[cand] {cand.id} {cand.text!r}\n"
                    f"[exist] {exist.id} {exist.text!r}\n  gpt-4o={a}  claude={b}\n\n")
        f.write(f"\n=== DOWNGRADED — mini false positives ({len(downgraded)}) ===\n\n")
        for n, (cand, exist, why) in enumerate(downgraded, 1):
            f.write(f"### {n}\n[cand] {cand.id} {cand.text!r}\n"
                    f"[exist] {exist.id} {exist.text!r}\n  {why}\n\n")
        f.write(f"\n=== GATED ({len(gated)}) ===\n\n")
        for n, (cand, exist, why) in enumerate(gated, 1):
            f.write(f"### {n}\n[cand] {cand.id} {cand.text!r}\n"
                    f"[exist] {exist.id} {exist.text!r}\n  {why}\n\n")
        f.write(f"\n=== MOOT ({len(moot)}) ===\n\n")
        for cand_id, exist_id, why in moot:
            f.write(f"{cand_id} vs {exist_id}: {why}\n")
    print("\nwrote rejudge_results.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
