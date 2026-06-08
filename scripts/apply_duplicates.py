"""Apply ONLY the DUPLICATE merges from a completed dry-run log.

Re-running the full consolidate walk just to apply would be slow, costly, and
non-deterministic (the LLM can return different verdicts). Instead we apply the
exact pairs the dry run already decided: parse the [LLM-DUPLICATE] /
[auto-DUPLICATE] blocks and, for each, invalidate the newer note (superseded_by
the older) and bump the older note's access — identical to the consolidate()
DUPLICATE branch. CONTRADICTS blocks are deliberately ignored here.

Usage:
    python apply_duplicates.py <clean_log.txt>            # dry preview
    python apply_duplicates.py <clean_log.txt> --apply    # write
"""
from __future__ import annotations

import io
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ai_memory.config import load_config
from ai_memory.core.service import MemoryService
from ai_memory.timestamps import now_iso

ID = r"([0-9a-fA-F]{8}-[0-9a-fA-F-]{27}|[0-9A-HJKMNP-TV-Za-z]{26})"
DUP_HEAD = re.compile(r"^\[(?:LLM-DUPLICATE|auto-DUPLICATE[^\]]*)\]\s+" + ID)
KEPT = re.compile(r"^\s*->\s*kept\s+" + ID)


def parse_pairs(path: str) -> list[tuple[str, str]]:
    """Return (newer_id, older_id) pairs: newer is invalidated, older kept."""
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    pairs: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        m = DUP_HEAD.match(lines[i])
        if m:
            newer = m.group(1)
            # the "-> kept <id>" line follows within the next couple of lines
            older = None
            for j in range(i + 1, min(i + 4, len(lines))):
                k = KEPT.match(lines[j])
                if k:
                    older = k.group(1)
                    break
            if older:
                pairs.append((newer, older))
        i += 1
    return pairs


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python apply_duplicates.py <clean_log.txt> [--apply]")
        return 2
    path = sys.argv[1]
    apply = "--apply" in sys.argv[2:]

    pairs = parse_pairs(path)
    newer_ids = [n for n, _ in pairs]
    older_ids = [o for _, o in pairs]
    newer_set, older_set = set(newer_ids), set(older_ids)

    print(f"parsed DUPLICATE pairs: {len(pairs)}")
    if len(newer_set) != len(newer_ids):
        print(f"[WARN] {len(newer_ids) - len(newer_set)} duplicate 'newer' ids in log")
    overlap = newer_set & older_set
    if overlap:
        print(f"[ABORT] {len(overlap)} ids are BOTH invalidated and kept — chain "
              f"inconsistency, refusing to apply: {sorted(overlap)[:5]}...")
        return 1
    print(f"unique notes to invalidate: {len(newer_set)}")
    print(f"unique survivor notes to bump: {len(older_set)}")

    config = load_config()
    service = MemoryService.build(config)
    service.start()
    try:
        store = service.store
        valid_ids = {n.id for n in store.list_valid_notes()}
        missing_newer = newer_set - valid_ids
        missing_older = older_set - valid_ids
        if missing_newer:
            print(f"[INFO] {len(missing_newer)} 'newer' notes already invalid/absent "
                  f"(will skip): {sorted(missing_newer)[:3]}...")
        if missing_older:
            print(f"[WARN] {len(missing_older)} survivor notes already invalid/absent: "
                  f"{sorted(missing_older)[:3]}...")

        to_apply = [(n, o) for n, o in pairs if n in valid_ids]
        print(f"\napplicable pairs (newer still valid): {len(to_apply)}")

        if not apply:
            print("\nDRY PREVIEW — no writes. Re-run with --apply to commit.")
            for n, o in to_apply[:5]:
                print(f"  invalidate {n}  ->  keep {o}")
            print(f"  ... ({len(to_apply)} total)")
            return 0

        now = now_iso()
        invalidated = 0
        for newer, older in to_apply:
            store.invalidate_note(newer, when=now, superseded_by=older)
            if older in valid_ids:
                store.bump_note_access(older, when=now)
            invalidated += 1
        print(f"\nAPPLIED: invalidated {invalidated} duplicate notes.")
        remaining = len(store.list_valid_notes())
        print(f"valid notes now: {remaining}")
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
