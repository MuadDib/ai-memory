"""Apply the CONFIRMED contradictions from rejudge_results.txt.

Parses the '=== CONFIRMED ===' section (cross-model agreed pairs) and, for each,
invalidates the OLD/superseded note (superseded_by the NEW note) and records the
conflict on the surviving NEW note via the contradicts column — mirroring the
live Phase 4 CONTRADICTS path and the retro consolidate() branch.

Usage:
    python apply_contradicts.py            # dry preview
    python apply_contradicts.py --apply    # write
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
NEW = re.compile(r"^\[NEW/supersedes\]\s+" + ID)
OLD = re.compile(r"^\[OLD/superseded\]\s+" + ID)


def parse_confirmed(path: str) -> list[tuple[str, str]]:
    """(new_id, old_id): new supersedes old. Only the CONFIRMED section."""
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    pairs, in_section, pending_new = [], False, None
    for line in lines:
        if line.startswith("=== CONFIRMED"):
            in_section = True
            continue
        if line.startswith("=== ") and not line.startswith("=== CONFIRMED"):
            in_section = False
        if not in_section:
            continue
        m = NEW.match(line)
        if m:
            pending_new = m.group(1)
            continue
        o = OLD.match(line)
        if o and pending_new:
            pairs.append((pending_new, o.group(1)))
            pending_new = None
    return pairs


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    path = "rejudge_results.txt"
    pairs = parse_confirmed(path)
    print(f"parsed CONFIRMED contradiction pairs: {len(pairs)}")
    for new, old in pairs:
        print(f"  supersede {old}  <-  {new}")

    config = load_config()
    service = MemoryService.build(config)
    service.start()
    try:
        store = service.store
        valid = {n.id: n for n in store.list_valid_notes()}
        actionable = []
        for new, old in pairs:
            if old not in valid:
                print(f"[SKIP] superseded note {old} already invalid/absent")
                continue
            if new not in valid:
                print(f"[SKIP] surviving note {new} not valid — cannot link/keep")
                continue
            actionable.append((new, old))

        print(f"\napplicable: {len(actionable)} / {len(pairs)}")
        if not apply:
            print("DRY PREVIEW — no writes. Re-run with --apply to commit.")
            return 0

        now = now_iso()
        n = 0
        for new, old in actionable:
            store.invalidate_note(old, when=now, superseded_by=new)
            survivor = valid[new]
            if old not in survivor.contradicts:
                survivor.contradicts.append(old)
                store.update_note(survivor)
            n += 1
        print(f"\nAPPLIED: superseded {n} contradicted notes (contradicts links recorded).")
        print(f"valid notes now: {len(store.list_valid_notes())}")
    finally:
        service.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
