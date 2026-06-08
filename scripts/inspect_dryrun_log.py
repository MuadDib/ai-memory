"""Decode a UTF-16LE consolidation dry-run log and emit a clean summary.

Usage: python inspect_dryrun_log.py <logfile>

Reads the log with the correct codec, keeps only printable lines, tallies the
verdict markers, and writes a UTF-8 copy alongside. Prints ONLY a compact
ASCII-safe summary so we never dump raw/mis-decoded bytes into the terminal.
"""
from __future__ import annotations

import collections
import io
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(path: str) -> None:
    with io.open(path, encoding="utf-16-le", errors="replace") as f:
        raw = f.read()

    # Drop the BOM and any stray NULs, keep non-empty lines.
    lines = [ln.rstrip("﻿\x00 ").strip() for ln in raw.splitlines()]
    lines = [ln for ln in lines if ln]

    # Write a clean UTF-8 copy for any later reading.
    out_path = path.rsplit(".", 1)[0] + "_clean.txt"
    with io.open(out_path, "w", encoding="utf-8", errors="replace") as f:
        f.write("\n".join(lines))

    markers = (
        "[auto-DUPLICATE",
        "[LLM-DUPLICATE]",
        "[LLM-CONTRADICTS]",
        "[GATED",
        "[DISPUTED",
        "[WARN]",
        "429",
        "Traceback",
        "UnicodeEncodeError",
        "candidates_checked",
        "Judging with",
    )
    counts = collections.Counter()
    for ln in lines:
        for m in markers:
            if m in ln:
                counts[m] += 1

    print(f"log: {path}")
    print(f"clean copy: {out_path}")
    print(f"total non-empty lines: {len(lines)}")
    print("--- marker counts ---")
    for m in markers:
        if counts[m]:
            print(f"  {m:<22} {counts[m]}")

    # Surface any final stats block (lines that look like the CLI report).
    report = [
        ln for ln in lines
        if ln.startswith(("candidates", "auto", "llm", "duplicates", "contradictions",
                          "disputed", "gated", "total", "LLM calls", "LLM tokens"))
        and ":" not in ln[:3]
    ]
    if report:
        print("--- final report block ---")
        for ln in report:
            print(f"  {ln}")

    # Did the run finish cleanly or die?
    tail = lines[-1] if lines else ""
    print("--- last line ---")
    print(f"  {tail[:200]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "consolidate_full_dryrun3.log")
