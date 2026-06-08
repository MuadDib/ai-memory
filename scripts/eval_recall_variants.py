"""Measure recall@k across recall-config variants (#1 rerank, #2 HyDE, #3 conviction).

Runs the default eval suite against the live corpus under each variant and prints a
pass-rate table, so the opt-in knobs can be validated/tuned with data before being
enabled by default. The rerank/HyDE variants make one extra LLM call per query.

Usage: python scripts/eval_recall_variants.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ai_memory.config import load_config
from ai_memory.core.service import MemoryService
from ai_memory.eval import load_suite, run_suite


def _run(label: str, cfg, suite) -> set:
    svc = MemoryService.build(cfg)
    svc.start()
    try:
        results = list(run_suite(suite, svc))
    finally:
        svc.stop()
    passed = [r for r in results if r.passed]
    fails = {r.case.id for r in results if not r.passed}
    pct = 100.0 * len(passed) / max(1, len(results))
    print(f"{label:30} {len(passed):2}/{len(results)}  ({pct:.1f}%)")
    return fails


def main() -> int:
    base = load_config()
    suite = load_suite(Path(__file__).parent.parent / "evals" / "default.yaml")

    def variant(**kw):
        return replace(base, recall=replace(base.recall, **kw))

    print(f"suite: {suite.suite} ({len(suite.cases)} cases)\n")
    _run("baseline (conviction on)", variant(), suite)
    _run("+ rerank (#1)", variant(rerank_enabled=True), suite)
    _run("+ hyde (#2)", variant(hyde_enabled=True), suite)
    both = _run("+ rerank + hyde", variant(rerank_enabled=True, hyde_enabled=True), suite)
    print("\nstill failing with rerank+hyde:", sorted(both))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
