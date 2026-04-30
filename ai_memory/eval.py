"""
Eval harness for the recall pipeline.

`load_suite(path)` parses a YAML file into an EvalSuite.
`run_suite(suite, service, ...)` is a generator that yields CaseResult per case.
`score_case(expected, hits)` is the single scoring predicate: case-insensitive
substring match against any hit.text in the top-k results.

The CLI (`ai-memory eval`) owns I/O, persistence, and exit codes.
This module is pure logic — no Click, no print, no sqlite writes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Generator

if TYPE_CHECKING:
    from ai_memory.core.service import MemoryService
    from ai_memory.core.models import RecallHit


@dataclass
class EvalCase:
    id: str
    query: str
    expected: str       # substring to find in any top-k hit text
    k: int = 8
    depth: str = "fast"
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalSuite:
    suite: str
    description: str
    cases: list[EvalCase]


@dataclass
class CaseResult:
    case: EvalCase
    passed: bool
    hits_count: int
    top_hit_text: str | None   # text of highest-scored hit, None if no hits
    latency_ms: int


def load_suite(path: Path) -> EvalSuite:
    """Parse a YAML eval suite file.  Validates required fields; raises ValueError on bad input."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "PyYAML is required for eval suites. Install it: pip install pyyaml"
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at the top level, got {type(raw)}")

    suite_name = str(raw.get("suite") or path.stem)
    description = str(raw.get("description") or "")
    raw_cases = raw.get("cases") or []
    if not isinstance(raw_cases, list):
        raise ValueError("'cases' must be a YAML list")

    cases: list[EvalCase] = []
    for i, c in enumerate(raw_cases):
        if not isinstance(c, dict):
            raise ValueError(f"Case {i} is not a mapping")
        missing = [f for f in ("id", "query", "expected") if not c.get(f)]
        if missing:
            raise ValueError(f"Case {i} is missing required fields: {missing}")
        cases.append(
            EvalCase(
                id=str(c["id"]),
                query=str(c["query"]),
                expected=str(c["expected"]),
                k=int(c.get("k") or 8),
                depth=str(c.get("depth") or "fast"),
                tags=[str(t) for t in (c.get("tags") or [])],
            )
        )

    return EvalSuite(suite=suite_name, description=description, cases=cases)


def run_suite(
    suite: EvalSuite,
    service: "MemoryService",
    k_override: int | None = None,
    depth_override: str | None = None,
) -> Generator[CaseResult, None, None]:
    """Run each case against service.recall(), yielding CaseResult as they complete."""
    for case in suite.cases:
        k = k_override if k_override is not None else case.k
        depth = depth_override if depth_override is not None else case.depth

        t0 = time.monotonic()
        hits = service.recall(query=case.query, depth=depth, k=k)
        latency_ms = int((time.monotonic() - t0) * 1000)

        passed = score_case(case.expected, hits[:k])
        top_text = hits[0].text if hits else None

        yield CaseResult(
            case=case,
            passed=passed,
            hits_count=len(hits),
            top_hit_text=top_text,
            latency_ms=latency_ms,
        )


def score_case(expected: str, hits: list["RecallHit"]) -> bool:
    """Return True if `expected` appears (case-insensitive) in any hit's text."""
    needle = expected.lower()
    return any(needle in (hit.text or "").lower() for hit in hits)
