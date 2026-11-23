"""Phase 5 clustering backends (ADR-0020) and dream progress logging."""
from __future__ import annotations

import logging
import random

import pytest

from ai_memory.core import dreaming
from ai_memory.core.dreaming import (
    _close_pairs_numpy,
    _close_pairs_python,
    _cluster_notes,
    _format_duration,
    _format_progress,
    _Progress,
)
from ai_memory.core.models import Note

np = pytest.importorskip("numpy")


def _note(note_id: str) -> Note:
    return Note(
        id=note_id,
        text=note_id,
        valid_from="2026-01-01T00:00:00Z",
        ingested_at="2026-01-01T00:00:00Z",
        embedding_model="test",
        source_episode_ids=["ep"],
    )


def _random_embeddings(n: int, dim: int, seed: int) -> list[list[float] | None]:
    rng = random.Random(seed)
    # A few tight groups plus noise, so the threshold actually produces pairs.
    centres = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(5)]
    out: list[list[float] | None] = []
    for i in range(n):
        if i % 17 == 0:
            out.append(None)
            continue
        c = centres[i % len(centres)]
        out.append([x + rng.gauss(0, 0.15) for x in c])
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_numpy_pairs_match_pure_python(seed: int, monkeypatch) -> None:
    # Small block size so multiple blocks (and the block boundary) are exercised.
    monkeypatch.setattr(dreaming, "_PAIR_BLOCK_ROWS", 7)
    emb = _random_embeddings(60, 16, seed)
    threshold = 0.65
    assert sorted(_close_pairs_numpy(emb, threshold)) == sorted(_close_pairs_python(emb, threshold))


def test_numpy_pairs_skip_mismatched_dimensions() -> None:
    emb: list[list[float] | None] = [[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01], None, [0.0, 0.0, 0.01]]
    assert sorted(_close_pairs_numpy(emb, 0.1)) == [(0, 2), (1, 4)]
    assert sorted(_close_pairs_python(emb, 0.1)) == [(0, 2), (1, 4)]


def test_numpy_pairs_threshold_is_inclusive() -> None:
    emb: list[list[float] | None] = [[0.0, 0.0], [3.0, 4.0]]  # exactly 5 apart
    assert _close_pairs_numpy(emb, 5.0) == [(0, 1)]
    assert _close_pairs_numpy(emb, 4.999999) == []


def test_cluster_notes_same_components_with_and_without_numpy(monkeypatch) -> None:
    emb = _random_embeddings(40, 8, 7)
    notes = [_note(f"n{i}") for i in range(len(emb))]

    def as_sets(clusters):
        return sorted(sorted(n.id for n in c) for c in clusters)

    fast = as_sets(_cluster_notes(notes, emb, dist_threshold=0.65))
    monkeypatch.setattr(dreaming, "_np", None)
    slow = as_sets(_cluster_notes(notes, emb, dist_threshold=0.65))
    assert fast == slow
    assert fast  # the fixture really does produce clusters


def test_cluster_notes_reports_progress(caplog) -> None:
    notes = [_note(f"n{i}") for i in range(4)]
    emb: list[list[float] | None] = [[0.0], [0.01], [5.0], [5.01]]
    with caplog.at_level(logging.INFO, logger="ai_memory.core.dreaming"):
        _cluster_notes(notes, emb, dist_threshold=0.1, report_progress=True)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Phase 5 clustering pairs (numpy) started (6 items)" in m for m in msgs)
    assert any("finished: 6/6" in m for m in msgs)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "0s"), (59.9, "59s"), (61, "1m01s"), (3600, "1h00m"), (3 * 3600 + 125, "3h02m"), (-5, "0s")],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert _format_duration(seconds) == expected


def test_format_progress_eta_is_linear() -> None:
    line = _format_progress("Phase 4", 25, 100, elapsed=600)
    assert line == "Phase 4: 25/100 (25.0%), elapsed 10m00s, eta ~30m00s"


def test_format_progress_edge_cases() -> None:
    assert "eta ?" in _format_progress("x", 0, 10, elapsed=5)
    assert "eta 0s" in _format_progress("x", 10, 10, elapsed=5)
    assert _format_progress("x", 3, 0, elapsed=5) == "x: 3 done, elapsed 5s"


def test_progress_throttles_to_interval(caplog) -> None:
    t = [0.0]
    with caplog.at_level(logging.INFO, logger="ai_memory.core.dreaming"):
        p = _Progress("Phase 3", 100, every_seconds=60, clock=lambda: t[0])
        for _ in range(10):
            t[0] += 10
            p.step()          # 10s apart → only the 60s and 120s marks log
        t[0] += 5
        p.finish()
    msgs = [r.getMessage() for r in caplog.records]
    assert msgs[0] == "dream progress: Phase 3 started (100 items)"
    periodic = [m for m in msgs if "eta" in m]
    assert len(periodic) == 1  # at t=60 (t=100 is only 40s later)
    assert "6/100 (6.0%)" in periodic[0]
    assert msgs[-1] == "dream progress: Phase 3 finished: 10/100 in 1m45s"
