"""Pure-function tests for the recall pipeline (no DB required)."""
from __future__ import annotations

from ai_memory.core.recall import reciprocal_rank_fusion


def test_rrf_single_ranker() -> None:
    scores = reciprocal_rank_fusion([["a", "b", "c"]], k_rrf=60)
    assert set(scores.keys()) == {"a", "b", "c"}
    # Best (rank 1) gets the highest score
    assert scores["a"] > scores["b"] > scores["c"]


def test_rrf_consensus_wins() -> None:
    """A doc that appears at rank 1 in both rankers should beat a rank-1-in-one-only doc."""
    ranker_a = ["x", "y", "z"]
    ranker_b = ["x", "z", "y"]
    scores = reciprocal_rank_fusion([ranker_a, ranker_b], k_rrf=60)
    assert scores["x"] > scores["y"]
    assert scores["x"] > scores["z"]


def test_rrf_handles_disjoint_results() -> None:
    """Docs unique to one ranker still get scored."""
    scores = reciprocal_rank_fusion([["a"], ["b"]], k_rrf=60)
    assert "a" in scores
    assert "b" in scores
    # Both at rank 1, equal score
    assert scores["a"] == scores["b"]


def test_rrf_empty_input() -> None:
    assert reciprocal_rank_fusion([], k_rrf=60) == {}
    assert reciprocal_rank_fusion([[]], k_rrf=60) == {}
