"""Pure-function tests for the recall pipeline (no DB required)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai_memory.config import RecallConfig
from ai_memory.core.models import Note, RecallHit
from ai_memory.core.recall import (
    RecallRequest,
    _conviction_boost,
    _hyde_embedding,
    _llm_rerank,
    _recency_boost,
    recall,
    reciprocal_rank_fusion,
)
from ai_memory.llm.interface import CompletionResult


def _hit(id_: str, text: str, score: float = 0.01) -> RecallHit:
    return RecallHit(item_type="note", id=id_, text=text, score=score)


def _rerank_llm(order_json: str) -> MagicMock:
    llm = MagicMock()
    llm.model_id = "mock"
    llm.complete.return_value = CompletionResult(
        text=order_json, input_tokens=1, output_tokens=1,
        model_id="mock", finish_reason="stop",
    )
    return llm


def test_rerank_reorders_pool_by_llm_order() -> None:
    hits = [_hit("a", "A"), _hit("b", "B"), _hit("c", "C")]
    llm = _rerank_llm('{"order": [2, 0, 1]}')
    out = _llm_rerank(llm=llm, query="q", hits=hits, top_n=3)
    assert [h.id for h in out] == ["c", "a", "b"]


def test_rerank_appends_omitted_indices_in_original_order() -> None:
    hits = [_hit("a", "A"), _hit("b", "B"), _hit("c", "C")]
    llm = _rerank_llm('{"order": [2]}')  # model only ranked one
    out = _llm_rerank(llm=llm, query="q", hits=hits, top_n=3)
    assert [h.id for h in out] == ["c", "a", "b"]  # c first, then a,b untouched


def test_rerank_keeps_tail_beyond_top_n_in_place() -> None:
    hits = [_hit("a", "A"), _hit("b", "B"), _hit("c", "C"), _hit("d", "D")]
    llm = _rerank_llm('{"order": [1, 0]}')
    out = _llm_rerank(llm=llm, query="q", hits=hits, top_n=2)
    assert [h.id for h in out] == ["b", "a", "c", "d"]  # tail c,d untouched


def test_rerank_failsafe_returns_input_on_bad_output() -> None:
    hits = [_hit("a", "A"), _hit("b", "B")]
    llm = _rerank_llm("not json at all")
    out = _llm_rerank(llm=llm, query="q", hits=hits, top_n=2)
    assert [h.id for h in out] == ["a", "b"]  # unchanged — never drops hits


def test_rerank_failsafe_on_provider_error() -> None:
    hits = [_hit("a", "A"), _hit("b", "B")]
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("provider down")
    out = _llm_rerank(llm=llm, query="q", hits=hits, top_n=2)
    assert [h.id for h in out] == ["a", "b"]


def test_rerank_noop_for_single_candidate() -> None:
    hits = [_hit("a", "A")]
    llm = _rerank_llm('{"order": [0]}')
    out = _llm_rerank(llm=llm, query="q", hits=hits, top_n=5)
    assert out == hits
    llm.complete.assert_not_called()


# --- #2 HyDE ----------------------------------------------------------------


def test_hyde_embeds_the_hypothetical_answer_not_the_question() -> None:
    llm = _rerank_llm("Igor works at Citywire as a team lead.")
    emb = MagicMock()
    emb.embed.return_value = [[0.5, 0.5]]
    out = _hyde_embedding(llm=llm, embedder=emb, query="where does Igor work")
    assert out == [0.5, 0.5]
    emb.embed.assert_called_once_with(["Igor works at Citywire as a team lead."])


def test_hyde_failsafe_returns_none_on_provider_error() -> None:
    llm = MagicMock()
    llm.complete.side_effect = RuntimeError("provider down")
    emb = MagicMock()
    out = _hyde_embedding(llm=llm, embedder=emb, query="q")
    assert out is None
    emb.embed.assert_not_called()  # never reached the embedder


def test_hyde_empty_answer_returns_none() -> None:
    llm = _rerank_llm("   ")
    emb = MagicMock()
    out = _hyde_embedding(llm=llm, embedder=emb, query="q")
    assert out is None


# --- #3 conviction boost ----------------------------------------------------


def test_conviction_weight_zero_is_identity() -> None:
    assert _conviction_boost(access_count=9, source_episodes=9, weight=0.0) == 1.0


def test_conviction_boost_rewards_corroboration_and_access() -> None:
    none = _conviction_boost(access_count=0, source_episodes=0, weight=0.2)
    some = _conviction_boost(access_count=2, source_episodes=3, weight=0.2)
    more = _conviction_boost(access_count=10, source_episodes=8, weight=0.2)
    assert none == 1.0            # log1p(0) = 0 -> no boost for an uncorroborated note
    assert more > some > none     # monotonic in the combined signal


def test_conviction_boost_is_gentle_log_scaled() -> None:
    # A heavily-corroborated note gets a bounded nudge, not a runaway multiplier.
    big = _conviction_boost(access_count=100, source_episodes=100, weight=0.2)
    assert 1.0 < big < 2.2


# --- Recency boost (ADR-0016) -------------------------------------------------

DAY = 86400
NOW = "2026-10-28T00:00:00Z"


def test_recency_boost_is_a_true_half_life() -> None:
    assert _recency_boost(age_seconds=0, half_life_days=90, floor=0.0) == 1.0
    assert _recency_boost(age_seconds=90 * DAY, half_life_days=90, floor=0.0) == pytest.approx(0.5)


def test_recency_boost_never_drops_below_floor() -> None:
    ancient = _recency_boost(age_seconds=10_000 * DAY, half_life_days=90, floor=0.95)
    assert 0.95 <= ancient < 0.951
    one_half_life = _recency_boost(age_seconds=90 * DAY, half_life_days=90, floor=0.95)
    assert one_half_life == pytest.approx(0.975)


def test_recency_boost_floor_one_disables_decay() -> None:
    assert _recency_boost(age_seconds=365 * DAY, half_life_days=90, floor=1.0) == 1.0


def test_recency_boost_treats_future_anchor_as_new() -> None:
    assert _recency_boost(age_seconds=-5 * DAY, half_life_days=90, floor=0.0) == 1.0


def _note(id_: str, ingested_at: str) -> Note:
    return Note(
        id=id_, text=id_, valid_from=ingested_at, ingested_at=ingested_at,
        source_episode_ids=["ep"], embedding_model="t",
    )


def _recall_ids(
    *, vector: list[Note], bm25: list[Note], config: RecallConfig | None = None
) -> list[str]:
    store = MagicMock()
    store.search_notes_vector.return_value = [(n, 0.5) for n in vector]
    store.search_notes_bm25.return_value = [(n, -1.0) for n in bm25]
    store.search_episodes_vector.return_value = []
    embedder = MagicMock()
    embedder.embed.return_value = [[0.0]]
    hits = recall(
        store=store, raw=MagicMock(), embedder=embedder,
        config=config or RecallConfig(), request=RecallRequest(query="q", k=8), now=NOW,
    )
    return [h.id for h in hits]


def test_old_strong_match_beats_fresh_single_ranker_match() -> None:
    """Regression for the 2026-10 eval collapse (4/27): a 6-month-old fact found
    by BOTH rankers lost to a fresh note found by only one, because unbounded
    decay outweighed RRF's whole ~3.6x score range."""
    old = _note("old-fact", "2026-04-30T00:00:00Z")
    fresh = _note("fresh-noise", "2026-10-27T00:00:00Z")
    assert _recall_ids(vector=[old], bm25=[old, fresh])[0] == "old-fact"
    # Plain half-life decay (no floor) still buries it — the floor is the fix.
    unfloored = RecallConfig(note_recency_floor=0.0)
    assert _recall_ids(vector=[old], bm25=[old, fresh], config=unfloored)[0] == "fresh-noise"


def test_recency_breaks_ties_between_equally_relevant_restatements() -> None:
    older = _note("powershell-5", "2026-04-30T00:00:00Z")
    newer = _note("powershell-7", "2026-10-20T00:00:00Z")
    # Mirror-image ranks: identical RRF score and conviction; only age differs.
    assert _recall_ids(vector=[older, newer], bm25=[newer, older])[:2] == [
        "powershell-7", "powershell-5",
    ]


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
