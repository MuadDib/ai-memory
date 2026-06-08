"""Regression test: notes and episodes must compete on a comparable score scale.

Until 2026-06, episode hits were scored via 1/(1+distance) (~0.4-0.9) while note
hits were scored via RRF*recency (~0.01-0.03) — a ~30x scale gap. Any episode
with a merely-OK vector match would mathematically bury a note that was the
actual best answer, even one corroborated by BOTH BM25 and vector search. The
gap was always there but invisible on a thin note corpus; the 2026-06-07 backlog
drain grew the note corpus enough to expose it (eval recall@k 96.3% -> 70.4%).

The fix folds episodes into the same reciprocal_rank_fusion as a third ranker,
so an item's TYPE never decides the outcome — its RANK across the signals that
found it does.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_memory.config import RecallConfig
from ai_memory.core.models import Episode, Note
from ai_memory.core.recall import RecallRequest, recall
from ai_memory.storage.raw_files import RawTranscriptStore
from ai_memory.storage.sqlite_store import SqliteStore
from ai_memory.timestamps import now_iso

pytest.importorskip("sqlite_vec")


class _FixedQueryEmbedder:
    """Always returns the same query vector — lets the test fully control distances."""

    model_id = "fake:embed"
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


@pytest.fixture()
def store(tmp_path: Path) -> SqliteStore:
    s = SqliteStore(tmp_path / "memory.db", embedding_dim=4)
    s.initialise()
    yield s
    s.close()


def test_doubly_corroborated_note_outranks_weakly_matched_episode(
    store: SqliteStore, tmp_path: Path
) -> None:
    """A note that wins BOTH the vector and BM25 rankers must outrank an episode
    that only weakly wins the (single) episode-vector ranker — even though the
    episode's raw vector distance is smaller than the note's.

    Engineered distances:
      - note:    embedding identical to the query  -> vector dist = 0      (rank 1)
                 text contains the query token     -> BM25 rank 1
      - episode: embedding close to the query      -> vector dist ~= 0.14 (rank 1,
                 well inside the 1.1 relevance floor)

    Pre-fix scoring: episode = 1/(1+0.14) ~= 0.88, note = (1/61 + 1/61)*recency
    ~= 0.033 -> the unrelated episode would win by ~25x despite the note being
    the doubly-corroborated, semantically exact match.

    Post-fix (unified RRF): note = 1/61 + 1/61 ~= 0.0328 (two rankers),
    episode = 1/61 ~= 0.0164 (one ranker) -> the note correctly wins.
    """
    now = now_iso()

    store.insert_note(
        Note(
            id="note-strong",
            text="Igor's zzqueryword preference is PostgreSQL",
            valid_from=now,
            ingested_at=now,
            embedding_model="t",
        ),
        [1.0, 0.0, 0.0, 0.0],
    )
    store.insert_episode(
        Episode(
            id="ep-weak",
            title="t",
            summary="An unrelated conversation about email replies and leave days",
            source="manual",
            started_at=now,
            raw_file="2026/06/ep-weak.jsonl",
            embedding_model="t",
        ),
        embedding=[0.9, 0.1, 0.0, 0.0],
    )

    raw = RawTranscriptStore(tmp_path / "raw")
    hits = recall(
        store=store,
        raw=raw,
        embedder=_FixedQueryEmbedder(),
        config=RecallConfig(),
        request=RecallRequest(query="zzqueryword", depth="fast", k=8),
        now=now,
    )

    by_id = {h.id: h for h in hits}
    assert "note-strong" in by_id, "the exact-match note should be retrieved at all"
    assert "ep-weak" in by_id, "the episode should still be retrieved (it's a real candidate)"
    assert by_id["note-strong"].score > by_id["ep-weak"].score, (
        "the doubly-corroborated note must outrank the weakly-matched episode — "
        f"got note={by_id['note-strong'].score:.4f} vs ep={by_id['ep-weak'].score:.4f}"
    )
    assert hits[0].id == "note-strong"
    assert hits[0].item_type == "note"
