"""Phase 5 promotion verdict cache (ADR-0021).

Uses a real temp SQLite store. The fake embedder returns the same vector for
every text, so every inserted note lands in one cluster.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_memory.config import DreamConfig
from ai_memory.core.dreaming import (
    _iso_days_before,
    _phase5_promote,
    _promotion_cluster_key,
    _promotion_verdict_version,
)
from ai_memory.core.models import Note
from ai_memory.ids import new_id
from ai_memory.llm.interface import CompletionResult
from ai_memory.storage.sqlite_store import SqliteStore

NOW = "2026-11-23T12:00:00Z"
DECLINE = '{"key": null, "value": null, "rationale": "not a durable user fact"}'
PROMOTE = '{"key": "preferred_database", "value": "PostgreSQL", "rationale": "x"}'


class _Llm:
    def __init__(self, response: str, model_id: str = "fake:quality") -> None:
        self.model_id = model_id
        self.response = response
        self.calls = 0

    def complete(self, system: str, messages: list, max_tokens: int = 4096) -> CompletionResult:
        self.calls += 1
        return CompletionResult(
            text=self.response, input_tokens=5, output_tokens=5, model_id=self.model_id
        )


@pytest.fixture()
def store(tmp_path: Path):
    s = SqliteStore(tmp_path / "memory.db", embedding_dim=4)
    s.initialise()
    yield s
    s.close()


def _add_notes(store: SqliteStore, n: int, episodes: tuple[str, ...] = ("epA", "epB")) -> None:
    for i in range(n):
        note = Note(
            id=new_id(), text=f"Igor prefers PostgreSQL ({i})", tags=[], entities=[],
            source_episode_ids=[episodes[i % len(episodes)]],
            valid_from=NOW, ingested_at=NOW, embedding_model="fake:embed",
        )
        store.insert_note(note, [0.1, 0.2, 0.3, 0.4])


def _run(store: SqliteStore, llm: _Llm, *, now: str = NOW, ttl: int = 30) -> tuple[int, list[str]]:
    journal: list[str] = []
    promoted, _ = _phase5_promote(
        store=store, llm=llm, quality_llm=llm,
        config=DreamConfig(promotion_verdict_ttl_days=ttl), now=now, journal=journal,
    )
    return promoted, journal


def test_declined_cluster_is_not_judged_again(store) -> None:
    _add_notes(store, 3)
    llm = _Llm(DECLINE)
    _run(store, llm)
    assert llm.calls == 1
    _promoted, journal = _run(store, llm)
    assert llm.calls == 1  # cache hit, no second verdict
    assert any("1 skipped (declined before, unchanged), 0 to judge" in line for line in journal)


def test_cluster_that_gains_a_note_is_judged_again(store) -> None:
    _add_notes(store, 3)
    llm = _Llm(DECLINE)
    _run(store, llm)
    _add_notes(store, 1)
    _run(store, llm)
    assert llm.calls == 2


def test_unparseable_verdict_is_not_cached(store) -> None:
    _add_notes(store, 3)
    llm = _Llm("not json at all")
    _run(store, llm)
    calls_first = llm.calls  # initial call + one validation retry
    _run(store, llm)
    assert llm.calls == 2 * calls_first  # judged again: a parse failure isn't a decline


def test_expired_verdict_is_judged_again(store) -> None:
    _add_notes(store, 3)
    llm = _Llm(DECLINE)
    _run(store, llm, now=NOW, ttl=30)
    _run(store, llm, now="2026-12-20T12:00:00Z", ttl=30)  # 27 days later: still cached
    assert llm.calls == 1
    _run(store, llm, now="2026-12-24T12:00:00Z", ttl=30)  # 31 days later: expired
    assert llm.calls == 2


def test_ttl_zero_disables_the_cache(store) -> None:
    _add_notes(store, 3)
    llm = _Llm(DECLINE)
    _run(store, llm, ttl=0)
    _run(store, llm, ttl=0)
    assert llm.calls == 2
    assert store.get_promotion_verdicts(
        [_promotion_cluster_key(store.list_valid_notes(), _promotion_verdict_version(llm.model_id))]
    ) == {}


def test_changing_promotion_model_invalidates_cache(store) -> None:
    _add_notes(store, 3)
    _run(store, _Llm(DECLINE, model_id="model-a"))
    other = _Llm(DECLINE, model_id="model-b")
    _run(store, other)
    assert other.calls == 1


def test_promoted_cluster_still_promotes(store) -> None:
    _add_notes(store, 3)
    llm = _Llm(PROMOTE)
    promoted, _ = _run(store, llm)
    assert promoted == 1
    assert {p.key for p in store.list_profile()} == {"preferred_database"}


def test_cluster_key_ignores_member_order_and_tracks_membership() -> None:
    a, b, c = (Note(id=x, text=x, valid_from=NOW, ingested_at=NOW, embedding_model="e")
               for x in ("a", "b", "c"))
    v = _promotion_verdict_version("m")
    assert _promotion_cluster_key([a, b], v) == _promotion_cluster_key([b, a], v)
    assert _promotion_cluster_key([a, b], v) != _promotion_cluster_key([a, b, c], v)
    assert _promotion_cluster_key([a, b], v) != _promotion_cluster_key([a, b], _promotion_verdict_version("n"))


def test_iso_days_before() -> None:
    assert _iso_days_before("2026-11-23T12:00:00Z", 30) == "2026-10-24T12:00:00Z"


def test_store_verdict_roundtrip_and_prune(store) -> None:
    store.record_promotion_verdict("k1", 3, "declined", "2026-10-01T00:00:00Z")
    store.record_promotion_verdict("k2", 4, "declined", "2026-11-20T00:00:00Z")
    assert store.get_promotion_verdicts(["k1", "k2", "k3"]) == {
        "k1": "2026-10-01T00:00:00Z", "k2": "2026-11-20T00:00:00Z",
    }
    assert store.prune_promotion_verdicts("2026-11-01T00:00:00Z") == 1
    assert store.get_promotion_verdicts(["k1", "k2"]) == {"k2": "2026-11-20T00:00:00Z"}
