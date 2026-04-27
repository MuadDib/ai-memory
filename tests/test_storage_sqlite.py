"""Round-trip tests for the SQLite store.

These run against a temporary file DB and exercise the schema-creation +
basic CRUD path. Vector and FTS search paths require sqlite-vec to be
installed at the system level — pytest will skip them if not available.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from ai_memory.core.models import Episode, Note, Profile
from ai_memory.storage.sqlite_store import SqliteStore


pytest.importorskip("sqlite_vec")


@pytest.fixture()
def store(tmp_path: Path) -> SqliteStore:
    s = SqliteStore(tmp_path / "memory.db", embedding_dim=4)
    s.initialise()
    yield s
    s.close()


def test_profile_roundtrip(store: SqliteStore) -> None:
    profile = Profile(key="name", value="Igor", updated_at=int(time.time()), source="manual")
    store.upsert_profile(profile)
    listed = store.list_profile()
    assert len(listed) == 1
    assert listed[0].key == "name" and listed[0].value == "Igor"


def test_note_roundtrip_and_invalidation(store: SqliteStore) -> None:
    now = int(time.time())
    note = Note(
        id="01HX1",
        text="Igor prefers PostgreSQL",
        tags=["preference", "db"],
        valid_from=now,
        ingested_at=now,
        embedding_model="test",
    )
    store.insert_note(note, [0.1, 0.2, 0.3, 0.4])
    fetched = store.get_note("01HX1")
    assert fetched is not None
    assert fetched.text == note.text
    assert fetched.tags == ["preference", "db"]
    assert fetched.valid_to is None

    store.invalidate_note("01HX1", when=now + 100, superseded_by=None)
    fetched = store.get_note("01HX1")
    assert fetched is not None and fetched.valid_to == now + 100


def test_episode_and_turn_roundtrip(store: SqliteStore) -> None:
    now = int(time.time())
    episode = Episode(
        id="01HE1", title="t", summary="s", source="manual",
        started_at=now, raw_file="2026/04/01HE1.jsonl", embedding_model="test",
    )
    store.insert_episode(episode, embedding=None)
    fetched = store.get_episode("01HE1")
    assert fetched is not None and fetched.summary == "s"

    recent = store.list_recent_episodes(limit=5)
    assert any(e.id == "01HE1" for e in recent)


def test_vector_search(store: SqliteStore) -> None:
    """Verify sqlite-vec returns nearest neighbours in expected order."""
    now = int(time.time())
    notes = [
        ("a", [1.0, 0.0, 0.0, 0.0]),
        ("b", [0.9, 0.1, 0.0, 0.0]),
        ("c", [0.0, 0.0, 1.0, 0.0]),
    ]
    for nid, emb in notes:
        store.insert_note(
            Note(id=nid, text=f"text-{nid}", valid_from=now, ingested_at=now, embedding_model="t"),
            emb,
        )
    hits = store.search_notes_vector([1.0, 0.0, 0.0, 0.0], k=2)
    assert [h[0].id for h in hits] == ["a", "b"]


def test_bm25_search(store: SqliteStore) -> None:
    now = int(time.time())
    for nid, text in [("a", "Postgres relational"), ("b", "DynamoDB key value"), ("c", "Postgres JSON columns")]:
        store.insert_note(
            Note(id=nid, text=text, valid_from=now, ingested_at=now, embedding_model="t"),
            [0.0, 0.0, 0.0, 0.0],
        )
    hits = store.search_notes_bm25("Postgres", k=5)
    ids = [h[0].id for h in hits]
    assert "a" in ids and "c" in ids
    assert "b" not in ids
