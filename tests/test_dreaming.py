"""Pure-function tests for the dreaming module.

These exercise the bits that don't need the real LLM, store, or embedder —
clustering math, JSON parsing tolerance, key sanitisation. The integration
phases themselves are covered by end-to-end tests once they exist.
"""
from __future__ import annotations

import time

from ai_memory.core.dreaming import (
    _cluster_notes,
    _euclid_distance,
    _chunk_turns,
    _safe_parse_facts,
    _safe_parse_json,
    _sanitise_profile_key,
)
from ai_memory.core.models import Note


# --- Distance --------------------------------------------------------------


def test_euclid_distance_basic() -> None:
    assert _euclid_distance([0.0, 0.0], [3.0, 4.0]) == 5.0
    assert _euclid_distance([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_euclid_distance_mismatched_dims() -> None:
    assert _euclid_distance([1.0], [1.0, 2.0]) == float("inf")


# --- Clustering ------------------------------------------------------------


def _note(id_: str, episode: str = "e1") -> Note:
    return Note(
        id=id_,
        text=f"text-{id_}",
        valid_from=int(time.time()),
        ingested_at=int(time.time()),
        embedding_model="test",
        source_episode_ids=[episode],
    )


def test_cluster_notes_groups_close_vectors() -> None:
    notes = [_note("a"), _note("b"), _note("c"), _note("d")]
    embeddings = [
        [1.0, 0.0],
        [1.001, 0.0],   # very close to a
        [10.0, 10.0],   # far away
        [10.001, 10.0], # close to c
    ]
    clusters = _cluster_notes(notes, embeddings, dist_threshold=0.1)
    cluster_ids = sorted(sorted(n.id for n in cluster) for cluster in clusters)
    assert cluster_ids == [["a", "b"], ["c", "d"]]


def test_cluster_notes_drops_singletons() -> None:
    notes = [_note("a"), _note("b")]
    embeddings = [[0.0, 0.0], [10.0, 10.0]]  # far apart
    assert _cluster_notes(notes, embeddings, dist_threshold=0.1) == []


def test_cluster_notes_handles_missing_embeddings() -> None:
    notes = [_note("a"), _note("b"), _note("c")]
    embeddings: list[list[float] | None] = [None, [0.0, 0.0], [0.0, 0.0]]
    clusters = _cluster_notes(notes, embeddings, dist_threshold=0.1)
    # Only b and c should pair up; a is excluded because it has no embedding.
    assert len(clusters) == 1
    assert sorted(n.id for n in clusters[0]) == ["b", "c"]


# --- Profile-key sanitisation ---------------------------------------------


def test_sanitise_key_lowercases_and_underscores() -> None:
    assert _sanitise_profile_key("Preferred Database") == "preferred_database"
    assert _sanitise_profile_key("Comm Style!") == "comm_style"
    assert _sanitise_profile_key("  --  ") == "fact"  # nothing useful left


def test_sanitise_key_collapses_repeated_separators() -> None:
    assert _sanitise_profile_key("foo___bar  baz") == "foo_bar_baz"


# --- JSON parsing ----------------------------------------------------------


def test_safe_parse_json_plain() -> None:
    assert _safe_parse_json('{"a": 1}', default={}) == {"a": 1}


def test_safe_parse_json_strips_markdown_fence() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert _safe_parse_json(raw, default={}) == {"a": 1}


def test_safe_parse_json_extracts_embedded_object() -> None:
    raw = 'Here is the JSON: {"key": "preferred_db", "value": "Postgres"} all done.'
    parsed = _safe_parse_json(raw, default={})
    assert parsed == {"key": "preferred_db", "value": "Postgres"}


def test_safe_parse_json_default_on_garbage() -> None:
    assert _safe_parse_json("not json at all", default={"fallback": True}) == {"fallback": True}


def test_safe_parse_facts_filters_non_dicts() -> None:
    raw = '[{"text": "fact 1", "tags": ["a"]}, "garbage", null, {"text": "fact 2"}]'
    facts = _safe_parse_facts(raw)
    assert len(facts) == 2
    assert facts[0]["text"] == "fact 1"
    assert facts[1]["text"] == "fact 2"


def test_safe_parse_facts_handles_bare_comma_separated_objects() -> None:
    """gpt-4o-mini regularly drops the outer [] of an array but keeps the items.

    Real example from a dream pass on a 260-turn cowork episode — the
    extract step returned `{"text":"..."},{"text":"..."}` with no array
    wrapper, which json.loads can't parse directly. The parser must wrap
    and retry rather than silently emitting zero candidates.
    """
    raw = '{"text": "fact one", "tags": ["a"]},{"text": "fact two", "tags": ["b"]}'
    facts = _safe_parse_facts(raw)
    assert len(facts) == 2
    assert facts[0]["text"] == "fact one"
    assert facts[1]["text"] == "fact two"
    assert facts[0]["tags"] == ["a"]


def test_safe_parse_facts_handles_single_object() -> None:
    """When the extractor finds only one fact some models return a bare
    object instead of a one-element array. Treat as a one-element list."""
    raw = '{"text": "the only fact", "tags": ["solo"]}'
    facts = _safe_parse_facts(raw)
    assert len(facts) == 1
    assert facts[0]["text"] == "the only fact"


def test_safe_parse_facts_handles_trailing_comma_on_bare_objects() -> None:
    """Some models leave a trailing comma after the last bare object."""
    raw = '{"text": "a", "tags": []},{"text": "b", "tags": []},'
    facts = _safe_parse_facts(raw)
    assert len(facts) == 2
    assert facts[0]["text"] == "a"
    assert facts[1]["text"] == "b"


def test_safe_parse_facts_empty_on_garbage() -> None:
    assert _safe_parse_facts("definitely not json") == []
    assert _safe_parse_facts("") == []


# --- Chunking -----------------------------------------------------------


def test_chunk_turns_short_episode_single_chunk() -> None:
    """Episodes shorter than chunk_size should not be chunked at all."""
    turns = list(range(20))
    chunks = _chunk_turns(turns, chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == turns


def test_chunk_turns_overlapping_windows() -> None:
    """Long episodes split into overlapping windows that cover all turns."""
    turns = list(range(260))
    chunks = _chunk_turns(turns, chunk_size=50, overlap=5)
    # Step = 45, so windows start at 0, 45, 90, 135, 180, 225 -> 6 chunks.
    assert len(chunks) == 6
    # Every turn appears in at least one chunk.
    seen = set()
    for chunk in chunks:
        seen.update(chunk)
    assert seen == set(turns)
    # Overlap exists at boundaries (chunk[0][-5:] should equal chunk[1][:5]).
    assert chunks[0][-5:] == chunks[1][:5]
    # Last chunk reaches the end.
    assert chunks[-1][-1] == 259


def test_chunk_turns_empty_input() -> None:
    assert _chunk_turns([], chunk_size=50, overlap=5) == []


def test_chunk_turns_exact_multiple() -> None:
    """When len(turns) lands exactly on a chunk boundary, no orphan tail."""
    turns = list(range(50))
    chunks = _chunk_turns(turns, chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == turns
