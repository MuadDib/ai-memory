"""ADR-0019 — incremental Phase 3 extraction for episodes that grow.

A long-lived session gets re-marked pending every time new turns are imported.
These tests pin that the next dream reads only the new turns (plus overlap),
folds a summary of them into the stored one, sizes the quality-model upgrade on
the new turns, and that the watermark is migrated, reset and race-safe.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_memory.config import DreamConfig
from ai_memory.core import dreaming
from ai_memory.core.dreaming import (
    EXTRACT_CHUNK_OVERLAP,
    DreamRequest,
    _extraction_start,
)
from ai_memory.core.ingest import remember
from ai_memory.core.models import Note
from ai_memory.llm.interface import CompletionResult
from ai_memory.storage.raw_files import RawTranscriptStore
from ai_memory.storage.sqlite_store import SqliteStore

ALL = "1970-01-01T00:00:00Z"


# --- _extraction_start (pure) ---------------------------------------------


@pytest.mark.parametrize(
    ("total", "extracted", "has_summary", "enabled", "expected"),
    [
        (10, 0, False, True, 0),     # never dreamed
        (10, 8, True, True, 3),      # resume 5 turns before the watermark
        (10, 3, True, True, 0),      # overlap clamps at the first turn
        (10, 10, True, True, 10),    # nothing new
        (10, 12, True, True, 0),     # watermark past the end: don't trust it
        (10, 8, False, True, 0),     # no stored summary to extend
        (10, 8, True, False, 0),     # incremental mode switched off
    ],
)
def test_extraction_start(total, extracted, has_summary, enabled, expected) -> None:
    assert EXTRACT_CHUNK_OVERLAP == 5  # the cases above assume it
    assert _extraction_start(
        total_turns=total, extracted_turns=extracted,
        has_summary=has_summary, enabled=enabled,
    ) == expected


# --- End to end -------------------------------------------------------------


def _cr(text: str, model: str) -> CompletionResult:
    return CompletionResult(
        text=text, input_tokens=5, output_tokens=5, model_id=model, finish_reason="stop"
    )


class _FakeEmbedder:
    model_id = "fake:embed"
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _RecordingLlm:
    """Records every user message; an optional hook runs on each extract call."""

    def __init__(self, model_id: str = "fake:base") -> None:
        self.model_id = model_id
        self.messages: list[str] = []
        self.on_extract = None

    def complete(self, system: str, messages: list, max_tokens: int = 4096) -> CompletionResult:
        user = messages[-1]["content"]
        self.messages.append(user)
        if "Extract atomic facts" in user:
            if self.on_extract is not None:
                self.on_extract()
            return _cr('[{"text": "Igor prefers PostgreSQL", "tags": ["technical"], '
                       '"entities": []}]', self.model_id)
        if "short title" in system:
            return _cr("A Short Title", self.model_id)
        if "Merge them" in user:
            return _cr("MERGED SUMMARY", self.model_id)
        return _cr(f"SUMMARY BY {self.model_id}", self.model_id)

    def extract_messages(self) -> list[str]:
        return [m for m in self.messages if "Extract atomic facts" in m]


@pytest.fixture()
def wiring(tmp_path: Path):
    store = SqliteStore(tmp_path / "memory.db", embedding_dim=4)
    store.initialise()
    raw = RawTranscriptStore(tmp_path / "raw")
    yield store, raw, _FakeEmbedder()
    store.close()


def _add_turns(store, raw, episode_id, first: int, last: int) -> str:
    for i in range(first, last + 1):
        result = remember(
            store=store, raw=raw, text=f"TURN-{i:02d} content", source="claude-code",
            episode_id=episode_id, now=f"2026-11-01T10:{i:02d}:00Z",
        )
        episode_id = result.episode_id
    return episode_id


def _extend(store, raw, episode_id, first: int, last: int) -> None:
    """Append turns and re-mark the episode pending, as the importer does."""
    _add_turns(store, raw, episode_id, first, last)
    ep = store.get_episode(episode_id)
    ep.consolidated_at = None
    store.update_episode(ep)


def _dream(store, raw, embedder, llm, quality=None, long_turns=0):
    return dreaming.dream(
        store=store, raw=raw, embedder=embedder, llm=llm, quality_llm=quality,
        config=DreamConfig(max_request_tokens=100000, long_episode_turns=long_turns),
        request=DreamRequest(trigger="manual"),
    )


def test_grown_episode_extracts_only_new_turns_and_merges_summary(wiring) -> None:
    store, raw, embedder = wiring
    ep_id = _add_turns(store, raw, None, 1, 8)
    _dream(store, raw, embedder, _RecordingLlm())
    assert store.get_episode(ep_id).extracted_turns == 8

    _extend(store, raw, ep_id, 9, 10)
    llm = _RecordingLlm()
    report = _dream(store, raw, embedder, llm)

    [extract] = llm.extract_messages()
    assert "TURN-01" not in extract and "TURN-03" not in extract
    assert "TURN-04" in extract  # overlap: 5 turns before watermark 8
    assert "TURN-10" in extract
    # The stored summary was folded in as Part 1 of the merge.
    [merge] = [m for m in llm.messages if "Merge them" in m]
    assert "[Part 1] SUMMARY BY fake:base" in merge
    ep = store.get_episode(ep_id)
    assert ep.summary == "MERGED SUMMARY"
    assert ep.extracted_turns == 10 and ep.consolidated_at is not None
    assert "read=7 (from turn 4)" in report.journal


def test_quality_model_sized_on_new_turns(wiring) -> None:
    store, raw, embedder = wiring
    ep_id = _add_turns(store, raw, None, 1, 8)
    base, quality = _RecordingLlm("fake:base"), _RecordingLlm("fake:quality")
    _dream(store, raw, embedder, base, quality=quality, long_turns=8)
    assert quality.extract_messages()  # 8 turns >= 8: first pass upgrades

    _extend(store, raw, ep_id, 9, 10)
    base, quality = _RecordingLlm("fake:base"), _RecordingLlm("fake:quality")
    _dream(store, raw, embedder, base, quality=quality, long_turns=8)
    # 10 turns in total, but only 7 are read: stays on the base model.
    assert not quality.extract_messages()
    assert base.extract_messages()


def test_incremental_off_restores_full_reextract(wiring) -> None:
    store, raw, embedder = wiring
    ep_id = _add_turns(store, raw, None, 1, 8)
    _dream(store, raw, embedder, _RecordingLlm())
    _extend(store, raw, ep_id, 9, 10)

    llm = _RecordingLlm()
    dreaming.dream(
        store=store, raw=raw, embedder=embedder, llm=llm, quality_llm=None,
        config=DreamConfig(max_request_tokens=100000, long_episode_turns=0,
                           incremental_extraction=False),
        request=DreamRequest(trigger="manual"),
    )
    [extract] = llm.extract_messages()
    assert "TURN-01" in extract and "TURN-10" in extract
    assert not [m for m in llm.messages if "Merge them" in m]


def test_pending_with_nothing_new_closes_without_llm(wiring) -> None:
    store, raw, embedder = wiring
    ep_id = _add_turns(store, raw, None, 1, 3)
    _dream(store, raw, embedder, _RecordingLlm())
    ep = store.get_episode(ep_id)
    ep.consolidated_at = None
    store.update_episode(ep)

    llm = _RecordingLlm()
    report = _dream(store, raw, embedder, llm)
    assert llm.messages == []
    assert store.get_episode(ep_id).consolidated_at is not None
    assert "no turns after watermark" in report.journal


def test_turns_arriving_mid_pass_keep_episode_pending(wiring) -> None:
    store, raw, embedder = wiring
    ep_id = _add_turns(store, raw, None, 1, 3)
    llm = _RecordingLlm()
    llm.on_extract = lambda: _add_turns(store, raw, ep_id, 4, 4)

    _dream(store, raw, embedder, llm)

    ep = store.get_episode(ep_id)
    assert ep.consolidated_at is None  # not stranded
    assert ep.extracted_turns == 3     # next pass reads from here


def test_redream_reset_clears_watermark(wiring) -> None:
    store, raw, embedder = wiring
    ep_id = _add_turns(store, raw, None, 1, 3)
    _dream(store, raw, embedder, _RecordingLlm())
    assert store.get_episode(ep_id).extracted_turns == 3

    store.reset_episode_for_redream(ep_id, purge_notes=True)
    assert store.get_episode(ep_id).extracted_turns == 0


# --- v4 -> v5 migration backfill -------------------------------------------


def test_migration_backfills_watermark(tmp_path: Path) -> None:
    db = tmp_path / "memory.db"
    store = SqliteStore(db, embedding_dim=4)
    store.initialise()
    raw = RawTranscriptStore(tmp_path / "raw")
    embedder = _FakeEmbedder()

    done = _add_turns(store, raw, None, 1, 4)   # consolidated: all 4 extracted
    grown = remember(store=store, raw=raw, text="x", source="grown",
                     now="2026-11-01T09:00:00Z").episode_id
    for minute in (1, 2, 30, 31):               # dreamed at 09:10, grew after
        remember(store=store, raw=raw, text="y", source="grown", episode_id=grown,
                 now=f"2026-11-01T09:{minute:02d}:00Z")
    fresh = remember(store=store, raw=raw, text="z", source="fresh").episode_id

    for ep_id, consolidated, summary in (
        (done, "2026-11-01T11:00:00Z", "s"), (grown, None, "s"), (fresh, None, ""),
    ):
        ep = store.get_episode(ep_id)
        ep.consolidated_at, ep.summary = consolidated, summary
        store.update_episode(ep)
    store.insert_note(
        Note(id="n1", text="fact", source_episode_ids=[grown],
             valid_from="2026-11-01T09:10:00Z", ingested_at="2026-11-01T09:10:00Z",
             embedding_model=embedder.model_id),
        embedder.embed(["fact"])[0],
    )
    # Simulate a pre-v5 DB: watermarks unset, version 5 not recorded.
    store.conn.execute("UPDATE episodes SET extracted_turns = 0")
    store.conn.execute("DELETE FROM schema_version WHERE version >= 5")
    store.conn.commit()
    store.close()

    store = SqliteStore(db, embedding_dim=4)
    store.initialise()
    try:
        assert store.get_episode(done).extracted_turns == 4
        assert store.get_episode(grown).extracted_turns == 3  # 09:00, 09:01, 09:02
        assert store.get_episode(fresh).extracted_turns == 0
    finally:
        store.close()
