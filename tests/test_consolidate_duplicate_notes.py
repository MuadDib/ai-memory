"""Tests for the retroactive note-consolidation script.

Exercises the merge/contradiction decision logic against a fake store + LLM —
mirrors the Phase 4 unit tests in test_dreaming.py (same verdict contract,
just applied to notes that are already both in storage).
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from ai_memory.core.models import Note
from ai_memory.llm.interface import CompletionResult
from ai_memory.timestamps import now_iso

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from consolidate_duplicate_notes import consolidate, ConsolidationStats  # noqa: E402


def _note(id_: str, text: str, ingested_at: str = "2026-01-01T00:00:00Z",
          tags: list[str] | None = None) -> Note:
    return Note(id=id_, text=text, tags=tags or [], valid_from=ingested_at,
                ingested_at=ingested_at, embedding_model="test")


def _mock_llm(*responses: str) -> MagicMock:
    llm = MagicMock()
    llm.model_id = "mock"
    llm.complete.side_effect = [
        CompletionResult(text=r, input_tokens=10, output_tokens=5,
                         model_id="mock", finish_reason="stop")
        for r in responses
    ]
    return llm


class _FakeStore:
    """Records invalidate/bump calls; serves notes + canned embeddings."""

    def __init__(self, notes: list[Note], embeddings: dict[str, list[float]]):
        self._notes = notes
        self._embeddings = embeddings
        self.invalidated: list[tuple[str, str | None]] = []
        self.bumped: list[str] = []
        self.updated: list[tuple[str, list[str]]] = []  # (id, contradicts) per update_note

    def list_valid_notes(self) -> list[Note]:
        return list(self._notes)

    def get_note_embedding(self, note_id: str):
        return self._embeddings.get(note_id)

    def invalidate_note(self, note_id, when, superseded_by):
        self.invalidated.append((note_id, superseded_by))

    def bump_note_access(self, note_id, when):
        self.bumped.append(note_id)

    def update_note(self, note: Note):
        self.updated.append((note.id, list(note.contradicts)))


def _service(notes, embeddings, llm, quality_llm=None):
    store = _FakeStore(notes, embeddings)
    return SimpleNamespace(store=store, llm=llm, quality_llm=quality_llm), store


def _log_collector():
    lines: list[str] = []
    return lines, lines.append


# --- auto-duplicate (distance pre-filter) ----------------------------------


def test_auto_duplicate_below_threshold_merges_without_llm():
    older = _note("a", "Igor drinks tea, not coffee.", "2026-01-01T00:00:00Z")
    newer = _note("b", "Igor drinks tea, not coffee!", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [1.0, 0.0]}  # distance 0.0
    llm = _mock_llm()  # must not be called
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.auto_duplicates == 1
    assert stats.llm_calls == 0
    assert store.invalidated == [("b", "a")]
    assert store.bumped == ["a"]
    llm.complete.assert_not_called()


# --- mid-band: LLM verdict drives the outcome ------------------------------


def test_llm_duplicate_verdict_invalidates_the_newer_note():
    """Both opinions agree DUPLICATE -> the newer note is merged into the older."""
    older = _note("a", "Igor lives and works in London.", "2026-01-01T00:00:00Z")
    newer = _note("b", "Igor is currently working in central London.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}  # mid-band distance
    duplicate_resp = '[{"existing_id": "a", "verdict": "DUPLICATE", "reason": "same fact"}]'
    llm = _mock_llm(duplicate_resp, duplicate_resp)
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.llm_calls == 2  # first verdict + adversarial confirmation
    assert stats.llm_duplicates == 1
    assert stats.contradictions == 0
    assert stats.disputed == 0
    assert store.invalidated == [("b", "a")]
    assert store.bumped == ["a"]


def test_llm_contradicts_verdict_invalidates_the_older_note():
    """Both opinions agree CONTRADICTS -> the older note is superseded."""
    older = _note("a", "The service runs on port 8080.", "2026-01-01T00:00:00Z")
    newer = _note("b", "The service now runs on port 9000.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    contradicts_resp = '[{"existing_id": "a", "verdict": "CONTRADICTS", "reason": "port changed"}]'
    llm = _mock_llm(contradicts_resp, contradicts_resp)
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.llm_calls == 2
    assert stats.contradictions == 1
    assert stats.llm_duplicates == 0
    assert stats.disputed == 0
    # The OLDER note is superseded by the newer one — newer survives.
    assert store.invalidated == [("a", "b")]
    assert store.bumped == []
    # The surviving note records the conflict (provenance parity with live
    # Phase 4, which sets contradicts=[existing_id] on the inserted note).
    assert store.updated == [("b", ["a"])]


def test_contradicts_does_not_record_link_on_dry_run():
    """Dry run computes the verdict but writes nothing — no contradicts link."""
    older = _note("a", "The service runs on port 8080.", "2026-01-01T00:00:00Z")
    newer = _note("b", "The service now runs on port 9000.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    contradicts_resp = '[{"existing_id": "a", "verdict": "CONTRADICTS", "reason": "port changed"}]'
    llm = _mock_llm(contradicts_resp, contradicts_resp)
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=False, log=log,
    )

    assert stats.contradictions == 1
    assert store.invalidated == []
    assert store.updated == []
    assert newer.contradicts == []  # in-memory note untouched on dry run


def test_disagreeing_second_opinion_is_disputed_and_skipped():
    """First call says DUPLICATE, second disagrees -> skipped, nothing written."""
    older = _note("a", "Keep responses: Short (unless depth is needed).", "2026-01-01T00:00:00Z")
    newer = _note("b", "Keep responses: Clear and structured.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    llm = _mock_llm(
        '[{"existing_id": "a", "verdict": "CONTRADICTS", "reason": "different value, same attribute"}]',
        '[{"existing_id": "a", "verdict": "COMPLEMENTS", "reason": "both can hold at once"}]',
    )
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.llm_calls == 2
    assert stats.disputed == 1
    assert stats.llm_duplicates == 0
    assert stats.contradictions == 0
    assert store.invalidated == []
    assert store.bumped == []


def test_no_second_opinion_acts_on_first_verdict_alone():
    older = _note("a", "Igor lives and works in London.", "2026-01-01T00:00:00Z")
    newer = _note("b", "Igor is currently working in central London.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    llm = _mock_llm('[{"existing_id": "a", "verdict": "DUPLICATE", "reason": "same fact"}]')
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log, second_opinion=False,
    )

    assert stats.llm_calls == 1
    assert stats.llm_duplicates == 1
    assert store.invalidated == [("b", "a")]


def test_llm_complements_verdict_leaves_both_notes_alone():
    older = _note("a", "Igor drinks tea, not coffee.", "2026-01-01T00:00:00Z")
    newer = _note("b", "Igor enjoys a solid cup of tea during long trips.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    llm = _mock_llm(f'[{{"existing_id": "a", "verdict": "COMPLEMENTS", "reason": "adds context"}}]')
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.llm_calls == 1
    assert stats.llm_duplicates == 0
    assert stats.contradictions == 0
    assert store.invalidated == []
    assert store.bumped == []


def test_verdict_with_unknown_existing_id_is_ignored_not_fatal():
    """A verdict referencing an existing_id absent from the shown neighbours
    must be skipped, not crash the run (live-corpus regression: gpt-4o
    occasionally echoes back an id that doesn't match any neighbour shown)."""
    older = _note("a", "Igor lives and works in London.", "2026-01-01T00:00:00Z")
    newer = _note("b", "Igor is currently working in central London.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    llm = _mock_llm(
        '[{"existing_id": "hallucinated-id-not-shown", "verdict": "DUPLICATE", "reason": "n/a"}]'
    )
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.llm_duplicates == 0
    assert stats.contradictions == 0
    assert stats.gated == 0
    assert store.invalidated == []


# --- preference-protection gate ---------------------------------------------
#
# 2026-06-07 review of a gpt-4o + adversarial-double-check dry run found a
# systematic bias: episodic notes (problem/workflow/project/fix) were wrongly
# judged DUPLICATE/CONTRADICTS against 'preference' notes (standing values),
# and the bias survived an independent second opinion (4/4 wrong; 0/3
# preference-vs-preference verdicts were). The gate overrides such verdicts
# using tag metadata the extract step already assigns — no extra LLM cost.


def test_preference_protection_gates_contradicts_from_non_preference_note():
    """A 'problem'-tagged note must not be allowed to supersede a 'preference' note."""
    older = _note("a", "The user wants the ai-memory connector to be the primary memory system.",
                  "2026-01-01T00:00:00Z", tags=["preference"])
    newer = _note("b", "The assistant used a different tool to save one fact.",
                  "2026-01-02T00:00:00Z", tags=["problem"])
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    llm = _mock_llm('[{"existing_id": "a", "verdict": "CONTRADICTS", "reason": "used a different tool"}]')
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.gated == 1
    assert stats.contradictions == 0
    assert stats.llm_duplicates == 0
    assert store.invalidated == []  # the standing preference survives untouched


def test_preference_protection_gates_duplicate_from_non_preference_note():
    """A 'workflow'-tagged note must not be merged away into a 'preference' note."""
    older = _note("a", "The user prefers the ai-memory connector to be the primary memory system.",
                  "2026-01-01T00:00:00Z", tags=["preference"])
    newer = _note("b", "The assistant suggested writing custom instructions for ai-memory.",
                  "2026-01-02T00:00:00Z", tags=["workflow"])
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    llm = _mock_llm('[{"existing_id": "a", "verdict": "DUPLICATE", "reason": "same topic"}]')
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.gated == 1
    assert stats.llm_duplicates == 0
    assert store.invalidated == []  # the episodic note is preserved, not discarded


def test_preference_vs_preference_duplicate_is_not_gated():
    """The gate must not block legitimate preference-vs-preference merges."""
    older = _note("a", "Direct, no sugar-coating.", "2026-01-01T00:00:00Z", tags=["preference"])
    newer = _note("b", "Prefers direct, concise answers.", "2026-01-02T00:00:00Z", tags=["preference"])
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    duplicate_resp = '[{"existing_id": "a", "verdict": "DUPLICATE", "reason": "same value"}]'
    llm = _mock_llm(duplicate_resp, duplicate_resp)
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.gated == 0
    assert stats.llm_duplicates == 1
    assert store.invalidated == [("b", "a")]


# --- pre-filter: clearly unrelated skips the LLM entirely -------------------


def test_clearly_unrelated_skips_llm_and_leaves_notes_alone():
    older = _note("a", "Igor drinks tea, not coffee.", "2026-01-01T00:00:00Z")
    newer = _note("b", "The Murphy bed needs a French cleat mount.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [-1.0, 0.0]}  # distance 2.0, far beyond ceiling
    llm = _mock_llm()
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.candidates_checked == 0
    assert stats.llm_calls == 0
    assert store.invalidated == []
    llm.complete.assert_not_called()


# --- dry run: report but don't write ----------------------------------------


def test_dry_run_reports_without_writing():
    older = _note("a", "Igor lives and works in London.", "2026-01-01T00:00:00Z")
    newer = _note("b", "Igor is currently working in central London.", "2026-01-02T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3]}
    duplicate_resp = '[{"existing_id": "a", "verdict": "DUPLICATE", "reason": "same fact"}]'
    llm = _mock_llm(duplicate_resp, duplicate_resp)
    service, store = _service([older, newer], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=False, log=log,
    )

    assert stats.llm_duplicates == 1
    # Decisions are computed (and logged) but never written.
    assert store.invalidated == []
    assert store.bumped == []


# --- chains: an invalidated note is skipped as a future candidate's neighbour


def test_invalidated_note_is_excluded_from_later_comparisons():
    a = _note("a", "Igor lives and works in London.", "2026-01-01T00:00:00Z")
    b = _note("b", "Igor is currently working in central London.", "2026-01-02T00:00:00Z")
    c = _note("c", "Igor works in central London most weekdays.", "2026-01-03T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3], "c": [0.7, 0.3]}
    # b gets merged into a; c must then be compared against a (the survivor),
    # not against the now-invalidated b. Each merge needs two agreeing calls.
    duplicate_resp = '[{"existing_id": "a", "verdict": "DUPLICATE", "reason": "same fact"}]'
    llm = _mock_llm(duplicate_resp, duplicate_resp, duplicate_resp, duplicate_resp)
    service, store = _service([a, b, c], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=None, apply=True, log=log,
    )

    assert stats.llm_calls == 4
    assert stats.llm_duplicates == 2
    assert store.invalidated == [("b", "a"), ("c", "a")]


# --- limit ------------------------------------------------------------------


def test_limit_stops_after_n_candidates():
    a = _note("a", "Igor lives and works in London.", "2026-01-01T00:00:00Z")
    b = _note("b", "Igor is currently working in central London.", "2026-01-02T00:00:00Z")
    c = _note("c", "Igor works in central London most weekdays.", "2026-01-03T00:00:00Z")
    embeddings = {"a": [1.0, 0.0], "b": [0.7, 0.3], "c": [0.7, 0.3]}
    llm = _mock_llm('[{"existing_id": "a", "verdict": "COMPLEMENTS", "reason": "n/a"}]')
    service, store = _service([a, b, c], embeddings, llm)
    _, log = _log_collector()

    stats = consolidate(
        service, duplicate_dist_below=0.15, unrelated_dist_above=1.05,
        neighbours_k=5, limit=1, apply=True, log=log,
    )

    assert stats.candidates_checked == 1
    assert stats.llm_calls == 1
