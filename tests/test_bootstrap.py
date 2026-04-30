"""Tests for ai_memory/bootstrap.py.

Pure-function tests cover _iter_bullets and _is_useless; integration-style
tests (with mocked MemoryService) cover the cross-file dedup guard.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pytest

from ai_memory.bootstrap import (
    BOOTSTRAP_DEDUP_DIST,
    BOOTSTRAP_DEDUP_MIDBAND,
    BootstrapResult,
    _is_useless,
    _iter_bullets,
    bootstrap_from_markdown,
)
from ai_memory.core.models import Note


# ---------------------------------------------------------------------------
# _is_useless
# ---------------------------------------------------------------------------

def test_is_useless_single_word_short() -> None:
    assert _is_useless("Python") is True


def test_is_useless_has_colon() -> None:
    # A bare token with a colon carries context — not useless.
    assert _is_useless("Name: Igor") is False


def test_is_useless_has_space() -> None:
    assert _is_useless("uses Python") is False


def test_is_useless_long_single_word() -> None:
    # Boundary is len < 20 — exactly 19 chars is still useless; 20 is not.
    assert _is_useless("a" * 19) is True
    assert _is_useless("a" * 20) is False


# ---------------------------------------------------------------------------
# _iter_bullets
# ---------------------------------------------------------------------------

def test_iter_bullets_simple() -> None:
    md = "- Name: Igor Valjevic\n- Location: London\n"
    facts = list(_iter_bullets(md))
    assert "Name: Igor Valjevic" in facts
    assert "Location: London" in facts


def test_iter_bullets_nested_with_label() -> None:
    md = "- Prefers:\n  - Direct answers\n  - Short responses\n"
    facts = list(_iter_bullets(md))
    assert "Prefers: Direct answers" in facts
    assert "Prefers: Short responses" in facts


def test_iter_bullets_heading_resets_parent() -> None:
    md = "## Section\n- Parent:\n  - Child\n## New Section\n- Uses Python daily\n"
    facts = list(_iter_bullets(md))
    # "Parent: Child" emitted before heading reset
    assert "Parent: Child" in facts
    # After heading reset, top-level bullet has no parent prefix
    assert "Uses Python daily" in facts
    assert "Parent: Uses Python daily" not in facts


def test_iter_bullets_skips_useless() -> None:
    md = "- Go\n- Uses Python daily\n"
    facts = list(_iter_bullets(md))
    assert "Go" not in facts          # single short word, no colon/space
    assert "Uses Python daily" in facts


def test_iter_bullets_strips_bold() -> None:
    md = "- **Name**: Igor\n"
    facts = list(_iter_bullets(md))
    assert any("Name" in f for f in facts)
    assert all("**" not in f for f in facts)


def test_iter_bullets_label_only_not_emitted() -> None:
    """A bullet ending in ':' is a sub-header label — should not be yielded."""
    md = "- Prefers:\n  - Concise answers\n"
    facts = list(_iter_bullets(md))
    assert "Prefers:" not in facts
    assert "Prefers" not in facts


# ---------------------------------------------------------------------------
# Cross-file dedup guard in bootstrap_from_markdown
# ---------------------------------------------------------------------------

def _make_llm_mock(verdict: str = "KEEP") -> MagicMock:
    """Return an LLM mock that responds with `verdict` from .complete().text."""
    llm = MagicMock()
    completion = MagicMock()
    completion.text = verdict
    llm.complete.return_value = completion
    return llm


def _make_service_mock(
    *,
    nearest_dist: float | None = None,
    llm_verdict: str = "KEEP",
) -> MagicMock:
    """Return a MemoryService mock wired up for bootstrap_from_markdown.

    `llm_verdict` controls what the LLM mock returns when a mid-band
    similarity check is triggered (BOOTSTRAP_DEDUP_DIST <= dist < BOOTSTRAP_DEDUP_MIDBAND).
    """
    svc = MagicMock()
    # config.exports_dir needs to be a real path we can write to.
    tmp = Path(tempfile.mkdtemp())
    svc.config.exports_dir = tmp
    svc.config.home = tmp

    svc.embedder.embed.return_value = [[0.0] * 1536]
    svc.embedder.model_id = "text-embedding-3-small"
    svc.llm = _make_llm_mock(llm_verdict)

    if nearest_dist is None:
        # No similar notes exist yet
        svc.store.search_notes_vector.return_value = []
    else:
        # Simulate an existing note at distance `nearest_dist`
        existing = MagicMock(spec=Note)
        existing.id = "aabbccdd-0000-0000-0000-000000000000"
        existing.text = "Existing fact text"
        svc.store.search_notes_vector.return_value = [(existing, nearest_dist)]

    return svc


def _write_md(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "test.md"
    p.write_text(content, encoding="utf-8")
    return p


def test_dedup_near_duplicate_skipped(tmp_path: Path) -> None:
    """A note whose nearest neighbour is within BOOTSTRAP_DEDUP_DIST is skipped."""
    svc = _make_service_mock(nearest_dist=BOOTSTRAP_DEDUP_DIST - 0.05)
    md = _write_md(tmp_path, "- Name: Igor\n")

    result = bootstrap_from_markdown(service=svc, file_path=md, source="test")

    assert result.notes_skipped == 1
    assert result.notes_inserted == 0
    svc.store.insert_note.assert_not_called()


def test_dedup_distant_note_inserted(tmp_path: Path) -> None:
    """A note beyond BOOTSTRAP_DEDUP_MIDBAND is inserted without any LLM call."""
    svc = _make_service_mock(nearest_dist=BOOTSTRAP_DEDUP_MIDBAND + 0.05)
    md = _write_md(tmp_path, "- Uses PostgreSQL for the main database\n")

    result = bootstrap_from_markdown(service=svc, file_path=md, source="test")

    assert result.notes_inserted == 1
    assert result.notes_skipped == 0
    assert result.notes_llm_deduped == 0
    svc.store.insert_note.assert_called_once()
    svc.llm.complete.assert_not_called()  # beyond mid-band: no LLM needed


def test_dedup_no_existing_notes_inserts(tmp_path: Path) -> None:
    """When the store is empty (no neighbours), every bullet is inserted."""
    svc = _make_service_mock(nearest_dist=None)
    md = _write_md(tmp_path, "- Location: London\n- Role: Senior developer\n")

    result = bootstrap_from_markdown(service=svc, file_path=md, source="test")

    assert result.notes_inserted == 2
    assert result.notes_skipped == 0
    assert result.notes_llm_deduped == 0


def test_dedup_threshold_boundary_exact(tmp_path: Path) -> None:
    """A neighbour at exactly BOOTSTRAP_DEDUP_DIST triggers the mid-band LLM check."""
    # LLM says KEEP → should be inserted
    svc = _make_service_mock(nearest_dist=BOOTSTRAP_DEDUP_DIST, llm_verdict="KEEP")
    md = _write_md(tmp_path, "- Uses Python for scripting\n")

    result = bootstrap_from_markdown(service=svc, file_path=md, source="test")

    # dist == threshold → not < threshold → mid-band LLM check → KEEP → inserted
    assert result.notes_inserted == 1
    assert result.notes_skipped == 0
    assert result.notes_llm_deduped == 0
    svc.llm.complete.assert_called_once()


# ---------------------------------------------------------------------------
# Mid-band LLM dedup tests
# ---------------------------------------------------------------------------

def test_dedup_midband_llm_says_duplicate(tmp_path: Path) -> None:
    """Mid-band note classified DUPLICATE by LLM is skipped."""
    dist = (BOOTSTRAP_DEDUP_DIST + BOOTSTRAP_DEDUP_MIDBAND) / 2
    svc = _make_service_mock(nearest_dist=dist, llm_verdict="DUPLICATE")
    md = _write_md(tmp_path, "- Role: C# programmer and team lead\n")

    result = bootstrap_from_markdown(service=svc, file_path=md, source="test")

    assert result.notes_llm_deduped == 1
    assert result.notes_inserted == 0
    assert result.notes_skipped == 0
    svc.store.insert_note.assert_not_called()


def test_dedup_midband_llm_says_keep(tmp_path: Path) -> None:
    """Mid-band note classified KEEP by LLM is inserted."""
    dist = (BOOTSTRAP_DEDUP_DIST + BOOTSTRAP_DEDUP_MIDBAND) / 2
    svc = _make_service_mock(nearest_dist=dist, llm_verdict="KEEP")
    md = _write_md(tmp_path, "- AWS (API Gateway, DynamoDB, Step Functions, Lambda)\n")

    result = bootstrap_from_markdown(service=svc, file_path=md, source="test")

    assert result.notes_inserted == 1
    assert result.notes_llm_deduped == 0
    svc.store.insert_note.assert_called_once()


def test_dedup_midband_llm_error_falls_back_to_keep(tmp_path: Path) -> None:
    """LLM exception in mid-band check falls back to inserting (never silently drops)."""
    dist = (BOOTSTRAP_DEDUP_DIST + BOOTSTRAP_DEDUP_MIDBAND) / 2
    svc = _make_service_mock(nearest_dist=dist)
    svc.llm.complete.side_effect = RuntimeError("API down")
    md = _write_md(tmp_path, "- Works with AWS services\n")

    result = bootstrap_from_markdown(service=svc, file_path=md, source="test")

    assert result.notes_inserted == 1
    assert result.notes_llm_deduped == 0


def test_dedup_midband_upper_boundary_no_llm(tmp_path: Path) -> None:
    """A note at exactly BOOTSTRAP_DEDUP_MIDBAND is inserted without LLM."""
    svc = _make_service_mock(nearest_dist=BOOTSTRAP_DEDUP_MIDBAND)
    md = _write_md(tmp_path, "- Enjoys cooking and baking at home\n")

    result = bootstrap_from_markdown(service=svc, file_path=md, source="test")

    assert result.notes_inserted == 1
    svc.llm.complete.assert_not_called()


def test_bootstrap_result_fields() -> None:
    """BootstrapResult dataclass exposes all counter fields, defaulting to 0."""
    r = BootstrapResult(episode_id="x", notes_inserted=3, profile_updates=1)
    assert r.notes_skipped == 0
    assert r.notes_llm_deduped == 0
