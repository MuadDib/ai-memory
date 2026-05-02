"""Pure-function tests for the Cowork transcript parser.

These exercise content-block flattening and timestamp parsing without
needing the storage stack. The end-to-end import is covered by an
integration test once we have a fixture session file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_memory.cowork_importer import (
    _flatten_blocks,
    _iter_session_files,
    _normalise_content,
    _parse_timestamp,
)


# --- _iter_session_files layout coverage -----------------------------------


def _touch(path: Path) -> Path:
    """Create a file and all parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


def test_iter_direct_layout(tmp_path: Path) -> None:
    """root IS the projects dir: root/<project>/<session>.jsonl (new Claude Code)."""
    f1 = _touch(tmp_path / "C--my-project" / "aabbccdd-0000-0000-0000-000000000001.jsonl")
    f2 = _touch(tmp_path / "C--my-project" / "aabbccdd-0000-0000-0000-000000000002.jsonl")
    found = set(_iter_session_files(tmp_path))
    assert found == {f1, f2}


def test_iter_nested_dotclaude_layout(tmp_path: Path) -> None:
    """Old Cowork UWP layout: root/.claude/projects/<project>/<session>.jsonl."""
    f = _touch(tmp_path / ".claude" / "projects" / "my-project" / "session-abc.jsonl")
    found = set(_iter_session_files(tmp_path))
    assert f in found


def test_iter_nested_projects_layout(tmp_path: Path) -> None:
    """Alternate layout: root/projects/<project>/<session>.jsonl."""
    f = _touch(tmp_path / "projects" / "my-project" / "session-abc.jsonl")
    found = set(_iter_session_files(tmp_path))
    assert f in found


def test_iter_no_duplicates_when_patterns_overlap(tmp_path: Path) -> None:
    """The seen-set must prevent a file from being yielded twice even if
    multiple glob patterns could match it."""
    # A file at root/projects/proj/s.jsonl is matched by both
    # rglob("projects/*/*.jsonl") and glob("*/*.jsonl") — only one yield expected.
    f = _touch(tmp_path / "projects" / "proj" / "s.jsonl")
    results = list(_iter_session_files(tmp_path))
    assert results.count(f) == 1


def test_iter_ignores_files_at_wrong_depth(tmp_path: Path) -> None:
    """A .jsonl file sitting directly under root (depth 1) is not a session file."""
    _touch(tmp_path / "stray.jsonl")
    found = set(_iter_session_files(tmp_path))
    assert found == set()


def test_iter_ignores_deeply_nested_non_session_files(tmp_path: Path) -> None:
    """A .jsonl buried 4+ levels deep that doesn't match any project layout
    should not appear — or if it does, the seen-set still prevents duplicates."""
    # This file is at a depth that glob("*/*.jsonl") won't reach and the
    # rglob patterns require a specific directory name, so it won't match.
    _touch(tmp_path / "a" / "b" / "c" / "d" / "deep.jsonl")
    found = set(_iter_session_files(tmp_path))
    assert found == set()


# --- Content normalisation -------------------------------------------------


def test_normalise_string_content() -> None:
    assert _normalise_content("hello world") == [{"type": "text", "text": "hello world"}]


def test_normalise_list_content_passthrough() -> None:
    blocks = [{"type": "text", "text": "a"}, {"type": "tool_use", "name": "x"}]
    assert _normalise_content(blocks) == blocks


def test_normalise_none_content() -> None:
    assert _normalise_content(None) == []


def test_normalise_list_with_strings_coerced() -> None:
    blocks = ["raw", {"type": "text", "text": "structured"}]
    out = _normalise_content(blocks)
    assert out[0] == {"type": "text", "text": "raw"}
    assert out[1] == {"type": "text", "text": "structured"}


# --- Flattening ------------------------------------------------------------


def test_flatten_text_only() -> None:
    blocks = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
    assert _flatten_blocks(blocks, include_tools=False) == "first\n\nsecond"


def test_flatten_skips_tool_use_by_default() -> None:
    blocks = [
        {"type": "text", "text": "before"},
        {"type": "tool_use", "name": "memory_recall", "input": {"query": "x"}},
        {"type": "text", "text": "after"},
    ]
    assert _flatten_blocks(blocks, include_tools=False) == "before\n\nafter"


def test_flatten_includes_tool_use_when_asked() -> None:
    blocks = [
        {"type": "text", "text": "before"},
        {"type": "tool_use", "name": "memory_recall", "input": {"query": "x"}},
    ]
    out = _flatten_blocks(blocks, include_tools=True)
    assert "before" in out
    assert "tool_use:memory_recall" in out


def test_flatten_skips_thinking_blocks() -> None:
    blocks = [
        {"type": "thinking", "text": "internal monologue"},
        {"type": "text", "text": "spoken answer"},
    ]
    assert _flatten_blocks(blocks, include_tools=False) == "spoken answer"


def test_flatten_drops_empty_text() -> None:
    blocks = [{"type": "text", "text": ""}, {"type": "text", "text": "real"}]
    assert _flatten_blocks(blocks, include_tools=False) == "real"


# --- Timestamp parsing -----------------------------------------------------


def test_parse_timestamp_iso() -> None:
    # ISO input passes through normalised (sub-second stripped, Z suffix kept).
    assert _parse_timestamp("2026-04-26T23:38:21Z") == "2026-04-26T23:38:21Z"


def test_parse_timestamp_epoch_int() -> None:
    # Unix epoch is converted to ISO 8601 UTC string.
    result = _parse_timestamp(1777246701)
    assert isinstance(result, str) and result.endswith("Z") and "T" in result


def test_parse_timestamp_epoch_float() -> None:
    result = _parse_timestamp(1777246701.5)
    assert isinstance(result, str) and result.endswith("Z") and "T" in result


def test_parse_timestamp_none() -> None:
    assert _parse_timestamp(None) is None


def test_parse_timestamp_garbage() -> None:
    assert _parse_timestamp("not a date") is None
