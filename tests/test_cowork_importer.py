"""Pure-function tests for the Cowork transcript parser.

These exercise content-block flattening and timestamp parsing without
needing the storage stack. The end-to-end import is covered by an
integration test once we have a fixture session file.
"""
from __future__ import annotations

from ai_memory.cowork_importer import (
    _flatten_blocks,
    _normalise_content,
    _parse_timestamp,
)


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
    # Drop sub-second precision; equality on the second is enough.
    assert _parse_timestamp("2026-04-26T23:38:21Z") == 1777246701


def test_parse_timestamp_epoch_int() -> None:
    assert _parse_timestamp(1777246701) == 1777246701


def test_parse_timestamp_epoch_float() -> None:
    assert _parse_timestamp(1777246701.5) == 1777246701


def test_parse_timestamp_none() -> None:
    assert _parse_timestamp(None) is None


def test_parse_timestamp_garbage() -> None:
    assert _parse_timestamp("not a date") is None
