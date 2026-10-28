"""Pure-function tests for the Cowork transcript parser.

These exercise content-block flattening and timestamp parsing without
needing the storage stack. The end-to-end import is covered by an
integration test once we have a fixture session file.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_memory.cowork_importer import (
    _coerce_turn,
    _derive_title,
    _flatten_blocks,
    _iter_session_files,
    _normalise_content,
    _parse_timestamp,
    import_cowork_sessions,
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


# --- Harness boilerplate stripping (ADR-0015) --------------------------------

_REMINDER = "<system-reminder>\nContents of CLAUDE.md: be direct\n</system-reminder>"


def _line(uuid: str, kind: str, content, ts: str = "2026-10-10T20:00:00Z") -> dict:
    return {"type": kind, "uuid": uuid, "timestamp": ts, "message": {"content": content}}


def test_coerce_user_turn_strips_reminder() -> None:
    turn = _coerce_turn(
        _line("u1", "user", f"{_REMINDER}\n\nlet's fix ai-memory"),
        include_tools=False, jsonl_stem="s",
    )
    assert turn is not None and turn["text"] == "let's fix ai-memory"


def test_coerce_reminder_only_user_turn_is_dropped() -> None:
    # Claude Code attaches reminders as a text block next to tool_result blocks.
    content = [{"type": "tool_result", "content": "ok"}, {"type": "text", "text": _REMINDER}]
    assert _coerce_turn(_line("u1", "user", content), include_tools=False, jsonl_stem="s") is None


def test_coerce_assistant_turn_untouched() -> None:
    turn = _coerce_turn(
        _line("a1", "assistant", [{"type": "text", "text": f"quoting {_REMINDER}"}]),
        include_tools=False, jsonl_stem="s",
    )
    assert turn is not None and "<system-reminder>" in turn["text"]


def test_derive_title_skips_wrapper_only_turns() -> None:
    turns = [
        {"role": "user", "text": "<task-notification><status>done</status></task-notification>"},
        {"role": "assistant", "text": "noted"},
        {"role": "user", "text": "now retitle the episodes"},
    ]
    assert _derive_title(Path("sess.jsonl"), turns) == "now retitle the episodes"


def test_derive_title_falls_back_to_stem() -> None:
    assert _derive_title(Path("sess.jsonl"), [{"role": "assistant", "text": "hi"}]) == "sess"


# --- End-to-end import against a temp store ----------------------------------


def _write_session(path: Path, lines: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def service(tmp_path: Path):
    pytest.importorskip("sqlite_vec")
    from ai_memory.storage.raw_files import RawTranscriptStore
    from ai_memory.storage.sqlite_store import SqliteStore

    store = SqliteStore(tmp_path / "memory.db", embedding_dim=4)
    store.initialise()
    yield SimpleNamespace(store=store, raw=RawTranscriptStore(tmp_path / "raw"))
    store.close()


def test_import_stores_clean_text_and_title(service, tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_session(root / "proj" / "sess1.jsonl", [
        _line("u1", "user", f"{_REMINDER}\n\nlet's fix ai-memory"),
        _line("a1", "assistant", [{"type": "text", "text": "on it"}]),
        _line("u2", "user", [{"type": "tool_result", "content": "x"}, {"type": "text", "text": _REMINDER}]),
    ])

    result = import_cowork_sessions(service=service, root=root)

    assert result.turns_inserted == 2  # reminder-only u2 dropped
    episode = service.store.get_episode("sess1")
    assert episode.title == "let's fix ai-memory"
    texts = [
        service.raw.read_turn(t.raw_file, t.byte_offset, t.byte_length)["text"]
        for t in service.store.get_turns_for_episode("sess1")
    ]
    assert texts == ["let's fix ai-memory", "on it"]


_SCHEDULED = (
    '<scheduled-task name="weekly-ai-memory-health" file="C:\\tasks\\SKILL.md">\n'
    "This is an automated run of a scheduled task. The user is not present.\n"
    "</scheduled-task>"
)


def test_import_skips_scheduled_task_runs(service, tmp_path: Path) -> None:
    root = tmp_path / "projects"
    _write_session(root / "proj" / "audit.jsonl", [
        _line("u1", "user", _SCHEDULED),
        _line("a1", "assistant", [{"type": "text", "text": "running the audit"}]),
    ])
    # The app puts a reminder ahead of the task prompt in some sessions.
    _write_session(root / "proj" / "watch.jsonl", [
        _line("u1", "user", f"{_REMINDER}\n\n{_SCHEDULED}"),
    ])
    # Discussing scheduled tasks later in a normal session must not match.
    _write_session(root / "proj" / "chat.jsonl", [
        _line("u1", "user", "why did my audit fail?"),
        _line("u2", "user", f"it started with {_SCHEDULED}"),
    ])

    result = import_cowork_sessions(service=service, root=root)

    assert result.sessions_skipped_automated == 2
    assert result.sessions_imported_new == 1
    assert service.store.get_episode("audit") is None
    assert service.store.get_episode("watch") is None
    assert service.store.get_episode("chat") is not None


def test_import_stops_extending_a_previously_imported_scheduled_run(service, tmp_path: Path) -> None:
    root = tmp_path / "projects"
    path = _write_session(root / "proj" / "audit.jsonl", [_line("u1", "user", "placeholder")])
    import_cowork_sessions(service=service, root=root)  # imported under the old rule
    _write_session(path, [
        _line("u1", "user", _SCHEDULED),
        _line("a1", "assistant", [{"type": "text", "text": "more audit output"}]),
    ])

    result = import_cowork_sessions(service=service, root=root)

    assert result.sessions_skipped_automated == 1 and result.turns_inserted == 0


def test_linear_scan_resumes_past_reminder_only_watermark(service, tmp_path: Path) -> None:
    # State written by the pre-ADR-0015 importer, whose watermark is a turn that
    # now strips to nothing. Force the linear-scan path (offset past EOF).
    from ai_memory.core.models import CoworkImportState

    root = tmp_path / "projects"
    _write_session(root / "proj" / "sess2.jsonl", [
        _line("u1", "user", "hello"),
        _line("u2", "user", _REMINDER),
        _line("a2", "assistant", [{"type": "text", "text": "new reply"}]),
    ])
    import_cowork_sessions(service=service, root=root)  # creates the episode
    service.store.upsert_cowork_import_state(CoworkImportState(
        session_id="sess2", last_turn_id="u2", last_byte_offset=10**9,
        last_imported_at="2026-10-10T20:00:00Z",
    ))
    service.store.conn.execute("DELETE FROM turns WHERE id = 'a2'")
    service.store.conn.commit()

    result = import_cowork_sessions(service=service, root=root)

    assert result.sessions_extended == 1 and result.turns_inserted == 1
    assert service.store.get_cowork_import_state("sess2").last_turn_id == "a2"
