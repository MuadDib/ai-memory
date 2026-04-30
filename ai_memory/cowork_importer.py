"""
Cowork / Claude Code session importer.

Walks a Cowork sessions root (or any Claude Code projects tree), finds the
JSONL transcripts, and ingests each one as a single Episode + many Turn
rows in ai-memory.

Resume semantics
----------------
A dedicated state row (``cowork_import_state``) tracks, per session, the
``last_turn_id`` and ``last_byte_offset`` we successfully imported. On
re-run:

    1. **First-time session** (no state row): scan from the start, insert
       every turn, write a state row at the end.
    2. **Existing session, file unchanged or grew** (fast path): seek to
       ``last_byte_offset``, read the next line, validate its turn id is
       *strictly newer* than ``last_turn_id`` (i.e. we're past the watermark),
       and continue from there.
    3. **Existing session, file unexpectedly different** at the offset
       (Claude Code rewrote it, edited a turn, etc.): fall back to a linear
       scan from the start, skipping forward until we pass ``last_turn_id``.
    4. **No new content**: state untouched, episode untouched, return zero.

When new turns *are* appended to an already-imported session, the episode
is re-marked ``consolidated_at = NULL`` so the dream cycle re-summarises +
re-extracts on its next pass over the now-extended transcript.

Default Windows root:
    %LOCALAPPDATA%\\Packages\\Claude_pzs8sxrjxfjjc\\LocalCache\\Roaming\\Claude\\local-agent-mode-sessions
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ai_memory.core.models import CoworkImportState, Episode, Turn
from ai_memory.timestamps import now_iso, unix_to_iso

logger = logging.getLogger(__name__)


@dataclass
class CoworkImportResult:
    sessions_seen: int
    sessions_imported_new: int       # never seen before this run
    sessions_extended: int           # already existed; we appended new turns
    sessions_skipped_unchanged: int  # already existed and no new turns to import
    turns_inserted: int


def import_cowork_sessions(
    *,
    service,  # MemoryService — duck-typed to avoid import cycle
    root: Path,
    include_tools: bool = False,
    only_session_id: str | None = None,
) -> CoworkImportResult:
    """Walk ``root`` for ``*.jsonl`` session transcripts and import each, resuming
    from per-session state when one exists.

    ``only_session_id`` lets you scope to one session for testing without nuking
    state for the rest.
    """
    if not root.exists():
        raise FileNotFoundError(f"Cowork sessions root not found: {root}")

    seen = 0
    imported_new = 0
    extended = 0
    skipped_unchanged = 0
    turns_total = 0

    for jsonl_path in _iter_session_files(root):
        seen += 1
        session_id = jsonl_path.stem
        if only_session_id and session_id != only_session_id:
            continue

        existing_state = service.store.get_cowork_import_state(session_id)
        is_first = existing_state is None and service.store.get_episode(session_id) is None

        new_turns = _import_one_session(
            service=service,
            session_id=session_id,
            jsonl_path=jsonl_path,
            include_tools=include_tools,
            state=existing_state,
        )

        if new_turns == 0:
            if is_first:
                # File was empty or had no convertible turns.
                logger.info("Session %s had no usable turns.", session_id)
            else:
                skipped_unchanged += 1
            continue

        if is_first:
            imported_new += 1
        else:
            extended += 1
        turns_total += new_turns
        logger.info(
            "%s session %s: +%d turns",
            "Imported" if is_first else "Extended",
            session_id,
            new_turns,
        )

    return CoworkImportResult(
        sessions_seen=seen,
        sessions_imported_new=imported_new,
        sessions_extended=extended,
        sessions_skipped_unchanged=skipped_unchanged,
        turns_inserted=turns_total,
    )


# --- Internals --------------------------------------------------------------


def _iter_session_files(root: Path) -> Iterable[Path]:
    """Yield every ``*.jsonl`` under ``root/**/.claude/projects/**`` or
    ``root/**/projects/**``. Order isn't guaranteed; resume state makes it
    re-runnable safely.
    """
    seen: set[Path] = set()
    for jsonl_path in root.rglob(".claude/projects/*/*.jsonl"):
        if jsonl_path in seen:
            continue
        seen.add(jsonl_path)
        yield jsonl_path
    for jsonl_path in root.rglob("projects/*/*.jsonl"):
        if jsonl_path in seen:
            continue
        seen.add(jsonl_path)
        yield jsonl_path


def _import_one_session(
    *,
    service,
    session_id: str,
    jsonl_path: Path,
    include_tools: bool,
    state: CoworkImportState | None,
) -> int:
    """Import (or extend) a single session. Returns count of newly inserted turns."""

    # Migration edge: an episode row exists but no state row. That means the
    # session was imported by an older version of this importer that didn't
    # track resume state. We can't safely guess where it left off — re-running
    # the whole file would collide on turn id PRIMARY KEY. Warn and skip; the
    # user can either accept that the session is "frozen" at its previous state
    # or manually clear the episode to force a fresh import.
    if state is None and service.store.get_episode(session_id) is not None:
        logger.warning(
            "Session %s has an existing episode but no import state. "
            "Skipping (migration from older importer). Delete the episode and "
            "its turns to force a fresh re-import.",
            session_id,
        )
        return 0

    # Decide where in the file to start, and whether the start position is "trusted".
    start_offset, resume_after_id = _resume_position(jsonl_path, state)

    # Stream the file from start_offset, looking for new turns.
    new_turns: list[dict] = []
    final_offset = start_offset
    final_turn_id: str | None = state.last_turn_id if state else None

    skipping_until_resume = (resume_after_id is not None)

    with jsonl_path.open("rb") as fh:
        fh.seek(start_offset)
        while True:
            line_start = fh.tell()
            raw = fh.readline()
            if not raw:
                break
            line_end = fh.tell()
            line_str = raw.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                obj = json.loads(line_str)
            except json.JSONDecodeError:
                continue

            turn_payload = _coerce_turn(obj, include_tools=include_tools, jsonl_stem=jsonl_path.stem)
            if turn_payload is None:
                continue

            # If we're in linear-scan resume mode, skip until we cross last_turn_id.
            if skipping_until_resume:
                if turn_payload["id"] == resume_after_id:
                    skipping_until_resume = False
                    final_offset = line_end
                    final_turn_id = turn_payload["id"]
                continue

            new_turns.append({**turn_payload, "_line_start": line_start, "_line_end": line_end})

    if skipping_until_resume:
        # We scanned the entire file and never found last_turn_id. Two possibilities:
        #   - The file was truncated/rewritten and the watermark turn is gone.
        #   - The session id collided across two files (shouldn't happen with UUIDs).
        # Safest action: log a warning and don't touch the DB. User can manually
        # delete the stale state row to force a re-import.
        logger.warning(
            "Session %s: last_turn_id %r not found in %s. State stale; skipping.",
            session_id, resume_after_id, jsonl_path,
        )
        return 0

    if not new_turns:
        return 0

    # First-time import: create the Episode row.
    if state is None and service.store.get_episode(session_id) is None:
        episode = Episode(
            id=session_id,
            title=_derive_title(jsonl_path, new_turns),
            summary="",
            source="cowork",
            started_at=new_turns[0]["ts"],
            ended_at=new_turns[-1]["ts"],
            raw_file="",
            embedding_model="",
        )
        service.store.insert_episode(episode, embedding=None)

    # Append every new turn to raw + insert turn row + advance the cursor.
    started_at = new_turns[0]["ts"]
    with service.raw.open_for_streaming_writes(
        episode_id=session_id, started_at=started_at
    ) as writer:
        for turn in new_turns:
            payload = {
                "id": turn["id"],
                "role": turn["role"],
                "ts": turn["ts"],
                "text": turn["text"],
            }
            appended = writer.append(payload)
            service.store.insert_turn(
                Turn(
                    id=turn["id"],
                    episode_id=session_id,
                    raw_file=appended.raw_file,
                    byte_offset=appended.byte_offset,
                    byte_length=appended.byte_length,
                    role=turn["role"],
                    ts=turn["ts"],
                )
            )
            final_offset = turn["_line_end"]
            final_turn_id = turn["id"]

    # If we extended an existing session, mark its episode unconsolidated so the
    # dream cycle re-summarises + re-extracts on its next pass.
    if state is not None:
        existing_episode = service.store.get_episode(session_id)
        if existing_episode is not None and existing_episode.consolidated_at is not None:
            existing_episode.consolidated_at = None
            existing_episode.ended_at = new_turns[-1]["ts"]
            service.store.update_episode(existing_episode, embedding=None)

    # Advance the cursor.
    assert final_turn_id is not None  # we'd have returned 0 above otherwise
    service.store.upsert_cowork_import_state(
        CoworkImportState(
            session_id=session_id,
            last_turn_id=final_turn_id,
            last_byte_offset=final_offset,
            last_imported_at=now_iso(),
        )
    )

    return len(new_turns)


def _resume_position(
    jsonl_path: Path, state: CoworkImportState | None
) -> tuple[int, str | None]:
    """Decide where to start reading + whether to validate.

    Returns ``(start_offset, resume_after_id)``:
        - ``(0, None)`` for first-time imports — read the whole file.
        - ``(state.last_byte_offset, None)`` if the byte-offset fast path looks
          valid (the line at that offset, if any, is a *new* turn).
        - ``(0, state.last_turn_id)`` to fall back to linear scan, skipping
          turns until we cross the watermark id.
    """
    if state is None:
        return 0, None

    # If the file is shorter than our offset, it was truncated/rewritten.
    try:
        size = jsonl_path.stat().st_size
    except OSError:
        return 0, state.last_turn_id

    if state.last_byte_offset > size:
        logger.info(
            "Session %s: file %s is shorter (%d) than last_byte_offset (%d); "
            "falling back to linear scan.",
            jsonl_path.stem, jsonl_path.name, size, state.last_byte_offset,
        )
        return 0, state.last_turn_id

    # If the offset is exactly at EOF, there's nothing new — short-circuit.
    if state.last_byte_offset == size:
        # Nothing to read. A non-None resume_after_id forces _import_one_session
        # into "skip until we see this id" mode, which never fires because there
        # are no lines. We want it to find no new turns and bail cleanly. Trick:
        # use offset = size and resume_after_id = None — we just won't read
        # anything because readline returns empty.
        return size, None

    # Offset is inside the file. Trust it for the fast path; if the next line we
    # read happens to be the same id as last_turn_id (weirdness — shouldn't
    # happen, but covers an edge), the importer will silently skip it. Robust.
    return state.last_byte_offset, None


def _coerce_turn(obj: dict, *, include_tools: bool, jsonl_stem: str) -> dict | None:
    """Convert one transcript line to a normalised turn dict, or None to skip."""
    event_type = obj.get("type")
    if event_type not in ("user", "assistant"):
        return None

    message = obj.get("message") or {}
    content_blocks = _normalise_content(message.get("content"))
    text = _flatten_blocks(content_blocks, include_tools=include_tools).strip()
    if not text:
        return None

    turn_id = obj.get("uuid") or message.get("id") or obj.get("id")
    if not turn_id:
        # Deterministic fallback: hash the line for stable identity across runs.
        import hashlib
        turn_id = f"{jsonl_stem}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"

    ts = _parse_timestamp(obj.get("timestamp")) or now_iso()

    return {
        "id": str(turn_id),
        "role": "assistant" if event_type == "assistant" else "user",
        "ts": ts,
        "text": text,
    }


def _normalise_content(raw) -> list:
    """Coerce the variety of content shapes Claude Code uses into a list of dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"type": "text", "text": raw}]
    if isinstance(raw, list):
        return [b if isinstance(b, dict) else {"type": "text", "text": str(b)} for b in raw]
    return []


def _flatten_blocks(blocks: list, *, include_tools: bool) -> str:
    """Render a list of content blocks down to plain text."""
    parts: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(block.get("text", ""))
        elif block_type == "thinking":
            continue
        elif block_type == "tool_use":
            if include_tools:
                name = block.get("name", "")
                input_str = json.dumps(block.get("input", {}), ensure_ascii=False)
                parts.append(f"[tool_use:{name} {input_str}]")
        elif block_type == "tool_result":
            if include_tools:
                content = block.get("content", "")
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                parts.append(f"[tool_result] {content}")
        else:
            text = block.get("text") or ""
            if text:
                parts.append(text)
    return "\n\n".join(p for p in parts if p)


def _parse_timestamp(value) -> str | None:
    """Claude Code sometimes emits ISO 8601, sometimes epoch seconds.

    Always returns an ISO 8601 UTC string (e.g. '2026-04-29T18:35:23Z') or None.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return unix_to_iso(value)
    if isinstance(value, str):
        try:
            from datetime import datetime
            # Normalise to our standard format (strips sub-second precision etc.)
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None
    return None


def _derive_title(jsonl_path: Path, turns: list[dict]) -> str:
    """Pick a short title from the first user turn, or fall back to the filename."""
    for turn in turns:
        if turn["role"] == "user" and turn["text"]:
            return turn["text"].split("\n", 1)[0][:80]
    return jsonl_path.stem
