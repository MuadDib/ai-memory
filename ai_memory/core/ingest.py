"""
Write pipeline: append a turn to raw storage, index it, and (optionally) start
a new episode if there isn't an open one.

`remember(text, source, episode_id?)` is the canonical entry point.

Episodes are opened automatically — if no `episode_id` is supplied and there
is no recent open episode for `source`, a new one is created. We don't try
to be clever about session boundaries on the write path; the dream cycle
clusters turns into proper episodes overnight (Phase 2 of the cycle).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from ulid import ULID

from ai_memory.core.models import Episode, Turn
from ai_memory.privacy import redact
from ai_memory.storage.raw_files import RawTranscriptStore

# Anything from `MemoryStore` we use as a structural type to avoid a circular
# import — the recall pipeline imports models too.
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_memory.storage.interface import MemoryStore


# Max age in seconds before we close an open episode and start a new one.
# 4 hours is generous; the dream cycle re-segments by topic anyway.
_EPISODE_MAX_AGE_SECONDS = 4 * 3600


@dataclass(frozen=True)
class RememberResult:
    """Returned by `remember`."""

    turn_id: str
    episode_id: str
    redactions: list[str]


def remember(
    *,
    store: "MemoryStore",
    raw: RawTranscriptStore,
    text: str,
    source: str,
    role: str = "user",
    episode_id: str | None = None,
    now: int | None = None,
) -> RememberResult:
    """Append one turn. Open a new episode if needed.

    Always runs the privacy redactor first; the un-redacted text is never
    persisted (anywhere — neither raw file nor index).
    """
    now = int(now if now is not None else time.time())
    redaction = redact(text)
    safe_text = redaction.text

    # Resolve / open episode
    if episode_id is None:
        episode_id = _find_or_open_episode(store, source=source, now=now)

    # Append to raw file
    payload = {
        "id": str(ULID()),
        "role": role,
        "ts": now,
        "text": safe_text,
    }
    appended = raw.append_turn(episode_id=episode_id, started_at=now, payload=payload)

    # Index the turn
    turn = Turn(
        id=payload["id"],
        episode_id=episode_id,
        raw_file=appended.raw_file,
        byte_offset=appended.byte_offset,
        byte_length=appended.byte_length,
        role=role,  # type: ignore[arg-type]
        ts=now,
    )
    store.insert_turn(turn)

    return RememberResult(
        turn_id=turn.id, episode_id=episode_id, redactions=redaction.redactions
    )


def _find_or_open_episode(store: "MemoryStore", source: str, now: int) -> str:
    """Return id of a recent unconsolidated episode for `source`, or create one."""
    recent = store.list_recent_episodes(limit=10)
    for ep in recent:
        if ep.source != source:
            continue
        if ep.consolidated_at is not None:
            continue
        if now - ep.started_at > _EPISODE_MAX_AGE_SECONDS:
            continue
        return ep.id

    new_id = str(ULID())
    episode = Episode(
        id=new_id,
        title="",  # populated by dream cycle
        summary="",  # populated by dream cycle
        source=source,
        started_at=now,
        ended_at=None,
        raw_file="",  # we set per-turn raw_file on the Turn rows; episodes can derive
        embedding_model="",
    )
    store.insert_episode(episode, embedding=None)
    return new_id
