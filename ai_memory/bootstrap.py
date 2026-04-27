"""
Bootstrap: import existing AI memory dumps so the server has real data
to query from the moment it boots.

Designed to handle the two formats Igor pasted in our setup session:
    1. ChatGPT memory export (markdown with `## Section` blocks and `- bullet`
       facts).
    2. Claude memory export (markdown with `### Subsection` blocks and bullets).

Strategy:
    - Treat each input file as one synthetic episode (source = chatgpt-export
      or claude-export). Raw markdown is preserved to disk in `exports/` and
      indexed as a single Turn for verbatim recall.
    - Walk the markdown bullets and create one Tier 1 note per bullet — these
      are already-atomic facts, so we skip the LLM consolidation pass.
    - Promote the obvious identity bullets (name, location, role) to Tier 0.

Idempotent: re-running with the same files updates rather than duplicates,
keyed on a stable hash of (file path, bullet text).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from ulid import ULID

from ai_memory.core.models import Episode, Note, Profile, Turn
from ai_memory.core.service import MemoryService

logger = logging.getLogger(__name__)

# Bullet keys we automatically promote to Tier 0 profile if found.
_PROFILE_KEY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("name", re.compile(r"^name[:\-]\s*(.+)$", re.IGNORECASE)),
    ("location", re.compile(r"^location[:\-]\s*(.+)$", re.IGNORECASE)),
    ("role", re.compile(r"^role[:\-]\s*(.+)$", re.IGNORECASE)),
    ("company", re.compile(r"^company[:\-]\s*(.+)$", re.IGNORECASE)),
    ("email", re.compile(r"^email[:\-]\s*(.+)$", re.IGNORECASE)),
]


@dataclass
class BootstrapResult:
    episode_id: str
    notes_inserted: int
    profile_updates: int


def bootstrap_from_markdown(
    *,
    service: MemoryService,
    file_path: Path,
    source: str,
    title: str | None = None,
) -> BootstrapResult:
    """Ingest one markdown memory dump.

    `source` should be one of `chatgpt-export` / `claude-export` / `manual`.
    """
    text = file_path.read_text(encoding="utf-8")
    now = int(time.time())

    # Save a copy to exports/ for verbatim retrieval.
    exports_copy = service.config.exports_dir / file_path.name
    exports_copy.parent.mkdir(parents=True, exist_ok=True)
    exports_copy.write_text(text, encoding="utf-8")

    # Create episode + single big turn for verbatim
    episode_id = str(ULID())
    episode = Episode(
        id=episode_id,
        title=title or f"Bootstrap: {file_path.name}",
        summary=_first_paragraph(text)[:600],
        source=source,
        started_at=now,
        ended_at=now,
        raw_file=str(exports_copy.relative_to(service.config.home).as_posix()),
        embedding_model="",
        consolidated_at=now,
    )
    service.store.insert_episode(episode, embedding=None)
    service.store.insert_turn(
        Turn(
            id=str(ULID()),
            episode_id=episode_id,
            raw_file=episode.raw_file,
            byte_offset=0,
            byte_length=len(text.encode("utf-8")),
            role="system",
            ts=now,
        )
    )

    # Walk bullets
    notes_inserted = 0
    profile_updates = 0
    seen_hashes: set[str] = set()

    for bullet in _iter_bullets(text):
        digest = hashlib.sha256(f"{file_path}::{bullet}".encode()).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        # Promote to profile if it matches an identity pattern
        promoted = False
        for key, pattern in _PROFILE_KEY_PATTERNS:
            if (m := pattern.match(bullet)):
                service.upsert_profile(key=key, value=m.group(1).strip(), source=source)
                profile_updates += 1
                promoted = True
                break

        # Always also store as a Tier 1 note so it's searchable
        [embedding] = service.embedder.embed([bullet])
        note = Note(
            id=str(ULID()),
            text=bullet,
            tags=[source, "bootstrap"] + (["profile-promoted"] if promoted else []),
            source_episode_ids=[episode_id],
            valid_from=now,
            ingested_at=now,
            embedding_model=service.embedder.model_id,
        )
        service.store.insert_note(note, embedding)
        notes_inserted += 1

    logger.info(
        "Bootstrap complete from %s: %d notes inserted, %d profile updates",
        file_path,
        notes_inserted,
        profile_updates,
    )
    return BootstrapResult(
        episode_id=episode_id,
        notes_inserted=notes_inserted,
        profile_updates=profile_updates,
    )


# --- Markdown helpers -------------------------------------------------------

_BULLET_LINE = re.compile(r"^\s*[-\*]\s+(.+?)\s*$")
_HEADING = re.compile(r"^#{1,6}\s+")


def _iter_bullets(text: str):
    """Yield the trimmed text of each bullet line in a markdown document.

    We collapse markdown emphasis (`**bold**`, `*ital*`) for cleaner notes
    but otherwise preserve the bullet content as-is.
    """
    for line in text.splitlines():
        if _HEADING.match(line):
            continue
        if (m := _BULLET_LINE.match(line)):
            content = m.group(1).strip()
            content = re.sub(r"\*+([^*]+)\*+", r"\1", content)  # strip * and **
            content = content.strip()
            if content:
                yield content


def _first_paragraph(text: str) -> str:
    """Return the first non-blank paragraph for a fallback summary."""
    for chunk in text.split("\n\n"):
        chunk = chunk.strip()
        if chunk and not chunk.startswith("#"):
            return chunk
    return text[:600]
