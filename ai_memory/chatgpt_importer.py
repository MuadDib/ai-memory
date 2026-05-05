"""
ChatGPT conversation export importer.

Reads the zip file produced by ChatGPT's "Export data" feature (Settings →
Data controls → Export) and ingests each conversation as an Episode + Turns
in ai-memory, exactly like cowork_importer does for Claude Code sessions.

Zip layout expected:
    conversations-000.json   ← array of conversation objects (may be split)
    conversations-001.json
    ...
    export_manifest.json     ← ignored
    user.json                ← ignored
    file-*                   ← media attachments, ignored

Conversation object shape (relevant fields only):
    {
      "id":           "<uuid>",
      "title":        "...",
      "create_time":  1234567890.0,   # Unix float
      "update_time":  1234567890.0,
      "current_node": "<node-uuid>",  # leaf of the canonical branch
      "mapping": {
        "<node-uuid>": {
          "id":       "<node-uuid>",
          "parent":   "<parent-uuid>" | null,
          "children": ["<child-uuid>", ...],
          "message":  null | {
            "id":          "<msg-uuid>",
            "author":      {"role": "system"|"user"|"assistant"|"tool"},
            "content":     {"content_type": "text", "parts": ["..."]},
            "create_time": 1234567890.0 | null,
            "status":      "finished_successfully" | ...
          }
        },
        ...
      }
    }

Resume semantics:
    Import state is stored in cowork_import_state keyed on conversation id.
    last_turn_id = the current_node of the conversation at last import time.
    If current_node matches → unchanged, skip.
    If different → conversation was extended; re-import the whole thing (safe
    because turn ids are stable UUIDs — INSERT OR IGNORE on the turn table
    avoids duplicates; episode is updated).

    On first import: insert episode + all turns; write state row.
"""
from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ai_memory.core.models import CoworkImportState, Episode, Turn
from ai_memory.ids import new_id
from ai_memory.timestamps import now_iso, unix_to_iso

logger = logging.getLogger(__name__)

# Roles we store; everything else (system, tool) is skipped by default.
_KEEP_ROLES = {"user", "assistant"}


@dataclass
class ChatGPTImportResult:
    conversations_seen: int
    conversations_imported_new: int
    conversations_extended: int
    conversations_skipped_unchanged: int
    turns_inserted: int


def import_chatgpt_zip(
    *,
    service,  # MemoryService — duck-typed to avoid import cycle
    zip_path: Path,
    include_tools: bool = False,
) -> ChatGPTImportResult:
    """Walk every conversation in ``zip_path`` and import into ai-memory."""
    if not zip_path.exists():
        raise FileNotFoundError(f"ChatGPT export zip not found: {zip_path}")

    seen = 0
    imported_new = 0
    extended = 0
    skipped = 0
    turns_total = 0

    for convo in _iter_conversations(zip_path):
        seen += 1
        convo_id = convo.get("id", "")
        if not convo_id:
            continue

        current_node = convo.get("current_node", "")
        existing_state: CoworkImportState | None = service.store.get_cowork_import_state(convo_id)

        # Unchanged: current_node hasn't moved since last import.
        if existing_state is not None and existing_state.last_turn_id == current_node:
            skipped += 1
            continue

        is_first = existing_state is None and service.store.get_episode(convo_id) is None

        n = _import_one_conversation(
            service=service,
            convo=convo,
            include_tools=include_tools,
        )

        if n == 0:
            continue

        if is_first:
            imported_new += 1
        else:
            extended += 1
        turns_total += n
        logger.info(
            "%s conversation %.8s (%s): +%d turns",
            "Imported" if is_first else "Extended",
            convo_id,
            (convo.get("title") or "")[:40],
            n,
        )

    return ChatGPTImportResult(
        conversations_seen=seen,
        conversations_imported_new=imported_new,
        conversations_extended=extended,
        conversations_skipped_unchanged=skipped,
        turns_inserted=turns_total,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _iter_conversations(zip_path: Path) -> Iterator[dict]:
    """Yield every conversation dict from all conversations-NNN.json files."""
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(e.filename for e in z.infolist() if e.filename.startswith("conversations-") and e.filename.endswith(".json"))
        for name in names:
            with z.open(name) as f:
                try:
                    batch = json.load(f)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed %s: %s", name, exc)
                    continue
                if isinstance(batch, list):
                    yield from batch
                elif isinstance(batch, dict):
                    yield batch


def _import_one_conversation(
    *,
    service,
    convo: dict,
    include_tools: bool,
) -> int:
    """Import or re-import one conversation. Returns count of turns written."""
    convo_id: str = convo["id"]
    current_node: str = convo.get("current_node", "")
    mapping: dict = convo.get("mapping", {})

    turns = _extract_linear_thread(mapping, current_node, include_tools=include_tools)
    if not turns:
        return 0

    now = now_iso()
    started_at = turns[0]["ts"]
    ended_at = turns[-1]["ts"]
    title = (convo.get("title") or convo_id)[:200]

    # Upsert episode — create on first import, update ended_at on extension.
    existing_episode = service.store.get_episode(convo_id)
    if existing_episode is None:
        episode = Episode(
            id=convo_id,
            title=title,
            summary="",
            source="chatgpt",
            started_at=started_at,
            ended_at=ended_at,
            raw_file="",
            embedding_model="",
        )
        service.store.insert_episode(episode, embedding=None)
    else:
        existing_episode.ended_at = ended_at
        existing_episode.consolidated_at = None  # re-dream on extension
        service.store.update_episode(existing_episode, embedding=None)

    # Write turns to raw store and insert Turn rows.
    # Turn ids are stable ChatGPT UUIDs so re-importing is idempotent as long
    # as the storage layer uses INSERT OR IGNORE (SqliteStore does).
    turns_written = 0
    with service.raw.open_for_streaming_writes(
        episode_id=convo_id, started_at=started_at
    ) as writer:
        for turn in turns:
            payload = {"id": turn["id"], "role": turn["role"], "ts": turn["ts"], "text": turn["text"]}
            appended = writer.append(payload)
            service.store.insert_turn(
                Turn(
                    id=turn["id"],
                    episode_id=convo_id,
                    raw_file=appended.raw_file,
                    byte_offset=appended.byte_offset,
                    byte_length=appended.byte_length,
                    role=turn["role"],
                    ts=turn["ts"],
                )
            )
            turns_written += 1

    # Update import state cursor.
    service.store.upsert_cowork_import_state(
        CoworkImportState(
            session_id=convo_id,
            last_turn_id=current_node,
            last_byte_offset=0,  # not used for ChatGPT; current_node is the watermark
            last_imported_at=now,
        )
    )

    return turns_written


def _extract_linear_thread(
    mapping: dict,
    current_node: str,
    *,
    include_tools: bool,
) -> list[dict]:
    """Walk current_node → parent → ... to get the canonical message chain.

    Returns turns in chronological order (root first).
    """
    chain: list[dict] = []
    node_id = current_node

    while node_id:
        node = mapping.get(node_id)
        if node is None:
            break
        msg = node.get("message")
        if msg:
            role = (msg.get("author") or {}).get("role", "")
            if role in _KEEP_ROLES or (include_tools and role == "tool"):
                text = _flatten_content(msg.get("content"), role=role)
                if text:
                    ts = _parse_ts(msg.get("create_time")) or now_iso()
                    turn_id = msg.get("id") or new_id()
                    chain.append({"id": turn_id, "role": role, "ts": ts, "text": text})
        node_id = node.get("parent") or ""

    chain.reverse()  # root → leaf
    return chain


def _flatten_content(content, *, role: str = "") -> str:
    """Extract plain text from a ChatGPT content block.

    content can be:
      {"content_type": "text", "parts": ["string", ...]}
      {"content_type": "code", "text": "..."}
      {"content_type": "tether_browsing_display", ...}  ← skip
      {"content_type": "multimodal_text", "parts": [...]}
      None
    """
    if not content or not isinstance(content, dict):
        return ""
    ct = content.get("content_type", "text")

    if ct == "text":
        parts = content.get("parts") or []
        texts = [p for p in parts if isinstance(p, str)]
        return "\n\n".join(t for t in texts if t.strip())

    if ct == "code":
        return content.get("text", "").strip()

    if ct == "multimodal_text":
        parts = content.get("parts") or []
        texts = [p for p in parts if isinstance(p, str)]
        return "\n\n".join(t for t in texts if t.strip())

    # tether_browsing_display, tether_quote, etc. — skip
    return ""


def _parse_ts(value) -> str | None:
    if value is None:
        return None
    try:
        return unix_to_iso(float(value))
    except (TypeError, ValueError):
        return None
