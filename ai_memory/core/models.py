"""
Domain models: plain dataclasses with explicit fields.

Deliberately not using ORM models — these are the canonical shape of data
flowing through the system. Storage adapters translate to/from these. A
future C# port maps each one to a record/POCO class with the same fields.

All timestamps are ISO 8601 UTC strings (e.g. '2026-04-29T18:35:23Z').
Lexicographically sortable, human-readable, and unambiguous.
Use ai_memory.timestamps for conversion helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Roles inside a raw transcript turn.
Role = Literal["user", "assistant", "system"]

# Source identifiers — which AI client wrote a memory.
# Free-form string; common values listed for consistency.
KnownSource = Literal[
    "claude-desktop",
    "claude-code",
    "claude-cowork",
    "cursor",
    "cline",
    "continue",
    "zed",
    "chatgpt-export",
    "claude-export",
    "manual",
]


@dataclass
class Profile:
    """Profile — durable user/agent fact.

    Stored as a key/value row plus mirrored into `profile.md` on update.
    """

    key: str
    value: str
    updated_at: str  # ISO 8601 UTC e.g. '2026-04-29T18:35:23Z'
    source: str | None = None


@dataclass
class Note:
    """Note — atomic semantic fact.

    Bi-temporal: `valid_from` / `valid_to` track when this fact held.
    `superseded_by` points at the note that replaced this one (if any).
    `contradicts` lists notes this one is in conflict with (resolved by
    invalidating the loser).
    """

    id: str
    text: str
    tags: list[str] = field(default_factory=list)
    source_episode_ids: list[str] = field(default_factory=list)
    valid_from: str = ""   # ISO 8601 UTC
    valid_to: str | None = None
    ingested_at: str = ""  # ISO 8601 UTC
    superseded_by: str | None = None
    contradicts: list[str] = field(default_factory=list)
    promoted_to_profile: bool = False
    embedding_model: str = ""
    access_count: int = 0
    last_accessed_at: str | None = None  # ISO 8601 UTC


@dataclass
class Episode:
    """Episode — per-session summary."""

    id: str
    title: str
    summary: str
    source: str  # which AI client; see KnownSource
    started_at: str   # ISO 8601 UTC
    ended_at: str | None = None
    raw_file: str = ""  # relative path within the raw/ tree
    embedding_model: str = ""
    consolidated_at: str | None = None  # None = not yet dreamed


@dataclass
class Turn:
    """Turn — verbatim raw turn, indexed by byte offset into a JSONL file."""

    id: str
    episode_id: str
    raw_file: str  # relative path within the raw/ tree
    byte_offset: int
    byte_length: int
    role: Role
    ts: str  # ISO 8601 UTC


@dataclass
class DreamLog:
    """Record of a single dream-cycle pass.

    Append-only. Free-text `journal` makes the system's behaviour
    introspectable — you can read what your memory thought about.
    """

    id: str
    started_at: str   # ISO 8601 UTC
    ended_at: str | None = None
    trigger: Literal["scheduled", "idle", "pressure", "manual"] = "manual"
    episodes_processed: int = 0
    notes_added: int = 0
    notes_invalidated: int = 0
    notes_promoted_to_profile: int = 0
    notes_pruned: int = 0
    llm_tokens_used: int = 0
    llm_cost_usd: float = 0.0
    journal: str = ""


@dataclass
class CoworkImportState:
    """Cursor for incremental Cowork / Claude Code session imports.

    The importer uses these to resume mid-session without re-ingesting turns
    we've already seen. ``last_byte_offset`` is a fast-path hint pointing at
    the position *after* the line that contained ``last_turn_id``; if the
    file has been rewritten and the offset no longer lines up with that id,
    the importer falls back to linear scan from the start.
    """

    session_id: str
    last_turn_id: str
    last_byte_offset: int
    last_imported_at: str  # ISO 8601 UTC


@dataclass
class RecallHit:
    """A single result from the recall pipeline.

    `score` is the fused score after RRF + recency boost. Higher = better.
    `provenance` carries the source episode id(s) so callers can trace
    a fact back to the conversation that produced it.
    """

    item_type: Literal["profile", "note", "episode", "turn"]
    id: str
    text: str
    score: float
    provenance: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """One eval-case result, persisted to the eval_results table for regression tracking."""

    id: str              # UUID4 per row
    run_id: str          # UUID4 shared across all cases in one CLI invocation
    suite: str           # suite name from YAML
    case_id: str         # case id from YAML
    query: str
    expected: str
    k: int
    depth: str
    passed: bool
    hits_count: int
    top_hit_text: str | None   # highest-scored hit text (for debugging failures)
    latency_ms: int
    run_at: str          # ISO 8601 UTC
