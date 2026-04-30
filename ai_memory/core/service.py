"""
MemoryService — orchestration layer.

Wires storage + embeddings + LLM + raw files into a single facade that
the MCP transport (and CLI, and tests) talks to. Pure business logic.

Construction:
    service = MemoryService.build(config)
    service.start()
    ...
    service.stop()

The `build` factory wires a sensible default stack from a Config object;
tests pass alternative components directly to `__init__` for substitution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from uuid import uuid4

from ai_memory.config import Config, ensure_home_layout
from ai_memory.core import dreaming as _dreaming
from ai_memory.core import ingest as _ingest
from ai_memory.core import recall as _recall
from ai_memory.core.models import Note, Profile, RecallHit
from ai_memory.embeddings.interface import Embedder
from ai_memory.embeddings.openai_embedder import OpenAIEmbedder
from ai_memory.llm.anthropic_llm import AnthropicLlm
from ai_memory.llm.interface import Llm
from ai_memory.storage.interface import MemoryStore
from ai_memory.storage.raw_files import RawTranscriptStore
from ai_memory.storage.sqlite_store import SqliteStore
from ai_memory.timestamps import now_iso

logger = logging.getLogger(__name__)


@dataclass
class MemoryService:
    """Top-level orchestrator. Pure functions delegated to ``core/`` modules."""

    config: Config
    store: MemoryStore
    raw: RawTranscriptStore
    embedder: Embedder
    llm: Llm

    # --- Lifecycle -------------------------------------------------------

    @classmethod
    def build(cls, config: Config) -> "MemoryService":
        """Wire a default stack from config: SQLite + OpenAI embeddings + Anthropic LLM."""
        ensure_home_layout(config)

        embedder: Embedder
        if config.embeddings.provider == "openai":
            embedder = OpenAIEmbedder(
                model=config.embeddings.model, dimensions=config.embeddings.dimensions
            )
        else:
            raise NotImplementedError(
                f"Embedding provider '{config.embeddings.provider}' not yet supported in v0.1"
            )

        llm: Llm
        if config.llm.provider == "anthropic":
            llm = AnthropicLlm(model=config.llm.model)
        elif config.llm.provider == "openai":
            from ai_memory.llm.openai_llm import OpenAILlm
            llm = OpenAILlm(model=config.llm.model)
        else:
            raise NotImplementedError(
                f"LLM provider '{config.llm.provider}' not yet supported in v0.1"
            )

        store = SqliteStore(config.database_path, embedding_dim=embedder.dimensions)
        raw = RawTranscriptStore(config.raw_dir)
        return cls(config=config, store=store, raw=raw, embedder=embedder, llm=llm)

    def start(self) -> None:
        self.store.initialise()
        logger.info(
            "ai-memory started: home=%s env=%s embedder=%s llm=%s",
            self.config.home,
            self.config.environment,
            self.embedder.model_id,
            self.llm.model_id,
        )

    def stop(self) -> None:
        self.store.close()

    # --- Public API ------------------------------------------------------

    def remember(
        self, *, text: str, source: str, role: str = "user", episode_id: str | None = None
    ) -> _ingest.RememberResult:
        """Persist a turn. Triggered by `memory_remember` MCP tool."""
        return _ingest.remember(
            store=self.store,
            raw=self.raw,
            text=text,
            source=source,
            role=role,
            episode_id=episode_id,
        )

    def recall(self, query: str, depth: str = "fast", k: int = 8) -> list[RecallHit]:
        """Hybrid search across the active memory tiers."""
        request = _recall.RecallRequest(query=query, depth=depth, k=k)
        return _recall.recall(
            store=self.store,
            raw=self.raw,
            embedder=self.embedder,
            config=self.config.recall,
            request=request,
        )

    def dream(
        self, *, trigger: str = "manual", since: str | None = None
    ) -> _dreaming.DreamReport:
        """Run a consolidation pass."""
        return _dreaming.dream(
            store=self.store,
            raw=self.raw,
            embedder=self.embedder,
            llm=self.llm,
            config=self.config.dream,
            request=_dreaming.DreamRequest(trigger=trigger, since=since),
        )

    def upsert_profile(self, key: str, value: str, source: str | None = None) -> None:
        """Insert / update a Tier 0 profile fact and re-mirror to profile.md."""
        self.store.upsert_profile(
            Profile(key=key, value=value, updated_at=now_iso(), source=source)
        )
        self._regenerate_profile_md()

    def list_profile(self) -> list[Profile]:
        return self.store.list_profile()

    def list_recent_dream_logs(self, limit: int = 10):
        return self.store.list_recent_dream_logs(limit)

    def list_recent_episodes(self, limit: int = 10):
        return self.store.list_recent_episodes(limit)

    def add_note(self, *, text: str, tags: list[str] | None = None) -> str:
        """Insert a Tier 1 note directly (used by bootstrap; dreaming uses this too)."""
        now = now_iso()
        [embedding] = self.embedder.embed([text])
        note = Note(
            id=str(uuid4()),
            text=text,
            tags=tags or [],
            valid_from=now,
            ingested_at=now,
            embedding_model=self.embedder.model_id,
        )
        self.store.insert_note(note, embedding)
        return note.id

    # --- Helpers ---------------------------------------------------------

    def _regenerate_profile_md(self) -> None:
        """Rewrite profile.md as a human-readable mirror of the profile table."""
        rows = sorted(self.store.list_profile(), key=lambda p: p.key)
        lines = ["# Profile (auto-generated)\n"]
        for row in rows:
            lines.append(f"- **{row.key}**: {row.value}")
        self.config.profile_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
