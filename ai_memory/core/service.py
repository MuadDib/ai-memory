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

import contextlib
import logging
from dataclasses import dataclass

from ai_memory.ids import new_id

from ai_memory.config import Config, ensure_home_layout
from ai_memory.core import dreaming as _dreaming
from ai_memory.core import ingest as _ingest
from ai_memory.core import recall as _recall
from ai_memory.core.models import Note, Profile, RecallHit
from ai_memory.embeddings.interface import Embedder
from ai_memory.embeddings.openai_embedder import OpenAIEmbedder
from ai_memory.llm.anthropic_llm import AnthropicLlm
from ai_memory.llm.interface import Llm
from ai_memory.process_lock import PidLock
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
    quality_llm: "Llm | None" = None

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

        # Rate-limit retry/backoff knobs shared by both providers.
        retry_kwargs = {
            "max_retries": config.dream.rate_limit_max_retries,
            "backoff_seconds": config.dream.rate_limit_backoff_seconds,
        }

        llm: Llm
        if config.llm.provider == "anthropic":
            llm = AnthropicLlm(model=config.llm.model, **retry_kwargs)
        elif config.llm.provider == "openai":
            from ai_memory.llm.openai_llm import OpenAILlm
            llm = OpenAILlm(model=config.llm.model, **retry_kwargs)
        else:
            raise NotImplementedError(
                f"LLM provider '{config.llm.provider}' not yet supported in v0.1"
            )

        # Optional quality-model LLM for long episodes (Phase 3 extract/summary).
        quality_llm: Llm | None = None
        if config.llm.quality_model:
            if config.llm.provider == "anthropic":
                quality_llm = AnthropicLlm(model=config.llm.quality_model, **retry_kwargs)
            elif config.llm.provider == "openai":
                from ai_memory.llm.openai_llm import OpenAILlm
                quality_llm = OpenAILlm(model=config.llm.quality_model, **retry_kwargs)

        store = SqliteStore(config.database_path, embedding_dim=embedder.dimensions)
        raw = RawTranscriptStore(config.raw_dir)
        return cls(
            config=config, store=store, raw=raw, embedder=embedder,
            llm=llm, quality_llm=quality_llm,
        )

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
        self, *, trigger: str = "manual", since: str | None = None,
        max_episodes: int | None = None,
    ) -> _dreaming.DreamReport:
        """Run a consolidation pass.

        Single-flight: every caller (daemon, CLI, MCP, PreCompact hook) goes
        through here, so a process-wide pidfile lock guarantees at most one pass
        at a time. `dream()` selects all pending episodes up front with no
        per-row claim, so two concurrent passes would double-process the same
        episodes — wasted LLM spend (painful on a throttled account) and
        near-duplicate notes. If a pass is already running we skip cleanly
        rather than pile on. See ADR-0013 / process_lock.py.
        """
        lock_path = self.config.home / "dream.pid"
        # No exe-name check: the real holder is base python (spawned by the
        # ai-memory.exe / venv launcher), whose path lacks "ai-memory", so a
        # name check would defeat the lock. See process_lock.py.
        lock = PidLock(lock_path, expected_name=None)
        if not lock.acquire():
            holder = ""
            with contextlib.suppress(OSError):
                holder = lock_path.read_text(encoding="utf-8").strip()
            msg = (
                "Dream pass skipped: another consolidation pass is already "
                f"running{f' (pid {holder})' if holder else ''}. "
                "Single-flight lock prevents concurrent passes from "
                "double-processing the same episodes."
            )
            logger.info(msg)
            return _dreaming.DreamReport(
                log_id="skipped",
                episodes_processed=0,
                notes_added=0,
                notes_invalidated=0,
                notes_promoted_to_profile=0,
                notes_pruned=0,
                journal=msg,
                episodes_failed=0,
            )
        try:
            return _dreaming.dream(
                store=self.store,
                raw=self.raw,
                embedder=self.embedder,
                llm=self.llm,
                quality_llm=self.quality_llm,
                config=self.config.dream,
                request=_dreaming.DreamRequest(
                    trigger=trigger, since=since, max_episodes=max_episodes
                ),
            )
        finally:
            lock.release()

    def upsert_profile(self, key: str, value: str, source: str | None = None) -> None:
        """Insert / update a profile fact and re-mirror to profile.md."""
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
        """Insert a note directly (used by bootstrap; dreaming uses this too)."""
        now = now_iso()
        [embedding] = self.embedder.embed([text])
        note = Note(
            id=new_id(),
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
