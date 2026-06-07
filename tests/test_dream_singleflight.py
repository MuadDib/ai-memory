"""service.dream() is single-flight (ADR-0013).

Every entry point (daemon, CLI `ai-memory dream`, MCP `memory_dream`, and the
PreCompact hook's `dream --trigger idle`) routes through `service.dream()`, so
the pidfile lock there is what actually prevents two passes from double-
processing the same episodes. These tests drive the real service with fakes.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ai_memory.config import Config
from ai_memory.core.ingest import remember
from ai_memory.core.service import MemoryService
from ai_memory.llm.interface import CompletionResult
from ai_memory.storage.raw_files import RawTranscriptStore
from ai_memory.storage.sqlite_store import SqliteStore


class _FakeEmbedder:
    model_id = "fake:embed"
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _CountingLlm:
    """Content-aware fake that records how many completions it served, so a
    skipped pass is distinguishable from a real one (real pass calls the LLM)."""

    model_id = "fake:llm"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, messages: list, max_tokens: int = 4096) -> CompletionResult:
        self.calls += 1
        user = messages[-1]["content"]
        if "Extract atomic facts" in user:
            text = '[{"text": "Igor prefers PostgreSQL", "tags": ["technical"], "entities": []}]'
        elif "short title" in system:
            text = "A Short Title"
        else:
            text = "A third-person summary of the conversation."
        return CompletionResult(
            text=text, input_tokens=5, output_tokens=5, model_id="fake:llm", finish_reason="stop"
        )


@pytest.fixture()
def service(tmp_path: Path):
    config = Config(home=tmp_path)
    store = SqliteStore(config.database_path, embedding_dim=4)
    store.initialise()
    raw = RawTranscriptStore(config.raw_dir)
    svc = MemoryService(
        config=config, store=store, raw=raw,
        embedder=_FakeEmbedder(), llm=_CountingLlm(), quality_llm=None,
    )
    # One pending episode so a non-skipped pass has real work to do.
    remember(store=store, raw=raw, text="I work with Postgres at Citywire", source="good")
    yield svc
    store.close()


def test_dream_skips_when_lock_already_held(service) -> None:
    lock_path = service.config.home / "dream.pid"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")  # a live holder

    report = service.dream(trigger="idle")

    assert report.log_id == "skipped"
    assert report.episodes_processed == 0
    assert "already" in report.journal.lower()
    assert service.llm.calls == 0  # no LLM work performed
    # The episode is left pending for whoever holds the lock to consolidate.
    pending = [
        e for e in service.store.episodes_since("1970-01-01T00:00:00Z")
        if e.consolidated_at is None
    ]
    assert len(pending) == 1


def test_dream_runs_and_releases_lock_when_free(service) -> None:
    report = service.dream(trigger="manual")

    assert report.log_id != "skipped"
    assert report.episodes_processed == 1
    assert service.llm.calls > 0
    # Lock file is cleaned up after the pass so the next one can run.
    assert not (service.config.home / "dream.pid").exists()

    # A subsequent pass can acquire the (now-free) lock and finds nothing pending.
    second = service.dream(trigger="manual")
    assert second.log_id != "skipped"
    assert second.episodes_processed == 0
