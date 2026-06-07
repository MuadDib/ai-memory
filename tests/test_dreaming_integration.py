"""End-to-end tests for the dream pass — the per-episode isolation and
journal-on-complete behaviour from ADR-0013.

These wire a real SqliteStore + RawTranscriptStore (temp dir) with a fake
embedder and a content-aware fake LLM, so we can drive `dreaming.dream()`
through Phases 3-7 without the network and force one episode to fail.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_memory.config import DreamConfig
from ai_memory.core import dreaming
from ai_memory.core.dreaming import DreamRequest
from ai_memory.core.ingest import remember
from ai_memory.llm.interface import CompletionResult
from ai_memory.storage.raw_files import RawTranscriptStore
from ai_memory.storage.sqlite_store import SqliteStore

_POISON = "POISON_PILL_MARKER"


def _cr(text: str) -> CompletionResult:
    return CompletionResult(
        text=text, input_tokens=5, output_tokens=5, model_id="fake:llm", finish_reason="stop"
    )


class _FakeEmbedder:
    model_id = "fake:embed"
    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class _FakeLlm:
    """Returns plausible output per call; raises when a turn carries the poison
    marker, simulating a provider error for exactly one episode."""

    model_id = "fake:llm"

    def __init__(self, fail_marker: str | None = None) -> None:
        self._fail_marker = fail_marker
        self.calls = 0

    def complete(self, system: str, messages: list, max_tokens: int = 4096) -> CompletionResult:
        self.calls += 1
        user = messages[-1]["content"]
        if self._fail_marker and self._fail_marker in user:
            raise RuntimeError("simulated provider failure")
        if "Extract atomic facts" in user:
            return _cr('[{"text": "Igor prefers PostgreSQL for relational work", '
                       '"tags": ["technical"], "entities": []}]')
        if "short title" in system:
            return _cr("A Short Title")
        return _cr("A third-person summary of what the participants discussed.")


@pytest.fixture()
def wiring(tmp_path: Path):
    store = SqliteStore(tmp_path / "memory.db", embedding_dim=4)
    store.initialise()
    raw = RawTranscriptStore(tmp_path / "raw")
    embedder = _FakeEmbedder()
    yield store, raw, embedder
    store.close()


def test_failed_episode_is_isolated_and_pass_completes(wiring) -> None:
    store, raw, embedder = wiring
    # Two separate episodes (distinct sources): one healthy, one poison-pilled.
    remember(store=store, raw=raw, text="I work at Citywire and like Postgres", source="good")
    remember(store=store, raw=raw, text=f"this turn {_POISON} will blow up", source="poison")

    llm = _FakeLlm(fail_marker=_POISON)
    cfg = DreamConfig(max_request_tokens=100000, long_episode_turns=0)

    report = dreaming.dream(
        store=store, raw=raw, embedder=embedder, llm=llm, quality_llm=None,
        config=cfg, request=DreamRequest(trigger="manual"),
    )

    # The healthy episode consolidated; the poison one failed but did NOT abort
    # the pass or block the healthy one.
    assert report.episodes_processed == 1
    assert report.episodes_failed == 1
    assert report.notes_added == 1

    episodes = {e.source: e for e in store.episodes_since("1970-01-01T00:00:00Z")}
    assert episodes["good"].consolidated_at is not None
    assert episodes["poison"].consolidated_at is None  # left pending for retry


def test_journal_on_complete_writes_exactly_one_finalized_row(wiring) -> None:
    store, raw, embedder = wiring
    remember(store=store, raw=raw, text="Igor uses .NET and AWS", source="good")
    remember(store=store, raw=raw, text=f"boom {_POISON}", source="poison")

    llm = _FakeLlm(fail_marker=_POISON)
    cfg = DreamConfig(max_request_tokens=100000, long_episode_turns=0)
    dreaming.dream(
        store=store, raw=raw, embedder=embedder, llm=llm, quality_llm=None,
        config=cfg, request=DreamRequest(trigger="manual"),
    )

    logs = store.list_recent_dream_logs(limit=10)
    # Exactly one row, fully finalized (ended_at set) — no orphans.
    assert len(logs) == 1
    assert logs[0].ended_at
    assert "EPISODE_FAILURE" in logs[0].journal


def test_clean_pass_has_no_failures(wiring) -> None:
    store, raw, embedder = wiring
    remember(store=store, raw=raw, text="Igor lives in London", source="good")

    llm = _FakeLlm(fail_marker=None)
    cfg = DreamConfig(max_request_tokens=100000, long_episode_turns=0)
    report = dreaming.dream(
        store=store, raw=raw, embedder=embedder, llm=llm, quality_llm=None,
        config=cfg, request=DreamRequest(trigger="manual"),
    )

    assert report.episodes_processed == 1
    assert report.episodes_failed == 0
    logs = store.list_recent_dream_logs(limit=10)
    assert len(logs) == 1
    assert logs[0].ended_at


# --- Phase 5 promotion hardening (ADR-0013 / Tier-0 poisoning fix) -----------

from ai_memory.core.dreaming import _phase5_promote  # noqa: E402
from ai_memory.core.models import Note  # noqa: E402
from ai_memory.ids import new_id  # noqa: E402
from ai_memory.timestamps import now_iso  # noqa: E402


class _PromoLlm:
    """Fake LLM for Phase 5: returns a fixed JSON for the promotion call and
    records how many times it was invoked."""

    def __init__(self, response: str) -> None:
        self.model_id = "fake:promo"
        self._response = response
        self.calls = 0

    def complete(self, system: str, messages: list, max_tokens: int = 4096) -> CompletionResult:
        self.calls += 1
        return CompletionResult(
            text=self._response, input_tokens=5, output_tokens=5, model_id="fake:promo"
        )


def _insert_cluster(store, embedder, texts: list[str], episode_ids: list[list[str]]) -> None:
    now = now_iso()
    for text, eps in zip(texts, episode_ids):
        note = Note(
            id=new_id(), text=text, tags=["preference"], entities=[],
            source_episode_ids=eps, valid_from=now, ingested_at=now,
            embedding_model=embedder.model_id,
        )
        store.insert_note(note, embedder.embed([text])[0])


def test_phase5_promotes_durable_fact_via_quality_model(wiring) -> None:
    store, _raw, embedder = wiring
    _insert_cluster(
        store, embedder,
        ["Igor prefers PostgreSQL", "Igor likes Postgres for relational work",
         "Igor's database of choice is PostgreSQL"],
        [["epA"], ["epB"], ["epA"]],
    )
    base = _PromoLlm("SHOULD-NOT-BE-USED")
    quality = _PromoLlm(
        '{"key": "preferred_database", "value": "PostgreSQL", "rationale": "endorsed across episodes"}'
    )
    promoted, _tokens = _phase5_promote(
        store=store, llm=base, quality_llm=quality, config=DreamConfig(),
        now=now_iso(), journal=[],
    )
    assert promoted == 1
    assert quality.calls == 1   # promotion verdict routed to the quality model
    assert base.calls == 0      # base model NOT used for the high-stakes verdict
    profile = {p.key: p.value for p in store.list_profile()}
    assert profile.get("preferred_database") == "PostgreSQL"


def test_phase5_declines_external_entity_cluster(wiring) -> None:
    store, _raw, embedder = wiring
    _insert_cluster(
        store, embedder,
        ["British Gas runs a Win Your Bill prize draw",
         "British Gas Trading Limited company number 03078711",
         "The British Gas promotion is open to UK residents aged 18 plus"],
        [["epA"], ["epB"], ["epA"]],
    )
    quality = _PromoLlm('{"key": null, "value": null, "rationale": "about an external company"}')
    promoted, _tokens = _phase5_promote(
        store=store, llm=quality, quality_llm=quality, config=DreamConfig(),
        now=now_iso(), journal=[],
    )
    assert promoted == 0
    profile = {p.key: p.value for p in store.list_profile()}
    assert "current_company" not in profile


def test_dream_respects_max_episodes_cap(wiring) -> None:
    store, raw, embedder = wiring
    # Three separate episodes (distinct sources).
    for i in range(3):
        remember(store=store, raw=raw, text=f"fact number {i} about Igor", source=f"src{i}")

    llm = _FakeLlm(fail_marker=None)
    cfg = DreamConfig(max_request_tokens=100000, long_episode_turns=0)
    report = dreaming.dream(
        store=store, raw=raw, embedder=embedder, llm=llm, quality_llm=None,
        config=cfg, request=DreamRequest(trigger="manual", max_episodes=1),
    )

    # Only one episode consolidated; the other two remain pending for the next batch.
    assert report.episodes_processed == 1
    pending = [
        e for e in store.episodes_since("1970-01-01T00:00:00Z") if e.consolidated_at is None
    ]
    assert len(pending) == 2
