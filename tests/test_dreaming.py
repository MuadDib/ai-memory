"""Pure-function tests for the dreaming module.

These exercise the bits that don't need the real LLM, store, or embedder —
clustering math, JSON parsing tolerance, key sanitisation. The integration
phases themselves are covered by end-to-end tests once they exist.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai_memory.core.dreaming import (
    _CandidateFact,
    _ExtractedFact,
    _IntegrateVerdict,
    _PromotionResult,
    _cluster_notes,
    _euclid_distance,
    _chunk_turns,
    _estimate_tokens,
    _llm_call_with_retry,
    _parse_extract_facts,
    _parse_promotion,
    _confirm_contradiction,
    _new_supersedes_cleanly,
    _note_conviction,
    _parse_verdicts,
    _phase4_integrate,
    _phase4b_resolve_contradictions,
    _safe_parse_facts,
    _safe_parse_json,
    _sanitise_profile_key,
    _split_transcript_by_budget,
    _summarise_episode,
    DUPLICATE_DIST_BELOW,
    INTEGRATE_VERDICT_SYSTEM,
    UNRELATED_DIST_ABOVE,
)
from ai_memory.config import DreamConfig
from ai_memory.core.models import Note
from ai_memory.llm.interface import CompletionResult, Message
from ai_memory.timestamps import now_iso, iso_to_dt


def _mock_llm(*responses: str) -> MagicMock:
    """Return a mock Llm that cycles through the given response texts."""
    llm = MagicMock()
    llm.model_id = "mock"
    llm.complete.side_effect = [
        CompletionResult(
            text=r, input_tokens=10, output_tokens=5, model_id="mock", finish_reason="stop"
        )
        for r in responses
    ]
    return llm


# --- Distance --------------------------------------------------------------


def test_euclid_distance_basic() -> None:
    assert _euclid_distance([0.0, 0.0], [3.0, 4.0]) == 5.0
    assert _euclid_distance([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_euclid_distance_mismatched_dims() -> None:
    assert _euclid_distance([1.0], [1.0, 2.0]) == float("inf")


# --- Clustering ------------------------------------------------------------


def _note(id_: str, episode: str = "e1") -> Note:
    return Note(
        id=id_,
        text=f"text-{id_}",
        valid_from=now_iso(),
        ingested_at=now_iso(),
        embedding_model="test",
        source_episode_ids=[episode],
    )


def test_cluster_notes_groups_close_vectors() -> None:
    notes = [_note("a"), _note("b"), _note("c"), _note("d")]
    embeddings = [
        [1.0, 0.0],
        [1.001, 0.0],   # very close to a
        [10.0, 10.0],   # far away
        [10.001, 10.0], # close to c
    ]
    clusters = _cluster_notes(notes, embeddings, dist_threshold=0.1)
    cluster_ids = sorted(sorted(n.id for n in cluster) for cluster in clusters)
    assert cluster_ids == [["a", "b"], ["c", "d"]]


def test_cluster_notes_drops_singletons() -> None:
    notes = [_note("a"), _note("b")]
    embeddings = [[0.0, 0.0], [10.0, 10.0]]  # far apart
    assert _cluster_notes(notes, embeddings, dist_threshold=0.1) == []


def test_cluster_notes_handles_missing_embeddings() -> None:
    notes = [_note("a"), _note("b"), _note("c")]
    embeddings: list[list[float] | None] = [None, [0.0, 0.0], [0.0, 0.0]]
    clusters = _cluster_notes(notes, embeddings, dist_threshold=0.1)
    # Only b and c should pair up; a is excluded because it has no embedding.
    assert len(clusters) == 1
    assert sorted(n.id for n in clusters[0]) == ["b", "c"]


# --- Profile-key sanitisation ---------------------------------------------


def test_sanitise_key_lowercases_and_underscores() -> None:
    assert _sanitise_profile_key("Preferred Database") == "preferred_database"
    assert _sanitise_profile_key("Comm Style!") == "comm_style"
    assert _sanitise_profile_key("  --  ") == "fact"  # nothing useful left


def test_sanitise_key_collapses_repeated_separators() -> None:
    assert _sanitise_profile_key("foo___bar  baz") == "foo_bar_baz"


# --- JSON parsing ----------------------------------------------------------


def test_safe_parse_json_plain() -> None:
    assert _safe_parse_json('{"a": 1}', default={}) == {"a": 1}


def test_safe_parse_json_strips_markdown_fence() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert _safe_parse_json(raw, default={}) == {"a": 1}


def test_safe_parse_json_extracts_embedded_object() -> None:
    raw = 'Here is the JSON: {"key": "preferred_db", "value": "Postgres"} all done.'
    parsed = _safe_parse_json(raw, default={})
    assert parsed == {"key": "preferred_db", "value": "Postgres"}


def test_safe_parse_json_default_on_garbage() -> None:
    assert _safe_parse_json("not json at all", default={"fallback": True}) == {"fallback": True}


def test_safe_parse_facts_filters_non_dicts() -> None:
    raw = '[{"text": "fact 1", "tags": ["a"]}, "garbage", null, {"text": "fact 2"}]'
    facts = _safe_parse_facts(raw)
    assert len(facts) == 2
    assert facts[0]["text"] == "fact 1"
    assert facts[1]["text"] == "fact 2"


def test_safe_parse_facts_handles_bare_comma_separated_objects() -> None:
    """gpt-4o-mini regularly drops the outer [] of an array but keeps the items.

    Real example from a dream pass on a 260-turn cowork episode — the
    extract step returned `{"text":"..."},{"text":"..."}` with no array
    wrapper, which json.loads can't parse directly. The parser must wrap
    and retry rather than silently emitting zero candidates.
    """
    raw = '{"text": "fact one", "tags": ["a"]},{"text": "fact two", "tags": ["b"]}'
    facts = _safe_parse_facts(raw)
    assert len(facts) == 2
    assert facts[0]["text"] == "fact one"
    assert facts[1]["text"] == "fact two"
    assert facts[0]["tags"] == ["a"]


def test_safe_parse_facts_handles_single_object() -> None:
    """When the extractor finds only one fact some models return a bare
    object instead of a one-element array. Treat as a one-element list."""
    raw = '{"text": "the only fact", "tags": ["solo"]}'
    facts = _safe_parse_facts(raw)
    assert len(facts) == 1
    assert facts[0]["text"] == "the only fact"


def test_safe_parse_facts_handles_trailing_comma_on_bare_objects() -> None:
    """Some models leave a trailing comma after the last bare object."""
    raw = '{"text": "a", "tags": []},{"text": "b", "tags": []},'
    facts = _safe_parse_facts(raw)
    assert len(facts) == 2
    assert facts[0]["text"] == "a"
    assert facts[1]["text"] == "b"


def test_safe_parse_facts_empty_on_garbage() -> None:
    assert _safe_parse_facts("definitely not json") == []
    assert _safe_parse_facts("") == []


# --- Chunking -----------------------------------------------------------


def test_chunk_turns_short_episode_single_chunk() -> None:
    """Episodes shorter than chunk_size should not be chunked at all."""
    turns = list(range(20))
    chunks = _chunk_turns(turns, chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == turns


def test_chunk_turns_overlapping_windows() -> None:
    """Long episodes split into overlapping windows that cover all turns."""
    turns = list(range(260))
    chunks = _chunk_turns(turns, chunk_size=50, overlap=5)
    # Step = 45, so windows start at 0, 45, 90, 135, 180, 225 -> 6 chunks.
    assert len(chunks) == 6
    # Every turn appears in at least one chunk.
    seen = set()
    for chunk in chunks:
        seen.update(chunk)
    assert seen == set(turns)
    # Overlap exists at boundaries (chunk[0][-5:] should equal chunk[1][:5]).
    assert chunks[0][-5:] == chunks[1][:5]
    # Last chunk reaches the end.
    assert chunks[-1][-1] == 259


def test_chunk_turns_empty_input() -> None:
    assert _chunk_turns([], chunk_size=50, overlap=5) == []


def test_chunk_turns_exact_multiple() -> None:
    """When len(turns) lands exactly on a chunk boundary, no orphan tail."""
    turns = list(range(50))
    chunks = _chunk_turns(turns, chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0] == turns


# --- Summary token budgeting (deadlock fix) ---------------------------------


def test_estimate_tokens_roughly_quarter_of_chars() -> None:
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("a" * 400) == 100


def test_split_transcript_small_returns_single_piece() -> None:
    transcript = "<turn n=1 role=user>hello</turn>"
    assert _split_transcript_by_budget(transcript, budget_chars=1000) == [transcript]


def test_split_transcript_breaks_at_turn_boundaries_under_budget() -> None:
    turns = "".join(f"<turn n={i} role=user>{'x' * 40}</turn>" for i in range(10))
    pieces = _split_transcript_by_budget(turns, budget_chars=120)
    # Every piece is under (or equal to) budget except an unavoidable single turn.
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= 120 or piece.count("<turn") == 1
    # Reassembly preserves every turn (no turn dropped or split mid-tag).
    assert "".join(pieces) == turns
    assert sum(p.count("<turn") for p in pieces) == 10


def test_summarise_episode_single_shot_when_small() -> None:
    """A small transcript = one LLM call, mode single-shot."""
    llm = _mock_llm("a concise summary")
    completion, tokens, mode = _summarise_episode(
        ep_llm=llm, transcript="<turn n=1 role=user>hi</turn>", max_request_tokens=25000
    )
    assert mode == "single-shot"
    assert completion.text == "a concise summary"
    assert tokens == 15  # 10 in + 5 out from _mock_llm
    assert llm.complete.call_count == 1


def test_summarise_episode_map_reduce_when_over_budget() -> None:
    """A transcript over budget is chunk-summarised then merged (map-reduce)."""
    transcript = "".join(f"<turn n={i} role=user>{'x' * 80}</turn>" for i in range(40))
    # Derive the real chunk count so the mock supplies exactly enough responses:
    # one per map call plus the final merge.
    chunks = _split_transcript_by_budget(transcript, 100 * 4)
    assert len(chunks) > 1  # sanity: this transcript really does need splitting
    llm = _mock_llm(*[f"part {i}" for i in range(len(chunks))], "merged summary")
    completion, tokens, mode = _summarise_episode(
        ep_llm=llm, transcript=transcript, max_request_tokens=100
    )
    assert mode == f"map-reduce({len(chunks)} chunks)"
    assert completion.text == "merged summary"
    # One call per chunk (map) + one merge (reduce); tokens summed across all.
    assert llm.complete.call_count == len(chunks) + 1
    assert tokens == 15 * (len(chunks) + 1)


# --- Pydantic parse functions -----------------------------------------------


def test_parse_extract_facts_valid() -> None:
    raw = '[{"text": "Igor uses PostgreSQL", "tags": ["technical"], "entities": ["postgresql"]}]'
    facts = _parse_extract_facts(raw)
    assert len(facts) == 1
    assert facts[0].text == "Igor uses PostgreSQL"
    assert facts[0].tags == ["technical"]
    assert facts[0].entities == ["postgresql"]


def test_parse_extract_facts_skips_empty_text() -> None:
    raw = '[{"text": "", "tags": []}, {"text": "valid fact", "tags": []}]'
    facts = _parse_extract_facts(raw)
    assert len(facts) == 1
    assert facts[0].text == "valid fact"


def test_parse_extract_facts_single_object() -> None:
    """LLM returns a bare object instead of an array — handled."""
    raw = '{"text": "only fact", "tags": ["preference"], "entities": []}'
    facts = _parse_extract_facts(raw)
    assert len(facts) == 1


def test_parse_extract_facts_raises_on_complete_garbage() -> None:
    with pytest.raises(ValueError):
        _parse_extract_facts("not json at all")


def test_parse_verdicts_valid() -> None:
    raw = '[{"existing_id": "abc", "verdict": "DUPLICATE", "reason": "same fact"}]'
    verdicts = _parse_verdicts(raw)
    assert len(verdicts) == 1
    assert verdicts[0].verdict == "DUPLICATE"
    assert verdicts[0].existing_id == "abc"


def test_parse_verdicts_invalid_enum_skipped() -> None:
    """Items with an invalid verdict value are skipped; valid ones are kept."""
    raw = (
        '[{"existing_id": "a", "verdict": "DUPLICATE", "reason": ""},'
        ' {"existing_id": "b", "verdict": "INVENTED_VERDICT", "reason": ""}]'
    )
    verdicts = _parse_verdicts(raw)
    assert len(verdicts) == 1
    assert verdicts[0].existing_id == "a"


def test_parse_verdicts_raises_when_all_invalid() -> None:
    raw = '[{"existing_id": "x", "verdict": "GARBAGE"}]'
    with pytest.raises(ValueError):
        _parse_verdicts(raw)


def test_parse_promotion_valid() -> None:
    raw = '{"key": "preferred_db", "value": "PostgreSQL", "rationale": "mentioned often"}'
    p = _parse_promotion(raw)
    assert p.key == "preferred_db"
    assert p.value == "PostgreSQL"


def test_parse_promotion_null_signals_decline() -> None:
    """LLM returning null means 'do not promote'."""
    p = _parse_promotion("null")
    assert p.key is None
    assert p.value is None


def test_parse_promotion_missing_fields_returns_empty() -> None:
    p = _parse_promotion("{}")
    assert p.key is None
    assert p.value is None


# --- Retry helper ------------------------------------------------------------


def test_llm_call_with_retry_success_first_attempt() -> None:
    llm = _mock_llm('[{"text": "a fact", "tags": [], "entities": []}]')
    result, tokens = _llm_call_with_retry(
        llm=llm,
        system="sys",
        user_msg="user",
        parse_fn=_parse_extract_facts,
        max_tokens=100,
        default=[],
    )
    assert len(result) == 1
    assert result[0].text == "a fact"
    assert llm.complete.call_count == 1
    assert tokens == 15  # 10 input + 5 output


def test_llm_call_with_retry_succeeds_on_second_attempt() -> None:
    """First response is invalid JSON; second is valid — retry succeeds."""
    llm = _mock_llm(
        "not json at all",
        '[{"text": "retried fact", "tags": [], "entities": []}]',
    )
    result, tokens = _llm_call_with_retry(
        llm=llm,
        system="sys",
        user_msg="user",
        parse_fn=_parse_extract_facts,
        max_tokens=100,
        default=[],
    )
    assert len(result) == 1
    assert result[0].text == "retried fact"
    assert llm.complete.call_count == 2
    assert tokens == 30  # two calls × 15 tokens each


def test_llm_call_with_retry_returns_default_after_two_failures() -> None:
    """Both attempts fail — returns the default without raising."""
    llm = _mock_llm("garbage", "still garbage")
    result, tokens = _llm_call_with_retry(
        llm=llm,
        system="sys",
        user_msg="user",
        parse_fn=_parse_extract_facts,
        max_tokens=100,
        default=[],
    )
    assert result == []
    assert llm.complete.call_count == 2


def test_llm_call_with_retry_passes_error_context_to_second_call() -> None:
    """Retry call must include the bad output + an error correction message."""
    llm = _mock_llm(
        "bad output",
        '[{"text": "fixed", "tags": [], "entities": []}]',
    )
    _llm_call_with_retry(
        llm=llm,
        system="sys",
        user_msg="original user msg",
        parse_fn=_parse_extract_facts,
        max_tokens=100,
        default=[],
    )
    _, retry_kwargs = llm.complete.call_args_list[1]
    retry_messages: list[Message] = retry_kwargs.get("messages") or llm.complete.call_args_list[1][0][1]
    # Retry should send a 3-turn conversation:
    # user (original), assistant (bad output), user (error correction)
    assert len(retry_messages) == 3
    assert retry_messages[0]["role"] == "user"
    assert retry_messages[1]["role"] == "assistant"
    assert retry_messages[1]["content"] == "bad output"
    assert retry_messages[2]["role"] == "user"
    assert "invalid" in retry_messages[2]["content"].lower()


# --- Verdict parsing edge cases ---------------------------------------------


def test_parse_verdicts_valid_all_four_types() -> None:
    """All four verdict values should parse cleanly."""
    for verdict in ("DUPLICATE", "CONTRADICTS", "COMPLEMENTS", "UNRELATED"):
        raw = f'[{{"existing_id": "x", "verdict": "{verdict}", "reason": "ok"}}]'
        result = _parse_verdicts(raw)
        assert len(result) == 1
        assert result[0].verdict == verdict


def test_parse_verdicts_missing_reason_field_is_ok() -> None:
    """reason is optional — should not cause a ValidationError."""
    raw = '[{"existing_id": "abc", "verdict": "DUPLICATE"}]'
    result = _parse_verdicts(raw)
    assert result[0].reason == ""


def test_parse_verdicts_case_sensitive_enum() -> None:
    """Lowercase verdict values are invalid and skipped."""
    raw = '[{"existing_id": "x", "verdict": "duplicate"}]'
    with pytest.raises(ValueError):
        _parse_verdicts(raw)


def test_parse_promotion_null_string_decoded_as_none() -> None:
    """JSON null literal → PromotionResult with key=None (decline signal)."""
    result = _parse_promotion("null")
    assert result.key is None
    assert result.value is None


# --- warnings param on _llm_call_with_retry ------------------------------------


def test_llm_call_with_retry_records_failure_in_warnings() -> None:
    """When both attempts fail, the error is recorded in the warnings list."""
    llm = _mock_llm("garbage", "still garbage")
    warnings: list[str] = []
    result, _ = _llm_call_with_retry(
        llm=llm,
        system="sys",
        user_msg="user",
        parse_fn=_parse_extract_facts,
        max_tokens=100,
        default=[],
        warnings=warnings,
    )
    assert result == []
    assert len(warnings) == 1
    assert "failed after retry" in warnings[0]


def test_llm_call_with_retry_no_warning_on_success() -> None:
    """Successful call must not add anything to the warnings list."""
    llm = _mock_llm('[{"text": "fact", "tags": [], "entities": []}]')
    warnings: list[str] = []
    result, _ = _llm_call_with_retry(
        llm=llm,
        system="sys",
        user_msg="user",
        parse_fn=_parse_extract_facts,
        max_tokens=100,
        default=[],
        warnings=warnings,
    )
    assert len(result) == 1
    assert warnings == []


# --- Phase 4 integration paths ------------------------------------------------


def _existing_note(
    id_: str = "existing-1", text: str = "Igor uses PostgreSQL", tags: list[str] | None = None
) -> Note:
    return Note(
        id=id_,
        text=text,
        tags=tags or [],
        valid_from=now_iso(),
        ingested_at=now_iso(),
        embedding_model="test",
    )


def _make_phase4_store(neighbours: list) -> "MagicMock":
    """Mock MemoryStore pre-wired with the given search_notes_vector return value."""
    from unittest.mock import MagicMock
    store = MagicMock()
    store.search_notes_vector.return_value = neighbours
    return store


def _make_phase4_embedder(n_candidates: int = 1) -> "MagicMock":
    from unittest.mock import MagicMock
    emb = MagicMock()
    emb.model_id = "test"
    emb.embed.return_value = [[0.1] * 4 for _ in range(n_candidates)]
    return emb


def test_phase4_clear_duplicate_no_llm() -> None:
    """Neighbour distance < DUPLICATE_DIST_BELOW → deduped without any LLM call."""
    existing = _existing_note()
    store = _make_phase4_store([(existing, DUPLICATE_DIST_BELOW - 0.01)])
    emb = _make_phase4_embedder()
    llm = _mock_llm()  # no responses needed — must not be called
    cand = _CandidateFact(text="Igor uses PostgreSQL", source_episode_id="ep1")

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    assert result["deduped"] == 1
    assert result["added"] == 0
    store.bump_note_access.assert_called_once()
    store.insert_note.assert_not_called()
    llm.complete.assert_not_called()


def test_phase4_clearly_unrelated_inserts_without_llm() -> None:
    """Nearest neighbour distance > UNRELATED_DIST_ABOVE → inserted without LLM."""
    existing = _existing_note()
    store = _make_phase4_store([(existing, UNRELATED_DIST_ABOVE + 0.01)])
    emb = _make_phase4_embedder()
    llm = _mock_llm()
    cand = _CandidateFact(text="A completely unrelated fact", source_episode_id="ep1")

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    assert result["added"] == 1
    store.insert_note.assert_called_once()
    llm.complete.assert_not_called()


def test_phase4_midband_contradicts_quarantines_keeps_both() -> None:
    """Mid-band + CONTRADICTS → new note inserted and linked to the existing, but
    BOTH kept valid (ADR-0014 quarantine: a single contradicting mention must not
    destroy a standing fact unattended)."""
    existing = _existing_note(text="Igor uses PostgreSQL as primary database")
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "CONTRADICTS", "reason": "db changed"}}]'
    )
    cand = _CandidateFact(
        text="Igor switched to MySQL as his primary database", source_episode_id="ep1"
    )

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    assert result["added"] == 1
    assert result["invalidated"] == 0          # ADR-0014: nothing destroyed
    assert result["quarantined"] == 1
    assert result["deduped"] == 0
    store.insert_note.assert_called_once()
    store.invalidate_note.assert_not_called()  # the standing fact survives
    # The new note records the conflict link back to the existing one.
    inserted_note = store.insert_note.call_args[0][0]
    assert existing.id in inserted_note.contradicts


def test_integrate_prompt_has_same_scope_guard() -> None:
    """ADR-0014 §1: the verdict prompt must instruct that a contradiction needs
    the SAME scope/timeframe/object — guards against silently reverting the rule
    that prevents the Q4-vs-FY / Pro-vs-non-Pro / general-vs-specific false class."""
    p = INTEGRATE_VERDICT_SYSTEM.lower()
    assert "same-scope" in p
    assert "timeframe" in p
    assert "general" in p and "specific" in p  # general-principle vs specific-instance


def test_phase4_contradicts_cross_model_confirmed_supersedes() -> None:
    """ADR-0014 §2: when the confirm model ALSO says CONTRADICTS, the existing note
    is superseded — destructive resolution is allowed only on cross-model agreement."""
    existing = _existing_note(text="The service runs on port 8080")
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "CONTRADICTS", "reason": "port changed"}}]'
    )
    confirm = _mock_llm(
        # 1st call: cross-model verdict (§2). 2nd call: coverage check (§3).
        f'[{{"existing_id": "{existing.id}", "verdict": "CONTRADICTS", "reason": "agree"}}]',
        '{"fully_superseded": true, "reason": "new restates old with the corrected value"}',
    )
    cand = _CandidateFact(text="The service runs on port 9000", source_episode_id="ep1")

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        confirm_llm=confirm, now=now_iso(), journal=[],
    )

    assert result["invalidated"] == 1
    assert result["quarantined"] == 0
    assert result["added"] == 1
    store.invalidate_note.assert_called_once()
    assert store.invalidate_note.call_args[0][0] == existing.id
    assert confirm.complete.call_count == 2  # §2 verdict + §3 coverage both consulted


def test_phase4_contradicts_cross_model_disagrees_quarantines() -> None:
    """ADR-0014 §2: when the confirm model does NOT agree (e.g. it sees the Q4-vs-FY
    scope mismatch), the pair is quarantined — both kept, nothing destroyed."""
    existing = _existing_note(text="AWS was 18% of Amazon revenue in 2025")
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "CONTRADICTS", "reason": "different %"}}]'
    )
    confirm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "COMPLEMENTS", "reason": "Q4 vs full year"}}]'
    )
    cand = _CandidateFact(
        text="AWS was 17% of Amazon net sales in Q4 2025", source_episode_id="ep1"
    )

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        confirm_llm=confirm, now=now_iso(), journal=[],
    )

    assert result["invalidated"] == 0
    assert result["quarantined"] == 1
    assert result["added"] == 1
    store.invalidate_note.assert_not_called()
    inserted_note = store.insert_note.call_args[0][0]
    assert existing.id in inserted_note.contradicts


def test_phase4_partial_information_quarantines_despite_cross_model_agreement() -> None:
    """ADR-0014 §3: even when both models agree it's a contradiction, if the new
    fact omits a still-true detail bundled in the old note, quarantine — don't
    destroy. (The WAL-mode + busy_timeout case.)"""
    existing = _existing_note(text="SQLite uses WAL mode and a 15s busy_timeout")
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "CONTRADICTS", "reason": "timeout changed"}}]'
    )
    confirm = _mock_llm(
        # §2 agrees it's a contradiction, but §3 coverage says the new fact drops info.
        f'[{{"existing_id": "{existing.id}", "verdict": "CONTRADICTS", "reason": "agree"}}]',
        '{"fully_superseded": false, "reason": "new drops the WAL-mode fact"}',
    )
    cand = _CandidateFact(text="busy_timeout set to 5000ms", source_episode_id="ep1")

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        confirm_llm=confirm, now=now_iso(), journal=[],
    )

    assert result["invalidated"] == 0       # §3 prevented the destroy
    assert result["quarantined"] == 1
    assert result["added"] == 1
    store.invalidate_note.assert_not_called()
    assert confirm.complete.call_count == 2  # verdict + coverage both ran
    inserted_note = store.insert_note.call_args[0][0]
    assert existing.id in inserted_note.contradicts


def test_new_supersedes_cleanly_failsafe_returns_false_on_error() -> None:
    """ADR-0014 §3 fail-safe: a coverage-check error must return False so a provider
    outage can never green-light a destructive supersede."""
    boom = MagicMock()
    boom.complete.side_effect = RuntimeError("provider down")
    ok, tok = _new_supersedes_cleanly(llm=boom, new_text="a", old_text="b")
    assert ok is False
    assert tok == 0


# --- ADR-0014 §5 conviction-gated resolution --------------------------------


def _conv_note(id_, text, *, episodes=1, access=0, contradicts=None) -> Note:
    return Note(
        id=id_, text=text, tags=[],
        source_episode_ids=[f"ep{i}" for i in range(episodes)],
        valid_from=now_iso(), ingested_at=now_iso(),
        access_count=access, contradicts=list(contradicts or []),
        embedding_model="test",
    )


class _ResolveStore:
    """Minimal store fake for the Phase 4b resolution pass."""

    def __init__(self, notes: list[Note]):
        self._notes = {n.id: n for n in notes}
        self.invalidated: list[tuple[str, str | None]] = []
        self.bumped: list[str] = []

    def list_valid_notes(self) -> list[Note]:
        return [n for n in self._notes.values() if n.valid_to is None]

    def get_note(self, nid: str):
        return self._notes.get(nid)

    def invalidate_note(self, note_id, when, superseded_by):
        self.invalidated.append((note_id, superseded_by))
        if note_id in self._notes:
            self._notes[note_id].valid_to = when

    def bump_note_access(self, note_id, when):
        self.bumped.append(note_id)


def test_note_conviction_rewards_corroboration_and_reinforcement() -> None:
    """A multi-episode, frequently-recalled note outranks a one-off, unused one."""
    now_dt = iso_to_dt(now_iso())
    strong = _conv_note("s", "x", episodes=3, access=5)
    weak = _conv_note("w", "y", episodes=1, access=0)
    cs = _note_conviction(strong, now_dt=now_dt, recency_half_life_days=90)
    cw = _note_conviction(weak, now_dt=now_dt, recency_half_life_days=90)
    assert cs > cw
    assert cs - cw >= 6  # +2 episodes +5 access, minus near-equal recency


def test_phase4b_resolves_when_conviction_separates_and_coverage_clean() -> None:
    """Genuine contradiction + decisive conviction gap + clean coverage → the
    lower-conviction note is superseded by the higher-conviction one."""
    strong = _conv_note("strong", "Service runs on port 9000",
                         episodes=3, access=5, contradicts=["weak"])
    weak = _conv_note("weak", "Service runs on port 8080", episodes=1, access=0)
    store = _ResolveStore([strong, weak])
    judge = _mock_llm(
        f'[{{"existing_id": "weak", "verdict": "CONTRADICTS", "reason": "port"}}]',
        '{"fully_superseded": true, "reason": "restates with corrected value"}',
    )

    resolved, _ = _phase4b_resolve_contradictions(
        store=store, judge=judge, config=DreamConfig(resolve_contradictions=True),
        now=now_iso(), journal=[],
    )

    assert resolved == 1
    assert store.invalidated == [("weak", "strong")]  # lower-conviction superseded
    assert store.bumped == ["strong"]


def test_phase4b_holds_when_conviction_gap_too_small() -> None:
    """Genuine contradiction but the two sides are evenly matched → leave both."""
    a = _conv_note("a", "Service runs on port 9000", episodes=1, access=0,
                   contradicts=["b"])
    b = _conv_note("b", "Service runs on port 8080", episodes=1, access=0)
    store = _ResolveStore([a, b])
    judge = _mock_llm(
        f'[{{"existing_id": "b", "verdict": "CONTRADICTS", "reason": "port"}}]'
    )

    resolved, _ = _phase4b_resolve_contradictions(
        store=store, judge=judge, config=DreamConfig(resolve_contradictions=True),
        now=now_iso(), journal=[],
    )

    assert resolved == 0
    assert store.invalidated == []


def test_phase4b_skips_false_contradiction_even_with_big_gap() -> None:
    """If the re-check says it is NOT a genuine contradiction (e.g. different
    scope), the pair is never resolved — conviction alone cannot destroy a fact."""
    strong = _conv_note("strong", "AWS was 17% in Q4 2025",
                        episodes=9, access=20, contradicts=["weak"])
    weak = _conv_note("weak", "AWS was 18% in 2025", episodes=1, access=0)
    store = _ResolveStore([strong, weak])
    judge = _mock_llm(
        f'[{{"existing_id": "weak", "verdict": "COMPLEMENTS", "reason": "Q4 vs full year"}}]'
    )

    resolved, _ = _phase4b_resolve_contradictions(
        store=store, judge=judge, config=DreamConfig(resolve_contradictions=True),
        now=now_iso(), journal=[],
    )

    assert resolved == 0
    assert store.invalidated == []


def test_confirm_contradiction_failsafe_returns_false_on_error() -> None:
    """ADR-0014 §2 fail-safe: a confirm-model error must return False (not confirmed)
    so a provider outage can never green-light a destructive supersede."""
    existing = _existing_note(text="x")
    boom = MagicMock()
    boom.complete.side_effect = RuntimeError("provider down")
    ok, tok = _confirm_contradiction(
        confirm_llm=boom, candidate_text="y", existing=existing
    )
    assert ok is False
    assert tok == 0


def test_phase4_midband_duplicate_via_llm_no_insert() -> None:
    """Mid-band + DUPLICATE verdict → deduped, no insert."""
    existing = _existing_note()
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "DUPLICATE", "reason": "same fact"}}]'
    )
    cand = _CandidateFact(
        text="Igor prefers PostgreSQL for relational data", source_episode_id="ep1"
    )

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    assert result["deduped"] == 1
    assert result["added"] == 0
    store.insert_note.assert_not_called()
    llm.complete.assert_called_once()


def test_phase4_value_change_now_reaches_llm() -> None:
    """Facts differing only in a value must reach the LLM, not get auto-deduped.

    Regression test for the DUPLICATE_DIST_BELOW tuning: real CONTRADICTS pairs
    (e.g. "port 8080" -> "port 9000") measure at ~0.63 L2 distance on
    text-embedding-3-small (see the dreaming.py threshold comments) — squarely
    in the mid-band, never close enough to trip the duplicate pre-filter. This
    pins that a mid-band distance reaches the LLM and can be classified
    CONTRADICTS rather than silently short-circuited as a duplicate.
    """
    existing = _existing_note(text="The service runs on port 8080")
    value_change_dist = 0.6  # measured CONTRADICTS pairs land around here — mid-band
    store = _make_phase4_store([(existing, value_change_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "CONTRADICTS", "reason": "port changed"}}]'
    )
    cand = _CandidateFact(
        text="The service runs on port 9000", source_episode_id="ep1"
    )

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    # LLM must have been consulted (key assertion for the threshold change).
    llm.complete.assert_called_once()
    assert result["quarantined"] == 1   # ADR-0014: kept both, not invalidated
    assert result["invalidated"] == 0
    assert result["added"] == 1


# --- Preference-protection gate ----------------------------------------------
#
# 2026-06-07 consolidation dry-run review found a systematic, repeatable LLM
# bias that survived even gpt-4o + an independent adversarial second opinion:
# a 'preference' note (a standing value) was wrongly judged DUPLICATE/
# CONTRADICTS by an episodic note (problem/workflow/project/fix) that merely
# *mentioned* the same topic — e.g. ruling a one-off "assistant used the wrong
# tool" incident note "contradicts" the user's stated standing preference for
# ai-memory as primary memory system. 4/4 sampled preference-vs-non-preference
# verdicts were wrong; 0/3 preference-vs-preference verdicts were. These tests
# pin the structural override that closes that gap without any extra LLM cost.


def test_phase4_verdict_with_unknown_existing_id_is_ignored_not_fatal() -> None:
    """A verdict referencing an existing_id absent from the shown neighbours
    must be skipped, not crash the integration pass (live-corpus regression:
    gpt-4o occasionally echoes back an id that doesn't match any neighbour)."""
    existing = _existing_note(id_="real-neighbour", text="Igor uses PostgreSQL")
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        '[{"existing_id": "hallucinated-id-not-in-neighbours", '
        '"verdict": "DUPLICATE", "reason": "looks similar"}]'
    )
    cand = _CandidateFact(text="Igor uses MySQL", source_episode_id="ep1")

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    # Verdict ignored -> falls through to a normal insert, no crash.
    assert result["deduped"] == 0
    assert result["added"] == 1
    store.bump_note_access.assert_not_called()
    store.invalidate_note.assert_not_called()


def test_phase4_preference_protection_overrides_contradicts_from_non_preference() -> None:
    """A 'problem'-tagged candidate must not be allowed to supersede a 'preference' note."""
    existing = _existing_note(
        text="The user wants the ai-memory connector to be the primary memory system.",
        tags=["preference"],
    )
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "CONTRADICTS", "reason": "used a different tool once"}}]'
    )
    cand = _CandidateFact(
        text="The assistant used a different tool to save one fact.",
        tags=["problem"],
        source_episode_id="ep1",
    )

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    # Overridden to insert-as-new — the standing preference survives untouched.
    assert result["invalidated"] == 0
    assert result["added"] == 1
    store.invalidate_note.assert_not_called()


def test_phase4_preference_protection_overrides_duplicate_from_non_preference() -> None:
    """A 'workflow'-tagged candidate must not be merged away into a 'preference' note."""
    existing = _existing_note(
        text="The user prefers the ai-memory connector to be the primary memory system.",
        tags=["preference"],
    )
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "DUPLICATE", "reason": "same topic"}}]'
    )
    cand = _CandidateFact(
        text="The assistant suggested writing custom instructions for ai-memory.",
        tags=["workflow"],
        source_episode_id="ep1",
    )

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    # Overridden to insert-as-new — the episodic detail is preserved, not discarded.
    assert result["deduped"] == 0
    assert result["added"] == 1
    store.bump_note_access.assert_not_called()
    store.insert_note.assert_called_once()


def test_phase4_preference_vs_preference_duplicate_still_applies() -> None:
    """The gate must not block legitimate preference-vs-preference merges."""
    existing = _existing_note(text="Direct, no sugar-coating", tags=["preference"])
    mid_dist = (DUPLICATE_DIST_BELOW + UNRELATED_DIST_ABOVE) / 2
    store = _make_phase4_store([(existing, mid_dist)])
    emb = _make_phase4_embedder()
    llm = _mock_llm(
        f'[{{"existing_id": "{existing.id}", "verdict": "DUPLICATE", "reason": "same value"}}]'
    )
    cand = _CandidateFact(
        text="Prefers direct, concise answers.",
        tags=["preference"],
        source_episode_id="ep1",
    )

    result, _ = _phase4_integrate(
        candidates=[cand], store=store, embedder=emb, llm=llm,
        now=now_iso(), journal=[],
    )

    assert result["deduped"] == 1
    assert result["added"] == 0
    store.bump_note_access.assert_called_once()
