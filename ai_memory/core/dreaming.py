"""
Dreaming module — sleep-style consolidation cycle.

This is the heavy LLM pipeline that turns raw turns into clean memory:
    Phase 1 — REPLAY     (read recent turns, score importance — implicit)
    Phase 2 — CLUSTER    (segment into episodes by topic/time — implicit, one
                          episode per write-side group; refinement deferred)
    Phase 3 — CONSOLIDATE (LLM summarises episodes, extracts atomic facts)
    Phase 4 — INTEGRATE  (dedupe via vector + LLM, contradict-resolution, link)
    Phase 5 — PROMOTE    (recurring patterns -> profile)
    Phase 6 — DECAY/PRUNE (soft-delete stale notes)
    Phase 7 — JOURNAL    (write dream_log entry)

Cost discipline: Phase 4 is where LLM cost can balloon. We bracket it tightly:
    - similarity > DUPLICATE_SIM_HIGH  -> trust as duplicate, no LLM
    - similarity < AMBIGUOUS_SIM_LOW   -> trust as unrelated, no LLM
    - middle band                       -> one LLM call per candidate
This keeps the LLM where its judgement actually matters.

Phase 5 clusters all valid notes whenever the dream pass runs. For a typical
single-user memory (thousands of notes) this is fast enough. For very large
memories we'd switch to incremental clustering — deferred.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Callable, Iterable, List, Literal, Tuple, TypeVar

from ai_memory.ids import new_id

from pydantic import BaseModel, ValidationError

from ai_memory.config import DreamConfig
from ai_memory.core.models import DreamLog, Note, Profile
from ai_memory.embeddings.interface import Embedder
from ai_memory.llm.interface import CompletionResult, Llm, Message
from ai_memory.storage.raw_files import RawTranscriptStore
from ai_memory.timestamps import iso_to_dt, now_iso

if TYPE_CHECKING:
    from ai_memory.storage.interface import MemoryStore

logger = logging.getLogger(__name__)


# --- Tunables (kept here, not in config, since they're pipeline internals) ---

# Phase 4 — vector-similarity bands (cosine sim ~ 1 - distance for normalized vecs).
# sqlite-vec returns L2 distance for unit vectors; sim ~ 1 - dist/2. We compare on
# the *distance* directly so we don't have to assume normalization — smaller is
# more similar.
#
# 2026-06-07 eval-driven re-measurement (drained corpus, text-embedding-3-small,
# unit vectors, 1907 valid notes — see remediation plan P1 "Tune Phase 4 dedup"):
# short English fact statements compress into a much narrower distance band than
# the old 0.10/0.55 split assumed. Controlled probe pairs:
#   identical text                                    -> 0.00
#   same fact, reworded ("drinks tea" vs "tea, not coffee") -> 0.87
#   same fact, richer context                         -> 0.64
#   CONTRADICTS (port 8080 vs port 9000)              -> 0.63
#   same topic, different subject (Igor/Emily, both "lives in London") -> 1.09
#   genuinely unrelated                               -> 1.41
# and the corpus's own nearest-neighbour distance distribution sits at
# min=0.56 / median=0.80 / p75=0.90. Live near-dup clusters already in storage
# (Citywire-role, WSL-environment, London-residency, tea-preference) measured
# pairwise at 0.55–1.06 — i.e. ABOVE the old UNRELATED_DIST_ABOVE=0.55, so they
# were classified "clearly unrelated" and inserted without ever reaching the LLM
# verdict. That mis-set ceiling — not DUPLICATE_DIST_BELOW — is *why* those
# clusters built up: paraphrased duplicates and contradictions both land well
# above 0.55, indistinguishable from each other by distance alone (0.87 vs 0.63).
# Only the LLM verdict can tell them apart; the pre-filters can only decide
# whether it's worth asking.
DUPLICATE_DIST_BELOW = 0.15   # near-byte-identical text only -> dedup without LLM
# Raised slightly from 0.10, but kept low and treated as a cheap optimisation,
# NOT a semantic-dedup mechanism: real paraphrased duplicates sit at 0.6-0.9,
# far above any safe auto-dedup line (auto-deduping there would also swallow
# genuine CONTRADICTS pairs, which measure even closer at ~0.63). This band
# only catches whitespace/punctuation-level rephrasings; everything else must
# go through the LLM verdict below.
UNRELATED_DIST_ABOVE = 1.05   # clearly unrelated -> insert without LLM
# Raised from 0.55 (which sat *below* the corpus's nearest-neighbour floor of
# 0.56, so almost nothing was ever filtered as "related" — Phase 4's LLM check
# was nearly dead code). 1.05 sits just below the measured "different subject,
# same topic" pair (1.09) and "genuinely unrelated" pair (1.41), while still
# routing real near-dups (<=0.9), CONTRADICTS (~0.63) and "same domain,
# different fact" (~1.0) candidates into the LLM verdict where they belong.
# Trade-off: this means *most* candidates now reach the LLM (median NN distance
# is 0.80, well inside the band) — a real per-pass cost increase on the
# rate-limited account, but it's the only way to get real semantic dedup with
# this embedding model's compressed distance range for short factual English.
INTEGRATE_NEIGHBOURS = 5      # how many existing notes to compare a candidate against

# Phase 5 — promotion clustering.
PROMOTION_CLUSTER_DIST = 0.65   # join two notes into a cluster if they're closer than this
PROMOTION_MIN_NOTES = 3         # cluster size required to consider promoting
PROMOTION_MIN_EPISODES = 2      # number of distinct episodes the cluster must span

# Phase 6 — decay and prune.
PRUNE_SCORE_THRESHOLD = 0.05    # below this, the note is a candidate for soft-delete
PRUNE_RECENT_RECALL_DAYS = 90   # don't prune if accessed within this window
PRUNE_MIN_AGE_DAYS = 30         # never prune a note younger than this

# Phase 3 — extract chunking. gpt-4o-mini's effective fact-extraction window
# is much smaller than its raw context window. With ~50 turns per chunk
# (~10k tokens) and a small overlap, each call stays sharp and grounded
# instead of degenerating into vague summaries + filler.
EXTRACT_CHUNK_TURNS = 50
EXTRACT_CHUNK_OVERLAP = 5


# --- Public dataclasses ----------------------------------------------------


@dataclass(frozen=True)
class DreamRequest:
    trigger: str = "manual"  # 'scheduled' | 'idle' | 'pressure' | 'manual'
    since: str | None = None  # ISO 8601 UTC; default: time of last completed dream
    max_episodes: int | None = None  # cap episodes processed this pass (None = all)


@dataclass
class DreamReport:
    log_id: str
    episodes_processed: int
    notes_added: int
    notes_invalidated: int
    notes_promoted_to_profile: int
    notes_pruned: int
    journal: str
    # Episodes that raised during Phase 3 and were left pending for retry.
    # Surfaced so the daemon can trip a circuit breaker on repeated failure
    # instead of hammering the same doomed work every poll tick.
    episodes_failed: int = 0


@dataclass
class _CandidateFact:
    """A fact extracted by Phase 3 that hasn't yet been integrated."""

    text: str
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    source_episode_id: str = ""


# --- Pydantic response models (schema-enforce LLM JSON output) ---------------

class _ExtractedFact(BaseModel):
    """One fact returned by the Phase 3 extract prompt."""
    text: str
    tags: List[str] = []
    entities: List[str] = []


class _IntegrateVerdict(BaseModel):
    """One verdict returned by the Phase 4 classify prompt."""
    existing_id: str
    verdict: Literal["DUPLICATE", "CONTRADICTS", "COMPLEMENTS", "UNRELATED"]
    reason: str = ""


class _PromotionResult(BaseModel):
    """Promotion decision returned by the Phase 5 promotion prompt."""
    key: str | None = None
    value: str | None = None
    rationale: str = ""


class _CoverageResult(BaseModel):
    """ADR-0014 §3 partial-information check: does the NEW fact fully preserve the
    still-valid content of the OLD fact, so OLD can be safely discarded?"""
    fully_superseded: bool
    reason: str = ""


_T = TypeVar("_T")
_MISSING = object()  # sentinel for _safe_parse_json failure detection


# Domain tag vocabulary — the only values the LLM is allowed to assign.
# Lowercase, validated at extraction time; anything else is dropped.
DOMAIN_TAGS: frozenset = frozenset({
    "preference", "project", "technical", "workflow", "personal",
    "problem", "fix", "person", "learning",
})


# --- Prompts (kept verbatim so dream-log entries can quote them later) -----

SUMMARY_SYSTEM = (
    "You analyse transcripts of conversations that have ALREADY HAPPENED. "
    "You are NOT a participant in the conversation. Never continue, reply "
    "to, or extend the dialogue. Never address the user. Never write "
    "follow-up questions, instructions, or next-step suggestions.\n\n"
    "Your output is a single concise paragraph (3-6 sentences) summarising "
    "what happened in the transcript: the topics discussed, the concrete "
    "decisions made, the tools and file paths involved, and any threads "
    "that were left open. Be specific, not generic. Refer to the "
    "participants in the third person ('the user', 'the assistant'). "
    "No greetings, no markdown headings, no bullet lists — just a "
    "third-person paragraph describing what occurred."
)

EXTRACT_SYSTEM = (
    "You analyse transcripts of conversations that have ALREADY HAPPENED "
    "and extract atomic facts. You are NOT a participant — never "
    "continue, reply to, or extend the conversation.\n\n"
    "Extract facts of THREE kinds and aim for a balanced spread across "
    "all three. Long transcripts (>100 turns) should yield 30-60 facts "
    "total; short ones can yield fewer. Prefer many short specific "
    "facts over a few generic ones.\n\n"
    "1. ABOUT THE USER — role, environment, preferences, hobbies, "
    "language stack. Examples: 'Igor works as a Senior Software Developer "
    "at Citywire', 'Igor prefers Postgres over MySQL'.\n\n"
    "2. ABOUT THE SYSTEM / PROJECT being built or discussed — concrete "
    "technical decisions, library names, file paths, versions, "
    "configurations, schemas, formulas. Be specific. Examples: "
    "'ai-memory stores notes in SQLite with the sqlite-vec extension "
    "and uses FTS5 for BM25 keyword search', 'Recall fuses vector and "
    "BM25 results via Reciprocal Rank Fusion with k=60'.\n\n"
    "3. PROBLEMS HIT AND FIXES APPLIED — specific gotchas, error "
    "messages, root causes, and the fix. These are the lessons-learned "
    "that make the memory worth having.\n\n"
    "Each fact object has THREE fields:\n"
    "  text     : one self-contained third-person sentence.\n"
    "  tags     : 1-3 domain tags chosen ONLY from this fixed list:\n"
    "               preference — something Igor prefers, values, likes, or avoids\n"
    "               project    — a specific work or personal project\n"
    "               technical  — stack, tools, languages, libs, config, architecture\n"
    "               workflow   — how Igor works: process, habits, tooling choices\n"
    "               personal   — life outside work: hobbies, lifestyle, background\n"
    "               problem    — a bug, error, failure, or challenge encountered\n"
    "               fix        — a solution, workaround, or resolution applied\n"
    "               person     — someone in Igor's personal or professional life\n"
    "               learning   — something Igor is actively studying or improving\n"
    "             Use ONLY these exact lowercase values. Any other value is invalid.\n"
    "  entities : 0-5 named things the fact is about (tools, projects, companies,\n"
    "             technologies, people). Normalise to lowercase-slug:\n"
    "             'AWS Lambda' → 'aws-lambda', 'Citywire' → 'citywire',\n"
    "             'Visual Studio Code' → 'vscode'.\n"
    "             PREFER REUSING existing entity names (provided below) over\n"
    "             inventing new ones. Only create a new entity slug as a last resort.\n\n"
    'Output a JSON array: [{"text": "...", "tags": [...], "entities": [...]}]. '
    "Skip pleasantries, raw code snippets, and one-off transient errors. "
    "Output ONLY the JSON array, no prose, no markdown fences."
)

# --- Entity slug helpers ---------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    """Normalise a raw entity string to a lowercase hyphen-slug."""
    return _SLUG_RE.sub("-", text.strip().lower()).strip("-")


def _normalise_entity(raw: str, vocab: list[str]) -> str:
    """Slug-normalise `raw`, then fuzzy-match against existing vocabulary.

    If the closest existing slug has similarity ≥ 0.85 it's returned instead,
    collapsing near-duplicates like 'aws-lambda' vs 'lambda-aws'.
    Returns an empty string if the input is blank after slugging.
    """
    slugged = _slug(raw)
    if not slugged:
        return ""
    if not vocab:
        return slugged
    best, best_score = slugged, 0.0
    for existing in vocab:
        score = SequenceMatcher(None, slugged, existing).ratio()
        if score > best_score:
            best_score, best = score, existing
    return best if best_score >= 0.85 else slugged


def _build_extract_system(entity_vocab: list[str]) -> str:
    """Return EXTRACT_SYSTEM with existing entity vocab appended (capped at 150 entries)."""
    if not entity_vocab:
        return EXTRACT_SYSTEM
    vocab_str = ", ".join(entity_vocab[:150])
    return EXTRACT_SYSTEM + f"\n\nExisting entity slugs (reuse these): {vocab_str}"


INTEGRATE_VERDICT_SYSTEM = (
    "You are a memory consolidation worker. You receive a NEW candidate fact "
    "and one or more EXISTING facts already stored. For each existing fact, "
    "classify the relationship using exactly one of these four verdicts:\n\n"
    "DUPLICATE — the new fact asserts the SAME claim as the existing fact: "
    "same subject, same attribute, same value — just rephrased or at a different "
    "detail level. Equivalent terms count as same value "
    "('Postgres' = 'PostgreSQL', 'eu-west-1' = 'Europe Ireland').\n\n"
    "CONTRADICTS — the new fact asserts a DIFFERENT value for the SAME attribute "
    "of the SAME subject. The key pattern is: [same subject] + [same attribute] "
    "+ [different value] → CONTRADICTS. The existing fact must be updated because "
    "it is no longer true. Examples:\n"
    "  old='Igor uses PostgreSQL as primary DB', new='Igor switched to MySQL' "
    "→ CONTRADICTS (same person, same attribute: primary DB, different value)\n"
    "  old='service runs on port 8080', new='service runs on port 9000' "
    "→ CONTRADICTS (same service, same attribute: port, different value)\n"
    "  old='Igor works at Citywire', new='Igor left Citywire and joined Acme' "
    "→ CONTRADICTS (same person, same attribute: employer, different value)\n"
    "  old='system uses Python 3.11', new='system upgraded to Python 3.12' "
    "→ CONTRADICTS (same system, same attribute: Python version, different value)\n"
    "Do NOT label as DUPLICATE when the value has changed — that is CONTRADICTS.\n"
    "Different subjects or compatible parallel facts are NOT contradictions "
    "(two services on different ports, two projects in different regions).\n\n"
    "SAME-SCOPE REQUIREMENT — a contradiction needs the SAME subject, attribute, "
    "scope, timeframe, and object. If the two facts differ in ANY of the following, "
    "they are NOT a contradiction — use COMPLEMENTS or UNRELATED:\n"
    "  - different TIMEFRAME: 'AWS was 17% in Q4 2025' vs 'AWS was 18% in 2025' "
    "(a quarter is not the full year) → not a contradiction\n"
    "  - different OBJECT / variant: 'OpenRun Pro maxes at 89 dB' vs 'OpenRun maxes "
    "at 85 dB' (different products) → not a contradiction\n"
    "  - different ENTITY: 'Roald Dahl's bibliography' vs 'Robert Dahl's bibliography' "
    "(different people) → not a contradiction\n"
    "  - GENERAL principle vs SPECIFIC instance: 'UK walls are usually 400mm, "
    "sometimes 600mm' vs 'this wall is 600mm' → not a contradiction\n"
    "  - two SEPARATE measurements / events: 'a dream pass used 127k tokens' vs "
    "'a dream pass used 150022 tokens' (different runs) → not a contradiction\n"
    "Only assert CONTRADICTS when the SAME thing, measured the SAME way, at the SAME "
    "scope, now genuinely has a different value.\n\n"
    "COMPLEMENTS — adds genuinely new information not inferable from the existing note.\n\n"
    "UNRELATED — completely different topic.\n\n"
    "Bias DUPLICATE over COMPLEMENTS when the underlying value is the same. "
    "Use CONTRADICTS whenever the value of an existing attribute has clearly changed.\n"
    'Output JSON: [{"existing_id": "...", "verdict": "DUPLICATE|CONTRADICTS|COMPLEMENTS|UNRELATED", "reason": "..."}]. '
    "Output ONLY the JSON array, no prose."
)

SUPERSEDE_COVERAGE_SYSTEM = (
    "You are a memory-consolidation SAFETY CHECK. A NEW fact has been judged to "
    "contradict and replace an OLD fact, and the OLD fact is about to be deleted. "
    "Your job is to prevent the loss of any still-true information.\n\n"
    "Decide: does the NEW fact preserve ALL the still-valid information in the OLD "
    "fact, so the OLD fact can be safely discarded? Only the single value being "
    "corrected may differ; EVERYTHING ELSE asserted by OLD must also be present in "
    "NEW. If OLD contains any additional attribute, qualifier, or related fact that "
    "NEW does not restate, answer false — both must be kept.\n\n"
    "Examples:\n"
    "  OLD='SQLite uses WAL mode and a 15s busy_timeout' "
    "NEW='busy_timeout set to 5000ms' → false (NEW drops the WAL-mode fact)\n"
    "  OLD='service runs on port 8080' NEW='service runs on port 9000' "
    "→ true (NEW fully restates OLD with only the corrected value)\n"
    "  OLD='Igor works at Citywire as a team lead' NEW='Igor left Citywire' "
    "→ false (NEW drops the role detail)\n"
    "  OLD='Python 3.11' NEW='upgraded to Python 3.12' → true\n\n"
    "When unsure, answer false — keeping a redundant note is cheap; deleting a "
    "still-true fact is not.\n"
    'Output JSON: {"fully_superseded": true|false, "reason": "..."}. '
    "Output ONLY the JSON object."
)

PROMOTION_SYSTEM = (
    "You are a careful memory consolidation worker deciding what to write into a "
    "PERMANENT user profile that is loaded into EVERY future conversation. A wrong "
    "entry silently poisons every future session, so decline unless you are confident.\n\n"
    "You receive a CLUSTER of related notes. PROMOTE only if the cluster describes a "
    "STABLE, DURABLE, FIRST-PERSON attribute of the user (Igor) HIMSELF: who he is, "
    "what he prefers, how he works, what he is skilled in or learning. The fact must "
    "still be true months from now and its subject must be IGOR — never an external "
    "company, product, market, news item, promotion, or one-off task.\n\n"
    "PROMOTE (about Igor): preferred tools / databases / languages, communication "
    "style, role or seniority, dev environment OS and shell, hobbies, lifestyle, "
    "what he is currently learning.\n"
    "  Good keys: preferred_database, communication_style, primary_language, "
    "role, dev_os, learning_language, favourite_sport.\n\n"
    "DO NOT PROMOTE — return null — for any of these:\n"
    "  - Facts whose SUBJECT is an external company/product/market Igor merely "
    "discussed. e.g. 'British Gas runs a Win Your Bill prize draw', 'British Gas "
    "Trading Limited company number 03078711', 'AWS was 18% of Amazon revenue', "
    "'AWS pricing model'. These describe the entity, not Igor.\n"
    "  - Volatile or project-specific implementation detail: file paths, repo "
    "structure, a project's current architecture, service settings, version "
    "numbers. e.g. 'ai-memory uses a four-tier model', 'the project venv lives "
    "under the project directory'. These change and are not durable personal facts.\n"
    "  - One-off events, tasks, promotions, emails, or troubleshooting steps.\n"
    "  - Anything where the subject of the fact is not Igor himself.\n\n"
    "When in doubt, decline. Missing a fact is cheap; poisoning the profile is not.\n"
    'Format: {"key": "snake_case_key", "value": "concise value", "rationale": '
    '"why this is a durable fact about Igor"} or {"key": null, "value": null} to '
    "decline. Output ONLY the JSON object."
)


# --- Public entry point ----------------------------------------------------


def dream(
    *,
    store: "MemoryStore",
    raw: RawTranscriptStore,
    embedder: Embedder,
    llm: Llm,
    quality_llm: "Llm | None" = None,
    confirm_llm: "Llm | None" = None,
    config: DreamConfig,
    request: DreamRequest,
    now: str | None = None,
) -> DreamReport:
    """Run a single dream pass. Inserts a DreamLog row and returns a report.

    `quality_llm`: when set, episodes with >= `config.long_episode_turns` turns
    use this LLM for Phase 3 (summary + extract) where extraction quality
    matters most.  Phase 4 and Phase 5 always use the base `llm`.

    `confirm_llm`: when set (ADR-0014 §2), a Phase 4 CONTRADICTS verdict only
    SUPERSEDES the existing note if this second model also agrees; otherwise the
    pair is quarantined. When None, ALL contradictions are quarantined.
    """
    now = now if now is not None else now_iso()
    log_id = new_id()

    # Journal-on-complete: the dream_log row is written exactly once, at the
    # END of the pass, fully populated. We deliberately do NOT insert a
    # placeholder row up front — a pass that died mid-way used to leave an
    # orphaned row (empty journal, ended_at=NULL), and a poison-pill episode
    # produced tens of thousands of them, rendering `dream-log` unreadable.
    log = DreamLog(id=log_id, started_at=now, trigger=request.trigger)

    # Fetch *all* unconsolidated episodes regardless of when they started.
    # An episode imported via the Cowork importer carries the chat's historical
    # started_at — possibly older than the last dream — but we still need to
    # consolidate it. The correctness rule is "consolidated_at IS NULL", not
    # "started_at >= since". `request.since` is preserved as an explicit hint
    # for callers who really want a date cutoff (rare).
    journal: List[str] = []
    tokens_used = 0
    notes_added = 0
    notes_invalidated = 0

    # --- Phase 1 + 2: collect un-consolidated episodes ------------------
    if request.since is not None:
        candidate_episodes = store.episodes_since(request.since)
    else:
        candidate_episodes = store.episodes_since("1970-01-01T00:00:00Z")
    episodes = [ep for ep in candidate_episodes if ep.consolidated_at is None]
    total_pending = len(episodes)
    if request.max_episodes is not None:
        episodes = episodes[: request.max_episodes]
    journal.append(
        f"Phase 1+2: processing {len(episodes)} of {total_pending} unconsolidated "
        f"episodes (scanned {len(candidate_episodes)} total)."
    )

    # --- Phase 3: consolidate each episode -----------------------------
    # Load entity vocab once before the loop so every extract call can
    # steer the LLM toward reusing known entity slugs.
    entity_vocab: list[str] = store.list_entity_vocab()
    extract_system = _build_extract_system(entity_vocab)

    candidate_facts: List[_CandidateFact] = []
    episodes_consolidated = 0
    episodes_failed = 0
    for ep in episodes:
        turns = list(store.get_turns_for_episode(ep.id))
        if not turns:
            continue

        # Per-episode isolation: one episode failing (rate limit, provider
        # error, bad data) must never abort the whole pass and block the rest
        # of the backlog. A failed episode is journalled and left pending
        # (consolidated_at stays NULL) so a later pass retries it.
        try:
            transcript = _render_transcript(raw, turns)

            # Select model: upgrade to quality_llm for long/dense episodes where
            # better extraction quality is worth the extra cost.
            long_threshold = (
                config.long_episode_turns if config.long_episode_turns > 0 else 10**9
            )
            ep_llm = (
                quality_llm
                if quality_llm is not None and len(turns) >= long_threshold
                else llm
            )
            if ep_llm is not llm:
                journal.append(
                    f"  Episode {ep.id}: {len(turns)} turns >= {long_threshold} — "
                    f"using quality model ({ep_llm.model_id}) for Phase 3."
                )

            # Summary: single-shot when the transcript fits the per-request
            # token budget; map-reduce (chunk -> merge) when it doesn't. This
            # guarantees no single request exceeds max_request_tokens, so a
            # long episode can never produce a permanent "request too large"
            # 429 (the deadlock this whole change fixes).
            summary_completion, summary_tokens, summary_mode = _summarise_episode(
                ep_llm=ep_llm,
                transcript=transcript,
                max_request_tokens=config.max_request_tokens,
            )
            tokens_used += summary_tokens

            # Extract: chunked. For short episodes _chunk_turns returns a single
            # window so behaviour is identical to the old single-shot path.
            turn_chunks = _chunk_turns(turns)
            all_facts: List[_ExtractedFact] = []
            chunk_sizes: List[int] = []
            chunk_warnings: List[str] = []  # structured failure log (Horcrux: typed failure result)
            for chunk_turns in turn_chunks:
                chunk_transcript = _render_transcript(raw, chunk_turns)
                extract_user_msg = (
                    "Extract atomic facts from the transcript chunk below "
                    "according to your instructions. The conversation has "
                    "already concluded — do not continue it. Output only the "
                    "JSON array.\n\n"
                    + chunk_transcript
                )
                chunk_facts, chunk_tokens = _llm_call_with_retry(
                    llm=ep_llm,
                    system=extract_system,
                    user_msg=extract_user_msg,
                    parse_fn=_parse_extract_facts,
                    max_tokens=4000,
                    default=[],
                    warnings=chunk_warnings,
                )
                tokens_used += chunk_tokens
                all_facts.extend(chunk_facts)
                chunk_sizes.append(len(chunk_facts))

            # Persist summary + mark consolidated only if we produced something.
            ep.summary = summary_completion.text.strip()

            # Generate a proper short title from the summary.  A tiny dedicated
            # call (≈10 output tokens on gpt-4o-mini) gives far better results
            # than slicing the first 80 chars of the paragraph.
            if ep.summary:
                title_completion = llm.complete(
                    system=(
                        "You are given a paragraph summarising a conversation. "
                        "Write a short title for it: 5-8 words, title case, no "
                        "punctuation at the end. Output only the title — no quotes, "
                        "no explanation, nothing else."
                    ),
                    messages=[Message(role="user", content=ep.summary)],
                    max_tokens=25,
                )
                tokens_used += title_completion.input_tokens + title_completion.output_tokens
                ep.title = title_completion.text.strip().strip('"').strip("'") or ep.summary[:80]
            else:
                ep.title = ""

            ep.embedding_model = embedder.model_id

            if ep.summary or all_facts:
                ep.consolidated_at = now

            try:
                [summary_embedding] = embedder.embed([ep.summary]) if ep.summary else [None]  # type: ignore[list-item]
            except Exception as exc:
                logger.warning("Failed to embed episode summary %s: %s", ep.id, exc)
                summary_embedding = None
            store.update_episode(ep, embedding=summary_embedding)

            this_ep_facts: List[_CandidateFact] = []
            for fact in all_facts:
                text = fact.text.strip()
                if not text:
                    continue
                # Domain tags: validate against controlled vocabulary, lowercase.
                # Drop any tag the LLM invented outside the allowed set; always
                # append the episode source as a provenance tag.
                raw_tags = [t.lower() for t in fact.tags]
                tags = [t for t in raw_tags if t in DOMAIN_TAGS] + [ep.source]
                # Entities: slug-normalise and fuzzy-match against known vocab.
                entities = [
                    norm for e in fact.entities
                    if (norm := _normalise_entity(e, entity_vocab))
                ]
                this_ep_facts.append(
                    _CandidateFact(text=text, tags=tags, entities=entities, source_episode_id=ep.id)
                )
            candidate_facts.extend(this_ep_facts)

            if ep.consolidated_at == now:
                episodes_consolidated += 1

            journal.append(
                f"  Episode {ep.id}: turns={len(turns)} chunks={len(turn_chunks)} "
                f"transcript_chars={len(transcript)} summary={len(ep.summary)} chars "
                f"(mode={summary_mode} finish={summary_completion.finish_reason}"
                f"{', refusal=' + summary_completion.refusal[:120] if summary_completion.refusal else ''}) "
                f"candidates={len(this_ep_facts)} (per-chunk={chunk_sizes})"
            )
            if chunk_warnings:
                # Structured failure log: surface extraction failures directly in the
                # dream journal so they're visible in `ai-memory dream-log` without
                # having to grep the Python logger output.
                for w in chunk_warnings:
                    journal.append(f"    EXTRACT_FAILURE: {w}")
            if len(this_ep_facts) == 0 and not chunk_warnings:
                journal.append(
                    "    note: 0 candidate facts extracted — "
                    "short episode, parse failure (see logs), or all facts deduplicated"
                )
        except Exception as exc:  # isolate the failure to this episode
            episodes_failed += 1
            logger.exception("Episode %s failed in Phase 3: %s", ep.id, exc)
            journal.append(
                f"  Episode {ep.id}: EPISODE_FAILURE in Phase 3 — "
                f"{type(exc).__name__}: {str(exc)[:200]}. Left pending for retry."
            )
            continue

    # --- Phase 4: integrate candidate facts ----------------------------
    integrated, phase4_tokens = _phase4_integrate(
        candidates=candidate_facts,
        store=store,
        embedder=embedder,
        llm=llm,
        confirm_llm=confirm_llm,
        now=now,
        journal=journal,
    )
    notes_added += integrated["added"]
    notes_invalidated += integrated["invalidated"]
    tokens_used += phase4_tokens

    # --- Phase 4b: conviction-gated resolution of quarantined ----------
    # contradictions (ADR-0014 §5). Off unless explicitly enabled; reuses the
    # cross-model confirmer when available, else the base llm.
    if config.resolve_contradictions:
        resolved, phase4b_tokens = _phase4b_resolve_contradictions(
            store=store,
            judge=confirm_llm or llm,
            config=config,
            now=now,
            journal=journal,
        )
        notes_invalidated += resolved
        tokens_used += phase4b_tokens

    # --- Phase 5: promote recurring patterns ---------------------------
    promoted, phase5_tokens = _phase5_promote(
        store=store,
        llm=llm,
        config=config,
        now=now,
        journal=journal,
        quality_llm=quality_llm,
    )
    tokens_used += phase5_tokens

    # --- Phase 6: decay and prune --------------------------------------
    pruned = _phase6_prune(store=store, config=config, now=now, journal=journal)

    # --- Phase 7: journal ----------------------------------------------
    journal.append(
        f"Pass complete: {episodes_consolidated} episodes consolidated, "
        f"{episodes_failed} failed (left pending). "
        f"LLM tokens used this pass: {tokens_used} ({llm.model_id})"
    )
    log.ended_at = now_iso()
    log.episodes_processed = episodes_consolidated
    log.notes_added = notes_added
    log.notes_invalidated = notes_invalidated
    log.notes_promoted_to_profile = promoted
    log.notes_pruned = pruned
    log.llm_tokens_used = tokens_used
    log.journal = "\n".join(journal)
    store.insert_dream_log(log)

    return DreamReport(
        log_id=log_id,
        episodes_processed=episodes_consolidated,
        notes_added=notes_added,
        notes_invalidated=notes_invalidated,
        notes_promoted_to_profile=promoted,
        notes_pruned=pruned,
        journal=log.journal,
        episodes_failed=episodes_failed,
    )


# --- Phase 4 ---------------------------------------------------------------


def _blocks_preference_override(existing_tags: List[str], candidate_tags: List[str]) -> bool:
    """True when a DUPLICATE/CONTRADICTS verdict should be overridden to insert-as-new.

    A note tagged ``preference`` encodes one of Igor's standing values — it
    should only ever be merged into or superseded by another statement of a
    preference, never by an episodic note (problem/workflow/project/fix) that
    merely *mentions* the same topic. The 2026-06-07 consolidation dry-run
    review found this exact asymmetry was a systematic, repeatable LLM bias
    (conflating "an event happened that deviates from a preference" with "the
    preference changed") that survived even gpt-4o + an independent adversarial
    second opinion: 4/4 sampled verdicts where a non-preference note challenged
    a preference note were wrong (including one that would have permanently
    destroyed a true standing preference); 0/3 preference-vs-preference
    verdicts were. This is a cheap structural gate using metadata the extract
    step already assigns — no extra LLM calls, and it closes a bias that
    re-querying the same judge demonstrably does not.
    """
    return "preference" in existing_tags and "preference" not in candidate_tags


def _phase4_integrate(
    *,
    candidates: List[_CandidateFact],
    store: "MemoryStore",
    embedder: Embedder,
    llm: Llm,
    confirm_llm: "Llm | None" = None,
    now: str,
    journal: List[str],
) -> Tuple[dict, int]:
    """Integrate candidate facts: dedup, contradict-resolve, insert.

    `confirm_llm` (ADR-0014 §2): a CONTRADICTS verdict only supersedes the
    existing note when this second model also returns CONTRADICTS for the same
    note; otherwise the pair is quarantined (both kept, linked). None → all
    contradictions quarantined.

    Returns ({added, invalidated, deduped, quarantined}, llm_tokens_used).
    """
    added = 0
    invalidated = 0
    deduped = 0
    quarantined = 0
    tokens = 0

    if not candidates:
        journal.append("Phase 4: no candidate facts to integrate.")
        return {"added": added, "invalidated": invalidated,
                "deduped": deduped, "quarantined": quarantined}, tokens

    # Embed all candidates in one batch — single API call instead of N.
    candidate_embeddings = embedder.embed([c.text for c in candidates])

    for cand, cand_emb in zip(candidates, candidate_embeddings):
        neighbours = store.search_notes_vector(
            cand_emb, k=INTEGRATE_NEIGHBOURS, only_valid=True
        )

        # Pre-filter: clear duplicate?
        if neighbours and neighbours[0][1] < DUPLICATE_DIST_BELOW:
            existing = neighbours[0][0]
            store.bump_note_access(existing.id, now)
            deduped += 1
            continue

        # Pre-filter: clearly unrelated to anything?
        if not neighbours or neighbours[0][1] > UNRELATED_DIST_ABOVE:
            _insert_candidate(store, cand, cand_emb, embedder, now)
            added += 1
            continue

        # Ambiguous middle band — ask the LLM.
        verdicts, t = _classify_candidate_vs_neighbours(
            llm=llm, candidate_text=cand.text, neighbours=neighbours
        )
        tokens += t

        action_taken = False
        for v in verdicts:
            if v.verdict not in ("DUPLICATE", "CONTRADICTS"):
                continue
            # The LLM occasionally returns an existing_id that doesn't match any
            # of the neighbours it was shown — ignore those rather than crashing.
            existing_note = next((n for n, _ in neighbours if n.id == v.existing_id), None)
            if existing_note is None:
                journal.append(
                    f"Phase 4: verdict referenced unknown existing_id {v.existing_id!r} "
                    f"— ignoring that verdict"
                )
                continue
            if _blocks_preference_override(existing_note.tags, cand.tags):
                journal.append(
                    f"Phase 4: preference-protection gate overrode {v.verdict} "
                    f"of {v.existing_id} (tags={existing_note.tags}) by "
                    f"non-preference candidate (tags={cand.tags}) -> inserting as new"
                )
                continue
            if v.verdict == "DUPLICATE":
                store.bump_note_access(v.existing_id, now)
                deduped += 1
                action_taken = True
                break  # done with this candidate
            # CONTRADICTS (ADR-0014): a single contradicting mention must not
            # destroy a standing fact unattended — the 2026-06-08 retro pass found
            # ~67% of the primary judge's contradictions were false. Supersede the
            # existing note ONLY if a second, ideally different-family model also
            # confirms the contradiction (§2 cross-model). Otherwise quarantine:
            # insert the new fact linked to the existing one, keep BOTH valid, and
            # defer resolution to the conviction gate / review digest (§5/§6).
            confirmed = False
            partial_info = False
            if confirm_llm is not None:
                confirmed, ctok = _confirm_contradiction(
                    confirm_llm=confirm_llm, candidate_text=cand.text,
                    existing=existing_note,
                )
                tokens += ctok
                if confirmed:
                    # §3 partial-information guard: even a cross-model-confirmed
                    # contradiction must not destroy the old note if the new one
                    # omits a still-true fact bundled into it (e.g. superseding a
                    # stale timeout would also delete a WAL-mode fact). If the new
                    # fact does not fully cover the old, downgrade to quarantine.
                    clean, stok = _new_supersedes_cleanly(
                        llm=confirm_llm, new_text=cand.text,
                        old_text=existing_note.text,
                    )
                    tokens += stok
                    if not clean:
                        confirmed = False
                        partial_info = True
            inserted_id = _insert_candidate(
                store, cand, cand_emb, embedder, now, contradicts=[v.existing_id]
            )
            added += 1
            if confirmed:
                store.invalidate_note(v.existing_id, when=now, superseded_by=inserted_id)
                invalidated += 1
                journal.append(
                    f"Phase 4: CONTRADICTS confirmed cross-model + full coverage — "
                    f"superseded {v.existing_id} with new {inserted_id} (ADR-0014 §2/§3)"
                )
            else:
                quarantined += 1
                if partial_info:
                    why = "partial information — old note has facts the new one omits"
                elif confirm_llm is not None:
                    why = "second model disagreed"
                else:
                    why = "no cross-model confirmer configured"
                journal.append(
                    f"Phase 4: CONTRADICTS quarantined ({why}) — kept both existing "
                    f"{v.existing_id} and new {inserted_id}; contradicts link "
                    f"recorded, neither invalidated (ADR-0014)"
                )
            action_taken = True
            break

        if not action_taken:
            # Either UNRELATED or COMPLEMENTS for everything — insert.
            _insert_candidate(store, cand, cand_emb, embedder, now)
            added += 1

    journal.append(
        f"Phase 4: integrated {len(candidates)} candidates -> "
        f"added={added}, invalidated={invalidated}, deduped={deduped}, "
        f"quarantined={quarantined}"
    )
    return {"added": added, "invalidated": invalidated, "deduped": deduped,
            "quarantined": quarantined}, tokens


def _confirm_contradiction(
    *, confirm_llm: Llm, candidate_text: str, existing: Note
) -> Tuple[bool, int]:
    """ADR-0014 §2: ask a second, independent model whether the candidate genuinely
    CONTRADICTS the existing note. Returns (confirmed, tokens_used).

    Fail-safe: any error, parse failure, or a non-CONTRADICTS verdict returns
    False. A failed or ambiguous confirmation must NEVER green-light a destructive
    supersede — the cost of a wrong destroy is higher than a redundant quarantine.
    """
    try:
        verdicts, tok = _classify_candidate_vs_neighbours(
            llm=confirm_llm, candidate_text=candidate_text, neighbours=[(existing, 0.0)]
        )
    except Exception:  # network / parse / provider error — do not destroy
        return False, 0
    v = next((x for x in verdicts if x.existing_id == existing.id), None)
    return (v is not None and v.verdict == "CONTRADICTS"), tok


def _parse_coverage(text: str) -> bool:
    """Parse {"fully_superseded": bool}. Raises ValueError on no/invalid JSON so the
    retry wrapper fires; returns the bool on success."""
    parsed = _safe_parse_json(text, default=_MISSING)
    if parsed is _MISSING or not isinstance(parsed, dict):
        raise ValueError(f"No valid JSON object in response: {text[:120]!r}")
    return _CoverageResult.model_validate(parsed).fully_superseded


def _new_supersedes_cleanly(
    *, llm: Llm, new_text: str, old_text: str
) -> Tuple[bool, int]:
    """ADR-0014 §3 partial-information guard. Returns (safe_to_destroy_old, tokens).

    True only if the model judges the NEW fact preserves every still-valid fact in
    the OLD fact, so superseding loses nothing. Fail-safe: any error or validation
    failure resolves to False (do NOT destroy — quarantine instead), consistent
    with §2: when unsure, never destroy.
    """
    try:
        result, tokens = _llm_call_with_retry(
            llm=llm,
            system=SUPERSEDE_COVERAGE_SYSTEM,
            user_msg=json.dumps({"old": old_text, "new": new_text}, ensure_ascii=False),
            parse_fn=_parse_coverage,
            max_tokens=200,
            default=False,
        )
    except Exception:  # network / provider error — do not destroy
        return False, 0
    return bool(result), tokens


def _note_conviction(note: Note, *, now_dt, recency_half_life_days: float) -> float:
    """ADR-0014 §5 conviction score for a quarantined note — higher = stronger
    standing. Combines independent corroboration (distinct source episodes),
    recall reinforcement (access_count), durability (promoted to profile), and a
    recency term in [0, 1]. Pure function so it is unit-testable."""
    corroboration = float(len(note.source_episode_ids))
    reinforcement = float(note.access_count)
    promoted = 2.0 if note.promoted_to_profile else 0.0
    ref = note.last_accessed_at or note.ingested_at or note.valid_from
    recency = 0.0
    if ref:
        try:
            age_days = max(0.0, (now_dt - iso_to_dt(ref)).total_seconds() / 86400.0)
            recency = math.exp(-age_days / max(1.0, recency_half_life_days))
        except Exception:
            recency = 0.0
    return corroboration + reinforcement + promoted + recency


def _phase4b_resolve_contradictions(
    *,
    store: "MemoryStore",
    judge: Llm,
    config: DreamConfig,
    now: str,
    journal: List[str],
) -> Tuple[int, int]:
    """ADR-0014 §5: resolve quarantined contradictions whose conviction has
    decisively separated. Returns (resolved_count, tokens_used).

    For each still-valid note linked via `contradicts` to a still-valid note:
      1. re-confirm it is a genuine contradiction (filters out false-contradictions
         that were quarantined — e.g. different-scope pairs — which must never be
         "resolved" by destroying a side);
      2. require the conviction gap between the two to exceed the configured
         minimum (evidence has actually accumulated for one side);
      3. require the higher-conviction side to fully cover the loser (§3 guard);
    then supersede the loser. Anything short of all three stays quarantined.
    """
    tokens = 0
    resolved = 0
    now_dt = iso_to_dt(now)
    half_life = float(config.decay_half_life_days)
    min_gap = config.contradiction_resolution_min_gap
    invalidated_ids: set = set()

    for note in store.list_valid_notes():
        if note.id in invalidated_ids or not note.contradicts:
            continue
        for old_id in list(note.contradicts):
            if old_id in invalidated_ids:
                continue
            old = store.get_note(old_id)
            if old is None or old.valid_to is not None:
                continue  # already resolved / gone

            genuine, t1 = _confirm_contradiction(
                confirm_llm=judge, candidate_text=note.text, existing=old
            )
            tokens += t1
            if not genuine:
                continue  # false-contradiction or no longer conflicting — keep both

            cn = _note_conviction(note, now_dt=now_dt, recency_half_life_days=half_life)
            co = _note_conviction(old, now_dt=now_dt, recency_half_life_days=half_life)
            if abs(cn - co) < min_gap:
                continue  # evidence has not separated them yet

            winner, loser = (note, old) if cn >= co else (old, note)
            clean, t2 = _new_supersedes_cleanly(
                llm=judge, new_text=winner.text, old_text=loser.text
            )
            tokens += t2
            if not clean:
                continue  # would lose still-true info — keep both

            store.invalidate_note(loser.id, when=now, superseded_by=winner.id)
            store.bump_note_access(winner.id, now)
            invalidated_ids.add(loser.id)
            resolved += 1
            journal.append(
                f"Phase 4b: resolved quarantined contradiction — superseded "
                f"{loser.id} (conviction {min(cn, co):.2f}) with {winner.id} "
                f"(conviction {max(cn, co):.2f}); gap {abs(cn - co):.2f} >= "
                f"{min_gap} (ADR-0014 §5)"
            )
            if loser.id == note.id:
                break  # the link holder was superseded — stop scanning its links

    journal.append(f"Phase 4b: resolved {resolved} quarantined contradiction(s).")
    return resolved, tokens


def _classify_candidate_vs_neighbours(
    *, llm: Llm, candidate_text: str, neighbours: List[Tuple[Note, float]]
) -> Tuple[List[_IntegrateVerdict], int]:
    """One LLM call that classifies the candidate against all K neighbours."""
    payload = {
        "candidate": candidate_text,
        "existing": [{"id": n.id, "text": n.text} for n, _ in neighbours],
    }
    verdicts, tokens = _llm_call_with_retry(
        llm=llm,
        system=INTEGRATE_VERDICT_SYSTEM,
        user_msg=json.dumps(payload, ensure_ascii=False),
        parse_fn=_parse_verdicts,
        max_tokens=800,
        default=[],
    )
    return verdicts, tokens


def _insert_candidate(
    store: "MemoryStore",
    cand: _CandidateFact,
    embedding: List[float],
    embedder: Embedder,
    now: str,
    contradicts: List[str] | None = None,
) -> str:
    """Persist a candidate as a new note. Returns the new note id."""
    note = Note(
        id=new_id(),
        text=cand.text,
        tags=cand.tags,
        entities=cand.entities,
        source_episode_ids=[cand.source_episode_id] if cand.source_episode_id else [],
        valid_from=now,
        ingested_at=now,
        embedding_model=embedder.model_id,
        contradicts=contradicts or [],
    )
    store.insert_note(note, embedding)
    return note.id


# --- Phase 5 ---------------------------------------------------------------


def _phase5_promote(
    *,
    store: "MemoryStore",
    llm: Llm,
    config: DreamConfig,
    now: str,
    journal: List[str],
    quality_llm: "Llm | None" = None,
) -> Tuple[int, int]:
    """Find recurring fact clusters and promote them to the profile.

    Returns (promotions_count, llm_tokens_used).
    """
    promoted = 0
    tokens = 0

    valid_notes = [n for n in store.list_valid_notes() if not n.promoted_to_profile]
    if len(valid_notes) < PROMOTION_MIN_NOTES:
        journal.append("Phase 5: not enough valid un-promoted notes to cluster.")
        return promoted, tokens

    # Pull embeddings for all valid notes (one indexed lookup each — cheap).
    embeddings: List[List[float] | None] = [
        store.get_note_embedding(n.id) for n in valid_notes
    ]

    # Cluster by simple greedy union-find on pairwise distance.
    clusters = _cluster_notes(valid_notes, embeddings, dist_threshold=PROMOTION_CLUSTER_DIST)

    min_episodes = max(PROMOTION_MIN_EPISODES, config.promotion_min_episodes)
    min_notes = max(PROMOTION_MIN_NOTES, config.promotion_min_endorsing_notes)

    journal.append(
        f"Phase 5: {len(valid_notes)} notes clustered at dist<={PROMOTION_CLUSTER_DIST} "
        f"→ {len(clusters)} multi-note clusters "
        f"(need ≥{min_notes} notes across ≥{min_episodes} episodes)"
    )

    for cluster in clusters:
        if len(cluster) < min_notes:
            continue
        episode_ids = {eid for n in cluster for eid in n.source_episode_ids}
        if len(episode_ids) < min_episodes:
            continue

        # Ask the LLM whether this cluster is worth promoting and what key/value to use.
        cluster_payload = [{"id": n.id, "text": n.text} for n in cluster]
        promotion, promo_tokens = _llm_call_with_retry(
            llm=quality_llm or llm,
            system=PROMOTION_SYSTEM,
            user_msg=json.dumps(cluster_payload, ensure_ascii=False),
            parse_fn=_parse_promotion,
            max_tokens=300,
            default=_PromotionResult(),
        )
        tokens += promo_tokens

        key = (promotion.key or "").strip()
        value = (promotion.value or "").strip()

        if not key or not value:
            continue  # LLM declined promotion

        store.upsert_profile(
            Profile(key=_sanitise_profile_key(key), value=value, updated_at=now, source="dream")
        )
        for n in cluster:
            n.promoted_to_profile = True
            store.update_note(n)
        promoted += 1
        journal.append(
            f"Phase 5: promoted cluster of {len(cluster)} notes "
            f"({len(episode_ids)} episodes) -> profile.{_sanitise_profile_key(key)} = {value!r}"
        )

    if promoted == 0:
        journal.append("Phase 5: no promotions this pass.")
    return promoted, tokens


def _cluster_notes(
    notes: List[Note],
    embeddings: List[List[float] | None],
    dist_threshold: float,
) -> List[List[Note]]:
    """Greedy union-find clustering on pairwise vector distance.

    O(n^2) — fine up to a few thousand valid notes. Switch to HNSW or
    incremental clustering when we outgrow that.
    """
    n = len(notes)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        if embeddings[i] is None:
            continue
        for j in range(i + 1, n):
            if embeddings[j] is None:
                continue
            d = _euclid_distance(embeddings[i], embeddings[j])  # type: ignore[arg-type]
            if d <= dist_threshold:
                union(i, j)

    buckets: dict[int, List[Note]] = {}
    for i, note in enumerate(notes):
        buckets.setdefault(find(i), []).append(note)
    return [v for v in buckets.values() if len(v) > 1]  # singletons aren't clusters


def _euclid_distance(a: List[float], b: List[float]) -> float:
    """L2 distance — same metric sqlite-vec returns. Avoids needing numpy."""
    if len(a) != len(b):
        return float("inf")
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


_KEY_SANITISE = re.compile(r"[^a-z0-9]+")
_KEY_COLLAPSE = re.compile(r"_+")


def _sanitise_profile_key(key: str) -> str:
    """Force snake_case; LLM sometimes returns 'Preferred Database' etc.

    Non-alphanumeric runs → single underscore; repeated underscores collapsed.
    """
    cleaned = _KEY_SANITISE.sub("_", key.strip().lower())
    cleaned = _KEY_COLLAPSE.sub("_", cleaned).strip("_")
    return cleaned or "fact"


# --- Phase 6 ---------------------------------------------------------------


def _phase6_prune(
    *, store: "MemoryStore", config: DreamConfig, now: str, journal: List[str]
) -> int:
    """Soft-delete stale, never-recalled, never-promoted notes."""
    half_life = max(1, config.decay_half_life_days) * 86400
    pruned = 0
    now_dt = iso_to_dt(now)

    valid_notes = store.list_valid_notes()
    for note in valid_notes:
        if note.promoted_to_profile:
            continue  # never prune promoted facts
        age_seconds = max(0, (now_dt - iso_to_dt(note.ingested_at)).total_seconds())
        if age_seconds < PRUNE_MIN_AGE_DAYS * 86400:
            continue  # too young to prune
        last_access = note.last_accessed_at or note.ingested_at
        if (now_dt - iso_to_dt(last_access)).total_seconds() < PRUNE_RECENT_RECALL_DAYS * 86400 and note.access_count > 0:
            continue  # recently used

        decay = math.exp(-age_seconds / half_life)
        score = decay * (1.0 + math.log(1.0 + note.access_count))
        threshold = max(PRUNE_SCORE_THRESHOLD, config.prune_threshold)
        if score < threshold:
            store.invalidate_note(note.id, when=now, superseded_by=None)
            pruned += 1

    journal.append(f"Phase 6: pruned {pruned} stale notes.")
    return pruned


# --- Helpers ----------------------------------------------------------------


def _chunk_turns(
    turns: List, chunk_size: int = EXTRACT_CHUNK_TURNS,
    overlap: int = EXTRACT_CHUNK_OVERLAP,
) -> List[List]:
    """Split a turn list into overlapping windows for piecewise extraction.

    Single-shot extraction over a 200+ turn episode causes the model to
    cherry-pick early-conversation facts and pad the rest with filler.
    Smaller windows keep each call grounded in concrete content. Overlap
    preserves a few turns of context across chunk boundaries so a fact
    introduced at the boundary isn't missed by both sides.
    """
    if not turns:
        return []
    if len(turns) <= chunk_size:
        return [list(turns)]
    step = max(1, chunk_size - overlap)
    chunks: List[List] = []
    start = 0
    while start < len(turns):
        end = start + chunk_size
        chunks.append(list(turns[start:end]))
        if end >= len(turns):
            break
        start += step
    return chunks


def _render_transcript(raw: RawTranscriptStore, turns: Iterable) -> str:
    """Render an episode's turns as a single delimited transcript.

    The format intentionally does NOT look like a live chat (no `user:` /
    `assistant:` line prefixes), because that triggers the LLM's "continue
    this dialogue" behaviour and overrides the system prompt. Instead we
    wrap in explicit XML tags and number each turn — this reads as data,
    not as an in-progress conversation.
    """
    lines: List[str] = ["<conversation_transcript>"]
    for index, turn in enumerate(turns, start=1):
        payload = raw.read_turn(turn.raw_file, turn.byte_offset, turn.byte_length)
        role = payload.get("role", "user")
        text = payload.get("text", "")
        lines.append(f"<turn n={index} role={role}>")
        lines.append(text)
        lines.append("</turn>")
    lines.append("</conversation_transcript>")
    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token). Good enough to decide chunking."""
    return len(text) // 4


def _split_transcript_by_budget(transcript: str, budget_chars: int) -> List[str]:
    """Split a rendered transcript into <=budget_chars pieces at turn boundaries.

    Cuts happen between `</turn>` markers so a single turn is never split across
    two requests. A pathologically large single turn may still exceed the budget
    on its own; that's acceptable — it's one turn, not a 142k-token episode.
    """
    if budget_chars <= 0 or len(transcript) <= budget_chars:
        return [transcript]
    # Re-attach the `</turn>` delimiter to every fragment except a trailing
    # empty one, so "".join(segments) == transcript exactly (no spurious tag).
    parts = transcript.split("</turn>")
    segments: List[str] = []
    for i, block in enumerate(parts):
        if i < len(parts) - 1:
            segments.append(block + "</turn>")
        elif block:
            segments.append(block)
    pieces: List[str] = []
    current = ""
    for segment in segments:
        if current and len(current) + len(segment) > budget_chars:
            pieces.append(current)
            current = segment
        else:
            current += segment
    if current:
        pieces.append(current)
    return pieces


def _summarise_episode(
    *,
    ep_llm: "Llm",
    transcript: str,
    max_request_tokens: int,
) -> "Tuple[CompletionResult, int, str]":
    """Summarise an episode without exceeding the per-request token budget.

    Returns (final_completion, total_tokens_used, mode). `mode` is 'single-shot'
    when the transcript fits the budget, otherwise 'map-reduce(N chunks)': each
    chunk is summarised and the partial summaries are merged. This guarantees no
    single request exceeds max_request_tokens, so a long episode can never
    trigger a permanent 'request too large' 429 — the root cause of the deadlock.
    """
    if _estimate_tokens(transcript) <= max_request_tokens:
        completion = ep_llm.complete(
            system=SUMMARY_SYSTEM,
            messages=[Message(role="user", content=(
                "Summarise the transcript below according to your instructions. "
                "The conversation has already concluded — do not continue it.\n\n"
                + transcript
            ))],
            max_tokens=1200,
        )
        return completion, completion.input_tokens + completion.output_tokens, "single-shot"

    # Map: summarise each chunk separately (each well under the budget).
    budget_chars = max(1, max_request_tokens) * 4
    chunks = _split_transcript_by_budget(transcript, budget_chars)
    total_tokens = 0
    partials: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        c = ep_llm.complete(
            system=SUMMARY_SYSTEM,
            messages=[Message(role="user", content=(
                f"This is PART {index} of {len(chunks)} of a long transcript whose "
                "conversation has already concluded — do not continue it. "
                "Summarise only what happens in this part, according to your "
                "instructions.\n\n"
                + chunk
            ))],
            max_tokens=600,
        )
        total_tokens += c.input_tokens + c.output_tokens
        if c.text.strip():
            partials.append(c.text.strip())

    # Reduce: merge the partial summaries into one coherent paragraph.
    merged_input = "\n\n".join(f"[Part {i}] {p}" for i, p in enumerate(partials, start=1))
    final = ep_llm.complete(
        system=SUMMARY_SYSTEM,
        messages=[Message(role="user", content=(
            "Below are ordered partial summaries of consecutive parts of one "
            "conversation that has already concluded. Merge them into a single "
            "coherent summary according to your instructions — do not continue "
            "the conversation.\n\n"
            + merged_input
        ))],
        max_tokens=1200,
    )
    total_tokens += final.input_tokens + final.output_tokens
    return final, total_tokens, f"map-reduce({len(chunks)} chunks)"


def _llm_call_with_retry(
    *,
    llm: "Llm",
    system: str,
    user_msg: str,
    parse_fn: "Callable[[str], _T]",
    max_tokens: int,
    default: "_T",
    warnings: "list[str] | None" = None,
) -> "Tuple[_T, int]":
    """Make an LLM call, validate with parse_fn, retry once on schema failure.

    Returns (result, total_tokens_used).

    parse_fn should raise ValueError or ValidationError for bad output.
    JSON-level quirks (markdown fences, prose preambles) are handled
    by _safe_parse_json before Pydantic sees the data; this retry
    fires only when the structure is wrong (missing fields, bad enum).
    """
    completion = llm.complete(
        system=system,
        messages=[Message(role="user", content=user_msg)],
        max_tokens=max_tokens,
    )
    tokens = completion.input_tokens + completion.output_tokens

    first_error: str = ""
    try:
        return parse_fn(completion.text), tokens
    except (ValueError, ValidationError) as err:
        first_error = str(err)
        logger.warning("LLM output validation failed (retrying once): %s", first_error)

    retry_completion = llm.complete(
        system=system,
        messages=[
            Message(role="user", content=user_msg),
            Message(role="assistant", content=completion.text),
            Message(
                role="user",
                content=(
                    f"Your previous response was invalid: {first_error}. "
                    "Fix it and output ONLY valid JSON matching the required schema."
                ),
            ),
        ],
        max_tokens=max_tokens,
    )
    tokens += retry_completion.input_tokens + retry_completion.output_tokens

    try:
        return parse_fn(retry_completion.text), tokens
    except (ValueError, ValidationError) as err2:
        msg = f"LLM output validation failed after retry — using default: {err2}"
        logger.error(msg)
        if warnings is not None:
            warnings.append(msg)
        return default, tokens


def _parse_extract_facts(text: str) -> List[_ExtractedFact]:
    """Parse and validate the Phase 3 extract JSON array.

    Raises ValueError if the text contains no valid JSON at all, so that
    _llm_call_with_retry knows to retry. Returns [] (no error) for a valid
    empty array — the LLM is allowed to find zero facts in a short transcript.
    """
    text = text.strip()
    if not text:
        return []
    parsed = _safe_parse_json(text, default=_MISSING)
    if parsed is _MISSING:
        raise ValueError(f"No valid JSON in response: {text[:120]!r}")
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    facts: List[_ExtractedFact] = []
    errors: List[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            f = _ExtractedFact.model_validate(item)
            if f.text.strip():
                facts.append(f)
        except ValidationError as e:
            errors.append(str(e))
    if errors and not facts:
        raise ValueError(f"All {len(errors)} fact(s) failed validation: {errors[0]}")
    return facts


def _parse_verdicts(text: str) -> List[_IntegrateVerdict]:
    """Parse and validate the Phase 4 verdict JSON array."""
    parsed = _safe_parse_json(text, default=_MISSING)
    if parsed is _MISSING:
        raise ValueError(f"No valid JSON in response: {text[:120]!r}")
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    verdicts: List[_IntegrateVerdict] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            verdicts.append(_IntegrateVerdict.model_validate(item))
        except ValidationError:
            pass
    if not verdicts and parsed:
        raise ValueError(f"No valid verdict objects parsed from {len(parsed)} items")
    return verdicts


def _parse_promotion(text: str) -> _PromotionResult:
    """Parse and validate the Phase 5 promotion JSON object."""
    parsed = _safe_parse_json(text, default=_MISSING)
    # LLM may return null/None to signal 'do not promote'
    if parsed is None:
        return _PromotionResult()
    if parsed is _MISSING:
        raise ValueError(f"No valid JSON in response: {text[:120]!r}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return _PromotionResult.model_validate(parsed)


def _safe_parse_facts(text: str) -> List[dict]:
    """Best-effort JSON parse of the extract step's output.

    Accepts:
        [{...}, {...}]      — well-formed JSON array
        {...}               — single fact object (some models drop the array
                              wrapper when they only have one fact)
        {...}, {...}, {...} — bare comma-separated objects (gpt-4o-mini does
                              this fairly often; handled by _safe_parse_json)
    """
    parsed = _safe_parse_json(text, default=[])
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _safe_parse_json(text: str, *, default):
    """Strip markdown fences, tolerate junk before/after, return default on failure."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ``` or ```json fence
        text = text.lstrip("`")
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # 1) Direct parse — handles well-formed input.
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2) Slice from first opener to last closer — handles prose preambles
    # like "Here are the facts: [...]".
    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                pass

    # 3) Bare comma-separated objects: `{"a":1},{"b":2}`. Smaller OpenAI
    # models (notably gpt-4o-mini) drop the outer `[ ]` of an array but
    # keep the items well-formed. Wrap and retry.
    stripped = text.strip().rstrip(",")
    if stripped.startswith("{") and stripped.endswith("}") and "}," in stripped:
        try:
            return json.loads("[" + stripped + "]")
        except (json.JSONDecodeError, ValueError):
            pass

    return default
