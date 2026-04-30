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
from typing import TYPE_CHECKING, Iterable, List, Tuple

from uuid import uuid4

from ai_memory.config import DreamConfig
from ai_memory.core.models import DreamLog, Note, Profile
from ai_memory.embeddings.interface import Embedder
from ai_memory.llm.interface import Llm, Message
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
DUPLICATE_DIST_BELOW = 0.10   # essentially identical -> dedup without LLM
UNRELATED_DIST_ABOVE = 0.55   # clearly unrelated -> insert without LLM
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


@dataclass
class DreamReport:
    log_id: str
    episodes_processed: int
    notes_added: int
    notes_invalidated: int
    notes_promoted_to_profile: int
    notes_pruned: int
    journal: str


@dataclass
class _CandidateFact:
    """A fact extracted by Phase 3 that hasn't yet been integrated."""

    text: str
    tags: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    source_episode_id: str = ""


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
    "about a user, and one or more EXISTING facts already stored. For each "
    "existing fact, classify the relationship: "
    "DUPLICATE (same fact, no new info), "
    "CONTRADICTS (directly conflicts), "
    "COMPLEMENTS (related, adds context, both can stay), "
    "UNRELATED (different topic). "
    'Output JSON: [{"existing_id": "...", "verdict": "DUPLICATE|CONTRADICTS|COMPLEMENTS|UNRELATED", "reason": "..."}]. '
    "Output ONLY the JSON array, no prose."
)

PROMOTION_SYSTEM = (
    "You are a memory consolidation worker. You receive a CLUSTER of related "
    "notes about a user. If the cluster represents a stable, durable fact "
    "worth promoting to the user's persistent profile, output a single canonical "
    "key/value pair as JSON. Otherwise output null fields. "
    'Format: {"key": "snake_case_key", "value": "concise value or null", "rationale": "why"}. '
    "Examples of good keys: preferred_database, communication_style, primary_language, "
    "current_company, dev_environment. Output ONLY the JSON object."
)


# --- Public entry point ----------------------------------------------------


def dream(
    *,
    store: "MemoryStore",
    raw: RawTranscriptStore,
    embedder: Embedder,
    llm: Llm,
    config: DreamConfig,
    request: DreamRequest,
    now: str | None = None,
) -> DreamReport:
    """Run a single dream pass. Inserts a DreamLog row and returns a report."""
    now = now if now is not None else now_iso()
    log_id = str(uuid4())

    log = DreamLog(id=log_id, started_at=now, trigger=request.trigger)
    store.insert_dream_log(log)

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
    journal.append(
        f"Phase 1+2: {len(episodes)} unconsolidated episodes (scanned "
        f"{len(candidate_episodes)} total)."
    )

    # --- Phase 3: consolidate each episode -----------------------------
    # Load entity vocab once before the loop so every extract call can
    # steer the LLM toward reusing known entity slugs.
    entity_vocab: list[str] = store.list_entity_vocab()
    extract_system = _build_extract_system(entity_vocab)

    candidate_facts: List[_CandidateFact] = []
    for ep in episodes:
        turns = list(store.get_turns_for_episode(ep.id))
        if not turns:
            continue
        transcript = _render_transcript(raw, turns)

        # Summary: single-shot on the whole transcript. Synthesising one
        # paragraph is well within gpt-4o-mini's effective range even at
        # 60k+ tokens; only enumerative tasks (extract) need chunking.
        summary_user_msg = (
            "Summarise the transcript below according to your instructions. "
            "The conversation has already concluded — do not continue it.\n\n"
            + transcript
        )
        summary_completion = llm.complete(
            system=SUMMARY_SYSTEM,
            messages=[Message(role="user", content=summary_user_msg)],
            max_tokens=1200,
        )
        tokens_used += summary_completion.input_tokens + summary_completion.output_tokens

        # Extract: chunked. For short episodes _chunk_turns returns a single
        # window so behaviour is identical to the old single-shot path.
        turn_chunks = _chunk_turns(turns)
        all_facts: List[dict] = []
        chunk_finish_reasons: List[str] = []
        chunk_sizes: List[int] = []
        any_extract_text = False
        for chunk_turns in turn_chunks:
            chunk_transcript = _render_transcript(raw, chunk_turns)
            extract_user_msg = (
                "Extract atomic facts from the transcript chunk below "
                "according to your instructions. The conversation has "
                "already concluded — do not continue it. Output only the "
                "JSON array.\n\n"
                + chunk_transcript
            )
            extract_completion = llm.complete(
                system=extract_system,
                messages=[Message(role="user", content=extract_user_msg)],
                max_tokens=4000,
            )
            tokens_used += extract_completion.input_tokens + extract_completion.output_tokens
            chunk_facts = _safe_parse_facts(extract_completion.text)
            all_facts.extend(chunk_facts)
            chunk_finish_reasons.append(extract_completion.finish_reason)
            chunk_sizes.append(len(chunk_facts))
            if extract_completion.text.strip():
                any_extract_text = True

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

        for fact in all_facts:
            text = (fact.get("text") or "").strip()
            if not text:
                continue
            # Domain tags: validate against controlled vocabulary, lowercase.
            # Drop any tag the LLM invented outside the allowed set; always
            # append the episode source as a provenance tag.
            raw_tags = [str(t).lower() for t in (fact.get("tags") or [])]
            tags = [t for t in raw_tags if t in DOMAIN_TAGS] + [ep.source]
            # Entities: slug-normalise and fuzzy-match against known vocab.
            raw_entities = [str(e) for e in (fact.get("entities") or [])]
            entities = [
                norm for e in raw_entities
                if (norm := _normalise_entity(e, entity_vocab))
            ]
            candidate_facts.append(
                _CandidateFact(text=text, tags=tags, entities=entities, source_episode_id=ep.id)
            )

        ep_candidates = len([f for f in candidate_facts if f.source_episode_id == ep.id])
        journal.append(
            f"  Episode {ep.id}: turns={len(turns)} chunks={len(turn_chunks)} "
            f"transcript_chars={len(transcript)} summary={len(ep.summary)} chars "
            f"(finish={summary_completion.finish_reason}"
            f"{', refusal=' + summary_completion.refusal[:120] if summary_completion.refusal else ''}) "
            f"candidates={ep_candidates} (per-chunk={chunk_sizes}, "
            f"finishes={chunk_finish_reasons})"
        )
        if ep_candidates == 0 and any_extract_text:
            journal.append(
                "    note: chunks produced text but parser found 0 facts — "
                "investigate response format"
            )

    # --- Phase 4: integrate candidate facts ----------------------------
    integrated, phase4_tokens = _phase4_integrate(
        candidates=candidate_facts,
        store=store,
        embedder=embedder,
        llm=llm,
        now=now,
        journal=journal,
    )
    notes_added += integrated["added"]
    notes_invalidated += integrated["invalidated"]
    tokens_used += phase4_tokens

    # --- Phase 5: promote recurring patterns ---------------------------
    promoted, phase5_tokens = _phase5_promote(
        store=store,
        llm=llm,
        config=config,
        now=now,
        journal=journal,
    )
    tokens_used += phase5_tokens

    # --- Phase 6: decay and prune --------------------------------------
    pruned = _phase6_prune(store=store, config=config, now=now, journal=journal)

    # --- Phase 7: journal ----------------------------------------------
    journal.append(f"LLM tokens used this pass: {tokens_used} ({llm.model_id})")
    log.ended_at = now_iso()
    log.episodes_processed = len(episodes)
    log.notes_added = notes_added
    log.notes_invalidated = notes_invalidated
    log.notes_promoted_to_profile = promoted
    log.notes_pruned = pruned
    log.llm_tokens_used = tokens_used
    log.journal = "\n".join(journal)
    store.update_dream_log(log)

    return DreamReport(
        log_id=log_id,
        episodes_processed=len(episodes),
        notes_added=notes_added,
        notes_invalidated=notes_invalidated,
        notes_promoted_to_profile=promoted,
        notes_pruned=pruned,
        journal=log.journal,
    )


# --- Phase 4 ---------------------------------------------------------------


def _phase4_integrate(
    *,
    candidates: List[_CandidateFact],
    store: "MemoryStore",
    embedder: Embedder,
    llm: Llm,
    now: str,
    journal: List[str],
) -> Tuple[dict, int]:
    """Integrate candidate facts: dedup, contradict-resolve, insert.

    Returns ({added, invalidated, deduped}, llm_tokens_used).
    """
    added = 0
    invalidated = 0
    deduped = 0
    tokens = 0

    if not candidates:
        journal.append("Phase 4: no candidate facts to integrate.")
        return {"added": added, "invalidated": invalidated, "deduped": deduped}, tokens

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
            verdict = (v.get("verdict") or "").upper()
            existing_id = v.get("existing_id") or ""
            if verdict == "DUPLICATE":
                store.bump_note_access(existing_id, now)
                deduped += 1
                action_taken = True
                break  # done with this candidate
            if verdict == "CONTRADICTS":
                # Insert the new fact, then invalidate the old one and link them.
                new_id = _insert_candidate(
                    store, cand, cand_emb, embedder, now, contradicts=[existing_id]
                )
                store.invalidate_note(existing_id, when=now, superseded_by=new_id)
                invalidated += 1
                added += 1
                action_taken = True
                break

        if not action_taken:
            # Either UNRELATED or COMPLEMENTS for everything — insert.
            _insert_candidate(store, cand, cand_emb, embedder, now)
            added += 1

    journal.append(
        f"Phase 4: integrated {len(candidates)} candidates -> "
        f"added={added}, invalidated={invalidated}, deduped={deduped}"
    )
    return {"added": added, "invalidated": invalidated, "deduped": deduped}, tokens


def _classify_candidate_vs_neighbours(
    *, llm: Llm, candidate_text: str, neighbours: List[Tuple[Note, float]]
) -> Tuple[List[dict], int]:
    """One LLM call that classifies the candidate against all K neighbours."""
    payload = {
        "candidate": candidate_text,
        "existing": [{"id": n.id, "text": n.text} for n, _ in neighbours],
    }
    completion = llm.complete(
        system=INTEGRATE_VERDICT_SYSTEM,
        messages=[Message(role="user", content=json.dumps(payload, ensure_ascii=False))],
        max_tokens=800,
    )
    verdicts = _safe_parse_json(completion.text, default=[])
    if not isinstance(verdicts, list):
        verdicts = []
    return verdicts, completion.input_tokens + completion.output_tokens


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
        id=str(uuid4()),
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
        f"Phase 5: {len(valid_notes)} notes clustered at dist≤{PROMOTION_CLUSTER_DIST} "
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
        completion = llm.complete(
            system=PROMOTION_SYSTEM,
            messages=[
                Message(role="user", content=json.dumps(cluster_payload, ensure_ascii=False))
            ],
            max_tokens=300,
        )
        tokens += completion.input_tokens + completion.output_tokens

        verdict = _safe_parse_json(completion.text, default={})
        key = (verdict.get("key") or "").strip() if isinstance(verdict, dict) else ""
        value = (verdict.get("value") or "").strip() if isinstance(verdict, dict) else ""

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
