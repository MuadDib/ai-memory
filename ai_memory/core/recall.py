"""
Recall pipeline: hybrid search (BM25 + vector, over both notes AND episodes)
fused with Reciprocal Rank Fusion, soft-boosted by recency.

Three depths:
    - "fast"     -> profile + top-k notes
    - "deep"     -> larger notes + episode sweep, scored together
    - "verbatim" -> deep + then fetch verbatim turns for the top hits

RRF formula:
    score(doc) = sum over rankers of  1 / (k_rrf + rank_in_ranker(doc))

We use k_rrf = 60 (the value Supermemory found generalises well).

Notes and episodes are fused through this SAME formula, as parallel rankers
(note-vector, note-BM25, episode-vector), so an item's TYPE never decides the
final ranking by scale artifact — only its RANK across the signals that found
it does. (An earlier version scored episodes via 1/(1+distance), landing on a
~30x larger scale than notes' RRF*recency scores; that let weakly-related
episodes mathematically bury well-corroborated notes once the corpus grew —
see the regression test in test_recall_merge.py for the concrete scenario.)

Recency boost:
    boost(doc) = exp(- age_days / half_life_days)
    final     = rrf_score * boost

Recency boost is applied only at the merge stage (not inside each ranker)
so the ranker-internal scores remain comparable.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_memory.config import RecallConfig
from ai_memory.core.models import RecallHit
from ai_memory.embeddings.interface import Embedder
from ai_memory.llm.interface import Message
from ai_memory.storage.raw_files import RawTranscriptStore
from ai_memory.timestamps import iso_to_dt, now_iso

if TYPE_CHECKING:
    from ai_memory.llm.interface import Llm
    from ai_memory.storage.interface import MemoryStore


logger = logging.getLogger(__name__)


Depth = str  # "fast" | "deep" | "verbatim"


@dataclass(frozen=True)
class RecallRequest:
    """Input to `recall`."""

    query: str
    depth: Depth = "fast"
    k: int = 8


def reciprocal_rank_fusion(
    rankers: list[list[str]], k_rrf: int
) -> dict[str, float]:
    """Fuse ranked lists into a single score per document id.

    `rankers` is a list of ranked id-lists, each in best-to-worst order.
    Returns {id: score} where higher = better.
    """
    scores: dict[str, float] = {}
    for ranked in rankers:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    return scores


RERANK_SYSTEM = (
    "You are a search reranker (a cross-encoder). Given a QUERY and a numbered "
    "list of candidate memory facts, return the candidate indices ordered from "
    "MOST to LEAST relevant for ANSWERING the query. Judge whether each candidate "
    "actually answers or directly informs the query — not mere topical overlap. "
    "A specific, direct answer outranks a general or tangential mention. Include "
    "every index exactly once.\n"
    'Output JSON only: {"order": [<index>, <index>, ...]}.'
)


def _llm_rerank(
    *, llm: "Llm", query: str, hits: list[RecallHit], top_n: int
) -> list[RecallHit]:
    """LLM cross-encoder rerank of the top `top_n` hits (the rest are left in place
    after them). Returns a reordered hits list. Fail-safe: returns the input
    UNCHANGED on any error or invalid output — reranking must never drop a hit."""
    pool = hits[:top_n]
    if len(pool) < 2:
        return hits
    payload = {
        "query": query,
        "candidates": [{"i": i, "text": h.text[:400]} for i, h in enumerate(pool)],
    }
    try:
        completion = llm.complete(
            system=RERANK_SYSTEM,
            messages=[Message(role="user", content=json.dumps(payload, ensure_ascii=False))],
            max_tokens=256,
        )
        text = completion.text
        s, e = text.find("{"), text.rfind("}")
        order = json.loads(text[s:e + 1]).get("order") if s != -1 and e != -1 else None
        if not isinstance(order, list):
            return hits
        reordered: list[RecallHit] = []
        seen: set[int] = set()
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < len(pool) and idx not in seen:
                reordered.append(pool[idx])
                seen.add(idx)
        for i, h in enumerate(pool):  # append any indices the model omitted
            if i not in seen:
                reordered.append(h)
        return reordered + hits[top_n:]
    except Exception:  # malformed JSON / provider error — keep the fused order
        logger.warning("rerank failed; falling back to fused order", exc_info=False)
        return hits


HYDE_SYSTEM = (
    "You help a memory search engine. Given a QUESTION, write ONE short, plausible "
    "declarative sentence that would directly ANSWER it — phrased as a stored fact, "
    "not a question. Invent reasonable specifics if needed; this text is only used "
    "to steer vector search toward the answer's phrasing and is never shown to "
    "anyone. Output the single sentence only, no preamble."
)


def _hyde_embedding(
    *, llm: "Llm", embedder: Embedder, query: str
) -> list[float] | None:
    """HyDE (#2): embed a hypothetical ANSWER to the query rather than the question,
    bridging the interrogative-query / declarative-fact gap. Returns the embedding,
    or None on any failure so the caller falls back to the plain query embedding."""
    try:
        completion = llm.complete(
            system=HYDE_SYSTEM,
            messages=[Message(role="user", content=query)],
            max_tokens=80,
        )
        hypo = (completion.text or "").strip()
        if not hypo:
            return None
        [emb] = embedder.embed([hypo])
        return emb
    except Exception:
        logger.warning("HyDE failed; falling back to plain query embedding", exc_info=False)
        return None


def _conviction_boost(*, access_count: int, source_episodes: int, weight: float) -> float:
    """Recall ranking boost (#3): reward facts corroborated by multiple source
    episodes and reinforced by past recall. A gentle multiplicative nudge >= 1.0
    (log-scaled so a handful of corroborations helps without one popular note
    dominating). weight=0 disables it (returns 1.0)."""
    if weight <= 0:
        return 1.0
    signal = max(0, access_count) + max(0, source_episodes)
    return 1.0 + weight * math.log1p(signal)


def recall(
    *,
    store: "MemoryStore",
    raw: RawTranscriptStore,
    embedder: Embedder,
    config: RecallConfig,
    request: RecallRequest,
    llm: "Llm | None" = None,
    now: str | None = None,
) -> list[RecallHit]:
    """Run the recall pipeline. Returns a ranked list of hits."""
    now = now if now is not None else now_iso()
    t0 = time.monotonic()

    # 1. Embed the query.
    #    NOTE: BM25 was previously overlapped with embedding via ThreadPoolExecutor,
    #    but that caused deadlocks when multiple recall threads accessed the shared
    #    sqlite3 connection concurrently (sqlite3 connections are not thread-safe).
    #    The ~5 ms BM25 saving is not worth the complexity; keep it sequential.
    #    With HyDE (#2, off by default) we vector-search on a hypothetical ANSWER's
    #    embedding instead of the question's; BM25 below still uses the raw query.
    _t_embed = time.monotonic()
    query_embedding = None
    if config.hyde_enabled and llm is not None:
        query_embedding = _hyde_embedding(llm=llm, embedder=embedder, query=request.query)
    if query_embedding is None:
        [query_embedding] = embedder.embed([request.query])
    _embed_ms = (time.monotonic() - _t_embed) * 1000

    # 2. BM25 (sequential, safe — single thread per recall invocation).
    _t_bm25 = time.monotonic()
    bm25_hits = store.search_notes_bm25(request.query, config.bm25_candidates)
    _bm25_ms = (time.monotonic() - _t_bm25) * 1000

    # 3. Vector search (needs the embedding from step 1).
    _t_vec = time.monotonic()
    vector_hits_raw = store.search_notes_vector(
        query_embedding, k=config.vector_candidates, only_valid=True
    )
    _vec_ms = (time.monotonic() - _t_vec) * 1000

    # Diagnostic: log the live distance distribution so the floor can be
    # tuned from data rather than guesswork. Cheap; runs every recall.
    if vector_hits_raw:
        dists = [d for _, d in vector_hits_raw]
        logger.info(
            "recall query=%r vector_candidates=%d dist_min=%.3f dist_p50=%.3f "
            "dist_max=%.3f floor=%.3f bm25_candidates=%d "
            "t_embed=%.0fms t_bm25=%.0fms t_vec=%.0fms",
            request.query,
            len(vector_hits_raw),
            min(dists),
            sorted(dists)[len(dists) // 2],
            max(dists),
            config.vector_distance_floor,
            len(bm25_hits),
            _embed_ms, _bm25_ms, _vec_ms,
        )
    else:
        logger.info(
            "recall query=%r vector_candidates=0 bm25_candidates=%d "
            "t_embed=%.0fms t_bm25=%.0fms t_vec=%.0fms",
            request.query, len(bm25_hits),
            _embed_ms, _bm25_ms, _vec_ms,
        )

    # Drop semantic-noise candidates: anything past the distance floor is
    # effectively unrelated even though the vector index returned it. BM25
    # hits are NOT filtered — a token match is a deterministic signal.
    vector_hits = [
        (note, dist) for note, dist in vector_hits_raw
        if dist <= config.vector_distance_floor
    ]

    # 3. Pull episode candidates now — they need to be in hand *before* fusing
    #    so they can be fused as a ranker rather than bolted on afterwards (see below).
    #
    # We tried gating this on a "weak hits" heuristic, but with text-embedding-3-small
    # over a small corpus, scores cluster in a narrow 0.014-0.016 band regardless
    # of how relevant the hit actually is. Absolute thresholds and spread-based
    # signals both misfire. So instead we always pay the one extra DB query and
    # include episodes; the merged ranking surfaces whichever tier had the better
    # match. The `depth` parameter only controls whether we *also* expand into
    # raw verbatim turns (depth=verbatim), which is genuinely expensive.
    _t_ep = time.monotonic()
    episode_hits_raw = store.search_episodes_vector(query_embedding, k=config.deep_k)
    _ep_ms = (time.monotonic() - _t_ep) * 1000
    if episode_hits_raw:
        ep_dists = [d for _, d in episode_hits_raw]
        logger.info(
            "recall episode_candidates=%d dist_min=%.3f dist_p50=%.3f dist_max=%.3f t_ep=%.0fms",
            len(ep_dists),
            min(ep_dists),
            sorted(ep_dists)[len(ep_dists) // 2],
            max(ep_dists),
            _ep_ms,
        )
    episodes_filtered = [
        (ep, dist) for ep, dist in episode_hits_raw
        if dist <= config.vector_distance_floor
    ]

    # 4. RRF fuse — notes (vector + BM25) and episodes (vector) all feed the SAME
    #    fusion as parallel rankers, landing on the SAME score scale.
    #
    #    Previously episodes were scored via 1/(1+distance) (~0.4-0.9) while notes
    #    were scored via RRF*recency (~0.01-0.03) — a ~30x scale gap. Any episode
    #    with a merely-OK vector match would mathematically bury a note that was
    #    the actual best answer (even one corroborated by BOTH BM25 and vector).
    #    That gap was always there but didn't bite until the 2026-06-07 backfill
    #    grew the note corpus enough to expose it (eval recall@k 96.3% -> 70.4%).
    #    Folding episodes into the fusion as a third ranker means an item's TYPE
    #    never decides the outcome — its RANK across the signals that found it does.
    vector_ids = [n.id for n, _ in vector_hits]
    bm25_ids = [n.id for n, _ in bm25_hits]
    episode_ids = [ep.id for ep, _ in episodes_filtered]
    fused = reciprocal_rank_fusion([vector_ids, bm25_ids, episode_ids], k_rrf=config.rrf_k)

    # 5. Resolve ids back to source objects (note and episode id spaces don't collide — both ULIDs).
    notes_by_id = {n.id: n for n, _ in vector_hits}
    notes_by_id.update({n.id: n for n, _ in bm25_hits})
    episodes_by_id = {ep.id: ep for ep, _ in episodes_filtered}

    # 6. Apply recency boost uniformly (notes anchor on ingested_at, episodes on started_at).
    half_life_seconds = max(1, config.recency_half_life_days * 86400)
    boosted: list[tuple[str, float]] = []
    for item_id, base_score in fused.items():
        if item_id in notes_by_id:
            note = notes_by_id[item_id]
            anchor = note.ingested_at
            # #3 conviction boost: corroboration (source episodes) + reinforcement
            # (access_count). Notes only; episodes have no comparable signal.
            conviction = _conviction_boost(
                access_count=note.access_count,
                source_episodes=len(note.source_episode_ids),
                weight=config.conviction_weight,
            )
        elif item_id in episodes_by_id:
            anchor = episodes_by_id[item_id].started_at
            conviction = 1.0
        else:
            continue
        age = max(0, (iso_to_dt(now) - iso_to_dt(anchor)).total_seconds())
        boost = math.exp(-age / half_life_seconds)
        boosted.append((item_id, base_score * boost * conviction))
    boosted.sort(key=lambda pair: pair[1], reverse=True)

    # 7. Take a candidate pool. When rerank is enabled we keep a WIDER pool
    #    (`rerank_candidates`) so the LLM cross-encoder can pull a just-below-k
    #    answer up into the top k; otherwise the pool is exactly k.
    rerank_on = config.rerank_enabled and llm is not None
    pool_n = max(request.k, config.rerank_candidates) if rerank_on else request.k
    top_k = boosted[:pool_n]
    note_hits: list[RecallHit] = []
    for item_id, score in top_k:
        if item_id in notes_by_id:
            note = notes_by_id[item_id]
            note_hits.append(
                RecallHit(
                    item_type="note",
                    id=item_id,
                    text=note.text,
                    score=score,
                    provenance={
                        "tags": note.tags,
                        "source_episode_ids": note.source_episode_ids,
                        "valid_from": note.valid_from,
                    },
                )
            )
        else:
            ep = episodes_by_id[item_id]
            note_hits.append(
                RecallHit(
                    item_type="episode",
                    id=item_id,
                    text=ep.summary,
                    score=score,
                    provenance={"source": ep.source, "started_at": ep.started_at},
                )
            )

    # Final relevance gate: drop anything below the absolute score floor (applied
    # on the fused*recency score, before any rerank reorders the survivors).
    if config.final_score_floor > 0:
        note_hits = [h for h in note_hits if h.score >= config.final_score_floor]

    # 7b. Cross-encoder rerank the wider pool, then cut to k (#1, off by default).
    if rerank_on:
        note_hits = _llm_rerank(
            llm=llm, query=request.query, hits=note_hits, top_n=pool_n
        )
    note_hits = note_hits[: request.k]

    # 8. For "verbatim" — pull the underlying turns for the top episodes.
    if request.depth == "verbatim":
        verbatim_hits: list[RecallHit] = []
        for hit in note_hits:
            if hit.item_type != "episode":
                continue
            turns = store.get_turns_for_episode(hit.id)
            for turn in turns:
                payload = raw.read_turn(turn.raw_file, turn.byte_offset, turn.byte_length)
                verbatim_hits.append(
                    RecallHit(
                        item_type="turn",
                        id=turn.id,
                        text=payload.get("text", ""),
                        score=hit.score,  # inherit episode score
                        provenance={
                            "episode_id": hit.id,
                            "role": turn.role,
                            "ts": turn.ts,
                        },
                    )
                )
        note_hits.extend(verbatim_hits)

    logger.info(
        "recall done hits=%d total=%.0fms  (embed=%.0fms bm25=%.0fms vec=%.0fms ep=%.0fms)",
        len(note_hits),
        (time.monotonic() - t0) * 1000,
        _embed_ms, _bm25_ms, _vec_ms, _ep_ms
    )
    return note_hits
