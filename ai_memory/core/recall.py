"""
Recall pipeline: hybrid search (BM25 + vector) fused with Reciprocal Rank Fusion,
soft-boosted by recency.

Three depths:
    - "fast"     -> Tier 0 (profile) + top-k Tier 1 notes
    - "deep"     -> larger Tier 1 + Tier 2 sweep, scored together
    - "verbatim" -> deep + then fetch verbatim turns for the top hits

RRF formula:
    score(doc) = sum over rankers of  1 / (k_rrf + rank_in_ranker(doc))

We use k_rrf = 60 (the value Supermemory found generalises well).

Recency boost:
    boost(doc) = exp(- age_days / half_life_days)
    final     = rrf_score * boost

Recency boost is applied only at the merge stage (not inside each ranker)
so the ranker-internal scores remain comparable.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_memory.config import RecallConfig
from ai_memory.core.models import RecallHit
from ai_memory.embeddings.interface import Embedder
from ai_memory.storage.raw_files import RawTranscriptStore

if TYPE_CHECKING:
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


def recall(
    *,
    store: "MemoryStore",
    raw: RawTranscriptStore,
    embedder: Embedder,
    config: RecallConfig,
    request: RecallRequest,
    now: int | None = None,
) -> list[RecallHit]:
    """Run the recall pipeline. Returns a ranked list of hits."""
    now = int(now if now is not None else time.time())
    t0 = time.monotonic()

    # 1. Embed the query.
    #    NOTE: BM25 was previously overlapped with embedding via ThreadPoolExecutor,
    #    but that caused deadlocks when multiple recall threads accessed the shared
    #    sqlite3 connection concurrently (sqlite3 connections are not thread-safe).
    #    The ~5 ms BM25 saving is not worth the complexity; keep it sequential.
    _t_embed = time.monotonic()
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

    # 3. RRF fuse.
    vector_ids = [n.id for n, _ in vector_hits]
    bm25_ids = [n.id for n, _ in bm25_hits]
    fused = reciprocal_rank_fusion([vector_ids, bm25_ids], k_rrf=config.rrf_k)

    # 4. Resolve back to Note objects (one of the rankers will have it).
    notes_by_id = {n.id: n for n, _ in vector_hits}
    notes_by_id.update({n.id: n for n, _ in bm25_hits})

    # 5. Apply recency boost.
    half_life_seconds = max(1, config.recency_half_life_days * 86400)
    boosted: list[tuple[str, float]] = []
    for note_id, base_score in fused.items():
        note = notes_by_id.get(note_id)
        if note is None:
            continue
        age = max(0, now - note.ingested_at)
        boost = math.exp(-age / half_life_seconds)
        boosted.append((note_id, base_score * boost))
    boosted.sort(key=lambda pair: pair[1], reverse=True)

    # 6. Take top k.
    top_k = boosted[: request.k]

    note_hits = [
        RecallHit(
            item_type="note",
            id=note_id,
            text=notes_by_id[note_id].text,
            score=score,
            provenance={
                "tags": notes_by_id[note_id].tags,
                "source_episode_ids": notes_by_id[note_id].source_episode_ids,
                "valid_from": notes_by_id[note_id].valid_from,
            },
        )
        for note_id, score in top_k
    ]

    # 7. Always also pull episode summaries.
    #
    # We tried gating this on a "weak hits" heuristic, but with text-embedding-3-small
    # over a small corpus, scores cluster in a narrow 0.014-0.016 band regardless
    # of how relevant the hit actually is. Absolute thresholds and spread-based
    # signals both misfire. So instead we always pay the one extra DB query and
    # include episodes; the merged ranking surfaces whichever tier had the better
    # match. The `depth` parameter now only controls whether we *also* expand
    # into raw verbatim turns (depth=verbatim), which is genuinely expensive.
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
    episode_hits = [
        RecallHit(
            item_type="episode",
            id=ep.id,
            text=ep.summary,
            score=1.0 / (1.0 + dist),  # convert distance -> score
            provenance={"source": ep.source, "started_at": ep.started_at},
        )
        for ep, dist in episode_hits_raw
        if dist <= config.vector_distance_floor
    ]
    note_hits.extend(episode_hits)

    # Final relevance gate: drop anything below the absolute score floor.
    if config.final_score_floor > 0:
        note_hits = [h for h in note_hits if h.score >= config.final_score_floor]

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
        "recall done hits=%d total=%.0fms  (embed=%.0fms bm25_wait=%.0fms vec=%.0fms ep=%.0fms)",
        len(note_hits),
        (time.monotonic() - t0) * 1000,
        _embed_ms, _bm25_ms, _vec_ms, _ep_ms
    )
    return note_hits
