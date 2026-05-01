# ADR-0012: Bootstrap Dedup Strategy — Two-Tier Vector + LLM

**Status:** Accepted  
**Date:** 2026-04-27

## Context

The bootstrap process (`ai-memory bootstrap`) reads structured markdown files and inserts facts as notes. On re-runs or across multiple source files, the same fact can appear multiple times in different phrasing. A hash-based dedup (`seen_hashes`) only catches exact-text duplicates within a single file pass. Cross-file semantic duplicates (e.g. "Igor is a Principal Architect" and "Igor's role: Principal Architect") pass through and pollute the notes index.

## Decision

Use a **two-tier dedup strategy** on each candidate note before insertion:

**Constants:**
- `BOOTSTRAP_DEDUP_DIST = 0.30` — hard vector threshold (distance < 0.30 → skip, no LLM)
- `BOOTSTRAP_DEDUP_MIDBAND = 0.40` — LLM check band upper bound

**Logic per candidate:**
1. Embed the candidate fact.
2. Search for the nearest existing note vector (`k=1`).
3. **Distance < 0.30**: Near-certain duplicate. Skip silently. No LLM call.
4. **0.30 ≤ distance < 0.40**: Ambiguous. Call LLM with both texts. LLM returns DUPLICATE or KEEP. If DUPLICATE: skip. If KEEP (or on LLM error): insert.
5. **Distance ≥ 0.40**: Clearly distinct. Insert without LLM call.

The LLM fallback-to-keep on error ensures a failed API call never silently drops data.

Thresholds were calibrated by computing pairwise cosine distances across the full bootstrap corpus (92 notes, 21 near-pairs identified). True duplicates clustered at 0.15–0.28; distinct-but-related facts clustered at 0.30–0.55.

## Consequences

- Cross-file semantic duplicates are caught without exact-text matching.
- The hard threshold (< 0.30) avoids LLM cost for obvious duplicates.
- The mid-band LLM check handles edge cases where vector distance alone is ambiguous.
- LLM errors are safe: they default to insert (no data loss), not skip.
- The thresholds are empirically tuned for `text-embedding-3-small` on this corpus. If the embedding model changes (ADR-0003), both thresholds must be re-evaluated.
- `BootstrapResult` carries `notes_llm_deduped` and `notes_skipped` counters for observability.

## Alternatives considered

- **Hash dedup only**: Fast, zero cost. Misses cross-file semantic duplicates. Rejected.
- **Vector threshold only (no LLM mid-band)**: A single threshold either misses near-duplicates (too high) or drops distinct facts (too low). The mid-band LLM check handles the ambiguous zone. Rejected as sole mechanism.
- **LLM check for all candidates**: Correct but expensive — O(n) LLM calls for a large bootstrap corpus. The two-tier approach reduces LLM calls to only the ambiguous mid-band. Rejected.
- **Phase 4 dedup only (no bootstrap dedup)**: Relies on the dream cycle to clean up; bootstrap-inserted duplicates persist until the next dream run. Rejected — bootstrap should be clean on insertion.
