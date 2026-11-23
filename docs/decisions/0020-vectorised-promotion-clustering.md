# ADR-0020: Vectorised Phase 5 Clustering and Dream Progress Logging

**Status:** Accepted
**Date:** 2026-11-23
**Related:** ADR-0013 (dream resilience), ADR-0019 (incremental extraction)
**Implementation:** `dreaming._close_pairs` (`_close_pairs_numpy` / `_close_pairs_python`),
`dreaming._Progress`, optional extra `ai-memory[fast]` (numpy).
Tests: `tests/test_phase5_clustering.py`.

## Context

Phase 5 groups every valid, un-promoted note into clusters by pairwise L2
distance (`PROMOTION_CLUSTER_DIST`). The pair search was a pure-Python double
loop. Each distance walked 1,536 floats in the interpreter, so a pass at about
14k notes did roughly 100M distance calls on one core: about 1.8 h on an idle
machine, measured on the live corpus, and more while the daemon is busy. The
time grows with the square of the note count. The code comment said "fine up to
a few thousand notes", and the corpus has outgrown that.

Separately, the daemon log gave no sign of progress inside a pass. During the
clustering step there are no API calls and no log lines for hours, so a slow
pass looks the same as a hung one.

## Decision

1. **Vectorise the pair search with numpy when it is installed.**
   - Embeddings are stacked into a float64 matrix and compared in blocks of 512
     rows using `|a|² + |b|² − 2a·b` (one matrix multiply per block, about 57 MB
     at 14k notes).
   - Candidates within `threshold + 1e-6` are rechecked with a direct difference,
     so pairs near the threshold land on the same side as the pure-Python path.
   - Embeddings with a missing vector or a different dimension never pair, the
     same as before.
   - Union-find components don't depend on the order pairs are found in, so both
     paths give the same clusters.
2. **Keep numpy optional** (`pip install -e .[fast]`). Without it the pure-Python
   loop runs unchanged, and the progress label names which backend ran.
3. **Log throttled progress for every long phase** with `_Progress`, at INFO, to
   the daemon's stdio log:
   - Phase 3 episodes, Phase 4 candidate facts, Phase 5 clustering pairs, and
     Phase 5 promotion verdicts.
   - Each logs a start line, at most one line a minute with count, percent,
     elapsed time and a linear ETA, and a finish line with the total time.

## Measured

Live corpus, read-only, 13,689 notes:

| Path | Time |
|---|---|
| numpy, full corpus | 2.8 s (1,674 close pairs) |
| pure Python, 800-note sample | 22.7 s |
| numpy, same 800-note sample | 0.047 s, identical pairs |
| pure Python, full corpus (extrapolated) | ~1.8 h |

## Portability

`porting-to-csharp.md` rules out numpy **as a hard dependency**. Making it
optional keeps that rule. The operation is a blocked matrix product and a
threshold, so the port has a clean equivalent:

- `System.Numerics.Tensors.TensorPrimitives.Dot` / `SquaredDistance`, or
- a plain SIMD loop over `float[]`.

Nothing in the algorithm depends on numpy.

## Consequences

- Phase 5 no longer dominates pass time. Its cost is mainly the LLM promotion
  verdicts.
- The work is still O(n²) comparisons, just about 1,000× cheaper each.
  Incremental clustering (compare only new notes against existing clusters)
  stays the long-term fix if the corpus reaches the hundreds of thousands.
- An install without `[fast]` silently keeps the slow path, but the
  `Phase 5 clustering pairs (pure-python)` progress line makes it visible.
- The ETA is linear in items done. Phase 3 episodes vary a lot in size, so its
  ETA is a rough guide only.
