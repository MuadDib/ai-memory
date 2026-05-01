# ADR-0002: SQLite + sqlite-vec as the Primary Store

**Status:** Accepted  
**Date:** 2026-04-27

## Context

The system targets a local-first, single-user deployment on Windows (and eventually Linux/Mac). It needs: relational storage, BM25 full-text search, and vector similarity search, all in a single process with no server.

## Decision

Use SQLite (via Python's `sqlite3`) with two extensions:
- **FTS5** (built-in): BM25 keyword search over note text and entities.
- **sqlite-vec**: ANN vector search over 1536-dim `float32` embeddings.

The database lives at `%LOCALAPPDATA%\ai-memory\memory.db`. Schema version is tracked in a `schema_version` table; migrations are appended to `SCHEMA_SQL` in `sqlite_store.py` and applied on every `initialise()` call.

## Consequences

- Zero external dependencies for storage — no Postgres, no Qdrant, no Redis.
- `sqlite-vec` must be installed as a Python package and loaded at runtime via `con.enable_load_extension(True)`. Any code that opens the DB for vector ops must call `sqlite_vec.load(con)`.
- Switching to Postgres (ADR planned for Phase 3 multi-device) requires implementing the `MemoryStore` interface (`storage/interface.py`) — the interface was designed with this swap in mind.
- Embedding dimension is encoded in the schema (`vec_size`). Changing the embedding model (ADR-0003) requires a schema migration and re-embedding all notes.

## Alternatives considered

- **Postgres + pgvector**: Better multi-user scaling, more SQL power. Ruled out for local Phase 1 (requires a running server). Planned for Phase 3.
- **Qdrant / Weaviate**: Excellent vector search, but adds an external server and breaks the single-file deployment story. Rejected for Phase 1.
- **ChromaDB**: Simpler API, but opaque storage (no raw SQL access for debugging). Rejected — "files for raw, DB for indexes" means the DB must be transparent.
