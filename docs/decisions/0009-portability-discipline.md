# ADR-0009: Portability Discipline — Interface-First, No Framework Lock-in

**Status:** Accepted  
**Date:** 2026-04-27

## Context

The system is written in Python but is intended to be portable to C# (and potentially other runtimes) in a future phase. This requires module boundaries that map cleanly to interfaces, and a deliberate avoidance of Python-ecosystem-specific framework dependencies in the core logic.

## Decision

All external dependencies (storage, embeddings, LLM, raw file I/O) are accessed through **explicitly defined interfaces** in `storage/interface.py` and `llm/`:

- `MemoryStore` — all database operations (notes, episodes, profile, FTS, vector search)
- `Embedder` — embedding generation (`embed(text) -> list[float]`)
- `Llm` — text completion (`complete(prompt, ...) -> CompletionResult`)
- `RawStore` — verbatim JSONL I/O

Rules:
1. **No LangChain / LlamaIndex in core.** Framework abstractions obscure the logic that needs to be ported.
2. **No framework-specific types crossing interface boundaries.** Core logic receives and returns plain Python types (dicts, dataclasses, primitives).
3. **New dependencies require justification in an ADR or PR comment.** The question is always: "Can this be a thin adapter behind an interface, or does it reach into core?"
4. **The C# porting guide** (`docs/porting-to-csharp.md`) is the secondary constraint. Any code that would require non-trivial redesign to port must be justified.

`MemoryService.build(config)` is the assembly seam. It wires concrete implementations to interfaces. Tests can substitute mock implementations without touching core logic.

## Consequences

- Core logic (recall, dreaming, bootstrap) is testable with in-memory mocks — no real DB or API calls needed.
- Swapping SQLite for Postgres, OpenAI embeddings for local sentence-transformers, or `gpt-4o-mini` for a local Ollama model requires only a new adapter class.
- The discipline adds boilerplate (interface definitions, adapter classes). This is intentional — the seams are the design.
- Graph databases, vector-native stores, and framework ORMs are deferred or rejected until a portability-compatible interface design is established.

## Alternatives considered

- **LangChain/LlamaIndex as the orchestration layer**: Faster initial development, but opaque internals and Python-specific. Porting would require redesign, not translation. Rejected.
- **No interfaces, direct SQLite calls everywhere**: Simple but locks storage to SQLite permanently. Rejected.
- **ORM (SQLAlchemy)**: Useful but Python-specific; generates non-portable query logic. Rejected for core. Acceptable in adapters if needed.
- **Microservices per tier**: Over-engineering for a local single-user system. Rejected for Phase 1.
