# ai-memory — orientation for Claude Code

This file is the entry point when Claude Code starts work in this repo. Read it first; everything else is in the docs it points to.

## What this is

A local-first shared AI memory backend with an MCP server interface. One persistent SQLite-backed memory that any MCP-aware AI client (Claude Desktop, Cursor, Cline, Continue, Zed, Claude Code itself) can read and write. Four-tier model — profile / atomic notes / episode summaries / verbatim turns — with a sleep-style "dream" consolidation cycle that turns raw conversation turns into structured memory overnight.

Stack: Python 3.12 with portability discipline so a future C# port is translation, not redesign. SQLite + sqlite-vec for vector search, FTS5 for BM25, fused via Reciprocal Rank Fusion (k=60). Recent-boosted at merge time. OpenAI for embeddings (`text-embedding-3-small`, 1536 dim) and LLM (`gpt-4o-mini` for cheap dream cycles).

## Where things live

The runnable copy is on the user's Windows machine at `C:\ai-mem\` — short path because deeper paths break Python venvs on Windows MAX_PATH. The Cowork outputs folder under `%APPDATA%\Claude\local-agent-mode-sessions\.../outputs\ai-memory\` is treated as source-of-truth; `sync.ps1` (in the outputs root) `robocopy /MIR`s it to `C:\ai-mem\` excluding `.venv` and `__pycache__`. Edit in Cowork outputs, sync to test.

Repo layout under `ai-memory/`:

- `ai_memory/cli.py` — Click subcommands. Top of file has a *deliberate* `authlib.jose` pre-import inside `redirect_stderr(StringIO())` to silence fastmcp's deprecation warning; comment explains why normal filters don't work.
- `ai_memory/config.py` — Layered config (env vars > `<home>/config.yaml` > defaults). `RecallConfig.vector_distance_floor = 1.1` is empirically tuned for `text-embedding-3-small` on this corpus.
- `ai_memory/core/recall.py` — Hybrid search pipeline (BM25 + vector → RRF → recency boost). Logs distance distributions on every call so future tuning is data-driven.
- `ai_memory/core/dreaming.py` — The seven-phase consolidation pipeline. Phase 3 (consolidate) chunks the extract step into 50-turn windows with 5-turn overlap; the summary stays single-shot. Three system prompts (`SUMMARY_SYSTEM`, `EXTRACT_SYSTEM`, `INTEGRATE_VERDICT_SYSTEM`, `PROMOTION_SYSTEM`) are first-class and quoted into dream-log entries.
- `ai_memory/core/service.py` — `MemoryService.build(config)` is the assembly seam. The MCP server and CLI both go through it; tests can also exercise it directly.
- `ai_memory/llm/` — Provider interface + OpenAI and Anthropic adapters. `CompletionResult` carries `text`, `input_tokens`, `output_tokens`, `model_id`, `finish_reason`, `refusal` — surface those when adding a new provider so dream-log diagnostics keep working.
- `ai_memory/storage/` — SQLite schema + sqlite-vec / FTS5 wiring + raw JSONL store for verbatim turns.
- `ai_memory/transport/mcp_server.py` — FastMCP wrapper. Zero business logic; every tool is a one-liner against `MemoryService`. Same service is callable from the CLI for tests.
- `ai_memory/cowork_importer.py` — Walks `.claude/projects/*/*.jsonl`. Incremental via the `cowork_import_state` table (`last_turn_id` + `last_byte_offset` per session). Re-runnable; extends episodes that grew and re-marks `consolidated_at = NULL` so dream picks up the delta.
- `tests/` — `test_recall.py`, `test_dreaming.py`, `test_storage_sqlite.py`, `test_privacy.py`, `test_cowork_importer.py`. Pure-function tests where possible; SQLite tests use temp DBs.

Runtime data lives at `%LOCALAPPDATA%\ai-memory\` (`memory.db`, `profile.md`, `raw/YYYY/MM/<episode>.jsonl`, `exports/`, `logs/`, `config.yaml`). Override via `AI_MEMORY_HOME`.

## Read these before changing anything significant

- **`../shared-ai-memory-proposal.md`** — the canonical design doc. Versioned via changelog blocks at the top. v4 (2026-04-27 afternoon) covers the recall-quality debug arc and the open follow-ups; v3 covers the build session; v2 covers the pre-build design with research synthesis from six existing-memory-MCP deep-dives.
- **`README.md`** — install/wiring instructions and the current command surface.
- **`docs/porting-to-csharp.md`** — the portability discipline that shapes module boundaries. Honour it when adding new code; if you cross it, justify it in the PR or in the proposal doc.

## Stable contracts (don't change without eval data)

These are listed in proposal v3 + v4:

- **MCP tool surface** — `memory_recall`, `memory_remember`, `memory_dream`, `memory_dream_log`, `memory_recent_episodes`, plus the `memory://profile` resource. Adding tools is fine; renaming breaks downstream client configs.
- **On-disk layout** at `%LOCALAPPDATA%\ai-memory\`. Schema version lives in the `schema_version` table; write migrations into `SCHEMA_SQL`.
- **Embeddings model = `text-embedding-3-small` at 1536 dim.** Switching the model invalidates every existing vector. Notes/episodes carry `embedding_model` per row to enable lazy or batch re-embed if we ever change.
- **Transcript rendering shape** (`<conversation_transcript><turn n=N role=R>...</turn>...</conversation_transcript>`) — the prompts depend on it. Changing it regresses dream quality. Verified by the recall-quality debug arc.
- **Three-category extract prompt structure** (user / system / problems-and-fixes). Re-organising the categories silently re-biases extraction.

## Open follow-ups (priority order, all from proposal v4)

1. **Phase 4 dedup tuning** — `DUPLICATE_DIST_BELOW = 0.10` is too strict; visible duplicates in the corpus. Target ~0.35 for `text-embedding-3-small`. **Do the eval harness first** so this isn't blind.
2. **Phase 4 contradiction detection** — `invalidated=0` even on direct contradictions. Mid-band similarity LLM-verdict branch isn't earning its call.
3. **Eval harness (#16, still pending)** — YAML of `query → expected_fact_substring`, `ai-memory eval` runs them, prints recall@k, persists to SQLite for regression tracking. Highest leverage; makes everything else cheap.
4. **`ai-memory redream --episode <id> [--purge-notes]` admin subcommand** — productise what the v4 debug session needed five hand-rolled Python heredocs for. Support `--all-pending` and `--latest` too.
5. **Auto-upgrade to `gpt-4o` for long episodes** — chunking got us across the line on `gpt-4o-mini` but Phase 4 quality looks model-bound. Per-call escalation rule.
6. **Pre-render redaction over the rendered transcript** — privacy filter currently runs *before persistence*, but the assembled transcript still has raw creds when sent to OpenAI for dream cycles.

Plus everything still pending from v3: incremental Phase 3 dreaming, `remember --from-stdin`, cross-encoder rerank (Phase 2), local sentence-transformers embedder (Phase 2), Linux daemon mode.

## Conventions

- Direct, no sugar-coating; pushback on weak ideas welcome (the user's stated style).
- Prefer surgical edits over rewrites. The codebase is intentionally small; keep it that way.
- Files for raw, DB for indexes — never make opaque storage the source of truth.
- Heavy LLM work happens in dream cycles, never on the recall hot path.
- Storage / embeddings / LLM all behind interfaces — swap implementations without touching call sites.
- No LangChain / LlamaIndex in core.
- Add a unit test for any new pure function; SQLite-touching code uses temp DB fixtures.

## Quick recovery recipes

These come up often enough they're worth pinning:

**Reset and re-dream a specific episode** (e.g. for prompt-iteration testing):

```python
import os, sqlite3
db = os.path.join(os.environ['LOCALAPPDATA'], 'ai-memory', 'memory.db')
con = sqlite3.connect(db)
ep_id = "<episode-id>"
con.execute("UPDATE episodes SET consolidated_at = NULL WHERE id = ?", (ep_id,))
con.execute("DELETE FROM notes WHERE source_episode_ids LIKE ?", (f'%{ep_id}%',))
con.commit()
```

Then `ai-memory dream`. (Pending: bake this into `ai-memory redream`.)

**Inspect a corpus quickly:** `ai-memory stats` for counts, `ai-memory recent-episodes 20` for episode list, `ai-memory dream-log 10` for recent passes.

**Tune the relevance floor:** every `recall` call logs distance min/median/max for both note-vector and episode-vector searches. Run `ai-memory recall <query>` with the log level at INFO and read the numbers; adjust `recall.vector_distance_floor` in `config.yaml`.
