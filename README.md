# ai-memory

Local-first shared AI memory backend with an MCP server interface. One persistent, queryable memory that any MCP-aware AI client (Claude Desktop, Cursor, Cline, Continue, Zed, Claude Code, ...) can read and write.

## Status

Phase 1 — running end-to-end on a real corpus. All core commands are operational. Current corpus: bootstrapped notes from exported AI memory dumps + imported Claude Code sessions consolidated via the dream cycle. Eval suite: 27 cases, all passing.

Latest completed work: Phase 4 contradiction detection fix (DUPLICATE_DIST_BELOW 0.20→0.10 + prompt rewrite with same-attribute/different-value pattern); Phase 5 promotion prompt guard against external-entity pollution; bootstrap cross-file semantic dedup (two-tier: vector threshold + LLM mid-band); gpt-4o auto-upgrade for long episodes (≥100 turns); `redream` and `backfill-entities` CLI subcommands; 12 ADRs in `docs/decisions/`.

## Why this exists

Existing memory MCP servers are either hosted-only with vendor lock-in (Supermemory), too thin to be useful (the official reference server), or strong on one tier but weak on others (Mem0, MemPalace, Graphiti, etc.).

This project explicitly models four tiers and a sleep-style consolidation cycle:

| Tier | What it is | Storage |
|---|---|---|
| 0 — Profile | Durable user/agent facts. Always loaded. | SQLite KV + `profile.md` mirror |
| 1 — Semantic notes | Atomic facts, hybrid-searched (BM25 + vector + RRF, k=60) | SQLite + sqlite-vec + FTS5 |
| 2 — Episodes | Per-session summaries | SQLite + sqlite-vec |
| 3 — Verbatim | Raw turns, append-only JSONL files. Pulled on demand. | Disk |

See [`docs/decisions/`](docs/decisions/) for the full set of architectural decision records.

## Quickstart (Windows)

```powershell
# 1. Install
git clone <this repo>
cd ai-memory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 2. Configure — OpenAI is used for both embeddings and dream-cycle LLM
$env:OPENAI_API_KEY = "sk-..."

# 3. Bootstrap from your existing AI memory exports (optional but recommended)
ai-memory bootstrap --chatgpt path\to\chatgpt-memory.md --claude path\to\claude-memory.md

# 4. Run the MCP server
ai-memory serve
```

## Wiring into Claude Desktop

Edit `%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-memory": {
      "command": "ai-memory",
      "args": ["serve"]
    }
  }
}
```

Restart Claude Desktop. You'll see `ai-memory` appear as a tools provider with `memory_recall`, `memory_remember`, `memory_dream`, etc.

## Wiring into Cursor

Add to `~/.cursor/mcp.json` (or via Settings → MCP):

```json
{
  "mcpServers": {
    "ai-memory": {
      "command": "ai-memory",
      "args": ["serve"]
    }
  }
}
```

Same database. Same memories. Same `profile.md`.

## File layout (what lives where)

```
%LOCALAPPDATA%\ai-memory\
├── memory.db                    # SQLite — profile, notes, episodes, turns, FTS, vec, dream log
├── profile.md                   # Mirror of Tier 0; human-readable
├── raw/2026/04/<episode>.jsonl  # Append-only raw turns
├── exports/                     # Bootstrap dumps you imported
├── logs/                        # Operational logs
└── config.yaml                  # All knobs (see below)
```

Override the root with `AI_MEMORY_HOME=C:\path\to\custom\dir`.

## Commands

| Command | What it does |
|---|---|
| `ai-memory serve` | Start the MCP server on stdio |
| `ai-memory bootstrap --chatgpt <path> --claude <path>` | Ingest existing memory exports (semantic dedup: vector threshold + LLM mid-band) |
| `ai-memory import-cowork [--root <path>] [--session-id <id>]` | Bulk-import past Claude Code transcripts into Tier 3; incremental via state cursor |
| `ai-memory dream [--trigger manual\|scheduled\|idle\|pressure]` | Trigger a consolidation pass manually |
| `ai-memory redream --episode <id\|prefix>` | Reset a specific episode and re-dream it (prompt-iteration testing) |
| `ai-memory redream --latest` | Reset and re-dream the most recently consolidated episode |
| `ai-memory redream --all-pending` | Re-dream all unconsolidated episodes |
| `ai-memory backfill-entities [--batch-size N] [--dry-run]` | LLM entity extraction for notes with empty entity lists |
| `ai-memory recall <query> [--depth fast\|deep\|verbatim] [--k N]` | Query memory from the CLI |
| `ai-memory profile` | Print the current Tier 0 profile |
| `ai-memory stats` | Corpus inventory — turns / episodes / notes / profile / pending-dream counts |
| `ai-memory recent-episodes [--limit N]` | List most-recent episodes with consolidated flag and summary |
| `ai-memory dream-log [--limit N]` | Show recent dream-cycle passes with episode/note/prune counts and journal |

## Configuration (`config.yaml`)

```yaml
llm:
  provider: openai
  model: gpt-4o-mini           # default model for all dream phases
  max_tokens: 4096
  quality_model: gpt-4o        # upgrade model for long episodes in Phase 3

dream:
  long_episode_turns: 100      # episodes >= this many turns use quality_model for Phase 3
```

All values have defaults — only override what you need. Full knob list in `ai_memory/config.py`.

## Design principles

See [`docs/decisions/`](docs/decisions/) for the full ADR set. Highlights:

- **Files for raw, DB for indexes.** Source of truth never lives only in opaque storage.
- **Hybrid search by default** — vector + BM25 + RRF (k=60), recency-boosted at merge.
- **Bi-temporal facts** (`valid_from` / `valid_to` / `ingested_at`). Updates invalidate, never delete.
- **Heavy LLM work in dream cycles only** — never on the recall hot path.
- **Storage / embeddings / LLM behind interfaces** — swap implementations without touching call sites.
- **No LangChain / LlamaIndex in core.** Plain functions over framework abstractions.
- **Designed for clean port to C#.** See [`docs/porting-to-csharp.md`](docs/porting-to-csharp.md).

## Privacy

- All data stays on disk. No cloud sync, no telemetry.
- Embeddings and LLM calls go to OpenAI by default. Swap to local sentence-transformers / Ollama for fully offline operation (interfaces in `ai_memory/embeddings/` and `ai_memory/llm/`).
- Privacy filter strips API keys, JWTs, and password-like patterns from text before persisting or embedding.

## License

MIT.
