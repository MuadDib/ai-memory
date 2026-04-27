# ai-memory

Local-first shared AI memory backend with an MCP server interface. One persistent, queryable memory that any MCP-aware AI client (Claude Desktop, Cursor, Cline, Continue, Zed, ...) can read and write.

## Status

Phase 1 (MVP) — running end-to-end on a real corpus. `remember`, `recall`, `bootstrap`, `import-cowork`, and `dream` (Phases 1–7) all work. As of 2026-04-27, recall-quality has been verified against a 260-turn imported session: query *"what was the BOM problem we hit"* returns the cause and the fix, both extracted via a chunked dream pass. Outstanding tuning lives in [`../shared-ai-memory-proposal.md`](../shared-ai-memory-proposal.md) (v3 → v4 changelog).

## Why this exists

Existing memory MCP servers are either hosted-only with vendor lock-in (Supermemory), too thin to be useful (the official reference server), or strong on one tier but weak on others (Mem0, MemPalace, Graphiti, etc.).

This project explicitly models four tiers and a sleep-style consolidation cycle:

| Tier | What it is | Storage |
|---|---|---|
| 0 — Profile | Durable user/agent facts. Always loaded. | SQLite KV + `profile.md` mirror |
| 1 — Semantic notes | Atomic facts, hybrid-searched (BM25 + vector + RRF) | SQLite + sqlite-vec + FTS5 |
| 2 — Episodes | Per-session summaries | SQLite + sqlite-vec |
| 3 — Verbatim | Raw turns, append-only JSONL files. Pull on demand. | Disk |

See [`docs/architecture.md`](docs/architecture.md) for the full design.

## Quickstart (Windows)

```powershell
# 1. Install
git clone <this repo>
cd ai-memory
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# 2. Configure
$env:OPENAI_API_KEY = "sk-..."         # for embeddings
$env:ANTHROPIC_API_KEY = "sk-ant-..."  # for dream-cycle consolidation

# 3. Bootstrap from your existing AI memory dumps (optional but recommended)
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

Restart Claude Desktop. You'll see `ai-memory` show up as a tools provider, with `memory_recall`, `memory_remember`, `memory_dream`, etc.

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
└── config.yaml                  # All knobs
```

Override the root with `AI_MEMORY_HOME=C:\path\to\custom\dir`.

## Commands

| Command | What it does |
|---|---|
| `ai-memory serve` | Start the MCP server on stdio |
| `ai-memory bootstrap --chatgpt <path> --claude <path>` | Ingest existing memory dumps (ChatGPT/Claude exports) |
| `ai-memory import-cowork [--root <path>] [--session-id <id>]` | Bulk-import past Cowork / Claude Code transcripts into Tier 3, incremental via state cursor |
| `ai-memory dream [--trigger manual\|scheduled\|idle\|pressure]` | Trigger a consolidation pass manually |
| `ai-memory dream --watch` | Run as a long-lived daemon — fires on schedule, idle, or pressure |
| `ai-memory recall <query> [--depth fast\|deep\|verbatim] [--k N]` | Query memory from the CLI (handy for sanity-checking) |
| `ai-memory profile` | Print the current Tier 0 profile |
| `ai-memory stats` | Quick corpus inventory — turns / episodes / notes / profile / pending-dream counts |
| `ai-memory recent-episodes [--limit N]` | List most-recent episodes with consolidated flag, source, summary head |
| `ai-memory dream-log [--limit N]` | Show recent dream-cycle passes with episode/note/prune counts |

Planned (not yet shipped): `rebuild`, `forget`, `export`, `remember --from-stdin`, `redream --episode <id>`. See proposal v4 follow-ups.

## Design principles

See [`docs/architecture.md`](docs/architecture.md). Highlights:

- Files for raw, DB for indexes. Source of truth never lives only in opaque storage.
- Hybrid search by default (vector + BM25 + RRF, k=60).
- Bi-temporal facts (`valid_from` / `valid_to` / `ingested_at`). Updates invalidate, never delete.
- Heavy LLM work happens in dream cycles, never on the recall hot path.
- Storage / embeddings / LLM all behind interfaces — swap implementations without touching call sites.
- No LangChain / LlamaIndex in core. Plain functions over framework abstractions.
- Designed for clean port to C# later. See [`docs/porting-to-csharp.md`](docs/porting-to-csharp.md).

## Privacy

- All data stays on disk by default.
- Embeddings call out to OpenAI by default; swap to local sentence-transformers for fully offline (Phase 2).
- Privacy filter strips API keys, JWTs, and password-like patterns from text before persisting / embedding.

## License

MIT.
