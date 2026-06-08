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

## Quickstart — Windows (PowerShell)

> **Note:** Clone to a short path. Deep paths break Python venvs on Windows due to MAX_PATH limits.

```powershell
# 1. Clone and install
git clone https://github.com/MuadDib/ai-memory.git C:\ai-mem\ai-memory
cd C:\ai-mem\ai-memory

# Allow venv activation scripts (only needed once per machine)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

py -3.11 -m venv .venv          # requires Python 3.11+; use py -3.12 if you have 3.12
.\.venv\Scripts\Activate.ps1
pip install -e .

# 2. Set your OpenAI API key
# Current session only:
$env:OPENAI_API_KEY = "sk-..."
# — or persist it for all future sessions:
[System.Environment]::SetEnvironmentVariable("OPENAI_API_KEY", "sk-...", "User")

# 3. Verify the install
ai-memory stats     # should print zeroes, not an error

# 4. Bootstrap from existing AI memory exports (optional but recommended)
ai-memory bootstrap --chatgpt path\to\chatgpt-memory.md --claude path\to\claude-memory.md

# 5. Wire into your MCP client (see below), then start the server
ai-memory serve
```

## Quickstart — Linux / macOS (bash)

```bash
# 1. Clone and install
git clone https://github.com/MuadDib/ai-memory.git ~/ai-memory
cd ~/ai-memory
python3.11 -m venv .venv        # requires Python 3.11+
source .venv/bin/activate
pip install -e .

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."
# Add to ~/.bashrc or ~/.zshrc to persist across sessions

# 3. Verify the install
ai-memory stats

# 4. Bootstrap (optional)
ai-memory bootstrap --chatgpt path/to/chatgpt-memory.md --claude path/to/claude-memory.md

# 5. Wire into your MCP client (see below), then start the server
ai-memory serve
```

## Wiring into MCP clients

The MCP server runs on stdio — your client spawns it as a subprocess. The config block is the same for all clients; only the config file location differs.

```json
{
  "mcpServers": {
    "ai-memory": {
      "command": "ai-memory",
      "args": ["serve"],
      "env": { "OPENAI_API_KEY": "sk-..." }
    }
  }
}
```

> If `ai-memory` isn't on your system PATH (e.g. you're using a venv), use the full path to the script:
> - Windows: `"command": "C:\\ai-mem\\ai-memory\\.venv\\Scripts\\ai-memory.exe"`
> - Linux/macOS: `"command": "/home/you/ai-memory/.venv/bin/ai-memory"`

| Client | Config file |
|---|---|
| **Claude Desktop** | `%APPDATA%\Claude\claude_desktop_config.json` (Windows) / `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) |
| **Claude Code** | `.claude/mcp.json` in your project, or `~/.claude/mcp.json` globally |
| **Cursor** | `~/.cursor/mcp.json` or Settings → MCP |
| **Cline / Continue / Zed** | Their respective MCP config files |

Restart the client after editing. All clients share the same `memory.db` — same memories, same `profile.md`, regardless of which tool wrote them.

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
  max_request_tokens: 25000    # per-request input ceiling (~4 chars/token). Larger
                               # transcripts are summarised map-reduce so one episode
                               # can never exceed the model's per-minute token (TPM)
                               # limit. Keep this comfortably under your model's TPM.
  rate_limit_max_retries: 5    # transient-429 retries before giving up on a call
  rate_limit_backoff_seconds: 2.0  # base for exponential backoff between those retries
```

All values have defaults — only override what you need. Full knob list in `ai_memory/config.py`.

> **Rate limits & long episodes.** `quality_model` (e.g. `gpt-4o`) on a low-TPM
> account is the classic deadlock trap: a single 800-turn episode's summary can
> exceed the per-minute token limit and 429 forever. `max_request_tokens` caps
> every Phase 3 request (summary is now map-reduced when oversized) and transient
> 429s are retried with backoff, so the cycle stays resilient. See
> [ADR-0013](docs/decisions/0013-dream-cycle-resilience-and-rate-limit-safety.md).

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
