# ADR-0010: MCP Tool Surface — Stable Contract

**Status:** Accepted  
**Date:** 2026-04-27

## Context

The system exposes memory operations to AI clients (Claude Desktop, Cursor, Cline, Continue, Zed, Claude Code) via the Model Context Protocol (MCP). Once a tool name is wired into a client's config file, renaming it breaks that client silently — the tool simply disappears from the client's tool list.

## Decision

The MCP tool surface is a **stable contract**. The following tools and resource are frozen at their current names and signatures:

| Name | Type | Description |
|------|------|-------------|
| `memory_recall` | Tool | Hybrid search over notes + episodes |
| `memory_remember` | Tool | Store a new note (Tier 1 insert) |
| `memory_dream` | Tool | Trigger a dream cycle for pending episodes |
| `memory_dream_log` | Tool | Retrieve recent dream-log entries |
| `memory_recent_episodes` | Tool | List recent episodes with metadata |
| `memory://profile` | Resource | Read the current Tier 0 profile document |

Rules:
- **Adding new tools is safe.** New tools appear in clients that poll for capabilities.
- **Renaming existing tools is a breaking change.** Requires a deprecation period with the old name aliased to the new one.
- **Changing parameter names or types is a breaking change.** Add optional parameters; don't rename or remove existing ones.
- **The MCP server (`transport/mcp_server.py`) contains zero business logic.** Every tool is a one-liner delegating to `MemoryService`. This ensures the same logic is testable via the CLI without MCP.

## Consequences

- Clients that wire `memory_recall` into their config files will continue to work across upgrades.
- New capabilities must be exposed as new tools, not modifications to existing tools.
- The transport layer (`mcp_server.py`) must stay thin — business logic that creeps into it can't be tested without an MCP runtime.

## Alternatives considered

- **Version the tool names** (`memory_recall_v2`): Confusing for end users. Rejected.
- **Single generic `memory` tool with an `action` parameter**: Harder for LLMs to discover capabilities; breaks the MCP tool-discovery model. Rejected.
- **Expose raw SQL via MCP**: Security risk; breaks abstraction. Rejected.
