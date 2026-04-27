# Porting ai-memory to C# (.NET 9)

This doc exists from day one because the Python build is explicitly designed
to be portable. If/when you want to live in your home language, the work
should read as a translation, not a rewrite.

## What translates 1:1

| Python concept | .NET equivalent |
|---|---|
| `dataclass` (e.g. `Note`, `Episode`) | `record` or `class` with init-only setters |
| `Protocol` (e.g. `Embedder`, `MemoryStore`, `Llm`) | `interface IEmbedder`, `IMemoryStore`, `ILlm` |
| `dict` config + `dataclass` Config | `Microsoft.Extensions.Configuration` + record options bound via `IOptions<T>` |
| `pyyaml` + env-var override | Same — `appsettings.yaml` + `IConfigurationBuilder.AddEnvironmentVariables("AI_MEMORY_")` |
| `sqlite3` stdlib + `sqlite-vec` | `Microsoft.Data.Sqlite` + `sqlite-vec` (load extension via `LoadExtension`) |
| `openai` python lib | `OpenAI` NuGet (`OpenAI.Embeddings`) |
| `anthropic` python lib | `Anthropic.SDK` NuGet (or call REST directly) |
| `fastmcp` | `ModelContextProtocol` C# SDK (Microsoft-maintained) |
| `click` CLI | `System.CommandLine` |
| `python-ulid` | `NUlid` NuGet |

## Module-by-module mapping

| Python module | C# namespace |
|---|---|
| `ai_memory.core.models` | `AiMemory.Core.Models` |
| `ai_memory.core.service` | `AiMemory.Core.MemoryService` |
| `ai_memory.core.recall` | `AiMemory.Core.Recall` |
| `ai_memory.core.ingest` | `AiMemory.Core.Ingest` |
| `ai_memory.core.dreaming` | `AiMemory.Core.Dreaming` |
| `ai_memory.storage.interface` | `AiMemory.Storage.IMemoryStore` |
| `ai_memory.storage.sqlite_store` | `AiMemory.Storage.Sqlite.SqliteStore` |
| `ai_memory.storage.raw_files` | `AiMemory.Storage.RawTranscriptStore` |
| `ai_memory.embeddings.interface` | `AiMemory.Embeddings.IEmbedder` |
| `ai_memory.embeddings.openai_embedder` | `AiMemory.Embeddings.OpenAiEmbedder` |
| `ai_memory.llm.interface` | `AiMemory.Llm.ILlm` |
| `ai_memory.llm.anthropic_llm` | `AiMemory.Llm.AnthropicLlm` |
| `ai_memory.transport.mcp_server` | `AiMemory.Transport.Mcp.McpServer` |
| `ai_memory.cli` | `AiMemory.Cli.Program` |
| `ai_memory.privacy` | `AiMemory.Privacy.Redactor` |
| `ai_memory.bootstrap` | `AiMemory.Bootstrap.MarkdownImporter` |

## Things that stay identical

- **SQL schema.** Copy the SQL string from `sqlite_store.py` to a `.sql` resource file. The schema is portable.
- **JSONL raw files.** Same on-disk format; both runtimes read/write JSON lines fine.
- **MCP tool signatures.** The wire protocol is JSON-RPC; tool names, arg shapes, and return shapes don't change.
- **Reciprocal Rank Fusion math.** Pure arithmetic — copy the algorithm verbatim.
- **Privacy regex patterns.** Copy the regex strings; both runtimes' regex engines understand them.

## Things that change shape

- **async/await.** Python file uses sync calls (sufficient for a single-user local server). The C# port should use `async` throughout — `Microsoft.Data.Sqlite` is sync, but everything else (HTTP, file IO at scale) benefits.
- **Dependency injection.** Python uses constructor injection by hand (`MemoryService.build`); C# uses `Microsoft.Extensions.DependencyInjection` (`services.AddSingleton<IEmbedder, OpenAiEmbedder>()`).
- **Logging.** `logging` → `ILogger<T>`.
- **Config validation.** `dataclass(frozen=True)` → `record` with required fields and `IValidateOptions<T>`.

## Things to deliberately avoid in the Python build

To keep the port mechanical, the Python code stays clear of:

- LangChain / LlamaIndex (no equivalents in .NET, and porting them would be huge)
- Any Python-specific magic: descriptors, metaclasses, `__init_subclass__`, etc.
- Async-only libs in the core (we use sync sqlite + sync openai-client)
- Pickle, dill, numpy as a hard dep, anything that doesn't have a clean .NET counterpart

## What you build first when porting

1. `Models` (records) — pure data, no deps.
2. `IMemoryStore` interface + `SqliteStore` implementation — get schema parity working with a quick test.
3. `IEmbedder` + `OpenAiEmbedder`. Verify dimensions match.
4. `MemoryService` with `Remember` and `Recall`. Hybrid search reuses the same RRF math.
5. MCP transport via the C# SDK.
6. `Bootstrap` markdown importer.
7. `Dreaming` last — it's the heaviest module and you want everything else proven first.

## Migration path (run both side-by-side)

Both runtimes can point at the same `memory.db` and `raw/` tree — the on-disk
format is the contract. So during a port you can:

1. Stand up the C# build pointing at a copy of your existing data dir.
2. Verify recall results match the Python build for a fixed query set.
3. Cut Claude Desktop / Cursor over to the C# server.
4. Decommission the Python build.

No migration step needed. That's the whole point of designing for it from day one.
