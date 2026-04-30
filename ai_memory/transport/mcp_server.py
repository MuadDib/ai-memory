"""
MCP server adapter — thin wrapper that exposes MemoryService over MCP/stdio.

Adds zero business logic. Every tool maps directly to one MemoryService
method. The same service is reachable from `cli.py` for tests and direct
use without booting an MCP transport.

Run via:
    ai-memory serve

which calls `serve_mcp(config)` below.
"""
from __future__ import annotations

import logging
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict

import anyio
from fastmcp import FastMCP

from ai_memory.config import Config
from ai_memory.core.service import MemoryService

logger = logging.getLogger(__name__)


def _attach_log_file(config: Config) -> None:
    """Add a FileHandler so MCP server logs land in a per-PID log file.

    Claude Desktop runs `ai-memory serve` as an stdio subprocess and discards
    its stderr (only stdout carries the MCP protocol).  Without this handler
    every logger.info/warning/error call from the server process is silently
    dropped — making the diagnostic log useless.  We attach the handler once,
    early in serve_mcp(), before any other log calls.

    Each server process gets its own file (service-stderr-<pid>.log) so that
    concurrent servers writing to the same path don't corrupt each other's
    output — interleaved writes on Windows produce truncated lines.
    """
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = config.logs_dir / f"service-stderr-{os.getpid()}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(fh)


def _server_version() -> str:
    try:
        from importlib.metadata import version
        return version("ai-memory")
    except Exception:
        return "unknown"


_SLOW_TOOL_THRESHOLD_S = 5.0  # warn if a tool call takes longer than this


@contextmanager
def _timed_tool(name: str, **kw):
    """Log start/finish/duration of every MCP tool call. Captures exceptions."""
    extra = "  ".join(f"{k}={v!r}" for k, v in kw.items())
    logger.info("tool.start  %s  %s", name, extra)
    t0 = time.monotonic()
    try:
        yield
        elapsed = time.monotonic() - t0
        level = logging.WARNING if elapsed >= _SLOW_TOOL_THRESHOLD_S else logging.INFO
        logger.log(level, "tool.finish %s  %.3fs", name, elapsed)
    except Exception:
        elapsed = time.monotonic() - t0
        logger.error(
            "tool.error  %s  %.3fs\n%s", name, elapsed, traceback.format_exc()
        )
        raise


def build_app(service: MemoryService) -> FastMCP:
    """Build a FastMCP app exposing the memory tools, bound to `service`."""
    app = FastMCP("ai-memory")

    # --- Tools -----------------------------------------------------------

    @app.tool()
    def memory_remember(text: str, source: str, role: str = "user") -> dict:
        """Store a piece of text as a turn in the active episode for `source`.

        Returns the new turn id, the episode id, a list of any redactions the
        privacy filter applied, and server_pid (the OS process id that handled
        this call — useful for diagnosing which server instance is active).
        """
        # Synchronous: memory_remember is fast SQLite + file I/O with no
        # external network calls. The thread offload (anyio.to_thread.run_sync)
        # caused event-loop starvation under concurrent calls and is not needed
        # here. Only memory_recall (which calls OpenAI for embeddings) needs it.
        with _timed_tool("memory_remember", source=source, role=role, chars=len(text)):
            result = service.remember(text=text, source=source, role=role)
            d = asdict(result)
            d["server_pid"] = os.getpid()
            logger.info("tool.remember  episode_id=%s server_pid=%d", result.episode_id, os.getpid())
            return d

    @app.tool()
    async def memory_recall(query: str, depth: str = "deep", k: int = 8) -> list[dict]:
        """Retrieve relevant memories.

        Always returns a merged ranking across atomic notes and
        episode summaries. `depth` only controls whether to also
        expand into raw conversation turns:

            - "fast" / "deep" -> notes + episode summaries (no raw turns)
            - "verbatim"      -> notes + episode summaries + raw turns from
                                 the matching episodes. Genuinely expensive
                                 — use only when you need exact wording.

        The fast/deep distinction was originally a tier-gate, but we found
        that gating episode summaries behind a heuristic mis-fired with small corpora
        and tight embedding score distributions. Now they're identical.
        """
        with _timed_tool("memory_recall", depth=depth, k=k):
            # Offload to a worker thread: the OpenAI embeddings call (~2 s) would
            # otherwise block the event loop and queue every other tool call behind it.
            hits = await anyio.to_thread.run_sync(
                lambda: service.recall(query=query, depth=depth, k=k)
            )
            return [asdict(h) for h in hits]

    @app.tool()
    def memory_dream(trigger: str = "manual") -> dict:
        """Run a consolidation pass on un-consolidated episodes.

        Heavy LLM work. Use sparingly during a session; let the scheduled /
        idle / pressure triggers do their job by default.
        """
        with _timed_tool("memory_dream", trigger=trigger):
            report = service.dream(trigger=trigger)
            return asdict(report)

    @app.tool()
    def memory_recent_episodes(limit: int = 10) -> list[dict]:
        """List the most recent episodes by start time."""
        with _timed_tool("memory_recent_episodes", limit=limit):
            episodes = service.list_recent_episodes(limit=limit)
            return [asdict(e) for e in episodes]

    @app.tool()
    def memory_dream_log(limit: int = 10) -> list[dict]:
        """Inspect recent dream-cycle passes."""
        with _timed_tool("memory_dream_log", limit=limit):
            logs = service.list_recent_dream_logs(limit=limit)
            return [asdict(l) for l in logs]

    # --- Resources -------------------------------------------------------

    @app.resource("memory://profile")
    def profile_resource() -> str:
        """The user's profile, as a human-readable markdown blob.

        Clients should attach this to context at session start so the agent
        knows who they're talking to.
        """
        rows = sorted(service.list_profile(), key=lambda p: p.key)
        if not rows:
            return "# Profile\n\n_(empty)_"
        lines = ["# Profile (auto-generated)\n"]
        for row in rows:
            lines.append(f"- **{row.key}**: {row.value}")
        return "\n".join(lines) + "\n"

    return app


def serve_mcp(config: Config) -> None:
    """Boot the service and start the MCP stdio loop. Blocking call."""
    # Claude Desktop discards MCP subprocess stderr (stdout carries the protocol).
    # Attach a per-PID FileHandler so our logs always land in service-stderr-<pid>.log.
    _attach_log_file(config)

    # No PID lock here — Claude Desktop legitimately spawns one server process
    # per connected window/project, so multiple instances are expected and fine.
    # SQLite WAL mode + busy_timeout=15 000 ms handles concurrent writers safely.
    # (The dream daemon keeps its own PID lock; dream cycles must not duplicate.)
    service = MemoryService.build(config)
    service.start()
    logger.info(
        "mcp-server started: pid=%d version=%s home=%s",
        os.getpid(),
        _server_version(),
        config.home,
    )
    try:
        app = build_app(service)
        # Disable FastMCP's docket worker background task.
        # FastMCP 2.14.7 runs worker.run_forever() as asyncio.create_task() on
        # the same event loop as our MCP tools. With memory:// (fakeredis), the
        # worker polls continuously and intermittently starves the event loop,
        # causing CallToolRequest messages to sit unread in stdin — triggering the
        # 4-minute MCP timeout. We register zero background tasks, so the worker
        # is pure overhead. Setting _is_mounted=True makes _docket_lifespan()
        # skip both Docket creation and Worker startup entirely.
        app._is_mounted = True
        app.run()  # FastMCP handles stdio loop
    finally:
        service.stop()
