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
from dataclasses import asdict

from fastmcp import FastMCP

from ai_memory.config import Config
from ai_memory.core.service import MemoryService

logger = logging.getLogger(__name__)


def build_app(service: MemoryService) -> FastMCP:
    """Build a FastMCP app exposing the memory tools, bound to `service`."""
    app = FastMCP("ai-memory")

    # --- Tools -----------------------------------------------------------

    @app.tool()
    def memory_remember(text: str, source: str, role: str = "user") -> dict:
        """Store a piece of text as a turn in the active episode for `source`.

        Returns the new turn id, the episode id, and a list of any redactions
        the privacy filter applied.
        """
        result = service.remember(text=text, source=source, role=role)
        return asdict(result)

    @app.tool()
    def memory_recall(query: str, depth: str = "deep", k: int = 8) -> list[dict]:
        """Retrieve relevant memories.

        Always returns a merged ranking across atomic notes (Tier 1) and
        episode summaries (Tier 2). `depth` only controls whether to also
        expand into raw conversation turns:

            - "fast" / "deep" -> notes + episode summaries (no raw turns)
            - "verbatim"      -> notes + episode summaries + raw turns from
                                 the matching episodes. Genuinely expensive
                                 — use only when you need exact wording.

        The fast/deep distinction was originally a tier-gate, but we found
        that gating tier 2 behind a heuristic mis-fired with small corpora
        and tight embedding score distributions. Now they're identical.
        """
        hits = service.recall(query=query, depth=depth, k=k)
        return [asdict(h) for h in hits]

    @app.tool()
    def memory_dream(trigger: str = "manual") -> dict:
        """Run a consolidation pass on un-consolidated episodes.

        Heavy LLM work. Use sparingly during a session; let the scheduled /
        idle / pressure triggers do their job by default.
        """
        report = service.dream(trigger=trigger)
        return asdict(report)

    @app.tool()
    def memory_recent_episodes(limit: int = 10) -> list[dict]:
        """List the most recent episodes by start time."""
        episodes = service.list_recent_episodes(limit=limit)
        return [asdict(e) for e in episodes]

    @app.tool()
    def memory_dream_log(limit: int = 10) -> list[dict]:
        """Inspect recent dream-cycle passes."""
        logs = service.list_recent_dream_logs(limit=limit)
        return [asdict(l) for l in logs]

    # --- Resources -------------------------------------------------------

    @app.resource("memory://profile")
    def profile_resource() -> str:
        """The user's Tier 0 profile, as a human-readable markdown blob.

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
    service = MemoryService.build(config)
    service.start()
    try:
        app = build_app(service)
        app.run()  # FastMCP handles stdio loop
    finally:
        service.stop()
