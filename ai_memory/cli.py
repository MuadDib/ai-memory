"""
Command-line interface.

Entry point for `ai-memory <subcommand>`. Each subcommand is a thin click
wrapper around a MemoryService method or the MCP server boot loop.
"""
from __future__ import annotations

# --- Third-party deprecation noise --------------------------------------
# fastmcp pulls in authlib.jose, which prints an AuthlibDeprecationWarning
# on every CLI invocation. authlib defeats normal warnings filters (it
# resets the filter list during its own import via simplefilter('always')),
# so we pre-import the module here with stderr redirected to a sink. Once
# it's cached in sys.modules, fastmcp's later import is a no-op and stays
# silent. Drop this whole block when fastmcp migrates off authlib.jose.
import contextlib as _contextlib
import io as _io

with _contextlib.redirect_stderr(_io.StringIO()):
    try:
        import authlib.jose  # noqa: F401
    except ImportError:
        pass

import logging
import sys
from pathlib import Path

import click

from ai_memory.bootstrap import bootstrap_from_markdown
from ai_memory.config import ensure_home_layout, load_config
from ai_memory.core.service import MemoryService
from ai_memory.timestamps import iso_to_dt
from ai_memory.transport.mcp_server import serve_mcp


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@click.group()
@click.option("--home", type=click.Path(path_type=Path), default=None,
              help="Override AI_MEMORY_HOME for this invocation.")
@click.pass_context
def main(ctx: click.Context, home: Path | None) -> None:
    """ai-memory: local-first shared AI memory with MCP server."""
    config = load_config(home_override=home)
    ensure_home_layout(config)
    _setup_logging(config.log_level)
    ctx.obj = config


@main.command()
@click.pass_obj
def serve(config) -> None:
    """Start the MCP server on stdio. Wire this into Claude Desktop / Cursor."""
    serve_mcp(config)


@main.command()
@click.option("--chatgpt", "chatgpt_path", type=click.Path(exists=True, path_type=Path),
              help="Path to a ChatGPT-style memory markdown export.")
@click.option("--claude", "claude_path", type=click.Path(exists=True, path_type=Path),
              help="Path to a Claude-style memory markdown export.")
@click.pass_obj
def bootstrap(config, chatgpt_path, claude_path):
    """Import existing AI memory dumps so the server starts with real data."""
    if chatgpt_path is None and claude_path is None:
        raise click.UsageError("Pass at least one of --chatgpt or --claude.")
    service = MemoryService.build(config)
    service.start()
    try:
        if chatgpt_path:
            r = bootstrap_from_markdown(
                service=service, file_path=chatgpt_path, source="chatgpt-export",
                title="ChatGPT memory export",
            )
            click.echo(f"chatgpt: episode={r.episode_id} notes={r.notes_inserted} profile={r.profile_updates}")
        if claude_path:
            r = bootstrap_from_markdown(
                service=service, file_path=claude_path, source="claude-export",
                title="Claude memory export",
            )
            click.echo(f"claude: episode={r.episode_id} notes={r.notes_inserted} profile={r.profile_updates}")
    finally:
        service.stop()


@main.command()
@click.option("--trigger", default="manual",
              type=click.Choice(["manual", "scheduled", "idle", "pressure"]))
@click.option("--watch", is_flag=True,
              help="Run as a long-lived daemon. Fires dream() on schedule, idle, or pressure.")
@click.pass_obj
def dream(config, trigger, watch):
    """Run a dream-cycle pass once, or as a watching daemon."""
    if watch:
        from ai_memory.daemon import run_watch
        sys.exit(run_watch(config))

    service = MemoryService.build(config)
    service.start()
    try:
        report = service.dream(trigger=trigger)
        click.echo(f"dream {report.log_id}: episodes={report.episodes_processed} "
                   f"notes_added={report.notes_added} pruned={report.notes_pruned}")
        click.echo("--- journal ---")
        click.echo(report.journal)
    finally:
        service.stop()


@main.command()
@click.argument("query")
@click.option("--depth", default="fast", type=click.Choice(["fast", "deep", "verbatim"]))
@click.option("--k", default=8, type=int)
@click.pass_obj
def recall(config, query, depth, k):
    """Query the memory from the command line (handy for sanity-checking)."""
    service = MemoryService.build(config)
    service.start()
    try:
        hits = service.recall(query=query, depth=depth, k=k)
        if not hits:
            click.echo("(no hits)")
            return
        for hit in hits:
            click.echo(f"[{hit.item_type} {hit.score:.4f}] {hit.text}")
    finally:
        service.stop()


@main.command()
@click.pass_obj
def profile(config):
    """Print the current Tier 0 profile."""
    service = MemoryService.build(config)
    service.start()
    try:
        for row in sorted(service.list_profile(), key=lambda p: p.key):
            click.echo(f"{row.key}: {row.value}")
    finally:
        service.stop()


@main.command("recent-episodes")
@click.option("--limit", default=10, type=int)
@click.pass_obj
def recent_episodes(config, limit):
    """List the most recent episodes by start time."""
    service = MemoryService.build(config)
    service.start()
    try:
        episodes = service.list_recent_episodes(limit=limit)
        if not episodes:
            click.echo("(no episodes)")
            return
        for ep in episodes:
            ts = iso_to_dt(ep.started_at).strftime("%Y-%m-%d %H:%M")
            consol = "C" if ep.consolidated_at else "."
            summary = (ep.summary or "").strip().replace("\n", " ")
            if len(summary) > 100:
                summary = summary[:97] + "..."
            click.echo(f"[{consol}] {ts}  {ep.source:20s}  {ep.id}  {summary}")
    finally:
        service.stop()


@main.command("dream-log")
@click.option("--limit", default=10, type=int)
@click.pass_obj
def dream_log(config, limit):
    """Show recent dream-cycle passes."""
    service = MemoryService.build(config)
    service.start()
    try:
        logs = service.list_recent_dream_logs(limit=limit)
        if not logs:
            click.echo("(no dream passes recorded)")
            return
        for entry in logs:
            ts = iso_to_dt(entry.started_at).strftime("%Y-%m-%d %H:%M")
            click.echo(
                f"{ts}  trigger={entry.trigger:10s}  "
                f"episodes={entry.episodes_processed}  "
                f"notes_added={entry.notes_added}  "
                f"pruned={entry.notes_pruned}"
            )
    finally:
        service.stop()


@main.command()
@click.pass_obj
def stats(config):
    """Quick corpus inventory — handy for diagnosing empty / weak recall results."""
    import sqlite3
    db_path = config.database_path
    if not db_path.exists():
        click.echo(f"(no database yet at {db_path})")
        return
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        for table in ("turns", "episodes", "notes", "profile", "cowork_import_state"):
            try:
                (n,) = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                click.echo(f"{table:25s} {n}")
            except sqlite3.OperationalError as exc:
                click.echo(f"{table:25s} (missing: {exc})")
        # Episodes pending consolidation
        try:
            (pending,) = cur.execute(
                "SELECT COUNT(*) FROM episodes WHERE consolidated_at IS NULL"
            ).fetchone()
            click.echo(f"{'episodes pending dream':25s} {pending}")
        except sqlite3.OperationalError:
            pass
        # Valid (uncontradicted) notes
        try:
            (valid,) = cur.execute(
                "SELECT COUNT(*) FROM notes WHERE valid_to IS NULL"
            ).fetchone()
            click.echo(f"{'notes (valid)':25s} {valid}")
        except sqlite3.OperationalError:
            pass
    finally:
        conn.close()


@main.command("import-cowork")
@click.option("--root", "root_path", type=click.Path(exists=True, path_type=Path),
              default=None,
              help="Root folder to scan for *.jsonl session transcripts. "
                   "Defaults to the packaged Cowork sessions directory under %LOCALAPPDATA%.")
@click.option("--include-tools", is_flag=True,
              help="Include tool_use / tool_result blocks in the imported text. "
                   "Default is to skip them (less noise for dream-cycle extraction).")
@click.option("--session-id", default=None,
              help="If set, only import the matching session id. Useful for testing.")
@click.pass_obj
def import_cowork(config, root_path, include_tools, session_id):
    """Bulk-import past Cowork / Claude Code chat transcripts into Tier 3.

    Each session becomes one Episode + many Turn rows. Re-runnable: sessions
    that already have a matching episode id are skipped. The dream cycle
    picks them up on its next pass.
    """
    import os
    if root_path is None:
        default = os.environ.get("LOCALAPPDATA", "")
        if not default:
            raise click.UsageError(
                "Could not infer default --root (LOCALAPPDATA not set). "
                "Pass --root explicitly."
            )
        root_path = Path(default) / "Packages" / "Claude_pzs8sxrjxfjjc" / \
            "LocalCache" / "Roaming" / "Claude" / "local-agent-mode-sessions"
        if not root_path.exists():
            raise click.UsageError(
                f"Default root not found at {root_path}. Pass --root <dir> to override."
            )

    from ai_memory.cowork_importer import import_cowork_sessions

    service = MemoryService.build(config)
    service.start()
    try:
        result = import_cowork_sessions(
            service=service,
            root=root_path,
            include_tools=include_tools,
            only_session_id=session_id,
        )
        click.echo(
            f"sessions seen={result.sessions_seen} "
            f"new={result.sessions_imported_new} "
            f"extended={result.sessions_extended} "
            f"unchanged={result.sessions_skipped_unchanged} "
            f"turns_inserted={result.turns_inserted}"
        )
    finally:
        service.stop()


if __name__ == "__main__":
    main()
