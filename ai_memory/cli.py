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
@click.option("--episode", "episode_ids", multiple=True,
              help="Episode ID (or unambiguous prefix) to reset and re-dream. Repeatable.")
@click.option("--latest", is_flag=True,
              help="Reset and re-dream the most recently consolidated episode.")
@click.option("--all-pending", is_flag=True,
              help="Re-dream all episodes not yet consolidated (no reset needed; just runs dream).")
@click.option("--keep-notes", is_flag=True,
              help="Reset consolidated_at without deleting extracted notes. "
                   "Notes from the re-dream will accumulate alongside old ones; "
                   "Phase 4 dedup will merge them. Default is to purge notes first.")
@click.option("--trigger", default="manual",
              type=click.Choice(["manual", "scheduled", "idle", "pressure"]))
@click.pass_obj
def redream(config, episode_ids, latest, all_pending, keep_notes, trigger):
    """Reset one or more episodes and re-run the dream cycle on them.

    Useful for prompt iteration: after tweaking EXTRACT_SYSTEM or
    INTEGRATE_VERDICT_SYSTEM, run redream to regenerate notes from scratch.

    Examples:\n
      ai-memory redream --latest\n
      ai-memory redream --episode 3d275d78\n
      ai-memory redream --episode abc123 --episode def456\n
      ai-memory redream --all-pending
    """
    if not episode_ids and not latest and not all_pending:
        raise click.UsageError(
            "Specify at least one of --episode, --latest, or --all-pending."
        )

    service = MemoryService.build(config)
    service.start()
    try:
        purge = not keep_notes

        if all_pending:
            # Just run dream — episodes already have consolidated_at = NULL
            click.echo("Running dream on all pending episodes (no reset)...")
        else:
            # Resolve episode IDs
            targets: list[str] = []

            if latest:
                all_eps = service.store.list_recent_episodes(limit=100)
                consolidated = [ep for ep in all_eps if ep.consolidated_at]
                if not consolidated:
                    raise click.ClickException("No consolidated episodes found.")
                # list_recent_episodes returns most-recent first
                targets.append(consolidated[0].id)

            for partial_id in episode_ids:
                all_eps = service.store.list_recent_episodes(limit=500)
                matches = [ep for ep in all_eps if ep.id.startswith(partial_id)]
                if not matches:
                    raise click.ClickException(
                        f"No episode found with id prefix {partial_id!r}."
                    )
                if len(matches) > 1:
                    ids = ", ".join(m.id for m in matches)
                    raise click.ClickException(
                        f"Ambiguous prefix {partial_id!r} matches {len(matches)} episodes: {ids}"
                    )
                targets.append(matches[0].id)

            # Deduplicate while preserving order
            seen: set[str] = set()
            targets = [t for t in targets if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

            for ep_id in targets:
                ep = service.store.get_episode(ep_id)
                label = (ep.title or ep_id[:12]) if ep else ep_id[:12]
                deleted = service.store.reset_episode_for_redream(ep_id, purge_notes=purge)
                if purge:
                    click.echo(f"Reset {ep_id[:12]}  ({label})  — {deleted} notes purged")
                else:
                    click.echo(f"Reset {ep_id[:12]}  ({label})  — notes kept")

        # Run dream
        report = service.dream(trigger=trigger)
        click.echo(
            f"\ndream {report.log_id}: episodes={report.episodes_processed} "
            f"notes_added={report.notes_added} pruned={report.notes_pruned}"
        )
        click.echo("--- journal ---")
        click.echo(report.journal)

    finally:
        service.stop()


@main.command()
@click.pass_obj
def profile(config):
    """Print the current profile."""
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
    """Bulk-import past Cowork / Claude Code chat transcripts as turns.

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


@main.command()
@click.option("--suite", "suite_path", type=click.Path(path_type=Path), default=None,
              help="Path to a YAML eval suite. Defaults to evals/default.yaml in the repo root.")
@click.option("--k", default=None, type=int,
              help="Override k for all cases (per-case k in YAML wins if set).")
@click.option("--depth", default=None, type=click.Choice(["fast", "deep", "verbatim"]),
              help="Override depth for all cases.")
@click.option("--run-id", default=None,
              help="Stable run identifier (UUID). Defaults to a fresh UUID4 per invocation.")
@click.option("--no-persist", is_flag=True,
              help="Skip writing results to the eval_results table.")
@click.pass_obj
def eval(config, suite_path, k, depth, run_id, no_persist):
    """Run the recall eval suite and report recall@k.

    Exits 0 if all cases pass, 1 if any case fails — suitable for CI gates.
    """
    import sqlite3 as _sqlite3
    from uuid import uuid4
    from ai_memory.eval import load_suite, run_suite
    from ai_memory.core.models import EvalResult
    from ai_memory.timestamps import now_iso

    # --- Resolve suite path -------------------------------------------------
    if suite_path is None:
        # Try repo-relative first, then $AI_MEMORY_HOME/evals/
        repo_default = Path(__file__).parent.parent / "evals" / "default.yaml"
        home_default = config.home / "evals" / "default.yaml"
        if repo_default.exists():
            suite_path = repo_default
        elif home_default.exists():
            suite_path = home_default
        else:
            raise click.UsageError(
                f"No suite found at {repo_default} or {home_default}. "
                "Pass --suite <path> to specify one explicitly."
            )

    suite = load_suite(suite_path)
    run_id = run_id or str(uuid4())

    service = MemoryService.build(config)
    service.start()

    results: list[EvalResult] = []
    case_width = max((len(c.id) for c in suite.cases), default=20)

    try:
        for cr in run_suite(suite, service, k_override=k, depth_override=depth):
            status = "PASS" if cr.passed else "FAIL"
            effective_k = k if k is not None else cr.case.k
            effective_depth = depth if depth is not None else cr.case.depth
            click.echo(
                f"{status}  [{effective_depth} k={effective_k:2d} {cr.latency_ms:4d}ms]"
                f"  {cr.case.id:{case_width}s}  {cr.case.query[:60]}"
            )

            er = EvalResult(
                id=str(uuid4()),
                run_id=run_id,
                suite=suite.suite,
                case_id=cr.case.id,
                query=cr.case.query,
                expected=cr.case.expected,
                k=effective_k,
                depth=effective_depth,
                passed=cr.passed,
                hits_count=cr.hits_count,
                top_hit_text=(cr.top_hit_text or "")[:500] if cr.top_hit_text else None,
                latency_ms=cr.latency_ms,
                run_at=now_iso(),
            )
            results.append(er)

    finally:
        service.stop()

    # --- Persist ------------------------------------------------------------
    if not no_persist and results:
        conn = _sqlite3.connect(config.database_path)
        for er in results:
            conn.execute(
                "INSERT INTO eval_results("
                "  id,run_id,suite,case_id,query,expected,"
                "  k,depth,passed,hits_count,top_hit_text,latency_ms,run_at"
                ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    er.id, er.run_id, er.suite, er.case_id, er.query, er.expected,
                    er.k, er.depth, int(er.passed), er.hits_count,
                    er.top_hit_text, er.latency_ms, er.run_at,
                ),
            )
        conn.commit()
        conn.close()

    # --- Summary ------------------------------------------------------------
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pct = 100.0 * passed / total if total else 0.0

    click.echo()
    click.echo("--- eval summary ---")
    click.echo(f"suite:     {suite.suite}  ({total} cases)")
    click.echo(f"run_id:    {run_id}")
    click.echo(f"passed:    {passed} / {total}")
    click.echo(f"recall@k:  {pct:.1f}%")
    if not no_persist and results:
        click.echo(f"persisted: {config.database_path.name}::eval_results")
    else:
        click.echo("(not persisted)")

    sys.exit(0 if passed == total else 1)


@main.command("backfill-entities")
@click.option("--batch-size", default=20, type=int,
              help="Notes per LLM call. Larger = fewer calls but slower per call.")
@click.option("--source-filter", default=None,
              help="Only backfill notes whose tags LIKE this value (e.g. 'bootstrap').")
@click.option("--dry-run", is_flag=True,
              help="Print what would be updated without writing to the DB.")
@click.pass_obj
def backfill_entities(config, batch_size, source_filter, dry_run):
    """Backfill entity tags on notes that have none.

    Sends notes in batches to the LLM and writes the extracted entity slugs
    back to the DB. Useful after importing bootstrap notes, which bypass the
    dream-cycle entity extraction.

    Examples:\n
      ai-memory backfill-entities --source-filter bootstrap\n
      ai-memory backfill-entities --dry-run
    """
    import json as _json
    from ai_memory.core.dreaming import _normalise_entity

    _ENTITY_SYSTEM = (
        "You extract named entities from short memory facts. "
        "You will receive a JSON array of objects, each with 'id' and 'text'. "
        "For EACH, identify 0-5 NAMED things the fact explicitly mentions. "
        "Valid entities: specific tools, technologies, frameworks, companies, "
        "projects, geographic places (cities, countries), proper nouns, and "
        "named activities with a recognised proper name (e.g. 'orienteering', "
        "'speleology', 'scuba-diving'). "
        "NOT entities: generic concepts (efficiency, learning, competence), "
        "abstract qualities (direct, concise, clarity), common nouns (tea, music, "
        "travel), job titles (technical-lead), or vague terms (automation, environment). "
        "Normalise each to a lowercase-hyphen-slug (e.g. 'postgresql', 'api-gateway', 'london'). "
        "Output ONLY a JSON array: "
        '[{"id": "...", "entities": ["slug1"]}]. '
        "Empty array [] when no named entity is present. No other text."
    )

    service = MemoryService.build(config)
    service.start()
    try:
        # Load entity vocab for fuzzy normalisation
        entity_vocab: list[str] = service.store.list_entity_vocab()

        # Fetch notes with no entities
        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(config.database_path)
        conn.row_factory = _sqlite3.Row
        q = (
            "SELECT id, text, tags FROM notes "
            "WHERE valid_to IS NULL "
            "  AND (entities IS NULL OR entities = '[]' OR entities = '')"
        )
        if source_filter:
            q += f" AND tags LIKE '%{source_filter}%'"
        q += " ORDER BY text"
        rows = conn.execute(q).fetchall()
        conn.close()

        if not rows:
            click.echo("No notes need entity backfill.")
            return

        click.echo(f"Notes to backfill: {len(rows)} (batch_size={batch_size})")

        total_updated = 0
        total_skipped = 0

        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start: batch_start + batch_size]
            payload = [{"id": r["id"], "text": r["text"]} for r in batch]

            try:
                completion = service.llm.complete(
                    system=_ENTITY_SYSTEM,
                    messages=[
                        __import__("ai_memory.llm.interface", fromlist=["Message"])
                        .Message(role="user", content=_json.dumps(payload, ensure_ascii=False))
                    ],
                    max_tokens=1024,
                )
                raw = _json.loads(completion.text.strip())
            except Exception as exc:
                click.echo(f"  batch {batch_start}: LLM/parse error — {exc}", err=True)
                continue

            # Index by id
            result_map = {item["id"]: item.get("entities", []) for item in raw
                          if isinstance(item, dict) and "id" in item}

            conn = _sqlite3.connect(config.database_path)
            for row in batch:
                raw_entities = result_map.get(row["id"], [])
                normalised = list({
                    norm for e in raw_entities
                    if (norm := _normalise_entity(str(e), entity_vocab))
                })
                if not normalised:
                    total_skipped += 1
                    continue
                if dry_run:
                    click.echo(f"  DRY  {row['id'][:8]}  {row['text'][:60]}  -> {normalised}")
                else:
                    conn.execute(
                        "UPDATE notes SET entities = ? WHERE id = ?",
                        (_json.dumps(normalised), row["id"]),
                    )
                    click.echo(f"  SET  {row['id'][:8]}  {row['text'][:60]}  -> {normalised}")
                total_updated += 1
            if not dry_run:
                conn.commit()
            conn.close()

        action = "Would update" if dry_run else "Updated"
        click.echo(f"\n{action} {total_updated} notes  ({total_skipped} had no extractable entities)")

    finally:
        service.stop()


if __name__ == "__main__":
    main()
