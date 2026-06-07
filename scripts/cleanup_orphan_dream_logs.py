"""Cleanup orphaned rows in the ``dream_log`` table.

An ORPHAN row is a dream pass that started (``started_at`` set) but never
finished (``ended_at IS NULL``). A bug caused one oversized episode to
"poison-pill" every dream pass, so each pass started, immediately failed,
and left behind an empty row with no ``ended_at``. Over time this produced
roughly 50,000 orphan rows that pollute the ``dream_log`` table. Rows that
have a non-null ``ended_at`` represent real, completed dream passes and
must always be kept.

This script can report on (dry run) or delete (``--apply``) those orphan
rows.
"""

import argparse
import sqlite3
import sys
from pathlib import Path


def delete_orphans(conn: sqlite3.Connection) -> int:
    """Delete rows from ``dream_log`` where ``ended_at IS NULL``.

    Commits the transaction and returns the number of rows deleted.
    """
    cursor = conn.execute("DELETE FROM dream_log WHERE ended_at IS NULL")
    conn.commit()
    return cursor.rowcount


def count_states(conn: sqlite3.Connection) -> tuple[int, int]:
    """Return ``(total_rows, orphan_rows)`` for the ``dream_log`` table."""
    total_rows = conn.execute("SELECT COUNT(*) FROM dream_log").fetchone()[0]
    orphan_rows = conn.execute(
        "SELECT COUNT(*) FROM dream_log WHERE ended_at IS NULL"
    ).fetchone()[0]
    return total_rows, orphan_rows


def _resolve_db_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    from ai_memory.config import load_config

    return load_config().database_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report on or delete orphaned dream_log rows (ended_at IS NULL)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually delete orphan rows. Without this flag, performs a dry run.",
    )
    parser.add_argument(
        "--db",
        metavar="PATH",
        default=None,
        help="Explicit path to the sqlite database file. "
        "If omitted, resolved via ai_memory.config.load_config().database_path",
    )
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db)

    if not db_path.exists():
        print(f"ERROR: database file does not exist: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        total_before, orphans_before = count_states(conn)
        print(f"Database: {db_path}")
        print(f"Total dream_log rows: {total_before}")
        print(f"Orphan rows (ended_at IS NULL): {orphans_before}")

        if not args.apply:
            print(
                f"DRY RUN: would delete {orphans_before} orphan row(s). "
                "No changes made. Re-run with --apply to actually delete."
            )
            return 0

        deleted = delete_orphans(conn)
        total_after, orphans_after = count_states(conn)
        print(f"Deleted {deleted} orphan row(s).")
        print(f"Total dream_log rows now: {total_after}")
        print(f"Orphan rows remaining: {orphans_after}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
