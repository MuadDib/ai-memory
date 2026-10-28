"""WAL-safe online backup of memory.db.

Copying memory.db as a file misses whatever still sits in memory.db-wal (often
hours of writes), and can tear if a checkpoint runs mid-copy. SQLite's backup
API reads a consistent snapshot through the WAL, so it's safe while the daemon
and MCP servers are running.

Usage: python scripts/backup_db.py [home]   (home defaults to AI_MEMORY_HOME or C:\\ai-mem\\data)
"""
from __future__ import annotations

import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    home = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("AI_MEMORY_HOME", r"C:\ai-mem\data"))
    src = home / "memory.db"
    dst = home / f"memory.db.bak-{datetime.now():%Y%m%d-%H%M%S}"

    source = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True)
    target = sqlite3.connect(dst)
    try:
        source.backup(target)
        check = target.execute("PRAGMA quick_check").fetchone()[0]
        notes = target.execute("SELECT count(*) FROM notes").fetchone()[0]
    finally:
        target.close()
        source.close()

    print(f"{dst}  ({dst.stat().st_size / 1048576:.1f} MB, quick_check={check}, notes={notes})")
    return 0 if check == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
