import sys
import pathlib
import sqlite3

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from cleanup_orphan_dream_logs import delete_orphans, count_states


def _make_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE dream_log (id TEXT, ended_at TEXT)")
    rows = [
        ("1", "2024-01-01T00:00:00"),
        ("2", "2024-01-02T00:00:00"),
        ("3", None),
        ("4", None),
        ("5", None),
    ]
    conn.executemany("INSERT INTO dream_log (id, ended_at) VALUES (?, ?)", rows)
    conn.commit()
    return conn


def test_count_states_reports_total_and_orphans():
    conn = _make_conn()
    assert count_states(conn) == (5, 3)


def test_delete_orphans_removes_only_orphan_rows():
    conn = _make_conn()
    deleted = delete_orphans(conn)
    assert deleted == 3

    remaining = conn.execute("SELECT id FROM dream_log ORDER BY id").fetchall()
    assert remaining == [("1",), ("2",)]
    assert count_states(conn) == (2, 0)
