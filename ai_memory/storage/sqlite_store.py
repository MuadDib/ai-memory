"""
SQLite + sqlite-vec + FTS5 implementation of MemoryStore.

Schema is defined inline (not via migrations yet) — version 0.1 is greenfield.
First production change after release adds an Alembic-style migration runner.

The vector column is stored in a dedicated `*_vec` virtual table created by
sqlite-vec. The FTS5 index is a content-shadow table on `notes`. We keep both
in sync via explicit upserts in `insert_note` / `update_note` — there is no
implicit trigger magic, so behaviour is portable.
"""
from __future__ import annotations

import json
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any

import sqlite_vec

from ai_memory.core.models import CoworkImportState, DreamLog, Episode, EvalResult, Note, Profile, Turn

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
  version    INTEGER PRIMARY KEY,
  applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS profile (
  key        TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  source     TEXT
);

CREATE TABLE IF NOT EXISTS notes (
  id                    TEXT PRIMARY KEY,
  text                  TEXT NOT NULL,
  tags                  TEXT,                         -- JSON array
  source_episode_ids    TEXT,                         -- JSON array
  valid_from            TEXT NOT NULL,
  valid_to              TEXT,
  ingested_at           TEXT NOT NULL,
  superseded_by         TEXT REFERENCES notes(id),
  contradicts           TEXT,                         -- JSON array
  promoted_to_profile   INTEGER NOT NULL DEFAULT 0,
  embedding_model       TEXT NOT NULL,
  access_count          INTEGER NOT NULL DEFAULT 0,
  last_accessed_at      TEXT,
  entities              TEXT NOT NULL DEFAULT '[]'  -- JSON array of lowercase-slug entity tags
);

CREATE INDEX IF NOT EXISTS notes_valid_to     ON notes(valid_to);
CREATE INDEX IF NOT EXISTS notes_ingested_at  ON notes(ingested_at);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
  text,
  content='notes',
  content_rowid='rowid',
  tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS episodes (
  id                TEXT PRIMARY KEY,
  title             TEXT,
  summary           TEXT NOT NULL,
  source            TEXT NOT NULL,
  started_at        TEXT NOT NULL,
  ended_at          TEXT,
  raw_file          TEXT NOT NULL,
  embedding_model   TEXT NOT NULL,
  consolidated_at   TEXT
);

CREATE INDEX IF NOT EXISTS episodes_started_at ON episodes(started_at);
CREATE INDEX IF NOT EXISTS episodes_consolidated_at ON episodes(consolidated_at);

CREATE TABLE IF NOT EXISTS turns (
  id          TEXT PRIMARY KEY,
  episode_id  TEXT NOT NULL REFERENCES episodes(id),
  raw_file    TEXT NOT NULL,
  byte_offset INTEGER NOT NULL,
  byte_length INTEGER NOT NULL,
  role        TEXT NOT NULL,
  ts          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS turns_by_episode ON turns(episode_id, ts);
CREATE INDEX IF NOT EXISTS turns_ts         ON turns(ts);

CREATE TABLE IF NOT EXISTS dream_log (
  id                          TEXT PRIMARY KEY,
  started_at                  TEXT NOT NULL,
  ended_at                    TEXT,
  trigger                     TEXT NOT NULL,
  episodes_processed          INTEGER NOT NULL DEFAULT 0,
  notes_added                 INTEGER NOT NULL DEFAULT 0,
  notes_invalidated           INTEGER NOT NULL DEFAULT 0,
  notes_promoted_to_profile   INTEGER NOT NULL DEFAULT 0,
  notes_pruned                INTEGER NOT NULL DEFAULT 0,
  llm_tokens_used             INTEGER NOT NULL DEFAULT 0,
  llm_cost_usd                REAL NOT NULL DEFAULT 0,
  journal                     TEXT
);

CREATE TABLE IF NOT EXISTS cowork_import_state (
  session_id        TEXT PRIMARY KEY,
  last_turn_id      TEXT NOT NULL,
  last_byte_offset  INTEGER NOT NULL,
  last_imported_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
  id            TEXT PRIMARY KEY,   -- UUID4 per row
  run_id        TEXT NOT NULL,      -- UUID4 shared across one CLI invocation
  suite         TEXT NOT NULL,
  case_id       TEXT NOT NULL,
  query         TEXT NOT NULL,
  expected      TEXT NOT NULL,
  k             INTEGER NOT NULL,
  depth         TEXT NOT NULL,
  passed        INTEGER NOT NULL,   -- 1 = pass, 0 = fail
  hits_count    INTEGER NOT NULL,
  top_hit_text  TEXT,
  latency_ms    INTEGER NOT NULL,
  run_at        TEXT NOT NULL       -- ISO 8601 UTC
);

CREATE INDEX IF NOT EXISTS eval_results_run_id  ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS eval_results_case_id ON eval_results(case_id, run_at);
"""

SCHEMA_VERSION = 4

# Migration SQL run once when upgrading from schema version 1 (Unix int
# timestamps) to version 2 (ISO 8601 UTC strings).  The WHERE guards make
# each statement idempotent — rows already containing 'T' are skipped.
_MIGRATION_V1_TO_V2 = """
UPDATE schema_version
   SET applied_at = REPLACE(datetime(CAST(applied_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
 WHERE applied_at NOT LIKE '%T%';

UPDATE profile
   SET updated_at = REPLACE(datetime(CAST(updated_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
 WHERE updated_at NOT LIKE '%T%';

UPDATE notes
   SET valid_from       = REPLACE(datetime(CAST(valid_from AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z',
       ingested_at      = REPLACE(datetime(CAST(ingested_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z',
       valid_to         = CASE WHEN valid_to IS NOT NULL AND valid_to NOT LIKE '%T%'
                               THEN REPLACE(datetime(CAST(valid_to AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
                               ELSE valid_to END,
       last_accessed_at = CASE WHEN last_accessed_at IS NOT NULL AND last_accessed_at NOT LIKE '%T%'
                               THEN REPLACE(datetime(CAST(last_accessed_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
                               ELSE last_accessed_at END
 WHERE valid_from NOT LIKE '%T%';

UPDATE episodes
   SET started_at      = REPLACE(datetime(CAST(started_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z',
       ended_at        = CASE WHEN ended_at IS NOT NULL AND ended_at NOT LIKE '%T%'
                              THEN REPLACE(datetime(CAST(ended_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
                              ELSE ended_at END,
       consolidated_at = CASE WHEN consolidated_at IS NOT NULL AND consolidated_at NOT LIKE '%T%'
                              THEN REPLACE(datetime(CAST(consolidated_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
                              ELSE consolidated_at END
 WHERE started_at NOT LIKE '%T%';

UPDATE turns
   SET ts = REPLACE(datetime(CAST(ts AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
 WHERE ts NOT LIKE '%T%';

UPDATE dream_log
   SET started_at = REPLACE(datetime(CAST(started_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z',
       ended_at   = CASE WHEN ended_at IS NOT NULL AND ended_at NOT LIKE '%T%'
                         THEN REPLACE(datetime(CAST(ended_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
                         ELSE ended_at END
 WHERE started_at NOT LIKE '%T%';

UPDATE cowork_import_state
   SET last_imported_at = REPLACE(datetime(CAST(last_imported_at AS INTEGER), 'unixepoch'), ' ', 'T') || 'Z'
 WHERE last_imported_at NOT LIKE '%T%';
"""


# Migration SQL run once when upgrading from schema version 2 to version 3.
# Adds the eval_results table to existing databases; idempotent via IF NOT EXISTS.
_MIGRATION_V2_TO_V3 = """
CREATE TABLE IF NOT EXISTS eval_results (
  id            TEXT PRIMARY KEY,
  run_id        TEXT NOT NULL,
  suite         TEXT NOT NULL,
  case_id       TEXT NOT NULL,
  query         TEXT NOT NULL,
  expected      TEXT NOT NULL,
  k             INTEGER NOT NULL,
  depth         TEXT NOT NULL,
  passed        INTEGER NOT NULL,
  hits_count    INTEGER NOT NULL,
  top_hit_text  TEXT,
  latency_ms    INTEGER NOT NULL,
  run_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS eval_results_run_id  ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS eval_results_case_id ON eval_results(case_id, run_at);
"""

# Migration v3→v4: adds entities column to notes (domain-split tag taxonomy).
_MIGRATION_V3_TO_V4 = """
ALTER TABLE notes ADD COLUMN entities TEXT NOT NULL DEFAULT '[]';
"""


def _fts5_escape(query: str) -> str:
    """Convert a free-text query into a safe FTS5 MATCH expression.

    FTS5 has its own mini query language where punctuation has meaning
    (apostrophes, quotes, parens, AND/OR/NOT keywords, NEAR(), etc.). When we
    just want a bag-of-words search over user-typed text we have to neutralise
    all of that.

    Strategy: pull out alphanumeric tokens via regex, drop everything else,
    and wrap each token in double-quotes so it's interpreted as a phrase.
    Multiple phrases are AND'd by FTS5 default semantics, which matches
    intuitive search behaviour.
    """
    import re
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    if not tokens:
        return ""
    return " ".join(f'"{t}"' for t in tokens)


def _pack_vec(vec: list[float]) -> bytes:
    """Pack a list of floats into the binary format sqlite-vec expects."""
    return struct.pack(f"{len(vec)}f", *vec)


class SqliteStore:
    """SQLite-backed MemoryStore using sqlite-vec for vector search and FTS5 for BM25.

    Conforms to the MemoryStore Protocol structurally; not declared explicitly
    (Protocols are duck-typed) so we keep import overhead minimal.
    """

    def __init__(self, db_path: Path, embedding_dim: int) -> None:
        """Open (or create) the database at `db_path` configured for `embedding_dim` vectors."""
        self._db_path = db_path
        self._embedding_dim = embedding_dim
        self._conn: sqlite3.Connection | None = None
        # Reentrant lock: sqlite3.Connection is NOT thread-safe. Multiple
        # anyio worker threads (concurrent recalls, simultaneous remember+recall)
        # must serialise all access through this lock.
        self._lock = threading.RLock()

    # --- Lifecycle -------------------------------------------------------

    def initialise(self) -> None:
        """Open the connection, load sqlite-vec, create schema, ensure vec/fts tables."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS notes_vec "
            f"USING vec0(note_id TEXT PRIMARY KEY, embedding float[{self._embedding_dim}])"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS episodes_vec "
            f"USING vec0(episode_id TEXT PRIMARY KEY, embedding float[{self._embedding_dim}])"
        )
        # Run v1→v2 migration if the DB is on the old Unix-int timestamp schema.
        current = conn.execute(
            "SELECT MAX(version) AS v FROM schema_version"
        ).fetchone()
        current_version = current["v"] if current and current["v"] is not None else 0
        if current_version < 2:
            conn.executescript(_MIGRATION_V1_TO_V2)
        if current_version < 3:
            conn.executescript(_MIGRATION_V2_TO_V3)
        if current_version < 4:
            # Only ALTER when upgrading an existing DB; fresh DBs already have
            # the column from SCHEMA_SQL, so guard against "duplicate column".
            existing_cols = {
                r[1] for r in conn.execute("PRAGMA table_info(notes)").fetchall()
            }
            if "entities" not in existing_cols:
                conn.executescript(_MIGRATION_V3_TO_V4)

        # Record schema version (idempotent insert)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version(version, applied_at)"
            " VALUES (?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))",
            (SCHEMA_VERSION,),
        )
        conn.commit()
        self._conn = conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteStore.initialise() must be called before use.")
        return self._conn

    # --- Profile ---------------------------------------------------------

    def upsert_profile(self, profile: Profile) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO profile(key, value, updated_at, source)
                VALUES(:key, :value, :updated_at, :source)
                ON CONFLICT(key) DO UPDATE SET
                  value      = excluded.value,
                  updated_at = excluded.updated_at,
                  source     = excluded.source
                """,
                profile.__dict__,
            )
            self.conn.commit()

    def delete_profile(self, key: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM profile WHERE key = ?", (key,))
            self.conn.commit()

    def list_profile(self) -> list[Profile]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM profile ORDER BY key").fetchall()
        return [_row_to_profile(r) for r in rows]

    # --- Notes -----------------------------------------------------------

    def insert_note(self, note: Note, embedding: list[float]) -> None:
        with self._lock:
            cur = self.conn.execute(
                """
                INSERT INTO notes(
                    id, text, tags, source_episode_ids, valid_from, valid_to,
                    ingested_at, superseded_by, contradicts, promoted_to_profile,
                    embedding_model, access_count, last_accessed_at, entities
                ) VALUES (
                    :id, :text, :tags, :source_episode_ids, :valid_from, :valid_to,
                    :ingested_at, :superseded_by, :contradicts, :promoted_to_profile,
                    :embedding_model, :access_count, :last_accessed_at, :entities
                )
                """,
                _note_to_row(note),
            )
            self.conn.execute(
                "INSERT INTO notes_fts(rowid, text) VALUES (?, ?)", (cur.lastrowid, note.text)
            )
            self.conn.execute(
                "INSERT INTO notes_vec(note_id, embedding) VALUES (?, ?)",
                (note.id, _pack_vec(embedding)),
            )
            self.conn.commit()

    def update_note(self, note: Note) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE notes SET
                    text = :text,
                    tags = :tags,
                    source_episode_ids = :source_episode_ids,
                    valid_from = :valid_from,
                    valid_to = :valid_to,
                    superseded_by = :superseded_by,
                    contradicts = :contradicts,
                    promoted_to_profile = :promoted_to_profile,
                    access_count = :access_count,
                    last_accessed_at = :last_accessed_at,
                    entities = :entities
                WHERE id = :id
                """,
                _note_to_row(note),
            )
            # Keep FTS in sync (rebuild row for this id)
            cur = self.conn.execute("SELECT rowid FROM notes WHERE id = ?", (note.id,)).fetchone()
            if cur:
                self.conn.execute("DELETE FROM notes_fts WHERE rowid = ?", (cur["rowid"],))
                self.conn.execute(
                    "INSERT INTO notes_fts(rowid, text) VALUES (?, ?)", (cur["rowid"], note.text)
                )
            self.conn.commit()

    def get_note(self, note_id: str) -> Note | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return _row_to_note(row) if row else None

    def search_notes_vector(
        self, embedding: list[float], k: int, only_valid: bool = True
    ) -> list[tuple[Note, float]]:
        valid_clause = "AND notes.valid_to IS NULL" if only_valid else ""
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT notes.*, notes_vec.distance AS distance
                FROM notes_vec
                JOIN notes ON notes.id = notes_vec.note_id
                WHERE notes_vec.embedding MATCH ?
                  AND k = ?
                  {valid_clause}
                ORDER BY notes_vec.distance
                """,
                (_pack_vec(embedding), k),
            ).fetchall()
        return [(_row_to_note(r), float(r["distance"])) for r in rows]

    def search_notes_bm25(
        self, query: str, k: int, only_valid: bool = True
    ) -> list[tuple[Note, float]]:
        fts_query = _fts5_escape(query)
        if not fts_query:
            return []  # nothing alphanumeric -> no BM25 hits
        valid_clause = "AND notes.valid_to IS NULL" if only_valid else ""
        with self._lock:
            rows = self.conn.execute(
                f"""
                SELECT notes.*, bm25(notes_fts) AS rank
                FROM notes_fts
                JOIN notes ON notes.rowid = notes_fts.rowid
                WHERE notes_fts MATCH ?
                  {valid_clause}
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, k),
            ).fetchall()
        return [(_row_to_note(r), float(r["rank"])) for r in rows]

    def invalidate_note(self, note_id: str, when: str, superseded_by: str | None) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE notes SET valid_to = ?, superseded_by = ? WHERE id = ?",
                (when, superseded_by, note_id),
            )
            self.conn.commit()

    def bump_note_access(self, note_id: str, when: str) -> None:
        """Bump access_count + last_accessed_at when a note is reused (e.g. dedup hit)."""
        with self._lock:
            self.conn.execute(
                "UPDATE notes SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?",
                (when, note_id),
            )
            self.conn.commit()

    def list_valid_notes(self) -> list[Note]:
        """Every note with valid_to IS NULL — used by the promotion clustering pass."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM notes WHERE valid_to IS NULL ORDER BY ingested_at"
            ).fetchall()
        return [_row_to_note(r) for r in rows]

    def list_entity_vocab(self) -> list[str]:
        """Return all distinct entity slugs across valid notes, sorted.

        Used by the dream cycle to inject existing vocabulary into the extract
        prompt so the LLM prefers reusing known entity names.
        """
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT DISTINCT je.value AS entity
                FROM notes, json_each(notes.entities) AS je
                WHERE notes.valid_to IS NULL
                  AND je.value != ''
                ORDER BY je.value
                """
            ).fetchall()
        return [str(r["entity"]) for r in rows]

    def get_note_embedding(self, note_id: str) -> list[float] | None:
        """Read back the stored vector for a note. Used to seed clustering against existing rows."""
        with self._lock:
            row = self.conn.execute(
                "SELECT embedding FROM notes_vec WHERE note_id = ?", (note_id,)
            ).fetchone()
        if row is None:
            return None
        blob = row["embedding"]
        if blob is None:
            return None
        # sqlite-vec stores as packed float32 little-endian.
        return list(struct.unpack(f"{len(blob) // 4}f", blob))

    # --- Episodes --------------------------------------------------------

    def insert_episode(self, episode: Episode, embedding: list[float] | None = None) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO episodes(
                    id, title, summary, source, started_at, ended_at,
                    raw_file, embedding_model, consolidated_at
                ) VALUES (
                    :id, :title, :summary, :source, :started_at, :ended_at,
                    :raw_file, :embedding_model, :consolidated_at
                )
                """,
                episode.__dict__,
            )
            if embedding is not None:
                self.conn.execute(
                    "INSERT INTO episodes_vec(episode_id, embedding) VALUES (?, ?)",
                    (episode.id, _pack_vec(embedding)),
                )
            self.conn.commit()

    def update_episode(self, episode: Episode, embedding: list[float] | None = None) -> None:
        """Replace an episode row in place; optionally re-embed its summary.

        Used by the dream cycle when Phase 3 attaches a real summary/title to a
        bootstrapped episode that was opened with empty fields.
        """
        with self._lock:
            self.conn.execute(
                """
                UPDATE episodes SET
                    title           = :title,
                    summary         = :summary,
                    source          = :source,
                    started_at      = :started_at,
                    ended_at        = :ended_at,
                    raw_file        = :raw_file,
                    embedding_model = :embedding_model,
                    consolidated_at = :consolidated_at
                WHERE id = :id
                """,
                episode.__dict__,
            )
            if embedding is not None:
                # Drop and re-insert the vector row — sqlite-vec doesn't expose UPDATE.
                self.conn.execute(
                    "DELETE FROM episodes_vec WHERE episode_id = ?", (episode.id,)
                )
                self.conn.execute(
                    "INSERT INTO episodes_vec(episode_id, embedding) VALUES (?, ?)",
                    (episode.id, _pack_vec(embedding)),
                )
            self.conn.commit()

    def get_episode(self, episode_id: str) -> Episode | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
        return _row_to_episode(row) if row else None

    def list_recent_episodes(self, limit: int) -> list[Episode]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def search_episodes_vector(
        self, embedding: list[float], k: int
    ) -> list[tuple[Episode, float]]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT episodes.*, episodes_vec.distance AS distance
                FROM episodes_vec
                JOIN episodes ON episodes.id = episodes_vec.episode_id
                WHERE episodes_vec.embedding MATCH ?
                  AND k = ?
                ORDER BY episodes_vec.distance
                """,
                (_pack_vec(embedding), k),
            ).fetchall()
        return [(_row_to_episode(r), float(r["distance"])) for r in rows]

    def episodes_since(self, ts: str) -> list[Episode]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM episodes WHERE started_at >= ? ORDER BY started_at", (ts,)
            ).fetchall()
        return [_row_to_episode(r) for r in rows]

    def mark_episode_consolidated(self, episode_id: str, when: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE episodes SET consolidated_at = ? WHERE id = ?", (when, episode_id)
            )
            self.conn.commit()

    # --- Turns -----------------------------------------------------------

    def insert_turn(self, turn: Turn) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO turns(id, episode_id, raw_file, byte_offset, byte_length, role, ts)
                VALUES (:id, :episode_id, :raw_file, :byte_offset, :byte_length, :role, :ts)
                """,
                turn.__dict__,
            )
            self.conn.commit()

    def get_turns_for_episode(self, episode_id: str) -> list[Turn]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM turns WHERE episode_id = ? ORDER BY ts", (episode_id,)
            ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def count_turns_since(self, ts: str) -> int:
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) AS n FROM turns WHERE ts >= ?", (ts,)).fetchone()
        return int(row["n"]) if row else 0

    # --- Dream log -------------------------------------------------------

    def insert_dream_log(self, entry: DreamLog) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO dream_log(
                    id, started_at, ended_at, trigger,
                    episodes_processed, notes_added, notes_invalidated,
                    notes_promoted_to_profile, notes_pruned,
                    llm_tokens_used, llm_cost_usd, journal
                ) VALUES (
                    :id, :started_at, :ended_at, :trigger,
                    :episodes_processed, :notes_added, :notes_invalidated,
                    :notes_promoted_to_profile, :notes_pruned,
                    :llm_tokens_used, :llm_cost_usd, :journal
                )
                """,
                entry.__dict__,
            )
            self.conn.commit()

    def update_dream_log(self, entry: DreamLog) -> None:
        with self._lock:
            self.conn.execute(
                """
                UPDATE dream_log SET
                    ended_at = :ended_at,
                    episodes_processed = :episodes_processed,
                    notes_added = :notes_added,
                    notes_invalidated = :notes_invalidated,
                    notes_promoted_to_profile = :notes_promoted_to_profile,
                    notes_pruned = :notes_pruned,
                    llm_tokens_used = :llm_tokens_used,
                    llm_cost_usd = :llm_cost_usd,
                    journal = :journal
                WHERE id = :id
                """,
                entry.__dict__,
            )
            self.conn.commit()

    def list_recent_dream_logs(self, limit: int) -> list[DreamLog]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM dream_log ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_dream_log(r) for r in rows]

    def last_dream_completed_at(self) -> str | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT MAX(ended_at) AS last FROM dream_log WHERE ended_at IS NOT NULL"
            ).fetchone()
        return str(row["last"]) if row and row["last"] is not None else None

    # --- Cowork importer state ------------------------------------------

    def get_cowork_import_state(self, session_id: str) -> CoworkImportState | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM cowork_import_state WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return CoworkImportState(
            session_id=row["session_id"],
            last_turn_id=row["last_turn_id"],
            last_byte_offset=int(row["last_byte_offset"]),
            last_imported_at=str(row["last_imported_at"]),
        )

    def upsert_cowork_import_state(self, state: CoworkImportState) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO cowork_import_state(
                    session_id, last_turn_id, last_byte_offset, last_imported_at
                ) VALUES (
                    :session_id, :last_turn_id, :last_byte_offset, :last_imported_at
                )
                ON CONFLICT(session_id) DO UPDATE SET
                  last_turn_id     = excluded.last_turn_id,
                  last_byte_offset = excluded.last_byte_offset,
                  last_imported_at = excluded.last_imported_at
                """,
                state.__dict__,
            )
            self.conn.commit()

    # --- Eval results -------------------------------------------------------

    def insert_eval_result(self, result: EvalResult) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO eval_results(
                    id, run_id, suite, case_id, query, expected,
                    k, depth, passed, hits_count, top_hit_text, latency_ms, run_at
                ) VALUES (
                    :id, :run_id, :suite, :case_id, :query, :expected,
                    :k, :depth, :passed, :hits_count, :top_hit_text, :latency_ms, :run_at
                )
                """,
                {**result.__dict__, "passed": int(result.passed)},
            )
            self.conn.commit()


# --- Row <-> dataclass helpers (kept private) ---------------------------


def _json_or_default(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _row_to_profile(row: sqlite3.Row) -> Profile:
    return Profile(
        key=row["key"],
        value=row["value"],
        updated_at=str(row["updated_at"]),
        source=row["source"],
    )


def _note_to_row(note: Note) -> dict:
    return {
        "id": note.id,
        "text": note.text,
        "tags": json.dumps(note.tags),
        "source_episode_ids": json.dumps(note.source_episode_ids),
        "valid_from": note.valid_from,
        "valid_to": note.valid_to,
        "ingested_at": note.ingested_at,
        "superseded_by": note.superseded_by,
        "contradicts": json.dumps(note.contradicts),
        "promoted_to_profile": 1 if note.promoted_to_profile else 0,
        "embedding_model": note.embedding_model,
        "access_count": note.access_count,
        "last_accessed_at": note.last_accessed_at,
        "entities": json.dumps(note.entities),
    }


def _row_to_note(row: sqlite3.Row) -> Note:
    return Note(
        id=row["id"],
        text=row["text"],
        tags=_json_or_default(row["tags"], []),
        source_episode_ids=_json_or_default(row["source_episode_ids"], []),
        valid_from=str(row["valid_from"]),
        valid_to=str(row["valid_to"]) if row["valid_to"] is not None else None,
        ingested_at=str(row["ingested_at"]),
        superseded_by=row["superseded_by"],
        contradicts=_json_or_default(row["contradicts"], []),
        promoted_to_profile=bool(row["promoted_to_profile"]),
        embedding_model=row["embedding_model"],
        access_count=int(row["access_count"]),
        last_accessed_at=str(row["last_accessed_at"]) if row["last_accessed_at"] is not None else None,
        entities=_json_or_default(row["entities"], []),
    )


def _row_to_episode(row: sqlite3.Row) -> Episode:
    return Episode(
        id=row["id"],
        title=row["title"] or "",
        summary=row["summary"],
        source=row["source"],
        started_at=str(row["started_at"]),
        ended_at=str(row["ended_at"]) if row["ended_at"] is not None else None,
        raw_file=row["raw_file"],
        embedding_model=row["embedding_model"],
        consolidated_at=str(row["consolidated_at"]) if row["consolidated_at"] is not None else None,
    )


def _row_to_turn(row: sqlite3.Row) -> Turn:
    return Turn(
        id=row["id"],
        episode_id=row["episode_id"],
        raw_file=row["raw_file"],
        byte_offset=int(row["byte_offset"]),
        byte_length=int(row["byte_length"]),
        role=row["role"],
        ts=str(row["ts"]),
    )


def _row_to_dream_log(row: sqlite3.Row) -> DreamLog:
    return DreamLog(
        id=row["id"],
        started_at=str(row["started_at"]),
        ended_at=str(row["ended_at"]) if row["ended_at"] is not None else None,
        trigger=row["trigger"],
        episodes_processed=int(row["episodes_processed"]),
        notes_added=int(row["notes_added"]),
        notes_invalidated=int(row["notes_invalidated"]),
        notes_promoted_to_profile=int(row["notes_promoted_to_profile"]),
        notes_pruned=int(row["notes_pruned"]),
        llm_tokens_used=int(row["llm_tokens_used"]),
        llm_cost_usd=float(row["llm_cost_usd"]),
        journal=row["journal"] or "",
    )
