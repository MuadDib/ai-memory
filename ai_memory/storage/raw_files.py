"""
Raw transcript storage: append-only JSONL files on disk, byte-offset indexed.

One file per episode, located at `<raw_dir>/<yyyy>/<mm>/<episode_id>.jsonl`.
Each line is a JSON object describing one turn:
    {"id": ..., "role": ..., "ts": ..., "text": ...}

The MemoryStore (SQLite) keeps an index of (episode_id, raw_file, byte_offset,
byte_length) per turn so we can seek and read a verbatim turn in O(1).

Why JSONL on disk instead of putting raw text in SQLite?
    - Append-only is trivial; no transactional contention with derived index
      writes.
    - Greppable, taillable, backup-able with regular file tools.
    - Source of truth is plain text — derived layers can be rebuilt.
    - SQLite stays small and fast; the multi-GB worth of raw conversation
      doesn't bloat its caches.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import IO


@dataclass(frozen=True)
class TurnAppendResult:
    """Returned by `append_turn` so the caller can populate a Turn row in the index."""

    raw_file: str  # relative path under raw_dir
    byte_offset: int
    byte_length: int


class RawTranscriptStore:
    """Append-only JSONL storage for raw conversation turns.

    Files are partitioned by year/month to keep directory listings reasonable
    over multi-year usage. The `raw_file` returned to the caller is always
    a relative path (POSIX-style with forward slashes) so it round-trips
    cleanly across operating systems.
    """

    def __init__(self, raw_dir: Path) -> None:
        self._raw_dir = raw_dir

    # --- Writes ----------------------------------------------------------

    def append_turn(
        self, episode_id: str, started_at: str, payload: dict
    ) -> TurnAppendResult:
        """Append one JSONL line to the episode's raw file. Returns offset/length."""
        rel_path = self._relative_path_for(episode_id, started_at)
        full_path = self._raw_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        encoded = line.encode("utf-8")

        with full_path.open("ab") as fh:
            fh.seek(0, 2)  # end of file
            offset = fh.tell()
            fh.write(encoded)

        return TurnAppendResult(
            raw_file=rel_path.as_posix(),
            byte_offset=offset,
            byte_length=len(encoded),
        )

    # --- Reads -----------------------------------------------------------

    def read_turn(self, raw_file: str, byte_offset: int, byte_length: int) -> dict:
        """Read one turn back, given the indexed location."""
        full_path = self._raw_dir / raw_file
        with full_path.open("rb") as fh:
            fh.seek(byte_offset)
            blob = fh.read(byte_length)
        return json.loads(blob.decode("utf-8"))

    def read_episode(self, raw_file: str) -> list[dict]:
        """Read every turn in an episode file (use for rebuild / export)."""
        full_path = self._raw_dir / raw_file
        with full_path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def open_for_streaming_writes(self, episode_id: str, started_at: str) -> "_TurnWriter":
        """Open a streaming writer for an episode — use when ingesting many turns at once.

        The writer keeps the file handle open across calls (one fewer fsync per turn)
        and is a context manager so it cleans up cleanly on exception.
        """
        rel_path = self._relative_path_for(episode_id, started_at)
        full_path = self._raw_dir / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return _TurnWriter(full_path, rel_path.as_posix())

    # --- Helpers ---------------------------------------------------------

    @staticmethod
    def _relative_path_for(episode_id: str, started_at: str) -> Path:
        # started_at is ISO 8601 UTC e.g. '2026-04-29T18:35:23Z' — take YYYY/MM directly.
        year, month = started_at[:4], started_at[5:7]
        return Path(year) / month / f"{episode_id}.jsonl"


class _TurnWriter:
    """Streaming-write context manager for a single episode file."""

    def __init__(self, full_path: Path, rel_path: str) -> None:
        self._full_path = full_path
        self._rel_path = rel_path
        self._fh: IO[bytes] | None = None

    def __enter__(self) -> "_TurnWriter":
        self._fh = self._full_path.open("ab")
        self._fh.seek(0, 2)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def append(self, payload: dict) -> TurnAppendResult:
        if self._fh is None:
            raise RuntimeError("Writer not opened (use as context manager).")
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        encoded = line.encode("utf-8")
        offset = self._fh.tell()
        self._fh.write(encoded)
        return TurnAppendResult(
            raw_file=self._rel_path,
            byte_offset=offset,
            byte_length=len(encoded),
        )
