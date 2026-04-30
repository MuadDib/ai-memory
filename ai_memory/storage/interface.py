"""
Storage interface: every backend (SQLite, Postgres, ...) implements this Protocol.

The interface is deliberately CRUD-flat — no clever query builders, no abstract
search DSLs. Search lives in the recall pipeline; the storage adapter only knows
how to fetch raw candidates and commit writes. This keeps adapters small and
easy to port.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ai_memory.core.models import CoworkImportState, DreamLog, Episode, Note, Profile, Turn


@runtime_checkable
class MemoryStore(Protocol):
    """Storage backend for indexed memory data.

    Raw transcripts (verbatim turn file content) live behind `RawTranscriptStore`,
    not here — this Protocol is for indexed/queryable rows only.
    """

    # --- Lifecycle -------------------------------------------------------

    def initialise(self) -> None:
        """Create tables, indexes, and load extensions if needed. Idempotent."""

    def close(self) -> None:
        """Release any held connections."""

    # --- Profile ---------------------------------------------------------

    def upsert_profile(self, profile: Profile) -> None: ...

    def delete_profile(self, key: str) -> None: ...

    def list_profile(self) -> list[Profile]: ...

    # --- Notes -----------------------------------------------------------

    def insert_note(self, note: Note, embedding: list[float]) -> None: ...

    def update_note(self, note: Note) -> None: ...

    def get_note(self, note_id: str) -> Note | None: ...

    def search_notes_vector(
        self, embedding: list[float], k: int, only_valid: bool = True
    ) -> list[tuple[Note, float]]:
        """Return up to k nearest notes with their distance scores."""

    def search_notes_bm25(
        self, query: str, k: int, only_valid: bool = True
    ) -> list[tuple[Note, float]]:
        """Return up to k notes by FTS5 BM25 score."""

    def invalidate_note(self, note_id: str, when: int, superseded_by: str | None) -> None: ...

    def bump_note_access(self, note_id: str, when: int) -> None:
        """Increment access_count and update last_accessed_at on a note."""

    def list_valid_notes(self) -> list[Note]:
        """Return every note with valid_to IS NULL — used by promotion clustering."""

    def get_note_embedding(self, note_id: str) -> list[float] | None:
        """Fetch the stored embedding for a note (used by clustering against existing rows)."""

    def list_entity_vocab(self) -> list[str]:
        """Return sorted list of all distinct entity slugs across valid notes.

        Injected into the dream-cycle extract prompt so the LLM prefers
        reusing known entity names over inventing new ones.
        """

    # --- Episodes --------------------------------------------------------

    def insert_episode(self, episode: Episode, embedding: list[float] | None = None) -> None: ...

    def update_episode(self, episode: Episode, embedding: list[float] | None = None) -> None:
        """Update an existing episode row (and re-embed its summary if `embedding` given)."""

    def get_episode(self, episode_id: str) -> Episode | None: ...

    def list_recent_episodes(self, limit: int) -> list[Episode]: ...

    def search_episodes_vector(
        self, embedding: list[float], k: int
    ) -> list[tuple[Episode, float]]: ...

    def episodes_since(self, ts: int) -> list[Episode]: ...

    def mark_episode_consolidated(self, episode_id: str, when: int) -> None: ...

    def reset_episode_for_redream(self, episode_id: str, purge_notes: bool = True) -> int:
        """Reset an episode so the dream cycle will reprocess it.

        Sets consolidated_at = NULL on the episode.  When `purge_notes` is
        True (default) also hard-deletes every note whose source_episode_ids
        contains this episode id, removing it from the FTS and vector indexes
        too.  Returns the number of notes deleted.
        """
        ...

    # --- Turns -----------------------------------------------------------

    def insert_turn(self, turn: Turn) -> None: ...

    def get_turns_for_episode(self, episode_id: str) -> list[Turn]: ...

    def count_turns_since(self, ts: int) -> int: ...

    # --- Dream log -------------------------------------------------------

    def insert_dream_log(self, entry: DreamLog) -> None: ...

    def update_dream_log(self, entry: DreamLog) -> None: ...

    def list_recent_dream_logs(self, limit: int) -> list[DreamLog]: ...

    def last_dream_completed_at(self) -> int | None: ...

    # --- Cowork importer state ------------------------------------------

    def get_cowork_import_state(self, session_id: str) -> CoworkImportState | None: ...

    def upsert_cowork_import_state(self, state: CoworkImportState) -> None: ...
