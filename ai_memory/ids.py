"""ID generation for all persisted entities.

All new IDs use ULID (Universally Unique Lexicographically Sortable Identifier):
  - 26 uppercase characters
  - Lexicographically sortable by creation time (no separate sort key needed)
  - Globally unique
  - URL-safe, no dashes

The cowork importer inherits episode IDs directly from Claude Code session
filenames, which are already ULIDs.  Using ULID everywhere keeps the format
consistent across all code paths.
"""
from ulid import ULID


def new_id() -> str:
    """Return a new ULID string."""
    return str(ULID())
