"""
Timestamp utilities — ISO 8601 UTC strings are the system-wide format.

All stored timestamps use: 2026-04-29T18:35:23Z
Lexicographically sortable, human-readable, unambiguous, no tz confusion.
"""
from __future__ import annotations

from datetime import datetime, timezone


def now_iso() -> str:
    """Current UTC time as an ISO 8601 string: '2026-04-29T18:35:23Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def unix_to_iso(ts: int | float) -> str:
    """Convert a Unix epoch (seconds) to an ISO 8601 UTC string."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_to_dt(s: str) -> datetime:
    """Parse an ISO 8601 UTC string to a timezone-aware datetime.

    Handles both the 'Z' suffix and '+00:00' offset forms.
    """
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso_to_unix(s: str) -> int:
    """Parse an ISO 8601 UTC string to a Unix timestamp int."""
    return int(iso_to_dt(s).timestamp())
