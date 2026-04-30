"""
Privacy filter — strip likely secrets from text before persisting / embedding.

This is a regex-based pre-pass, not a guarantee. It catches the obvious
patterns (API keys, JWTs, AWS credentials, common bearer-token shapes).
For sensitive deployments add allow/deny lists, named-entity scrubbers,
or both.

Borrowed in spirit from agentmemory's "strip secrets before compression"
pipeline. Better to over-redact than to embed a working credential.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern is (label, regex). Order matters: more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # JWT (3 base64url segments separated by dots)
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")),
    # AWS access key id
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # AWS secret (40-char base64-ish; high false-positive risk — keep narrow)
    (
        "AWS_SECRET",
        re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])"),
    ),
    # Anthropic API key (must come before OpenAI — sk-ant- also matches sk-)
    ("ANTHROPIC_API_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    # OpenAI API key
    ("OPENAI_API_KEY", re.compile(r"\bsk-(?!ant-)[A-Za-z0-9_\-]{20,}\b")),
    # GitHub fine-grained PAT
    ("GITHUB_PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    # Generic bearer header
    ("BEARER", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.]{20,}\b")),
    # GUIDs are sometimes secrets, sometimes not — leave alone here.
]


@dataclass(frozen=True)
class RedactionResult:
    """Result of a privacy-filter pass."""

    text: str
    redactions: list[str]  # human-readable labels of what was stripped


def redact(text: str) -> RedactionResult:
    """Replace detected secrets with `[REDACTED:<LABEL>]` placeholders.

    Returns the redacted text plus a list of labels for what was found.
    Order is preserved; multiple matches of the same kind appear once
    per occurrence (so callers can tell whether anything was redacted
    and what kinds).
    """
    found: list[str] = []
    for label, pattern in _PATTERNS:

        def _replace(match: re.Match[str], _label: str = label) -> str:
            found.append(_label)
            return f"[REDACTED:{_label}]"

        text = pattern.sub(_replace, text)
    return RedactionResult(text=text, redactions=found)
