"""
Harness-injected markup in Claude Code / Cowork user turns.

Claude Code writes things into the *user* side of a transcript that the user
never typed: `<system-reminder>` blocks (CLAUDE.md dumps, environment info,
attribution rules — often several KB on the first turn of every session),
`<local-command-caveat>` boilerplate, slash-command wrappers, background
task notifications. Left in, they became every episode's title
("<system-reminder>") and ~7% of all user-turn text sent to the dream LLM.

Two pure functions:

- `strip_injected(text)` removes the pure-boilerplate blocks. Applied to user
  turns at import time and again at dream render time (so turns imported
  before this existed are cleaned too). See ADR-0015.
- `title_candidate(text)` picks a human-meaningful title line from a user
  turn, skipping wrappers that aren't the user's own words.

Only ever apply these to role=user text: assistant turns can legitimately
quote these tags (e.g. a conversation *about* this module).
"""
from __future__ import annotations

import re

# Blocks that are pure harness boilerplate — never the user's words, never
# worth persisting or sending to an LLM.
_BOILERPLATE_TAGS = ("system-reminder", "local-command-caveat")

# Blocks that may carry useful context (kept in stored text) but are not the
# user's own words, so they make poor titles.
_NON_TITLE_TAGS = (
    "task-notification",
    "local-command-stdout",
    "local-command-stderr",
    "pasted_content",
    "artifact-view-context",
)


def _block_re(tags: tuple[str, ...]) -> re.Pattern[str]:
    names = "|".join(re.escape(t) for t in tags)
    # Paired open/close of the same tag, attributes allowed, non-greedy body.
    return re.compile(rf"<({names})(?:\s[^>]*)?>.*?</\1>", re.DOTALL)


_BOILERPLATE_RE = _block_re(_BOILERPLATE_TAGS)
_NON_TITLE_RE = _block_re(_NON_TITLE_TAGS)
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_ANY_TAG_RE = re.compile(r"</?[A-Za-z][\w-]*(?:\s[^>]*)?>")
_COMMAND_NAME_RE = re.compile(r"<command-name>(.*?)</command-name>", re.DOTALL)
_COMMAND_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)

TITLE_MAX_CHARS = 80


def strip_injected(text: str) -> str:
    """Remove harness boilerplate blocks from a user turn's text.

    Unpaired tags (e.g. a user merely *mentioning* `<system-reminder>`) are
    left alone — only complete open/close pairs are removed.
    """
    if "<" not in text:
        return text
    cleaned = _BOILERPLATE_RE.sub("", text)
    if cleaned == text:
        return text
    return _EXCESS_BLANK_LINES_RE.sub("\n\n", cleaned).strip()


_SCHEDULED_TASK_PREFIXES = ("<scheduled-task", "This is an automated run of a scheduled task")


def is_scheduled_task_run(first_user_text: str) -> bool:
    """True when a session's FIRST user turn is a scheduled-task launch prompt.

    These runs are unattended and repeat: the prompt is the task file and the
    replies are point-in-time results, so dreaming them turned the task's
    instructions into "facts about the user" (2026-10 health reviews). Anchored
    at the start, after boilerplate, so a session that merely discusses
    scheduled tasks never matches.
    """
    return strip_injected(first_user_text).lstrip().startswith(_SCHEDULED_TASK_PREFIXES)


def title_candidate(text: str) -> str | None:
    """Return a short title from a user turn, or None if it has no own words.

    Slash-command turns render as "/name args". Notification / pasted /
    command-output blocks are skipped; any remaining tags are stripped.
    """
    text = strip_injected(text)

    name_match = _COMMAND_NAME_RE.search(text)
    if name_match:
        name = name_match.group(1).strip()
        if name and not name.startswith("/"):
            name = "/" + name
        args_match = _COMMAND_ARGS_RE.search(text)
        args = args_match.group(1).strip() if args_match else ""
        line = " ".join(f"{name} {args}".split())
        return line[:TITLE_MAX_CHARS] or None

    text = _NON_TITLE_RE.sub("", text)
    text = _ANY_TAG_RE.sub("", text)
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:TITLE_MAX_CHARS]
    return None
