# ADR-0015: Strip Harness-Injected Markup from User Turns

**Status:** Accepted
**Date:** 2026-10-10
**Related:** ADR-0001 (four-tier model — raw files as source of truth),
ADR-0006 (transcript rendering), ADR-0007 (chunked extraction)
**Implementation:** `ai_memory/harness_markup.py` (`strip_injected`,
`title_candidate`); used by `cowork_importer._coerce_turn` / `_derive_title` and
`dreaming._render_transcript`. Tests: `tests/test_harness_markup.py`,
`tests/test_cowork_importer.py`.

## Context

Claude Code (CLI and the desktop app's Code tab) writes text into the **user**
side of a session transcript that the user never typed:

- `<system-reminder>…</system-reminder>` — CLAUDE.md contents, environment
  info, attribution rules, tool notices. Several KB on the first turn of every
  desktop session, and attached as a lone text block next to `tool_result`
  blocks mid-session.
- `<local-command-caveat>…</local-command-caveat>` — fixed boilerplate.
- Slash-command wrappers (`<command-name>`, `<command-args>`, …),
  `<task-notification>`, `<local-command-stdout>` — harness-generated but
  carrying real signal (what the user invoked, what a background task found).

A survey of the live `~/.claude/projects` tree (1,005 user turns) found 7.4% of
all user-turn characters inside `<system-reminder>` blocks. Effects:

1. Every desktop-session episode got the import-time title `<system-reminder>`
   (first line of the first user turn).
2. The same boilerplate was rendered into every dream transcript — wasted
   tokens, and a source of junk facts (e.g. "Igor is using a scratch workspace
   for this session", extracted from an environment reminder).
3. Reminder-only "user turns" were persisted as if the user had spoken.

## Decision

1. **Strip pure boilerplate at import.** For `role=user` turns only, remove
   complete `<system-reminder>` and `<local-command-caveat>` open/close pairs
   before persistence. A turn that becomes empty is not imported. Assistant
   turns are never touched (they can legitimately quote these tags).
2. **Strip again at dream render time** (`_render_transcript`, user turns only),
   so turns imported before this ADR are cleaned for dream / redream without
   rewriting raw files. The rendering *shape* from ADR-0006 is unchanged — only
   turn content shrinks.
3. **Keep, but don't title from, signal-bearing wrappers.** Slash commands,
   task notifications, command stdout and `<pasted_content>` stay in stored
   text. Titles come from `title_candidate`: slash commands render as
   `/name args`; notification / stdout / pasted blocks are skipped in favour of
   the user's own words; remaining tags are dropped.
4. **Resume safety.** In the importer's linear-scan fallback, the watermark is
   matched on the transcript's own turn id as well as the coerced turn, so a
   watermark imported pre-ADR that now strips to nothing is still found.

## Consequences

- The raw tier no longer holds verbatim harness boilerplate for new imports.
  This is a deliberate exception to "raw files are verbatim": the stripped
  content is machine-generated, duplicated per session, and recoverable from
  its own source (CLAUDE.md, the app). The original Claude Code JSONL is
  untouched on disk (subject to Claude Code's own retention).
- Existing raw files and turn rows are **not** rewritten. Old turns stay
  verbatim on disk and in `depth=verbatim` recall; only dream input is
  cleaned.
- Import-time titles are placeholders anyway — the dream pass replaces them with
  an LLM title — so existing bad titles need no backfill once dreamed.
- Matching is pair-based and non-greedy. A user merely *mentioning*
  `<system-reminder>` is untouched; user-pasted text containing a complete pair
  would lose that span (accepted).
- New harness tags need adding to the lists in `harness_markup.py`.
