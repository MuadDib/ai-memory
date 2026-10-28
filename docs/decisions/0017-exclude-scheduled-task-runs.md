# ADR-0017: Exclude Scheduled-Task Runs from Memory

**Status:** Accepted
**Date:** 2026-10-28
**Related:** ADR-0015 (harness-injected markup), the
the 2026-10-17 and 2026-10-28 weekly health reviews (local-only, in
`docs/reviews/private/`)
**Implementation:** `harness_markup.is_scheduled_task_run`;
`cowork_importer.import_cowork_sessions` (skip) and
`dreaming._is_scheduled_task_episode` (Phase 3 guard). Tests:
`tests/test_harness_markup.py`, `tests/test_cowork_importer.py`,
`tests/test_dreaming_integration.py`.

## Context

Claude Code scheduled tasks run unattended. The session's first user turn is
the task file, wrapped in `<scheduled-task name="…" file="…">` and sometimes
preceded by a `<system-reminder>`. The replies are point-in-time results.

The weekly health audit is one of these tasks, and its own sessions were
imported and dreamed. That added 23 notes on 10-17 and 10 on 10-23 restating
the audit's instructions as facts about the user. One of them ranked #2 for
"ai-memory dream daemon". The other scheduled tasks (website watchers)
produce the same kind of transient state. Each run repeats, so the damage
grows every week.

## Decision

1. **The importer skips any session whose first user turn**, after stripping
   boilerplate, starts with `<scheduled-task` or "This is an automated run of a
   scheduled task". This is checked on every run, so a run imported before the
   rule also stops being extended. The check is anchored at the start, so a
   normal session that *discusses* a scheduled task is unaffected.
2. **Phase 3 applies the same test** to episodes already imported. It marks
   them consolidated without extraction. Turns are kept, and
   `redream --episode` undoes it.

## Consequences

- What a scheduled task is *for* is captured only from the interactive session
  where the user set it up. Results of scheduled runs (e.g. "nothing has
  changed yet") never become memory. If a task's output should be remembered, the task
  should call `memory_remember` explicitly.
- Notes already extracted from earlier runs are **not** deleted by this
  change. Cleaning them up is a separate, user-approved step.
