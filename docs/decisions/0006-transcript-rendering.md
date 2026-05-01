# ADR-0006: Transcript Rendering Format

**Status:** Accepted  
**Date:** 2026-04-27

## Context

The dream cycle's LLM phases (Summary, Extract) receive conversation turns rendered as a single string. The rendering format affects how well the LLM understands speaker identity, turn boundaries, and turn ordering. The format must be stable — changing it silently regresses dream quality.

## Decision

Render transcripts using a structured XML-like envelope:

```xml
<conversation_transcript>
<turn n=1 role=user>Turn text here</turn>
<turn n=2 role=assistant>Turn text here</turn>
...
</conversation_transcript>
```

- `n` is the 1-based turn index within the episode.
- `role` is `user` or `assistant`.
- Turn text is included verbatim (after privacy filtering).
- The outer `<conversation_transcript>` tag scopes the content for the LLM and prevents leakage of the transcript format into extracted facts.

This format is tested as a stable contract. The extract and summary prompts reference `<turn>` tags by role and index when citing evidence.

## Consequences

- The LLM can reliably distinguish speakers and reference specific turns when extracting facts.
- The format is visually scannable for human debugging.
- Changing attribute names (`n`, `role`), the tag names, or the wrapper tag name will regress the extract and summary prompts — treat as a stable contract (see CLAUDE.md).
- Privacy filtering must run before rendering, not after, to prevent raw credentials from reaching the LLM (see open follow-up: pre-render redaction).

## Alternatives considered

- **Plain text with speaker prefixes** (`User: ...` / `Assistant: ...`): Simpler but harder to parse reliably; no turn indexing; no unambiguous boundary. Rejected after quality testing showed worse extraction recall.
- **JSON array of turn objects**: Machine-readable but token-inefficient for long transcripts; the LLM doesn't need JSON structure here. Rejected.
- **Markdown with `###` headings per turn**: Verbose; headings bleed into extracted facts. Rejected.
- **No rendering structure (raw concatenation)**: LLM confuses speaker attribution on multi-turn exchanges. Rejected.
