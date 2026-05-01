# ADR-0008: Tag Taxonomy — Fixed Vocabulary + Entity Tags

**Status:** Accepted  
**Date:** 2026-04-27

## Context

Notes and episodes are tagged to support filtered recall and analytics. Without a fixed vocabulary, LLMs invent inconsistent tags (e.g. "aws", "Amazon Web Services", "cloud-infra") that fragment the index. Tags must be stable enough to use as filter keys in recall queries.

## Decision

Use a **fixed domain tag vocabulary** (9 tags) plus free-form entity tags extracted separately:

**Domain tags** (mutually exclusive primary, combinable secondary):

| Tag | Meaning |
|-----|---------|
| `profile` | Personal identity, preferences, biographical facts |
| `technical` | Code, architecture, tools, debugging |
| `project` | Specific named project work |
| `learning` | Concepts explained, tutorials, references |
| `problem-fix` | Diagnosed problems and their solutions |
| `decision` | Explicit choices made (what and why) |
| `preference` | Stated likes/dislikes, working style |
| `external` | Facts about third parties, services, the world |
| `meta` | Facts about the memory system itself |

**Entity tags**: Named proper nouns (people, places, tools, organisations) extracted by Phase 3 and backfilled via `ai-memory backfill-entities`. Stored in the `entities` column as a comma-separated list, indexed in FTS5 for keyword recall.

The extract prompt enforces the fixed vocabulary. The backfill command uses a separate tightened prompt that restricts entity extraction to named proper nouns/places/tools/activities — not abstract concepts.

## Consequences

- Consistent tag vocabulary enables filtered recall and statistics (`ai-memory stats`).
- New domain tags require updating the extract prompt and re-dreaming affected episodes.
- Entity extraction can over-fire (abstract concepts) if the prompt is loose — the tightened prompt is the result of fixing this in production.
- Tags are not used as the primary recall signal — RRF over BM25+vector is. Tags supplement recall with hard filters.

## Alternatives considered

- **Free-form LLM-generated tags**: Inconsistent; fragments the index. Rejected.
- **Hierarchical taxonomy** (e.g. `technical/python/async`): More expressive but harder to enforce via prompt. Deferred.
- **No tags, entities only**: Loses domain-level filtering. Rejected.
- **Ontology/controlled vocabulary system** (SKOS, etc.): Overkill for a local single-user system. Rejected.
