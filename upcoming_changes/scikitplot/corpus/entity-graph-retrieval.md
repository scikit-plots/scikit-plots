---
title: "corpus: no entity graph for multi-hop retrieval"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no entity graph for multi-hop retrieval

## Summary

Documents are retrieved independently; nothing links entities (people, organisations, concepts) across documents.

## Why it matters

Questions that join facts from several documents (who worked with whom, which drug was used in which trial) need a graph of entities and relations.

## Current evidence

- `python -m scikitplot api scikitplot.corpus` lists no public name matching `knowledge_graph|entity_graph|graphrag`.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Optional entity and relation extraction into a graph store (in-memory, exportable), community summaries, and a retriever that expands from query entities along edges with a hop budget.

## Affected paths and ownership

- `scikitplot/corpus/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Entity resolution across spellings
- Huge hubs (degree caps)
- Contradictory relations

## Proposed direction

Start with co-occurrence graphs from existing NER; export to standard graph formats.

## Verification / acceptance criteria

- Multi-hop questions on a synthetic set are answered with cited paths

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
