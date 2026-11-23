---
title: "corpus: no late-interaction (multi-vector) retrieval"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no late-interaction (multi-vector) retrieval

## Summary

Retrieval scores a query against one vector per chunk (plus BM25). Late-interaction models keep one vector per token and score with maximum similarity, which retrieves exact terms and long documents better.

## Why it matters

Single-vector retrieval misses passages whose relevant part is a few words in a long chunk; late interaction closes much of the gap to cross-encoders at index time cost.

## Current evidence

- `python -m scikitplot api scikitplot.corpus` lists no public name matching `colbert|late_interaction|multi_vector|maxsim`.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

An index type storing token vectors (compressed), MaxSim scoring, and the same `RetrievalIndex.search` contract and fusion policies; memory and disk bounded by settings.

## Affected paths and ownership

- `scikitplot/corpus/` (retrieval index)
- `scikitplot/annoy` (candidate generation)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Very long documents (token cap)
- Index size (compression, pruning)
- Mixed languages

## Proposed direction

Candidate generation from the existing index, MaxSim re-scoring of the candidates.

## Verification / acceptance criteria

- On a labelled set, recall@k is reported against the single-vector index with the existing metrics

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
