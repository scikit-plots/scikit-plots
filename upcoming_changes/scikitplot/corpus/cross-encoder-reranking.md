---
title: "corpus: the rerank stage ships no reranker (no local cross-encoder)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: the rerank stage ships no reranker (no local cross-encoder)

## Summary

The execution plan has a rerank stage (`StageKind.RERANK`, `PlannedHit.rerank_score`) that calls a user-supplied function; the only example (`_custom_hooks.py`) calls a hosted service. No local reranker ships.

## Why it matters

Reranking the top candidates is the most effective single step for answer quality in retrieval pipelines.

## Current evidence

- `grep -rEil 'class .*CrossEncoder|CrossEncoder\(' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).
- Related code exists: `scikitplot/corpus/_custom_hooks.py`, `scikitplot/corpus/_execution.py`.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

An optional reranking stage after fusion (local cross-encoder or endpoint), with a candidate count, a time budget and the original scores kept for evaluation.

## Affected paths and ownership

- `scikitplot/corpus/` (retrieval)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Model unavailable (stage skipped with a warning)
- Long passages (truncation rule)
- Ties

## Proposed direction

A component for the existing stage; evaluation through the existing metrics.

## Verification / acceptance criteria

- nDCG@k with and without the stage is reported on a labelled set

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
