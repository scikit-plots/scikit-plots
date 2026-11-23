---
title: "api: no plots for retrieval evaluation"
status: open
kind: "other"
area: "scikitplot/api"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "feature"
towncrier_fragment: ""
---

# api: no plots for retrieval evaluation

## Summary

`scikitplot.corpus` computes `recall_at_k` and `ndcg_at_k`, but there is no plot of these metrics over k or across systems.

## Why it matters

scikit-plots is a plotting library; retrieval systems need the same quality plots classifiers get.

## Current evidence

- `python -m scikitplot api scikitplot.api` lists no public name matching `retrieval|recall_at|ndcg`.
- Already public in `scikitplot.corpus`: `ndcg_at_k`, `recall_at_k` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Plots of recall@k, precision@k, nDCG@k and MRR over k for several systems, with optional confidence bands over queries, and a per-query difference plot.

## Affected paths and ownership

- `scikitplot/api`
- `scikitplot/corpus` (metrics)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Queries with no relevant document
- Large k ranges
- Graded relevance

## Proposed direction

Use the corpus metrics; same style options as the classifier plots.

## Verification / acceptance criteria

- Plots of a synthetic evaluation match the metric values

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.api` / `feature`, one bullet, when the change lands.
