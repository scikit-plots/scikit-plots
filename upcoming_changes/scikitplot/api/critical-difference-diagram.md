---
title: "api: no critical-difference diagram for comparing many models over many datasets"
status: open
kind: "other"
area: "scikitplot/api"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "feature"
towncrier_fragment: ""
---

# api: no critical-difference diagram for comparing many models over many datasets

## Summary

Comparing several models across datasets is usually shown with average ranks and a critical-difference diagram (Friedman test with a post-hoc test); there is no such plot.

## Why it matters

Benchmark tables of many models are hard to read and invite unsupported claims.

## Current evidence

- `grep -rEil 'nemenyi|critical_difference|friedman' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

`plot_critical_difference(scores, ...)` computing ranks, the Friedman test, a post-hoc test (Nemenyi or a Holm-corrected Wilcoxon) and drawing the diagram with cliques of non-significant differences.

## Affected paths and ownership

- `scikitplot/api`
- `scikitplot/stats`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Ties in ranks
- Two models only (test not applicable)
- Missing results for a dataset

## Proposed direction

Statistics in `stats`, drawing in `api`.

## Verification / acceptance criteria

- The diagram for a published example matches its ranks and cliques

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.api` / `feature`, one bullet, when the change lands.
