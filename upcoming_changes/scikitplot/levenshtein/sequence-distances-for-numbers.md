---
title: "levenshtein: no distances for numeric sequences (dynamic time warping)"
status: open
kind: "other"
area: "scikitplot/levenshtein"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.levenshtein"
towncrier_type: "feature"
towncrier_fragment: ""
---

# levenshtein: no distances for numeric sequences (dynamic time warping)

## Summary

The distance facade compares strings. Time series and other numeric sequences need dynamic time warping and related elastic distances.

## Why it matters

Sequence similarity search, clustering and nearest-neighbour classification of time series need these distances with the same backend choice and limits.

## Current evidence

- `grep -rEil '\bdtw\b|dynamic_time_warping' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

`dtw_distance(a, b, window=...)` and a pairwise matrix function with the facade's backend selection and exact Python reference, plus lower bounds for pruning.

## Affected paths and ownership

- `scikitplot/levenshtein` (or a sibling module)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Sequences of different length
- Multivariate series
- Very long series (window required)

## Proposed direction

Pure-Python reference first, compiled backend optional, backends checked equal.

## Verification / acceptance criteria

- Results equal the reference on random series; the window bounds the cost

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.levenshtein` / `feature`, one bullet, when the change lands.
