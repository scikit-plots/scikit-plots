---
title: "api: metric plots show no confidence bands"
status: open
kind: "other"
area: "scikitplot/api"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "feature"
towncrier_fragment: ""
---

# api: metric plots show no confidence bands

## Summary

ROC, precision-recall and calibration plots draw point estimates only. `stats.bootstrap` exists but no plot uses it.

## Why it matters

A curve without its uncertainty overstates differences, especially on small or imbalanced test sets.

## Current evidence

- `grep -rEil 'bootstrap' scikitplot/api --include=*.py` (tests and vendored code excluded) finds nothing.
- Already public in `scikitplot.stats`: `bootstrap` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

An option (`ci=0.95, n_boot=..., seed=...`) on the metric plots computing bootstrap bands with a fixed seed and drawing them, plus the interval of the summary metric in the legend.

## Affected paths and ownership

- `scikitplot/api/metrics/`
- `scikitplot/stats`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Very small test sets (wide bands)
- Class missing in a resample
- Many classes (cost)

## Proposed direction

Shared resampling helper with a seed; vectorised where possible.

## Verification / acceptance criteria

- Bands are reproducible with the same seed and cover the reference interval

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.api` / `feature`, one bullet, when the change lands.
