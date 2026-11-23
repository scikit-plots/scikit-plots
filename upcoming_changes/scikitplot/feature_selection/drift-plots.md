---
title: "feature_selection: drift is measured but not plotted"
status: open
kind: "other"
area: "scikitplot/feature_selection"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.feature_selection"
towncrier_type: "feature"
towncrier_fragment: ""
---

# feature_selection: drift is measured but not plotted

## Summary

`DriftPolicy` and `ReferenceProfile` compute population stability (PSI) against a training profile; no plot shows drift per feature or over time.

## Why it matters

Drift monitoring is read as a chart (which features moved, when); numbers alone hide the trend.

## Current evidence

- `python -m scikitplot api scikitplot.feature_selection` lists no public name matching `plot_`.
- Already public in `scikitplot.feature_selection`: `ReferenceProfile`, `DriftPolicy` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Plots: per-feature PSI bars with the warn and alert thresholds, distribution comparison for one feature, and PSI over time windows.

## Affected paths and ownership

- `scikitplot/feature_selection`
- `scikitplot/api`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Categorical features with many levels
- Empty bins
- Time windows with few samples

## Proposed direction

Draw from the existing profile objects.

## Verification / acceptance criteria

- The plots of a synthetic drift show the thresholds and the drifted feature first

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.feature_selection` / `feature`, one bullet, when the change lands.
