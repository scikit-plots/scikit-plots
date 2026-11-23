---
title: "stats: no paired tests for comparing classifiers (McNemar, Cochran's Q, DeLong)"
status: open
kind: "other"
area: "scikitplot/stats"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.stats"
towncrier_type: "feature"
towncrier_fragment: ""
---

# stats: no paired tests for comparing classifiers (McNemar, Cochran's Q, DeLong)

## Summary

`scikitplot.stats` has bootstrap, Kuiper and robust statistics, but no test for whether two classifiers differ on the same samples: McNemar (two models), Cochran's Q (several), DeLong (two ROC AUCs).

## Why it matters

Model comparison plots without a significance test invite over-reading small differences; these are the standard tests for paired predictions.

## Current evidence

- `python -m scikitplot api scikitplot.stats` lists no public name matching `mcnemar|cochran|delong`.
- Already public in `scikitplot.stats`: `bootstrap` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Functions returning statistic, p-value and the contingency or covariance used, exact and asymptotic variants where they exist, and a plot that annotates compared curves with the result.

## Affected paths and ownership

- `scikitplot/stats`
- `scikitplot/api` (annotations on ROC and calibration plots)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Zero discordant pairs
- Small samples (exact test)
- Ties in scores (DeLong)

## Proposed direction

Implement with SciPy distributions; validate against published examples.

## Verification / acceptance criteria

- Results match reference values from the literature to 1e-10

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.stats` / `feature`, one bullet, when the change lands.
