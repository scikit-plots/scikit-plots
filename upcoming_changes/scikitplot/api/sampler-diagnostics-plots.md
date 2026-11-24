---
title: "api: no diagnostics plots for MCMC samplers"
status: open
kind: "other"
area: "scikitplot/api"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "feature"
towncrier_fragment: ""
---

# api: no diagnostics plots for MCMC samplers

## Summary

Bayesian models fitted with Markov chain samplers (Hamiltonian Monte Carlo and others) are checked with trace, rank and autocorrelation plots and the R-hat and effective sample size statistics; none exist here.

## Why it matters

Convergence problems go unnoticed without these checks.

## Current evidence

- `grep -rEil 'r_hat|rhat|effective_sample_size' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Plots for chains given as arrays (chains x draws x parameters): trace, rank histograms, autocorrelation, and a summary table with split R-hat and bulk and tail effective sample size.

## Affected paths and ownership

- `scikitplot/api`
- `scikitplot/stats`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- One chain only
- Divergent transitions marked
- Thousands of parameters (selection)

## Proposed direction

Implement the statistics from their published definitions; arrays only, no sampler dependency.

## Verification / acceptance criteria

- R-hat and ESS match reference implementations on test chains

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.api` / `feature`, one bullet, when the change lands.
