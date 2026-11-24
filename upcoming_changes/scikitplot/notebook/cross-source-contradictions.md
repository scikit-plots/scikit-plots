---
title: "Notebook: contradictions between sources are not reported"
status: open
kind: "other"
area: "scikitplot/notebook"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Notebook: contradictions between sources are not reported

## Summary

When two sources disagree on a fact, retrieval returns whichever ranks higher.

## Why it matters

Users need to know that sources disagree, which is often the most valuable finding of a review.

## Current evidence

- `grep -rEil 'contradiction_report|cross_source' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A report of claim pairs from different sources that contradict each other (entailment-based), with both citations; answers mention disagreement when it exists.

## Affected paths and ownership

- `scikitplot/notebook`
- `scikitplot/corpus` (evaluation)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Different dates or versions of the same fact
- Units and number formats
- Negation

## Proposed direction

Reuse the faithfulness evaluator in pairwise mode on retrieved claims.

## Verification / acceptance criteria

- Two synthetic sources with one conflicting number produce one reported contradiction

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
