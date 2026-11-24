---
title: "api: list default arguments shared between calls"
status: open
kind: "bug"
area: "scikitplot/api"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "fix"
towncrier_fragment: ""
---

# api: list default arguments shared between calls

## Summary

`plot_pca_2d_projection(..., dimensions=[0, 1])` uses list literals as defaults. Python creates a default once, at definition time, and every call that omits the argument shares it.

## Why it matters

If the function or a caller mutates the list, the next call silently starts from the changed value: a classic source of results that depend on call order.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list mutable-default` (main cba9383):
- `scikitplot/api/decomposition/_projection.py:48`: `def plot_pca_2d_projection(`

## Root cause / current understanding

Defaults written as list literals.

## Expected behavior

Defaults are `None` (or immutable tuples) and the function builds a new list per call; the docstring states the effective default.

## Affected paths and ownership

- `scikitplot/api`

## Constraints and non-goals

Same effective defaults and results for every existing call.

## Edge cases to cover

- Two calls in a row, the first passing a value that is mutated internally
- A caller passing the same list object twice

## Proposed direction

Use `None` with an explicit fallback or a tuple; enable ruff `B006` for the module.

## Verification / acceptance criteria

- `sweep_static.py --list mutable-default` prints nothing for the area
- A test calls the function twice and checks the second call is unaffected

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.api` / `fix`, one bullet, when the change lands.
