---
title: "annoy: importing the vector index loads pandas, SciPy and scikit-learn"
status: open
kind: "other"
area: "scikitplot/annoy"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.annoy"
towncrier_type: "efficiency"
towncrier_fragment: ""
---

# annoy: importing the vector index loads pandas, SciPy and scikit-learn

## Summary

`import scikitplot.annoy` loads pandas, SciPy and scikit-learn (0.607 s here; 2.3 s cold), although an approximate-nearest-neighbour index needs NumPy at most.

## Why it matters

Every corpus or MCP process that uses the index pays this start-up time and memory; a lightweight partial distribution of the index cannot be lighter than its import.

## Current evidence

- `sweep_runtime.py --module annoy`: heavy = ['pandas', 'scipy', 'sklearn'].
- `python -X importtime -c 'import scikitplot.annoy'` shows the scikit-learn check-build modules under the index's import chain.

## Root cause / current understanding

The index imports shared utilities (`scikitplot.utils`, `_lib`) whose import chain reaches scikit-learn and pandas.

## Expected behavior

`import scikitplot.annoy` loads only NumPy and the compiled extension; features that need pandas import it when used.

## Affected paths and ownership

- `scikitplot/annoy/`
- `scikitplot/utils/`
- `scikitplot/_lib/`

## Constraints and non-goals

Public API unchanged.

## Edge cases to cover

- Pickling an index (may need helpers)
- DataFrame input to an index method (lazy import there)

## Proposed direction

Find the import chain with `-X importtime`, move the heavy imports inside the functions that need them, and add a test asserting `sys.modules` after import.

## Verification / acceptance criteria

- `sweep_runtime.py --module annoy` reports heavy = []

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.annoy` / `efficiency`, one bullet, when the change lands.
