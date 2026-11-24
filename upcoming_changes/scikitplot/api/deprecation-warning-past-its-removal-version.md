---
title: "api: a deprecation warning promises removal in 0.4.0; the package is at 0.5.dev0"
status: open
kind: "other"
area: "scikitplot/api"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "api"
towncrier_fragment: ""
---

# api: a deprecation warning promises removal in 0.4.0; the package is at 0.5.dev0

## Summary

Importing `scikitplot.api` or `scikitplot.decile` emits `DeprecationWarning: This module was deprecated in version 0.3.0 ... This module will be removed in 0.4.0`, from `scikitplot/api/plotters.py:55`.

## Why it matters

The promise was not kept and the warning fires for users who never touch the deprecated module, because the import chain reaches it; warnings that never resolve teach users to ignore warnings.

## Current evidence

- `sweep_runtime.py`: `api` warnings = ['DeprecationWarning: This module was deprecated in version 0.3.0 and its functions are spread throughout different modules. Please check the documentation and update your function calls as soon as possible. This module will be removed in 0.4.0']
- `decile` warnings = ['DeprecationWarning: This module was deprecated in version 0.3.0 and its functions are spread throughout different modules. Please check the documentation and update your function calls as soon as possible. This module will be removed in 0.4.0']

## Root cause / current understanding

The removal planned for 0.4.0 did not happen; the module is imported eagerly.

## Expected behavior

The deprecated module is removed (no backward compatibility, campaign rule), or at least imported only when used.

## Affected paths and ownership

- `scikitplot/api/plotters.py`
- `scikitplot/api/__init__.py`
- `scikitplot/decile/`

## Constraints and non-goals

Removal needs its fragment and API reference update (Rule 172).

## Edge cases to cover

- Code importing `scikitplot.api.plotters` directly
- Pickled objects referring to the old path

## Proposed direction

Remove `plotters.py` and its re-exports, or move the warning to attribute access.

## Verification / acceptance criteria

- `sweep_runtime.py` reports no warnings for `api` and `decile`

## Documentation impact

Release notes (removal).

## Release-note promotion

Required: `scikitplot.api` / `api`, one bullet, when the change lands.
