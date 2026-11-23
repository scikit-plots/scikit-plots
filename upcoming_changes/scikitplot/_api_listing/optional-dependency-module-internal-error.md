---
title: "`api` listing crashes on a module whose names need an optional extra"
status: open
kind: "bug"
area: "scikitplot/_api_listing.py"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "fix"
towncrier_fragment: ""
---

# `api` listing crashes on a module whose names need an optional extra

## Summary

`python -m scikitplot api scikitplot.experimental` exits 70 with `Internal error: ModuleNotFoundError: No module named 'pydantic'` on an install without that extra.

## Why it matters

The listing is the documented way to read the installed API; it should list every name and mark the ones that need an extra, not stop.

## Current evidence

- `sweep_runtime.py --api`: `experimental` api status 70.
- Output: `Internal error: ModuleNotFoundError: No module named 'pydantic'` plus a hint that the install may predate the module, which is wrong here.

## Root cause / current understanding

Describing a name imports it; an `ImportError` of an optional dependency is treated as an internal error.

## Expected behavior

Names whose access raises `ImportError` are listed with 'needs <extra>' (from the error or the extras map), and the command exits 0.

## Affected paths and ownership

- `scikitplot/_api_listing.py`

## Constraints and non-goals

The listing still imports only `scikitplot` modules.

## Edge cases to cover

- A lazy `__getattr__` raising `ModuleNotFoundError`
- A name whose import raises another error (a real bug: report it)

## Proposed direction

Catch `ImportError` per name, render a 'needs extra' line, and keep the exit status 0.

## Verification / acceptance criteria

- `python -m scikitplot api scikitplot.experimental` exits 0 without pydantic and marks the names

## Documentation impact

none

## Release-note promotion

Required: `unknown` / `fix`, one bullet, when the change lands.
