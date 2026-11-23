---
title: "mcp: public names without a docstring (main)"
status: open
kind: "docs-contract"
area: "scikitplot/mcp"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# mcp: public names without a docstring (main)

## Summary

`scikitplot.mcp.__all__` exports `main` with no docstring; the API listing and the API reference show them with no description.

## Why it matters

The engineering rules require NumPyDoc docstrings for every public name; users and assistants read the listing to choose a class.

## Current evidence

- `sweep_runtime.py` (main cba9383): `mcp` undocumented = ['main'].

## Root cause / current understanding

Classes added or aliased without a docstring.

## Expected behavior

Each name has a NumPyDoc docstring (summary, parameters, examples where useful).

## Affected paths and ownership

- `scikitplot/mcp`

## Constraints and non-goals

Docstrings only.

## Edge cases to cover

- Aliases that should point to the documented class instead

## Proposed direction

Write the docstrings; add a test that every callable in `__all__` has one.

## Verification / acceptance criteria

- `sweep_runtime.py --module mcp` reports undocumented = []

## Documentation impact

API reference.

## Release-note promotion

Not required (no shipped behaviour changes).
