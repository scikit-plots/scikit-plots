---
title: "`api` listing prints typing reprs for modules with evaluated annotations (stats)"
status: open
kind: "docs-contract"
area: "scikitplot/_api_listing.py"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# `api` listing prints typing reprs for modules with evaluated annotations (stats)

## Summary

The listing of `scikitplot.stats` shows signatures such as `ndarray[tuple[typing.Any, ...], dtype[_ScalarT]]` and `collections.abc.Callable`. `_api_listing.py` states that texts are never built from a format CPython or `typing` owns, because generated skills are compared byte for byte (Rule 168).

## Why it matters

These reprs change between Python and NumPy versions, so the usage skill of a module with evaluated annotations changes without any code change, and CI comparisons fail spuriously.

## Current evidence

- `python -m scikitplot api scikitplot.stats` (main cba9383) prints `binom_conf_interval(k: int | ndarray[tuple[typing.Any, ...], dtype[_ScalarT]], ...` and `kuiper(data: ArrayLike, cdf: collections.abc.Callable = ...)`.
- The stats modules do not use `from __future__ import annotations`, so annotations are objects, not source strings.

## Root cause / current understanding

String annotations are printed as written; evaluated ones fall through to their repr.

## Expected behavior

Annotations render from source text (`inspect.get_annotations(..., eval_str=False)` or the function's source) or the module uses postponed evaluation, so the text is the same on every Python.

## Affected paths and ownership

- `scikitplot/_api_listing.py`
- `scikitplot/stats/`

## Constraints and non-goals

No change to the rendering of string annotations.

## Edge cases to cover

- Annotations from vendored code
- `numpy.typing` aliases

## Proposed direction

Add `from __future__ import annotations` to the stats modules, and make the listing prefer source text for evaluated annotations.

## Verification / acceptance criteria

- The stats listing contains no `typing.` or `collections.abc.` text
- The usage skill generated on two Pythons is identical

## Documentation impact

none

## Release-note promotion

Not required (no shipped behaviour changes).
