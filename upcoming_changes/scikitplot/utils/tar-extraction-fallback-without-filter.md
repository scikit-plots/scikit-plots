---
title: "utils: tar extraction falls back to an unfiltered extractall"
status: open
kind: "security"
area: "scikitplot/utils"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.utils"
towncrier_type: "fix"
towncrier_fragment: ""
---

# utils: tar extraction falls back to an unfiltered extractall

## Summary

`scikitplot/utils/_fixes.py::tarfile_extractall` calls `extractall(path, filter="data")` and, when that raises `TypeError` (a Python without extraction filters), calls `extractall(path)` with no filter at all.

## Why it matters

Unfiltered extraction follows absolute paths, `..` and links outside the target (the class of CVE-2007-4559). The fallback is reached exactly on the older Pythons that also lack every other mitigation.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list extractall-without-filter` lists `scikitplot/utils/_fixes.py:368: tarfile.extractall(path)`.
- The comment above it says the filter prevents 'the most dangerous security issues'; the `except TypeError` branch drops it.

## Root cause / current understanding

A compatibility shim kept the old call as a fallback instead of validating members itself.

## Expected behavior

On every supported Python, extraction refuses members outside the target, absolute paths, device files and links that escape, and raises a clear error naming the member.

## Affected paths and ownership

- `scikitplot/utils/_fixes.py`
- callers of `tarfile_extractall`

## Constraints and non-goals

Use the same rules as `corpus._archive_handler` (which already validates members); no new dependency.

## Edge cases to cover

- Member `../../etc/passwd`
- Absolute member path
- Symlink to `/`
- Hard link to a file outside
- A Python patch release that has `filter=` (3.8.17+, 3.9.17+, 3.10.12+, 3.11.4+)

## Proposed direction

When `filter=` is unavailable, iterate members, resolve each target and refuse anything outside, links included; or raise and require a Python with filters.

## Verification / acceptance criteria

- A test with a hostile tar passes on the floor Python and on the newest
- No `extractall` without `filter=` remains (`sweep_static.py --fail-on extractall-without-filter`)

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.utils` / `fix`, one bullet, when the change lands.
