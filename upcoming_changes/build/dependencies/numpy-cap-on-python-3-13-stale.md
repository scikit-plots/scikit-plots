---
title: "The NumPy upper bound for Python 3.13 rests on a premise that no longer holds"
status: open
kind: "packaging"
area: "build/dependencies"
discovered_during: "Real-data testing setup on CPython 3.13 (2026-11-23)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "other"
towncrier_fragment: ""
---

# The NumPy upper bound for Python 3.13 rests on a premise that no longer holds

## Summary

`pyproject.toml` (build requirements, `dependencies`, and one more list)
and `requirements/default.txt`, `requirements/build.txt` pin
`numpy <= 2.4.6; python_version == '3.13'`. The justification says NumPy
"has not published a 3.13-compatible wheel beyond 2.4.6". NumPy 2.5.0 to
2.5.5 are on PyPI with `cp313` wheels (checked 2026-11-23), so Python 3.13
users are held back while every other Python gets the latest NumPy.

## Why it matters

The dependency policy requires every upper bound to carry a valid
justification and the library to work across the declared range. A stale
cap also blocks users who need NumPy 2.5 features or fixes alongside
scikit-plots on 3.13, and makes the resolver pick older versions of other
packages that require newer NumPy.

## Current evidence

`main` at 13b457d, installed from source into a clean CPython 3.13 venv
(Linux x86-64), with the test extra; the full installed suite
(`pytest --pyargs scikitplot -n auto`, `-o addopts=""`) under each NumPy:

| NumPy | Passed | Failed | Errors | Skipped |
|---|---|---|---|---|
| 2.4.6 | 23,073 | 165 | 294 | 2,413 |
| 2.5.5 | 23,073 | 165 | 294 | 2,413 |

The failed sets are identical (no test fails on one version only); the
165 failures and 294 errors come from missing optional extras and no
network in the sandbox. The compiled extensions were built against 2.4.6
and ran unchanged with 2.5.5.

## Root cause / current understanding

Verified: the bound was set when 2.4.6 was the newest tested 3.13 wheel
(finding F-R14-02) and its own condition ("re-evaluate on each NumPy minor
release") was not acted on for the 2.5 series. Not verified here: macOS,
Windows, ARM64, free-threaded 3.13t, and building against 2.5 headers.

## Expected behavior

The 3.13 marker joins the general `numpy>=2.0.0` line, or keeps a bound
whose justification names a reproduced failure.

## Affected paths and ownership

`pyproject.toml` (three lists), `requirements/default.txt`,
`requirements/build.txt`, `docker/env_uv/py313` lock. Owner:
`maintenances/_build_utils`.

## Constraints and non-goals

Build against the oldest supported NumPy headers as now; do not raise
the lower bound. Regenerate requirements with the existing generator.

## Edge cases to cover

Free-threaded 3.13t wheels, Windows ARM64 (PR 948), the `env-xeus`
markers, Pyodide.

## Proposed direction

Remove the 3.13-only line in all lists; let CI's minimum-and-latest jobs
cover 3.13 with NumPy 2.5.x; regenerate `requirements/` and the uv lock.

## Verification / acceptance criteria

CI wheel and test jobs on 3.13 (all platforms) pass with the latest NumPy;
`generate_requirements` check is clean; the uv lock for 3.13 resolves
NumPy 2.5.x.

## Documentation impact

Install page's supported-versions table, if it lists NumPy per Python.

## Release-note promotion

Required (installed dependency range changes); section to choose at
implementation.
