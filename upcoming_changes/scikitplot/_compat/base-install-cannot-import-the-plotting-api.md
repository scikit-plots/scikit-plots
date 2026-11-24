---
title: "A base install cannot import the plotting API: the vendored seaborn copy imports seaborn"
status: open
kind: "packaging"
area: "scikitplot/_compat"
discovered_during: "Real-life probe on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot._compat"
towncrier_type: "fix"
towncrier_fragment: ""
---

# A base install cannot import the plotting API: the vendored seaborn copy imports seaborn

## Summary

On `pip install scikit-plots` (no extras), the README quick start `sp.metrics.plot_precision_recall(...)` fails with `AttributeError: Module 'scikitplot' has no attribute 'metrics'. No module named 'seaborn'`. `scikitplot.api`, `decile`, `seaborn` and `visualkeras` all fail to import.

## Why it matters

The package's main feature does not work after the install the README tells users to run; the error does not say which extra to install.

## Current evidence

- Clean virtual environment, `pip install` of main (13b457d, CPython 3.13), no extras: the quick start raises the error above.
- Traceback: `scikitplot/_compat/__init__.py` line 18 imports `scikitplot/externals/_seaborn`, whose `utils.py` line 17 runs `from seaborn._core.typing import deprecated` (the real seaborn).
- `pyproject.toml` lists `seaborn` only in the `core` and `usage-skills` extras.

## Root cause / current understanding

The vendored `_seaborn` package is not self-contained: it imports seaborn itself, and `_compat` imports it eagerly.

## Expected behavior

A base install imports `scikitplot.api` and runs the README quick start; or the README installs the extra and the error names it (`pip install "scikit-plots[core]"`).

## Affected paths and ownership

- `scikitplot/_compat/__init__.py`
- `scikitplot/externals/_seaborn/`
- `README.md`
- `pyproject.toml` (dependencies, extras)

## Constraints and non-goals

Choose one: make the vendored copy self-contained, declare seaborn as a base dependency, or require the extra explicitly; the README must match.

## Edge cases to cover

- Base install on 3.10 and 3.15
- Partial distributions under `libs/`
- Pyodide (no seaborn wheel)

## Proposed direction

Recommended: remove the real-seaborn imports from the vendored copy (or import it lazily where only seaborn functions need it); add a CI job that runs the README quick start on a base install.

## Verification / acceptance criteria

- A base install runs the README quick start
- `sweep_runtime.py` reports no import error for `api`, `decile`, `seaborn`, `visualkeras` on a base install

## Documentation impact

README install section.

## Release-note promotion

Required: `scikitplot._compat` / `fix`, one bullet, when the change lands.
