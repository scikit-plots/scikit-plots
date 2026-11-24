---
title: "Bare `except:` clauses in first-party code"
status: open
kind: "reliability"
area: "scikitplot"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# Bare `except:` clauses in first-party code

## Summary

Two handlers use `except:` with no exception type: they also catch `KeyboardInterrupt` and `SystemExit`.

## Why it matters

Ctrl-C during the protected call is swallowed; a requested exit does not happen.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list bare-except` (main cba9383):
- `scikitplot/_build_utils/_meson_features/module.py:109`: `except:`
- `scikitplot/_lib/decorator.py:181`: `except:`

## Root cause / current understanding

Handlers written without a type.

## Expected behavior

`except Exception:` at most, narrowed where possible.

## Affected paths and ownership

- `scikitplot/_build_utils/_meson_features/module.py`
- `scikitplot/_lib/decorator.py`

## Constraints and non-goals

Behaviour unchanged for ordinary exceptions.

## Edge cases to cover

- KeyboardInterrupt inside the protected block

## Proposed direction

Name the exceptions; enable ruff `E722` repository-wide.

## Verification / acceptance criteria

- `sweep_static.py --fail-on bare-except` exits 0

## Documentation impact

none

## Release-note promotion

Not required (no shipped behaviour changes).
