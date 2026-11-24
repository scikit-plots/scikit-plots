---
title: "_sphinx_ai_assistant: tests shipped in the package fail collection outside a checkout"
status: open
kind: "packaging"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant"
discovered_during: "Real-life probe on main cba9383 (2026-11-24)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# _sphinx_ai_assistant: tests shipped in the package fail collection outside a checkout

## Summary

`pytest --pyargs scikitplot` on an installed package reports 7 collection errors: `RuntimeError: AI-assistant tests require a repository/workspace root containing both 'scikitplot/' and 'maintenances/'` (raised by `tests/_paths.py`).

## Why it matters

The wheel ships tests that cannot run from it; downstream packagers and users running the installed test suite get errors instead of skips.

## Current evidence

- Installed main (13b457d), `pytest --pyargs scikitplot -o addopts=''`: 7 collection errors with the message above.

## Root cause / current understanding

Tests read repository files (maintenance records) and fail hard when they are absent.

## Expected behavior

Such tests skip with a reason when the checkout is absent, or are excluded from the wheel.

## Affected paths and ownership

- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/tests/_paths.py`
- wheel contents (`meson.build` install rules)

## Constraints and non-goals

In a checkout the tests still run and fail loudly when files are missing.

## Edge cases to cover

- Editable install inside a checkout
- sdist build

## Proposed direction

Turn the `RuntimeError` into `pytest.skip(..., allow_module_level=True)` when not in a checkout.

## Verification / acceptance criteria

- `pytest --pyargs scikitplot` on an installed wheel has no collection errors from this package

## Documentation impact

none

## Release-note promotion

Not required (no shipped behaviour changes).
