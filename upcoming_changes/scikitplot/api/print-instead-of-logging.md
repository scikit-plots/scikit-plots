---
title: "api: library code prints to standard output instead of logging (13 calls)"
status: open
kind: "other"
area: "scikitplot/api"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# api: library code prints to standard output instead of logging (13 calls)

## Summary

13 `print()` calls inside functions of the plotting API write directly to standard output.

## Why it matters

A caller cannot silence, redirect or level-filter `print`; it corrupts output that is meant to be parsed (a CLI's JSON, a notebook cell's result, an MCP server's stdio transport) and bypasses `scikitplot.logging`, which every other submodule uses.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list print-in-library` (main cba9383), unacknowledged findings:
- `scikitplot/api/metrics/_regression/_residuals.py:216`: `print(f"Fitted mean-mu (μ): {mean:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:217`: `print(f"Fitted std (σ)    : {std:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:239`: `print(f"Fitted lambda (λ): {lambda_fitted:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:269`: `print(f"Fitted shape shape-alpha (k or α): {shape:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:270`: `print(f"Fitted scale scale-beta  (θ or β): {scale:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:299`: `print(f"Fitted shape (λ): {shape:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:300`: `print(f"Fitted location (μ): {loc:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:301`: `print(f"Fitted scale (λ): {scale:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:332`: `print(f"Fitted mean-mu (μ): {mean:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:333`: `print(f"Tweedie var_power :{var_power:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:363`: `print(f"Fitted scale (λ): {scale:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:388`: `print(f"Fitted shape sigma (σ)       : {shape:.4f}")`
- `scikitplot/api/metrics/_regression/_residuals.py:389`: `print(f"Fitted scale exp(mu) (exp(μ)): {scale:.4f}")`

## Root cause / current understanding

Messages written for interactive use were never moved to the package's logger.

## Expected behavior

Library functions log through `logging.getLogger(__name__)` (or return the text); only command-line entry points print, and they print to the stream the command documents.

## Affected paths and ownership

- `scikitplot/api`

## Constraints and non-goals

Output a user relies on (a returned report) stays available, as a return value or an explicit `verbose`/`file=` argument.

## Edge cases to cover

- An MCP stdio server importing the area (stdout must carry only protocol messages)
- Notebook use with logging unconfigured (the message must still be reachable)

## Proposed direction

Replace `print` with the module logger at the right level, or return the information; add ruff `T201` for the area's library modules.

## Verification / acceptance criteria

- `sweep_static.py --list print-in-library` prints nothing unacknowledged for the area
- A test captures stdout of the public functions and finds it empty

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.api` / `enhancement`, one bullet, when the change lands.
