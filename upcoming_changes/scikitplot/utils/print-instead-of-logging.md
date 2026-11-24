---
title: "utils: library code prints to standard output instead of logging (11 calls)"
status: open
kind: "other"
area: "scikitplot/utils"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.utils"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# utils: library code prints to standard output instead of logging (11 calls)

## Summary

11 `print()` calls inside functions of utilities write directly to standard output.

## Why it matters

A caller cannot silence, redirect or level-filter `print`; it corrupts output that is meant to be parsed (a CLI's JSON, a notebook cell's result, an MCP server's stdio transport) and bypasses `scikitplot.logging`, which every other submodule uses.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list print-in-library` (main cba9383), unacknowledged findings:
- `scikitplot/utils/_show_versions.py:425`: `print("\nSystem Information:")`
- `scikitplot/utils/_show_versions.py:427`: `print(f"{k:>25}: {v}")`
- `scikitplot/utils/_show_versions.py:429`: `print("\nPython Dependencies:")`
- `scikitplot/utils/_show_versions.py:431`: `print(f"{k:>25}: {v}")`
- `scikitplot/utils/_show_versions.py:434`: `print("\nEnvironment Variables:")`
- `scikitplot/utils/_show_versions.py:437`: `print(f"{k:>25}: {val}")`
- `scikitplot/utils/_show_versions.py:440`: `print("\nGPU Information:")`
- `scikitplot/utils/_show_versions.py:442`: `print(f"{k:>25}: {v}")`
- `scikitplot/utils/_show_versions.py:445`: `print("\nThreadpoolctl Information:")`
- `scikitplot/utils/_show_versions.py:448`: `print(f"{k:>25}: {v}")`
- `scikitplot/utils/_show_versions.py:449`: `print()`

## Root cause / current understanding

Messages written for interactive use were never moved to the package's logger.

## Expected behavior

Library functions log through `logging.getLogger(__name__)` (or return the text); only command-line entry points print, and they print to the stream the command documents.

## Affected paths and ownership

- `scikitplot/utils`

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

Required: `scikitplot.utils` / `enhancement`, one bullet, when the change lands.
