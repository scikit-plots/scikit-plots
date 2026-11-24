---
title: "decile: library code prints to standard output instead of logging (16 calls)"
status: open
kind: "other"
area: "scikitplot/decile"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.decile"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# decile: library code prints to standard output instead of logging (16 calls)

## Summary

16 `print()` calls inside functions of the decile and model-plot utilities write directly to standard output.

## Why it matters

A caller cannot silence, redirect or level-filter `print`; it corrupts output that is meant to be parsed (a CLI's JSON, a notebook cell's result, an MCP server's stdio transport) and bypasses `scikitplot.logging`, which every other submodule uses.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list print-in-library` (main cba9383), unacknowledged findings:
- `scikitplot/decile/kds/_kds.py:92`: `print(`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:656`: `print(`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:669`: `print("The label with smallest class is %s" % select_targetclass[0])`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:677`: `print(`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:681`: `print("compare models")`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:691`: `print(f"The label with smallest class is {select_targetclass}")`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:700`: `print("compare datasets")`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:710`: `print(f"The label with smallest class is {select_targetclass}")`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:719`: `print("compare target classes")`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:1087`: `print(text[:-1])`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:1446`: `print(text[:-1])`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:1792`: `print(text[:-1])`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:2156`: `print(text[:-1])`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:2895`: `print(text[:-1])`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:3268`: `print(text[:-1])`
- `scikitplot/decile/modelplotpy/_modelplotpy.py:3641`: `print(text[:-1])`

## Root cause / current understanding

Messages written for interactive use were never moved to the package's logger.

## Expected behavior

Library functions log through `logging.getLogger(__name__)` (or return the text); only command-line entry points print, and they print to the stream the command documents.

## Affected paths and ownership

- `scikitplot/decile`

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

Required: `scikitplot.decile` / `enhancement`, one bullet, when the change lands.
