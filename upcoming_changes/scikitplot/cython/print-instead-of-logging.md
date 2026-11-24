---
title: "cython: library code prints to standard output instead of logging (96 calls)"
status: open
kind: "other"
area: "scikitplot/cython"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.cython"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# cython: library code prints to standard output instead of logging (96 calls)

## Summary

96 `print()` calls inside functions of the runtime Cython builder write directly to standard output.

## Why it matters

A caller cannot silence, redirect or level-filter `print`; it corrupts output that is meant to be parsed (a CLI's JSON, a notebook cell's result, an MCP server's stdio transport) and bypasses `scikitplot.logging`, which every other submodule uses.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list print-in-library` (main cba9383), unacknowledged findings:
- `scikitplot/cython/_templates/probe/compile_probe.py:29`: `print("compile entrypoints:", list(api))`
- `scikitplot/cython/_templates/probe/compile_probe.py:32`: `print("no compile_and_load entrypoint found")`
- `scikitplot/cython/_templates/probe/compile_probe.py:45`: `print(f"add(2,3) = {val}")`
- `scikitplot/cython/_templates/probe/compile_probe.py:46`: `print(f"first used_cache={used1}  second used_cache={used2}")`
- `scikitplot/cython/_templates/probe/compile_probe.py:48`: `print("VERDICT:", "OK" if ok else "CHECK")`
- `scikitplot/cython/_templates/probe/probe_abi001_runtime.py:45`: `print(f"contract declared (unload=False): {'OK' if declared else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_abi001_runtime.py:57`: `print(f"free-threaded rejected + opt-in: {'OK' if ft else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_abi001_runtime.py:69`: `print(f"subinterpreter rejected + opt-in: {'OK' if si else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_abi001_runtime.py:72`: `print("VERDICT:", "OK" if ok else "CHECK")`
- `scikitplot/cython/_templates/probe/probe_api002_stability.py:35`: `print(f"total coverage: {'OK' if cover else 'FAIL'} ({len(missing)} untiered)")`
- `scikitplot/cython/_templates/probe/probe_api002_stability.py:47`: `print(f"tiers partition surface: {'OK' if partition else 'FAIL'} "`
- `scikitplot/cython/_templates/probe/probe_api002_stability.py:56`: `print(f"accessors behave: {'OK' if acc else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_api002_stability.py:59`: `print("VERDICT:", "OK" if ok else "CHECK")`
- `scikitplot/cython/_templates/probe/probe_batch001_partial.py:52`: `print(f"fail-fast partial + resume token: {'OK' if ff else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_batch001_partial.py:57`: `print(f"collect records failures: {'OK' if col else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_batch001_partial.py:63`: `print(f"resume via only: {'OK' if res else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_batch001_partial.py:66`: `print("VERDICT:", "OK" if ok else "CHECK")`
- `scikitplot/cython/_templates/probe/probe_cache001_concurrency.py:55`: `print("  ", r)`
- `scikitplot/cython/_templates/probe/probe_cache001_concurrency.py:59`: `print("VERDICT: a worker failed")`
- `scikitplot/cython/_templates/probe/probe_cache001_concurrency.py:66`: `print(f"keys={keys} values={vals} used_cache={used}")`
- `scikitplot/cython/_templates/probe/probe_cache001_concurrency.py:67`: `print(f"published entries={len(entries)} staging_leftovers={len(staging_leftovers)}")`
- `scikitplot/cython/_templates/probe/probe_cache001_concurrency.py:76`: `print("VERDICT:", "OK" if ok else "CHECK")`
- `scikitplot/cython/_templates/probe/probe_cache002_containment.py:46`: `print(f"absolute artifact refused: {'OK' if r1 is None else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_cache002_containment.py:51`: `print(f"traversal artifact refused: {'OK' if r2 is None else 'FAIL'}")`
- `scikitplot/cython/_templates/probe/probe_cache002_containment.py:58`: `print(f"contained artifact selected: {'OK' if r3 == art else 'FAIL'}")`
- ... and 71 more (`sweep_static.py --list print-in-library`).

## Root cause / current understanding

Messages written for interactive use were never moved to the package's logger.

## Expected behavior

Library functions log through `logging.getLogger(__name__)` (or return the text); only command-line entry points print, and they print to the stream the command documents.

## Affected paths and ownership

- `scikitplot/cython`

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

Required: `scikitplot.cython` / `enhancement`, one bullet, when the change lands.
