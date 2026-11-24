---
title: "_lib: library code prints to standard output instead of logging (28 calls)"
status: open
kind: "other"
area: "scikitplot/_lib"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot._lib"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# _lib: library code prints to standard output instead of logging (28 calls)

## Summary

28 `print()` calls inside functions of GPU availability checks and helpers write directly to standard output.

## Why it matters

A caller cannot silence, redirect or level-filter `print`; it corrupts output that is meant to be parsed (a CLI's JSON, a notebook cell's result, an MCP server's stdio transport) and bypasses `scikitplot.logging`, which every other submodule uses.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list print-in-library` (main cba9383), unacknowledged findings:
- `scikitplot/_lib/_array_api.py:998`: `print("Checking CuPy availability...")`
- `scikitplot/_lib/_array_api.py:1004`: `print(f"GPUs Available (CuPy): {gpu_count}")`
- `scikitplot/_lib/_array_api.py:1006`: `print("No GPUs available for CuPy. Required GPU.")`
- `scikitplot/_lib/_array_api.py:1012`: `print("Matrix multiplication result (CuPy):", c)`
- `scikitplot/_lib/_array_api.py:1014`: `print(f"CuPy check failed: {e!s}")`
- `scikitplot/_lib/_array_api.py:1020`: `print("Checking TensorFlow availability...")`
- `scikitplot/_lib/_array_api.py:1028`: `print(f"GPUs Available (TensorFlow): {len(gpus)}")`
- `scikitplot/_lib/_array_api.py:1030`: `print("No GPUs available for TensorFlow. Using CPU.")`
- `scikitplot/_lib/_array_api.py:1033`: `print(f"CPUs Available (TensorFlow): {len(cpus)}")`
- `scikitplot/_lib/_array_api.py:1039`: `print("Matrix multiplication result (TensorFlow):", c.numpy())`
- `scikitplot/_lib/_array_api.py:1041`: `print(f"TensorFlow check failed: {e!s}")`
- `scikitplot/_lib/_array_api.py:1047`: `print("Checking JAX availability...")`
- `scikitplot/_lib/_array_api.py:1058`: `print(`
- `scikitplot/_lib/_array_api.py:1062`: `print("No GPUs available for JAX. Using CPU.")`
- `scikitplot/_lib/_array_api.py:1065`: `print(f"CPUs Available (JAX): {len(cpus)}")`
- `scikitplot/_lib/_array_api.py:1071`: `print("Matrix multiplication result (JAX):", c)`
- `scikitplot/_lib/_array_api.py:1073`: `print(f"JAX check failed: {e!s}")`
- `scikitplot/_lib/_array_api.py:1079`: `print("Checking PyTorch availability...")`
- `scikitplot/_lib/_array_api.py:1087`: `print(f"GPUs Available (PyTorch): {gpu_count}")`
- `scikitplot/_lib/_array_api.py:1090`: `print("No GPUs available for PyTorch. Using CPU.")`
- `scikitplot/_lib/_array_api.py:1093`: `print(f"CPUs Available (PyTorch): {cpu_count}")`
- `scikitplot/_lib/_array_api.py:1099`: `print("Matrix multiplication result (PyTorch):", c)`
- `scikitplot/_lib/_array_api.py:1101`: `print(f"PyTorch check failed: {e!s}")`
- `scikitplot/_lib/_array_api.py:1153`: `print(f"\n--- {name} Check ---")`
- `scikitplot/_lib/_array_api.py:1158`: `print(f"{name} check failed with error: {err}")`
- ... and 3 more (`sweep_static.py --list print-in-library`).

## Root cause / current understanding

Messages written for interactive use were never moved to the package's logger.

## Expected behavior

Library functions log through `logging.getLogger(__name__)` (or return the text); only command-line entry points print, and they print to the stream the command documents.

## Affected paths and ownership

- `scikitplot/_lib`

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

Required: `scikitplot._lib` / `enhancement`, one bullet, when the change lands.
