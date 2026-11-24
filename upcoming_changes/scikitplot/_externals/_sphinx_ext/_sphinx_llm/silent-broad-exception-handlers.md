---
title: "_sphinx_ext/_sphinx_llm: broad exception handlers that do nothing (3 places)"
status: open
kind: "reliability"
area: "scikitplot/_externals/_sphinx_ext/_sphinx_llm"
discovered_during: "Package sweep, tools/maint_tools/sweep_static.py on main cba9383 (2026-11-24)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# _sphinx_ext/_sphinx_llm: broad exception handlers that do nothing (3 places)

## Summary

In the documentation LLM extension, 3 handlers catch `Exception` (or everything) and only `pass` or `continue`, with no log line and no `# noqa` recording a decision.

## Why it matters

Every failure inside those blocks, including bugs (`TypeError`, `KeyError`) and resource errors, disappears; the caller sees a partial result and nobody can tell why.

## Current evidence

- `python -B tools/maint_tools/sweep_static.py --list silent-broad-except` (main cba9383), unacknowledged findings:
- `scikitplot/_externals/_sphinx_ext/_sphinx_llm/adapters/inventory.py:59`: `except Exception:  # ruff: ignore[blind-except, try-except-continue]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_llm/core/generator.py:348`: `except Exception:  # ruff: ignore[blind-except]`
- `scikitplot/_externals/_sphinx_ext/_sphinx_llm/core/generator.py:370`: `except Exception:  # ruff: ignore[blind-except]`

## Root cause / current understanding

Best-effort clean-up and optional steps were written with the broadest handler.

## Expected behavior

Each handler names the exceptions it expects; an unexpected one propagates or is logged at debug level with the operation and path; a deliberate best-effort catch carries `# noqa: BLE001` with the reason.

## Affected paths and ownership

- `scikitplot/_externals/_sphinx_ext/_sphinx_llm`

## Constraints and non-goals

Keep best-effort clean-up best effort (it must not mask the original error); only narrow and log.

## Edge cases to cover

- Clean-up during an exception already being raised
- Permission errors on read-only files
- Interpreter shutdown

## Proposed direction

Narrow each handler (`OSError`, `ValueError`, ...), log at debug level, and acknowledge the remaining deliberate ones with a reason.

## Verification / acceptance criteria

- `sweep_static.py --list silent-broad-except` prints nothing unacknowledged for the area

## Documentation impact

none

## Release-note promotion

Not required (no shipped behaviour changes).
