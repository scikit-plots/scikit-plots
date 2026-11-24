---
title: "mcp: names in `__all__` raise a bare ModuleNotFoundError without the extra"
status: open
kind: "other"
area: "scikitplot/mcp"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.mcp"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# mcp: names in `__all__` raise a bare ModuleNotFoundError without the extra

## Summary

Without the MCP extra, `scikitplot.mcp.create_server`, `SearchService`, `CitationOutput` and `SearchDocsOutput` raise `ModuleNotFoundError: No module named 'pydantic'`; the message does not name the extra to install.

## Why it matters

The corpus readers say which package or extra to install; the MCP module does not, so users search for a missing package by its import name.

## Current evidence

- `sweep_runtime.py`: `mcp` lazy_failures = ['CitationOutput: ModuleNotFoundError', 'SearchDocsOutput: ModuleNotFoundError', 'SearchService: ModuleNotFoundError', 'create_server: ModuleNotFoundError']

## Root cause / current understanding

The lazy `__getattr__` lets the dependency's own error through.

## Expected behavior

`ImportError` naming `pip install "scikit-plots[mcp]"`, chained from the original error.

## Affected paths and ownership

- `scikitplot/mcp/__init__.py`

## Constraints and non-goals

Still an `ImportError` subclass (callers may catch it).

## Edge cases to cover

- Partially installed extra (one dependency missing)

## Proposed direction

Wrap the lazy import with a message naming the extra, as `scikitplot.corpus` does.

## Verification / acceptance criteria

- The message names the extra; a test without pydantic checks it

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.mcp` / `enhancement`, one bullet, when the change lands.
