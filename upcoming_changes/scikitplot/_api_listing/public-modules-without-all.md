---
title: "Public modules without `__all__` cannot be listed (cexperimental, cexternals, environment_variables, ml_package_versions, neighbors, version)"
status: open
kind: "docs-contract"
area: "scikitplot"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "other"
towncrier_fragment: ""
---

# Public modules without `__all__` cannot be listed (cexperimental, cexternals, environment_variables, ml_package_versions, neighbors, version)

## Summary

6 public (no leading underscore) modules define no `__all__`: `cexperimental`, `cexternals`, `environment_variables`, `ml_package_versions`, `neighbors`, `version`. `python -m scikitplot api scikitplot.<module>` exits 1 with 'declares no __all__'.

## Why it matters

The listing is how users and assistants learn the installed API (MEMORY.md); a public module it cannot list is either undocumented API or should be private.

## Current evidence

- `sweep_runtime.py --api`: `cexperimental` api status 1
- `sweep_runtime.py --api`: `cexternals` api status 1
- `sweep_runtime.py --api`: `environment_variables` api status 1
- `sweep_runtime.py --api`: `ml_package_versions` api status 1
- `sweep_runtime.py --api`: `neighbors` api status 1
- `sweep_runtime.py --api`: `version` api status 1
- `python -m scikitplot api scikitplot.neighbors` prints `Error: scikitplot.neighbors declares no __all__`.

## Root cause / current understanding

Modules added before the listing contract, or placeholders.

## Expected behavior

Each public module declares `__all__` (possibly empty with a docstring saying why), or is renamed private.

## Affected paths and ownership

- `scikitplot/cexperimental`
- `scikitplot/cexternals`
- `scikitplot/environment_variables`
- `scikitplot/ml_package_versions`
- `scikitplot/neighbors`
- `scikitplot/version`

## Constraints and non-goals

Adding names to `__all__` follows Rule 172 (API reference, usage skill, fragment).

## Edge cases to cover

- Compiled extension modules (`.pyx`) exposing names
- Modules re-exported by `scikitplot/__init__.py`

## Proposed direction

Decide per module: declare `__all__` or make private; add a test that every public module has `__all__`.

## Verification / acceptance criteria

- `sweep_runtime.py --api` reports api status 0 for every public module

## Documentation impact

API reference.

## Release-note promotion

Required: `scikitplot` / `other`, one bullet, when the change lands.
