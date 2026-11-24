---
title: "Public API spells 'verbosity' 3 ways (log_level, quiet, verbose)"
status: open
kind: "api"
area: "scikitplot"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "api"
towncrier_fragment: ""
---

# Public API spells 'verbosity' 3 ways (log_level, quiet, verbose)

## Summary

Across public callables, the idea 'verbosity' is named `log_level`, `quiet`, `verbose`. `verbose` is common; `log_level` and `quiet` appear for the same control.

## Why it matters

A user who learned one submodule guesses wrong in the next; wrappers and configuration files have to map names; documentation and generated skills repeat the inconsistency.

## Current evidence

- `python -B tools/maint_tools/sweep_runtime.py --json out.json` (main cba9383), cluster report:
- `log_level` in 1 public callable(s): `mcp.RuntimeConfig`
- `quiet` in 1 public callable(s): `stats.knuth_bin_width`
- `verbose` in 18 public callable(s): `api.plot_learning_curve`, `api.test`, `config.test`, `cython.build_extension_module`, `cython.build_extension_module_result`, `cython.build_extension_package_from_code_result` ...

## Root cause / current understanding

Submodules were written at different times against different upstream conventions; there is no naming register.

## Expected behavior

One canonical name per idea (proposed: `verbose`), documented in a naming register in the engineering rules and used by every new or changed public callable; where two scopes differ (a per-file and a per-archive limit) the name says the scope with one pattern.

## Affected paths and ownership

- the public callables listed above
- `plugins/_shared/skills/scikitplot-engineering-rules/` (naming register)
- `tools/maint_tools/sweep_runtime.py` (`CLUSTERS`, the check)

## Constraints and non-goals

Renaming public parameters is an API decision; the campaign rule is no backward compatibility, so a rename lands with its fragment and usage-skill update (Rule 172).

## Edge cases to cover

- Classes whose `__init__` takes the old name positionally
- Configuration dataclasses serialised to JSON with the old key
- Names that look alike but differ in meaning (keep them, document why)

## Proposed direction

Decide the canonical name, rename in one change per submodule, add the cluster to a `sweep_runtime` gate so a new divergent spelling fails review.

## Verification / acceptance criteria

- `sweep_runtime.py` reports no cluster for this idea, or only the documented exceptions

## Documentation impact

API reference, usage skills, the naming register.

## Release-note promotion

Required: `scikitplot` / `api`, one bullet, when the change lands.
