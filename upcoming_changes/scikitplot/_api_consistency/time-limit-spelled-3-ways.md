---
title: "Public API spells 'time limit' 3 ways (max_seconds, timeout, timeout_s)"
status: open
kind: "api"
area: "scikitplot"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "api"
towncrier_fragment: ""
---

# Public API spells 'time limit' 3 ways (max_seconds, timeout, timeout_s)

## Summary

Across public callables, the idea 'time limit' is named `max_seconds`, `timeout`, `timeout_s`. `timeout` (seconds) is the standard-library convention; `max_seconds` and `timeout_s` say the same with other spellings.

## Why it matters

A user who learned one submodule guesses wrong in the next; wrappers and configuration files have to map names; documentation and generated skills repeat the inconsistency.

## Current evidence

- `python -B tools/maint_tools/sweep_runtime.py --json out.json` (main cba9383), cluster report:
- `max_seconds` in 3 public callable(s): `agent.Limits`, `agent.delegate`, `corpus.StageBudget`
- `timeout` in 13 public callable(s): `corpus.AnyDownloader`, `corpus.BaseDownloader`, `corpus.CustomDownloader`, `corpus.DownloadPlan`, `corpus.DownloadPolicy`, `corpus.GitHubDownloader` ...
- `timeout_s` in 2 public callable(s): `cython.build_lock`, `mlflow.wait_tracking_ready`

## Root cause / current understanding

Submodules were written at different times against different upstream conventions; there is no naming register.

## Expected behavior

One canonical name per idea (proposed: `timeout`), documented in a naming register in the engineering rules and used by every new or changed public callable; where two scopes differ (a per-file and a per-archive limit) the name says the scope with one pattern.

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
