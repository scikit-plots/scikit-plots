---
title: "No shared policy profile across submodules"
status: open
kind: "other"
area: "scikitplot"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "feature"
towncrier_fragment: ""
---

# No shared policy profile across submodules

## Summary

Limits and safety settings are configured per class: readers (`max_file_bytes`, `ocr`, `transcribe`), downloaders (`timeout`, `max_bytes`), agents (`Limits`), cleanprompt (`RedactionPolicy`), retrieval (`FusionPolicy`, `CutPolicy`), embeddings (`CachePolicy`). There is no profile that sets them together (strict, default, permissive) or a file that a team can review.

## Why it matters

A deployment cannot state 'no network, 64 MiB per file, OCR on, redaction strict' once; settings drift between components and are invisible in review.

## Current evidence

- `grep -rEil 'policy_profile|PolicyProfile' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.
- Already public in `scikitplot.corpus`: `PipelineGuard`, `BackendPlan`, `BackendPolicy`, `backend_policy`, `plan_backend_chain`, `DownloadPolicy` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A policy profile (TOML) with named sections per submodule, validated against each component's parameters, loaded explicitly (never implicitly from the environment), shown and checked by `python -m scikitplot policy show|check|init`, and recorded in run reports.

## Affected paths and ownership

- `scikitplot/corpus`
- `scikitplot/agent`
- `scikitplot/cleanprompt`
- `scikitplot/mcp`
- `scikitplot/_cli`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Unknown keys (error)
- A component's own argument overriding the profile (explicit wins, recorded)
- Profiles inheriting from a base

## Proposed direction

Define the schema from the existing policy classes; one loader; components accept a profile object.

## Verification / acceptance criteria

- `policy check` rejects a profile with an unknown key; a run report lists the effective settings

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot` / `feature`, one bullet, when the change lands.
