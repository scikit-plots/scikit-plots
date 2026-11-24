---
title: "Generation: no record of each model's licence and allowed uses"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation: no record of each model's licence and allowed uses

## Summary

Open-weight models come with their own licences (some forbid commercial use, some require attribution); nothing records which model produced an asset under which licence.

## Why it matters

Assets made with a non-commercial model can end up in a commercial product without anyone noticing.

## Current evidence

- `grep -rEil 'commercial_use|licen[cs]e_registry' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A registry of backends' models with licence identifiers and flags (commercial use, attribution, redistribution), checked against a policy before a run, and copied into each asset's metadata.

## Affected paths and ownership

- `scikitplot/genai`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Unknown licence (refused unless allowed)
- Licence changes between model versions
- Fine-tuned derivatives

## Proposed direction

Registry as data with per-backend entries; policy check before execution.

## Verification / acceptance criteria

- A run under a commercial-use policy refuses a non-commercial model

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
