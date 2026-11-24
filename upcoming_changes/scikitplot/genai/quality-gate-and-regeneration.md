---
title: "Generation: no quality gate that regenerates weak outputs"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation: no quality gate that regenerates weak outputs

## Summary

Generated assets are not checked: no rule or model judges whether an image matches its brief, speech is intelligible, or text follows its constraints, and nothing regenerates failures within a retry budget.

## Why it matters

Bad assets reach users; manual review does not scale to batches.

## Current evidence

- `grep -rEil 'multimodal_judge|regeneration_policy' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Checks per modality (rules: size, duration, text constraints; model judges: image-text agreement, speech-to-text round trip), a pass threshold, regeneration with a new seed up to a limit, and every verdict recorded with the asset.

## Affected paths and ownership

- `scikitplot/genai`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Judges disagreeing
- Infinite regeneration (limit)
- Checks that are themselves expensive (budgeted)

## Proposed direction

Pluggable checks; the speech round trip reuses the corpus transcription.

## Verification / acceptance criteria

- A stub backend that fails half its outputs is regenerated until the gate passes or the limit is hit; verdicts are recorded

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
