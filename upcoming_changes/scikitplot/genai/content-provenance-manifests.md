---
title: "Generation: generated media carry no provenance manifest"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation: generated media carry no provenance manifest

## Summary

Generated images, audio and video are not labelled as generated; no content-credential manifest records the model, the request and edits.

## Why it matters

Regulation and platforms increasingly require disclosure of AI-generated media; provenance also proves which asset came from which request.

## Current evidence

- `grep -rEil 'c2pa|content_credential' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Optional content-credential manifests (C2PA-style) embedded or written beside each asset, plus visible or metadata disclosure flags, and reading of manifests in the corpus image reader.

## Affected paths and ownership

- `scikitplot/genai`
- `scikitplot/corpus/_readers/_image.py` (reading manifests)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Formats without embedding support (sidecar file)
- Edits after generation
- Stripped metadata

## Proposed direction

Sidecar JSON first, embedding through an optional library.

## Verification / acceptance criteria

- Each generated asset has a manifest naming model, request hash and time

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
