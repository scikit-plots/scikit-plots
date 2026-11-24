---
title: "Generation: no asset store for media outputs (local and S3-compatible)"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation: no asset store for media outputs (local and S3-compatible)

## Summary

The agent's artifact stores keep JSON artifacts; generated media (images, audio, video) need a content-addressed store with metadata, local or S3-compatible.

## Why it matters

Assets must be deduplicated, referenced from manifests and served; ad-hoc file names collide and leak paths.

## Current evidence

- `python -m scikitplot api scikitplot.agent` lists no public name matching `\bS3|BinaryStore|MediaStore`.
- Already public in `scikitplot.agent`: `ArtifactStore`, `DirectoryArtifactStore`, `MemoryArtifactStore` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A store keyed by content hash with metadata (request, model, seed, licence, provenance), local folder and S3-compatible backends, and signed or local URLs for serving.

## Affected paths and ownership

- `scikitplot/genai`
- `scikitplot/agent` (artifact stores)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Large files (streaming upload)
- Same content from two requests (one object, two references)
- Deletion policy

## Proposed direction

Extend the artifact store protocol to binary objects; S3 through `fsspec`.

## Verification / acceptance criteria

- Two identical outputs are stored once; metadata round-trips

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
