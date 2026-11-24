---
title: "corpus: no sources for S3-compatible and cloud object storage"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no sources for S3-compatible and cloud object storage

## Summary

Folders on S3-compatible storage (and GCS, Azure Blob) cannot be read directly; downloaders cover URLs, GitHub and two cloud drives.

## Why it matters

Production corpora and generated assets live in object storage; copying them locally first doubles storage and time.

## Current evidence

- `grep -rEil 'import fsspec|fsspec\.open|s3fs\.' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).
- `scikitplot/_compat/optional_deps.py` lists `fsspec` and `s3fs` as optional dependencies; no corpus source uses them.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A source over `fsspec` URLs (`s3://`, `gs://`, `az://`) that lists, filters and streams objects into the existing readers, with credentials from the environment, the same byte limits, and resumable listing.

## Affected paths and ownership

- `scikitplot/corpus/_sources/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Millions of objects (paginated listing)
- Objects changing during a build (ETag check)
- Credentials missing (clear error)
- Requester-pays buckets

## Proposed direction

Implement with `fsspec` as an optional extra; reuse `CorpusBuilder` source expansion rules (CX-53).

## Verification / acceptance criteria

- A local S3-compatible test server (moto or MinIO) serves a folder that builds like a local one

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
