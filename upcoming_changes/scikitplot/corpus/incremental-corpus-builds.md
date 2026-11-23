---
title: "corpus: a rebuild reads every file again"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: a rebuild reads every file again

## Summary

`CorpusBuilder.add()` appends sources the caller names, and embeddings have a disk cache (`CachePolicy`), but nothing records which files of a folder changed, were added or were removed since the last build, and the index is rebuilt from scratch on every `add()` (its docstring says incremental index updates are not supported).

## Why it matters

Rebuilding a large archive after adding one file costs the full time (61 s for 692 documents here, hours for media with OCR).

## Current evidence

- `grep -rEil 'st_mtime|mtime' scikitplot/corpus/_corpus_builder.py --include=*.py` (tests and vendored code excluded) finds nothing.
- Already public in `scikitplot.corpus`: `CachePolicy` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

An optional build manifest (path, size, modification time, content hash, reader and settings fingerprint) so unchanged sources are reused, removed ones dropped and changed ones re-read; the result equals a full rebuild.

## Affected paths and ownership

- `scikitplot/corpus/_corpus_builder.py`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- A file changed but same size and time (hash decides)
- Reader version or settings changed (all affected files re-read)
- Interrupted build (resume)
- Moved file (same hash, new path)

## Proposed direction

Content-addressed manifest beside the output; equality test against a full build.

## Verification / acceptance criteria

- Adding one file to a built folder re-reads one file; the output equals a full rebuild

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
