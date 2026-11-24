---
title: "corpus: no reader for EPUB ebooks"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no reader for EPUB ebooks

## Summary

`.epub` files are refused by `DocumentReader.create`. Books, manuals and course material are distributed as EPUB.

## Why it matters

Folders that hold these files are indexed without them, and the user is told only by an error per file (or nothing, inside an archive: CX-49).

## Current evidence

- `DocumentReader.create('file.epub')` raises `ValueError: No DocumentReader registered for extension '.epub'`: the reader registry on main (84 entries) has no `.epub`.
- `grep -rEil 'ebooklib|\bepub\b' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A reader that follows the spine order, reads XHTML chapters through the HTML path, keeps chapter titles as sections and refuses DRM-protected content clearly.

## Affected paths and ownership

- `scikitplot/corpus/_readers/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- DRM-protected files
- Fixed-layout EPUB
- Embedded fonts and images (skipped)

## Proposed direction

Implement as an optional reader with the shared limits and a hostile-sample case.

## Verification / acceptance criteria

- A synthetic file of the format yields its text and structure; a malformed one raises a clear error

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
