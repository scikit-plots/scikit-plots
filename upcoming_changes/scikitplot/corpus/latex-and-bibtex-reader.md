---
title: "corpus: no reader for LaTeX sources and BibTeX bibliographies"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no reader for LaTeX sources and BibTeX bibliographies

## Summary

`.tex` files are refused by `DocumentReader.create`. Research archives hold papers as LaTeX and references as BibTeX.

## Why it matters

Folders that hold these files are indexed without them, and the user is told only by an error per file (or nothing, inside an archive: CX-49).

## Current evidence

- `DocumentReader.create('file.tex')` raises `ValueError: No DocumentReader registered for extension '.tex'`: the reader registry on main (84 entries) has no `.tex`.
- `DocumentReader.create('file.bib')` raises `ValueError: No DocumentReader registered for extension '.bib'`: the reader registry on main (84 entries) has no `.bib`.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A LaTeX reader that keeps section structure and math as text, drops macros it cannot expand, and a BibTeX reader yielding one record per entry with fields as metadata.

## Affected paths and ownership

- `scikitplot/corpus/_readers/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- `\input` of other files (refused or resolved within the folder)
- Custom macros
- Malformed BibTeX entries

## Proposed direction

Implement as an optional reader with the shared limits and a hostile-sample case.

## Verification / acceptance criteria

- A synthetic file of the format yields its text and structure; a malformed one raises a clear error

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
