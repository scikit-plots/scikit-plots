---
title: "corpus: PDF tables are read as running text"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: PDF tables are read as running text

## Summary

PDF pages are read as text in layout order; a table's rows and columns are flattened, so a number loses its row and column header.

## Why it matters

Financial, clinical and statistical reports keep their key facts in tables; question answering over them needs cells with headers.

## Current evidence

- `grep -rEil 'camelot|extract_tables|find_tables' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).
- A caller can pass a `custom_extractor` (the `corpus/_base.py` docstring shows one built on pdfplumber); no reader produces table chunks.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Optional table detection on PDF pages producing table chunks (rows with headers, CSV or Markdown text, cell coordinates) next to the page text, with `section_type` set to table.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_pdf.py`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Tables spanning pages
- Merged cells
- Rotated tables
- Scanned tables (OCR first)

## Proposed direction

Start with ruled tables (line detection from the PDF's drawing operators); measure with the reader benchmark.

## Verification / acceptance criteria

- A synthetic PDF table yields one row per record with headers

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
