---
title: "corpus: PAGE-XML and hOCR (OCR formats beside ALTO) are not recognised"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: PAGE-XML and hOCR (OCR formats beside ALTO) are not recognised

## Summary

ALTO is read by default (PR 953), but the other two common OCR exchange formats are not: PAGE-XML (root `PcGts`) goes to the generic `XMLReader`, which repeats nested text (CX-46), and hOCR (`.hocr`, or HTML with `ocr_page` classes) is refused.

## Why it matters

Digitisation projects and OCR engines produce PAGE-XML and hOCR as often as ALTO; their reading order, line geometry and confidence are lost.

## Current evidence

- `DocumentReader.create('file.hocr')` raises `ValueError: No DocumentReader registered for extension '.hocr'`: the reader registry on main (84 entries) has no `.hocr`.
- `grep -rEil 'PcGts|ocr_page' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Readers detected by root element or class (as `alto="auto"`), yielding regions in reading order with coordinates and confidence, and hyphenation handled as for ALTO.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_xml.py`
- `scikitplot/corpus/_readers/_alto.py` (shared line logic)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Reading order given explicitly (PAGE `ReadingOrder`)
- Word confidence below a threshold
- Mixed scripts

## Proposed direction

Reuse the ALTO line and dehyphenation helpers.

## Verification / acceptance criteria

- Synthetic PAGE-XML and hOCR pages give the same text as their ALTO equivalent

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
