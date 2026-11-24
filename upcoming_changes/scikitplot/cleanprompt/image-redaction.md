---
title: "cleanprompt: personal data in images cannot be redacted"
status: open
kind: "other"
area: "scikitplot/cleanprompt"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "feature"
towncrier_fragment: ""
---

# cleanprompt: personal data in images cannot be redacted

## Summary

cleanprompt redacts text, including text extracted from documents. Personal data visible in images (scanned forms, screenshots, burned-in DICOM annotations, photos of IDs) stays in the image.

## Why it matters

Sharing images for annotation or model input leaks what the text path removed.

## Current evidence

- `grep -rEil 'redact_image|bounding_box' scikitplot/cleanprompt --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

OCR with word boxes, the same detectors on the OCR text, and boxes filled on a copy of the image; a report of what was covered; faces optional through a detector backend.

## Affected paths and ownership

- `scikitplot/cleanprompt`
- `scikitplot/corpus/_readers/_image.py` (OCR boxes)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Rotated text
- Low OCR confidence (box anyway or flag)
- Multi-page TIFF

## Proposed direction

Reuse the image reader's OCR with bounding boxes.

## Verification / acceptance criteria

- A synthetic form with an email address has the address box filled; the report lists it

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.cleanprompt` / `feature`, one bullet, when the change lands.
