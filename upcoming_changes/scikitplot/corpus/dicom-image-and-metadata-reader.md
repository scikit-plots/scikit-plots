---
title: "corpus: no reader for DICOM images and their metadata"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no reader for DICOM images and their metadata

## Summary

`.dcm` files (CT, MR, X-ray, ultrasound) are refused. A DICOM object holds pixel data and hundreds of tagged attributes (modality, body part, study description, acquisition parameters) that are the searchable part of an imaging archive.

## Why it matters

Clinical and research imaging archives cannot be indexed, filtered by metadata or linked to their reports; DICOM is the format every scanner writes.

## Current evidence

- `DocumentReader.create('file.dcm')` raises `ValueError: No DocumentReader registered for extension '.dcm'`: the reader registry on main (84 entries) has no `.dcm`.
- `grep -rEil 'pydicom|dicom' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A `DICOMReader` (optional `pydicom`, pixel codecs optional) yields one record per instance: selected tags as metadata (configurable allow-list), the free-text tags (study and series description, comments) as text, an optional rendered image (window/level applied) for OCR or captioning, and a multi-frame series as frames with their index; private and personal tags are dropped unless the caller allows them.

## Affected paths and ownership

- `scikitplot/corpus/_readers/` (new reader)
- `scikitplot/cleanprompt` (personal data in free-text tags)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Multi-frame and enhanced multi-frame objects
- Compressed transfer syntaxes (JPEG 2000, JPEG-LS, RLE) without a codec installed
- DICOMDIR media
- Missing pixel data (structured reports, presentation states)
- Huge files (whole-slide imaging): byte and frame limits

## Proposed direction

Reader on `pydicom` with lazy pixel decoding; tag allow-list and redaction through cleanprompt; register `.dcm` and DICOM files without suffix by the `DICM` preamble when asked.

## Verification / acceptance criteria

- A synthetic DICOM (pydicom test data) yields its tags and text; personal tags are absent by default
- A hostile-sample case: truncated file, oversize frame count

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
