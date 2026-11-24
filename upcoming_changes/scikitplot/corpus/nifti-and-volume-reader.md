---
title: "corpus: no reader for research volumes (NIfTI)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no reader for research volumes (NIfTI)

## Summary

Neuroimaging and segmentation work stores volumes as NIfTI (`.nii`, `.nii.gz`), with a header describing voxel sizes and orientation.

## Why it matters

Research datasets and model outputs (segmentations, measurements) cannot be catalogued next to their reports.

## Current evidence

- `DocumentReader.create('file.nii')` raises `ValueError: No DocumentReader registered for extension '.nii'`: the reader registry on main (84 entries) has no `.nii`.
- `grep -rEil 'nibabel' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A reader (optional `nibabel`) yielding header metadata (shape, voxel spacing, orientation, data type) and, when asked, summary statistics or rendered slices for captioning; `.nii.gz` handled under the archive expansion limit.

## Affected paths and ownership

- `scikitplot/corpus/_readers/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Very large 4-D volumes (memory map, never load whole)
- Gzip bombs
- Non-standard orientation codes

## Proposed direction

Header-first reader with memory-mapped data access.

## Verification / acceptance criteria

- A small synthetic volume yields its header fields; a 4-D volume is not loaded into memory

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
