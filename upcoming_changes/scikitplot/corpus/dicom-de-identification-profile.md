---
title: "corpus: no DICOM de-identification profile for tags and UIDs"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no DICOM de-identification profile for tags and UIDs

## Summary

cleanprompt detects many clinical identifiers in text (MRN, NPI, DOB, ICD10, DIAGNOSIS), but DICOM identifiers live in structured tags: patient name and ID, accession number, dates, institution, device serial numbers and the study, series and instance UIDs that link records.

## Why it matters

Imaging data cannot leave a hospital without de-identification; removing tags by hand breaks the UID links that tie series to studies.

## Current evidence

- `grep -rEil 'PS3\.15|0010,0010|accession_number' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.
- Already public in `scikitplot.cleanprompt`: `Handle`, `VaultScrubber`, `restore`, `RedactionResult`, `Vault` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A tag-level profile (the standard's basic profile with options: retain longitudinal dates shifted, retain UIDs remapped) applied before any text leaves the reader: UIDs replaced by stable pseudonyms, dates shifted per patient, burned-in text flagged for image redaction, and a report of what was changed.

## Affected paths and ownership

- `scikitplot/corpus/_readers/` (DICOM reader)
- `scikitplot/cleanprompt` (shared pseudonym vault)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- UIDs referenced inside sequences
- Same patient across files (consistent pseudonyms)
- Private tags with unknown meaning (removed)
- Burned-in annotations in pixel data

## Proposed direction

Implement the profile as data (tag -> action) with the vault from cleanprompt for consistent pseudonyms; test against the standard's attribute table.

## Verification / acceptance criteria

- Every identifying tag of a synthetic file is removed or replaced; UID references stay consistent

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
