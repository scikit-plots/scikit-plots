---
title: "corpus: no reader for FHIR resources and bundles"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no reader for FHIR resources and bundles

## Summary

Clinical data exchanged as FHIR (JSON bundles of Patient, Observation, DiagnosticReport, DocumentReference) is refused or read as raw JSON (CX-50).

## Why it matters

Reports, observations and their codes are the text that imaging and lab data need for retrieval and question answering.

## Current evidence

- `grep -rEil 'fhir' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A reader that walks a bundle, yields one record per resource with its type, codes (LOINC, SNOMED CT, ICD) and narrative text, and resolves attachments (base64 PDFs) through the existing readers under the expansion limits.

## Affected paths and ownership

- `scikitplot/corpus/_readers/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Huge bundles (streaming parse)
- Contained resources and references
- Base64 attachments (size limits)
- NDJSON bulk export

## Proposed direction

Stream-parse JSON; narrative through the HTML path; attachments through `DocumentReader.create`.

## Verification / acceptance criteria

- A synthetic bundle yields one record per resource with its codes

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
