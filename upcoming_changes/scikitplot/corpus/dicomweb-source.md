---
title: "corpus: no DICOMweb source (query and retrieve studies over HTTP)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no DICOMweb source (query and retrieve studies over HTTP)

## Summary

Imaging archives (PACS, research platforms) expose studies through DICOMweb (QIDO-RS search, WADO-RS retrieve). The corpus can download from URLs, GitHub and cloud drives, but not query a DICOMweb server.

## Why it matters

Pulling a cohort means exporting files by hand; the corpus cannot select studies by metadata at the source.

## Current evidence

- `grep -rEil 'dicomweb|wado|qido' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A source that searches a DICOMweb endpoint (study, series, instance level, with filters), retrieves metadata first and pixel data only when asked, under the same network policy (SSRF checks, timeouts, byte limits, credentials from the environment).

## Affected paths and ownership

- `scikitplot/corpus/_sources/`
- `scikitplot/corpus/_downloader/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Paging of large result sets
- Authentication tokens that expire mid-run
- Partial retrieval failures (resume)
- Servers returning multipart responses

## Proposed direction

Implement on the existing downloader and URL policy; metadata-only mode first.

## Verification / acceptance criteria

- A recorded fixture server answers a study query; the source yields the instances' metadata

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
