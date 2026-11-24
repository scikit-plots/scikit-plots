---
title: "corpus: no near-duplicate detection across documents"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no near-duplicate detection across documents

## Summary

Archives hold the same text many times (versions, forwarded emails, re-exported PDFs, animated-image frames). Exact deduplication by hash misses near duplicates.

## Why it matters

Retrieval returns the same passage several times; evaluation sets leak into training; storage grows.

## Current evidence

- `grep -rEil 'minhash|simhash' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Optional MinHash/LSH (or SimHash) signatures per chunk, a threshold, and a policy (drop, keep first, keep longest, group) with a report of the clusters.

## Affected paths and ownership

- `scikitplot/corpus/` (filter or deduplication stage)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Short chunks (signature unreliable)
- Different scripts of the same text
- Millions of chunks (LSH bands)

## Proposed direction

Standard-library MinHash with a fixed seed; LSH banding; deterministic cluster order.

## Verification / acceptance criteria

- Two copies of a document with one edited sentence are grouped; distinct documents are not

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
