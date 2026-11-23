---
title: "Public API spells 'size limit' 5 ways (max_bytes, max_content_bytes, max_file_bytes, max_response_bytes, max_total_bytes)"
status: open
kind: "api"
area: "scikitplot"
discovered_during: "Package sweep, tools/maint_tools/sweep_runtime.py on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot"
towncrier_type: "api"
towncrier_fragment: ""
---

# Public API spells 'size limit' 5 ways (max_bytes, max_content_bytes, max_file_bytes, max_response_bytes, max_total_bytes)

## Summary

Across public callables, the idea 'size limit' is named `max_bytes`, `max_content_bytes`, `max_file_bytes`, `max_response_bytes`, `max_total_bytes`. Byte limits are spelled five ways; some differ in scope (one file, one response, a whole archive), which the name should say consistently.

## Why it matters

A user who learned one submodule guesses wrong in the next; wrappers and configuration files have to map names; documentation and generated skills repeat the inconsistency.

## Current evidence

- `python -B tools/maint_tools/sweep_runtime.py --json out.json` (main cba9383), cluster report:
- `max_bytes` in 12 public callable(s): `corpus.AnyDownloader`, `corpus.BaseDownloader`, `corpus.CachePolicy`, `corpus.CustomDownloader`, `corpus.DownloadPlan`, `corpus.DownloadPolicy` ...
- `max_content_bytes` in 1 public callable(s): `corpus.WebReader`
- `max_file_bytes` in 11 public callable(s): `corpus.ALTOReader`, `corpus.AudioReader`, `corpus.ImageReader`, `corpus.MarkdownReader`, `corpus.OfficeReader`, `corpus.PDFReader` ...
- `max_response_bytes` in 1 public callable(s): `corpus.WebReader`
- `max_total_bytes` in 2 public callable(s): `corpus.ZipReader`, `corpus.extract_archive`

## Root cause / current understanding

Submodules were written at different times against different upstream conventions; there is no naming register.

## Expected behavior

One canonical name per idea (proposed: `max_bytes`), documented in a naming register in the engineering rules and used by every new or changed public callable; where two scopes differ (a per-file and a per-archive limit) the name says the scope with one pattern.

## Affected paths and ownership

- the public callables listed above
- `plugins/_shared/skills/scikitplot-engineering-rules/` (naming register)
- `tools/maint_tools/sweep_runtime.py` (`CLUSTERS`, the check)

## Constraints and non-goals

Renaming public parameters is an API decision; the campaign rule is no backward compatibility, so a rename lands with its fragment and usage-skill update (Rule 172).

## Edge cases to cover

- Classes whose `__init__` takes the old name positionally
- Configuration dataclasses serialised to JSON with the old key
- Names that look alike but differ in meaning (keep them, document why)

## Proposed direction

Decide the canonical name, rename in one change per submodule, add the cluster to a `sweep_runtime` gate so a new divergent spelling fails review.

## Verification / acceptance criteria

- `sweep_runtime.py` reports no cluster for this idea, or only the documented exceptions

## Documentation impact

API reference, usage skills, the naming register.

## Release-note promotion

Required: `scikitplot` / `api`, one bullet, when the change lands.
