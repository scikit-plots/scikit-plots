---
title: "corpus: no Arrow table or Parquet export"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no Arrow table or Parquet export

## Summary

Exports exist for Polars, JSONL and ML frameworks, but not Apache Arrow tables or Parquet files, the interchange formats of DuckDB, Spark, Polars and most data platforms.

## Why it matters

Large corpora written as JSONL are slow to reload and lose types; Parquet keeps types, compresses and supports partial reads.

## Current evidence

- `python -m scikitplot api scikitplot.corpus` lists no public name matching `arrow|parquet`.
- Already public in `scikitplot.corpus`: `documents_to_polars` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

`to_arrow(documents)` with a versioned schema (schema version in the metadata), `to_parquet(documents, path, partition_by=...)` and `from_parquet` round trip; embeddings as fixed-size lists.

## Affected paths and ownership

- `scikitplot/corpus/_export/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Mixed metadata types across documents
- Very large text fields
- Schema evolution between releases (versioned field)

## Proposed direction

Implement on `pyarrow` (optional); reuse the field mapping of the Polars export.

## Verification / acceptance criteria

- Round trip keeps every field and type; a column projection reads only `text`

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
