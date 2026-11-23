---
title: "corpus: no SQL access to a built corpus"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no SQL access to a built corpus

## Summary

Analysts filter and aggregate corpora with SQL (count by source, find documents by metadata); the corpus offers no SQL view.

## Why it matters

Every ad-hoc question needs Python code; joins with other tables (patients, products) are impossible without an export.

## Current evidence

- `grep -rEil 'duckdb' scikitplot/corpus --include=*.py` (tests and vendored code excluded) finds nothing.
- Related code exists: `scikitplot/datasets/_data_loader.py`.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

An optional DuckDB view over the Arrow/Parquet export (`corpus.sql("select ...")`) and documentation of the schema.

## Affected paths and ownership

- `scikitplot/corpus/_export/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Concurrent writes while querying
- Large embeddings columns (excluded by default)

## Proposed direction

After the Arrow export: register it in DuckDB in-process (`scikitplot.datasets.load_duckdb` already wraps DuckDB for tables).

## Verification / acceptance criteria

- A query over a synthetic corpus returns the expected counts

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
