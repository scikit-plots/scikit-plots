---
title: "corpus: no export to a Spark DataFrame"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no export to a Spark DataFrame

## Summary

Documents export to pandas-like structures, Polars, JSONL, Hugging Face datasets, PyTorch and TensorFlow, but not to Spark, where large clinical and enterprise corpora are processed.

## Why it matters

A corpus built here cannot feed a Spark pipeline without writing and re-reading files; schema and metadata types are lost on the way.

## Current evidence

- `python -m scikitplot api scikitplot.corpus` lists no public name matching `spark`.
- `grep -rEil 'pyspark' scikitplot/corpus --include=*.py` (tests and vendored code excluded) finds nothing.
- Already public in `scikitplot.corpus`: `to_huggingface_dataset`, `documents_to_polars` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

`to_spark_dataframe(documents, spark)` with an explicit schema (text, doc_id, metadata as a struct or map, embeddings as arrays), built through Arrow without collecting everything on the driver, and the reverse `from_spark_dataframe` for round trips.

## Affected paths and ownership

- `scikitplot/corpus/_export/`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Millions of documents (partitioned writes)
- Embeddings of differing length
- Metadata keys with mixed types
- No `pyspark` installed (clear install hint)

## Proposed direction

Build on an Arrow export (see the Arrow note) and `spark.createDataFrame` from Arrow batches.

## Verification / acceptance criteria

- Round trip of a synthetic corpus through a local Spark session keeps every field

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
