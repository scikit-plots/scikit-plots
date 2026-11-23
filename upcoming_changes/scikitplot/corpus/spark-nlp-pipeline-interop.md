---
title: "corpus: no interop with Spark-based NLP pipelines"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no interop with Spark-based NLP pipelines

## Summary

Clinical NLP at scale often runs as Spark pipelines (entity recognition, assertion status, relation extraction, de-identification models). The corpus has no adapter to hand documents to such a pipeline and read the annotations back.

## Why it matters

Annotations computed at scale cannot enrich corpus documents (entities, codes, negation) for retrieval and evaluation.

## Current evidence

- `grep -rEil 'sparknlp|spark_nlp' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

An adapter that converts documents to the pipeline's input DataFrame and maps the annotation columns back to document metadata (span, label, confidence), with the open-source pipeline library optional and licensed models supplied by the user.

## Affected paths and ownership

- `scikitplot/corpus/_export/`
- `scikitplot/corpus/_enrichers/` (or equivalent)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings. Licensed models are never downloaded or bundled by the package.

## Edge cases to cover

- Pipelines that change text offsets
- Licensed model packages absent (clear error, no download)
- Annotations overlapping cleanprompt redactions

## Proposed direction

Start from the Spark DataFrame export; map annotations by character offsets.

## Verification / acceptance criteria

- A local pipeline with an open-source model annotates a synthetic note; spans map back onto the document text

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
