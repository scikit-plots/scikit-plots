---
title: "corpus: no retrieval by document structure (vectorless navigation)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: no retrieval by document structure (vectorless navigation)

## Summary

Long structured documents (reports, standards, books) are retrieved chunk by chunk. A structure-first method navigates the table of contents (a tree of sections) with a model choosing branches, without embeddings.

## Why it matters

Answers that depend on a section's context, or on documents whose terms embeddings miss, fail with chunk retrieval; structure navigation also explains why a passage was chosen.

## Current evidence

- `grep -rEil 'toc_tree|section_tree|tree_index' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A section tree built from headings (ALTO, PDF outlines, HTML headings, Markdown), a navigation policy (rule-based or model-driven with a budget), and the path through the tree as the citation.

## Affected paths and ownership

- `scikitplot/corpus/_chunkers/`
- `scikitplot/corpus/` (retrieval)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Documents without headings (fallback to chunks)
- Very deep trees (budget)
- Cycles in cross-references

## Proposed direction

Build the tree from existing section metadata; navigation as an alternative retriever behind the same interface.

## Verification / acceptance criteria

- On structured documents, answers cite a section path; budget limits are honoured

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
