---
title: "api: no projection plot for embeddings and corpora"
status: open
kind: "other"
area: "scikitplot/api"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.api"
towncrier_type: "feature"
towncrier_fragment: ""
---

# api: no projection plot for embeddings and corpora

## Summary

PCA projections exist for estimators; there is no plot for an embedding matrix (a corpus's chunks, a model's representations) coloured by metadata, with optional UMAP or t-SNE.

## Why it matters

Inspecting a corpus's coverage and clusters is a first step of retrieval debugging.

## Current evidence

- `python -m scikitplot api scikitplot.api` lists no public name matching `embedding|umap|tsne`.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

`plot_embedding_projection(X, labels=..., method='pca'|'umap'|'tsne', seed=...)` with sampling for large inputs and hover labels in interactive backends.

## Affected paths and ownership

- `scikitplot/api`
- `scikitplot/corpus` (embeddings)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Millions of points (sampling)
- UMAP not installed (PCA fallback with a note)
- Cosine vs Euclidean

## Proposed direction

Reuse the PCA plot's style; optional dependencies lazily.

## Verification / acceptance criteria

- A synthetic two-cluster set is drawn as two groups with a fixed seed

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.api` / `feature`, one bullet, when the change lands.
