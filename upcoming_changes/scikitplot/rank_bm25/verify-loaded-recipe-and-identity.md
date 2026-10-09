---
title: "Verify Rank-BM25 recipe and identity when loading artifacts"
status: open
kind: "persistence-integrity"
area: "scikitplot/rank_bm25"
discovered_during: "source-grounded Rank-BM25 user-guide synchronization"
release_note: "likely"
towncrier_section: "scikitplot.rank_bm25"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Verify Rank-BM25 recipe and identity when loading artifacts

## Summary

``BM25.load`` documents refusal of state from the wrong recipe, and persisted
state records both ``recipe`` and ``identity``, but the current loader checks
only the schema before reconstructing the requested class.

## Current evidence

``_state`` writes ``schema``, ``recipe``, ``identity``, parameters, analyzer,
statistics, and IDs.  ``load`` verifies ``schema == "bm25-index/1"`` but does
not compare ``state["recipe"]`` with ``cls.RECIPE_ID`` and does not recompute
or verify ``state["identity"]``.

## Why it matters

Calling one scorer class's ``load`` on another recipe's generation can construct
an incoherent object or fail later at query time instead of refusing the wrong
artifact at the load boundary.  Modified metadata can likewise disagree with
its recorded identity without detection.

## Expected behavior

Loading should reject a generation whose declared recipe is incompatible with
the requested class.  Decide whether the recorded identity is an integrity
check, informational metadata, or both, and implement/document that contract
explicitly.

## Constraints and edge cases

- existing valid ``bm25-index/1`` artifacts;
- loading via publication root versus generation directory;
- all three concrete scorer classes;
- missing/unknown recipe identifiers;
- altered parameters or analyzer declaration;
- forward compatibility if a future schema changes identity semantics.

## Verification / acceptance criteria

- cross-recipe load attempts fail immediately with an actionable ``ValueError``;
- valid artifacts still round trip;
- the docstring and user guide describe only checks the loader actually makes;
- any identity verification has deterministic tests for tampered metadata.
