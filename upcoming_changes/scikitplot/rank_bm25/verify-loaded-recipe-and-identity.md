---
title: "Verify Rank-BM25 recipe and identity when loading artifacts"
status: promoted
kind: "reliability"
area: "scikitplot/rank_bm25"
discovered_during: "source-grounded Rank-BM25 user-guide synchronization"
release_note: "required"
towncrier_section: "scikitplot.rank_bm25"
towncrier_type: "api"
towncrier_fragment: "docs/source/whats_new/upcoming_changes/scikitplot.rank_bm25/909.api.rst"
---

# Verify Rank-BM25 recipe and identity when loading artifacts

## Summary

``BM25.load`` documents refusal of state from the wrong recipe, and persisted
state records both ``recipe`` and ``identity``, but the current loader checks
only the schema before reconstructing the requested class.

## Current evidence

Re-verified 2026-11-04 on the tree with PR 909 applied (schemas
``bm25-index/1`` and ``bm25-index/2`` readable; recipes now
``okapi-epsilon-floor/2``, ``bm25l-delta/2``, ``bm25plus-delta/2``).

``_state`` writes ``schema``, ``recipe``, ``identity``, parameters, analyzer,
statistics, and IDs. ``load`` checks ``schema in _READABLE_SCHEMAS`` and the
array digests, but never compares ``state["recipe"]`` with ``cls.RECIPE_ID``
and never recomputes ``state["identity"]``:

```text
BM25Okapi(...).save(d); BM25Plus.load(d)
  -> AttributeError: 'BM25Plus' object has no attribute 'delta'
     (raised while rebuilding the engine, not a refusal at the boundary)

BM25Plus.load(tests/data/v1-index)       # recorded "bm25plus-delta/1"
  -> loads as bm25plus-delta/2; recorded identity != loaded.identity
```

The second case is new with the recipe bump: an artifact built under a
previous formula version loads silently and is scored with the current one,
and its recorded ``identity`` no longer matches the object's.

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

- existing valid ``bm25-index/1`` and ``bm25-index/2`` artifacts;
- an artifact whose recipe is an older version of the requested class's
  recipe (``bm25plus-delta/1`` loaded by ``BM25Plus``): pre-1.0 policy is
  current-version-only, so refusing it is consistent;
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

## Resolution (2026-11-04)

`BM25.load` now checks, before rebuilding the engine: the class has a formula
(`BM25.load` is a `TypeError`); `state["recipe"] == cls.RECIPE_ID` (another
scorer's recipe names the class to use; an earlier version or unknown recipe
says to rebuild); the parameters are exactly the recipe's and inside their
domains (`_set_parameters`, the same table the constructors now use); and the
recorded `identity` equals the recomputed one. `bm25-index/1` is no longer
read (every such generation has a `/1` recipe, so it could only be refused);
the shipped `tests/data/v1-index` fixture now proves that refusal. Tests:
13 new in `tests/test_persistence_v2.py`; removing either the recipe or the
identity check fails 7 of them. Remove this note once the PR merges.
