---
title: "Probability fusion under a SecurityFilter scope"
status: open
kind: "api"
area: "scikitplot/corpus"
discovered_during: "opt-in probability fusion for hybrid search (PR 906)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Probability fusion under a SecurityFilter scope

## Summary

`RetrievalConfig(probability_fusion=...)` raises `ValueError` when the index is
bound to a `SecurityFilter` scope, so scoped deployments can only use rank
fusion.

## Why it matters

Scoped indexes are the multi-tenant case, where calibrated probabilities (for
thresholds and abstention) are most useful.

## Current evidence

`corpus/tests/test__probability_hybrid.py::test_scoped_index_is_refused`.

## Root cause / current understanding

Verified: under a scope, BM25 document count, document frequencies and average
length are computed over the admitted rows (`_BM25Index.query(scope=...)`), so
the same document gets a different BM25 score than unscoped. A calibrator fitted
on unscoped scores maps the wrong scale. Dense cosine scores do not change with
the scope.

## Expected behavior

A calibrator can be bound to the scope it was fitted under, and probability
fusion runs when the query's scope matches it.

## Affected paths and ownership

`scikitplot/corpus/_similarity/_similarity.py` (`_search_hybrid_with_legs`),
`scikitplot/corpus/_logodds_fusion.py` (`ProbabilityFusion`),
`scikitplot/rank_bm25/_calibration.py` (population declaration).

## Constraints and non-goals

Must not weaken scope isolation; must not infer scope identity from a string
the caller typed. Rank fusion behaviour is unchanged.

## Edge cases to cover

Scope label equal but filter different; dense-only fusion under a scope (no
BM25 dependence); calibrator fitted unscoped used scoped (still refused).

## Proposed direction

Key calibrators by the scope's canonical filter digest (`filter_to_dict` +
canonical encoding); allow dense-only probability fusion under any scope.

## Verification / acceptance criteria

Matching scope digest fuses; any mismatch raises naming both digests; the
dense-only case passes under a scope.

## Documentation impact

Corpus retrieval guide, "Probability fusion".

## Release-note promotion

Required: `scikitplot.corpus`, `feature`.
