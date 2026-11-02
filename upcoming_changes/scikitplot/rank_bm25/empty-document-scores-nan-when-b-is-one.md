---
title: "Empty documents score NaN when b == 1"
status: open
kind: "bug"
area: "scikitplot/rank_bm25"
discovered_during: "precomputed scoring engine equivalence work (PR 906)"
release_note: "required"
towncrier_section: "scikitplot.rank_bm25"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Empty documents score NaN when b == 1

## Summary

With `b=1`, every recipe scores an empty document `NaN` for every query,
including queries whose terms the document cannot hold.

## Why it matters

`NaN` propagates into sums, fusion and metrics. `get_top_n` places it last
only because a stable `argsort` happens to; a caller reading `get_scores`
gets `NaN` with a `RuntimeWarning`.

## Current evidence

```python
from scikitplot.rank_bm25 import BM25Okapi

BM25Okapi([["a", "b"], [], ["b", "c", "c"]], b=1.0).get_scores(["b", "zz"])
# array([0.038..., nan, 0.028...])
```

Same for `BM25L` and `BM25Plus`.

## Root cause / current understanding

Verified: the length normaliser `1 - b + b * length / avgdl` is 0 when
`length == 0` and `b == 1`; the recipes then evaluate `0 / 0`.

## Expected behavior

An empty document holds no query term; its score should be the recipe's
absent-term value (0 for Okapi, `idf * delta`-style lower bound for the delta
recipes), finite, with no warning.

## Affected paths and ownership

`scikitplot/rank_bm25/_rank_bm25.py` (`_term_contribution` of each recipe),
`scikitplot/rank_bm25/_engine.py` (`irregular_rows`, which currently
reproduces the `NaN` on purpose).

## Constraints and non-goals

A score change, so a recipe version bump (`.../2`) is needed for `identity`
to distinguish old and new indexes. Do not change scores of non-empty
documents.

## Edge cases to cover

`b=1` and `b<1`; all-empty corpus is already refused; delta recipes; save and
reload; engine and reference paths.

## Proposed direction

Define the zero-length case explicitly in each formula (frequency is 0, so
use the absent-term value), bump recipe versions, then drop the irregular-row
path from the engine.

## Verification / acceptance criteria

No `NaN` for any recipe and `b` in `[0, 1]`; engine still bit-identical to
reference; identity differs between recipe versions.

## Documentation impact

Rank-BM25 user guide: parameter domain notes.

## Release-note promotion

Required: `scikitplot.rank_bm25`, `fix`.
