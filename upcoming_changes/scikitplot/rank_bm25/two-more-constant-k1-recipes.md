---
title: "rank_bm25: two more constant-k1 recipes (log(N/df) idf; unscaled tf with log(1 + ...) idf)"
status: open
kind: "api"
area: "scikitplot/rank_bm25"
discovered_during: "BM25-Adpt and BM25T design review (pull request 1027)"
release_note: "required"
towncrier_section: "scikitplot.rank_bm25"
towncrier_type: "feature"
towncrier_fragment: ""
---

# rank_bm25: two more constant-k1 recipes

## Summary

`scikitplot.rank_bm25` ships five recipes. Two further, fully specified
variants with one `k1` for every term are not available:

- A: idf `log(N / df)`; term part `(k1 + 1) tf / (k1 (1 - b + b dl / avgdl) + tf)`.
- B: idf `log(1 + (N - df + 0.5) / (df + 0.5))`; term part
  `tf / (k1 (1 - b + b dl / avgdl) + tf)` (no `(k1 + 1)` factor, which scales
  every score by a constant and leaves rankings unchanged).

## Why it matters

Both are common reference points when comparing lexical rankers; a user who
needs scores comparable with a system using either cannot reproduce them with
the current recipes (Okapi floors negative idfs; BM25L and BM25+ shift the
term part).

## Current evidence

- `scikitplot/rank_bm25/_rank_bm25.py`: `_SCORERS` lists five classes; none
  computes either idf above.
- `RECIPES` in `_identity.py` holds no matching recipe.

## Root cause / current understanding

Not implemented; deferred from pull request 1027 to keep that change to the
two adaptive-k1 recipes.

## Expected behavior

Two classes, each with its own recipe identifier at version 1, constructor
parameters `k1` and `b` with the same domains as `BM25Okapi`, scored by the
precomputed engine bit for bit like the reference path, persisted and reloaded
like the other recipes.

## Affected paths and ownership

- `scikitplot/rank_bm25/_rank_bm25.py` (classes, `_SCORERS`)
- `scikitplot/rank_bm25/_identity.py` (`RECIPES`)
- `scikitplot/_build_utils/generate_apis_reference/apis_reference.py.in`
  (then regenerate `docs/source/apis_reference.py`)

## Constraints and non-goals

- Constant `k1`: the three-argument `_term_contribution`; no change to the
  existing recipes' expression trees or identifiers.
- A: `log(N / df)` is 0 for a term in every document; that is the definition,
  not floored.
- Not in scope: variants whose formula cannot be stated exactly from a
  reliable source, and multi-field (BM25F) scoring, which needs a different
  input contract.

## Edge cases to cover

- A term in every document (A: weight 0; B: positive).
- Empty documents with `b == 1` (absent-term value, as for the other recipes).
- Engine equivalence on random corpora; reload equality; a generation of one
  recipe refused by the other class.

## Proposed direction

Add both to the shared `CLASSES` lists of the existing tests and give each a
small idf test against the formula written inline.

## Verification / acceptance criteria

The rank_bm25 suite green on the newest CPython and on 3.8; engine output
bit-identical to `backend="reference"`; doctests pass.

## Documentation impact

`scikitplot/rank_bm25/__init__.py` list of algorithms; API reference section
"BM25 Variations".

## Release-note promotion

Required: `docs/source/whats_new/upcoming_changes/scikitplot.rank_bm25/<PR>.feature.rst`.
