---
title: "Unify Rank-BM25 query-token validation across public methods"
status: open
kind: "reliability"
area: "scikitplot/rank_bm25"
discovered_during: "source-grounded Rank-BM25 user-guide synchronization"
release_note: "likely"
towncrier_section: "scikitplot.rank_bm25"
towncrier_type: "fix"
towncrier_fragment: ""
---

# Unify Rank-BM25 query-token validation across public methods

## Summary

The documented query contract is a sequence of string tokens, but validation is
not applied consistently across the public query methods.

## Current evidence

- ``BM25Okapi.get_scores``, ``BM25L.get_scores``, and ``BM25Plus.get_scores``
  call ``_require_query_tokens`` and reject a bare string.
- ``get_top_n`` reaches ``get_scores`` and therefore inherits that guard.
- ``get_top_ids`` calls ``_sparse_scores`` directly; ``_sparse_scores`` does not
  call ``_require_query_tokens``.
- ``candidate_count`` iterates the supplied query directly.
- each concrete ``get_batch_scores`` implementation iterates the supplied query
  without calling ``_require_query_tokens``.

A bare string can therefore be rejected on one public path and silently treated
as a sequence of characters on another.

## Expected behavior

All public query entry points should enforce one token-sequence contract before
scoring or candidate lookup.

## Edge cases to cover

- ``str``, ``bytes`` and ``bytearray`` queries;
- generators of valid string tokens;
- non-string tokens;
- empty queries;
- repeated query terms;
- parity across all three concrete recipes.

## Proposed direction

Validate at a shared boundary used by dense, sparse, candidate-count, and batch
paths rather than duplicating slightly different checks in each scorer.

## Verification / acceptance criteria

- every public query method accepts the same valid token sequences;
- every public query method rejects bare strings and non-string tokens with the
  same error shape;
- sparse/dense ranking behavior remains unchanged for valid queries.
