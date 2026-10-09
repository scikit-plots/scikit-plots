---
title: "Resolve the Rank-BM25 analyzer persistence API boundary"
status: open
kind: "api-contract"
area: "scikitplot/rank_bm25"
discovered_during: "source-grounded Rank-BM25 user-guide synchronization"
release_note: "likely"
towncrier_section: "scikitplot.rank_bm25"
towncrier_type: "api"
towncrier_fragment: ""
---

# Resolve the Rank-BM25 analyzer persistence API boundary

## Summary

Durable persistence of an index built from raw text requires the internal
``Analyzer`` declaration, but ``Analyzer`` is not exported from the public
``scikitplot.rank_bm25`` namespace.

## Current evidence

- ``BM25._record_analyzer`` recognizes ``_identity.Analyzer`` and records it as
  the durable tokenization declaration.
- ``BM25.save`` refuses an index built with a bare callable tokenizer and tells
  the user to rebuild with an ``Analyzer``.
- ``BM25.load`` reconstructs ``Analyzer`` state from the artifact.
- ``scikitplot.rank_bm25.__all__`` exposes only ``BM25``, ``BM25Okapi``,
  ``BM25L``, and ``BM25Plus``; the only import path for ``Analyzer`` is the
  private ``scikitplot.rank_bm25._identity`` module.

## Why it matters

The supported persistence path currently directs users toward an object that
has no stable public import path.  Applications must either use a private API,
pre-tokenize outside the index, or avoid persistence after callable tokenization.

## Expected behavior

Choose and document one coherent contract:

1. make the analyzer declaration a supported public API (including its identity
   semantics and compatibility policy), or
2. redesign persistence so users do not need a private object to persist a
   reproducible text-analysis declaration.

## Constraints and edge cases

- Preserve loading of existing ``bm25-index/1`` artifacts unless a migration is
  explicitly designed.
- Keep bare callables usable for in-memory indexing.
- Do not pretend a callable can be fingerprinted safely from ``repr`` or code
  object details.
- Keep analyzer behavior and its recorded identity coupled.

## Verification / acceptance criteria

- the documented persistence path uses only public imports;
- save/load round trips a declared analysis policy;
- incompatible analyzer behavior moves the build identity;
- API/reference and user-guide documentation agree on the supported path.
