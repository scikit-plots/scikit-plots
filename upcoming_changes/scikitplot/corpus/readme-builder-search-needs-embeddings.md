---
title: "The README's CorpusBuilder example raises: search defaults to hybrid, the build has no embeddings"
status: open
kind: "docs-contract"
area: "scikitplot/corpus"
discovered_during: "PR 919: research self-test of corpus and mcp (finding CX-09)"
release_note: "unknown"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# The README's CorpusBuilder example raises: search defaults to hybrid, the build has no embeddings

## Summary

`scikitplot/corpus/README.md`, section 2, builds with
`BuilderConfig(chunker="paragraph", normalize=True, enrich=True,
build_index=True)` and then calls `builder.search("financial protection")`.
Run as written on a one-file folder it raises `RuntimeError:
match_mode='hybrid' requires embeddings. Set embed=True in BuilderConfig
and rebuild.`

## Why it matters

It is the README's "end-to-end convenience" example: the first thing a user
copies fails. The error itself is correct and actionable.

## Current evidence

- `CorpusBuilder.search(..., match_mode="hybrid")` is the default
  (`_corpus_builder.py`), and `BuilderConfig.embed` defaults to off.
- The example, run verbatim on `main` at `6734def`: the `RuntimeError`
  above.
- No other page or gallery example calls `CorpusBuilder.search` this way
  (the one in `galleries/examples/corpus/plot_corpus_who_per_file_script.py`
  is commented out).

## Root cause / current understanding

The example and the default disagree: a default `search` needs a build
option that the example (and the default config) does not set.

## Expected behavior

The README example runs as written and returns hits.

## Affected paths and ownership

- `scikitplot/corpus/README.md` (section 2)
- possibly `CorpusBuilder.search` (`_corpus_builder.py`)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

Do not weaken the error for a hybrid or semantic search on an index without
embeddings: an explicit `match_mode` that cannot run must still raise.

## Edge cases to cover

Default config; `embed=True`; an explicit `match_mode` of each value.

## Proposed direction

Two options, the maintainer's choice:
(a) the README passes `match_mode="keyword"` (or sets `embed=True` and says
it needs an embedding model): documentation only;
(b) `search` takes `match_mode=None`, meaning hybrid when the build has
embeddings and keyword when it has none, while an explicit mode keeps
raising: a default change (pre-1.0, no compatibility path).

## Verification / acceptance criteria

The README example executed in a test (or a doctest) on a one-file folder.

## Documentation impact

`scikitplot/corpus/README.md`; with (b), the `search` docstring.

## Release-note promotion

(a): `documentation`, type `fix`; (b): `scikitplot.corpus`, type `api`.
