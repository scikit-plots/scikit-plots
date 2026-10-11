---
name: levenshtein-maintainer
description: Maintain scikitplot.levenshtein, the lazy edit-distance facade. Use for unit-cost Levenshtein metrics, backend discovery and fallback, license-safe automatic selection, deterministic ranking, score cutoffs, backend provenance, the optional Corpus retrieval adapter, docs/gallery, and release verification.
---

# `scikitplot.levenshtein` maintainer

Work from the wide checkout containing `scikitplot/`, `maintenances/`,
`skills/`, `docs/` and `galleries/`.

## Read first

1. `maintenances/levenshtein/MAINTAINING.md`
2. `maintenances/levenshtein/_maintenance/RESUME.md`
3. `maintenances/levenshtein/_maintenance/DESIGN.md`
4. `maintenances/levenshtein/REVIEW.json`
5. `maintenances/levenshtein/_maintenance/VERIFICATION.md`

Then run the focused checker/tests from `VERIFICATION.md`.

## Mental model

There are four implementation choices, but only three are automatically safe:

```text
auto
  ├── internal       bundled `scikitplot.cexternals._editdistance`
  ├── rapidfuzz      optional MIT accelerator
  └── python         dependency-free fallback

explicit only
  └── levenshtein    external GPL-2.0-or-later package
```

Never put the GPL backend into the automatic chain.

## Public metric contract

This facade implements unit-cost Levenshtein distance only:

```text
insert = 1
delete = 1
substitute = 1
```

For sequences `a` and `b`:

```text
distance
similarity = max(len(a), len(b)) - distance
normalized_distance = distance / max(len(a), len(b))
normalized_similarity = 1 - normalized_distance
```

Empty/empty normalized distance is `0.0`; normalized similarity is `1.0`.

Do not silently introduce Unicode normalization, case folding, tokenization,
transpositions or weighted costs into these functions.

## Import contract

`import scikitplot.levenshtein` must not import RapidFuzz, external
`Levenshtein`, or Corpus.

`make_corpus_scorer` owns a lazy adapter. Keep the Corpus import inside the
returned scorer.

## Backend failure contract

An explicitly requested unavailable backend:

- `strict=False`: warn and use the safe automatic fallback;
- `strict=True`: raise.

A selected accelerator that crashes at runtime currently warns and falls back
to pure Python. `LV-001` tracks the fact that the next safe accelerator is not
tried.

Do not close `LV-001` by adding nested try/except blocks in `rank()`. Build one
per-operation execution plan so `LV-002` (repeated resolution in ranking) is
closed by the same architecture.

## Ranking contract

Tie order is stable:

```text
(-normalized_similarity, distance, original_index)
```

`limit=0` must not consume the input iterable.

`score_cutoff` is a normalized-similarity threshold in `[0, 1]`. It is
currently applied after exact distance computation. Keep docs honest if that
changes.

## Adding an optional accelerator

Before changing `auto`, verify:

- current license;
- import-time behavior;
- supported sequences;
- exact unit-cost semantics;
- empty inputs;
- Unicode;
- runtime errors;
- deterministic conformance against the pure-Python implementation.

An accelerator can be supported explicitly without being selected
automatically.

## Gallery and docs

Examples must:

- be deterministic;
- use local literal data only;
- require no download/network;
- assert the result they demonstrate;
- identify backend availability without importing optional accelerators at
  module import.

Docs should distinguish:

- **availability fallback** from **runtime fallback**;
- **semantic score cutoff** from a backend performance optimization;
- the standalone facade from the optional Corpus bridge.

## Verification rule

A source-reading concern is a hypothesis until a named test or executable probe
demonstrates it. An unrun release lane is `UNVERIFIED`, not PASS.
