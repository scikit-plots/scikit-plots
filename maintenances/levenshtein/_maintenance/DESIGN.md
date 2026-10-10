# Design

## Purpose

`scikitplot.levenshtein` is a small, dependency-tolerant facade for unit-cost
Levenshtein edit distance, normalized metrics, deterministic ranking and an
optional Corpus retrieval adapter.

It is **not** a clone of RapidFuzz or the external `Levenshtein` project.

## Public surface

The canonical public names are exported by `_core.__all__`:

- `BackendInfo`
- `Match`
- `available_backends`
- `backend_info`
- `closest`
- `distance`
- `make_corpus_scorer`
- `normalized_distance`
- `normalized_similarity`
- `rank`
- `similarity`

## Backend policy

### Automatic

```text
internal bundled Cython
        ↓ unavailable
RapidFuzz (MIT)
        ↓ unavailable
pure Python (BSD-3-Clause)
```

### Explicit only

The external `Levenshtein` backend is GPL-2.0-or-later and therefore is not
eligible for automatic selection.

Supporting a backend and automatically selecting it are different decisions.

## Failure policy

Availability failure for an explicitly named backend:

- `strict=False`: warning + safe automatic fallback;
- `strict=True`: `ImportError`.

Runtime failure for a selected accelerator:

- `strict=False`: warning + pure Python fallback;
- `strict=True`: re-raise.

Open finding `LV-001` tracks the difference between this runtime path and the
full safe preference order.

## Ranking policy

Normalized similarity is calculated after exact unit-cost distance. Ranking is:

```text
(-similarity, distance, original_index)
```

This makes ties stable and deterministic.

`score_cutoff` is defined in normalized similarity space `[0, 1]`.
It currently filters after exact distance calculation. This is a semantic
contract, not a performance promise.

## Input policy

The facade accepts strings, bytes and sequence-like values. The pure-Python and
bundled internal implementations support general sequences. Optional external
backends can impose narrower runtime type constraints; fail-soft mode can then
fall back to Python.

No Unicode normalization or case folding is implicit. Users should preprocess
through `key=` for ranking when they want such semantics.

## Corpus boundary

`make_corpus_scorer` imports `scikitplot.corpus` only when the returned scorer
is called. Levenshtein stays independently importable.

## Non-goals

- Damerau-Levenshtein/transpositions;
- weighted edit operations in the public facade;
- tokenization/normalization hidden inside distance;
- approximate median strings;
- a broad clone of `rapidfuzz.process`;
- automatic GPL backend selection.
