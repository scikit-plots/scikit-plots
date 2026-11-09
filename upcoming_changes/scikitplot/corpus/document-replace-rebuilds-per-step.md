---
title: "CorpusDocument.replace recomputes its field names, and each normalizer step rebuilds the document"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 920: profile of CorpusBuilder.build over the research corpus after CX-08 (finding CX-12)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "efficiency"
towncrier_fragment: ""
---

# CorpusDocument.replace recomputes its field names, and each normalizer step rebuilds the document

## Summary

After CX-08, `CorpusDocument.replace` (`scikitplot/corpus/_schema.py`) is
the largest cumulative cost of `CorpusBuilder.build` over the research
corpus: 2.55 s of a profiled 8.36 s, over 44,264 calls (two normalizer
steps for each of 22,132 documents). Two parts:

1. every call computes `{f.name for f in fields(self)}` to check the
   changed names: 0.62 s of the 2.55 s;
2. every normalizer step returns `doc.replace(normalized_text=...)`, so a
   chain of N steps builds N frozen documents, each through
   `dataclasses.replace` and `__post_init__`: 1.6 s.

## Why it matters

Normalising is on by default in the builder; its cost grows with the number
of steps times the number of chunks.

## Current evidence

`cProfile` of the build above (CPython 3.13.18, PR 920 tree):
`_schema.py:2284(replace)` 44,264 calls, 2.554 s cumulative; its callees
`dataclasses.fields` 0.618 s and `dataclasses.replace` 1.605 s; `copy.copy`
of `metadata` 0.046 s.

## Root cause / current understanding

(1) The field names of a dataclass are fixed when the class is made; they
are recomputed per call. (2) The normalizer protocol is document in,
document out, so each step pays for a full frozen copy even when it changes
only `normalized_text`.

## Expected behavior

(1) The same `ValueError` for an unknown name, from names computed once.
(2) The same normalised documents, with fewer intermediate documents.

## Affected paths and ownership

- `scikitplot/corpus/_schema.py` (`CorpusDocument.replace`)
- `scikitplot/corpus/_normalizers/_normalizer.py` (the chain)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

(1) is exact and local. (2) changes how steps compose and touches a
protocol (a step may read fields an earlier step set, such as `language`):
it needs a design decision before any change.

## Edge cases to cover

(1) Unknown names, every known name, the mutable-copy rule for `metadata`
and the list fields. (2) A step reading `normalized_text` written by the
previous one; a step setting another field; an empty chain.

## Proposed direction

(1) A class-level frozen set of field names. (2) For the maintainer to
decide: a chain that composes text-only steps and builds one document,
or keep one document per step.

## Verification / acceptance criteria

(1) Same errors and documents; the profile without `dataclasses.fields`
under `replace`. (2) Byte-identical normalised output on the research
corpus and the corpus suite unchanged.

## Documentation impact

None for (1); the normalizer guide for (2).

## Release-note promotion

Required: `scikitplot.corpus`, type `efficiency`.
