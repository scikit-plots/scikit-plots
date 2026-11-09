---
title: "Each normalizer step rebuilds the frozen document"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 920: profile of CorpusBuilder.build over the research corpus after CX-08 (finding CX-12, part 2; part 1 fixed in PR 922)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "efficiency"
towncrier_fragment: ""
---

# Each normalizer step rebuilds the frozen document

## Summary

Every normalizer step returns `doc.replace(normalized_text=...)`, so a chain
of N steps builds N frozen `CorpusDocument`s, each through
`dataclasses.replace` and `__post_init__`. Part 1 of CX-12 (field names
recomputed on every `replace`) was fixed in PR 922.

## Why it matters

Normalising is on by default in the builder; the cost grows with the number
of steps times the number of chunks.

## Current evidence

`cProfile` of `CorpusBuilder(BuilderConfig(chunker="paragraph",
build_index=True)).build(...)` over 64 English text files (CPython 3.13.18,
PR 920 tree): 44,264 `replace` calls (two steps for each of 22,132
documents), of which `dataclasses.replace` took 1.6 s of a profiled 8.4 s.

## Root cause / current understanding

The normalizer protocol is document in, document out, so each step pays for
a full frozen copy even when it changes only `normalized_text`.

## Expected behavior

The same normalised documents, with fewer intermediate documents.

## Affected paths and ownership

- `scikitplot/corpus/_normalizers/_normalizer.py` (the chain)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

It changes how steps compose and touches a protocol (a step may read fields
an earlier step set, such as `language`): decision D14 in
`maintenances/corpus/todo/todo.md` comes first.

## Edge cases to cover

A step reading `normalized_text` written by the previous one; a step setting
another field; an empty chain; a custom step.

## Proposed direction

With D14 (a): compose text-only steps and build one document at the end; a
step that sets another field keeps its own document.

## Verification / acceptance criteria

Byte-identical normalised output on the research corpus; the corpus suite
unchanged; fewer `replace` calls in the profile.

## Documentation impact

The normalizer guide.

## Release-note promotion

Required: `scikitplot.corpus`, type `efficiency`.
