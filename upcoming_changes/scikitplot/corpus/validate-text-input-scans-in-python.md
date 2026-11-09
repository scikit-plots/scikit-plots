---
title: "The chunkers' input guard looks for lone surrogates one character at a time in Python"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 920: profile of CorpusBuilder.build over the research corpus after CX-08 (finding CX-11)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "efficiency"
towncrier_fragment: ""
---

# The chunkers' input guard looks for lone surrogates one character at a time in Python

## Summary

`_validate_text_input` (`scikitplot/corpus/_chunkers/_sentence.py`), the
input guard that the `chunk()` of the `Sentence`, `Word`, `Paragraph`,
`FixedWindow`, `Semantic` and `Recursive` chunkers call, checks for NUL with
`in` (C) but then walks the whole document through `enumerate` and `ord` to
find a lone surrogate (U+D800 to U+DFFF).

## Why it matters

After CX-08, profiling `CorpusBuilder(BuilderConfig(chunker="paragraph",
build_index=True)).build(...)` over 64 English text files (3.35 M
characters) gives `_validate_text_input` 0.55 s of own time and 0.84 s in
all for 69 calls (one per document): the third largest own time of the
build, for a check that finds nothing in valid text.

## Current evidence

- Profile above (CPython 3.13.18, PR 920 tree; the evidence log of the
  round names the command).
- The loop compares every code point, so its cost grows with the document,
  and it runs in Python for every character.

## Root cause / current understanding

The NUL check uses `in`; the surrogate check does not have a C-level
equivalent in the code, although `re.search("[\ud800-\udfff]", text)` finds
the same first surrogate (its `start()` is the same code-point index).

## Expected behavior

The same `ValueError`, with the same message (first surrogate, its code
point and its position), raised for exactly the same inputs; the search
done in C.

## Affected paths and ownership

- `scikitplot/corpus/_chunkers/_sentence.py` (`_validate_text_input`)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

No change of which inputs are refused or of any message. The NUL check
stays first.

## Edge cases to cover

A surrogate at position 0 and at the end; two surrogates (the first is
named); a high and a low surrogate; a surrogate pair written as two code
points; NUL and a surrogate together (NUL wins); private-use code points
(U+E000, used by the abbreviation placeholder) are not refused; empty text.

## Proposed direction

A module-level compiled pattern for the surrogate range and one `search`;
an equivalence test against the per-character definition over generated
inputs.

## Verification / acceptance criteria

Equivalence on the generated set; the profile shows the guard no longer
among the largest costs; corpus suite unchanged.

## Documentation impact

None.

## Release-note promotion

Required: `scikitplot.corpus`, type `efficiency`.
