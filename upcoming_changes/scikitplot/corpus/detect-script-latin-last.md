---
title: "detect_script tests Latin last, per character: 30% of CorpusBuilder.build on English text"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 919: research self-test of corpus and mcp (finding CX-08)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "efficiency"
towncrier_fragment: ""
---

# detect_script tests Latin last, per character: 30% of CorpusBuilder.build on English text

## Summary

`detect_script` (`scikitplot/corpus/_chunkers/_custom_tokenizer.py`) walks
the first `sample_size` characters of every chunk through about 25 code-point
range tests, and the Latin test is the last branch. Every ASCII letter
therefore runs every comparison before it is counted.

## Why it matters

Measured on 64 English markdown/text files (3.35 M characters, 22,034
paragraph chunks), `CorpusBuilder(BuilderConfig(chunker="paragraph",
build_index=True)).build(...)` under `cProfile`: `detect_script` spent
3.57 s of the build's 11.94 s (24,609 calls, 6.27 M `ord` calls), the
largest single cost.

## Current evidence

- Profile above (CPython 3.13, `main` at `6734def`).
- A prototype ASCII fast path returned the same `ScriptType` as the current
  function on 168,176 cases (22,034 real chunks, 20,000 random strings over
  U+0000 to U+2FFFF without surrogates, 10 edge cases, each at
  `majority_threshold` 0, 0.55, 1.0 and 1.01): 0 mismatches; 3.29 s to
  0.47 s over the 22,034 chunks.

## Root cause / current understanding

In an ASCII sample only `A-Z` and `a-z` are counted, and only as Latin, so
the result is decided by one question: does the sample hold an ASCII letter?
The per-character chain answers it the slowest way.

## Expected behavior

Identical results for every input and every `majority_threshold`, decided
for an ASCII sample in C (`str.isascii`, one regular-expression search):
`UNKNOWN` with no letter, otherwise `LATIN` when `1.0 >= majority_threshold`
and `MIXED` when not.

## Affected paths and ownership

- `scikitplot/corpus/_chunkers/_custom_tokenizer.py` (`detect_script`)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

No change of any result; the non-ASCII branches stay as they are, in the
same order.

## Edge cases to cover

Empty text; no letters (`"123 !@#"`); one letter after 499 non-letters and
after 500 (the sample boundary); ASCII followed by non-ASCII inside and
outside the sample; thresholds 0, 0.55, 1.0 and above 1.

## Proposed direction

A fast path at the top of `detect_script` for `sample.isascii()`; the
existing chain unchanged for every other sample. An equivalence test over
generated inputs against the chain as the oracle.

## Verification / acceptance criteria

Equivalence on the generated set (0 mismatches); the profile above shows
`detect_script` no longer the largest cost; corpus suite unchanged.

## Documentation impact

None.

## Release-note promotion

Required: `scikitplot.corpus`, type `efficiency`.
