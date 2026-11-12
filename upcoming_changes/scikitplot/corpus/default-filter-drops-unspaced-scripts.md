---
title: "DefaultFilter drops text written without spaces between words"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 932: what each reader does with broken and non-UTF-8 files"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# DefaultFilter drops text written without spaces between words

## Summary

`DefaultFilter` drops a document whose whitespace-separated token count is
below `min_words` (3). Chinese, Japanese and Thai are written without spaces
between words, so a whole document in them counts as one token and is
dropped (finding CX-22).

## Why it matters

A 14,400-byte Chinese text file gives one raw chunk and zero documents; a
ZIP member in Chinese disappears the same way. Nothing says why except the
omitted count.

## Current evidence

- `TextReader` on 4,800 Chinese characters: 1 raw chunk, 0 documents, on
  `main` and on the PR 932 branch.
- `ZipReader` over a member `zh.md`: no document for it.

## Root cause / current understanding

`min_words` counts `str.split()` tokens, which is a word count only for
scripts that separate words with spaces.

## Expected behavior

A substantial document in any script passes the default filter.

## Affected paths and ownership

- `scikitplot/corpus/_base.py` (`DefaultFilter`)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

The filter stays deterministic, with no word segmenter.

## Edge cases to cover

Mixed Latin and CJK, Japanese with kana, Thai, Korean (which uses spaces),
emoji-only text, short CJK headings.

## Proposed direction

A decision between: (a) count each character of a script without word
spaces (Han, Hiragana, Katakana, Thai, Lao, Khmer, Myanmar) as a token;
(b) pass a document that meets `min_chars` regardless of `min_words` when
most of its letters are in such a script; (c) document `min_words=0` for
such corpora.

## Verification / acceptance criteria

The Chinese test file gives documents; English results unchanged.

## Documentation impact

`DefaultFilter` docstring.

## Release-note promotion

Required: `scikitplot.corpus`, type `fix`.
