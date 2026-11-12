---
title: "A text file that is not UTF-8 decodes differently with and without chardet installed"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 932: what each reader does with broken and non-UTF-8 files"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# A text file that is not UTF-8 decodes differently with and without chardet installed

## Summary

`TextReader` with `encoding=None` asks chardet, when it is installed, to
name the codec of a file that is neither marked nor UTF-8. chardet is an
optional, statistical guesser, so the same file gives one text where it is
installed and another where it is not (finding CX-21).

## Why it matters

A corpus built on two machines, or before and after an unrelated install,
holds different text, different chunks and different ids for the same files.
Since PR 932 this concerns only files that are not UTF-8; before it, a UTF-8
file over 8 KiB could be affected too.

## Current evidence

- A KOI8-R file: chardet (when confident) gives Cyrillic; without it the
  reader decodes cp1252 and warns.
- chardet is installed in the system CPython 3.13 here and absent from the
  other interpreters.

## Root cause / current understanding

The detection chain has one step that depends on the environment.

## Expected behavior

The same bytes and reader settings give the same text everywhere.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_text.py` (`_detect_encoding`)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

No new required dependency.

## Edge cases to cover

Single-byte Cyrillic and Greek, Shift-JIS, GB18030, short files where a
guesser has little to go on.

## Proposed direction

A decision between: (a) drop chardet from the default chain and offer it
only when asked, for example `encoding="detect"`; (b) keep it, record the
codec used on every document so a difference can be traced; (c) leave it.

## Verification / acceptance criteria

The test suite gives one text per file with chardet present and absent.

## Documentation impact

`TextReader` docstring.

## Release-note promotion

Required: `scikitplot.corpus`, type `fix`.
