---
title: "corpus: TextReader silently decodes legacy encodings as Latin-1 (CX-54)"
status: open
kind: "bug"
area: "scikitplot/corpus"
discovered_during: "Real-life probe on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# corpus: TextReader silently decodes legacy encodings as Latin-1 (CX-54)

## Summary

`TextReader` reads files in cp1254 (Turkish), Shift-JIS and GB18030 as mojibake without a warning; only UTF-8 and BOM-marked UTF-16 come out right.

## Why it matters

Legacy encodings are common in government, academic and East Asian archives; mojibake is indexed and embedded as if it were text, and searches never match.

## Current evidence

- Probe (main cba9383): Turkish cp1254 `Iğdır'da şöyle bir çay içtik` reads as `Iðdýr'da þöyle bir çay içtik`.
- Shift-JIS `日本語のテキスト...` reads as `\x93ú\x96{\x8cê...`; GB18030 `这是一个中文...` reads as `ÕâÊÇÒ»¸ö...`.
- UTF-16 with BOM reads correctly.

## Root cause / current understanding

When strict UTF-8 fails, the reader falls back to a single-byte decoding instead of detecting the encoding or reporting the failure.

## Expected behavior

Encoding chosen by: an explicit `encoding=` argument, a BOM, strict UTF-8, then a detector (optional `charset-normalizer`) with a confidence threshold; below it, a warning naming the file and the chosen encoding.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_text.py`

## Constraints and non-goals

Same text for every UTF-8 file; detection deterministic.

## Edge cases to cover

- Short files (detection unreliable)
- Mixed encodings in one file
- cp1252 vs Latin-1 vs cp1254 (Turkish letters)
- EUC-KR, Big5, KOI8-R

## Proposed direction

Add a detection step and an `encoding` parameter; record the encoding in chunk metadata (already present as `encoding`).

## Verification / acceptance criteria

- The four probe files read correctly with the detector installed; without it a warning names the fallback
- Reader snapshot: no change for UTF-8 files

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.corpus` / `fix`, one bullet, when the change lands.
