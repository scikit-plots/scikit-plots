---
title: "rank_bm25: no Unicode case folding; a str.lower tokenizer breaks Turkish İ"
status: open
kind: "bug"
area: "scikitplot/rank_bm25"
discovered_during: "Real-life probe on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.rank_bm25"
towncrier_type: "fix"
towncrier_fragment: ""
---

# rank_bm25: no Unicode case folding; a str.lower tokenizer breaks Turkish İ

## Summary

`BM25Okapi` scores pre-tokenised corpora case-sensitively, so `istanbul` never matches `İstanbul`; the common `tokenizer=lambda s: s.lower().split()` turns `İstanbul` into `i̇stanbul` (an i followed by U+0307), which still does not match.

## Why it matters

Keyword retrieval in Turkish and Azerbaijani silently misses documents; `str.lower` also leaves German `ß` and Greek final sigma unfolded.

## Current evidence

- Probe (main cba9383): corpus `['İstanbul büyük bir şehir', 'Ankara başkent', 'ISTANBUL traffic']`, query `['istanbul']`: scores `[0, 0, 0]` (pre-split) and `[0, 0, 0.576]` with `str.lower`.
- `'İstanbul'.lower()` is `'i̇stanbul'`.

## Root cause / current understanding

Tokenisation is left to the caller; the documented examples lower-case with `str.lower`.

## Expected behavior

A default analyzer that applies NFKC and `casefold`, with language-aware folding (`tr`, `az`: `İ`→`i`, `I`→`ı`) when `Analyzer(language=...)` says so; documented examples use it.

## Affected paths and ownership

- `scikitplot/rank_bm25/`
- the corpus keyword index that builds on it

## Constraints and non-goals

Scores for already-folded ASCII corpora unchanged.

## Edge cases to cover

- Turkish dotted and dotless i
- German ß/ss
- Greek final sigma
- Full-width Latin
- Combining accents (NFC vs NFD)

## Proposed direction

Implement folding in `Analyzer`, use it when no tokenizer is given, and add the examples above as tests.

## Verification / acceptance criteria

- The probe query scores the two Istanbul documents above zero
- Existing BM25 tests unchanged

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.rank_bm25` / `fix`, one bullet, when the change lands.
