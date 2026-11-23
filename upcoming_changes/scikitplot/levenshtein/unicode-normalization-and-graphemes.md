---
title: "levenshtein: composed and decomposed accents differ by 2; distances count code points"
status: open
kind: "other"
area: "scikitplot/levenshtein"
discovered_during: "Real-life probe on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.levenshtein"
towncrier_type: "enhancement"
towncrier_fragment: ""
---

# levenshtein: composed and decomposed accents differ by 2; distances count code points

## Summary

`distance('café', 'cafe\u0301')` is 2 although both render as `café`; an emoji family sequence and a single emoji are 4 apart. Distances count code points; there is no normalisation or grapheme option.

## Why it matters

Fuzzy matching of names and words from different sources (macOS file names are NFD, most text NFC) reports false differences; deduplication misses true duplicates.

## Current evidence

- Probe (main cba9383): `scikitplot.levenshtein.distance('café', 'cafe\u0301') == 2`; `distance('👨‍👩‍👧', '👨') == 4`.

## Root cause / current understanding

Inputs are compared as given.

## Expected behavior

`normalize=` (`None`, `'NFC'`, `'NFKC'`, `'casefold'`) and `units=` (`'codepoint'`, `'grapheme'`) options, defaults unchanged, applied identically by every backend.

## Affected paths and ownership

- `scikitplot/levenshtein/`
- `corpus.LevenshteinRanker`

## Constraints and non-goals

Backends must agree byte for byte (the facade's contract).

## Edge cases to cover

- Hangul syllables vs jamo
- Regional-indicator flags
- Zero-width joiner sequences
- Very long strings (grapheme segmentation cost)

## Proposed direction

Normalise in the facade before dispatch; graphemes via a small segmenter or an optional dependency.

## Verification / acceptance criteria

- `distance('café','cafe\u0301', normalize='NFC') == 0`; every backend agrees

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.levenshtein` / `enhancement`, one bullet, when the change lands.
