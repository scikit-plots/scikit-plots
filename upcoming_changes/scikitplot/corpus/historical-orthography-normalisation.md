---
title: "corpus: historical spellings are not normalised for search (long s, old ligatures)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Real-life probe on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: historical spellings are not normalised for search (long s, old ligatures)

## Summary

Text with historical characters keeps them: `Die Geſchichte des Hauſes` is indexed with `ſ`, so a search for `geschichte` does not find it. There is no option to add a normalised search text next to the original.

## Why it matters

ALTO and TEI support made historical archives (Fraktur newspapers, early prints) readable; without normalisation their text is unsearchable by modern spelling.

## Current evidence

- Probe (main cba9383): a UTF-8 file `Die Geſchichte des Hauſes und die Wiſſenſchaft` read by `TextReader`: `'ſ' in text` is True and `'geschichte' in text.lower()` is False.

## Root cause / current understanding

Readers keep characters as encoded (correct for `raw_text`); nothing derives a search form.

## Expected behavior

An optional normaliser (`ſ`→`s`, `ꝛ`→`r`, `æ`→`ae` by language profile, NFKC for ligatures) that writes `text` for search while `raw_text` keeps the original, recorded in metadata.

## Affected paths and ownership

- `scikitplot/corpus/_normalizers/` (or the filter stage)

## Constraints and non-goals

Off by default; reversible (original kept).

## Edge cases to cover

- Greek polytonic
- Latin abbreviations with combining marks
- Long s inside URLs or code

## Proposed direction

Add a `HistoricOrthographyNormalizer` with per-language tables.

## Verification / acceptance criteria

- The probe text is found by `geschichte` with the normaliser on; `raw_text` unchanged

## Documentation impact

none

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
