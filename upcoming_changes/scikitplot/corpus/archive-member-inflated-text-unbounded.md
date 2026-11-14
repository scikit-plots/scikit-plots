---
title: "A text member of a ZIP has no text budget: 54 KB yields 24 M characters (CX-41)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 938: hostile-sample gate (test__hostile_samples.py)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# A text member of a ZIP has no text budget: 54 KB yields 24 M characters (CX-41)

## Summary

A ZIP of about 54 KB holding one text member that inflates to 24 MiB is read in full: 24 M characters, more than 20 s (a 147 KB archive of 48 MiB: 50 M characters, 100 s, 240 MiB traced).

## Current evidence

- Gate sample `zip-compressed-member`: over the 20 s ceiling.
- 146,592-byte archive, 48 MiB member: read, 50,331,648 characters, 100.8 s, 239.5 MiB traced peak.
- `max_total_bytes` (2 Gi) is the only bound: a 2 MB archive may yield 2 Gi characters.

## Root cause / current understanding

CX-36 left `max_text_chars` at `None` for text, XML and ALTO because their output cannot outgrow their input. Inside an archive the member's input is the inflated bytes, which deflate makes about 1000 times the archive's size; `ZipReader` itself sets no `max_text_chars` or `max_text_chars_per_byte` default, and the ratio is 'not applied when the input is not a file on disk'.

## Expected behavior

`ZipReader` applies the amplification defaults to the archive as a whole: `max_text_chars` 256 Mi and `max_text_chars_per_byte` against the archive's size on disk (with the 1 Mi floor), as `OfficeReader` and `PDFReader` do. The value of the ratio for archives (real text compresses 3 to 10 times, so 50 is already generous) is the maintainer's to confirm.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_zip.py` (defaults; the archive total in `_iter_raw_chunks`).

## Acceptance

The fix removes the sample's entry from `_OPEN` in
`scikitplot/corpus/_readers/tests/test__hostile_samples.py` (a strict
`xfail`: the gate fails if the entry stays after the fix).
