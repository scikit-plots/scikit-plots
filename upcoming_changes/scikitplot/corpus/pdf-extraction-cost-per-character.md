---
title: "A 63 KB PDF takes minutes to reach the text limit with pdfminer (CX-40)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 938: hostile-sample gate (test__hostile_samples.py)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# A 63 KB PDF takes minutes to reach the text limit with pdfminer (CX-40)

## Summary

A PDF of 62,853 bytes (one compressed content stream of 40,000 characters shared by 400 pages) is refused at the ratio limit (3.1 M characters), after 217 s.

## Current evidence

- Measured here: 217.5 s, 55 MiB traced, CPython 3.13, pdfminer backend (the default chosen by `DocumentReader.create`).
- 5 pages of the same stream: 16 s under the profiler, all in `pdfminer.pdfinterp.process_page` (about 3 s a page of 40,000 characters).
- That is about 70 us a character; CX-37 recorded about 9 us and accepted 'a file of N bytes may cost what 50 x N characters cost'.
- Ordinary files too (real-data benchmark, 2026-11-23, `main` at 13b457d): 16 PDFs took 27.2 s with the default backend against 3.5 s for `pdftotext` (poppler); the slowest were a 0.8 MB, 30,667-word PDF (7.3 s against 0.2 s) and a 2.2 MB, 123,097-word PDF (8.3 s against 2.5 s). Word recall was equal or better with poppler on 15 of 16 files (the exception is CX-47's figure text). This supports option (c).

## Root cause / current understanding

The ratio limit bounds output, not the cost of producing it, and the cost per character depends on the page: many characters on one page cost more than linear time in the layout analysis. CX-37 accepted the residual at 9 us a character; at 70 us the same decision allows 63 KB -> 3.6 minutes.

## Expected behavior

A decision of the maintainer (options, recommended first): (a) a per-file time budget for extraction (`max_seconds`, checked between pages; refuse or truncate as `on_text_limit` says); (b) a lower default ratio for `PDFReader`; (c) prefer the faster backend when both are installed; (d) keep the CX-37 residual and raise the gate's ceiling for this sample.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_pdf.py` (backend choice, page loop); `_base.py` if a time budget is shared by readers.

## Acceptance

The fix removes the sample's entry from `_OPEN` in
`scikitplot/corpus/_readers/tests/test__hostile_samples.py` (a strict
`xfail`: the gate fails if the entry stays after the fix).
