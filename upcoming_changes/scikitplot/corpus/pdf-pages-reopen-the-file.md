---
title: "PDFReader reopens the file and walks its page tree for every page"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 928: second research self-test (text, PDF, image and video files through corpus)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "efficiency"
towncrier_fragment: ""
---

# PDFReader reopens the file and walks its page tree for every page

## Summary

`_extract_page_text_pdfminer` (`scikitplot/corpus/_readers/_pdf.py`) calls
`pdfminer.high_level.extract_pages(path, page_numbers=[n])` once per page.
Each call opens the file again and walks the whole page tree to reach page
`n`, so reading a document costs pages times pages (finding CX-14).

## Why it matters

A 277-page, 2.1 MiB text PDF took 75 s through `CorpusPipeline`; pdfminer
reads the same file in one pass in 7.9 s with the same 277 pages.

## Current evidence

- `cProfile` of `CorpusPipeline(chunker=ParagraphChunker()).run(pdf)`
  (CPython 3.13.18, pdfminer.six as installed): 554 `extract_pages` calls for
  277 pages, 77,284 `create_pages` calls (277 x 279), 257 s of the profiled
  345 s under `PDFPage.get_pages`.
- One page alone, as the reader calls it: 0.28 s for page 10, 100 and 270
  alike (the fixed cost of the walk).
- One pass, `for page in extract_pages(path)`: 277 pages, 7.9 s.

## Root cause / current understanding

The reader's unit of work is one page (auto mode falls back to pypdf per
page), and the pdfminer helper has no state between pages.

## Expected behavior

The same text per page, from one pass over the file per backend; the
per-page fallback to pypdf kept.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_pdf.py`
- Owner: `maintenances/corpus/`

## Constraints and non-goals

Same chunks, page numbers and fallback behaviour; pages still streamed, not
all held in memory; a page pdfminer fails on must not stop the others.

## Edge cases to cover

A page without text (fallback), an encrypted file, a file pdfminer cannot
open, one page, `prefer_backend` of each value, why 554 calls for 277 pages.

## Proposed direction

Keep one `extract_pages` iterator per document and advance it page by page;
compare the text of every page with the per-page calls on several PDFs.

## Verification / acceptance criteria

Per-page text identical to `main` on the test PDFs; the 277-page file in
about the one-pass time; corpus suite unchanged.

## Documentation impact

None.

## Release-note promotion

Required: `scikitplot.corpus`, type `efficiency`.
