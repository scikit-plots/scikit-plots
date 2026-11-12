---
title: "The text of a PDF page from pdfminer can differ from one read to the next"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 930: comparing every page of the test PDFs before and after the one-pass read"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# The text of a PDF page from pdfminer can differ from one read to the next

## Summary

`PDFReader` asks pdfminer.six for a page's layout with the default layout
parameters. For some pages the order of the text boxes, and so the page's
text, changes between two identical calls in one process (finding CX-16).

## Why it matters

The same file can give a different `text` for a page on each ingestion, so a
content hash, a document id derived from the text, a chunk boundary or a
stored embedding is not reproducible for that page.

## Current evidence

- A 277-page text PDF, page 36 (a figure with many small labels of the same
  size): 6 identical `extract_pages(path, page_numbers=[36])` calls in one
  process gave 4 different texts; the characters are the same, their order
  is not. The other 276 pages were identical on every read.
- pdfminer.six 20260107, CPython 3.13.18; the same with `PYTHONHASHSEED=0`.
- With `LAParams(boxes_flow=None)` the 6 calls gave 1 text.

## Root cause / current understanding

`LTLayoutContainer.group_textboxes` keeps candidate pairs in a heap of
`(idx, dist, id(obj1), id(obj2), obj1, obj2)`. Pairs at the same distance
are ordered by `id()`, a memory address, which differs from call to call.

## Expected behavior

The same file and reader settings give the same text for every page on
every read.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_pdf.py` (`_PdfminerPages`)
- Owner: `maintenances/corpus/`

## Constraints and non-goals

Not a change of pdfminer itself. The reading order of ordinary pages should
not get worse.

## Edge cases to cover

Multi-column pages, tables, figures with repeated labels, a page with one
text box.

## Proposed direction

A decision between: (a) a reader field for the layout parameters, default
unchanged, documented as the way to a reproducible read; (b) read with
`boxes_flow=None` by default (boxes ordered by position: reproducible, but
the reading order of multi-column pages changes); (c) keep pdfminer's
grouping and order boxes that tie by their position before joining them, if
that can be shown to remove every difference.

## Verification / acceptance criteria

Repeated reads of the test PDFs give identical text for every page; the
reading order on a multi-column sample is compared before and after.

## Documentation impact

`PDFReader` docstring, if a field is added or a default changes.

## Release-note promotion

Required: `scikitplot.corpus`, type `fix`.
