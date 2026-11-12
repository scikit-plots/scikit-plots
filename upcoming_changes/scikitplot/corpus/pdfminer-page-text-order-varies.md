---
title: "PDFReader's default read of a PDF page can differ from one read to the next"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 930: comparing every page of the test PDFs before and after the one-pass read"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# PDFReader's default read of a PDF page can differ from one read to the next

## Summary

`PDFReader` asks pdfminer.six for a page's layout with pdfminer's default
layout parameters unless `laparams` is given. For some pages the order of
the text boxes, and so the page's text, changes between two identical reads
(finding CX-16). PR 930 added `laparams`; `{"boxes_flow": None}` gives the
same text on every read. What is left is the default.

## Why it matters

The same file can give a different `text` for a page on each ingestion, so a
content hash, a document id derived from the text, a chunk boundary or a
stored embedding is not reproducible for that page.

## Current evidence

- 8 PDFs, 457 pages, 4 reads each with pdfminer.six 20260107 on CPython
  3.13.18: 7 files had no unstable page. A 277-page text PDF had 2 to 4
  (pages 16, 18, 36 and 96 over two sets of reads): figures with many small
  labels of the same size. The characters are the same, their order is not.
- The same with `PYTHONHASHSEED=0`.
- With `laparams={"boxes_flow": None}`: no unstable page in any file.

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

A decision between: (b) read with `boxes_flow=None` by default (boxes
ordered by position: reproducible, but pdfminer no longer groups boxes into
columns, so the reading order of multi-column pages changes); (c) keep
pdfminer's grouping and order boxes that tie by position before joining
them, if that can be shown to remove every difference; (d) leave the default
and rely on the documented `laparams` setting.

## Verification / acceptance criteria

Repeated reads of the test PDFs give identical text for every page; the
reading order on a multi-column sample is compared before and after.

## Documentation impact

`PDFReader` docstring, if the default changes.

## Release-note promotion

Required: `scikitplot.corpus`, type `fix`.
