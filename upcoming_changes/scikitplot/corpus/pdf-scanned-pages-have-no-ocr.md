---
title: "A PDF of scanned pages gives no text: PDFReader has no OCR path"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "PR 931: what PDFReader reports for files it gets nothing from"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# A PDF of scanned pages gives no text: PDFReader has no OCR path

## Summary

`PDFReader` extracts the text layer of a PDF. A file whose pages are images
(a scan, a slide deck exported as pictures) has none, so it gives no
document. `ImageReader` runs OCR, but only on image files (finding CX-19).

## Why it matters

In the second research set, 3 of 8 PDFs gave no usable text: two gave no
page with text (51 and 27 pages) and one gave 6 characters per page. Since
PR 931 the reader says so in a warning; the content is still not ingested.

## Current evidence

- `PDFReader` on the 51-page and 27-page files: 0 chunks, the warning "none
  of the N page(s) read ... has extractable text".
- The 9-page file: 8 chunks of 6 characters each (page labels), so it is not
  detected as a scan.

## Root cause / current understanding

Rendering a PDF page to an image needs a renderer that neither pdfminer.six
nor pypdf provides; the corpus extra has none.

## Expected behavior

A way to get text from scanned pages through the OCR backends `ImageReader`
already has, chosen explicitly, with the page number kept.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_pdf.py`, `scikitplot/corpus/_readers/_image.py`
- Owner: `maintenances/corpus/`

## Constraints and non-goals

No new required dependency. OCR is slow: it must be opt-in and bounded, as
`ImageReader.max_frames` is. Options for the renderer follow the rule
"options for a named backend" in `maintenances/MAINTAINING.md`.

## Edge cases to cover

A file that mixes text pages and scanned pages; a page whose only text is a
page label; very large pages; an encrypted scan.

## Proposed direction

A decision between: (a) OCR the images embedded in a page (pypdf can list
them; no renderer needed, but a page built from several images or vector
drawings is read in pieces); (b) an optional renderer dependency behind a
`<renderer>_kwargs` parameter, pages rendered and passed to the OCR backend;
(c) leave it to the caller and document how to convert a scan to images for
`ImageReader`.

## Verification / acceptance criteria

The two scanned research PDFs give text per page with the chosen option on;
with it off, behaviour is unchanged.

## Documentation impact

`PDFReader` docstring and the corpus README's reader table.

## Release-note promotion

Required: `scikitplot.corpus`, type `feature`.
