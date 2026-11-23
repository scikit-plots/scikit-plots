---
title: "PDFReader's default backend drops text drawn inside form XObjects (CX-47)"
status: open
kind: "bug"
area: "scikitplot/corpus"
discovered_during: "Real-data snapshot and extraction benchmark of a private mixed research archive (2026-11-23)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# PDFReader's default backend drops text drawn inside form XObjects (CX-47)

## Summary

The pdfminer page loop keeps only top-level `LTTextContainer` elements.
Text that a PDF draws inside a form XObject is laid out by pdfminer as an
`LTFigure` and is never read, with no warning.

## Why it matters

Design and presentation tools often place text in form XObjects. Such a PDF
reads as almost empty with the default backend while pypdf, pypdfium2 and
poppler read it fully. The result is "ok", so nothing tells the user.

## Current evidence

Synthetic reproducer (`main` at 13b457d, pdfminer.six, reportlab to build):

```python
c = canvas.Canvas("form_text.pdf")
c.drawString(72, 760, "Top level line one two three")
c.beginForm("box")
c.drawString(72, 700, "alpha beta gamma delta")
c.endForm()
c.doForm("box")
c.showPage()
c.save()
# default / prefer_backend="pdfminer": ['Top level line one two three']
# prefer_backend="pypdf":             ['Top level line one two three\nalpha beta gamma delta']
```

Real data: one 10-page PDF exported from a design tool. Word recall against
`pdftotext` was 0.28 (304 of 1004 words); per page the reader got 10 to 24
words where pypdf and pypdfium2 got 92 to 182. Walking into `LTFigure`
restored 92 to 174 words per page; with `LAParams(all_texts=True)`, 99 to
184. Of the 16 PDFs in the archive, this was the only one affected.

## Root cause / current understanding

Verified: `_PdfminerPages.text` in `scikitplot/corpus/_readers/_pdf.py`
joins `element.get_text()` for `element in page if isinstance(element,
LTTextContainer)`; `LTFigure` children (`LTChar`, or text boxes when
`all_texts=True`) are skipped. Passing `pdfminer_kwargs={"laparams":
{"all_texts": True}}` alone does not help, because the loop still skips the
figure.

## Expected behavior

Text inside figures is read in reading order with the rest of the page by
the default backend, so the three backends agree within layout differences.

## Affected paths and ownership

`scikitplot/corpus/_readers/_pdf.py` (`_PdfminerPages.text`). Owner:
`maintenances/corpus`.

## Constraints and non-goals

- Keep `page_number` zero-based and page text stable across reads (CX-16).
- Keep the `pdfminer_ties="stable"` ordering for text boxes.
- Do not read image XObjects (that is OCR, a separate setting).

## Edge cases to cover

Nested forms (a figure inside a figure), a form used twice on one page
(text appears twice, as drawn), a figure holding only images, rotated
figures, `all_texts=True` and `False`, a page with only figure text (must
no longer trigger the "probably scanned" warning).

## Proposed direction

Recurse into `LTFigure`: yield text containers found inside, and join runs
of loose `LTChar` into a line; or default `LAParams(all_texts=True)` and
recurse. Prefer the second if its text order matches pypdf on the fixtures.

## Verification / acceptance criteria

- The reproducer yields both lines with the default backend.
- A test fails on the current code (mutation check: drop the recursion).
- Reader snapshot on `main` and the branch: only PDFs with figure text
  change; no new errors or unstable files.

## Documentation impact

`PDFReader` Notes (what each backend reads).

## Release-note promotion

Required: `scikitplot.corpus` / `fix`.
