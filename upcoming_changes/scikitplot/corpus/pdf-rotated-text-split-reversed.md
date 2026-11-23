---
title: "PDFReader's default backend splits rotated text into reversed single characters (CX-48)"
status: open
kind: "bug"
area: "scikitplot/corpus"
discovered_during: "Real-data extraction benchmark of a private mixed research archive (2026-11-23)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# PDFReader's default backend splits rotated text into reversed single characters (CX-48)

## Summary

With the pdfminer backend (the default), text drawn at 90 degrees, such as
a chart's vertical axis label, comes out as one character per line in
reverse order. pypdf reads the same text as a word.

## Why it matters

Years, units and labels on charts become unsearchable ("2015" is stored as
`5 1 0 2`) and the page gains many one-character tokens that dilute keyword
scores and embeddings.

## Current evidence

Synthetic reproducer (`main` at 13b457d; reportlab to build):

```python
c.saveState()
c.translate(100, 400)
c.rotate(90)
c.drawString(0, 0, "Year 2015")
c.restoreState()
# pdfminer: 'Horizontal label text\n5\n1\n0\n2\nr\na\ne\nY\n...'
# pypdf:    'Horizontal label text\nYear 2015\n...'
```

Real data: a 6-page report with charts. Against `pdftotext`, the reader's
output had 661 extra word tokens, mostly the digits and single letters of
rotated axis labels, and 159 missing ones, mostly four-digit years
(word precision 0.81).

## Root cause / current understanding

Verified as pdfminer layout behavior with the reader's default `LAParams()`
(`detect_vertical=False`): each rotated glyph becomes its own text line, in
the order of the glyphs' positions. Not yet verified: whether
`detect_vertical=True` or a rotation-aware ordering fixes every angle.

## Expected behavior

Text drawn at 90 and 270 degrees is read as words in writing order; text at
other angles at least keeps each word's characters together.

## Affected paths and ownership

`scikitplot/corpus/_readers/_pdf.py` (pdfminer page text, its `LAParams`).
Owner: `maintenances/corpus`.

## Constraints and non-goals

Do not change horizontal text order or `pdfminer_ties="stable"`. True
vertical scripts (CJK set vertically) must stay readable.

## Edge cases to cover

90, 180, 270 and 45 degree text; a page with both horizontal and rotated
text; CJK vertical writing; ligatures inside rotated text.

## Proposed direction

Measure `detect_vertical=True` and grouping `LTChar` by the text matrix's
rotation on the fixtures; alternatively take rotated runs from the pypdf
backend. Choose by word recall against a reference on the fixtures.

## Verification / acceptance criteria

The reproducer yields `Year 2015` with the default backend; a test fails on
the current code; snapshot shows only PDFs with rotated text changing.

## Documentation impact

`PDFReader` Notes.

## Release-note promotion

Required: `scikitplot.corpus` / `fix`.
