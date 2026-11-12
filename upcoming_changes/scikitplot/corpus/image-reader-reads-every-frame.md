---
title: "ImageReader copies and OCRs every frame of an animated image, without a bound"
status: open
kind: "reliability"
area: "scikitplot/corpus"
discovered_during: "PR 928: second research self-test (text, PDF, image and video files through corpus)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "fix"
towncrier_fragment: ""
---

# ImageReader copies and OCRs every frame of an animated image, without a bound

## Summary

`ImageReader._extract_frames` (`scikitplot/corpus/_readers/_image.py`)
copies every frame of a multi-frame image into a list, and each frame is
then OCRed. The reader has no limit on frames, pixels or time
(finding CX-15).

## Why it matters

An 800 x 1111 animated GIF of 280 frames did not finish in 60 s (one frame
takes 0.2 to 0.9 s with Tesseract) and its raw frames take about 237 MiB.
The research set holds 83 GIFs with 11,257 frames (median 147, most 390):
hours of OCR for a folder a user points the builder at. An input can also
be made to exhaust memory.

## Current evidence

- `CorpusPipeline(chunker=ParagraphChunker()).run(gif)` stopped by a 60 s
  limit inside `_ocr_tesseract` (CPython 3.13.18, pytesseract).
- `grep` for a frame limit in `_image.py`: none; `max_file_bytes` bounds the
  file, not the frames it expands to.
- Consecutive frames differ in OCR text (frames 0 and 1 of the sample), so
  dropping repeated frames by text is not a rule that removes the cost.

## Root cause / current understanding

Multi-frame support was written for multi-page TIFF, where every page is
content; an animation has the same shape and far more frames.

## Expected behavior

A decision for the maintainer: (a) a `max_frames` limit (default to be
chosen) that raises a clear error, or yields the first frames and reports
the rest as omitted in the result; (b) frames streamed one at a time
instead of copied into a list, with or without (a); (c) animated formats
read as their first frame only unless asked.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_image.py`
- Owner: `maintenances/corpus/`

## Constraints and non-goals

Multi-page TIFF keeps every page by default. No silent truncation: what is
not read is reported.

## Edge cases to cover

One frame; multi-page TIFF; animated GIF, WebP and PNG; a limit of zero or
one; a frame that fails to decode.

## Proposed direction

(b) first (exact, memory only), then the limit of (a) once its default and
its behaviour at the limit are decided.

## Verification / acceptance criteria

Peak memory of one frame, not of all; the sample GIF bounded by the limit;
TIFF results unchanged.

## Documentation impact

The image reader's parameters in the corpus guide.

## Release-note promotion

Required: `scikitplot.corpus`, type `fix` (or `api` with a new parameter).
