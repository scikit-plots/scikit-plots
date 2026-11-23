---
title: "Text from images is read by default but not from scanned PDFs, audio or video (CX-52)"
status: open
kind: "api"
area: "scikitplot/corpus"
discovered_during: "Real-data snapshot and media sample of a private mixed research archive (2026-11-23)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "api"
towncrier_fragment: ""
---

# Text from images is read by default but not from scanned PDFs, audio or video (CX-52)

## Summary

Readers that need an engine to get text disagree on the default.
`ImageReader` runs OCR by default; `PDFReader` has `ocr="off"`;
`AudioReader` and `VideoReader` have `transcribe=False`. A photo of a page
gives text, a scan of the same page in a PDF gives none.

## Why it matters

A mixed folder is read inconsistently: what the corpus contains depends on
the container, not the content. Each reader does warn, but a user indexing
a folder sees one warning per file and must learn four settings.

## Current evidence

`main` at 13b457d, tesseract installed, no whisper. Real data: 25 of 25
sampled JPGs gave OCR text (1.7 to 6.3 s each); 4 image-only PDFs gave no
text with a warning to pass `ocr='auto'`; 6 MP4 files and 1 MP3 gave no
chunks in 0.3 s with a warning to set `transcribe=True`.

## Root cause / current understanding

Verified: the defaults are set per reader (`_image.py`, `_pdf.py`
`ocr="off"`, `_audio.py`/`_video.py` `transcribe=False`) with no shared
policy.

## Expected behavior

One documented policy for engine-based text, the same for every reader,
selectable per call and per `CorpusBuilder`: for example
`"off"` / `"auto"` (use an installed engine within the reader's budgets,
warn once when none is installed) / `"force"`.

## Affected paths and ownership

`scikitplot/corpus/_readers/_image.py`, `_pdf.py`, `_audio.py`,
`_video.py`, `scikitplot/corpus/_corpus_builder.py` (passing the policy).
Owner: `maintenances/corpus`.

## Constraints and non-goals

- Changes defaults: a maintainer decision.
- Budgets stay (`ocr_max_pages`, `max_frames`, audio duration limits).
- Do not download models implicitly.

## Edge cases to cover

Engine missing, engine present but language pack missing, a PDF with some
text pages and some scanned pages, a video with a subtitle file, a very
long audio file, a run with no network.

## Proposed direction

Options, recommended first: (a) `"auto"` everywhere: read with an
installed engine, otherwise warn once per run with the install hint;
(b) `"off"` everywhere including images, with one per-run warning that
counts the files left unread; (c) keep per-reader defaults and document
the table.

## Verification / acceptance criteria

The same image saved as PNG and as a one-page PDF gives the same text
under the default policy; tests for engine missing and present.

## Documentation impact

Corpus user guide (one table of engine-based readers and the policy),
each reader's docstring.

## Release-note promotion

Required: `scikitplot.corpus` / `api`.
