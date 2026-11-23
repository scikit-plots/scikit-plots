---
title: "ImageReader OCRs every frame of an animated image and emits near-duplicate chunks (CX-51)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Real-data media sample of a private mixed research archive (2026-11-23)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "efficiency"
towncrier_fragment: ""
---

# ImageReader OCRs every frame of an animated image and emits near-duplicate chunks (CX-51)

## Summary

For an animated GIF, `ImageReader` runs OCR on each of the first 64 frames
(`max_frames=64`) and yields one chunk per frame. Consecutive frames of an
animation usually show mostly the same text, so most of the time and most
of the chunks repeat.

## Why it matters

Time: about 1.5 s a frame, 94 to 116 s for one 800x1000 GIF; a folder of
127 such GIFs would take about 4 hours. Quality: dozens of overlapping
chunks per image crowd out other sources in a search.

## Current evidence

`main` at 13b457d, tesseract 5, CPython 3.13, real data:

- One GIF, 80 frames: the first 64 frames all differ by bytes, 64 chunks,
  94.2 s.
- Same GIF, 16 frames: 2,197 OCR words but only 408 distinct ones (5.4x
  repetition); word overlap (Jaccard) between consecutive frames 0.44 to
  0.77.
- Two GIFs in the media sample: 64 chunks each, 108 s and 116 s.

## Root cause / current understanding

Verified: frames are selected by count only (`max_frames`), each frame is
OCR'd independently and becomes its own chunk. There is no check whether a
frame's content changed, and no merge of overlapping frame texts.

## Expected behavior

Cost grows with the number of distinct visual states, not with the frame
count, and the output holds each distinct text once (with the frames it
was seen on), under explicit and documented settings.

## Affected paths and ownership

`scikitplot/corpus/_readers/_image.py` (frame loop, `max_frames`). Owner:
`maintenances/corpus`.

## Constraints and non-goals

- Deterministic: the same file gives the same frames and chunks on every
  read (no time-based sampling).
- Keep `max_frames` as the hard ceiling.
- Video frames and subtitles are out of scope.

## Edge cases to cover

A GIF whose frames are identical (one OCR call), a slide-show GIF (each
slide distinct), a fade between two slides, frame 0 blank, a 1-frame GIF,
animated WebP and APNG, partial-frame GIFs (disposal methods: compose the
frame before comparing).

## Proposed direction

(a) Skip a frame whose composed image matches the last OCR'd frame within a
small, documented difference (a downscaled grayscale difference or a
perceptual hash); (b) merge consecutive chunks whose word overlap passes a
threshold into one chunk with `frame_range`; (c) expose both as
parameters with defaults chosen on fixtures. Measure OCR calls, time and
distinct-word recall against OCR of every frame.

## Verification / acceptance criteria

On a fixture GIF: at least the same distinct words, fewer OCR calls, the
same output on two reads; a test that fails on the current code; media
snapshot time before and after.

## Documentation impact

`ImageReader` docstring (frames, new parameters).

## Release-note promotion

Required: `scikitplot.corpus` / `efficiency`.
