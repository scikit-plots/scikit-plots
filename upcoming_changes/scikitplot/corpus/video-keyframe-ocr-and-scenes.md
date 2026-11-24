---
title: "corpus: video text on screen is not read (no keyframe OCR)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: video text on screen is not read (no keyframe OCR)

## Summary

`VideoReader` reads subtitles or a transcript. Text shown on screen (slides, captions burned in, code) is not read; there is no scene detection to pick frames.

## Why it matters

Lecture and tutorial videos carry most of their content on slides.

## Current evidence

- `grep -rEil 'keyframe|scene_detect' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Optional scene detection (frame difference) and OCR of one frame per scene, with timestamps, merged with the transcript by time; frame redundancy handled as in CX-51.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_video.py`
- `scikitplot/corpus/_readers/_image.py`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Static slides for minutes (one OCR)
- Fast cuts
- Very long videos (budget)

## Proposed direction

Reuse the image OCR path and the CX-51 frame deduplication.

## Verification / acceptance criteria

- A synthetic video with two slides yields two OCR chunks with timestamps

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
