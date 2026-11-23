---
title: "corpus: transcripts carry no speaker turns"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: transcripts carry no speaker turns

## Summary

Audio and video transcription produces text without who spoke when.

## Why it matters

Interviews, meetings and podcasts are only useful with speaker turns; citations need the speaker.

## Current evidence

- `grep -rEil 'diariz' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Optional diarisation backend producing speaker labels per segment, merged with transcript segments by time, labels anonymised (`SPEAKER_1`) unless names are supplied.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_audio.py`
- `scikitplot/corpus/_readers/_video.py`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Overlapping speech
- One speaker only
- Long recordings (chunked)

## Proposed direction

Backend interface; off by default.

## Verification / acceptance criteria

- A two-speaker synthetic recording yields segments with two labels

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
