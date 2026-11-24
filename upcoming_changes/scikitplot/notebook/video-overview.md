---
title: "Notebook: no narrated slide overview (video)"
status: open
kind: "other"
area: "scikitplot/notebook"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Notebook: no narrated slide overview (video)

## Summary

A short narrated video (slides from the outline with narration) summarises a notebook visually; nothing produces slides or video here.

## Why it matters

Training and onboarding material is consumed as video; producing it by hand repeats work the notebook already did.

## Current evidence

- `grep -rEil 'video_overview' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Slides generated from the outline (with figures from sources), narration through TTS, assembly with `ffmpeg`, subtitles from the script.

## Affected paths and ownership

- `scikitplot/notebook`
- generation interface

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Figures with licences that forbid reuse
- Long narration per slide
- No ffmpeg installed

## Proposed direction

Compose slides, TTS and ffmpeg; keep each step's artifact.

## Verification / acceptance criteria

- A synthetic notebook yields a video with subtitles matching the script

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
