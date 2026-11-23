---
title: "Notebook: no spoken overview (two-voice script and audio)"
status: open
kind: "other"
area: "scikitplot/notebook"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Notebook: no spoken overview (two-voice script and audio)

## Summary

A spoken overview of a notebook (a two-host conversation script rendered to audio) is a popular way to consume sources; there is no script generator or text-to-speech path.

## Why it matters

Audio overviews make long material accessible while commuting and for users with visual impairments.

## Current evidence

- `grep -rEil 'audio_overview|two_voice' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A script generator (turns with speaker roles and citations), rendering through the text-to-speech interface, chapter markers, and a transcript file next to the audio.

## Affected paths and ownership

- `scikitplot/notebook`
- generation interface (text-to-speech note)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Very long notebooks (segment and summarise)
- Pronunciation of names and terms
- Languages without a voice installed

## Proposed direction

Script first (testable without audio), audio through the TTS backend.

## Verification / acceptance criteria

- A synthetic notebook yields a script whose every claim is cited and an audio file of the expected length

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
