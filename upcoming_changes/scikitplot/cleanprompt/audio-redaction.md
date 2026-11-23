---
title: "cleanprompt: personal data in audio cannot be redacted"
status: open
kind: "other"
area: "scikitplot/cleanprompt"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.cleanprompt"
towncrier_type: "feature"
towncrier_fragment: ""
---

# cleanprompt: personal data in audio cannot be redacted

## Summary

Spoken personal data (names, numbers) in recordings is not redacted; transcripts can be, but the audio keeps it.

## Why it matters

Call recordings and interviews must be shared without personal data.

## Current evidence

- `grep -rEil 'bleep|redact_audio' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Transcription with word times, detectors on the transcript, and the matching time ranges silenced or replaced by a tone, with a report.

## Affected paths and ownership

- `scikitplot/cleanprompt`
- `scikitplot/corpus/_readers/_audio.py`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Word times missing (segment level)
- Spelled-out numbers
- Several speakers

## Proposed direction

Reuse the audio transcription with word timestamps.

## Verification / acceptance criteria

- A synthetic recording with a spoken phone number has that range silenced

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.cleanprompt` / `feature`, one bullet, when the change lands.
