---
title: "Generation: no safety checks on prompts and outputs"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation: no safety checks on prompts and outputs

## Summary

Requests and outputs are not screened (prompts asking for disallowed content, outputs containing personal data or unsafe material). cleanprompt redacts personal data in text, but nothing applies it to generation prompts or checks generated media.

## Why it matters

A media pipeline run over user-supplied records can produce harmful or privacy-violating assets at scale.

## Current evidence

- `grep -rEil 'moderation' scikitplot/agent --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A policy step before generation (cleanprompt on prompts, a configurable moderation check) and after (personal data in generated text, OCR of generated images, speech-to-text of generated audio), with refusals recorded.

## Affected paths and ownership

- `scikitplot/genai`
- `scikitplot/cleanprompt`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- False positives on medical vocabulary
- Languages the checks do not cover (marked)
- Moderation backend unavailable (fail closed)

## Proposed direction

Reuse cleanprompt; moderation backends behind the provider protocol.

## Verification / acceptance criteria

- A prompt with an email address is redacted before reaching the backend; a stub output with a phone number is flagged

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
