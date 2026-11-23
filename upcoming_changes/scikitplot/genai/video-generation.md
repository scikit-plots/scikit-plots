---
title: "Generation: no video generation and editing backend"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation: no video generation and editing backend

## Summary

Short videos cannot be generated or assembled (from images and narration, or from a prompt).

## Why it matters

The read side covers the modality; without the write side, notebooks, agents and media pipelines stop at text.

## Current evidence

- `grep -rEil 'text_to_video|video_generation' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).
- Private base: the documentation proxy `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/` already has a provider registry (`_providers/`: anthropic, openai, gemini, huggingface, self_hosted, custom, `policy.py`, `registry.py`, `executor.py`), binary output executors (`_providers/artifact_output.py`), asynchronous audio jobs (`_utils/_audio_generation.py`), a video-generation contract with a stub backend (`_utils/_video_generation.py`), document generation (`_utils/_document_generation.py`), telemetry and storage; all of it serves the docs web app only.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A video backend for prompt-to-video and an assembly path (images plus audio plus subtitles via ffmpeg), with duration and resolution budgets.

## Affected paths and ownership

- `scikitplot/genai`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- GPU memory limits (fit or refuse)
- Long durations (segments)
- Audio and video drift

## Proposed direction

Implement on the generation interface with a stub backend for tests and one optional real backend.

## Verification / acceptance criteria

- The stub passes the conformance tests; the optional backend produces an artifact with provenance and cost

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
