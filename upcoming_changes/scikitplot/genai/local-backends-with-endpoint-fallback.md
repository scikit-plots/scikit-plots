---
title: "Generation backends: local first, endpoint fallback"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation backends: local first, endpoint fallback

## Summary

No public API runs a generation model in process (a diffusion pipeline, a local speech model) or falls back to an OpenAI-compatible endpoint when local resources are missing. The documentation proxy has hosted and self-hosted providers, private to it and server-based.

## Why it matters

Users need privacy and cost control (local) and availability (fallback) without changing code.

## Current evidence

- `grep -rEil 'class .*(Generation|Generator)(Request|Backend)' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).
- Private base: the documentation proxy `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/` already has a provider registry (`_providers/`: anthropic, openai, gemini, huggingface, self_hosted, custom, `policy.py`, `registry.py`, `executor.py`), binary output executors (`_providers/artifact_output.py`), asynchronous audio jobs (`_utils/_audio_generation.py`), a video-generation contract with a stub backend (`_utils/_video_generation.py`), document generation (`_utils/_document_generation.py`), telemetry and storage; all of it serves the docs web app only.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Backends: local pipelines (optional extras per modality), OpenAI-compatible HTTP endpoints (configurable base URL, key from the environment), an ordered fallback chain with reasons recorded per request; the docs assistant's provider code reused or merged, not duplicated.

## Affected paths and ownership

- `scikitplot/genai`
- `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/_providers/` (existing provider layer)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- GPU absent (local backend unavailable)
- Endpoint rate limits
- Different output formats per backend (normalised)

## Proposed direction

Extract a public provider protocol from the private provider layer; implement local backends behind extras.

## Verification / acceptance criteria

- The fallback chain chooses the endpoint when the local backend reports unavailable; the result records why

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
