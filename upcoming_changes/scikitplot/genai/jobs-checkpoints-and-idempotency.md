---
title: "Generation: no job model with checkpoints, resume and idempotency keys"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation: no job model with checkpoints, resume and idempotency keys

## Summary

Long generation runs (one asset per record, thousands of records) need to survive crashes: a killed run must resume without redoing finished items, and a repeated request must not generate twice.

## Why it matters

Without per-item checkpoints, a failure at item 900 of 1000 repeats 900 paid generations.

## Current evidence

- `grep -rEil 'idempotency_key' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).
- Already public in `scikitplot.agent`: `ArtifactStore`, `DirectoryArtifactStore`, `MemoryArtifactStore`, `StateStore` (the base to build on).
- Private base: the documentation proxy `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/` already has a provider registry (`_providers/`: anthropic, openai, gemini, huggingface, self_hosted, custom, `policy.py`, `registry.py`, `executor.py`), binary output executors (`_providers/artifact_output.py`), asynchronous audio jobs (`_utils/_audio_generation.py`), a video-generation contract with a stub backend (`_utils/_video_generation.py`), document generation (`_utils/_document_generation.py`), telemetry and storage; all of it serves the docs web app only.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Jobs with per-item state in a store (the agent's artifact stores are a base), idempotency keys derived from the request, resume that skips finished items, and a final manifest; retries with backoff for transient errors only.

## Affected paths and ownership

- `scikitplot/genai`
- `scikitplot/agent` (state and artifact stores)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Process killed mid-write (atomic writes)
- Request changed between runs (new key)
- Non-deterministic backends (result kept, not regenerated)

## Proposed direction

Reuse `FileStateStore`/`FileArtifactStore` semantics; keys are hashes of the canonical request.

## Verification / acceptance criteria

- A run killed at half resumes and generates only the missing half; the manifest equals an uninterrupted run

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
