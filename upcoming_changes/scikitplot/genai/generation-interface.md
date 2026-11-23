---
title: "Generation exists only inside the documentation proxy: no public interface for text, image, audio and video"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation exists only inside the documentation proxy: no public interface for text, image, audio and video

## Summary

The public submodules read every media type (any-to-text) but generate none. The documentation proxy has provider, job and artifact code for audio, video and documents, private to its web app; notebooks, agents, MCP tools and users cannot call it.

## Why it matters

Production media pipelines (personalised assets from records, overviews from notebooks, illustrated reports) need one interface across backends with costs, seeds and provenance.

## Current evidence

- `grep -rEil 'class .*(Generation|Generator)(Request|Backend)' scikitplot --include=*.py` finds nothing in the public submodules (the documentation extensions under `_externals/` and build helpers left out).
- Private base: the documentation proxy `scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/` already has a provider registry (`_providers/`: anthropic, openai, gemini, huggingface, self_hosted, custom, `policy.py`, `registry.py`, `executor.py`), binary output executors (`_providers/artifact_output.py`), asynchronous audio jobs (`_utils/_audio_generation.py`), a video-generation contract with a stub backend (`_utils/_video_generation.py`), document generation (`_utils/_document_generation.py`), telemetry and storage; all of it serves the docs web app only.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A small interface: `GenerationRequest` (modality, prompt or inputs, parameters, seed, budget), `GenerationResult` (artifact, model and version, seed, cost, time, provenance), backends registered by name with capability flags, and a deterministic stub backend for tests.

## Affected paths and ownership

- new `scikitplot/genai`
- `scikitplot/agent` (tools)
- `scikitplot/mcp` (served tools)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Backends that cannot honour a seed (marked non-deterministic)
- Requests over budget (refused before running)
- Partial results (video segments)

## Proposed direction

Extract the proxy's provider protocol, artifact executors and job lifecycle into a public module (the proxy becomes one client of it); then add backends behind optional extras.

## Verification / acceptance criteria

- The stub backend passes a shared conformance test suite every backend must pass; the docs proxy uses the public module

## Documentation impact

New user guide chapter.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
