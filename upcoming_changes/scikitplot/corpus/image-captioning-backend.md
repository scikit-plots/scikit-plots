---
title: "corpus: images without text yield nothing (no captioning backend)"
status: open
kind: "other"
area: "scikitplot/corpus"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.corpus"
towncrier_type: "feature"
towncrier_fragment: ""
---

# corpus: images without text yield nothing (no captioning backend)

## Summary

`ImageReader` reads text in images (OCR). A photo, chart or diagram without text yields no chunk: 3 of 8 sampled PNGs gave 0 chunks.

## Why it matters

Figures, charts and photos are invisible to retrieval and to a source-grounded notebook; multimodal answers need at least a description.

## Current evidence

- Media sample on a private archive (main 13b457d): 3 of 8 PNG files gave 0 chunks with OCR on.
- `python -m scikitplot api scikitplot.corpus` lists no public name matching `caption`.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

An optional captioning backend (a local vision-language model or an OpenAI-compatible endpoint) producing a description chunk marked as generated (`source_type`, model, confidence), never mixed with OCR text.

## Affected paths and ownership

- `scikitplot/corpus/_readers/_image.py`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Images with both text and content (OCR and caption both kept, labelled)
- Large images (resized deterministically)
- Model unavailable offline (warning, no chunk)

## Proposed direction

Backend interface shared with the generation notes; disabled by default.

## Verification / acceptance criteria

- A test with a stub backend yields one labelled caption chunk per image

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.corpus` / `feature`, one bullet, when the change lands.
