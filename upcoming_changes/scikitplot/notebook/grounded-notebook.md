---
title: "No source-grounded notebook: sources, notes and cited answers in one object"
status: open
kind: "other"
area: "scikitplot/notebook"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# No source-grounded notebook: sources, notes and cited answers in one object

## Summary

The pieces exist separately (readers, retrieval, MCP serving, agents), but nothing combines a chosen set of sources into a notebook that answers only from them, with citations to exact spans, keeps notes, and refuses when the sources do not cover a question.

## Why it matters

Users want to upload or point at sources and ask questions with verifiable answers; today that means wiring several submodules by hand.

## Current evidence

- `grep -rEil 'class Notebook' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A `Notebook` with: sources added from any reader (file, folder, URL, archive), per-source enable or disable, built index, `ask()` returning an answer with span citations and a refusal when coverage is too low, saved notes, and save/load of the whole notebook.

## Affected paths and ownership

- new `scikitplot/notebook` (or within `corpus`)
- `scikitplot/corpus`
- `scikitplot/agent`
- `scikitplot/mcp`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Sources in several languages
- A source removed after an answer cited it
- Very large notebooks (incremental index)
- No model configured (retrieval-only answers)

## Proposed direction

Compose existing parts behind one class; model access through the generation interface.

## Verification / acceptance criteria

- A notebook of three synthetic sources answers with span citations and refuses an off-topic question

## Documentation impact

New user guide section and gallery example.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
