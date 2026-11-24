---
title: "Notebook: no generated study outputs (briefing, FAQ, timeline, study guide, quiz, flashcards, mind map)"
status: open
kind: "other"
area: "scikitplot/notebook"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Notebook: no generated study outputs (briefing, FAQ, timeline, study guide, quiz, flashcards, mind map)

## Summary

Turning sources into a briefing document, an FAQ, a timeline, a study guide, quiz questions, flashcards or a mind map is a common use of grounded notebooks; nothing produces these here.

## Why it matters

Each output is a structured transformation of the same cited material; without it users copy answers by hand.

## Current evidence

- `grep -rEil 'flashcard|study_guide|mind_map' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Generators that return structured data (each item with citations), a Markdown rendering, and exporters (CSV for flashcards, JSON for quizzes, a graph for mind maps), all grounded and refusing uncovered items.

## Affected paths and ownership

- `scikitplot/notebook`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Sources too short for the requested size
- Contradicting sources (both cited)
- Language of output differing from sources

## Proposed direction

Templates as data, one generator per output type, the same faithfulness evaluator as answers.

## Verification / acceptance criteria

- Each output on a synthetic notebook has a citation per item

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
