---
title: "agent: no long-term memory beyond run state"
status: open
kind: "other"
area: "scikitplot/agent"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.agent"
towncrier_type: "feature"
towncrier_fragment: ""
---

# agent: no long-term memory beyond run state

## Summary

`MemoryStore` and `FileStateStore` persist one run's state. There is no long-term memory across runs: episodic (what happened), semantic (facts learned), procedural (how-to steps), with consolidation and forgetting.

## Why it matters

Assistants that learn from earlier sessions need memory with provenance and expiry; storing everything in run state does not scale or protect privacy.

## Current evidence

- `grep -rEil 'episodic|procedural_memory' scikitplot/agent --include=*.py` (tests and vendored code excluded) finds nothing.
- Already public in `scikitplot.agent`: `MemoryStore` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A memory store with typed entries (kind, content, source, time, confidence, expiry), retrieval by relevance and recency, consolidation (merge duplicates, summarise), forgetting by policy and on request, and cleanprompt applied before storage.

## Affected paths and ownership

- `scikitplot/agent`
- `scikitplot/corpus` (retrieval for recall)

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Contradicting memories (newer wins, both kept with times)
- Requests to forget (hard delete)
- Personal data (redacted or refused)

## Proposed direction

Build on the store protocols; retrieval through the corpus index.

## Verification / acceptance criteria

- Memories written in one run are recalled in the next with their source; a forget request removes them

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.agent` / `feature`, one bullet, when the change lands.
