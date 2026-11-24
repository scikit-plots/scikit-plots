---
title: "mcp: tools cannot report progress or be cancelled"
status: open
kind: "other"
area: "scikitplot/mcp"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.mcp"
towncrier_type: "feature"
towncrier_fragment: ""
---

# mcp: tools cannot report progress or be cancelled

## Summary

MCP tools served here return one result per call. Long-running work (a corpus build, a media generation job) cannot send progress notifications, be cancelled, or be polled by job id.

## Why it matters

GPU-backed tools take minutes; clients time out or block, and a user cannot stop a costly job.

## Current evidence

- `grep -rEil 'progressToken|notifications/progress' scikitplot/mcp --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Tools that start a job and return its id, progress notifications with the client's progress token, cancellation, and a status tool; the job model from the generation notes underneath.

## Affected paths and ownership

- `scikitplot/mcp`
- `scikitplot/genai`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Client without progress support (polling)
- Cancellation after completion
- Server restart (jobs resume from the store)

## Proposed direction

Add job tools over the job model; follow the protocol's progress and cancellation messages.

## Verification / acceptance criteria

- A stub long job reports progress three times and stops when cancelled

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.mcp` / `feature`, one bullet, when the change lands.
