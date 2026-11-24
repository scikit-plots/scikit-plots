---
title: "No single setting that guarantees nothing leaves the machine"
status: open
kind: "other"
area: "scikitplot/notebook"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# No single setting that guarantees nothing leaves the machine

## Summary

Readers, embedding engines, downloaders, MCP and the docs assistant each have their own network and provider settings; there is no one switch that forbids every outbound call and every remote model.

## Why it matters

Sensitive sources (clinical, legal, personal) must be processed locally; users cannot verify that every component respects that.

## Current evidence

- `grep -rEil 'local_only' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A `local_only` profile that every component checks before any network access or remote model call (refusing with a clear error), plus a run report listing every outbound call attempted.

## Affected paths and ownership

- `scikitplot/corpus`
- `scikitplot/agent`
- `scikitplot/mcp`
- `scikitplot/cleanprompt`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Model downloads at first use
- Telemetry of third-party libraries
- DNS lookups

## Proposed direction

A shared policy object (see the policy-profile note) checked by a central network helper.

## Verification / acceptance criteria

- With the profile on, a test that patches `socket.connect` sees no call during a full notebook run

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
