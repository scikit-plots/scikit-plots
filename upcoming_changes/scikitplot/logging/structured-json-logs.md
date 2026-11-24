---
title: "logging: no structured (JSON) log output"
status: open
kind: "other"
area: "scikitplot/logging"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "scikitplot.logging"
towncrier_type: "feature"
towncrier_fragment: ""
---

# logging: no structured (JSON) log output

## Summary

The package logger formats human-readable lines. There is no JSON formatter with stable fields (time, level, logger, message, and context such as run id, source path, document id).

## Why it matters

Services (MCP servers, batch builds) ship logs to collectors that parse JSON; correlation across components needs a shared run id.

## Current evidence

- `python -m scikitplot api scikitplot.logging` lists no public name matching `json`.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

An optional JSON formatter and a context helper that adds run and document identifiers, redaction of configured fields, and documentation of the field names.

## Affected paths and ownership

- `scikitplot/logging`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Non-serialisable extra fields
- Very long messages
- Messages containing personal data (redaction hook)

## Proposed direction

Formatter class plus `contextvars` for run context.

## Verification / acceptance criteria

- A test parses every emitted line as JSON with the documented fields

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `scikitplot.logging` / `feature`, one bullet, when the change lands.
