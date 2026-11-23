---
title: "Generation and agents: no tracing spans or service-level metrics"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation and agents: no tracing spans or service-level metrics

## Summary

Agent runs write an audit log; there are no standard tracing spans (per step: model, tokens, cost, time) or service-level indicators (latency percentiles, failure rate) that common observability tools read.

## Why it matters

Production pipelines need a dashboard and alerts; an audit log alone cannot show where time and money go across a batch.

## Current evidence

- `grep -rEil 'opentelemetry' scikitplot/agent --include=*.py` (tests and vendored code excluded) finds nothing.
- Already public in `scikitplot.agent`: `AuditSink`, `JsonlAuditSink`, `MemoryAuditSink` (the base to build on).
- The documentation proxy has its own `_utils/_telemetry.py`, private to it.

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

Optional OpenTelemetry spans for each step (reading, retrieval, generation, judging), metrics for latency and cost, and an SLO helper that evaluates a run against targets; off by default with no exporter configured.

## Affected paths and ownership

- `scikitplot/genai`
- `scikitplot/agent`
- `scikitplot/corpus`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- No exporter configured (no overhead)
- Sensitive attributes (redacted)
- High-cardinality labels

## Proposed direction

Thin wrapper over the OpenTelemetry API (optional), emitting from the existing audit points.

## Verification / acceptance criteria

- A run with an in-memory exporter produces one span per step with cost and time attributes

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
