---
title: "Generation: no batch fan-out from records with cost and time budgets"
status: open
kind: "other"
area: "scikitplot/genai"
discovered_during: "Capability review against current practice, gap probed on main cba9383 (2026-11-24)"
release_note: "required"
towncrier_section: "unknown"
towncrier_type: "feature"
towncrier_fragment: ""
---

# Generation: no batch fan-out from records with cost and time budgets

## Summary

Personalised media for many records (one brief per row) needs fan-out with concurrency limits, a total budget and per-item budgets; the agent has `FanOut` and `Limits` for tool calls but nothing ties them to generation cost.

## Why it matters

Uncontrolled fan-out exhausts quotas or money; quality-versus-cost trade-offs cannot be measured.

## Current evidence

- `grep -rEil 'cost_per_|cost_report' scikitplot --include=*.py` (tests and vendored code excluded) finds nothing.
- Already public in `scikitplot.agent`: `Agent`, `Limits`, `FanOut` (the base to build on).

## Root cause / current understanding

Not implemented: the capability was outside the submodule's scope so far.

## Expected behavior

A batch runner over a table of records: templated requests, concurrency limit, budget in money and time (refuses to start when the estimate exceeds it, stops cleanly when reached), and a cost report per item and in total.

## Affected paths and ownership

- `scikitplot/genai`
- `scikitplot/agent`

## Constraints and non-goals

Optional extra, imported lazily; no new base dependency; the same size, time and expansion limits as the existing readers and sources; no network access unless the caller asks for it; the licence of every library and model checked before it becomes a default; deterministic output for the same input and settings.

## Edge cases to cover

- Estimates wrong by a factor (actual cost tracked)
- Mixed modalities per record
- Empty or invalid records

## Proposed direction

Build on `agent.FanOut` and `Limits`; costs from backend metadata.

## Verification / acceptance criteria

- A 1,000-record synthetic batch with a stub backend stays within budget and reports cost per item

## Documentation impact

User guide and API reference of the submodule; its usage skill.

## Release-note promotion

Required: `unknown` / `feature`, one bullet, when the change lands.
