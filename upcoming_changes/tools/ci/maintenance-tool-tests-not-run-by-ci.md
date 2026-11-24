---
title: "Maintenance tool tests run only by hand"
status: open
kind: "reliability"
area: "tools"
discovered_during: "Real-life probe on main cba9383 (2026-11-24)"
release_note: "not-required"
towncrier_section: "unknown"
towncrier_type: "unknown"
towncrier_fragment: ""
---

# Maintenance tool tests run only by hand

## Summary

`maintenances/*/_maintenance/tests/` (reader snapshot, reader benchmark, contract) and the records checker tests run in no automatic workflow: `pr_check_records.yml` is `workflow_dispatch` only.

## Why it matters

A change to a tool or a record can break the tools the next session relies on, and nobody sees it until a person runs them.

## Current evidence

- `.github/workflows/pr_check_records.yml`: `on: workflow_dispatch` (pull_request and push commented out).
- `grep -rn 'maintenances/corpus/_maintenance/tests' .github/workflows` finds nothing.
- `maintenances/corpus/MAINTAINING.md` says to run the tool tests by hand.

## Root cause / current understanding

The workflow was made manual; no other job runs these tests.

## Expected behavior

A pull-request job (path-filtered to `maintenances/**`, `tools/**`, `plugins/**`) runs the records check and every `_maintenance/tests` suite with the standard library only.

## Affected paths and ownership

- `.github/workflows/pr_check_records.yml`

## Constraints and non-goals

Fast (under a minute), no build, pinned actions (zizmor and action-validator clean).

## Edge cases to cover

- A fork without the repository condition

## Proposed direction

Re-enable `pull_request` with path filters and add the tool test commands.

## Verification / acceptance criteria

- A PR touching `maintenances/corpus/_maintenance/tools/` runs `test_reader_snapshot.py` and `test_reader_benchmark.py`

## Documentation impact

none

## Release-note promotion

Not required (no shipped behaviour changes).
