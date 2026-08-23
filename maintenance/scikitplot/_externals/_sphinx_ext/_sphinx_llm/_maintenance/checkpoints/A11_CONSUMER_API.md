# A11 — Consumer Api

Status: **NOT_STARTED**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_llm**

## Objective

Freeze a stable internal facade for document/artifact lookup.

## Prerequisites

- `A10`

## Scope

- manifest/page lookup API
- capability/status API

## Non-goals

- assistant internals

## Required evidence before editing

- Record the exact source snapshot/commit being reviewed.
- Re-run the maintenance tracker before changing production code.
- Name the logical contract(s) touched by this checkpoint.
- Record current behavior with a test, build artifact, source anchor, or explicit `UNVERIFIED` status.
- If external/upstream behavior matters, pin the upstream revision used as evidence.

## Execution record schema

Fill these fields in this file when the checkpoint becomes active:

```yaml
checkpoint: A11
status: NOT_STARTED | IN_PROGRESS | BLOCKED | COMPLETE | DEFERRED | SUPERSEDED
started_at: null
completed_at: null
source_anchor: null
upstream_anchor: null
production_code_modified: false
contracts_touched: []
files_read: []
files_changed: []
findings_opened: []
findings_closed: []
risks: []
rollback: null
```

## Verification gates

- [ ] consumer does not import upstream private modules
- [ ] API tests

## Closure rule

A checkpoint is not `COMPLETE` until:

1. its evidence is reproducible from the repository;
2. every changed contract has a regression gate;
3. `REGISTRY.md`, `STATE.json`, and relevant tracker files agree;
4. any remaining limitation is explicitly `DEFERRED` or `BLOCKED`, never hidden;
5. the next bounded checkpoint is `A12`.
