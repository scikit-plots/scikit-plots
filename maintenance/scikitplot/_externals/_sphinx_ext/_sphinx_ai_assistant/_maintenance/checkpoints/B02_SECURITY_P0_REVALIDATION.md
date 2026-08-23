# B02 — Security P0 Revalidation

Status: **NOT_STARTED**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Reproduce/adjudicate every open P0 against the exact current source snapshot.

## Prerequisites

- `B01`

## Scope

- proxy/model/worker/browser/build config
- existing security index evidence

## Non-goals

- fix all P0s in one checkpoint

## Required evidence before editing

- Record the exact source snapshot/commit being reviewed.
- Re-run the maintenance tracker before changing production code.
- Name the logical contract(s) touched by this checkpoint.
- Record current behavior with a test, build artifact, source anchor, or explicit `UNVERIFIED` status.
- If external/upstream behavior matters, pin the upstream revision used as evidence.

## Execution record schema

Fill these fields in this file when the checkpoint becomes active:

```yaml
checkpoint: B02
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

- [ ] each finding has source evidence, invariant, owner, exploit condition, regression gate

## Closure rule

A checkpoint is not `COMPLETE` until:

1. its evidence is reproducible from the repository;
2. every changed contract has a regression gate;
3. `REGISTRY.md`, `STATE.json`, and relevant tracker files agree;
4. any remaining limitation is explicitly `DEFERRED` or `BLOCKED`, never hidden;
5. the next bounded checkpoint is `B03`.
