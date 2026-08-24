# B01 — Sphinx Llm Dependency Boundary

Status: **NOT_STARTED**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Define exact producer-consumer interface and block reverse/private dependencies.

## Prerequisites

- `B00`

## Scope

- stable facade/artifact lookup contract
- fallback priority contract

## Non-goals

- move code before producer A11 is ready

## Required evidence before editing

- Record the exact source snapshot/commit being reviewed.
- Re-run the maintenance tracker before changing production code.
- Name the logical contract(s) touched by this checkpoint.
- Record current behavior with a test, build artifact, source anchor, or explicit `UNVERIFIED` status.
- If external/upstream behavior matters, pin the upstream revision used as evidence.

## Execution record schema

Fill these fields in this file when the checkpoint becomes active:

```yaml
checkpoint: B01
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

- [ ] boundary/import tests specified
- [ ] INTEGRATION_CONTRACT current

## Closure rule

A checkpoint is not `COMPLETE` until:

1. its evidence is reproducible from the repository;
2. every changed contract has a regression gate;
3. `REGISTRY.md`, `STATE.json`, and relevant tracker files agree;
4. any remaining limitation is explicitly `DEFERRED` or `BLOCKED`, never hidden;
5. the next bounded checkpoint is `B02`.

---

## Revised intent (pivot)

This checkpoint previously existed to *define* the boundary between the assistant
and `_sphinx_llm`. Under the capability pivot it becomes **sever, and prove
severed**.

The severing is already true in code and false only in documentation:

- 0 references in `__init__.py`, `_static/ai-assistant.js`, `_static/__init__.py`
  or any test;
- 19 files mention `_sphinx_llm`, every one of them documentation or maintenance
  record.

## Exit criteria

```text
[ ] no import of _sphinx_llm anywhere in assistant runtime source
[ ] no assistant surface degrades when _sphinx_llm is absent from `extensions`
[ ] MAINTAINING.md, DEPENDENCY_MAP.md, INTEGRATION_CONTRACT.md, RULESET.md,
    SUBMODULE_STRUCTURE.md rewritten for a two-member family
[ ] the gate proves absence rather than asserting it
```
