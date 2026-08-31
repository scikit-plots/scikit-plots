# B07 — Feedback Provenance

Status: **COMPLETE VIA B22 / RUN 6**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Harden feedback/training persistence, consent versioning, and poisoning boundary.

## Prerequisites

- Original sequence prerequisite: `B06`.
- Security campaign execution: B18 Run 6 / `B22_FEEDBACK_CONTRIBUTION_PROVENANCE_RETENTION.md` implemented the bounded provenance boundary before the later release-wide B05/B06 parity pass.

## Scope

- dataset schema
- write routes
- consent/provenance/authenticity

## Non-goals

- representation work

## Required evidence before editing

- Record the exact source snapshot/commit being reviewed.
- Re-run the maintenance tracker before changing production code.
- Name the logical contract(s) touched by this checkpoint.
- Record current behavior with a test, build artifact, source anchor, or explicit `UNVERIFIED` status.
- If external/upstream behavior matters, pin the upstream revision used as evidence.

## Execution record schema

Execution was completed through B22 / Run 6. The authoritative implementation/evidence record is `B22_FEEDBACK_CONTRIBUTION_PROVENANCE_RETENTION.md`.

```yaml
checkpoint: B07
status: COMPLETE
started_at: 2026-08-29
completed_at: 2026-08-29
source_anchor: B18 Run 6 overlay
production_code_modified: true
contracts_touched: [AIA-C13, AIA-C20]
findings_closed: [AIA-008, SEC-P0-06, SEC-P0-07, SEC-P0-18]
risks:
  - process-local quarantine is not a durable multi-replica review store
  - physical erasure after promotion is not guaranteed across append-only/mirrored storage
  - Cloudflare contribution/review parity remains incomplete
rollback: restore Run 5 baseline only together with re-opening the Run 6 findings/contracts
```

## Verification gates

- [x] untrusted contribution labeled
- [x] consent/provenance schema tests

## Closure rule

A checkpoint is not `COMPLETE` until:

1. its evidence is reproducible from the repository;
2. every changed contract has a regression gate;
3. `REGISTRY.md`, `STATE.json`, and relevant tracker files agree;
4. any remaining limitation is explicitly `DEFERRED` or `BLOCKED`, never hidden;
5. the next bounded checkpoint is `B08`.

## Run 6 closure note

B22 additionally proves minimal opt-in rating telemetry, server-side over-collection stripping, exact versioned consent, quarantine-before-durable-storage, capability-protected pending deletion, separate review promotion, explicit model evidence, and fail-closed training eligibility. The original B07 wording is retained only as the historical checkpoint identity.
