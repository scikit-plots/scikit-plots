# B04 — Prompt Authority

Status: **COMPLETE VIA B20 / RUN 4**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Move authoritative model policy server-side and treat page/retrieval/user content as untrusted data.

## Prerequisites

- `B03`

## Scope

- browser request shape
- proxy/model role handling
- direct endpoint behavior

## Non-goals

- CORS/share changes

## Required evidence before editing

- Record the exact source snapshot/commit being reviewed.
- Re-run the maintenance tracker before changing production code.
- Name the logical contract(s) touched by this checkpoint.
- Record current behavior with a test, build artifact, source anchor, or explicit `UNVERIFIED` status.
- If external/upstream behavior matters, pin the upstream revision used as evidence.

## Execution record schema

Fill these fields in this file when the checkpoint becomes active:

```yaml
checkpoint: B04
status: COMPLETE
started_at: 2026-08-29
completed_at: 2026-08-29
source_anchor: scikitplot__sphinx_ai_assistant_b18_run3_global_share_authority_overlay.zip
upstream_anchor: null
production_code_modified: true
contracts_touched: [AIA-C05, AIA-C12, AIA-C18]
files_read: [browser JS, HF proxy, Cloudflare Worker, dev proxy, direct HF model service]
files_changed: [see B20_PROMPT_AUTHORITY_CREDENTIAL_BINDING.md]
findings_opened: []
findings_closed: [SEC-P0-04, SEC-P0-05]
risks: [custom endpoints remain outside bundled trust boundary, CORS/resource/logging parity deferred]
rollback: revert B20/Run4 as one security-breaking protocol increment
```

## Verification gates

- [x] direct caller cannot set system authority
- [x] malicious page content remains reference data

Implementation and evidence live in `B20_PROMPT_AUTHORITY_CREDENTIAL_BINDING.md`.

## Closure rule

A checkpoint is not `COMPLETE` until:

1. its evidence is reproducible from the repository;
2. every changed contract has a regression gate;
3. `REGISTRY.md`, `STATE.json`, and relevant tracker files agree;
4. any remaining limitation is explicitly `DEFERRED` or `BLOCKED`, never hidden;
5. the next bounded checkpoint is `B05`.
