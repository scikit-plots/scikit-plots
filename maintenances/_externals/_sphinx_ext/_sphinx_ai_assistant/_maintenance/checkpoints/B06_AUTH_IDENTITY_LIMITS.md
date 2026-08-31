# B06 — Auth Identity Limits

Status: **COMPLETE — Run 11 / B27**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Separate share read/edit authority, trusted proxy identity, and pre-buffer request/resource limits.

## Prerequisites

- `B05`

## Scope

- share capability model
- forwarded headers
- body limits
- rate identity

## Non-goals

- feedback data model

## Required evidence before editing

- Record the exact source snapshot/commit being reviewed.
- Re-run the maintenance tracker before changing production code.
- Name the logical contract(s) touched by this checkpoint.
- Record current behavior with a test, build artifact, source anchor, or explicit `UNVERIFIED` status.
- If external/upstream behavior matters, pin the upstream revision used as evidence.

## Execution record schema

Fill these fields in this file when the checkpoint becomes active:

```yaml
checkpoint: B06
status: COMPLETE
started_at: 2026-08-29
completed_at: 2026-08-29
source_anchor: scikitplot__sphinx_ai_assistant_b18_run10_global_link_lifecycle_hardening_overlay.zip sha256=cd3897bc7078fd8a1df4cbb7c8e3282772868062b7cbe10c450f723c2ed673fc
upstream_anchor: Cloudflare KV same-key limit + KV consistency + Workers Rate Limiting API revalidated 2026-08-29
production_code_modified: true
contracts_touched: [AIA-C09, AIA-C10, AIA-C11, AIA-C18]
files_read: [_hf_spaces_proxy/app.py, _hf_spaces_model/app.py, dev_proxy.py, _cf_worker/index.js]
files_changed: [_hf_spaces_proxy/app.py, _hf_spaces_model/app.py, dev_proxy.py, _cf_worker/index.js, _cf_worker/wrangler.toml, tests/test_b05_b06_release_security.py, tests/test_chat_authority.mjs]
findings_opened: [SEC-P0-31]
findings_closed: [SEC-P0-08, SEC-P0-09, SEC-P0-23]
risks: [HF process-local and Worker KV abuse gates are not globally authoritative quotas, IP-like identities can represent shared NAT/proxy populations, direct model service should remain behind the intended proxy/private deployment boundary]
rollback: revert B27 body/origin/limiter changes together; never reintroduce post-allocation body checks as the only guard
```

## Verification gates

- [x] read token cannot edit — preserved Run 3 distinct Share edit capability
- [x] spoofed IP denied/ignored — HF defaults to direct peer and trusts XFF only by explicit ingress declaration; every HF rate-limited public route uses the same helper; Worker uses Cloudflare edge identity
- [x] oversize request bounded — Content-Length precheck plus incremental stream ceiling on HF proxy/model and Worker; loopback dev proxy rejects oversized declared bodies before read

## Closure rule

A checkpoint is not `COMPLETE` until:

1. its evidence is reproducible from the repository;
2. every changed contract has a regression gate;
3. `REGISTRY.md`, `STATE.json`, and relevant tracker files agree;
4. any remaining limitation is explicitly `DEFERRED` or `BLOCKED`, never hidden;
5. the next bounded checkpoint is `B07`.


## Run 11 closure evidence

All HF proxy public body routes now pass through one incremental reader; the direct model service independently streams to the same hard 16 MiB maximum ceiling; the Worker streams with `ReadableStream.getReader()` and cancels on overflow. In-memory HF rate-limit identity maps are hard-bounded and fail closed for new identities at capacity. The Worker no longer rewrites one KV counter key per request; unique TTL event keys avoid KV's documented one-write-per-second same-key ceiling. Distributed rate limiting remains intentionally a soft abuse boundary (`SEC-P0-31`), not accounting or authentication.
