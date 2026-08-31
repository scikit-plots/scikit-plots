# B05 — Proxy Routing Cors

Status: **COMPLETE — Run 11 / B27**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Bind credentials to approved destinations and unify least-privilege origin policy across relays.

## Prerequisites

- `B04`

## Scope

- BACKEND_URL/credential routing
- proxy CORS
- worker CORS
- relay parity

## Non-goals

- share authorization

## Required evidence before editing

- Record the exact source snapshot/commit being reviewed.
- Re-run the maintenance tracker before changing production code.
- Name the logical contract(s) touched by this checkpoint.
- Record current behavior with a test, build artifact, source anchor, or explicit `UNVERIFIED` status.
- If external/upstream behavior matters, pin the upstream revision used as evidence.

## Execution record schema

Fill these fields in this file when the checkpoint becomes active:

```yaml
checkpoint: B05
status: COMPLETE
started_at: 2026-08-29
completed_at: 2026-08-29
source_anchor: scikitplot__sphinx_ai_assistant_b18_run10_global_link_lifecycle_hardening_overlay.zip sha256=cd3897bc7078fd8a1df4cbb7c8e3282772868062b7cbe10c450f723c2ed673fc
upstream_anchor: Cloudflare Workers/KV/HTTP-header documentation revalidated 2026-08-29
production_code_modified: true
contracts_touched: [AIA-C07, AIA-C08, AIA-C18]
files_read: [_hf_spaces_proxy/app.py, _hf_spaces_model/app.py, dev_proxy.py, _cf_worker/index.js, _cf_worker/wrangler.toml]
files_changed: [_hf_spaces_proxy/app.py, _hf_spaces_proxy/_shared_logic.py, _hf_spaces_proxy/README.md, _hf_spaces_model/app.py, _hf_spaces_model/README.md, dev_proxy.py, _cf_worker/index.js, _cf_worker/wrangler.toml, tests/test_b05_b06_release_security.py]
findings_opened: [SEC-P0-31]
findings_closed: [SEC-P0-02]
risks: [Origin is browser abuse control not authentication, explicit wildcard remains an operator escape hatch, custom external endpoints remain outside bundled policy]
rollback: revert B27 production changes together; do not restore wildcard defaults independently
```

## Verification gates

- [x] malicious destination denied before credential attach — preserved Run 4 destination/redirect gates
- [x] origin matrix green — exact defaults + early browser-Origin rejection across HF proxy/model, Worker, and loopback dev proxy

## Closure rule

A checkpoint is not `COMPLETE` until:

1. its evidence is reproducible from the repository;
2. every changed contract has a regression gate;
3. `REGISTRY.md`, `STATE.json`, and relevant tracker files agree;
4. any remaining limitation is explicitly `DEFERRED` or `BLOCKED`, never hidden;
5. the next bounded checkpoint is `B06`.


## Run 11 closure evidence

B05 closes the remaining bundled-origin policy gap without treating CORS as authentication. Explicit disallowed browser `Origin` values are rejected before handlers perform inference or writes; requests without `Origin` remain valid for server-to-server clients and must satisfy the endpoint's real auth/capability policy. `ALLOWED_ORIGINS=*` remains only an explicit insecure compatibility escape hatch. Credential-destination binding remains the already-proved Run 4 contract.


### Run 16.2.4 additive-origin clarification

The official documentation origin `https://scikit-plots.github.io` is now always retained by the bundled HF proxy and Worker. `ALLOWED_ORIGINS` contributes additional validated exact HTTP(S) origins; it no longer replaces the official package-owned origin. Explicit `*` remains the only wildcard escape hatch. This closes the operational failure mode where a deployment environment override could make the shipped Global Share button fail with `403 Origin not allowed` on the official documentation site.
