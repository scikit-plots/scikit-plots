# B29 — Share Fixed-Path Fragment Transport

Status: **COMPLETE — Run 13 current generated transport**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Remove the public Global Share read capability from current HTTP request paths so ordinary CDN/provider/reverse-proxy URL logs do not receive it, while preserving the Run 9–10 artifact lifecycle contract and bounded compatibility for already-issued legacy path links.

## Source anchor

- Input archive: `scikitplot__sphinx_ai_assistant_b18_run12_contribution_lifecycle_withdrawal_overlay.zip`
- SHA-256: `f55b5cea20ccf8790990bde541c618786bd64cc9087401a73f88945d2f83d215`
- Review date: 2026-08-29

## Landed controls

1. **Fragment-backed public URL** — new Global Share URLs are `/v1/share#share=<locator>`. URI fragments are browser-local and are not part of the HTTP request target.
2. **Fixed viewer path** — `GET /v1/share` returns a no-store/noindex viewer shell. It reads only an exact allowlisted locator shape from `location.hash` and uses DOM `textContent` for untrusted conversation values.
3. **Fixed operation paths** — current read/status/update/revoke calls use `/v1/share/read`, `/status`, `/update`, and `/revoke`. The public locator is bounded JSON body data rather than request-path data.
4. **Capability separation preserved** — update/revoke still require `X-Share-Edit-Token`; the private edit capability remains page-memory-only and server storage retains only its digest.
5. **Cross-backend locator parity** — exact 32-hex HF IDs and canonical 36-character Worker UUIDs are accepted; loose `[0-9a-f-]{32,36}` matching is not the current fixed-viewer contract.
6. **Lifecycle ledger migration** — session recovery accepts only exact matching `#share=<id>` fixed-viewer URLs plus legacy path links. Query/userinfo/foreign fragments remain rejected.
7. **Write-abuse parity** — fixed Worker `/update` applies the same Share write-rate gate as create/HF update.
8. **Cloudflare logging defense in depth** — bundled Wrangler disables automatic invocation logs while leaving privacy-minimized custom application events available. This reduces URL telemetry but is not a claim about every external WAF/full-body trace product.
9. **Legacy compatibility is explicit** — old `HEAD/GET/PATCH/DELETE /v1/share/{id}` routes remain so already-issued links/clients do not break. They are deprecated capability-bearing paths and remain a bounded migration residual until expiry/revoke/removal.

## Security boundary

The current transport closes **request-path capability exposure for newly generated links**. It does not claim that a provider configured to record full request bodies, packet captures, browser history/extensions, or arbitrary third-party telemetry cannot observe a capability. Those are separate deployment/data-flow controls.

Legacy path links necessarily reveal their capability in the first/each legacy path request; operators that require immediate elimination must disable the legacy routes after their migration window.

## Verification

- `tests/test_run13_share_capability_transport.py` — fixed URLs/routes, viewer CSP/no-store/DOM isolation, HF runtime flow, body limits, Worker/Wrangler parity, and client source contract.
- `tests/test_share_fixed_transport.mjs` — real Worker create -> viewer -> status/read/update/revoke flow using canonical Worker UUIDs.
- `tests/test_global_share_capability.mjs` — client fixed-path locator/update/status/revoke and Worker parity.
- `tests/test_share_conversation_dom.mjs` — fragment URL lifecycle tracking/recovery without edit-token persistence.
- `tests/test_mutation.py` — positive controls for reintroducing capability-bearing update/revoke/status paths and rejecting fragment ledger entries.
- broad non-Sphinx, syntax, maintenance, Sphinx-environment, and packaged-copy gates are recorded in `VERIFICATION.md`.

## Contracts

- `AIA-C27 ShareCapabilityTransport = HOLDS` for current generated links and fixed-path operations.
- `AIA-022 / SEC-P0-15 = PARTIAL (CURRENT TRANSPORT CLOSED)` because application/current URL-path leakage is closed, but legacy capability-bearing routes remain intentionally available during migration.
- `SEC-P1-34 = OPEN/BOUNDED LEGACY` tracks retirement of legacy `/v1/share/{id}` capability-bearing routes.

## Residuals

- Retire legacy path routes after the old-link compatibility window if the deployment requires zero capability-bearing request paths.
- Treat full request-body/WAF/packet tracing as sensitive or disable/redact it; fixed paths protect URL/request-target logging, not arbitrary payload capture.
- Strict distributed rate accounting remains `SEC-P0-31`.
- Shared contribution receipt authority/provider-history erasure, representative-browser E2E, and canonical Sphinx-enabled release verification remain independent residuals.

## Rollback

Revert viewer, fixed backend routes, client locator transport, Worker observability config, tests, and maintenance together. Do not restore current-generated capability-bearing URLs while claiming `AIA-C27` HOLDS.


## Run 14 supersession note

B30 narrows B29's compatibility statement: legacy routes no longer apply to every object. Generation-2 objects are rejected on `/v1/share/{id}`, legacy PATCH is retired, and fixed update migrates old objects. B29 remains the historical fixed-path transport checkpoint; B30 owns the monotonic legacy drain.
