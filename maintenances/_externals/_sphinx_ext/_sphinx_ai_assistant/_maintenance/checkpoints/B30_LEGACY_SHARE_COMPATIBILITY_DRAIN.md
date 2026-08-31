# B30 — Legacy Share Compatibility Drain

Status: **COMPLETE — Run 14 bounded migration ratchet**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Prevent the deprecated capability-bearing `/v1/share/{id}` transport from becoming a permanent alternate API while preserving already-issued pre-generation links for their bounded remaining TTL.

## Source anchor

- Input archive: `scikitplot__sphinx_ai_assistant_b18_run13_share_fixed_path_fragment_transport_overlay.zip`
- SHA-256: `5f627345c7e3be66784fd935b9bbe1ab315b7fb8ba9adf4d4fb260b698e7606c`
- Review date: 2026-08-29

## Landed controls

1. **Server-owned transport generation** — every new Share and every fixed-path update is stamped generation 2 (`transport_version` / `transportVersion`).
2. **Generation-gated legacy access** — legacy `HEAD`/`GET`/authenticated `DELETE` serve only pre-generation entries (missing generation metadata or explicit generation 1). Unknown/tampered generations fail closed; generation 2 returns `404` on capability-bearing paths.
3. **Legacy PATCH retired** — deprecated path mutation returns `410` and cannot extend TTL or refresh the capability-bearing transport. Supported mutation is `POST /v1/share/update` with the locator in a bounded body.
4. **One-way migration** — fixed `/update` accepts an old object, preserves edit authority, writes generation 2, and immediately makes the old path ineligible.
5. **Bounded natural drain** — no new generation-2 object can enter legacy compatibility. Remaining pre-generation objects age out under the pre-existing maximum 365-day object TTL or disappear earlier through revoke/expiry.
6. **Standards-based deprecation signaling** — eligible legacy responses use RFC 9745 Structured Field Date `Deprecation: @1787961600`, object-expiry `Sunset`, and a fixed-viewer successor link. These headers are migration hints, never authorization controls.
7. **HF/Worker parity** — both bundled Share implementations enforce the same generation ratchet and retired legacy update semantics.
8. **Lifecycle client compatibility** — the browser may still parse an old URL from its bounded lifecycle ledger, but all current status/update/revoke operations use fixed paths; fixed update upgrades old storage generation.

## Security boundary

Run 14 prevents newly created or migrated objects from being served through capability-bearing paths. It cannot retroactively stop the *first request* to an already-issued old URL from exposing that old locator to infrastructure request-URL logging; the locator is literally embedded in the historical URL. That exposure drains as pre-generation objects expire/revoke. Full request-body/WAF/packet telemetry remains deployment-owned.

## Verification

- `tests/test_run14_legacy_share_retirement.py` — HF generation stamping, generation-2 legacy rejection, RFC deprecation/sunset headers, retired PATCH, fixed-path migration, bounded legacy revoke, Worker source parity.
- `tests/test_share_legacy_retirement.mjs` — real Worker KV pre-generation -> deprecated read/status -> retired PATCH -> fixed update -> generation-2 legacy rejection, plus legacy revoke.
- `tests/test_share_server_authority.py` — current lifecycle authority now tests fixed status/read/update/revoke rather than relying on deprecated paths.
- `tests/test_global_share_capability.mjs` — Worker/client source contract updated so legacy PATCH retirement is a required invariant.
- Working-tree acceptance: **60** focused tests, **17/17** Worker legacy-runtime assertions, **37** Node harness wrappers, **193** mutation tests, and **633 passed / 3 skipped** across the complete runnable non-Sphinx suite; syntax/config/maintenance gates GREEN.
- Sphinx-inclusive working-tree attempt: **1099 passed, 3 skipped, 5 failed, 62 errors**; every failure/error remains the missing-`sphinx` `test___init__.py` environment wall.
- Candidate packaged-copy acceptance: **GREEN** — 60 focused, 17/17 Worker runtime, 37 Node harness, 193 mutation, 633 passed / 3 skipped non-Sphinx, syntax/config/maintenance GREEN; **228 files**, exact two-root layout, zero cache/bytecode contamination.
- Candidate Sphinx-inclusive attempt: **1099 passed, 3 skipped, 5 failed, 62 errors**, all on the missing-`sphinx` environment wall.
- Diff from Run 13 input: **23 changed paths** (3 added, 20 modified, 0 removed). Delivery is conditioned on a fresh extraction of the final metadata-bearing archive reproducing the same gates; its final SHA-256 is recorded externally with the delivered artifact.

## Contracts

- `AIA-C27 ShareCapabilityTransport = HOLDS` and is strengthened: generation-2 objects cannot be served through legacy capability-bearing paths.
- `AIA-C28 LegacyShareCompatibilityDrain = HOLDS` at the bundled application boundary.
- `AIA-022 / SEC-P0-15 = PARTIAL (CURRENT + GENERATION-2 PATH EXPOSURE CLOSED)` because historical pre-generation URLs can still expose their locator until their bounded lifetime ends; deployment payload-level telemetry is separate.
- `SEC-P1-34 = PARTIAL / DRAINING LEGACY` rather than OPEN: no new object enters compatibility, legacy PATCH cannot extend lifetime, and fixed update permanently migrates an old object.

## Residuals

- Remove the legacy route code after deployments prove the pre-generation population has drained; until then direct old links still necessarily place their old locator in the request path.
- Disable/redact full-body/WAF/packet telemetry if deployed; fixed paths protect ordinary request-target logging, not arbitrary payload capture.
- Strict distributed rate accounting remains `SEC-P0-31`.
- Shared contribution receipt authority/provider-history erasure, representative-browser E2E, and canonical Sphinx-enabled release verification remain independent residuals.

## Rollback

Do not remove generation stamping/gating while keeping fragment URLs: that would silently recreate a permanent capability-bearing alternate route for every new Share. If emergency legacy compatibility must be widened, document it as a security regression and bound it operationally.
