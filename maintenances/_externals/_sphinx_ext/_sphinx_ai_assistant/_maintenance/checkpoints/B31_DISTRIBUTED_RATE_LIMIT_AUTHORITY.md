# B31 — Distributed Rate-Limit Authority

Status: **COMPLETE AT BUNDLED/CAPABILITY BOUNDARY — Run 15; HF activation remains deployment-conditional**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Add a truthful shared rate-limit decision plane for horizontally scaled deployments without relabeling the existing process-local/KV compatibility gates as globally authoritative accounting.

## Source anchor

- Input archive: `scikitplot__sphinx_ai_assistant_b18_run14_legacy_share_compatibility_drain_overlay.zip`
- SHA-256: `ae7fbb4750cf7cd9bfd5b3c1c8695cec39ce62d75c11aeefcdbd6f45ef0cb023`
- Review date: 2026-08-29

## Landed controls

1. **HF shared Redis backend** — optional `RATE_LIMIT_BACKEND=redis` uses one atomic Lua fixed-window operation for chat/Share/feedback/contribution route families across replicas sharing one Redis consistency domain.
2. **HF fail-closed authority mode** — `RATE_LIMIT_REQUIRE_SHARED=true` rejects local mode and returns bounded HTTP 503 when Redis is absent, misconfigured, unavailable, or fails at runtime; there is no silent Redis-to-process-local quota split.
3. **Private shared identity keys** — raw peer/XFF-derived identity is HMAC-SHA256'd with a dedicated server secret before any Redis key is emitted. Public health/discovery exposes backend/shared/authoritative/readiness booleans only, never Redis URLs/secrets.
4. **Worker sharded Durable Object authority** — one `RateLimitBucket` Durable Object is addressed per route-family + HMAC-derived identity. The object owns strongly coordinated window state and rejects the first request beyond the configured limit even when callers arrive through different Worker/PoP environments.
5. **Worker bundled fail-closed deployment** — `wrangler.toml` binds `RATE_LIMIT_DO` and sets `RATE_LIMIT_REQUIRE_AUTHORITATIVE=true`; missing binding, missing HMAC secret, or Durable Object failure yields 503 before provider spend/write work.
6. **Explicit KV compatibility fallback** — unique TTL event keys remain only when authoritative mode is not required (custom/local compatibility). Their result is labeled `authoritative:false`; Workers Rate Limiting binding is intentionally not substituted for strict shared accounting.
7. **Bounded semantics** — this is abuse-control quota authority, not authenticated human identity, billing accounting, or a claim of atomicity across unrelated/Active-Active Redis domains.
8. **Fresh-deploy dependency/config parity** — HF requirements include the async Redis client; README/app configuration documents the shared-mode activation and HMAC requirements. Worker Wrangler exports the SQLite-backed Durable Object class and dedicated identity secret.

## Security boundary

The bundled Worker deployment now requires a shared Durable Object decision plane. HF ships the shared Redis implementation but remains local by default for single-instance compatibility; horizontal HF deployments close the distributed quota boundary only when Redis is configured and `RATE_LIMIT_REQUIRE_SHARED=true`. A deployment that leaves HF in local mode must continue to describe its counters as process-local abuse controls.

## Verification

- `tests/test_run15_distributed_rate_authority.py` — **11 passed**: shared Redis behavior, HMAC identity privacy, route scope separation, truthful manifest, fail-closed HF source contract, Worker/Wrapper config contract, dependency laziness, plus four positive-control authority mutants.
- `tests/test_run15_worker_rate_authority.mjs` — **15/15 assertions**: Durable Object limit boundary, bounded retry, alarm cleanup, two simulated PoP environments sharing one budget, provider-spend denial, identity independence, health truth, and three fail-closed deployment/runtime cases.
- Existing authority/privacy regression plane — **81 passed** after superseding the stale B27 assertion that KV must be the primary Worker limiter.
- JS harness + client mutation plane — **231 passed** = 38 Node harness wrappers + 193 existing client mutation gates.
- Complete runnable non-Sphinx working tree — **645 passed / 3 skipped**.
- Sphinx-inclusive working-tree boundary: **1111 passed / 3 skipped / 5 failed / 62 errors**; every failure/error remains confined to `test___init__.py` and terminates on missing `sphinx`.
- Candidate/final packaged-byte evidence is recorded below after archive acceptance.
- Diff from Run 14 input: **27 changed paths** (5 added, 22 modified, 0 removed).
- Candidate packaged-copy acceptance: **GREEN** — 81 focused Python regressions, 15/15 Worker authority assertions, 38 Node harness wrappers, 193 client mutation gates, 645 passed / 3 skipped non-Sphinx, syntax/config/maintenance GREEN; **233 files**, exact two-root layout, zero cache/bytecode contamination.
- Candidate Sphinx-inclusive attempt: **1111 passed / 3 skipped / 5 failed / 62 errors**, all on the missing-`sphinx` environment boundary.
- Final metadata-bearing archive acceptance: **GREEN on fresh extraction** — 81 focused, 15/15 Worker authority, 38 Node harnesses, 193 client mutation gates, 645 passed / 3 skipped non-Sphinx, syntax/config/maintenance GREEN; 233 files, two roots, zero cache/bytecode contamination. Final Sphinx-inclusive bytes reproduce **1111 / 3 / 5 / 62**, all missing `sphinx`. The archive SHA-256 is recorded externally with delivery because embedding a file's own digest would be self-referential.

## Contracts

- `AIA-C25 DistributedAbuseLimitSemantics = HOLDS`: local/KV gates remain truthfully non-authoritative, while shared backends advertise authority only when actually selected/ready.
- `AIA-C29 DistributedRateLimitAuthority = HOLDS_WHEN_AUTHORITATIVE_MODE_REQUIRED`: Worker bundled config requires DO authority; HF horizontal deployments require Redis + fail-closed shared mode.
- `AIA-028 / SEC-P0-31 = PARTIAL (RUN 15 IMPLEMENTATION LANDED; HF DEPLOYMENT ACTIVATION REQUIRED)`, not universally CLOSED.

## Residuals

- Production HF horizontal deployments must provision one intended Redis consistency domain, dedicated HMAC key, TLS/ACL/network controls, and set `RATE_LIMIT_REQUIRE_SHARED=true`; repository code cannot prove an operator deployed those resources.
- Multi-region/Active-Active Redis semantics and billing-grade exact accounting are not claimed.
- Run 16/B32 now lands optional shared Redis contribution receipt authority. External Redis durability/production activation, provider-history/global erasure, legacy Share route removal after drain, representative-browser E2E, and canonical Sphinx-enabled release verification remain independent residuals.

## Rollback

Never fall back silently from a required shared limiter to local/KV counters. If an authoritative backend must be disabled operationally, either fail closed or explicitly change the deployment policy/status to soft abuse-control mode and record the reduced guarantee.
