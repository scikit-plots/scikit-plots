# Session log — `scikitplot.mcp` review

Newest at top. One terse entry per session.

Status legend: CONFIRMED = behaviour verified; CANDIDATE = to confirm;
RESOLVED = fixed + gated; GATED = needs a maintainer decision / live env.

---

## 2026-08-05 — Docker CLI, health endpoint, and editable-build diagnosis

RESOLVED MCP-SRV-001 deployment gap and the reported Docker startup failure:

- Added explicit `--docker` defaults (Streamable HTTP, `0.0.0.0`, `/mcp`,
  `/healthz`) without pretending that container binding supplies authentication.
- Added validated environment-variable configuration, effective-config output,
  and a proxy-independent `--probe` mode with bounded response parsing.
- Added a minimal, no-cache `/healthz` custom route with no sensitive runtime
  disclosure.
- Added strict path collision/format validation and remote-bind acknowledgement.
- Added CLI and server-shell tests; full local gate is **69 passed**.
- Documented that the Meson editable loader runs before `__main__`, so the Ninja
  rebuild cannot be caught by the MCP CLI; production Docker must install a wheel.
- Added a multi-stage Dockerfile and hardened Compose reference.

The real MCP v2 package could not be installed from the sandbox package mirror,
so the real-SDK network round-trip remains a dependency-enabled CI gate. The
implementation was checked against the official MCP v2 `MCPServer.run`,
`streamable_http_app`, and `custom_route` APIs.

---

## 2026-08 — orientation + handoff scaffolding (no code changes)

Set up the module's durable memory so a fresh chat can run the review:
- Wrote `mcp/MAINTAINING.md` (was empty): contracts, safety chokepoint,
  verification gates, DRY/composition rules, compat rule, resume steps.
- Added `_maintenance/METHODOLOGY.md` (review process) and this log.
- Seeded `_maintenance/MCP_REVIEW_GUIDE.md` from a first grounded read of
  `_core.py` / `_hybrid.py` / `_corpus_annoy.py` / `__init__.py`.

Grounded facts established this session:
- 24 tests green (13 `test_mcp_core.py` + 11 `test_hybrid.py`).
- All mcp modules carry `from __future__ import annotations`; 3.8→3.15+ clean
  (no evaluated subscripted generics, `|`-unions, or version-gated APIs).
- CONFIRMED candidate MCP-SEC-001: `urlparse('//evil.com/x').scheme == ''`, so
  `_safe_uri` admits protocol-relative citation URLs.
- CONFIRMED candidate MCP-CORE-002: `isinstance(float('nan'), float)` is True, so
  non-finite scores flow into the (invalid) JSON payload.

Seeded candidate findings (none fixed yet): MCP-SEC-001, MCP-CORE-002 (P2,
confirmed); MCP-HYB-001, MCP-COUP-001 (P2, candidate); MCP-CORE-003, MCP-HYB-002
(P3); MCP-SRV-001 (D1), MCP-INT-001 (D2) gated.

**Next session:** start with MCP-SEC-001 and MCP-CORE-002 (both in-sandbox
verifiable on the stdlib-only core), then MCP-HYB-001. Defer the gated items
until D1/D2 are decided. Follow `METHODOLOGY.md`.


## 2026-08-06 — idempotent protocol and live-test hardening

- Reproduced the zero-result mismatch: `count=0` with one synthetic passage.
- Moved no-result status text to `message` and enforced equal count/passages/citations lengths.
- Added output-model invariants, strict input types, deterministic repeated-call tests, and a single version source (`0.2.1`).
- Added `--self-test`, restored machine-readable CLI output, and made health-probe failures quiet and controlled.
- Added real MCP SDK in-memory protocol tests and made live HTTP tests explicitly opt-in.
- Hardened shell wrappers and bounded load tests; documented immutable index generations and canary promotion.
- Verification: two consecutive self-contained runs each produced `81 passed, 2 skipped`; compileall and shell syntax checks passed.

## 2026-08-06 — strict live wire validation (`0.2.2`)

- User default run: `84 passed, 1 skipped`.
- User opt-in live run: eleven live checks passed before JSON Boolean `k=true`
  was accepted as integer `1` by non-strict MCP/Pydantic validation.
- Replaced tool input annotations with `StrictStr` and `StrictInt`; retained
  independent `SearchService` validation for direct callers.
- Added SDK-double, real-SDK, and live regressions for Boolean, numeric-string,
  float, null, array, missing, and extra arguments.
- Added a local-versus-live version precheck to prevent testing changed source
  against a stale running container.
- Restored optional MCP dependency skips and added a source manifest integrity
  test.
