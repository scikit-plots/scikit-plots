# B40 — Runtime Isolation & Secret-Boundary Hardening

Status: **IMPLEMENTED — prefinal exact bytes GREEN; final exact-byte acceptance pending**

## Why this checkpoint exists

Run 18 separated network feedback consent from the public `ai-assistant-feedback`
event, but `document` was still used as an internal coordination bus for model,
profile, conversation, contribution, reasoning and editor lifecycle events. That
made internal state observable to arbitrary same-origin page listeners even when
page integration was Off. Separately, `Origin: null` Share compatibility did not
separate read-only local-file viewing from mutation authority, and browser-entered
Share/Feedback bearer tokens were memory-only but enabled by default.

## Landed boundaries

1. **Private assistant event bus.** Internal lifecycle coordination no longer
   depends on `document`. A public `CustomEvent` is emitted only when the reader
   holds the current versioned page-integration permission, and its detail is a
   bounded per-event projection. Raw model objects, provider model ids, active
   endpoint keys/URLs, bearer tokens, Q&A/note text and stable conversation ids
   are not projected.
2. **Page integration consent v2.** The current permission key is
   `ai-assistant-page-integration-consent`, version `2.0.0`. The old feedback-only
   key does not migrate authority. Network telemetry consent remains independent.
3. **Attachment integration follows the same permission.** The `+` attachment
   hook does not publish an event while page integration is Off.
4. **Opaque-origin read/write split.** `SHARE_ALLOW_OPAQUE_ORIGIN=true` permits
   only the bounded Share viewer/read surface. Create/update/revoke/status-
   capability operations require the additional
   `SHARE_ALLOW_OPAQUE_ORIGIN_WRITE=true`; strict HF deployments refuse that
   write opt-in. Worker behavior matches the route/method split.
5. **Runtime bearer entry defaults Off.** New Sphinx setting
   `ai_assistant_allow_runtime_tokens=False` controls browser-entered Share and
   Feedback bearer compatibility. The endpoint registry enforces the policy
   centrally, so DevTools/API injection cannot bypass a hidden/disabled field.
   Build-time credentials remain prohibited and runtime tokens never enter Web
   Storage.
6. **Standalone Share viewer hardening.** HF and Worker viewer responses deny
   framing and deny camera, microphone, geolocation, payment, USB and Topics
   permissions in addition to existing CSP/no-referrer/no-store controls.

## Truthful residual

This checkpoint reduces repository-controlled same-origin observation but does
**not** create a browser security boundary against arbitrary compromised
same-origin JavaScript. A stronger future architecture would place sensitive
assistant state in a separate-origin iframe/service with a narrow validated
message/capability protocol. This remains an explicit architectural residual,
not a closed claim.

## Working-tree verification

- `test_run21_runtime_isolation_secret_boundary.py`: **8 passed**
- `test_run21_runtime_isolation_secret_boundary.mjs`: **21/21**
- focused compatibility/CORS/Node wrappers after contract updates: **57 passed**
- complete runnable non-Sphinx suite: **742 passed, 3 skipped**
- registered Node harnesses: **44/44**
- mutation/logging/privacy positive controls: **236/236**
- packaged Python compile: **67/67**
- browser + Worker JavaScript syntax: **GREEN**
- Wrangler TOML: **GREEN**, `observability.logs.invocation_logs=false`
- supply-chain + release-evidence policy TOML: **GREEN**
- offline supply-chain verifier + release-subject printer: **GREEN**
- frozen runtime-source SHA-256: `0798bd9861d896a796c76ac03e0d828cb8e83cce7060c3c57069e40c605438c1`
- Sphinx-inclusive environment boundary: **1208 passed, 3 skipped, 5 failed, 62 errors**; every non-green case is `test___init__.py` terminating on missing `sphinx`
- mutation anchor retargeted from public `document` bus to private-bus chokepoint
- proxy version: **7.0.0**
- controlled Run 20 → Run 21 diff: **4 added · 29 modified · 0 removed = 33 paths**
- prepackage membership: **279 files**, exactly two roots and zero generated cache/bytecode

Candidate, prefinal and exact-final-byte evidence is recorded separately in
`VERIFICATION.md` after each immutable package cycle.

## Candidate-byte acceptance

Independent candidate extraction reproduced **8/8 B40 Python**, **21/21 B40
browser**, **44/44 Node registry**, **236/236 mutation/logging/privacy**, **742
passed + 3 skipped** runnable non-Sphinx, **67-file** compile, release-subject /
supply-chain / JS / TOML / maintenance GREEN, and Sphinx-inclusive **1208 passed,
3 skipped, 5 failed, 62 errors** with missing `sphinx` in `test___init__.py` as
the only non-green family. Candidate archive hygiene: **279 files**, exact two
roots, zero cache/bytecode, ZIP integrity GREEN.

## Prefinal-byte acceptance

Independent metadata-bearing prefinal extraction reproduced **8/8 B40 Python**,
**21/21 B40 browser**, **44/44 Node registry**, **236/236 mutation/logging/privacy**,
**742 passed + 3 skipped** runnable non-Sphinx, **67-file** compile, release-subject /
supply-chain / JS / TOML / maintenance GREEN, and Sphinx-inclusive **1208 passed,
3 skipped, 5 failed, 62 errors** with missing `sphinx` in `test___init__.py` as
the only non-green family. Archive hygiene remained **279 files**, two exact roots,
zero generated cache/bytecode and ZIP integrity GREEN. Final-byte acceptance and
SHA-256 are intentionally recorded externally so the final archive is not rewritten.
