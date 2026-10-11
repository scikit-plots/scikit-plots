# B41 — Separate-Origin Isolation & Capability Messaging

Status: **FINAL METADATA FREEZE READY — exact-final-byte acceptance recorded externally**

## Scope

Close the repository-controlled portion of `SEC-P1-41` by providing an opt-in browser-enforced origin boundary without weakening the B37/B40 privacy, contribution, telemetry or token contracts.

## Implemented

- distinct-origin sandboxed iframe runtime;
- fail-closed no-downgrade parent behavior;
- exact HELLO/INIT validation and one transferred MessagePort;
- protocol `1.0.0`, bounded monotonic capability envelopes and closed capability allowlist;
- bounded visible page-context/canonical Markdown adapter with query/fragment removal;
- parent-origin scoped local/session storage;
- isolated page identity/docs-root abstraction;
- B40 consent-gated public event forwarding with independent host validation;
- strict bootstrap CSP baseline and production isolation deployment guide;
- immutable/sanitized host bootstrap snapshot with secret-key and prototype-pollution-key rejection.

## Working-tree freeze verification

- B41 Python: **11/11**.
- B41 executable browser: **34/34**.
- registered Node harnesses: **45/45**.
- mutation/privacy: **212/212**.
- runnable non-Sphinx: **754 passed, 3 skipped**.
- packaged Python compile across both roots: **68/68**.
- browser / isolation-host / isolated-frame / Worker JavaScript syntax: **GREEN**.
- Wrangler TOML: **GREEN**, `observability.logs.invocation_logs=false`.
- supply-chain + release-evidence policy TOML: **GREEN**.
- offline supply-chain verifier + canonical release subjects: **GREEN**.
- proxy version remains **7.0.0** and runtime-source SHA-256 remains `0798bd9861d896a796c76ac03e0d828cb8e83cce7060c3c57069e40c605438c1`.
- maintenance drift: **GREEN**.
- Sphinx-inclusive boundary: **1220 passed, 3 skipped, 5 failed, 62 errors**; every non-green case remains in `test___init__.py` and terminates on unavailable `sphinx`.
- the pre-B41 `test_css_and_js_added` single-script assertion was intentionally ratcheted to require `ai-assistant-isolation-host.js` before `ai-assistant.js`; the corrected test passes.
- controlled Run 21 → Run 22 diff: **9 added · 20 modified · 0 removed = 29 paths**.
- prepackage membership: **288 files**, exactly `scikitplot/` + `maintenances/`, zero cache/bytecode after cleanup.
- candidate / prefinal / exact-final-byte package cycle remains pending.

## Residuals

`SEC-P1-41` closure is conditional on isolated mode being enabled. `SEC-P1-42` (fully compromised parent integrity/presentation) and `SEC-P1-43` (real deployment header/CORS evidence) remain open.
## Candidate packaged-byte acceptance

**GREEN — candidate archive independently re-extracted.** Exact candidate bytes reproduced B41 Python **11/11**, B41 executable browser **34/34**, registered Node **45/45**, mutation/privacy **212/212**, runnable non-Sphinx **754 passed, 3 skipped**, **68-file** two-root Python compile, browser/isolation-host/isolated-frame/Worker syntax, Wrangler `invocation_logs=false`, supply-chain + release-evidence TOML, release subjects and maintenance drift GREEN. Sphinx-inclusive remained **1220 passed, 3 skipped, 5 failed, 62 errors**, with every non-green case confined to the established missing-`sphinx` `test___init__.py` boundary. Candidate archive hygiene: **288 files**, exact two roots, ZIP integrity GREEN, zero packaged cache/bytecode.
## Metadata-bearing prefinal packaged-byte acceptance

**GREEN — prefinal archive independently re-extracted.** Exact prefinal bytes reproduced B41 Python **11/11**, B41 executable browser **34/34**, registered Node **45/45**, mutation/privacy **212/212**, runnable non-Sphinx **754 passed, 3 skipped**, **68-file** two-root Python compile, browser/isolation-host/isolated-frame/Worker syntax, Wrangler `invocation_logs=false`, supply-chain + release-evidence TOML, release subjects and maintenance drift GREEN. Sphinx-inclusive remained **1220 passed, 3 skipped, 5 failed, 62 errors**, with unavailable `sphinx` as the only failure/error family. Archive hygiene remained **288 files**, exact two roots, ZIP integrity GREEN and zero cache/bytecode. Final delivery will be rebuilt once from this metadata freeze; final SHA-256 stays external to avoid self-reference.
