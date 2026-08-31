# B42 — Hostile-Parent & Egress Boundary Hardening

Status: **WORKING-TREE FREEZE GREEN — package cycle pending**

## Scope

Narrow the repository-controlled portions of `SEC-P1-42` and the B41 deployment boundary without claiming that a parent compromised before bridge startup or external response headers are trustworthy.

## Implemented

- protocol v2 (`2.0.0`) with the bootstrap channel nonce generated inside the isolated frame by WebCrypto;
- no bootstrap capability in the iframe URL and no `Math.random()` security fallback;
- capture-phase host handshake listener installed before iframe attachment and valid HELLO consumption using snapshotted native event primitives;
- build-generated, exact-origin parent policy with a deny-all source default and closed schema;
- sandbox tightened by removing `allow-popups-to-escape-sandbox` and retaining no top-navigation authority;
- frame-self HTTP(S) navigation interception with external navigation restricted to `_blank` + `noopener,noreferrer`;
- assistant-service fetch wrapper forces `credentials="omit"` by default and permits at most `same-origin` under an explicit site-owner compatibility opt-in;
- canonical documentation reads remain the one deliberate same-origin credentialed path, but use redirect-error/cache-no-store semantics and streaming byte/character ceilings;
- cross-origin microphone delegation is independently default-Off and unavailable voice UI is suppressed;
- isolated Web Storage scope now includes both validated parent origin and normalized documentation root, preventing project collisions under one docs origin.

## Working-tree verification

- B42 Python: **11/11**.
- B42 executable browser: **35/35**.
- B37–B41 compatibility Python: **61/61**.
- registered Node harnesses: **46/46**.
- mutation/logging/privacy: **236/236**.
- runnable non-Sphinx: **766 passed, 3 skipped**.
- Sphinx-inclusive after updating the stale build-finished-hook contract: **1232 passed, 3 skipped, 5 failed, 62 errors**; every non-green case remains in `test___init__.py` and terminates on unavailable `sphinx`.
- complete two-root Python compile: **69/69**;
- browser / isolation-host / isolated-frame / Worker JavaScript syntax: **GREEN**;
- Wrangler TOML: **GREEN**, `observability.logs.invocation_logs=false`;
- supply-chain + release-evidence TOML and offline verifier/release subjects: **GREEN**; proxy version remains **7.0.0**, runtime-source SHA-256 remains `0798bd9861d896a796c76ac03e0d828cb8e83cce7060c3c57069e40c605438c1`;
- maintenance drift: **GREEN**;
- controlled Run 22 → Run 23 diff: **5 added · 20 modified · 0 removed = 25 paths**;
- prepackage membership: **293 files**, exactly `scikitplot/` + `maintenances/`, zero cache/bytecode after cleanup;
- candidate / prefinal / exact-final-byte package cycle remains pending.

## Security disposition

B42 closes a newly identified repository-controlled isolation escape/egress class (`SEC-P1-44`): weak/fallback bootstrap entropy, URL-carried bootstrap capability, frame-self navigation across the origin boundary, popup sandbox escape, ambient assistant-service credentials, and unscoped cross-origin microphone delegation.

`SEC-P1-42` remains **OPEN / ARCHITECTURAL** for a parent compromised before the host bridge initializes: it can falsify source page content, monkeypatch primitives before snapshot, remove/cover/clickjack the frame, or deny service. `SEC-P1-43` remains **OPEN / DEPLOYMENT EVIDENCE** for real CSP/frame-ancestors/CORS/CDN/reverse-proxy behavior.

## Candidate packaged-byte acceptance

**GREEN — independently re-extracted candidate bytes.** Exact candidate bytes reproduced B42 Python **11/11**, B42 browser **35/35**, registered Node **46/46**, mutation/logging/privacy **236/236**, runnable non-Sphinx **766 passed, 3 skipped**, **69-file** two-root Python compile, browser/isolation-host/isolated-frame/Worker JavaScript syntax, Wrangler `invocation_logs=false`, supply-chain + release-evidence TOML, offline supply-chain/release subjects and maintenance drift GREEN. Sphinx-inclusive remained **1232 passed, 3 skipped, 5 failed, 62 errors**, with every non-green case confined to the established unavailable-`sphinx` `test___init__.py` boundary. Candidate hygiene: **293 files**, exact two roots, ZIP integrity GREEN, zero packaged cache/bytecode.

## Metadata-bearing prefinal packaged-byte acceptance

**GREEN — independently re-extracted prefinal bytes.** Exact prefinal bytes reproduced B42 Python **11/11**, B42 browser **35/35**, registered Node **46/46**, mutation/logging/privacy **236/236**, runnable non-Sphinx **766 passed, 3 skipped**, **69-file** two-root Python compile, browser/isolation-host/isolated-frame/Worker JavaScript syntax, Wrangler `invocation_logs=false`, supply-chain + release-evidence TOML, offline supply-chain/release subjects and maintenance drift GREEN. Sphinx-inclusive remained **1232 passed, 3 skipped, 5 failed, 62 errors**, all unavailable-`sphinx` only. Archive hygiene remained **293 files**, exact two roots, ZIP integrity GREEN and zero packaged cache/bytecode. Final delivery is rebuilt once from this metadata freeze; exact-final-byte acceptance and SHA-256 remain external to avoid self-reference.
