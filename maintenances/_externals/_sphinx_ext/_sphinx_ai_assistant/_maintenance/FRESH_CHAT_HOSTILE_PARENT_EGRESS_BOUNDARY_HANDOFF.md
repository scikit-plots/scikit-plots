# Fresh-chat handoff — Run 23 / B42 Hostile-Parent & Egress Boundary Hardening

Input release: Run 22 / B41 exact final SHA-256 `6cc673e6a5dd727312e6c9c653189dd040e1640cb6ed85c17871393c1ea1105f`.

B42 hardens the browser-enforced isolation boundary rather than pretending a hostile parent can be made trustworthy. Protocol v2 moves bootstrap nonce generation into the isolated frame and requires WebCrypto; the nonce is not present in the iframe URL. A build-generated closed-schema parent-origin policy defaults to deny-all. The host installs its capture listener before frame attachment and consumes a valid HELLO before later page listeners.

The sandbox no longer grants popup escape. HTTP(S) frame-self navigation is intercepted so the assistant frame cannot navigate onto the documentation origin and collapse the SOP boundary; external navigation uses `_blank` with `noopener,noreferrer`. Assistant-service requests centrally omit ambient credentials by default and an explicit compatibility opt-in permits at most `same-origin`, never `include`. Canonical documentation reads remain intentionally same-origin but are streaming-bounded. Microphone delegation is a separate default-Off site-owner decision. Isolated storage is partitioned by parent origin plus normalized docs root.

Current working-tree gates before packaging: B42 Python 11/11, B42 browser 35/35, B37–B41 compatibility Python 61/61, Node 46/46, mutation/privacy 236/236, runnable non-Sphinx 766 passed / 3 skipped, and Sphinx-inclusive 1232 passed / 3 skipped / 5 failed / 62 errors with unavailable `sphinx` as the only failure/error family.

Do not close `SEC-P1-42`: compromise before host startup, source-page falsification/clickjacking/removal and denial of service remain architectural parent risks. Do not close `SEC-P1-43`: actual deployment headers/CORS remain external evidence.

Prepackage freeze: 69/69 two-root Python compile, all browser/bridge/Worker JS syntax GREEN, Wrangler/policy TOML + supply-chain/release-subject + maintenance GREEN, controlled diff 5 added / 20 modified / 0 removed = 25 paths, 293 files under the two canonical roots and zero cache/bytecode. Candidate → prefinal → final exact-byte cycle remains pending.

## Candidate packaged-byte acceptance

**GREEN — independently re-extracted candidate bytes.** Exact candidate bytes reproduced B42 Python **11/11**, B42 browser **35/35**, registered Node **46/46**, mutation/logging/privacy **236/236**, runnable non-Sphinx **766 passed, 3 skipped**, **69-file** two-root Python compile, browser/isolation-host/isolated-frame/Worker JavaScript syntax, Wrangler `invocation_logs=false`, supply-chain + release-evidence TOML, offline supply-chain/release subjects and maintenance drift GREEN. Sphinx-inclusive remained **1232 passed, 3 skipped, 5 failed, 62 errors**, with every non-green case confined to the established unavailable-`sphinx` `test___init__.py` boundary. Candidate hygiene: **293 files**, exact two roots, ZIP integrity GREEN, zero packaged cache/bytecode.

## Metadata-bearing prefinal packaged-byte acceptance

**GREEN — independently re-extracted prefinal bytes.** Exact prefinal bytes reproduced B42 Python **11/11**, B42 browser **35/35**, registered Node **46/46**, mutation/logging/privacy **236/236**, runnable non-Sphinx **766 passed, 3 skipped**, **69-file** two-root Python compile, browser/isolation-host/isolated-frame/Worker JavaScript syntax, Wrangler `invocation_logs=false`, supply-chain + release-evidence TOML, offline supply-chain/release subjects and maintenance drift GREEN. Sphinx-inclusive remained **1232 passed, 3 skipped, 5 failed, 62 errors**, all unavailable-`sphinx` only. Archive hygiene remained **293 files**, exact two roots, ZIP integrity GREEN and zero packaged cache/bytecode. Final delivery is rebuilt once from this metadata freeze; exact-final-byte acceptance and SHA-256 remain external to avoid self-reference.
