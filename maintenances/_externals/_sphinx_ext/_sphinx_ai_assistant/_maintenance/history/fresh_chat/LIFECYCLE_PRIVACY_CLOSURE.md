# Fresh-chat handoff — Run 18 / B37 lifecycle and privacy closure

Use this handoff only after reading `STATE.json`, `RULESET.md`, `TRACKER.json`,
`SECURITY_FINDINGS_INDEX.md`, and checkpoint
`checkpoints/B37_LIFECYCLE_PRIVACY_CLOSURE.md`.

## Exact source anchor

Run 18 started from the exact Run 17 delivery archive:

- archive: `scikitplot__sphinx_ai_assistant_b36_run17_dataset_contribution_ux_telemetry_consent_overlay(1).zip`
- SHA-256: `9539ff404725f09a957b9fe0e11e0c9b5b759d9058b797a6bf04c4c3e944abe9`
- roots: exactly `scikitplot/` + `maintenances/`
- members: 249 files

## Run 18 architecture

1. Share/contribution CREATEs use a pre-request operation envelope.
2. Raw revoke/delete capabilities remain browser-side during current CREATE;
   the server receives only their SHA-256 digests.
3. Exact operation replay is idempotent; payload or capability-digest mismatch
   is conflict; ambiguous create result is `outcome_unknown` and retries the
   same envelope.
4. Contribution management capability has explicit Save/Import receipt UX and
   is never silently persisted in localStorage.
5. Public feedback DOM integration has a separate versioned permission from
   network feedback telemetry; both default Off.
6. Reviewed contribution limits are enforced before consent/storage and
   over-limit data is rejected rather than silently truncated.
7. Transcript restore is explicit per-tab opt-in and bounded; microphone device
   ID is session-scoped.
8. Endpoint diagnostics strip query/fragment/userinfo and secret-like query
   parameter names are rejected.
9. HF Global Share uses memory/SQLite/Redis `ShareStore` with truthful
   durability/shared semantics and fail-closed required modes.
10. Redis Share expiry cleanup is atomic. Worker KV remains explicitly
    eventual-consistency storage, not global transactional authority.

## Working-tree verification already reached

- B37 Python: 6 passed
- B37 browser/source: 56/56
- focused compatibility/mutation: 229 passed
- runnable non-Sphinx: 711 passed, 3 skipped
- browser/Worker syntax: GREEN
- proxy version: 6.7.0

Do not call Run 18 final until the clean archive has been rebuilt, independently
extracted, all release gates rerun against those exact bytes, cache/bytecode
absence verified, controlled diff/member count recorded, and the final SHA-256
computed externally.

## Deliberate next residuals

- SEC-P0-10 supply-chain/container hardening: digest-pinned base image,
  hash-locked dependencies, non-root/rootless/read-only runtime, SBOM/CVE scan.
- provider history/backups/cache/global erasure proof.
- deployment evidence for Redis TLS/ACL/persistence/replication and WAF/body
  logging.
- representative Playwright/WebDriver E2E.
- stronger separate-origin browser isolation if arbitrary documentation-origin
  script compromise is in scope.
- local Sphinx-inclusive suite remains environment-blocked until `sphinx` is
  installed.
