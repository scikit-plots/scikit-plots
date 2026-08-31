# B32 — Shared Contribution Receipt Authority

Status: **COMPLETE AT BUNDLED SHARED-COORDINATION BOUNDARY — Run 16; external Redis durability/production activation remain deployment-conditional**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Extend the Run 12 contribution receipt lifecycle from a truthful single-instance SQLite control plane to an optional shared atomic authority for horizontally scaled proxy replicas, without pretending a Redis lease can fence external Git/provider side effects or that shared coordination proves Redis persistence/backup durability.

## Source anchor

- Input archive: `scikitplot__sphinx_ai_assistant_b18_run15_distributed_rate_limit_authority_overlay.zip`
- SHA-256: `fe3b54b59c199ef47b5ba75465d66a27d352b89c130eec44b6efec77a6ea1cf3`
- Review date: 2026-08-29

## Landed controls

1. **Shared Redis receipt ledger** — `CONTRIBUTION_LEDGER_BACKEND=redis` provides one atomic receipt lifecycle domain for all replicas pointed at the same intended Redis consistency domain. Create/capacity, promotion claim/finalization, withdrawal claim/finalization and active-index maintenance are executed through bounded Lua transactions.
2. **Cluster-safe key ownership** — all multi-key ledger keys share the `{contribution}` Redis hash tag so the transaction key set belongs to one cluster slot; pending receipt count remains capped and pending-byte admission recomputes over that bounded active set rather than trusting a drift-prone cross-key byte counter.
3. **Private shared receipt identifiers** — raw receipt capability IDs never become Redis keys/index members. A dedicated `CONTRIBUTION_LEDGER_KEY_SECRET` HMAC-SHA256 pseudonymizes the external identifier before shared-state use. Delete/review capabilities remain separate authorities.
4. **Fail-closed shared-required mode** — `CONTRIBUTION_REQUIRE_SHARED=true` rejects contribution intake when the selected ledger is not ready + shared + authoritative. Redis configuration/runtime failures do not silently fall back to memory/SQLite while retaining a shared-authority claim.
5. **Atomic operation claims** — Redis stores only a claim digest and lease metadata. The raw operation claim exists only in the active process and must match before finalization/failure transitions; stale workers cannot finalize a newer claim.
6. **No unsafe promotion lease takeover** — an expired `promoting` lease becomes `promotion_uncertain` / reconciliation-required. It is never automatically reassigned, because Redis cannot fence a paused worker after that worker has already issued an external Git/HF/GitLab/Bitbucket mutation.
7. **Ambiguous provider writes fail safe** — transport/timeouts and other mutation outcomes where the provider may have accepted the write are normalized as transient/ambiguous storage errors and transition promotion to `promotion_uncertain`, not back to ordinary quarantine/re-promotion.
8. **Privacy-safe uncertainty resolution** — a participant management capability can drive `promotion_uncertain` toward a withdrawal tombstone. Withdrawal is monotonic: an expired withdrawal claim can be retried toward `withdrawn`, but uncertainty is never restored to training eligibility.
9. **Direct-ledger parity** — even direct Redis `delete_pending()` cannot reinterpret an expired promotion claim as quarantined; it enters reconciliation-required uncertainty just like the HTTP lifecycle path.
10. **Durability claim split** — Redis reports shared/authoritative coordination but deliberately does **not** self-certify persistence durability. SQLite remains the bundled single-instance restart-durable option; external Redis persistence/AOF/RDB/replication/backup policy is deployment evidence.

## Security boundary

Shared receipt authority and external-provider side effects form a distributed saga, not one database transaction. Redis can serialize receipt state transitions across replicas, but it cannot revoke a Git commit/API mutation already emitted by a worker that later pauses or loses the response. Therefore uncertain promotion outcomes are non-promotable and must be operationally reconciled or driven toward withdrawal. This is fail-safe privacy behavior, not automatic exactly-once provider mutation.

`CONTRIBUTION_REQUIRE_SHARED=true` is the explicit horizontal-deployment gate. A deployment that leaves the memory or SQLite backend active must continue to describe its receipt authority as process-local/single-instance respectively.

## Verification

- `tests/test_run16_shared_contribution_authority.py` — **16 passed**: two-replica atomic promotion ownership, HMAC-only Redis keyspace, shared manifest truth, fail-closed shared-required policy, promotion/withdrawal uncertainty handling, direct-delete uncertainty parity, transient provider mutation path, SQLite uncertainty restart behavior, and positive-control mutants.
- Focused authority/privacy/storage/request-boundary plane — **81 passed**.
- Client mutation catalogue — **193 passed**; logging/privacy mutation catalogue — **7 passed**.
- Node/JS harness wrappers — **38 passed**.
- Complete runnable non-Sphinx working tree — **661 passed / 3 skipped**.
- Syntax/config: browser JS + Worker JS GREEN; modified proxy/model Python compile GREEN; Wrangler TOML authoritative-rate/invocation-log policy parses GREEN; maintenance drift GREEN.
- Sphinx-inclusive working-tree boundary: **1127 passed / 3 skipped / 5 failed / 62 errors**; every failure/error remains confined to `test___init__.py` and terminates on missing `sphinx`.
- Diff from Run 15 input: **24 changed paths** (2 added, 22 modified, 0 removed).
- Packaged-byte evidence is recorded after the final acceptance cycle below.

## Contracts

- `AIA-C26 ContributionReceiptLifecycle = HOLDS`: Run 12 local lifecycle invariants remain; Run 16 extends shared coordination without weakening withdrawal/deletion semantics.
- `AIA-C30 SharedContributionReceiptAuthority = HOLDS_WHEN_SHARED_MODE_REQUIRED`: all participating replicas consult one atomic Redis receipt domain, receipt IDs are HMAC-pseudonymized, and required shared-backend failure is fail-closed.
- `AIA-C20 PrivacyDataLifecycle = PARTIAL`: shared transactional authority now lands, while provider-history/global erasure and external control-plane durability remain separate guarantees.
- `AIA-023 / SEC-P0-26 = PARTIAL (RUN 16 SHARED AUTHORITY LANDED; EXTERNAL REDIS DURABILITY/ACTIVATION EVIDENCE REMAINS)`.
- `SEC-P0-34 = CLOSED — Run 16`: ambiguous provider mutation outcomes no longer reopen ordinary promotion eligibility.

## Residuals

- Horizontal production must provision one intended Redis consistency domain, dedicated contribution HMAC secret, TLS/ACL/network controls, and `CONTRIBUTION_REQUIRE_SHARED=true`; repository code cannot prove those resources are actually deployed.
- Redis persistence/replication/backup/recovery semantics must be verified separately before claiming cross-crash durable receipt authority for that external domain.
- Provider-history/backups/caches/replicas remain outside current-view removal/global erasure guarantees.
- Real Redis-server/cluster integration remains deployment/canonical-environment evidence; the bundled regression plane pins Lua/source contracts and executes shared-state semantics with an independent semantic client in this environment.
- Legacy Share route drain/removal, representative-browser E2E and canonical Sphinx-enabled release verification remain independent residuals.

## Rollback

Never resolve an ambiguous/expired promotion claim by silently returning it to ordinary `quarantined` eligibility. If shared authority must be disabled, either fail closed or explicitly return to a documented single-instance mode; do not retain a shared-authority claim while falling back to local state.


## Candidate package acceptance

- Candidate archive: **235 files**, exact `scikitplot/` + `maintenances/` roots, zero cache/bytecode contamination, ZIP integrity GREEN.
- Fresh candidate extraction: **81 focused**, **38 Node harness**, **193 client mutation**, **7 logging/privacy mutation**, **661 passed / 3 skipped** complete non-Sphinx; syntax/config/maintenance GREEN.
- Candidate Sphinx-inclusive boundary: **1127 passed / 3 skipped / 5 failed / 62 errors**, all confined to missing-`sphinx` `test___init__.py` surface.
- Final metadata-bearing archive must be rebuilt and independently re-extracted before delivery.


## Final package acceptance

**GREEN — final delivery archive independently re-extracted.** The self-consistent metadata-bearing delivery archive reproduced **81 focused**, **38 Node harness**, **193 client mutation**, **7 logging/privacy mutation**, **661 passed / 3 skipped** complete non-Sphinx, syntax/config/maintenance GREEN, and Sphinx-inclusive **1127 / 3 / 5 / 62** on the same missing-`sphinx` boundary. Archive hygiene remained **235 files**, exact `scikitplot/` + `maintenances/` roots, zero cache/bytecode contamination, and ZIP integrity GREEN. Its SHA-256 is recorded externally at delivery to avoid self-reference.
