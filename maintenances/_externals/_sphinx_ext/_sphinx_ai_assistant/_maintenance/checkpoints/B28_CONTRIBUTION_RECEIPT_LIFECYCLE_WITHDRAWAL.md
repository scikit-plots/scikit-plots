# B28 — Contribution Receipt Lifecycle & Training Withdrawal

Status: **COMPLETE — Run 12**
Type: **bounded maintenance/change campaign checkpoint**
Subsystem: **_sphinx_ai_assistant**

## Objective

Close the single-instance contribution receipt lifecycle gap without overstating deletion guarantees: make receipt state restart-durable when explicitly configured, prevent duplicate promotion races, preserve user management authority after promotion, and make withdrawal enforceable in ordinary training dataset construction.

## Source anchor

- Input archive: `scikitplot__sphinx_ai_assistant_b18_run11_cors_identity_resource_parity_overlay.zip`
- SHA-256: `df2ac5e3410e466e8e35ee4aaccb055d782e01d7aadaacb305bd7ad7947f85d3`
- Review date: 2026-08-29

## Landed controls

1. **Receipt ledger abstraction** — contribution lifecycle authority moves behind `_contribution_ledger.py`. Compatibility `memory` remains bounded/process-local; optional `sqlite` is local transactional/restart-durable.
2. **Durability fail-closed switch** — `CONTRIBUTION_REQUIRE_DURABLE=true` rejects intake unless the configured ledger reports durable storage. SQLite is deliberately not described as a shared multi-replica authority.
3. **Capability hashing** — the receipt delete/withdraw capability remains client-held; the ledger stores its digest, not the token. Raw Q&A is cleared from ledger state after promotion.
4. **Atomic lifecycle claims** — promotion is `quarantined -> promoting -> eligible`; withdrawal is `eligible -> withdrawing -> withdrawn`. A second concurrent promotion/withdrawal cannot independently persist the same lifecycle transition.
5. **Content-free lifecycle status** — authenticated `GET /v1/contribute/{receipt}` reports lifecycle only and never returns raw contribution content.
6. **Truthful pending deletion** — deleting a still-quarantined receipt removes its content from the active review ledger. It does not claim forensic erasure from database pages/WAL, storage media, backups, provider logs, or infrastructure snapshots.
7. **Post-promotion withdrawal** — the same user receipt capability persists privacy-minimal `withdraw` tombstones keyed only by server-owned contribution dedup keys. Ordinary dataset construction applies last-write-wins and excludes both the withdrawn eligible row and the tombstone from training output.
8. **Current-view provider cleanup** — withdrawal best-effort deletes the promoted record from the current Hugging Face/GitHub/GitLab/Bitbucket view and suppresses stale in-process mirror retries for the withdrawn record. Versioned repository history is explicitly outside this guarantee.
9. **SQLite deletion defense in depth** — `secure_delete=ON`, bounded receipt cardinality, and WAL checkpoint/truncation are used around sensitive lifecycle clearing. These controls improve local residue handling but are not evidence of forensic/global physical erasure.
10. **UI semantics** — one receipt action truthfully changes from pending removal to training withdrawal after promotion; browser copy explicitly distinguishes current-view removal from history/backup erasure.
11. **Crash-state reclamation** — SQLite startup reclaims stale `promoting`/`withdrawing` ownership from a prior process so receipts cannot remain permanently busy after a crash. Promotion replays use the receipt's original intake time for a stable provider path.
12. **Replay-stable withdrawal** — withdrawal tombstones accept receipt-stable timestamps/path time so retry/restart does not create a growing family of logically identical withdrawal artifacts.
13. **Terminal tombstone retention** — deleted/expired/withdrawn lifecycle tombstones are retained only for a bounded status window, preventing terminal history from permanently consuming receipt capacity.
14. **Withdrawal-vs-mirror race suppression** — provider retry suppression is rechecked inside the per-target write lock so a retry that waited behind withdrawal cannot resurrect an eligible current-view file after deletion.

## Verification

- `tests/test_contribution_lifecycle_control_plane.py` — memory atomicity, SQLite reconstruction, status authorization, durable-required fail-closed behavior, withdrawal tombstone/training exclusion, idempotent withdrawal, and truthful UI contract.
- `tests/test_feedback_contribution_privacy.py` / `.mjs` — privacy/provenance and receipt-capability semantics.
- `tests/test_storage_multisource.py` — current-view removal across Hugging Face, GitHub, GitLab, and Bitbucket plus Docker module packaging.
- `tests/test_deduplicate_multisource.py` — withdrawal tombstones participate in LWW but are never emitted into training output.
- `tests/test_mutation.py` — contribution UI mutants prove post-promotion management and non-erasure wording cannot silently regress.
- broad non-Sphinx, syntax, maintenance, and packaged-copy gates are recorded in `VERIFICATION.md`.

## Contracts

- `AIA-C20 PrivacyDataLifecycle = PARTIAL` — strengthened: single-instance restart durability + enforceable training withdrawal now land, while shared multi-replica authority and provider-history/global erasure remain deployment residuals.
- `AIA-C26 ContributionReceiptLifecycle = HOLDS` at the bundled single-instance ledger/dataset boundary.
- `SEC-P0-26 = PARTIAL` — local transactional/restart durability closes; shared multi-replica transactional authority remains.
- `SEC-P0-27 = PARTIAL` — training withdrawal + current-view cleanup lands; provider history/backups/global erasure remain unproved.
- `SEC-P0-32 = CLOSED` — duplicate promotion race eliminated by atomic lifecycle claim.
- `SEC-P0-33 = CLOSED` — pending-delete physical-erasure overclaim removed from contract/UI/docs.

## Residuals

- Use an external shared transactional control plane before horizontally scaled contribution collection; bundled SQLite is one-node authority only.
- Provider-complete deletion from versioned history, replicas, backups, caches, and infrastructure remains unproved and must never be promised as global erasure.
- Strict distributed rate accounting remains `SEC-P0-31`.
- Run 13/B29 closes path-capability logging for current generated Share transport; legacy capability-bearing routes remain the bounded `SEC-P0-15` migration residual.
- Representative-browser E2E and canonical Sphinx-enabled release verification remain outstanding.

## Rollback

Revert ledger, route, dataset-tombstone, provider-removal, browser, docs, and tests together. Do not restore a post-promotion receipt dead end, a read-then-write promotion race, or wording that equates active-ledger/current-view removal with global physical erasure.


## Run 16 supersession note

B32 now supplies the optional shared Redis transactional receipt authority that B28 intentionally left as a deployment residual. B28's SQLite statements remain correct for the local backend. External Redis persistence/backup durability and provider-complete erasure remain separate, unclosed guarantees.
