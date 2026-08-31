# B54 — Provider-Native Contribution Review

Status: **WORKING-TREE GREEN — final exact-byte replay pending**

## Scope

Replace the operator-invisible process-local quarantine payload as the only review surface with an optional provider-native review workflow. The configured **Primary** storage target owns review authority; the configured canonical branch remains the only training-eligible source of truth.

## Configuration

- `CONTRIBUTION_REVIEW_MODE=ledger` — compatibility/default path; existing lifecycle is unchanged.
- `CONTRIBUTION_REVIEW_MODE=provider-pr` — create/recover one native provider review per contribution.
- `CONTRIBUTION_REVIEW_TOKEN` remains an optional server-side automation/fallback capability for `/promote`; ordinary reviewers can merge or close in the provider UI.

## Provider mapping

| Primary provider | Quarantine/review surface | Accept | Reject |
| --- | --- | --- | --- |
| Hugging Face | Hub Pull Request (`refs/pr/*`) | merge PR | close PR |
| GitHub | opaque branch + Pull Request | merge PR | close PR |
| GitLab | opaque branch + Merge Request | merge MR | close MR |
| Bitbucket | opaque branch + Pull Request | merge PR | decline PR |

The branch/title key is receipt-derived SHA-256 material only; contribution text, page text, identity, model text and notes never enter branch names or review titles.

## Lifecycle

`provider-pr` makes review-provider state authoritative for training eligibility:

- open/draft review -> internal `quarantined`, user-facing **IN REVIEW**, not training eligible;
- provider merge -> monotonically ratchets the receipt to `eligible` on the next status/withdrawal observation;
- provider close/decline -> user-facing **NOT ACCEPTED**; it never becomes training eligible;
- participant delete while pending -> closes/declines the provider review and clears the active ledger payload;
- participant withdrawal after merge -> existing withdrawal tombstone/current-view-removal semantics remain authoritative.

The exact bytes placed in review are already normalized with `trainingStatus="eligible"`; those bytes become canonical only if the review is merged. The mutable ledger retains the quarantined copy until acceptance/deletion/expiry so management capability semantics remain compatible.

## Compatibility boundaries

- Existing `ledger` deployments are unchanged by default.
- Feedback persistence remains direct and is not routed through code review.
- Only the configured `role="primary"` target opens a native review; mirrors are replicas, not competing approval authorities.
- Native review does not imply physical erasure from Git/provider history, caches or backups after close/delete/withdrawal.
- Provider-specific permissions and protected-branch policy remain deployment facts; failures are bounded/classified and do not leak provider response bodies.
- If the local/shared contribution ledger is process-local, the remote review payload is durable in provider storage but receipt/delete-capability coordination still requires Redis/SQLite durability according to the existing B32 policy.

## Working-tree verification

- B54 provider-review tests: **13/13**.
- focused provider/storage/contribution/Node plane: **118 passed**.
- registered Node harnesses: **50/50**.
- runnable non-Sphinx: **824 passed, 3 skipped**.
- Sphinx-inclusive: **1290 passed, 3 skipped, 5 failed, 62 errors**; all 67 non-green cases are the established missing-`sphinx` family in `test___init__.py`.
- complete two-root Python compile: **72/72**.
- browser/isolation-host/isolated-frame/Worker JS syntax: **GREEN**.
- Wrangler TOML and `invocation_logs=false`: **GREEN**.
- supply-chain policy / release-evidence policy / offline supply-chain verifier: **GREEN**.
- proxy: **7.3.0**.
- runtime-source SHA-256: `12480c65bed724a6ae87ebb8ee952fb559820aa3963a98b262e9ceeea0ab8912`.

- clean source freeze: **308 files**, exactly `scikitplot/` + `maintenances/`, zero cache/bytecode;
- exact Run 34 → Run 35 source diff: **3 added · 15 modified · 0 removed = 18 paths**.

Exact final ZIP hash is recorded only after independent extraction and final-byte replay.
