# Fresh-chat handoff — B61 review continuity & withdrawal recovery

Use the latest B61 artifact as the source anchor.

## Current behavior

- `CONTRIBUTION_REVIEW_MODE=provider-pr` uses one native provider review per receipt.
- unchanged re-submit is a no-op; changed reviewed content updates the same PR/MR.
- normal provider status/update uses persisted review IDs, not review-list scanning.
- contribution management actions reappear when the sheet is reopened in the same tab.
- users can save a private JSON receipt or copy a private `aicm2.…` withdrawal code.
- private receipt/code import is available under **Recover withdrawal access**.
- schema-v2 private receipt import fails closed when its `serviceHint` does not match the
  currently configured contribution endpoint.
- users can separately copy a non-secret support reference containing receipt,
  provider/review number and stable `ct_<review-key>.jsonl` path.
- provider review URL and participant management secret are not exposed in that support
  reference.

## Important residual

Portable participant authority does not imply infinite server retention. Pending review
lifecycle expiry or a lost process-local ledger can make an old capability unresolvable;
maintainer support via the non-secret review/path reference is the fallback. Prefer
SQLite/Redis for production receipt durability.
