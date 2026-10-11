# Fresh-chat handoff — Run 35 / B54

Run 35 adds an opt-in provider-native contribution review workflow across every storage provider currently supported by the extension: Hugging Face, GitHub, GitLab and Bitbucket.

Core invariants:

1. `CONTRIBUTION_REVIEW_MODE=ledger` preserves the historical workflow; `provider-pr` opts into native code review.
2. Only the configured Primary target owns review authority. The canonical target branch is the only training-eligible source of truth.
3. Hugging Face uses Hub PRs; GitHub/Bitbucket use Pull Requests; GitLab uses Merge Requests.
4. Review branch/title identifiers are opaque receipt-derived hashes and contain no contribution/user text.
5. Merge is acceptance; close/decline is rejection. A manual merge in the provider UI is detected and ratchets local lifecycle state to `eligible` before later participant management.
6. Pending participant delete closes the provider review before clearing the mutable ledger. Post-merge withdrawal retains the existing tombstone/current-view-removal contract and never promises Git-history erasure.
7. Provider response bodies remain bounded/private under B44. Review APIs do not weaken token, URL-authority, logging or response-size boundaries.
8. Mirrors remain replication targets and never become independent approval authorities.

Proxy version: **7.3.0**.
Runtime-source SHA-256: `12480c65bed724a6ae87ebb8ee952fb559820aa3963a98b262e9ceeea0ab8912`.

Before delivery, purge cache/bytecode, package exactly `scikitplot/` + `maintenances/`, independently extract the final ZIP, replay B54/provider/storage/Node, runnable non-Sphinx, compile/syntax/TOML/supply-chain/maintenance and the known Sphinx boundary, then compute the immutable ZIP SHA-256 without rewriting it.
