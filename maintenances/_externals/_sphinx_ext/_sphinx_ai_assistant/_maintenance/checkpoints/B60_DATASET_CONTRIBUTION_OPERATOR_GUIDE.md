# B60 — Dataset Contribution & Operator Guide

Status: **GREEN**

## Scope

Make the dataset contribution lifecycle understandable without reading proxy source code.
This run is documentation-first and keeps runtime behavior unchanged.

## Documentation architecture

A new top-level guide now owns the human workflow:

- `DATASET_CONTRIBUTION_GUIDE.md`

It explains:

- reader contribution scopes and exact-JSON review;
- explicit consent and quarantine/in-review semantics;
- private management receipt behavior;
- `provider-pr` versus `ledger` modes;
- Primary-versus-Mirror authority;
- Hugging Face, GitHub, GitLab, and Bitbucket native review mapping;
- approval, rejection, pending deletion, post-merge withdrawal, provider failure,
  and proxy-restart scenarios;
- durable receipt authority (`sqlite` / `redis`);
- Variables versus Secrets;
- deployment recipes and troubleshooting.

The existing deep storage guide remains authoritative for provider topology,
migration, deduplication, and dataset assembly:

- `_hf_spaces_proxy/DATASET_COLLECTION_GUIDANCE.md`

Its opening lifecycle policy and contribution test section were updated so they no
longer describe review-token promotion as the only approval path after B54.

## README learning path

The top-level `README.md` now starts with a goal-oriented navigation table and
incremental deployment levels. It explicitly distinguishes static features, stub
panel use, live proxy mode, and dataset contribution.

The README also adds:

- provider-native dataset review overview;
- Primary review authority and mirror semantics;
- management receipt actions;
- receipt durability warning for process-local memory;
- Variables/Secrets summary;
- minimal `RECORD_STORAGE_TARGETS` example;
- contribution troubleshooting.

The HF Space proxy README links to the top-level contribution guide for human
workflow and to `DATASET_COLLECTION_GUIDANCE.md` for deep storage operations.

## Runtime contract clarified

No runtime behavior changed. The documented contract matches B54/B59:

- `CONTRIBUTION_REVIEW_MODE=provider-pr` opens a native review on the Primary;
- canonical branch is the training-eligibility boundary;
- manual merge is detected during later status/management synchronization;
- provider close/decline remains training-ineligible and is surfaced by the browser
  as **NOT ACCEPTED** through `reviewStatus`;
- Mirrors do not independently approve/reject;
- external Primary merges are not synchronously replicated to Mirrors;
- provider PR durability does not replace durable receipt-lifecycle storage.

## Verification

- documentation links resolve inside the packaged source tree;
- all Markdown code fences are balanced;
- no token values or credential examples are introduced;
- existing dataset/storage documentation contract tests remain green;
- runtime source delta: none.
