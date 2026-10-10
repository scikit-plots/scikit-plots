# B36 — Dataset contribution UX and conversation record

Status: **IMPLEMENTED — Run 17 working tree; final package verification pending**

## Trigger

The content-bearing contribution action was hidden inside **Share → More actions**
while the per-answer feedback popup used the ambiguous label **Save to dataset**.
That information architecture conflated three different purposes:

1. privacy-minimal rating telemetry;
2. human-facing Share/export;
3. explicit content donation for dataset review and possible training/evaluation.

The server lifecycle was already quarantine/review/capability based, but the
browser did not expose that purpose boundary clearly or make whole-conversation
contribution a first-class reviewed operation.

## Product / purpose contract

- **Feedback telemetry** is per-answer rating/event mechanics only. It does not
  contain question, answer, note, model, page, or conversation content.
- **Share** is export/human-sharing only and owns no dataset contribution button,
  controller, delete token, or training endpoint action.
- **Dataset contribution** is a dedicated first-class sheet. Main-panel,
  per-answer, and Endpoint Configuration shortcuts all converge on the same
  controller.
- The feedback popup says **Send rating telemetry** and exposes a separate
  **Contribute this Q&A…** action. Enabling telemetry never implies contribution
  consent.


## Telemetry permission hardening

Run 17 also closes a purpose-boundary residual discovered during implementation:
local rating was already network-off by default, but the server accepted feedback
requests without explicit consent evidence and the public `ai-assistant-feedback`
DOM event exposed the full local feedback tuple to page listeners.

Current contract:

- local ratings work with zero network telemetry;
- historical boolean telemetry preferences do not migrate authority;
- only a current structured browser consent record (`1.0.0`) enables telemetry;
- rating/retraction helpers self-gate even if a caller forgets the outer UI check;
- HF and Worker `/v1/feedback` require schema 4 plus the current consent marker,
  version, and positive grant timestamp before rate-limit/storage work;
- the public feedback DOM event is rating-only and content-free;
- turning telemetry off stops future network sends and does not claim remote
  erasure of earlier accepted telemetry;
- operator persistence policy never substitutes for reader permission.

This consent marker is purpose evidence for the bundled application, not
cryptographic proof that a particular human clicked the switch.

## Contribution scopes

The canonical **Contribute to dataset** sheet supports:

- **This Q&A** — one selected Q&A record;
- **Rated answers** — only explicitly rated Q&A records;
- **Whole conversation** — exactly one ordered conversation record.

Every scope uses the same flow:

```text
select scope
  -> build exact schema-v4 JSON
  -> Inspect JSON
  -> local privacy preflight over that same object
  -> explicit consent 2.0.0
  -> POST /v1/contribute
  -> quarantine receipt
  -> delete pending / withdraw training use
```

Quick access means quick entry into this review flow, never one-click silent
content submission.

## Schema v4 contract

Current browser contributions use:

- `schemaVersion = 4`;
- `consentVersion = "2.0.0"`;
- `recordType = "qa" | "conversation"`.

A conversation record contains one ordered `messages[]` array. Only `user` and
`assistant` messages are eligible. Runtime/error/system/tool material is not
reinterpreted as training dialogue. Assistant messages carry their own
client-reported model evidence and optional existing rating/note metadata.

The contribution envelope does not add a stable browser session/conversation
identifier, feedback event chain, Share capability, receipt-delete capability,
or endpoint credential. A sanitized page reference remains optional context.

Legacy schema v2/v3 contribution clients remain accepted with historical consent
`1.0.0` for compatibility. That old consent is not accepted for schema v4's
broader conversation-record contract.

## Lifecycle contract preserved

B36 changes intake representation and browser ownership, not the hardened
receipt state machine:

- raw accepted content enters `trainingStatus="quarantined"`;
- only independent review authority may promote it to `eligible`;
- pending deletion removes active review-ledger content without forensic-erasure
  overclaim;
- post-promotion withdrawal suppresses ordinary training output and attempts
  provider current-view removal without claiming provider history/backups erased;
- Run 16 shared Redis authority, `promotion_uncertain`, and monotonic withdrawal
  remain unchanged.

## UI information architecture

- Main conversation actions: **Share** and **Contribute** are peers.
- Feedback popup: **Send rating telemetry** / **Contribute this Q&A…** /
  **Detailed feedback** are separate controls.
- Endpoint Configuration: **Runtime & Data** contains separate **Feedback
  telemetry** and **Dataset contributions** blocks plus an **Open contribution
  sheet** shortcut.
- Usage Policy explicitly states that telemetry and dataset contribution are
  different operations.
- User-facing endpoint copy says **Dataset contribution endpoint**; internal key
  `training` remains for backward-compatible configuration/profile schema.

## Verification — working tree

- contribution + feedback/lifecycle focused Python plane: **42 passed**;
- dedicated contribution source/UX harness: **43 passed**;
- dedicated contribution mini-DOM workflow: **27 passed**;
- executable telemetry-consent browser harness: **42 passed**;
- feedback/contribution privacy source harness: **40 passed**;
- JavaScript harness registry: **42 passed**;
- mutation + logging/privacy positive controls: **212 passed**;
- complete runnable non-Sphinx tree: **704 passed, 3 skipped**;
- Sphinx-inclusive boundary: **1170 passed, 3 skipped, 5 failed, 62 errors**, all failures/errors confined to missing `sphinx`;
- proxy deployment version: **6.6.1**;
- controlled diff from Run 16.2.7: **6 added, 32 modified, 0 removed = 38 paths**;
- browser source size at candidate freeze: **32,322 JS lines / 17,422 CSS lines**.

Final ZIP extraction evidence is recorded in `VERIFICATION.md` after the clean archive cycle; SHA-256 remains external to avoid a self-referential archive.


## Candidate package acceptance

**GREEN — clean candidate extraction.** The exact candidate bytes reproduced the 43/27/42/40 browser gates, 42 focused Python tests, 42 Node harnesses, 212 mutation/privacy checks, 704 passed + 3 skipped runnable tree, 54-file Python compile, maintenance GREEN, and the unchanged missing-Sphinx boundary of 1170/3/5/62. Candidate membership is 249 files with exactly two roots and no packaged cache/bytecode contamination.


## Metadata-bearing prefinal acceptance

**GREEN.** Exact prefinal extraction reproduces all Run-17 focused gates, 704 passed + 3 skipped runnable tree, maintenance GREEN, and the unchanged missing-Sphinx boundary 1170/3/5/62. Final delivery bytes are rebuilt after this metadata update and rechecked independently.


## Final delivery acceptance

**GREEN — exact final delivery archive independently re-extracted.** The delivered bytes reproduce the 43/27/42/40 browser gates, 42 focused Python tests, 42 Node harnesses, 212 mutation/privacy checks, 704 passed + 3 skipped runnable tree, 54-file compile, JS/Worker syntax, maintenance GREEN, Sphinx boundary 1170/3/5/62, and 249-file two-root cache-clean archive hygiene. SHA-256 remains external.
