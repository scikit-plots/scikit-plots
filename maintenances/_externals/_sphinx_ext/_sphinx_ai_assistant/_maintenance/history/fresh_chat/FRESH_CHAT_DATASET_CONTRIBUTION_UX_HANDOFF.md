> **Run 18 supersession notice (2026-08-30):** This Run-17 handoff remains
> historical evidence. Continue current work from
> `FRESH_CHAT_LIFECYCLE_PRIVACY_CLOSURE_HANDOFF.md` and B37.

# Fresh Chat Handoff — Dataset Contribution UX Redesign

Status: **ACTIVE DESIGN + IMPLEMENTATION HANDOFF**
Baseline: **Run 16.2.7 — Global Forget Unlock**
Next bounded implementation: **B36 / Run 17 — Dataset Contribution UX + Conversation Record**

## Why this exists

The existing contribution control is technically functional but is placed under
`Share conversation -> More actions -> Contribute rated answers…`. That hierarchy
is conceptually wrong and makes an explicit training/evaluation contribution
feature nearly undiscoverable. This file is the durable fresh-chat source of truth
if conversation context collapses.

## Product vocabulary and ownership

Three user jobs must remain distinct:

1. **Feedback** — rate an individual answer and optionally send privacy-minimal
   telemetry. Ordinary feedback is not a training-data channel.
2. **Share** — export or share a conversation with another person/device. Share
   must own no training/dataset contribution controls.
3. **Dataset contribution** — explicitly donate selected content for quarantine,
   review, and possible training/evaluation use. Content-bearing contribution
   always requires a separate privacy review and explicit consent.

Never collapse these concepts into one button, toggle, or endpoint description.

## Target UI architecture

### Main conversation surface

Provide a first-class quick action near Share:

`Share | Contribute`

`Contribute` opens one canonical **Contribute to dataset** slide-over sheet.
Do not hide the action under `More actions`.

### Feedback popup

Replace the misleading label:

`Save to dataset` -> `Send rating telemetry`

The toggle continues to control only `/v1/feedback` persistence. It must not send
question text, answer text, notes, model metadata, page metadata, or conversation
identity merely because the toggle is enabled.

Add a separate row:

`Contribute this Q&A…`

That row opens the same canonical contribution sheet with the `This Q&A` scope.

Keep `Detailed feedback` as a separate local/editing affordance.

### Contribution sheet scopes

The canonical sheet supports:

- **This Q&A** — context-sensitive when opened from one answer.
- **Rated answers** — existing explicit-content behavior, one record per rated Q&A.
- **Whole conversation** — one structured conversation JSON contribution record.

The sheet must display counts/size, optional user note, exact JSON inspection,
privacy preflight, consent, submission state, and lifecycle receipt actions.

## Contribution sheet sequence

1. Resolve the configured contribution endpoint.
2. Select contribution scope.
3. Build one normalized client payload from current in-memory transcript/feedback.
4. Show what will be included and what is structurally excluded.
5. Allow an optional bounded contribution note.
6. `Inspect JSON` shows the exact value that will be submitted.
7. Run the existing privacy preflight against that exact value.
8. Require current versioned explicit consent.
9. POST `/v1/contribute`.
10. Render quarantine receipt and management actions.
11. Preserve delete-pending / withdraw-training semantics from Runs 12/16.

Quick access means quick access to this review workflow — never silent submission.

## Security and privacy invariants — MUST PRESERVE

- Ordinary `/v1/feedback` remains privacy-minimal telemetry.
- Rating telemetry ON does **not** imply Q&A or conversation contribution.
- Contribution content leaves the browser only after an explicit contribution
  action, privacy preflight, and current-version consent.
- Share and Contribution are separate control planes.
- Bearer tokens, endpoint credentials, edit/revoke/delete capabilities, raw
  storage keys, raw query strings, raw fragments, and browser-private identifiers
  are never user-selectable contribution fields.
- Safe page metadata uses the existing `_sanitizePage` boundary.
- Contribution receipts/delete tokens remain capability material and must never
  enter ordinary telemetry, Share URLs, dataset rows, console logs, or persistent
  browser storage.
- New conversation records enter `quarantined`; they are not training-eligible
  until the existing authorised promotion path completes.
- Withdrawal remains monotonic even when provider/history erasure is uncertain.
- Provider history/backups/caches must not be described as physically erased
  unless independently evidenced.
- Ambiguous provider write outcomes retain the Run 16 reconciliation semantics.

## Server/schema direction

Extend the current contribution schema rather than creating a second endpoint.

### Existing Q&A record family

Keep existing explicitly consented Q&A records compatible.

Suggested normalized descriptor:

`recordType = "qa"`

### New conversation record family

One conversation contribution should remain one dataset record:

```json
{
  "recordType": "conversation",
  "messages": [
    {"role": "user", "content": "...", "ts": 0},
    {
      "role": "assistant",
      "content": "...",
      "ts": 0,
      "model": {"id": "...", "provider": "..."},
      "feedback": {"ratingValue": 1, "ratingSlug": "helpful", "note": "..."}
    }
  ],
  "note": "optional user contribution note"
}
```

Per-message model metadata is required because a user may change models during a
conversation. Do not assume one envelope model describes every assistant turn.

Conversation records must preserve ordered multi-turn structure and may include
ratings/notes attached to their assistant message. Errors should be explicitly
excluded or included by a named policy, not accidentally mixed with assistant
training content.

### Compatibility

- Existing schema v2/v3 Q&A contribution clients remain accepted.
- New clients use a bumped schema/consent version only after server + client land
  together.
- Stored old contribution rows remain readable through existing normalization.

## Endpoint Configuration redesign

Do not move the contribution form into Endpoint Configuration.

Rename the operator section from `Runtime & Feedback` toward `Runtime & Data` and
show two explicit blocks:

### Feedback telemetry

- `Send rating telemetry` toggle
- explanation that content stays local unless separately contributed

### Dataset contributions

- contribution endpoint/readiness
- quarantine/review readiness
- dataset/storage readiness
- `Open contribution sheet` quick action

User-facing label should be `Dataset contribution endpoint`, while the internal
profile key may remain `training` for compatibility.

## Usage Policy role

Usage Policy explains the flow but is not the primary action surface.

It should state clearly:

`Feedback telemetry != Dataset contribution`

and describe:

`Contribution -> quarantine -> review -> promotion -> possible training/evaluation`

plus delete-pending, withdrawal, and provider-history limitations.

## Suggested implementation order

1. Write this handoff before source edits. **DONE**
2. Extract reusable contribution payload/lifecycle helpers from the Share sheet.
3. Build the dedicated `Contribute to dataset` sheet using the existing rated-Q&A
   schema first.
4. Remove Share -> More actions contribution controls.
5. Add a first-class Contribute main/menu shortcut.
6. Rename feedback toggle and add `Contribute this Q&A…` entry.
7. Add `Rated answers` + `This Q&A` scopes against the existing server contract.
8. Add schema v4 conversation records + consent v2 only after the UI/controller is
   stable.
9. Add `Whole conversation` scope using one conversation record.
10. Update Endpoint Configuration and Usage Policy copy/readiness surface.
11. Add adversarial privacy, schema, DOM, lifecycle, and positive-control mutant
    tests.
12. Synchronize maintenance rules/contracts/findings/checkpoint/state.
13. Build a clean two-root overlay ZIP, re-extract, and retest exact packaged bytes.

## Acceptance criteria

- No `Contribute rated answers…` control remains in the Share sheet.
- Share source has no contribution form/controller ownership.
- Feedback popup says `Send rating telemetry`, not `Save to dataset`.
- A visible `Contribute` action opens the canonical contribution sheet.
- Per-answer `Contribute this Q&A…` opens that same sheet in answer scope.
- Rated-answer contribution continues to pass the existing quarantine/receipt
  lifecycle tests.
- Whole-conversation contribution is one normalized conversation JSON record.
- Rating telemetry alone cannot make content training-eligible or content-bearing.
- Exact submitted JSON is inspectable before submission and is the value checked by
  privacy preflight.
- The full non-Sphinx suite and mutation/privacy gates remain green.



## Telemetry non-negotiables

- 👍/👎 and detailed feedback remain fully usable locally when telemetry is Off.
- Network feedback is explicit opt-in, default Off, versioned, and fail-closed.
- Do not revive `ai-assistant-feedback-telemetry=true` or the older
  `ai-assistant-feedback-persist` boolean as authority. Current permission lives
  in the structured `ai-assistant-feedback-telemetry-consent` record.
- Every official feedback POST/retraction carries schema 4 plus
  `telemetryConsent=true`, consent version `1.0.0`, and grant timestamp. HF and
  Worker reject missing/stale/malformed consent before rate-limit/storage work.
- `_postFeedback` and `_postFeedbackRetract` must self-gate, not rely only on UI
  call sites. Turning Off means no future feedback network request.
- `ai-assistant-feedback` remains a compatibility DOM hook but is rating-only;
  never put Q&A, note, model, page, stable conversation IDs, tokens, or
  capabilities back into its public detail.
- `FEEDBACK_PERSIST_ENABLED=true` means only that the operator permits storage
  after a consented request arrives; it does not enable browser telemetry.
- Feedback telemetry permission and dataset contribution consent are separate
  purposes and must never be merged.
