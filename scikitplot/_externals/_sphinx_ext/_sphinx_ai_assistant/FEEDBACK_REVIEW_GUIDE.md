# Feedback review guide

This guide explains the feedback system used by the Sphinx AI Assistant: local
ratings, anonymous rating telemetry, maintainer feedback review, and dataset
contribution. These are deliberately separate control planes.

## One picture

```text
Reader rates one answer
        |
        +---------------------> Local rating state
        |                       always available
        |                       zero network required
        |
        +-- telemetry consent ON -------------------> /v1/feedback
        |                                             privacy-minimal metadata
        |                                             no Q&A or written note
        |
        +-- review + model-improvement consent ON -> /v1/feedback/review
                                                      exactly one Q&A
                                                      rating + optional note
                                                      normalized quality signal
                                                      provider PR/MR
                                                      training-eligible only after merge

Dataset contribution remains separate:
explicit scope + payload inspection + privacy review + contribution consent
        -> /v1/contribute
        -> provider review
        -> training-eligible only after authorized merge/promotion
```

The most important invariant is:

> A rating does not imply telemetry consent, review/model-improvement consent,
> or dataset-contribution consent. Training use of feedback is authorized only
> by the explicit review/model-improvement permission and a maintainer merge.

## Feedback & contribution workspace

The panel uses one presentation shell with three tabs:

```text
[ Feedback ] [ Dataset contribution ] [ Activity ]
```

They share visual language, not authority.

### Feedback

Feedback is always about exactly **one question and one assistant answer**.
It contains:

- one rating;
- quick or detailed rating mode;
- an optional written feedback note;
- originating model attribution (`provider` + concrete model name) and bounded page evidence when explicitly shared with maintainers.

Feedback records live under the configured `feedback/` storage path. An open
review is not training-eligible. If the reader granted the current explicit
review/model-improvement consent and a maintainer merges the PR/MR, that single
Q&A becomes training-eligible together with its quality signal.

The Feedback tab includes **Inspect feedback payload**, with local-only **Inspect JSON**,
**Copy JSON to clipboard**, and **Download JSON file** actions. This preview is the
exact client review payload. It includes the Q&A, rating/quality inputs, optional note,
and the model attribution belonging to the assistant turn that actually produced the
answer. Inspect/copy/download never submit anything. If originating model attribution
is unavailable, review sharing fails closed instead of creating an ambiguous training row.

### Dataset contribution

Dataset contribution can contain:

- one Q&A;
- rated answers; or
- a whole structured conversation.

It uses its own privacy preflight and contribution consent. Approved
contribution records live under `contributions/` and can become eligible only
through the configured review lifecycle.

### Activity

Activity shows the review receipts remembered by the current tab while keeping
feedback and dataset contribution visibly separate. Use **Manage** to return to
the appropriate control plane.

## Quick feedback

The quick thumbs buttons are optimized for the common path.

### Review sharing Off

```text
click Helpful
    -> local rating changes
    -> quick/detailed controls synchronize
    -> no feedback-review request
```

Anonymous telemetry is evaluated independently. If telemetry is also Off, the
click causes no feedback network request at all.

### Review sharing On

After the reader explicitly enables **Share feedback for review & model improvement**:

```text
first quick rating
    -> open one provider feedback review
    -> revision 1

same rating again
    -> no-op
    -> no new commit
    -> no new review

change quick rating
    -> update the same review
    -> revision 2
```

Turning review sharing On does **not** retroactively upload a rating that was
already local. The Feedback tab exposes **Share current feedback** for that
explicit transition.

## Detailed feedback

Detailed feedback uses the same logical feedback item as the quick buttons.
The textarea remains a local draft until the reader presses the feedback submit
button.

```text
quick Helpful
    -> review #27 revision 1

detailed form
rating = Mostly helpful
note = "Example needs one more edge case"
    -> Save feedback
    -> review #27 revision 2

change detailed rating again
    -> review #27 revision 3
```

Typing does not create provider commits. Only explicit feedback actions do.

## Synchronized rating controls

Quick and detailed controls are two views of one local feedback state.

```text
quick Helpful selected
        |
        +-- choose detailed Not helpful
                |
                +-- quick Helpful resets
                +-- detailed Not helpful becomes selected
```

Withdrawal clears the local feedback state and resets both surfaces.

## Provider-native feedback review

Set the Space Variable:

```text
FEEDBACK_REVIEW_MODE=provider-pr
```

The Primary storage provider owns the review authority:

| Primary provider | Review object | Accept | Reject |
|---|---|---|---|
| Hugging Face Dataset | Pull Request | Merge | Close |
| GitHub | Pull Request | Merge | Close |
| GitLab | Merge Request | Merge | Close |
| Bitbucket Cloud | Pull Request | Merge | Decline |

The feedback review uses a stable opaque identity and path such as:

```text
feedback/2026/09/01/fb_<opaque-review-key>.jsonl
```

User text is not placed in branch names or review titles.

A merged feedback review means **accepted feedback and approved training use**
for that single Q&A. The review ref carries the future canonical bytes, but the
API reports `trainingEligible=false` until the provider actually merges it.
After merge the canonical row is:

```text
_source = feedback
trainingStatus = eligible
feedbackReview = true
qualityScore = 0.0 .. 1.0
qualityPercent = 0 .. 100
```

The original signed `ratingValue`, `ratingSlug`, `ratingTitle`, and scale bounds
are retained. `qualityScore` is derived server-side as:

```text
(ratingValue - ratingScaleMin) / (ratingScaleMax - ratingScaleMin)
```

Examples:

| Rating scale | Selected value | qualityScore | qualityPercent |
|---|---:|---:|---:|
| `[-1, +1]` | `-1` | `0.00` | `0%` |
| `[-1, +1]` | `+1` | `1.00` | `100%` |
| `[-2,-1,0,+1,+2]` | `0` | `0.50` | `50%` |
| `[-2,-1,0,+1,+2]` | `+1` | `0.75` | `75%` |

This normalized value is a quality/weight signal, not an instruction to discard
low-rated examples. Training/evaluation pipelines can use poor answers as negative
or preference examples while keeping the raw rating for future recomputation.

## One review, many revisions

Feedback review continuity prevents reviewer queue spam.

```text
one logical feedback receipt
        |
        +-- revision 1
        +-- revision 2
        +-- revision 3
        |
        -> one PR/MR
```

Hugging Face updates the existing PR ref. GitHub, GitLab, and Bitbucket update
the existing source branch behind the PR/MR.

The server persists the provider review locator with the feedback receipt so
normal status/update operations use direct review lookup instead of scanning a
large repository review queue.

## Withdrawal

### Pending feedback

```text
IN REVIEW
    -> Withdraw feedback
    -> close/decline provider review
    -> remove pending lifecycle record
    -> local rating controls reset
```

### Merged feedback

```text
ELIGIBLE
    -> Withdraw feedback
    -> request current canonical feedback/training-view removal
    -> WITHDRAWN
```

The system does not claim physical erasure of provider Git history, backups,
logs, caches, or infrastructure snapshots.

## Anonymous rating telemetry

The **Send anonymous rating telemetry** switch controls a different endpoint:

```text
POST /v1/feedback
```

### Off

Ratings still work locally. No rating telemetry is sent.

### On

The browser may send privacy-minimal mechanics such as:

- rating value/label;
- quick vs detailed mode;
- answer index;
- edit/supersession mechanics;
- bounded event timestamp;
- current versioned telemetry-consent marker.

It intentionally excludes:

- question text;
- answer text;
- written feedback note;
- model identity;
- page URL;
- stable conversation identity.

### Why the switch may appear to have no repository effect

Browser telemetry permission and server telemetry persistence are separate.
If:

```text
FEEDBACK_PERSIST_ENABLED=false
```

then a consented `/v1/feedback` request can be validated and accepted while the
operator intentionally stores no telemetry row. This does not affect
`/v1/feedback/review`.

The Feedback workspace and Endpoint Configuration surface both states so users
can distinguish:

```text
Browser telemetry permission: On/Off
Server telemetry persistence:  On/Off
Maintainer review permission:   On/Off
Maintainer review readiness:    Ready/Not ready
```

## Variables

Recommended non-secret Space Variables include:

| Variable | Default / example | Purpose |
|---|---|---|
| `FEEDBACK_REVIEW_MODE` | `provider-pr` | Enable provider-native maintainer feedback review. Use `disabled` to disable it. |
| `FEEDBACK_PERSIST_ENABLED` | `false` | Independently enable persistence of anonymous rating telemetry. |
| `FEEDBACK_REVIEW_RATE_LIMIT_PER_HOUR` | `20` | Bound content-bearing feedback-review creation/update attempts. |
| `FEEDBACK_REVIEW_TTL_SECONDS` | `604800` | Feedback-review receipt lifetime. |
| `RECORD_STORAGE_TARGETS` | provider JSON | Define the Primary and optional Mirrors. |
| `ALLOWED_ORIGINS` | deployment origins | Additional/replacement browser origins. |
| `ALLOWED_ORIGINS_MODE` | `additive` | CORS origin merge policy. |

For restart-durable feedback management, configure the feedback review ledger or
inherit the contribution ledger settings. Typical controls include:

```text
FEEDBACK_REVIEW_LEDGER_BACKEND=sqlite
FEEDBACK_REVIEW_LEDGER_SQLITE_PATH=/data/feedback-review.sqlite3
```

For replicas/shared authority, use the supported Redis ledger configuration and
require durable/shared mode according to the deployment policy.

## Secrets

Provider storage credentials remain server-side. For example:

```text
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY=<private provider token>
AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR=<private provider token>
```

The browser never receives these values.

The participant's feedback-review management capability is a separate private
authority. Do not place it in URLs, PR titles, issues, repository files, logs,
or screenshots.

## Primary and Mirrors

Only the **Primary** is the review authority. Mirrors must not independently
approve the same feedback item.

```text
Hugging Face PRIMARY
    -> feedback PR #27
    -> maintainer merge
    -> canonical feedback record

GitHub MIRROR
    -> not an independent approval queue
```

This keeps one unambiguous reviewer decision per feedback lifecycle.

## Reviewer workflow

For each feedback PR/MR:

1. Read the latest revision.
2. Use earlier commits only when edit history is useful.
3. Merge to accept into the maintainer feedback dataset.
4. Close/decline to reject.
5. Do not interpret merge as training consent.

At larger scale, the storage/review contract intentionally keeps the logical
feedback identity separate from the provider review locator. That leaves room
for a future bounded batching strategy without changing browser consent or the
feedback row schema.

## Troubleshooting

### Quick rating works but no PR/MR appears

Check all of these:

1. **Share feedback for review & model improvement** is On.
2. The feedback workspace reports review service **Ready**.
3. `FEEDBACK_REVIEW_MODE=provider-pr` is active.
4. `RECORD_STORAGE_TARGETS` has a writable Primary.
5. The Primary provider token has repository write/review permission.
6. The browser origin is CORS-allowed.

Anonymous telemetry being On is not sufficient and is intentionally unrelated.

### Telemetry is On but no feedback file appears

Check `FEEDBACK_PERSIST_ENABLED`. The default is false. This is expected to have
no effect on content-bearing maintainer review.

### Updating a rating opens another review

That is not expected for the same active management receipt. Check that the
receipt ledger is not being lost between requests and that the provider review
locator is persisted. A process-local `memory` ledger can lose continuity after
a restart.

### Review was merged but the UI still says In review

Use **Check status**. Manual provider merges are detected on status/withdrawal
refresh and the lifecycle is ratcheted forward.

### User wants the feedback removed

Use **Withdraw feedback** while the management receipt is still available. If
that capability is unavailable, the maintainer must locate the provider review
or canonical feedback record through server-side repository history/support
processes. Never ask a user to publish a private management capability.

## Security invariants

The implementation should continue to enforce all of the following:

- local rating does not require network permission;
- telemetry consent and feedback-review consent use different versioned keys;
- feedback-review consent never grants dataset-contribution consent;
- feedback review contains exactly one Q&A;
- provider credentials stay server-side;
- one active feedback lifecycle updates one review instead of opening duplicates;
- unchanged content produces no provider commit;
- feedback records never become training-eligible;
- withdrawal authority is distinct from maintainer merge/close authority;
- Mirrors never become independent review authorities;
- logs do not contain Q&A bodies, provider tokens, or participant management capabilities.
