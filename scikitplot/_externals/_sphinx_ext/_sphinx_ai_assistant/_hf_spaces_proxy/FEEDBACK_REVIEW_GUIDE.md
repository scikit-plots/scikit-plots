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

The Feedback tab includes a centralized **Inspect payload** surface with separate
**JSON** and **JSONL** format tabs. **JSON** shows the exact browser review-request
envelope. **JSONL** shows the pre-save projection of the canonical cloud feedback row in
an expanded readable view; Copy/Download still emits strict one-object-per-line NDJSON.
Downloaded filenames identify both the artifact role and the tab format:
`ai-feedback-review-request-json-<timestamp>.json` for the JSON request tab and
`ai-feedback-review-cloud-projection-jsonl-<timestamp>.jsonl` for the JSONL projection
tab. Cloud provider storage may retain opaque lifecycle-safe names internally; human
exports should use role-bearing names such as `cloud-record-jsonl` or
`cloud-merged-jsonl`. The active format is named directly in the size badge and
Copy/Download actions. The tabs support pointer,
Arrow-key, Home, and End navigation with a single keyboard focus target. Long content wraps
inside the local inspector, and syntax emphasis is created only with inert text nodes.
When anonymous rating telemetry is enabled, its privacy-minimal JSONL row appears as a
nested, separately labeled record because it is not the maintainer-review payload.
Inspect/copy/download never submit anything. If originating model attribution is
unavailable, review sharing fails closed instead of creating an ambiguous training row.


For maintainer review, model attribution is intentionally narrower than the full model
configuration. Only the model id, provider, and concrete model name cross the review
boundary. Endpoint URLs, model information links, descriptions, labels, credentials,
query strings, fragments, and other transport/configuration metadata are not part of the
feedback-review payload. Source-page evidence is independently canonicalized to an
HTTP(S) origin + path without userinfo, query, or fragment. The server repeats these
normalizations even for older or hostile clients.

### Cloud merged feedback view

Cloud feedback remains **one authoritative provider record per review**. The
**individual canonical provider feedback records** are the source authority. Do not replace
those files with one mutable monolithic JSONL: individual records are the safer authority
for review revisions, concurrent submissions, deduplication, withdrawal, and current-view
removal.

For analysis, CI, or maintainer download, derive a deterministic merged view from the
canonical cloud branch:

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --feedback-review-cloud-merged
```

The default human-facing artifact name is
`ai-feedback-review-cloud-merged-jsonl-<UTC timestamp>.jsonl`. The command also writes
`<filename>.manifest.json`, which records the content SHA-256, byte count, row count,
artifact role, and an explicit statement that the merged file is **derived, not
authoritative**. Provider credentials remain environment-only and are never written to
either artifact.

The merged view contains only current canonical rows that are all of the following:

- `_source == "feedback"`;
- `feedbackReview == true`;
- `recordType == "qa"`;
- `trainingStatus == "eligible"`;
- not a retraction/withdrawal tombstone.

The existing storage deduplication and feedback-lineage resolver runs before filtering, so
superseded revisions, retractions, forks, and malformed lineage are resolved with the same
fail-closed rules used for training reads. `--include-unreviewed` is intentionally
incompatible with this merged-review mode.

A custom filename may be supplied with `--output`; the sidecar manifest follows that
filename. The cloud provider's opaque `fb_<hash>.jsonl` files remain internal lifecycle
identities and are not rewritten or deleted by this export.

### Dataset contribution

Dataset contribution can contain:

- one Q&A;
- rated answers; or
- a whole structured conversation.

It uses its own privacy preflight and contribution consent. Approved
contribution records live under `contributions/` and can become eligible only
through the configured review lifecycle.

### Activity

Activity is a bounded, tab-local management ledger for feedback reviews and dataset
contributions. Entries are shown newest-first. Use **Manage** to return to the
appropriate control plane, **Forget** to discard one private in-tab management
receipt, or **Forget all** at the section header to clear that tracked family.
Forget actions never delete, close, merge, withdraw, or otherwise alter remote data.
Because forgetting may discard the only in-tab withdrawal capability, both actions
require a second confirming click.

Terminal reviews are not kept as permanent history. When a status check reports a
review as merged/reviewed, closed/rejected, deleted, withdrawn, expired, or the
receipt/provider review is no longer available (404/410), its tab-local tracking
entry is removed automatically. Activity performs no background polling.

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

When **Maintainer feedback review** / **Share with maintainers** is On (built-in
initial value **True**, configurable with
``ai_assistant_panel_feedback_review_default`` and overridable by the reader):

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
already local. The next explicit quick or detailed feedback save is the operation
that creates or updates the maintainer review. Merely changing a workspace setting
or typing in the optional note does not submit content.


### Browser initial defaults

The documentation build can set the initial state of reader-facing controls without
overriding a choice already stored in the browser:

| Setting | Built-in initial value | Sphinx config |
|---|---:|---|
| Anonymous rating telemetry | Off | ``ai_assistant_panel_feedback_telemetry_default`` |
| Maintainer feedback review | On | ``ai_assistant_panel_feedback_review_default`` |
| Page integration events | Off | ``ai_assistant_panel_page_integration_default`` |
| Streaming responses | On | ``ai_assistant_panel_streaming_default`` |
| Remember conversation in this tab | On | ``ai_assistant_panel_remember_conversation`` |

``ai_assistant_panel_api_streaming`` is separate: it is the hard SSE capability
ceiling. If it is False, the reader cannot enable streaming even when their
Streaming responses preference is On. Explicit browser ON/OFF values are retained
so a configured default cannot silently resurrect a setting the reader turned off.

## Feedback revision lineage

Canonical dataset schema v5 keeps storage identity separate from semantic rating
identity. Every current rating revision carries a bounded, self-contained lineage:

```text
feedbackId       = current rating/revision event
feedbackChainId  = stable root feedbackId for this answer
prevFeedbackId   = immediate predecessor (backward-compatible scalar)
prevFeedbackIds  = complete ordered ancestry, oldest -> newest
editCount        = revision depth; normally len(prevFeedbackIds)
```

Example:

```text
f1  chain=f1  prev=null  history=[]       edit=0
 |
f2  chain=f1  prev=f1    history=[f1]     edit=1
 |
f3  chain=f1  prev=f2    history=[f1,f2]  edit=2
```

`prevFeedbackId` intentionally remains a scalar so historical readers do not break;
`prevFeedbackIds[]` supplies the retrospective chain. The browser keeps the bounded
lineage companion state in same-tab `sessionStorage` only when **Remember conversation
in this tab** is enabled, so a restored transcript does not silently forget rating
ancestry. Turning that setting Off clears both transcript and lineage restoration
state.

The deduplication tool resolves storage lifecycle first and semantic rating lineage
second. A later semantic revision wins even when an older representation comes from
a higher-priority source. Source priority applies only when feedback and contribution
contain the same terminal `feedbackId`. Same-revision forks, cycles, duplicate IDs with
conflicting ancestry, and other explicit malformed v5 lineage fail closed instead of
being guessed from timestamps.

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

The Feedback workspace is the single visible owner for telemetry and maintainer-review
permissions. Endpoint Configuration no longer duplicates those consent switches; it keeps
runtime/endpoint and optional page-integration controls. The Feedback workspace distinguishes:

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

1. **Maintainer feedback review** / **Share with maintainers** is On.
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
- feedback review records become training-eligible only while review-sharing permission is active and after authorized maintainer merge; anonymous telemetry never becomes training-eligible;
- withdrawal authority is distinct from maintainer merge/close authority;
- Mirrors never become independent review authorities;
- logs do not contain Q&A bodies, provider tokens, or participant management capabilities.

## Release archive witness boundary

Feedback behavior is unchanged by Run 162. Archive-health witnessing and retention-root
recovery operate only on release-security evidence and do not grant feedback telemetry,
dataset contribution, or browser-runtime authority.
