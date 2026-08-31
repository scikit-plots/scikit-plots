# Dataset Contribution and Review Guide

This guide explains the complete dataset-contribution workflow for the Sphinx AI
Assistant: what a reader submits, where the data goes, how maintainers review it,
when it becomes training-eligible, how withdrawal works, and how the same workflow
maps to Hugging Face, GitHub, GitLab, and Bitbucket Cloud.

Use this guide when you see **Contribute to dataset**, **Submit for review**,
`Status: QUARANTINED`, or `Status: IN REVIEW` in the assistant UI.

For low-level storage topology, deduplication, migration, and provider API details,
see [`_hf_spaces_proxy/DATASET_COLLECTION_GUIDANCE.md`](./_hf_spaces_proxy/DATASET_COLLECTION_GUIDANCE.md).

---

## 1. The short version

For human-reviewed datasets, configure:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
```

Then a contribution follows this lifecycle:

```text
Reader
  |
  | Contribute to dataset
  | inspect exact JSON
  | privacy preflight
  | explicit consent
  v
POST /v1/contribute
  |
  v
PRIMARY repository review
  |
  | Hugging Face -> Pull Request
  | GitHub       -> Pull Request
  | GitLab       -> Merge Request
  | Bitbucket    -> Pull Request
  v
IN REVIEW / QUARANTINED
trainingEligible = false
  |
  +-----------------------+
  |                       |
  | maintainer merges     | maintainer closes/declines
  v                       v
APPROVED / ELIGIBLE       NOT ACCEPTED
canonical branch          never training-eligible
  |
  | contributor can later withdraw
  v
WITHDRAWN
```

The configured canonical branch, normally `main`, is the training-eligibility
boundary. An open review is **not** training data.

---

## 2. Three control planes that must stay separate

The assistant deliberately separates three kinds of user action:

| Control plane | Contains conversation content? | Network by default? | Training eligible? |
|---|---:|---:|---:|
| Share/export | only when the reader explicitly shares/exports | depends on chosen share mode | No |
| Rating feedback | No Q&A content in telemetry | Off until explicit telemetry permission | No |
| Dataset contribution | Yes, exact reviewed content | Only after explicit contribution consent | Only after review approval |

A thumbs-up/down rating is not a dataset contribution. A shared conversation is
not a dataset contribution. Only the dedicated **Contribute to dataset** workflow
can submit content for training review.

---

## 3. What the reader does

The contribution sheet supports three scopes:

- **This Q&A** — one selected user/assistant exchange.
- **Rated answers** — only Q&A records the reader explicitly rated.
- **Whole conversation** — one ordered `recordType="conversation"` record with
  `messages[]`; it is not fragmented into unrelated turns.

Before submission the reader can:

1. choose the scope;
2. optionally add reviewer context;
3. inspect the exact JSON that will be submitted;
4. run the privacy preflight;
5. explicitly consent to dataset review/use;
6. press **Submit for review**.

A successful provider-native submission shows wording similar to:

```text
Submitted for review

Status: IN REVIEW · queued in the huggingface review workflow;
not training-eligible unless merged into the canonical branch.

Save private receipt | Copy private withdrawal code | Check status
Copy support reference | Delete pending / withdraw training use
```

In compatibility `ledger` mode it instead shows `Status: QUARANTINED` because the
content is waiting in the lifecycle ledger for an authorized promotion operation.

---

## 4. The management receipt

The reader receives a private management capability after submission. The UI keeps
it available while the tab still owns the active review, and automatically restores
the management controls when the contribution sheet is reopened. The reader can also
move that authority outside browser state in two equivalent private forms:

- **Save private receipt** — a JSON capability file;
- **Copy private withdrawal code** — a compact `aicm2.…` representation of the same capability.

Both are secrets. A third path, **Copy support reference**, is deliberately non-secret
and exists only so a maintainer can locate the native review or stable `.jsonl` file.

Conceptually it contains:

```json
{
  "schemaVersion": 2,
  "kind": "dataset-contribution-management",
  "receiptId": "<opaque receipt>",
  "deleteToken": "<private management capability>",
  "reviewProvider": "huggingface",
  "reviewId": "7",
  "reviewPath": "contributions/2026/08/31/ct_<stable-review-key>.jsonl",
  "reviewRevision": 3,
  "expiresAt": 0,
  "consentVersion": "2.0.0"
}
```

Treat this file like a private capability:

- do not publish it;
- do not paste it into issues, logs, or repository commits;
- do not store it in documentation `conf.py`;
- anyone who obtains it may be able to manage that contribution.

The server stores only the management-token hash, not the raw token.

The controls mean:

| Control | Purpose | Safe to share? |
|---|---|---|
| **Save private receipt** | Save the full management capability as JSON | **No** |
| **Copy private withdrawal code** | Save the same capability as compact text | **No** |
| **Recover withdrawal access** | Import a private receipt or paste a private code later | Local action only |
| **Check status** | Check open/closed/merged lifecycle state | No content is returned |
| **Copy support reference** | Copy receipt + provider review number + stable `.jsonl` path | **Yes**, it contains no withdrawal capability |
| **Copy maintainer removal request** | Produce ready-to-send removal wording using only the support reference | **Yes** |
| **Delete pending / withdraw training use** | Close/remove pending review or withdraw an approved record | Requires private capability |

### If the panel is closed

Closing and reopening the AI panel does not intentionally remove the management UI.
If the tab still holds the active review capability, reopening **Contribute to dataset**
rehydrates the status/withdrawal actions automatically.

If the tab/browser state is gone, open **Recover withdrawal access** and either import
the saved private JSON receipt or paste the private `aicm2.…` withdrawal code.

### If the private capability no longer resolves

A saved receipt/code is portable but does not extend server-side retention. Pending
ledger entries can expire according to deployment policy, and an in-memory ledger can
be lost on restart. In that case use **Copy support reference** (saved earlier) or the
visible repository review/file reference when available and contact a maintainer. A
reference looks like:

```text
Dataset contribution support reference
Receipt: <opaque receipt>
Provider: huggingface
Review: #7
Record: contributions/2026/08/31/ct_<stable-review-key>.jsonl
Revision: 3
```

The support reference deliberately omits the private delete/withdraw token. It is safe
to paste into an issue or support message; the private receipt/code is not.

---

## 5. `provider-pr` versus `ledger`

### Recommended: `provider-pr`

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
```

The proxy opens the Primary provider's normal review object. Maintainers use the
provider UI they already know. No custom review dashboard is required.

Benefits:

- review bytes are durable in the provider rather than only in process memory;
- the canonical branch remains clean until approval;
- normal repository review history is available to maintainers;
- merge is the explicit eligibility decision;
- provider credentials stay server-side;
- the browser receives safe review status, not maintainer credentials.

### Compatibility: `ledger`

```text
CONTRIBUTION_REVIEW_MODE=ledger
```

The historical mode keeps content in the contribution lifecycle ledger until an
authorized promotion call writes an eligible record to configured storage.

Use this mode only when you intentionally want an API/operator-driven promotion
workflow. `CONTRIBUTION_REVIEW_TOKEN` is required for the promotion endpoint.

### How to confirm the active mode

Open the proxy status endpoint:

```text
GET https://your-proxy.example/
```

Look for:

```json
{
  "training": {
    "contribute_ready": true,
    "contribution_review_mode": "provider-pr",
    "canonical_branch": "main"
  }
}
```

If it still says `ledger`, no provider PR/MR will be created automatically.

---


### Review continuity: one management receipt, one review thread

`provider-pr` mode treats the saved management receipt as the continuity key for
one logical contribution review. This prevents a reader from creating a new
provider review every time the same conversation is submitted again from the
same tab/session authority.

| Reader action | Proxy behavior | Reviewer result |
|---|---|---|
| First submit | Open one native PR/MR | One new review appears |
| Submit again with identical reviewed bytes | Return `reviewUpdate="unchanged"` | No new PR/MR and no new commit |
| Conversation changed, then submit again | `PUT` the pending contribution and commit revision N+1 to the same review | Same PR/MR, latest commit is authoritative |
| Check status | Look up the persisted provider review ID directly | No queue scan in the normal path |
| Withdraw while pending | Close/decline the same review | Review cannot be merged accidentally |
| Review already merged/closed | Update fails closed | Reader must explicitly start a new follow-up review |

The UI changes **Submit for review** to **Update existing review** while that
receipt remains active. It also shows the remembered review revision. This
continuity is tab/session scoped because the management capability is sensitive;
it is not a global browser fingerprint.

The provider implementation keeps the native review identity stable:

- Hugging Face commits revisions to `refs/pr/<number>`;
- GitHub commits revisions to the existing Pull Request source branch;
- GitLab commits revisions to the existing Merge Request source branch;
- Bitbucket commits revisions to the existing Pull Request source branch.

The contribution file path is stable for that receipt as well. New content
replaces the pending review file instead of creating `ct_<new-content-hash>.jsonl`
for every revision. Earlier provider commits remain the review history, while the
latest revision is what a maintainer should approve.

> **Deliberate boundary:** this is capability-scoped continuity, not global content deduplication. Two independent readers can submit identical content as
> two independent reviews because each must retain independent withdrawal and
> management authority. The server must not silently merge those authorities
> merely because the bytes match.

For large queues, the proxy stores the provider-native review ID in the receipt
ledger when the review is opened. Status, update, merge, and withdrawal then use
direct review lookup. Repository-wide PR/MR scanning remains only a bounded
legacy/recovery fallback for receipts created before review locators were stored.

## 6. Primary and Mirrors

`RECORD_STORAGE_TARGETS` may configure one Primary and several Mirrors.
Exactly one target must have:

```json
"role": "primary"
```

Only the **Primary** decides eligibility. Mirrors are not independent review authorities.

Example:

```json
[
  {
    "id": "hf-primary",
    "label": "Hugging Face Dataset",
    "provider": "huggingface",
    "role": "primary",
    "repo": "example-org/assistant-contributions",
    "branch": "main",
    "paths": {
      "feedback": "feedback",
      "contributions": "contributions"
    },
    "token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
    "token_type": "fine-grained",
    "expose_links": true
  },
  {
    "id": "github-mirror",
    "label": "GitHub Mirror",
    "provider": "github",
    "role": "mirror",
    "repo": "example-org/assistant-records",
    "branch": "main",
    "paths": {
      "feedback": "feedback",
      "contributions": "contributions"
    },
    "token_env": "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR",
    "expose_links": true
  }
]
```

With this topology:

```text
Submit
  |
  v
Hugging Face PRIMARY
  |
  v
HF Pull Request   <- review authority
  |
  | merge
  v
HF main           <- eligible authority

GitHub MIRROR     <- not an independent approval authority
```

Do **not** expect one user submission to create a review on every mirror. That
would create conflicting eligibility decisions.

Current implementation also does not synchronously fan a maintainer's external
provider-UI merge out to all Mirrors. If immediate mirror convergence is required,
run an explicit post-merge reconciliation/replication process or operate with a
single Primary during the review stage.

---

## 7. Provider-specific reviewer workflow

The proxy uses a receipt-derived opaque review key such as:

```text
ai-contrib-<24-hex-key>
Dataset contribution <24-hex-key>
```

User questions, answers, names, page titles, and notes are not used in branch or
review titles.

### Hugging Face

Primary configuration:

```json
{
  "provider": "huggingface",
  "role": "primary",
  "repo": "example-org/assistant-contributions",
  "branch": "main",
  "token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY"
}
```

Review location:

```text
Dataset repository -> Community -> Pull Requests
```

The proxy creates the change using the Hub PR mechanism (`create_pr=True`). Hugging
Face PRs use special repository refs rather than requiring contributor forks.

Reviewer actions:

- **Merge** -> contribution becomes eligible on the next lifecycle/status sync.
- **Close** -> browser status is shown as **NOT ACCEPTED**; it remains
  training-ineligible.

Official references:

- https://huggingface.co/docs/hub/en/repositories-pull-requests-discussions
- https://huggingface.co/docs/huggingface_hub/main/guides/community

### GitHub

Review location:

```text
Repository -> Pull requests
```

The proxy creates an isolated branch and opens a PR whose base is the configured
canonical branch.

Reviewer actions:

- **Merge pull request** -> eligible.
- **Close pull request** -> not accepted.

Official reference:

- https://docs.github.com/en/rest/pulls

### GitLab

Review location:

```text
Project -> Merge requests
```

The proxy creates a source branch and opens a merge request targeting the
configured canonical branch.

Reviewer actions:

- **Merge** -> eligible.
- **Close** -> not accepted.

For self-managed GitLab, only the GitLab target supports a validated custom
`api_base`.

Official reference:

- https://docs.gitlab.com/api/merge_requests/

### Bitbucket Cloud

Review location:

```text
Repository -> Pull requests
```

The proxy creates a source branch and opens a pull request with the configured
canonical branch as destination.

Reviewer actions:

- **Merge** -> eligible.
- **Decline** -> not accepted.

Official reference:

- https://developer.atlassian.com/cloud/bitbucket/rest/api-group-pullrequests/

---

## 8. Scenario guide

### Scenario A — normal approval

```text
Reader submits
  -> provider review opens
  -> status IN REVIEW
  -> maintainer inspects exact contribution file
  -> maintainer merges
  -> canonical branch contains eligible record
  -> reader presses Check status
  -> status APPROVED / ELIGIBLE
```

The contribution file is written to its final canonical `contributions/...` path
inside the isolated review ref. Before merge, that path is not part of `main`.

### Scenario B — maintainer rejects it

```text
Reader submits
  -> review opens
  -> maintainer closes/declines without merge
  -> trainingEligible remains false
  -> Check status shows NOT ACCEPTED
```

No training job should read closed review refs. Dataset construction accepts only
eligible records from the canonical dataset view by default.

### Scenario C — reader withdraws before review

The reader selects **Delete pending / withdraw training use** while the review is
still open.

The proxy:

1. checks whether the review was merged first;
2. if still pending, closes/rejects the provider review;
3. removes the active pending payload from the lifecycle ledger;
4. returns `status="deleted"`.

This is a logical/current-view deletion guarantee, not a promise of forensic
physical erasure from provider caches, snapshots, Git object history, or backups.

### Scenario D — reader withdraws after approval

If the review was already merged, the same management capability changes meaning:

```text
ELIGIBLE
  |
  | contributor withdrawal
  v
WITHDRAWING
  |
  v
withdrawal tombstone + best-effort current-view removal
  |
  v
WITHDRAWN
```

The dataset builder applies the withdrawal using the server-owned dedup key and
excludes withdrawn content from normal training output. Versioned repository
history is not represented as globally erased.

### Scenario E — provider review creation fails

A submission can enter the local lifecycle ledger but fail to open its native
provider review, for example because of a missing repository permission.

The proxy returns HTTP 503 with a retry-oriented message instead of pretending a
review exists. Current create-once clients can retry the same operation without
intentionally duplicating the contribution.

Check:

- `CONTRIBUTION_REVIEW_MODE=provider-pr`;
- Primary token exists;
- token has write/review permissions;
- repository ID and branch are correct;
- repository allows PR/MR creation;
- proxy logs show a bounded reason code, not secret values.

### Scenario F — Space restarts while a review is open

The provider PR/MR is durable at the provider, but the contributor management
receipt also depends on the contribution ledger.

If startup reports:

```text
backend=memory durability=process_local shared=False
```

then a restart can lose the local receipt lifecycle authority even though the
provider review still exists.

For one restart-durable instance, use SQLite on persistent storage:

```text
CONTRIBUTION_LEDGER_BACKEND=sqlite
CONTRIBUTION_LEDGER_SQLITE_PATH=/data/contribution-lifecycle.sqlite3
CONTRIBUTION_REQUIRE_DURABLE=true
```

For several replicas, use one shared Redis authority:

```text
CONTRIBUTION_LEDGER_BACKEND=redis
CONTRIBUTION_LEDGER_REDIS_URL=<secret Redis URL>
CONTRIBUTION_LEDGER_KEY_SECRET=<strong secret>
CONTRIBUTION_REQUIRE_DURABLE=true
CONTRIBUTION_REQUIRE_SHARED=true
```

Redis durability/backup guarantees still depend on the Redis deployment itself.

---

## 9. Variables and Secrets

The server-side deployment owns storage. Browser `conf.py` must never contain
provider write credentials.

### Variables — non-sensitive configuration

| Variable | Typical value | Why |
|---|---|---|
| `CONTRIBUTION_REVIEW_MODE` | `provider-pr` | Use native repository review |
| `RECORD_STORAGE_TARGETS` | JSON array | Primary/mirror topology and token **names** |
| `TRAINING_DATASET_REPO` | `org/repo` | Legacy HF-only repository identifier |
| `ALLOWED_MODELS` | comma-separated IDs | Exact model admission list |
| `HF_SPACES_MODEL_NAMESPACES` | `example-org` | Path-2 namespace routing |
| `ALLOWED_ORIGINS` | exact browser origins | Custom docs sites |
| `ALLOWED_ORIGINS_MODE` | `replace` for forks | Own the downstream CORS boundary |
| `HF_TOKEN_TYPE` | `fine-grained` / `read` / `write` | Non-secret token classification |

### Secrets — credentials/capabilities

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | model inference credential |
| `AI_RECORD_STORAGE_TOKEN_HF_PRIMARY` | example HF repository write credential |
| `AI_RECORD_STORAGE_TOKEN_GITHUB_PRIMARY` | example GitHub Primary credential |
| `AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR` | example GitHub Mirror credential |
| `AI_RECORD_STORAGE_TOKEN_GITLAB_*` | GitLab target credential |
| `AI_RECORD_STORAGE_TOKEN_BITBUCKET_*` | Bitbucket target credential |
| `CONTRIBUTION_REVIEW_TOKEN` | optional API-driven merge/promotion capability |
| `CONTRIBUTION_LEDGER_KEY_SECRET` | shared Redis receipt pseudonymization key |

`RECORD_STORAGE_TARGETS` contains only a `token_env` name such as:

```json
"token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY"
```

Never put the corresponding token value inside the JSON.

---

## 10. Minimal deployment recipes

### Hugging Face Primary only

Variables:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
RECORD_STORAGE_TARGETS=[{"id":"hf-primary","label":"Hugging Face Dataset","provider":"huggingface","role":"primary","repo":"example-org/assistant-contributions","branch":"main","paths":{"feedback":"feedback","contributions":"contributions"},"token_env":"AI_RECORD_STORAGE_TOKEN_HF_PRIMARY","token_type":"fine-grained","expose_links":true}]
```

Secrets:

```text
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY=<repo-scoped write token>
```

### GitHub Primary only

Variables:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
RECORD_STORAGE_TARGETS=[{"id":"github-primary","label":"GitHub Records","provider":"github","role":"primary","repo":"example-org/assistant-records","branch":"main","paths":{"feedback":"feedback","contributions":"contributions"},"token_env":"AI_RECORD_STORAGE_TOKEN_GITHUB_PRIMARY","expose_links":true}]
```

Secrets:

```text
AI_RECORD_STORAGE_TOKEN_GITHUB_PRIMARY=<repo-scoped token>
```

### Hugging Face Primary + GitHub Mirror

Variables:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
RECORD_STORAGE_TARGETS=[{"id":"hf-primary","label":"Hugging Face Dataset","provider":"huggingface","role":"primary","repo":"example-org/assistant-contributions","branch":"main","paths":{"feedback":"feedback","contributions":"contributions"},"token_env":"AI_RECORD_STORAGE_TOKEN_HF_PRIMARY","token_type":"fine-grained","expose_links":true},{"id":"github-mirror","label":"GitHub Mirror","provider":"github","role":"mirror","repo":"example-org/assistant-records","branch":"main","paths":{"feedback":"feedback","contributions":"contributions"},"token_env":"AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR","expose_links":true}]
```

Secrets:

```text
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY=<HF repository write token>
AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR=<GitHub repository token>
```

The HF Primary receives the review. The GitHub Mirror does not independently
approve or reject it.

---

## 11. First deployment checklist

Before inviting real contributions:

- [ ] `GET /` reports `contribute_ready: true`.
- [ ] `contribution_review_mode` is `provider-pr` if native review is intended.
- [ ] Exactly one storage target has `role: primary`.
- [ ] The Primary repository exists and the configured branch exists.
- [ ] Provider tokens are Secrets, not `conf.py` values.
- [ ] `RECORD_STORAGE_TARGETS` contains token env-var names, never token values.
- [ ] The dataset/repository is private if submissions may contain non-public material.
- [ ] A synthetic submission creates a real PR/MR on the Primary.
- [ ] The contribution is absent from canonical `main` before merge.
- [ ] Closing/declining the review leaves it training-ineligible.
- [ ] Merging the review makes **Check status** report approved/eligible.
- [ ] Pending deletion closes the open review.
- [ ] Post-merge withdrawal removes the record from ordinary training output.
- [ ] Receipt storage is durable enough for your deployment (`sqlite` or `redis` for production use).
- [ ] `deduplicate_dataset.py` produces training output only from eligible, non-withdrawn records.

---

## 12. Troubleshooting

### `Submitted for review` appears, but no provider PR/MR exists

Check the proxy status first:

```json
"contribution_review_mode": "ledger"
```

If it says `ledger`, this is expected. Set:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
```

and restart the service.

If it already says `provider-pr`, check the Primary token/repository permissions
and server logs for `contribute.review_open_fail`.

### The review exists on HF but not on the GitHub Mirror

Expected when HF is Primary. Only the Primary owns review authority.

### The review was merged manually, but the browser still says IN REVIEW

Press **Check status**. The proxy re-reads the provider review state and ratchets
the receipt to eligible when it observes a merge.

If the receipt ledger was lost after a restart, the saved receipt may return 404
even though the PR still exists. Use durable receipt storage for production.

### The review was closed, but the internal lifecycle still says `quarantined`

Provider close/decline is represented by the safe `reviewStatus` while the pending
receipt remains manageable until deletion/expiry. The browser intentionally renders
this combination as **NOT ACCEPTED** and it remains `trainingEligible=false`.

### `dataset_repo` is null on `/`

When `RECORD_STORAGE_TARGETS` is active, `training.dataset_repo` is the legacy
single-HF field and may be null. Use:

```json
"storage": {
  "configured": true,
  "target_count": 1,
  "primary_ready": true
}
```

as the provider-neutral readiness signal.

### Startup says receipt backend is `memory`

The workflow can function, but receipt management is process-local. A restart can
make saved management receipts unresolved. Configure SQLite or Redis before relying
on long-lived review windows.

---

## 13. Security invariants

These are architecture requirements, not optional UX preferences:

1. Provider write tokens are server-side only.
2. `main` (or the configured canonical branch) is the eligibility authority.
3. An open PR/MR is never training eligible.
4. Mirrors never independently decide eligibility.
5. Feedback telemetry cannot silently become training content.
6. Dataset content requires explicit, versioned contribution consent.
7. The reader sees the exact JSON before submission.
8. Receipt/delete capability is separate from reviewer authority.
9. Repository metadata uses opaque receipt-derived identifiers, not user content.
10. Withdrawal means enforceable exclusion from normal training output; it does
    not promise physical erasure of all versioned history, caches, backups, or
    infrastructure snapshots.

---

## 14. Related documentation

- [`README.md`](./README.md) — extension overview and first setup.
- [`_example_conf.py`](./_example_conf.py) — Sphinx configuration reference.
- [`_hf_spaces_proxy/README.md`](./_hf_spaces_proxy/README.md) — proxy deployment and Variables/Secrets.
- [`_hf_spaces_proxy/DATASET_COLLECTION_GUIDANCE.md`](./_hf_spaces_proxy/DATASET_COLLECTION_GUIDANCE.md) — deep multi-store operations, migration, deduplication, and provider details.
- [`_hf_spaces_proxy/deduplicate_dataset.py`](./_hf_spaces_proxy/deduplicate_dataset.py) — canonical dataset reader/deduplicator.
