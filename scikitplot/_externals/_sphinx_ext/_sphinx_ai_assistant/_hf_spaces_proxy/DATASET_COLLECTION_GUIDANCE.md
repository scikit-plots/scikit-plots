# Dataset Collection and Multi-Store Operations Guide

## Schema v4 privacy and contribution lifecycle — authoritative

> **Current policy:** where older historical sections below describe feedback as
> training input, cross-source feedback/contribution joins, unversioned consent,
> immediate contribution persistence, or review-token-only promotion, this section
> supersedes them. For the human workflow first read
> [`../DATASET_CONTRIBUTION_GUIDE.md`](../DATASET_CONTRIBUTION_GUIDE.md).

- Ordinary `/v1/feedback` is rating telemetry only. Query, answer, note, model,
  page and conversation identity are discarded server-side; durable feedback is
  opt-in and marked `trainingStatus="telemetry"`.
- Schema v4 keeps feedback telemetry and explicit content contribution
  structurally separate. Contribution has two record families: `recordType="qa"`
  and one ordered `recordType="conversation"` with `messages[]`. Whole-conversation
  mode is not exploded into unrelated training rows.
- **Contribute to dataset** is the only content-bearing contribution surface.
  **This Q&A**, **Rated answers**, and **Whole conversation** converge on exact-JSON
  inspection, privacy preflight, explicit versioned consent, quarantine/review,
  private management receipt, pending delete, and post-approval withdrawal.
- Schema v4 contribution consent is `2.0.0`. Legacy schema v2/v3 clients may use
  historical consent `1.0.0` only for the legacy contract.
- `/v1/contribute` always creates a lifecycle receipt first and starts at
  `trainingStatus="quarantined"`; quarantined content is never ordinary training
  input. The lifecycle backend may be `memory`, `sqlite`, or `redis`.
- **Recommended human-review mode:** `CONTRIBUTION_REVIEW_MODE=provider-pr`. The
  Primary storage target receives a native Hugging Face/GitHub/GitLab/Bitbucket
  review object. The configured canonical branch (`main` by default) is the
  eligibility boundary. Merge means eligible. Close/decline remains
  training-ineligible and the browser renders it as **NOT ACCEPTED**.
- **Compatibility mode:** `CONTRIBUTION_REVIEW_MODE=ledger`. Content stays in the
  lifecycle ledger until the authenticated `/promote` endpoint, protected by
  `CONTRIBUTION_REVIEW_TOKEN`, writes/promotes eligible bytes.
- Only the **Primary** owns review authority. Mirrors do not independently approve
  or reject the same contribution. An external Primary UI merge is detected on a
  later status/management check and ratchets the receipt to `eligible`. Current
  code does not synchronously fan that external merge out to mirrors.
- The receipt has a separate delete/withdraw capability. Before approval, DELETE
  closes the pending provider review when applicable and removes active receipt
  content. After approval, the same capability records a privacy-minimal
  withdrawal tombstone and attempts best-effort current-view removal.
- Provider-review durability and receipt durability are different. A PR/MR can
  survive a proxy restart while `CONTRIBUTION_LEDGER_BACKEND=memory` loses the
  contributor-management authority. Use persistent SQLite for one instance or
  shared Redis for replicas when long-lived receipt management matters.
- `deduplicate_dataset.py` fails closed: ordinary training output accepts only
  `trainingStatus="eligible"` and applies later withdrawal state.
- Physical deletion from versioned repository history, database pages/WAL,
  backups, CDN/provider logs, or infrastructure snapshots is **not guaranteed**.
  Withdrawal is an enforceable training-exclusion/current-view operation, not a
  claim of global forensic erasure.
- `CONTRIBUTION_REQUIRE_DURABLE=true` can fail closed unless receipt storage is
  restart-durable. `CONTRIBUTION_REQUIRE_SHARED=true` independently requires a
  shared transactional receipt authority.


**Component:** scikit-plots Sphinx AI Assistant proxy
**Scope:** Feedback and consent-gated contribution records
**Storage:** Hugging Face, GitHub, GitLab, Bitbucket Cloud, or a primary + mirrors
**Guide version:** 4.0
**Verified against implementation:** 2026-08-31
**Audience:** first-time operator → maintainer → senior platform engineer

---

## 0. Start here: the whole system in one picture

The most important concept is that **browser routing** and **record storage** are
two different layers.

```text
Documentation browser
        |
        | POST /v1/feedback
        | POST /v1/contribute
        v
AI proxy / app.py
        |
        | validates + normalizes once
        | creates one canonical record ID/content
        v
PRIMARY storage target          <- acceptance boundary
        |
        +--------------------+
        |                    |
        v                    v
Mirror A                  Mirror B
(optional)                (optional)
```

Examples:

```text
HF only
Browser -> app.py -> Hugging Face PRIMARY
```

```text
GitHub only
Browser -> app.py -> GitHub PRIMARY
```

```text
Recommended redundant setup
Browser -> app.py -> Hugging Face PRIMARY -> GitHub MIRROR
```

> **Important — one Primary only.**
> A configuration may contain up to 8 storage targets, but exactly one must
> have `"role": "primary"`. A record is accepted when the Primary succeeds.
> Mirror failure does not undo a successful Primary write.

> **Attention — the browser does not choose HF vs GitHub.**
> `conf.py` and the Endpoint Configuration panel choose **which proxy endpoint**
> the browser calls. Storage selection is server-side through
> `RECORD_STORAGE_TARGETS`. This keeps write credentials out of public docs,
> JavaScript, localStorage, and generated HTML.

> **Hint — if you are new to this, start with one provider.**
> Get HF-only or GitHub-only working first. Add a mirror only after an operator-side
> storage verification confirms the Primary can persist a **reviewed promoted** record.
> Public Service diagnostics intentionally reports only coarse readiness, not private
> storage topology.

---

## 1. The four configuration surfaces

| Surface | Purpose | Contains credentials? | Typical owner |
|---|---|---:|---|
| Sphinx `conf.py` / AI panel | Browser → proxy endpoint routing | **No** | docs maintainer |
| Hugging Face Space **Variables** | Non-sensitive server configuration | No | operator |
| Hugging Face Space **Secrets** | Tokens/API credentials | **Yes** | operator/security |
| `RECORD_STORAGE_TARGETS` | Primary/mirror topology | Token **names**, never token values | operator |

### 1.1 `conf.py` is not a storage credential store

A normal production docs configuration can stay as simple as:

```python
ai_assistant_endpoint_profiles = {
    "default": {
        "label": "Production Proxy",
        "base": "https://scikit-plots-ai.hf.space",
        "chat": None,
        "share": None,
        "feedback": None,
        "training": None,
    },
}

ai_assistant_endpoint_default_profile = "default"
```

The browser resolves the inherited routes to the proxy, for example:

```text
https://scikit-plots-ai.hf.space/v1/feedback
https://scikit-plots-ai.hf.space/v1/contribute
```

`app.py` then decides whether those records go to HF, GitHub, or several stores.

> **Important — never put write tokens in `conf.py`.**
> Sphinx output is publishable content. A token placed in docs configuration can
> leak through source control, generated artifacts, CI logs, or client assets.

---

## 2. Secret vs Variable: what belongs where

Hugging Face officially distinguishes **Variables** for non-sensitive values
from **Secrets** for access tokens, API keys, and credentials. Space Variables
are readable; Secret values are write-only in the settings interface.

### Recommended classification

| Name | Put in | Required? | Why |
|---|---|---:|---|
| `RECORD_STORAGE_TARGETS` | **Variable** | New multi-store mode | Contains topology and env-var names, not tokens |
| `AI_RECORD_STORAGE_TOKEN_HF_PRIMARY` | **Secret** | if HF target | Actual HF credential |
| `AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR` | **Secret** | if GitHub target | Actual GitHub credential |
| `AI_RECORD_STORAGE_TOKEN_GITLAB_*` | **Secret** | if GitLab target | Actual GitLab credential |
| `AI_RECORD_STORAGE_TOKEN_BITBUCKET_*` | **Secret** | if Bitbucket target | Actual Bitbucket credential |
| `TRAINING_DATASET_REPO` | **Variable** | Legacy HF mode only | Repo ID is normally non-sensitive |
| `HF_DATASET_TOKEN` | **Secret** | Legacy HF mode | Actual HF credential |
| `HF_DATASET_TOKEN_TYPE` | **Variable** | Legacy HF mode | Classification only |
| `FEEDBACK_PERSIST_ENABLED` | **Variable** | Recommended | Server persistence flag; never substitutes for reader telemetry consent |

> **Attention — your existing `TRAINING_DATASET_REPO` Secret works, but it does
> not normally need to be a Secret.**
> `scikit-plots/ai-assistant-contributions` is configuration metadata, not a
> credential. Prefer a Variable unless even the repository identifier itself is
> confidential.

> **Attention — `AI_RECORD_STORAGE_TOKEN_HF_PRIMARY_TYPE` is normally a
> Variable, not a Secret.**
> Better still: if your target JSON already contains
> `"token_type": "fine-grained"`, you do not need the separate `_TYPE`
> variable at all.

### When `RECORD_STORAGE_TARGETS` itself should be a Secret

Normally it is safe as a Variable because it contains values like:

```json
{
  "repo": "scikit-plots/ai-assistant-records",
  "token_env": "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR"
}
```

—not the token value.

If repository names/topology are themselves confidential, you may store the
whole JSON as a Space Secret instead. `app.py` receives both Variables and
Secrets as environment variables.

`expose_links` remains a storage-target compatibility setting, but Run 5 public
discovery no longer auto-publishes repository/storage URLs from server topology.
Public links should be supplied intentionally through documentation/profile
configuration when they are meant to be visible to readers.

---

## 3. Current configuration precedence

### 3.1 Provider-neutral mode is authoritative

When this is non-empty:

```text
RECORD_STORAGE_TARGETS
```

it defines the live storage topology.

`TRAINING_DATASET_REPO`, `HF_DATASET_TOKEN`, `HF_WRITE_TOKEN`, and `HF_TOKEN`
remain available only for legacy compatibility/fallback behavior; they do not
replace the explicit targets in `RECORD_STORAGE_TARGETS`.

The proxy has been hardened so keeping `TRAINING_DATASET_REPO` during migration
does **not** trigger misleading legacy “missing HF dataset token” warnings while
`RECORD_STORAGE_TARGETS` is active.

### 3.2 Legacy mode

If `RECORD_STORAGE_TARGETS` is empty and `TRAINING_DATASET_REPO` is set, the
proxy synthesizes one target:

```text
id       = hf-primary
provider = huggingface
role     = primary
repo     = TRAINING_DATASET_REPO
```

Token precedence is:

```text
HF_DATASET_TOKEN       preferred
        |
        v
HF_WRITE_TOKEN         legacy alias
        |
        v
HF_TOKEN               compatibility fallback
```

This keeps existing deployments operational.

---

# Part I — Beginner setup

## 4. Recipe A: Hugging Face only — legacy/simple mode

Use this if you want the fewest settings.

### Step 1 — Create or choose the HF Dataset repository

Example:

```text
scikit-plots/ai-assistant-contributions
```

The proxy writes JSONL records into that Dataset repository.

### Step 2 — Create the HF token

For production, choose a **Fine-grained** User Access Token scoped so it can
write the target Dataset repository.

The three HF token types are:

| HF token type | Record persistence | Recommendation |
|---|---:|---|
| `fine-grained` | Yes, when its resource scope permits writing this repo | **Preferred** |
| `read` | **No** | Blocked by proxy |
| `write` | Yes | Works, but broader than necessary |

The proxy performs a repo-specific write-capability preflight with modern
`huggingface_hub` and can report `verified`, `denied`, or an unverified fallback
state. A real successful commit also proves capability.

> **Implementation note — fresh proxy deployments install `huggingface_hub>=1.0,<2`.**
> That range provides `HfApi.auth_check(..., write=True)` for non-mutating,
> repo-specific write verification. The runtime still feature-detects the method
> so older environments fail soft to first-commit verification rather than
> crashing.

### Step 3 — Add Space settings

**Variable**:

```text
TRAINING_DATASET_REPO=scikit-plots/ai-assistant-contributions
```

**Secret**:

```text
HF_DATASET_TOKEN=<hf-repo-write-token>
```

**Variable**:

```text
HF_DATASET_TOKEN_TYPE=fine-grained
```

**Optional telemetry Variable**:

```text
FEEDBACK_PERSIST_ENABLED=true
```

This opts the server into durable **rating telemetry only**. It does not store
question/answer/comment/model/page/session content and does not make feedback
training-eligible. Leave it unset/false unless durable rating telemetry is an
explicit operational requirement.

For explicit contribution review/promotion also configure the Secret:

```text
CONTRIBUTION_REVIEW_TOKEN=<separate high-entropy operator capability>
```

### Step 4 — restart and inspect status

Changing Space configuration causes the app to restart. After it is healthy:

```bash
BASE=https://scikit-plots-ai.hf.space
curl -s "$BASE/" | python -m json.tool
```

Look only for the privacy-minimized readiness summary, for example:

```json
"training": {
  "configured": true,
  "target_count": 1,
  "primary_ready": true
}
```

The public status surface intentionally does **not** enumerate provider/repository
URLs, token classes, or storage links. Use operator-side configuration/logs/provider
consoles for private topology verification.

---

## 5. Recipe B: Hugging Face only — new storage-target mode

This is the preferred foundation if you expect to add mirrors later.

### Secrets

```text
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY=<hf-repo-write-token>
```

### Variables

Optional durable rating telemetry (still training-ineligible):

```text
FEEDBACK_PERSIST_ENABLED=true
```

Explicit contribution promotion additionally requires the Secret
`CONTRIBUTION_REVIEW_TOKEN`; raw accepted contributions remain in mutable
quarantine until that review authority promotes them.

```text
RECORD_STORAGE_TARGETS=[
  {
    "id": "hf-primary",
    "label": "Hugging Face Dataset",
    "provider": "huggingface",
    "role": "primary",
    "repo": "scikit-plots/ai-assistant-contributions",
    "branch": "main",
    "paths": {
      "feedback": "feedback",
      "contributions": "contributions"
    },
    "token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
    "token_type": "fine-grained",
    "expose_links": true
  }
]
```

`TRAINING_DATASET_REPO` is not required for the new write path. You may keep it
as a Variable during migration/rollback:

```text
TRAINING_DATASET_REPO=scikit-plots/ai-assistant-contributions
```

### Alternative token-type configuration

These two forms are equivalent for the target parser.

**Preferred — keep type beside the target:**

```json
"token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
"token_type": "fine-grained"
```

**Alternative — omit `token_type` and use a Variable:**

```text
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY_TYPE=fine-grained
```

Do **not** configure both unless you intentionally want redundant documentation.
If both exist, the explicit JSON `token_type` wins.

---

## 6. Recipe C: GitHub only

### Step 1 — create a repository

Example:

```text
scikit-plots/ai-assistant-records
```

A private repository is usually appropriate for **reviewed, promoted** contribution
records and any intentionally durable rating telemetry. Raw contribution intake
must remain in the mutable quarantine/control-plane stage rather than being
committed directly into repository history.

### Step 2 — create a fine-grained GitHub PAT

Use a fine-grained Personal Access Token with:

```text
Resource owner:       owner of the target repository
Repository access:    Only select repositories
Selected repository:  scikit-plots/ai-assistant-records
Contents permission:  Read and write
```

The storage adapter uses GitHub's **Create or update file contents** endpoint,
which officially requires repository `Contents: write` for fine-grained tokens.

> **Important — branch protection still applies.**
> A token may have `Contents: write` yet still be unable to commit directly to a
> protected branch. Test the exact branch configured in `RECORD_STORAGE_TARGETS`.

### Step 3 — add Secret

```text
AI_RECORD_STORAGE_TOKEN_GITHUB_PRIMARY=<github-repo-token>
```

### Step 4 — add Variable

```text
RECORD_STORAGE_TARGETS=[
  {
    "id": "github-primary",
    "label": "GitHub Records",
    "provider": "github",
    "role": "primary",
    "repo": "scikit-plots/ai-assistant-records",
    "branch": "main",
    "paths": {
      "feedback": "feedback",
      "contributions": "contributions"
    },
    "token_env": "AI_RECORD_STORAGE_TOKEN_GITHUB_PRIMARY",
    "expose_links": true
  }
]
```

No HF dataset token is required for record persistence in this topology.
`HF_TOKEN` may still be needed independently for model inference.

---

## 7. Recipe D: Hugging Face Primary + GitHub Mirror — recommended redundant setup

This matches the topology you are using.

### 7.1 Correct Space settings

#### Secret 1 — HF write credential

```text
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY=<hf-repo-write-token>
```

#### Secret 2 — GitHub write credential

```text
AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR=<github-repo-token>
```

#### Variable — target topology

```json
[
  {
    "id": "hf-primary",
    "label": "Hugging Face Dataset",
    "provider": "huggingface",
    "role": "primary",
    "repo": "scikit-plots/ai-assistant-contributions",
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
    "repo": "scikit-plots/ai-assistant-records",
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

Store that JSON as the Space **Variable**:

```text
RECORD_STORAGE_TARGETS
```

#### Optional migration Variable

```text
TRAINING_DATASET_REPO=scikit-plots/ai-assistant-contributions
```

It is harmless while `RECORD_STORAGE_TARGETS` is active and useful for rollback
or old discovery consumers. It no longer drives the active write topology.

#### Optional rating-telemetry Variable

```text
FEEDBACK_PERSIST_ENABLED=true
```

This stores privacy-minimal rating telemetry only. It never enables Q&A/comment
collection and never changes `trainingStatus` to `eligible`.

### 7.2 What you do **not** need

Because the JSON already says:

```json
"token_type": "fine-grained"
```

you do not need:

```text
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY_TYPE
```

If you prefer the `_TYPE` Variable, remove `token_type` from the JSON first.

### 7.3 Expected runtime topology

```text
PRIMARY
Hugging Face Dataset
scikit-plots/ai-assistant-contributions
        |
        +---- MIRROR ----> GitHub
                           scikit-plots/ai-assistant-records
```

---

## 8. Recipe E: GitHub Primary + Hugging Face Mirror

This is valid when GitHub should define acceptance and HF is only a replica.

```json
[
  {
    "id": "github-primary",
    "label": "GitHub Records",
    "provider": "github",
    "role": "primary",
    "repo": "scikit-plots/ai-assistant-records",
    "branch": "main",
    "token_env": "AI_RECORD_STORAGE_TOKEN_GITHUB_PRIMARY"
  },
  {
    "id": "hf-mirror",
    "label": "Hugging Face Dataset Mirror",
    "provider": "huggingface",
    "role": "mirror",
    "repo": "scikit-plots/ai-assistant-contributions",
    "branch": "main",
    "token_env": "AI_RECORD_STORAGE_TOKEN_HF_MIRROR",
    "token_type": "fine-grained"
  }
]
```

The success contract changes accordingly:

```text
GitHub primary failure  -> request is not accepted
GitHub primary success  -> request accepted
HF mirror failure       -> accepted, mirror becomes degraded/retries
```

---

# Part II — Verify before collecting real data

## 9. Test 1: service discovery

```bash
BASE=https://scikit-plots-ai.hf.space
curl -s "$BASE/" | python -m json.tool
```

Or with `jq`:

```bash
curl -s "$BASE/" | jq '.storage'
```

For HF + GitHub, expect two targets and one Primary:

```text
hf-primary      provider=huggingface  role=primary
github-mirror   provider=github       role=mirror
```

### HF token diagnostics

The HF target exposes classification/capability only, never the token value:

```json
"token": {
  "type": "fine-grained",
  "write_capability": "verified"
}
```

Possible capability states include:

| State | Meaning |
|---|---|
| `verified` | repo-specific write preflight or successful commit proved access |
| `unverified` | token appears usable but older client/network prevented preflight |
| `broad-write` | classic Write token; functional but broader |
| `denied-read-token` | declared Read token; persistence blocked |
| `denied` | explicit permission rejection such as 401/403 |
| `missing-token` | configured token env var has no value |

> **Hint — `fine-grained` is a token type, not a guarantee by itself.**
> The token must also be scoped to a resource on which its owner has permission
> to write.

---

## 10. Test 2: real contribution intake and native review

Prefer the browser's **Contribute to dataset** sheet for normal operation because
it performs exact-JSON inspection, privacy preflight, and the current consent UX.
For an operator smoke test, use a clearly synthetic schema-v4 record:

```bash
BASE=https://scikit-plots-ai.hf.space

curl -sS "$BASE/v1/contribute" \
  -H "Content-Type: application/json" \
  -d '{
    "schemaVersion": 4,
    "consentFlag": true,
    "consentVersion": "2.0.0",
    "page": "https://example.invalid/storage-smoke-test",
    "model": null,
    "records": [
      {
        "recordType": "qa",
        "answerIndex": 0,
        "query": "synthetic storage smoke test",
        "answer": "synthetic answer for review only",
        "message": "operator smoke test; remove after verification"
      }
    ]
  }' | python -m json.tool
```

In recommended `provider-pr` mode, expect a lifecycle response shaped like:

```json
{
  "accepted": true,
  "status": "quarantined",
  "rows": 1,
  "receiptId": "<opaque receipt>",
  "deleteToken": "<private management capability>",
  "consentVersion": "2.0.0",
  "reviewMode": "provider-pr",
  "reviewProvider": "huggingface",
  "reviewStatus": "open",
  "trainingEligible": false
}
```

The exact response may contain additional content-free lifecycle metadata. Keep the
`deleteToken` private; current browser clients generate/hold their management
capability locally and do not require the server to echo it.

### Verify the Primary review — not `main`

Do **not** expect the new contribution on the canonical branch immediately. Open
the Primary provider's review UI:

```text
Hugging Face -> Dataset -> Community / Pull Requests
GitHub       -> Repository -> Pull requests
GitLab       -> Project -> Merge requests
Bitbucket    -> Repository -> Pull requests
```

The review should contain a contribution path such as:

```text
contributions/YYYY/MM/DD/ct_<recordId>.jsonl
```

but that path exists only on the isolated review ref until approval.

For a provider-native smoke test:

1. verify the review is open;
2. verify the contribution is absent from canonical `main`;
3. inspect the exact JSON/JSONL bytes;
4. merge the review;
5. call **Check status** in the browser or the receipt-status endpoint;
6. confirm the lifecycle becomes `eligible`;
7. confirm the canonical branch now contains the record.

If you close/decline instead, the contribution remains training-ineligible and the
browser renders the provider review state as **NOT ACCEPTED**.

### Mirror expectations

If the topology is HF Primary + GitHub Mirror, the smoke test creates an HF review
only. The GitHub Mirror is not an independent review authority. Current code does
not synchronously replicate a human Primary merge to Mirrors; use explicit
reconciliation when immediate mirror convergence is required.

### Compatibility `ledger` mode

If `GET /` reports:

```json
"contribution_review_mode": "ledger"
```

the same intake remains in the receipt ledger and no provider review is expected.
Only the separately authorized promotion path can write the eligible record.


## 11. Test 3: rating telemetry and contribution quarantine

### Rating telemetry

The browser does not send rating telemetry until the reader explicitly enables
**Send rating telemetry**. The permission is versioned and stored as a structured
browser consent record; missing, malformed, stale-version, or historical boolean
preferences fail closed to **Off**. Every network telemetry request carries
`telemetryConsent=true`, `telemetryConsentVersion="1.0.0"`, and the grant timestamp;
the bundled proxy/Worker reject requests that omit the current consent contract.
With that user permission active and `FEEDBACK_PERSIST_ENABLED=true`, a rating may produce:

```text
feedback/YYYY/MM/DD/fb_<recordId>.jsonl
```

Inspect the row and verify it contains rating/event mechanics only: no query,
answer, note, model, page URL, or conversation identifier, and
`trainingStatus="telemetry"`. Turning telemetry off must stop future rating and
retraction network requests. Storage failure must not break the local rating UI.
The public `ai-assistant-feedback` DOM event is content-free for the same reason:
page listeners receive bounded rating mechanics, not the Q&A/note/model/page tuple.

### Contribution receipt lifecycle

`POST /v1/contribute` always starts with `status="quarantined"`, a lifecycle
receipt, and a separate management capability. The content is training-ineligible
at this stage. What happens next depends on `CONTRIBUTION_REVIEW_MODE`.

With `provider-pr`, the proxy opens a native review on the **Primary**:

```text
quarantined + reviewStatus=open
        |
        +-- maintainer merge ------> eligible
        |
        +-- close / decline -------> quarantined + reviewStatus=closed/rejected
                                     (browser: NOT ACCEPTED)
```

The proxy writes the future eligible bytes to their final canonical path inside
the isolated provider review ref. They do not become part of the canonical branch
until merge. A later **Check status** observes a manual merge and atomically
ratchets the lifecycle receipt to `eligible`.

With compatibility `ledger` mode, only the authenticated promotion endpoint may
claim the receipt and write durable `trainingStatus="eligible"` rows.

After approval, raw contributed content is cleared from the receipt ledger;
lifecycle metadata, server-owned dedup keys, provider paths, and the
delete-capability hash remain. The same receipt capability then means **withdraw
from training use**: the proxy writes privacy-minimal withdrawal tombstones and
attempts current-view deletion. `deduplicate_dataset.py` applies the later
withdrawal through last-write-wins and does not emit the tombstone itself.

For restart durability on a single writable instance, configure:

```text
CONTRIBUTION_LEDGER_BACKEND=sqlite
CONTRIBUTION_LEDGER_SQLITE_PATH=/data/contribution-lifecycle.sqlite3
CONTRIBUTION_REQUIRE_DURABLE=true
CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS=86400
```

On restart, SQLite reclaims interrupted `promoting`/`withdrawing` states. Promotion replays target the same receipt-stable provider path derived from the original intake timestamp rather than creating a second dated training file. Deleted/expired/withdrawn lifecycle tombstones are retained only for the configured bounded status window; eligible receipts remain managed until withdrawal or an explicitly managed migration to the shared Redis receipt authority.

The SQLite backend uses transactional state changes plus `secure_delete` and WAL
checkpoint/truncation as defense in depth. Those controls still do **not** prove
forensic erasure from storage media, host snapshots, or backups. SQLite is also
not a shared multi-replica authority. Horizontally scaled collection should use
the Redis receipt authority described in **Shared receipt authority across replicas**
below, with external Redis persistence/durability verified separately.

---

## 12. Test 4: deliberately break the mirror

This is an important production readiness test.

1. Keep the HF Primary token valid.
2. Temporarily replace/revoke the GitHub Mirror token.
3. Submit one synthetic contribution and confirm it is quarantined/in review.
4. If using `provider-pr`, merge the Primary provider review. If using `ledger`,
   promote the quarantine receipt with the separate review capability.

Expected approval/promotion behavior:

```text
HF Primary      success
GitHub Mirror   degraded
reviewed record promoted/accepted
```

The response may show:

```json
"mirrors": {
  "github-mirror": "degraded"
}
```

Restore the GitHub token after the test.

> **Important — mirrors are not an atomic transaction.**
> There is no cross-provider rollback. A successful Primary is authoritative.
> This is intentional: HF, GitHub, GitLab, and Bitbucket have independent commit
> APIs and failure semantics.

---

# Part III — Record organization and data quality

## 13. Folder structure: current and legacy

### Current multi-store layout

```text
repo-root/
├── feedback/
│   └── YYYY/
│       └── MM/
│           └── DD/
│               └── fb_<record-id>.jsonl
└── contributions/
    └── YYYY/
        └── MM/
            └── DD/
                └── ct_<record-id>.jsonl
```

`record-id` is derived from the SHA-256 of the canonical bytes and truncated to
24 hexadecimal characters by the current storage implementation.

### Legacy layout remains readable

Older repositories may contain:

```text
feedback/<unix-ms>.jsonl
contributions/<unix-ms>.jsonl
```

`deduplicate_dataset.py` reads recursively, so old flat files and new dated
files can coexist during migration.

### Custom folder paths

Each target can override:

```json
"paths": {
  "feedback": "records/feedback",
  "contributions": "records/contributions"
}
```

Paths are server-validated as relative repository folders. `..`, backslashes,
empty segments, excessive depth, and unsafe names are rejected.

---

## 14. Contribution record identity and deduplication

There are two intentionally separate collection paths:

| Path | Endpoint | Typical folder | Content semantics |
|---|---|---|---|
| Rating telemetry | `POST /v1/feedback` | `feedback/` | bounded rating/event mechanics only; never training-eligible |
| Explicit contribution | `POST /v1/contribute` | `contributions/` | user-reviewed Q&A or one ordered conversation record |

Feedback telemetry no longer carries query/answer content, so it is **not** joined
with contributions to form training examples. It remains `trainingStatus="telemetry"`
and is excluded from ordinary training output.

Contribution dedup keys are server-owned and receipt-scoped rather than stable
browser conversation identifiers:

```text
Q&A record:           <receipt-id>:<answer-index>
Conversation record:  <receipt-id>:conversation
```

The same key is reused by later eligible/withdrawal lifecycle records so
last-write-wins can suppress a withdrawn contribution without exposing a stable
client identity. Within a key, the newest server `_ts` wins. A whole conversation
remains one record throughout quarantine, promotion, and withdrawal.

---

## 15. Retraction/edit behavior

An edited rating can create a chain such as:

```text
rate A
  |
  v
retract A
  |
  v
rate B
```

Relevant v2 fields include:

```text
feedbackId
prevFeedbackId
editCount
action
_ts
```

Retraction tombstones participate in last-write-wins so an explicitly removed
rating can suppress an older rating. A tombstone is then **always removed** from
the clean training output.

If a retraction reaches the server but the replacement rating never arrives,
no training example is emitted for that key. This is safer than resurrecting a
rating the user explicitly withdrew.

---

# Part IV — `deduplicate_dataset.py`: old and new workflows

## 16. Legacy HF command remains supported

Existing automation can continue using:

```bash
python deduplicate_dataset.py \
  --repo-id scikit-plots/ai-assistant-contributions \
  --output clean_dataset.jsonl
```

`--repo-id` still defaults to provider `huggingface`.

For a private dataset, prefer an environment token rather than passing a raw
credential on the command line:

```bash
export HF_DATASET_READ_TOKEN=<hf-repo-write-token>

python deduplicate_dataset.py \
  --repo-id scikit-plots/ai-assistant-contributions \
  --token-env HF_DATASET_READ_TOKEN \
  --output clean_dataset.jsonl
```

A read-only token is sufficient for **deduplication/download**. It does not need
record-write permission.

> **Attention — `--token` is kept only for compatibility.**
> Command-line secrets can appear in shell history or process inspection. Use
> `--token-env` for new scripts.

---

## 17. Local snapshot/clone mode

New simplified form:

```bash
python deduplicate_dataset.py \
  --local-dir /tmp/ai-assistant-records \
  --output clean_dataset.jsonl
```

The old form is still accepted:

```bash
python deduplicate_dataset.py \
  --repo-id scikit-plots/ai-assistant-contributions \
  --local-dir /tmp/ai-assistant-records \
  --output clean_dataset.jsonl
```

In local mode, `--repo-id` is now optional because it was never required to read
the local files.

---

## 18. GitHub direct read

For a private GitHub repo, create a separate fine-grained token with **Contents:
read** if practical. Training/dedup does not need the write token used by
`app.py`.

```bash
export GITHUB_DATASET_READ_TOKEN=<github-repo-token>

python deduplicate_dataset.py \
  --provider github \
  --repo-id scikit-plots/ai-assistant-records \
  --branch main \
  --token-env GITHUB_DATASET_READ_TOKEN \
  --output clean_dataset.jsonl
```

The script downloads the GitHub repository archive through the official archive
endpoint, then performs guarded local extraction.

---

## 19. Use the same topology as `app.py`

This is the cleanest operational workflow.

Assume the environment already contains:

```text
RECORD_STORAGE_TARGETS
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY
AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR
```

Then:

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --output clean_dataset.jsonl
```

### Default behavior: Primary only

This intentionally reads only:

```text
role=primary
```

Why? Because mirrors contain copies of the same accepted records. Reading every
store naively would multiply training samples.

---

## 20. Read a specific mirror

Useful for confirming that the GitHub mirror can independently produce the same
clean dataset:

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --target-id github-mirror \
  --output clean_from_github.jsonl
```

Then compare outputs, for example:

```bash
sha256sum clean_dataset.jsonl clean_from_github.jsonl
```

If both repositories are synchronized and file ordering/source state is
identical, the clean outputs should match.

---

## 21. Read all Primary + Mirrors safely

For audit/recovery:

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --all-targets \
  --stats-only
```

Or produce a union:

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --all-targets \
  --output clean_union.jsonl
```

Multi-source safeguards:

1. Byte-identical mirrored files are loaded only once.
2. Legacy differently-named files containing the same normalized record are
   suppressed once across sources.
3. If two providers contain the **same canonical record-file ID** but different
   bytes, the script fails closed instead of guessing which copy is correct.
4. Normal `_dedup_key` rules still run after mirror suppression.

> **Important — `--all-targets` is read-only recovery/audit, not reverse sync.**
> It can construct a safe union for analysis. It does **not** push missing files
> from GitHub back to HF or vice versa.

---

## 22. Load topology from a file

You may keep the non-secret JSON in a file:

```bash
python deduplicate_dataset.py \
  --targets-file record_storage.json \
  --output clean_dataset.jsonl
```

`record_storage.json` should contain exactly the same JSON array accepted by
`RECORD_STORAGE_TARGETS`.

The file must contain only:

```json
"token_env": "AI_RECORD_STORAGE_TOKEN_..."
```

Never put raw tokens in that JSON file.

---

## 23. Direct provider support in the cleaner

| Provider | Direct `--provider` read | Storage write adapter | Notes |
|---|---:|---:|---|
| Hugging Face | Yes | Yes | Dataset snapshot API |
| GitHub | Yes | Yes | repository archive / Contents API |
| GitLab | Yes | Yes | repository archive / Repository Files API |
| Bitbucket Cloud | Yes | Yes | branch archive / source commit API |

For GitLab self-managed direct reads:

```bash
python deduplicate_dataset.py \
  --provider gitlab \
  --repo-id group/subgroup/project \
  --api-base https://gitlab.example.com/api/v4 \
  --token-env GITLAB_DATASET_READ_TOKEN
```

For Bitbucket, Bearer-capable repository/API access tokens are compatible with
the current adapter. A local clone remains a good fallback for unusual auth
setups.

---

## 24. Cleaner download safety

Remote archive handling includes:

- compressed archive size limit (`--max-archive-mb`, default 512 MiB),
- extracted-size limit (`--max-extract-mb`, default 2048 MiB),
- max archive member count,
- no absolute paths,
- no `..` traversal,
- no symlinks/hardlinks,
- no device/FIFO members,
- no provider response bodies or credentials in error logs.

Tune the limits only when you understand the expected repository size:

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --max-archive-mb 1024 \
  --max-extract-mb 4096 \
  --output clean_dataset.jsonl
```

---

# Part V — Multiple providers and senior-level tuning

## 25. Primary choice: HF or GitHub?

### HF as Primary

Good when:

- the records are fundamentally an ML dataset,
- downstream training already uses Hugging Face,
- Dataset browsing/versions are useful,
- GitHub is primarily backup/audit.

Recommended topology for this project:

```text
HF PRIMARY -> GitHub MIRROR
```

### GitHub as Primary

Good when:

- Git history/review is the operational system of record,
- repository policies/auditing are central,
- HF is mainly a downstream dataset replica.

```text
GitHub PRIMARY -> HF MIRROR
```

Do not choose a Primary merely because one provider is “first” in the JSON.
Set `role` explicitly.

---

## 26. Current write reliability behavior

The coordinator currently uses:

| Mechanism | Current behavior |
|---|---|
| Primary requirement | exactly one |
| Max targets | 8 |
| Per-target serialization | `asyncio.Lock` |
| Immediate attempts | 2 |
| Immediate retry backoff | 0.25 s exponential base |
| Mirror delayed retries | after ~2 s, 10 s, 30 s |
| Circuit breaker | opens after 3 failures |
| Circuit cool-down | 60 s |
| Client disconnect protection | accepted storage task kept by server + shielded |
| Mirror queue durability | in-memory only |

### Why per-target serialization matters

GitHub explicitly warns that conflicting repository Contents operations should
be serialized. The coordinator therefore uses one lock per target rather than
blindly running concurrent writes to the same repository target.

### What broken-pipe protection does

If a browser navigates away or the request socket closes after persistence has
been accepted, the server keeps a strong reference to the storage task and
shields it from request cancellation.

### What it does **not** guarantee

An in-memory retry task is not a durable queue.

```text
browser disconnect       -> protected
network blip             -> retries
mirror temporary failure -> retries/circuit
whole Space/container dies -> in-memory retry state is lost
```

For strict at-least-once mirror delivery across process/container restarts, add
an external durable queue/spool in a future architecture.

---

## 27. Multi-provider expansion examples

### HF Primary + GitHub + GitLab mirrors

```json
[
  {
    "id": "hf-primary",
    "provider": "huggingface",
    "role": "primary",
    "repo": "org/dataset",
    "token_env": "AI_RECORD_STORAGE_TOKEN_HF_PRIMARY",
    "token_type": "fine-grained"
  },
  {
    "id": "github-mirror",
    "provider": "github",
    "role": "mirror",
    "repo": "org/records",
    "token_env": "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR"
  },
  {
    "id": "gitlab-mirror",
    "provider": "gitlab",
    "role": "mirror",
    "repo": "org/records",
    "token_env": "AI_RECORD_STORAGE_TOKEN_GITLAB_MIRROR"
  }
]
```

### GitLab token note

The current adapter writes with the GitLab **Repository Files API**. Official
GitLab documentation states the `api` scope provides read-write access through
that API; `write_repository` is for Git-over-HTTP and does not by itself provide
API authentication. Prefer a project-scoped access token when your GitLab plan
and governance permit it.

### Bitbucket token note

The adapter writes through the Bitbucket Cloud source/commit API using Bearer
authentication. Use a repository-scoped token/API token with repository Write
permission rather than a broadly privileged user credential when possible.

---

## 28. Public links and private repository topology

Run 5 changed public discovery to a privacy-minimized contract. Server storage
topology is not an operator dashboard and repository/provider URLs are no longer
automatically exposed merely because a target has `expose_links=true`.

If readers are intentionally meant to see a public Dataset/repository link, publish
that identifier through the documentation/profile configuration. Private/confidential
server topology remains operator-side. Credentials are never public either way.

---

## 29. Branch strategy

Default:

```json
"branch": "main"
```

For high-volume collection, consider a dedicated data branch:

```json
"branch": "records"
```

Advantages:

- separates append-only records from project code,
- avoids branch protections intended for source code,
- reduces accidental human edits,
- makes retention/archive policy easier.

> **Attention — create the target branch first.**
> Provider APIs differ in behavior when asked to write to a branch that does not
> exist.

---

## 30. Path strategy

Simple:

```json
"paths": {
  "feedback": "feedback",
  "contributions": "contributions"
}
```

Namespaced:

```json
"paths": {
  "feedback": "records/feedback",
  "contributions": "records/contributions"
}
```

Keep the logical meaning consistent across mirrors whenever possible. It makes
manual comparison and incident recovery easier, even though the canonical file
content/ID is provider-neutral.

---

# Part VI — Migration without downtime

## 31. Legacy HF → HF Primary + GitHub Mirror

### Phase 0 — existing working configuration

```text
TRAINING_DATASET_REPO
HF_DATASET_TOKEN
HF_DATASET_TOKEN_TYPE
```

Verify one successful record first.

### Phase 1 — create GitHub repo/token

Do not modify the working HF path yet.

### Phase 2 — introduce `RECORD_STORAGE_TARGETS`

Configure HF as Primary using the same Dataset repo and a dedicated
`AI_RECORD_STORAGE_TOKEN_HF_PRIMARY` Secret.

### Phase 3 — add GitHub as Mirror

Add the second target and token Secret.

### Phase 4 — run a synthetic contribution

Verify same record ID on both providers.

### Phase 5 — run mirror-failure test

Verify Primary acceptance remains correct.

### Phase 6 — keep legacy variable briefly

Keeping:

```text
TRAINING_DATASET_REPO
```

for a rollback window is safe. The multi-target config remains authoritative.

### Phase 7 — optional cleanup

After older consumers are no longer needed, you may remove the legacy variable
and legacy token slots.

---

# Part VII — Troubleshooting

## 32. Startup / discovery troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| no `storage.targets` | old proxy build or no configured storage | deploy current proxy / inspect env |
| `missing-token` | `token_env` is configured but Secret missing | add exact Secret name |
| HF `denied-read-token` | token type is Read | use Fine-grained repo-write token |
| HF `denied` | repo permission/organization policy/invalid token | inspect HF token scope and repo membership |
| HF `unverified` | preflight unavailable or transient issue | test real write; modern Hub v1.x is recommended |
| GitHub 404/403/degraded | wrong repo/token/repository selection | verify selected repo and Contents permission |
| GitHub 409/422 | conflict/branch issue | verify branch and serialized writes |
| mirror stays degraded | credential/provider/branch failure | fix target, then submit/test again |

---

## 33. Configuration parser troubleshooting

### `TARGET_PRIMARY_COUNT`

You have zero or more than one Primary.

Correct:

```json
[
  {"id": "a", "role": "primary", ...},
  {"id": "b", "role": "mirror", ...}
]
```

### `TARGET_TOKEN_ENV`

Storage token environment names are intentionally restricted to:

```text
AI_RECORD_STORAGE_TOKEN_[A-Z0-9_...]
```

Do not point a target at arbitrary environment names such as `HOME` or an
unrelated secret.

### `TARGET_PATH`

Check folder names for traversal, empty segments, or unsupported characters.

---

## 34. Contribution succeeds on HF but not GitHub

That is a degraded mirror, not a lost Primary record.

Check:

1. GitHub token exists as a Space Secret.
2. Fine-grained PAT selected the correct repository.
3. Repository `Contents` permission is Read and write.
4. Branch exists.
5. Branch protection permits the token identity to commit.
6. Service diagnostics no longer shows circuit-open after cooldown.

Do not manually re-submit the user contribution simply to fix a mirror unless
you understand the duplicate consequences. The record ID/path is designed for
idempotent retries.

---

## 35. Cleaner says mirrors disagree

`deduplicate_dataset.py --all-targets` intentionally fails if the same canonical
record-file ID has different bytes on two stores.

Treat this as an integrity incident:

1. Do not train from the ambiguous union.
2. Compare the two files manually.
3. Determine which store contains the accepted canonical bytes.
4. Preserve both versions for audit before repairing anything.
5. Re-run the cleaner after reconciliation.

The tool deliberately does not auto-pick Primary in this case because silent
conflict resolution could hide corruption or unauthorized modification.

---

# Part VIII — Operational checklist

## 36. First-time operator checklist

- [ ] Proxy deployment is current.
- [ ] Exactly one storage target is Primary.
- [ ] Every `token_env` has a matching Space Secret.
- [ ] No raw token appears inside `RECORD_STORAGE_TARGETS`.
- [ ] HF persistence token is Fine-grained with target Dataset write permission.
- [ ] GitHub PAT is limited to selected repo + Contents read/write.
- [ ] `FEEDBACK_PERSIST_ENABLED=true` only if durable privacy-minimal rating telemetry is intentionally required.
- [ ] `CONTRIBUTION_REVIEW_TOKEN` is a separate operator capability when contribution promotion is enabled.
- [ ] Public discovery exposes only coarse storage readiness, not private storage topology.
- [ ] Synthetic contribution returns `status="quarantined"` and is not present in durable training storage.
- [ ] Pending-delete capability removes a still-quarantined synthetic receipt from the active review ledger without claiming forensic/global erasure.
- [ ] Post-promotion DELETE returns `status="withdrawn"`, persists privacy-minimal withdrawal tombstones, and ordinary dataset output no longer contains that contribution.
- [ ] Provider current-view removal is inspected separately from repository-history/backups; no global erasure claim is made.
- [ ] Authorized promotion creates `trainingStatus="eligible"` rows in Primary/Mirrors.
- [ ] Feedback persistence test, if enabled, creates only telemetry rows with no Q&A/comment/page/session content.
- [ ] `deduplicate_dataset.py --from-storage-config --stats-only` succeeds and excludes telemetry/quarantined/unreviewed rows.
- [ ] Raw/quarantined records are never used directly for training.

---

## 37. Senior/operator periodic checklist

- [ ] Review token expiry and rotate before expiration.
- [ ] Keep one token per application/purpose where feasible.
- [ ] Re-check repo scopes after repository/org policy changes.
- [ ] Monitor target status / last failure / pending retries.
- [ ] Periodically compare Primary and mirror clean outputs.
- [ ] Run `--all-targets --stats-only` after incidents.
- [ ] Treat mirror conflicts as integrity incidents, not ordinary duplicates.
- [ ] Keep archive/extraction limits appropriate to expected dataset growth.
- [ ] Consider a durable external queue if mirror delivery must survive container loss.
- [ ] Review retention/privacy policy before promoting quarantined contribution content; never train from raw intake.

---

# Part IX — Verified implementation behavior

## 38. What is implemented now

### `app.py` / `_utils/_storage.py`

- provider-neutral target schema;
- Hugging Face, GitHub, GitLab, Bitbucket Cloud write adapters;
- exactly one Primary + optional Mirrors;
- legacy HF synthesis when `RECORD_STORAGE_TARGETS` is absent;
- Fine-grained / Read / Write HF token classification;
- repo-write preflight with modern `huggingface_hub`;
- Read-token write blocking;
- canonical content-derived record ID;
- date-sharded stable file paths;
- per-target serialization;
- bounded immediate retries;
- mirror delayed retries;
- circuit state;
- accepted-task shielding from browser disconnect;
- public provider links generated server-side;
- credentials excluded from discovery;
- multi-target config suppresses obsolete legacy HF-token warnings.

### `deduplicate_dataset.py`

- old `--repo-id` HF command remains valid;
- old `--repo-id + --local-dir` remains valid;
- `--local-dir` now works without a dummy repo ID;
- direct HF/GitHub/GitLab/Bitbucket source modes;
- `--token-env` preferred while `--token` remains compatible;
- `--from-storage-config` uses the same target parser as `app.py`;
- Primary selected by default;
- `--target-id` reads a chosen mirror;
- `--all-targets` safely unions mirrors;
- identical mirror files suppressed;
- exact legacy duplicate records across mirrors suppressed;
- same canonical record ID + different bytes fails closed;
- guarded remote archive extraction;
- historical schema normalization to current schema v4 when `_utils/_dataset_schema.py` is available;
- default exclusion of feedback telemetry/quarantined/legacy-unreviewed records;
- training acceptance only for `trainingStatus="eligible"` contributions;
- tombstone/retraction handling where applicable;
- deterministic NDJSON output.

---

## 39. What is intentionally **not** implemented

### No browser-side storage credential management

The AI panel consumes only coarse public storage readiness plus any repository
identifier the documentation author explicitly chose to publish. It cannot read
private server storage topology, change server write tokens, or choose an arbitrary
repository.

### No automatic bidirectional repository sync

Current replication is:

```text
Primary -> Mirrors
```

not:

```text
HF <-> GitHub <-> GitLab <-> Bitbucket
```

The cleaner can read a union for recovery, but does not mutate repositories.
This prevents sync loops and ambiguous conflict resolution.

### No durable retry queue across container death

Delayed mirror retry state is in memory. Add an external queue/spool if this
becomes a hard delivery requirement.

---

# Part X — Official reading / verification sources

The implementation and this guide were cross-checked against the following
provider documentation on **2026-08-28**.

## Hugging Face

1. **Spaces — Secrets and environment variables**
   https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets-and-environment-variables
   Confirms Variables are for non-sensitive config and Secrets are for tokens,
   API keys, and sensitive credentials.

2. **User Access Tokens**
   https://huggingface.co/docs/hub/security-tokens
   Defines Fine-grained, Read, and Write roles; recommends Fine-grained tokens
   for production and warns against exposing token values.

3. **`HfApi.auth_check`**
   https://huggingface.co/docs/huggingface_hub/main/package_reference/hf_api
   Current API supports `write=True` to verify content-write permission on a
   repository.

4. **`snapshot_download` / download guide**
   https://huggingface.co/docs/huggingface_hub/main/guides/download
   Documents downloading a complete Dataset repository snapshot.

5. **Spaces variables/secrets from `huggingface_hub`**
   https://huggingface.co/docs/huggingface_hub/en/guides/manage-spaces

## GitHub

6. **Create or update repository file contents**
   https://docs.github.com/en/rest/repos/contents#create-or-update-file-contents
   Confirms fine-grained tokens need `Contents` repository permission with
   write access and warns that conflicting content operations should be
   serialized.

7. **Manage fine-grained Personal Access Tokens**
   https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
   Documents selecting a resource owner, restricting repository access, and
   granting the minimum permissions needed.

8. **Download repository archive**
   https://docs.github.com/en/rest/repos/contents#download-a-repository-archive-tar
   Confirms private archive download with a fine-grained token requires
   `Contents: read`.

## GitLab

9. **Repository Files API**
   https://docs.gitlab.com/api/repository_files/
   Documents repository-file create/update APIs and states the `api` token
   scope provides read-write access through this API.

10. **Access token scopes**
    https://docs.gitlab.com/security/tokens/access_token_scopes/
    Distinguishes `api`, `read_api`, `read_repository`, and `write_repository`.

11. **Repository archive API**
    https://docs.gitlab.com/api/repositories/#retrieve-file-archive-from-a-repository

## Bitbucket Cloud

12. **Bitbucket Cloud REST authentication**
    https://developer.atlassian.com/cloud/bitbucket/rest/
    Documents repository/project/workspace access tokens and Bearer auth.

13. **Repository access-token permissions**
    https://support.atlassian.com/bitbucket-cloud/docs/repository-access-token-permissions/
    Documents repository Read and Write scopes.

14. **API token permissions**
    https://support.atlassian.com/bitbucket-cloud/docs/api-token-permissions/

15. **Repository/source API**
    https://developer.atlassian.com/cloud/bitbucket/rest/api-group-source/
    Documents browsing source and creating commits by uploading files.

16. **Repository archive download**
    https://support.atlassian.com/bitbucket-cloud/kb/how-to-download-repositories-using-the-api/

---

## 40. Recommended scikit-plots production baseline

For this project today, the balanced setup is:

```text
Browser
   |
   v
scikit-plots-ai.hf.space
   |
   v
Hugging Face Dataset PRIMARY
scikit-plots/ai-assistant-contributions
   |
   +----> GitHub MIRROR
          scikit-plots/ai-assistant-records
```

Use:

```text
HF:      Fine-grained token scoped to the target Dataset with write permission
GitHub:  Fine-grained PAT restricted to the mirror repo + Contents read/write
```

Store actual tokens only as Space **Secrets**. Store the topology JSON as a
**Variable** unless its repository metadata is itself confidential.

For training:

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --output clean_dataset.jsonl
```

This reads the Primary by default, applies schema normalization, suppresses
feedback/contribution duplicates, removes retraction tombstones, and produces
the clean NDJSON artifact to feed downstream training/analysis.

For periodic mirror audit:

```bash
python deduplicate_dataset.py \
  --from-storage-config \
  --all-targets \
  --stats-only
```

That is the operational separation to preserve:

```text
collect once -> persist Primary -> replicate Mirrors -> deduplicate Primary -> train
                                      |
                                      +-> audit mirrors separately
```


### Shared receipt authority across replicas

For horizontally scaled HF proxy replicas, use one shared Redis consistency domain:

```text
CONTRIBUTION_LEDGER_BACKEND=redis
CONTRIBUTION_LEDGER_REDIS_URL=rediss://<shared-redis>
CONTRIBUTION_LEDGER_KEY_SECRET=<dedicated-random-secret-at-least-32-bytes>
CONTRIBUTION_REQUIRE_SHARED=true
```

The Redis ledger atomically coordinates receipt creation, promotion claims, pending
delete, withdrawal claims, and terminal transitions across replicas. Receipt IDs are
HMAC-pseudonymized before becoming Redis key material, and operation claims use
bounded leases. Withdrawal can continue monotonically after an expired lease; an
expired **promotion** claim instead becomes `promotion_uncertain` and blocks
re-promotion, because Redis cannot fence a paused worker from an external Git/provider
side effect.

This closes cross-replica **coordination authority** only. The proxy does not infer
Redis AOF/RDB/managed-service persistence guarantees, backup retention, or disaster
recovery policy. Keep `CONTRIBUTION_REQUIRE_DURABLE` as a separate deployment
requirement and verify the external Redis durability contract independently.


## Run 25 / B44 provider response boundaries

Storage-provider control traffic is a small metadata plane, not a bulk-download plane. Its response ceiling defaults to **4 MiB** and is hard-clamped to **16 MiB**. GitHub/GitLab/Bitbucket mutations that need no response body do not consume one; metadata reads are stream-bounded before JSON parsing. Hugging Face Hub control calls run with a scoped bounded lower-level HTTP client and restore the prior SDK client factory immediately afterward. Intentional large offline dataset/model downloads remain separately scoped (`SEC-P2-49`).


## Provider-native review workflow

For human-maintained datasets, prefer `CONTRIBUTION_REVIEW_MODE=provider-pr`. The proxy opens the provider's normal code-review object against the configured canonical branch instead of writing submitted content directly to that branch. This works with Hugging Face Pull Requests, GitHub Pull Requests, GitLab Merge Requests, and Bitbucket Cloud Pull Requests. Merge = eligible; close/decline = reject. Keep the dataset repository private when submissions may contain non-public material, and keep provider tokens server-side only.

The Primary target is the review authority. Mirrors remain ordinary replication targets; do not treat a mirror branch as an independent eligibility decision domain. In `provider-pr` mode, a maintainer merge changes the Primary canonical branch only; current code does not synchronously fan that external merge out to Mirrors. If immediate mirror convergence is required, use one Primary during review or run an explicit post-merge reconciliation/replication job. Mirrors never decide eligibility.
