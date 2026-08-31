---
title: sphinx-ai-assistant proxy
emoji: 🔁
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - inference-api
license: bsd-3-clause
short_description: Thin OpenAI-compatible proxy for sphinx-ai-assistant
---

# sphinx-ai-assistant proxy

Server-authoritative proxy for the **sphinx-ai-assistant** Sphinx
extension.  Runs as a free CPU Docker Space on HuggingFace.  Accepts
unauthenticated requests from the browser widget, resolves the upstream model
backend, injects the required auth header server-side, and returns the
response — keeping all tokens out of the browser.

---

> **Security contract:** browser requests use `scikitplot-chat-v1`; the proxy constructs the authoritative system role and rejects client `system`/`developer`/tool authority.

## How it works — routing decision tree

Every `POST /v1/chat/completions` is routed through three ordered paths.
The first matching path wins.

```
Browser  ──POST /v1/chat/completions──▶  This proxy
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                     BACKEND_URL set?   model namespace      fallback
                          │             in NAMESPACES?           │
                       Path 1              Path 2             Path 3
                          │                   │                   │
                   Custom backend      ai-model Space      HF Serverless
                   (DMR / Ollama /     ZeroGPU CPU/GPU     Inference API
                    any HTTP server)   (scikit-plots/*)    (Qwen/*, etc.)
                          │                   │                   │
                   BACKEND_AUTH_TOKEN    HF_SPACES_AUTH_TOKEN    HF_TOKEN (HF router only)
                   (if set)             (Path-2 Space      (required)
                                         handles it)
```

Each path has its own independent read timeout so slow CPU inference
(Path 2, ~4–5 min) and fast GPU inference (Path 3, ~30–90 s) coexist
on the same proxy without interfering.

---

## Files in this Space

| File | Purpose |
|---|---|
| `app.py` | FastAPI proxy application — deployment entrypoint and route handlers |
| `deduplicate_dataset.py` | Canonical dataset reader/deduplicator CLI |
| `_utils/` | Private Python implementation package used by `app.py` and the deduplication CLI |
| `_utils/_shared_logic.py` | Shared proxy/service routing, validation, and redaction helpers |
| `_utils/_share_contract.py` | Global Share schema validation, rendering, and edit-capability helpers |
| `_utils/_share_store.py` | Global Share lifecycle authority (`memory` / `sqlite` / `redis`) with capability-digest and bounded-store semantics |
| `_utils/_redis_security.py` | One Redis URL/TLS policy shared by rate, Share, and contribution authorities; strict mode forces verified TLS |
| `_utils/_contribution_ledger.py` | Capability-hashed contribution receipt lifecycle authority |
| `_utils/_dataset_schema.py` | Canonical feedback/contribution schema and normalization |
| `_utils/_storage.py` | Provider-neutral record storage coordinator |
| `Dockerfile` | Digest-pinned two-stage, non-root production container |
| `.dockerignore` | Deny-by-default Docker build-context allowlist |
| `requirements.txt` | Human-readable exact direct-dependency manifest |
| `requirements.lock` | Complete Linux/amd64 CPython 3.11 binary-wheel closure with SHA-256 hashes |
| `security/` | Offline lock/SBOM verifier, policy, CycloneDX Python SBOM, and networked release gates |
| `docker-compose.hardened.reference.yml` | Operator reference for read-only/rootless/capability-dropped deployment |
| `README.md` | This file — HF Space metadata + full documentation |
| `../DATASET_CONTRIBUTION_GUIDE.md` | Reader + maintainer contribution lifecycle, native review, receipt management, and scenario guide |
| `DATASET_COLLECTION_GUIDANCE.md` | Deep multi-store operations, provider topology, migration, deduplication, and training-data assembly |

> **Critical** — commit the complete `_utils/` package with `app.py`. The Dockerfile copies
> `_utils/` as a directory so helper dependencies cannot be accidentally omitted one-by-one.
> The only supported root-level Python entrypoints are `app.py` and `deduplicate_dataset.py`.

---

## Endpoints

| Method | Path | Purpose | Notes |
|---|---|---|---|
| `GET`  | `/`                    | Status page — routing config, token states, `contribute_ready` flag | Always 200 while running |
| `GET`  | `/health`              | Minimal liveness probe for container orchestrators | Always `{"status":"ok"}` |
| `HEAD` | `/`                    | Health-monitor probe (no body) | Required by HF uptime monitor |
| `HEAD` | `/health`              | Health-monitor probe (no body) | Required by HF uptime monitor |
| `POST` | `/`                    | Backward-compat alias for `/v1/chat/completions` | Identical behaviour |
| `POST` | `/v1/chat/completions` | Primary proxy — routes to Path 1 / 2 / 3 | Negotiates SSE vs JSON; never relabels JSON as SSE |
| `POST` | `/v1/feedback`         | Receive 👍/👎 rating; optionally persist the canonical record through the configured Primary + Mirrors | Rate-limited: 30/IP/hour |
| `POST` | `/v1/share`            | Validate/store a structured snapshot; returns a fixed-path fragment URL and supports recoverable create-once semantics | Server owns representation/MIME; storage is `memory`, `sqlite`, or `redis` according to deployment policy |
| `GET`  | `/v1/share`            | Serve the fixed Share viewer shell | Public locator stays in `#share=...`; fragment is browser-local |
| `POST` | `/v1/share/read`       | Resolve/render a public read-only snapshot | Locator is JSON body data, never request-path data |
| `POST` | `/v1/share/status`     | Content-free lifecycle probe | Locator is JSON body data; `no-store` |
| `POST` | `/v1/share/update`     | Replace snapshot while preserving public fragment URL | Requires `X-Share-Edit-Token`; rate-limited |
| `POST` | `/v1/share/revoke`     | Revoke the share | Requires `X-Share-Edit-Token` |
| legacy | `/v1/share/{uuid}`     | Bounded pre-generation compatibility: `HEAD`/`GET` and authenticated `DELETE`; `PATCH` is retired with `410` | Only objects lacking generation-2 metadata are eligible; responses advertise `Deprecation`/`Sunset`/successor headers |
| `POST` | `/v1/contribute`       | Consent-gated contribution intake into the mutable receipt ledger | Rate-limited: 5/client/hour |
| `PUT` | `/v1/contribute/{receipt}` | Update the same pending review | Requires `X-Contribution-Delete-Token`; identical content is a no-op, changed content becomes a new revision on the same PR/MR |
| `GET` | `/v1/contribute/{receipt}` | Read content-free receipt lifecycle status | Requires `X-Contribution-Delete-Token` |
| `DELETE` | `/v1/contribute/{receipt}` | Delete still-pending intake or withdraw an already-promoted contribution from training use | Requires `X-Contribution-Delete-Token`; never claims repository-history erasure |
| `POST` | `/v1/contribute/{receipt}/promote` | Optional API-driven merge/promotion | Requires `CONTRIBUTION_REVIEW_TOKEN`; in `provider-pr` mode it merges the native PR/MR instead of creating a second direct commit |

> **Dataset operators:** start with [`../DATASET_CONTRIBUTION_GUIDE.md`](../DATASET_CONTRIBUTION_GUIDE.md)
> for the reader/maintainer workflow and scenario guide. Use
> [`DATASET_COLLECTION_GUIDANCE.md`](./DATASET_COLLECTION_GUIDANCE.md) for deep
> provider-storage, migration, deduplication, and multi-store operations.

### Global Share security contract

`POST /v1/share` accepts only a structured schema-v2 `snapshot`, an allowlisted
`format` (`html`, `json`, `txt`, `yaml`, or `toml`), and `ttlDays`. The client cannot select the
response MIME type, extension, or submit arbitrary rendered HTML for hosting.
The server canonicalizes the snapshot again before storage and renders the
response itself on fixed `POST /v1/share/read`. New public URLs use
`/v1/share#share=<locator>`; the fragment is consumed by the browser viewer and
current read/status/update/revoke requests place the locator in a bounded JSON
body on a fixed endpoint path. This keeps the public read capability out of
ordinary request-URL/access-log fields. Every new or fixed-path-updated object is
stamped with Share transport generation 2. Legacy capability-bearing path routes
serve only pre-generation objects; generation-2 objects return `404` on those
paths. Legacy `PATCH` is retired (`410`) so old path transport cannot extend its
own lifetime; `POST /v1/share/update` migrates an old object to generation 2.
Legacy `HEAD`/`GET`/authenticated `DELETE` remain only while a pre-generation
object is live and advertise `Deprecation`, object-expiry `Sunset`, and a fixed
viewer successor link. Expiry remains explicit: current fixed-path and eligible
legacy operations return `410 Gone` for confirmed expiry.

A successful legacy create returns two separate capabilities:

- `uuid` / `url` — public **read-only** capability;
- `editToken` — private mutation capability required in `X-Share-Edit-Token`
  for current fixed `/update` and `/revoke` operations (and bounded legacy
  `DELETE` during migration). The server stores only its SHA-256 digest.

Current browser clients use the stronger Run-18 recovery envelope. Before the
request, the browser generates the public locator and private management token,
then sends only the locator, an operation ID, and `SHA-256(managementToken)` in
bounded headers. The raw edit/revoke token stays in page memory and is never
part of the create request or create response. If the create response is lost,
the exact same operation can be retried; identical payload + operation +
management digest resolves the existing Share, while conflicting reuse returns
`409`. HTTP/network failure is therefore represented as **outcome unknown**, not
as proof that no public object exists.

Application logs intentionally omit both capabilities and conversation text.
The bundled Uvicorn command disables access logs because a path-based bearer
URL would otherwise appear in the request line. Operators must apply the same
rule to any upstream CDN/reverse-proxy access logs.

Resource defaults are independent from chat payload limits: 500 KiB per Share,
256 live entries, and 16 MiB aggregate storage. Set
`SHARE_MAX_BODY_BYTES`, `SHARE_MAX_ENTRIES`, and `SHARE_MAX_TOTAL_BYTES` lower
when appropriate. The default `SHARE_STORE_BACKEND=memory` is compatibility-only
and is truthfully reported as process-local/non-durable. Use `sqlite` for one
restart-durable local authority, or `redis` for one shared atomic authority
across replicas. `SHARE_REQUIRE_DURABLE=true` and `SHARE_REQUIRE_SHARED=true`
fail writes closed when the selected deployment cannot substantiate those
properties. Redis persistence is never inferred merely because Redis is shared;
set `SHARE_REDIS_DURABILITY_CONFIRMED=true` only after the operator has verified
the external Redis AOF/RDB/managed-service durability contract.

`SHARE_PUBLIC_BASE_URL` may be set to the externally visible
HTTPS base when reverse-proxy ASGI metadata is internal. On Hugging Face Spaces
the proxy automatically prefers the platform-provided `SPACE_HOST` and emits
`https://<SPACE_HOST>`, so the internal HTTP listener does not leak into public
Share URLs. An explicit `http://<same SPACE_HOST>` value is upgraded to HTTPS;
unrelated remote HTTP bases still fail closed. `TRUST_X_FORWARDED_FOR` defaults
to false; enable it only when the ingress proxy is known to overwrite
caller-supplied forwarding headers.


### Pending-review continuity

In `provider-pr` mode the first accepted submission binds the receipt to the
provider-native review ID. Subsequent updates authenticated by the same
participant management capability use `PUT /v1/contribute/{receipt}`.

- identical reviewed content -> `reviewUpdate="unchanged"`, no provider commit;
- changed reviewed content -> `reviewUpdate="updated"`, `reviewRevision += 1`;
- Hugging Face -> update `refs/pr/N`;
- GitHub/GitLab/Bitbucket -> update the existing source branch;
- merged/closed review -> HTTP 409; the proxy does not silently open a replacement;
- normal status/update/withdraw -> direct lookup by persisted provider review ID;
- bounded repository scanning -> legacy/recovery fallback only.

This makes the provider's ordinary PR/MR page the reviewer dashboard even at
large queue sizes: one receipt stays in the same review thread rather than
opening one new thread per click.

### Participant recovery and maintainer support references

The browser can persist participant management authority outside tab state as either a
private JSON receipt or a compact **private withdrawal code** using the `aicm2.…` format. Both carry the
management capability and must remain secret. They are accepted only against the
currently configured contribution endpoint; an imported receipt never redirects the
browser to an arbitrary endpoint from the file.

Provider-native responses/status also return bounded **non-secret** locator metadata:
`reviewProvider`, numeric `reviewId`, and the stable `reviewPath` such as
`contributions/YYYY/MM/DD/ct_<review-key>.jsonl`. Provider review URLs, repository
tokens, contribution content, and the management token are not included in that
support metadata. This gives participants a safe reference to send maintainers if an
old receipt can no longer be resolved.

A saved capability does not change `CONTRIBUTION_QUARANTINE_TTL_SECONDS` or ledger
durability. Use SQLite/Redis for durable receipt management; treat the non-secret
support reference as the fallback for an expired/lost pending lifecycle record.

### `POST /v1/contribute` — payload schema

The current browser sends **schema v4** with consent **2.0.0**. The endpoint
supports two explicit contribution record families. A whole conversation is one
ordered record:

```json
{
  "schemaVersion": 4,
  "consentFlag": true,
  "consentVersion": "2.0.0",
  "page": "https://your-docs-site/index.html",
  "model": null,
  "records": [
    {
      "recordType": "conversation",
      "messages": [
        {"role": "user", "content": "What is a confusion matrix?", "ts": 1781002584000},
        {
          "role": "assistant",
          "content": "A confusion matrix is...",
          "ts": 1781002584724,
          "model": {"id": "...", "provider": "...", "model": "..."},
          "feedback": {"ratingValue": 2, "ratingLabel": "helpful", "note": "optional"}
        }
      ],
      "message": "optional contribution-level reviewer note",
      "ts": 1781002585000
    }
  ]
}
```

The dedicated **Contribute to dataset** sheet also supports **This Q&A** and
**Rated answers** scopes. Those produce `recordType="qa"` records using the
historical `query` / `answer` shape. The current v4 envelope always requires
`consentVersion="2.0.0"`; legacy schema v2/v3 clients remain accepted with their
historical `1.0.0` consent for compatibility, but legacy consent is not accepted
for the broader v4 contract.

Ordinary `/v1/feedback` remains privacy-minimal rating telemetry and never turns
into content contribution automatically. The browser keeps ratings local unless
the reader explicitly enables **Send rating telemetry**. Current telemetry uses a
versioned permission (`telemetryConsent=true`, `telemetryConsentVersion="1.0.0"`)
and the proxy rejects feedback requests that do not carry that current consent
contract. Enabling telemetry does not contribute the question, answer, note,
page, model, or conversation content. Turning telemetry off stops future sends;
it does not claim erasure of telemetry already accepted by a remote provider.

Each accepted contribution record receives a server receipt-scoped dedup key,
but raw accepted content is **not** written to provider repositories immediately.
It enters the mutable contribution receipt ledger first with
`trainingStatus="quarantined"` in the mutable receipt ledger. In the optional `provider-pr` workflow, the proxy also prepares the future `trainingStatus="eligible"` bytes on a provider-native review ref; the canonical branch remains unchanged until merge. Only an independently authorized promotion or native provider merge makes the record eligible.

In `provider-pr` mode reviewers use the repository UI they already know: Hugging Face Pull Requests, GitHub Pull Requests, GitLab Merge Requests, or Bitbucket Pull Requests. Closing/declining rejects the pending review; merging into the configured canonical branch (`main` by default) is the training-eligibility boundary. The browser never receives provider credentials or maintainer review authority.

The compatibility `ledger` mode keeps the historical behavior where only an independently authorized promotion writes
an `eligible` row beneath `contributions/YYYY/MM/DD/<record-id>.jsonl` in the
Primary/Mirrors. Whole-conversation mode contributes one ordered conversation
record; rated-answer mode contributes only explicitly rated Q&A records; the
single-Q&A shortcut contributes only the selected Q&A after the same inspect,
privacy-review, and consent flow.

The receipt's delete capability remains meaningful after promotion. While
pending it removes the content from the active review ledger. After promotion
the same capability records a privacy-minimal `withdraw` tombstone keyed by the
server-owned contribution dedup key and attempts best-effort removal from each
provider's current branch/view. Dataset construction applies the withdrawal
through last-write-wins and excludes both the withdrawn eligible row and the
tombstone itself from training output. Versioned repository history, database
pages/WAL remnants, backups, CDN/provider logs, and infrastructure snapshots are
**not** claimed physically erased.

---

## Configuration

Configure the Space from **Settings → Variables and secrets**. Hugging Face exposes
Variables as non-sensitive deployment configuration and keeps Secrets private. See
[Managing Spaces variables and secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets).

### Variables and secrets

Use one simple rule: **configuration goes in Variables; credentials and capabilities go in Secrets**.
Both surfaces become environment variables inside the container, so putting a non-secret value
in Secrets still works technically, but it makes maintenance harder and hides which settings are
actually sensitive.

#### Variables — public / non-sensitive

| Name | Required? | Typical value | Notes |
|---|---:|---|---|
| `RECORD_STORAGE_TARGETS` | Multi-store deployments | JSON array | Provider-neutral Primary + Mirrors topology. May contain repo IDs and `token_env` **names**, never token values. If repository topology itself is confidential, it is acceptable to store this JSON as a Secret instead. |
| `TRAINING_DATASET_REPO` | Legacy HF-only mode | `scikit-plots/ai-assistant-contributions` | Repository identifier, not a credential. If this is currently stored as a Secret, it can be moved to Variables unless you intentionally treat the repo identity as confidential. Ignored as the active topology when `RECORD_STORAGE_TARGETS` is set. |
| `ALLOWED_MODELS` | No | comma-separated model IDs | Exact Path-3 model allow-list. Do not use `*`; an explicit model list prevents the proxy from becoming a general-purpose inference relay. |
| `HF_SPACES_MODEL_NAMESPACES` | No | `scikit-plots` | Model owner prefixes routed through the configured Path-2 model Space. |
| `ALLOWED_ORIGINS` | Custom sites only | comma-separated origins | Exact browser origins such as `https://docs.example.org`. Origins contain only scheme + host (+ optional port): no path, query, fragment, or trailing page URL. |
| `ALLOWED_ORIGINS_MODE` | No | `additive` | `additive` keeps the bundled Scikit-plots origins and adds `ALLOWED_ORIGINS`; `replace` trusts only `ALLOWED_ORIGINS` and is the recommended mode for forks/downstream sites that want their own CORS boundary. |
| `CONTRIBUTION_REVIEW_MODE` | No | `provider-pr` or `ledger` | Native provider PR/MR review or historical ledger review. |

| `HF_TOKEN_TYPE` | Recommended | `fine-grained` / `read` / `write` | Non-secret classification label for `HF_TOKEN`; avoids `unknown` startup diagnostics. |
| `HF_DATASET_TOKEN_TYPE` | Legacy HF persistence | `fine-grained` | Non-secret classification label for the dataset-persistence token. |

The proxy has two built-in browser origins because this extension is currently deployed on both
Scikit-plots documentation sites:

```text
https://scikit-plots.github.io
https://scikit-plots-learn.readthedocs.io
```

With the default `ALLOWED_ORIGINS_MODE=additive`, both remain trusted and any
`ALLOWED_ORIGINS` entries are appended. A downstream/open-source deployment can take complete
control without editing `app.py`:

```text
# Custom/fork deployment: trust only these sites
ALLOWED_ORIGINS=https://docs.example.org,https://learn.example.org
ALLOWED_ORIGINS_MODE=replace
```

To add a site while retaining the two Scikit-plots defaults:

```text
ALLOWED_ORIGINS=https://preview.example.org
ALLOWED_ORIGINS_MODE=additive
```

Never use `ALLOWED_ORIGINS=*` in production. CORS is a browser abuse boundary, not authentication;
server-to-server clients without an `Origin` header still rely on their own token/capability controls.

#### Secrets — private / server-only

| Name | Required? | Purpose |
|---|---:|---|
| `HF_TOKEN` | Path 3 inference | Hugging Face inference credential. Prefer least privilege; it should not also be your broad repository-write token. |
| `AI_RECORD_STORAGE_TOKEN_HF_PRIMARY` | When referenced by `RECORD_STORAGE_TARGETS` | Example HF Primary write credential. The exact name is configurable through each target's `token_env`. |
| `AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR` | When referenced by `RECORD_STORAGE_TARGETS` | Example GitHub Mirror write credential. Use an independent least-privilege token. |
| `AI_RECORD_STORAGE_TOKEN_*` | Per configured target | Provider-specific HF/GitHub/GitLab/Bitbucket write credential. Only names with this prefix are accepted by storage target configuration. |
| `HF_DATASET_TOKEN` | Legacy HF-only persistence | Preferred legacy dataset token; fine-grained to the target dataset repository. |
| `CONTRIBUTION_REVIEW_TOKEN` | Optional | API-driven review/promotion capability. Not required when maintainers review entirely through provider-native PR/MR UI. |
| `RATE_LIMIT_IDENTITY_SECRET` | Redis rate limiting | HMAC key used to pseudonymize shared rate-limit identities. |
| `CONTRIBUTION_LEDGER_KEY_SECRET` | Redis contribution ledger | HMAC key used to pseudonymize receipt identifiers. |
| Redis URLs containing credentials | When Redis is used | Keep `RATE_LIMIT_REDIS_URL`, `SHARE_STORE_REDIS_URL`, and `CONTRIBUTION_LEDGER_REDIS_URL` private when they contain usernames/passwords/tokens. |

A practical Scikit-plots Space layout is therefore:

```text
# Variables
RECORD_STORAGE_TARGETS=<provider-neutral JSON topology>
ALLOWED_MODELS=openai/gpt-oss-20b,Qwen/Qwen2.5-Coder-7B-Instruct,Qwen/Qwen2.5-Coder-32B-Instruct,scikit-plots/gpt-oss-20b,scikit-plots/Qwen2.5-Coder-7B-Instruct,scikit-plots/Qwen2.5-Coder-32B-Instruct
HF_SPACES_MODEL_NAMESPACES=scikit-plots
ALLOWED_ORIGINS_MODE=additive
# ALLOWED_ORIGINS may remain empty because both current Scikit-plots sites are built in.

# Legacy-only variable (not needed as the active topology when RECORD_STORAGE_TARGETS is used)
TRAINING_DATASET_REPO=scikit-plots/ai-assistant-contributions

# Secrets
HF_TOKEN=<inference-token>
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY=<repo-scoped-write-token>
AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR=<repo-scoped-write-token>
```

The proxy never reads provider token values from `RECORD_STORAGE_TARGETS`; it reads only the
configured `token_env` name and then resolves that environment variable server-side. Keep token
values out of `conf.py`, generated Sphinx HTML, JavaScript, logs, repository URLs, and commit metadata.

Start with [../DATASET_CONTRIBUTION_GUIDE.md](../DATASET_CONTRIBUTION_GUIDE.md) for contribution review and lifecycle behavior. Then use [DATASET_COLLECTION_GUIDANCE.md](./DATASET_COLLECTION_GUIDANCE.md) for exact HF/GitHub/GitLab/Bitbucket storage, Primary+Mirror, migration, testing, and deduplication recipes.

### Tokens

| Variable | Required? | Scope | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (for Path 3) | Read / inference | Inference-only token forwarded to the HF model backend. Prefer a read-only or fine-grained token that has no dataset-write permission. |
| `HF_TOKEN_TYPE` | No | — | Declares `fine-grained` \| `read` \| `write`. A classic `write` inference token triggers a least-privilege warning. |
| `HF_DATASET_TOKEN` | No | Dataset persistence | **Preferred dataset token.** Use a fine-grained token scoped to write only the target dataset repo. A classic `write` token is also supported, but is broader than necessary. |
| `HF_DATASET_TOKEN_TYPE` | No | — | Declares `fine-grained` \| `read` \| `write`. `read` is rejected for persistence. Fine-grained access is checked against the target dataset when the installed `huggingface_hub` supports `auth_check(..., write=True)`; otherwise the first commit verifies capability. |
| `HF_WRITE_TOKEN` | No | Legacy alias | Backward-compatible alias for `HF_DATASET_TOKEN`. Despite its historical name, it may contain a repo-scoped fine-grained token; it does **not** have to be a classic `write` token. |
| `HF_WRITE_TOKEN_TYPE` | No | Legacy alias | Type declaration for legacy `HF_WRITE_TOKEN`. Prefer `HF_DATASET_TOKEN_TYPE` for new deployments. |

> **Why separate inference and dataset tokens?** Least privilege. `HF_TOKEN` can be
> forwarded to model backends, so it should not be able to modify repositories.
> `HF_DATASET_TOKEN` never leaves the proxy and should be fine-grained to the
> one dataset repo it needs to update.

Dataset-token precedence is:

```text
HF_DATASET_TOKEN        preferred
        ↓
HF_WRITE_TOKEN          legacy alias
        ↓
HF_TOKEN                backward-compatible single-token fallback
```

The `/` status page reports token **type/capability**, never token values:

```json
"tokens": {
  "hf_token_set":          true,
  "hf_token_type":         "fine-grained",
  "hf_dataset_token_set":  true,
  "hf_dataset_token_type": "fine-grained",
  "hf_write_token_set":    false,
  "hf_write_token_type":   "unknown",
  "least_privilege_mode":  true
},
"training": {
  "dataset_repo":     "scikit-plots/ai-assistant-contributions",
  "contribute_ready": true
}
```

The provider-neutral `storage.targets[].token.write_capability` field adds the
important distinction between a token label and what it can actually do:

- `fine-grained` + `verified` — preferred; repo-specific write access verified.
- `fine-grained` + `unverified` — verification unavailable/transient; first real commit proves capability.
- `read` + `denied-read-token` — blocked before any persistence attempt.
- `write` — supported, but shown as broad permission so operators can tighten it.
- `denied` — an explicit 401/403 permission check failed; persistence is not attempted.

`least_privilege_mode: false` means dataset writes are falling back to the
inference token. This remains backward compatible, but is not recommended for
production.

### Routing

| Variable | Required? | Default | Description |
|---|---|---|---|
| `BACKEND_URL` | No | `""` | **Path 1** — Explicit upstream URL.  All requests forwarded here when set.  Use for Docker Model Runner, Ollama, or any custom backend. |
| `HF_SPACES_MODEL_URL` | No | — | **Path 2** — Custom ZeroGPU Space URL (e.g. `https://scikit-plots-ai-model.hf.space/v1/chat/completions`). Receives requests whose `model` matches `HF_SPACES_MODEL_NAMESPACES`. |
| `HF_SPACES_MODEL_NAMESPACES` | No | `scikit-plots` | Comma-separated model owner prefixes routed to Path 2 (e.g. `scikit-plots,my-org`). |
| `ALLOWED_MODELS` | No | `DEFAULT_MODEL` + bundled provider models | Exact model IDs accepted by the strict chat contract before routing. The bundled defaults include the configured `DEFAULT_MODEL` plus `Qwen/Qwen2.5-Coder-7B-Instruct`, `Qwen/Qwen2.5-Coder-32B-Instruct`, and `openai/gpt-oss-20b`; Path-2 namespaces remain an additional explicit admission rule. |
| `HF_BASE` | No | `https://router.huggingface.co` | **Path 3** — HF Serverless API base URL.  Only used when Path 1 and Path 2 do not match. |
| `DEFAULT_MODEL` | No | `scikit-plots/Qwen2.5-Coder-7B-Instruct` | Fallback model when the request body omits `"model"`. |

### Reasoning capabilities

Reasoning controls are **opt-in**. Leave `REASONING_ENABLED=false` (the default)
unless the selected upstream is known to accept the advertised fields. A malformed
mode or budget bound falls back safely and logs only a fixed diagnostic code /
option name — never the supplied value, endpoint, model id, request body, or user
content.

| Variable | Required? | Default | Description |
|---|---|---|---|
| `REASONING_ENABLED` | No | `false` | Advertise optional Effort / Thinking capability through `/health`. When false, the panel leaves provider defaults untouched. |
| `REASONING_EFFORT_PARAM` | No | `reasoning_effort` | Top-level request field used for the Effort value. Set empty to advertise no Effort support. |
| `REASONING_THINKING_PARAM` | No | `""` | Top-level request field used for Thinking. Empty means Thinking is unsupported even when Effort is enabled. |
| `REASONING_THINKING_MODE` | No | `budget` | Payload adapter for the Thinking field: `boolean`, `adaptive`, or `budget`. Configure only a shape the upstream is verified to accept. |
| `REASONING_BUDGET_MIN` | No | `500` | Lower token-budget bound advertised for `budget` mode; clamped to the safe 500–16000 range. |
| `REASONING_BUDGET_MAX` | No | `16000` | Upper token-budget bound advertised for `budget` mode; clamped to the safe 500–16000 range and never below the minimum. |

`boolean` sends the configured Thinking field as `true`; `adaptive` sends
`{"type": "adaptive"}`; `budget` sends
`{"type": "enabled", "budget_tokens": N}`. These are adapter shapes, not
claims that every provider or every model supports them. If an optional reasoning
shape is rejected before output begins, the browser retries once without optional
Effort / Thinking fields and opens a per-model in-memory fallback circuit.

### Record / training data collection

| Variable | Put in | Required? | Description |
|---|---|---:|---|
| `RECORD_STORAGE_TARGETS` | Variable | New multi-store mode | Provider-neutral JSON topology with exactly one Primary and optional Mirrors. Contains token environment-variable **names**, never token values. |
| `TRAINING_DATASET_REPO` | Variable | Legacy HF mode only | Backward-compatible HF Dataset repo ID. When `RECORD_STORAGE_TARGETS` is present, the explicit target topology is authoritative; this value may remain for rollback/older discovery consumers. |
| `FEEDBACK_PERSIST_ENABLED` | Variable | No | Server-side persistence permission for privacy-minimal `/v1/feedback` telemetry. Default is `false`; it cannot override the browser user-consent gate, and feedback is never training-eligible. |
| `CONTRIBUTION_REVIEW_MODE` | Variable | `ledger` | `ledger` keeps the historical local/DB quarantine. `provider-pr` creates a native provider review ref immediately after consent; only merge to the canonical branch makes it eligible. |
| `CONTRIBUTION_REVIEW_TOKEN` | Secret | Optional | Operator-only token for API-driven merge/promotion. In `provider-pr` mode maintainers may instead merge/close directly in the provider UI. |
| `CONTRIBUTION_LEDGER_BACKEND` | Variable | No | `memory` (default), `sqlite` (local transactional/restart-durable), or `redis` (shared transactional authority across replicas in one Redis consistency domain). Redis persistence durability is deployment-conditional and is not inferred by the proxy. |
| `CONTRIBUTION_LEDGER_SQLITE_PATH` | Variable | When using SQLite | Local SQLite database path. Keep it on deployment-owned persistent storage if restart durability is required. |
| `CONTRIBUTION_REQUIRE_DURABLE` | Variable | No | Default `false`. When `true`, contribution intake fails closed unless the configured receipt ledger reports durable storage. Bundled Redis shared mode deliberately does not self-certify external persistence durability. |
| `CONTRIBUTION_REQUIRE_SHARED` | Variable | No | Default `false`. When `true`, contribution intake fails closed unless the active ledger is a ready shared authoritative backend. Use with `CONTRIBUTION_LEDGER_BACKEND=redis` for horizontal replicas. |
| `CONTRIBUTION_LEDGER_REDIS_URL` | Secret | Redis only | Shared receipt-control Redis URL. Keep separate from public configuration; `rediss://` is mandatory when `REDIS_REQUIRE_TLS=true` or `DEPLOYMENT_PROFILE=strict`; certificate and hostname verification are forced by code. |
| `CONTRIBUTION_LEDGER_KEY_SECRET` | Secret | Redis only | Dedicated >=32-byte HMAC secret used to pseudonymize receipt IDs before they become Redis key material. Do not reuse provider or review tokens. |
| `CONTRIBUTION_OPERATION_LEASE_SECONDS` | Variable | No | Shared lifecycle claim lease, bounded to 30–900 seconds (default 120). Withdrawal claims may be retried toward the monotonic withdrawn state. An expired promotion claim becomes reconciliation-required rather than being automatically reassigned, because external Git/provider writes cannot be fenced by Redis alone. |
| `CONTRIBUTION_LEDGER_MAX_RECEIPTS` | Variable | No | Bounded receipt cardinality for the active ledger; protects memory, SQLite, and Redis control-plane state from unbounded intake. |
| `CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS` | Variable | No | Default `86400`. Retains deleted/expired/withdrawn lifecycle tombstones for bounded status/history before reclaiming capacity; eligible receipts are not auto-retired. |

Actual provider credentials belong in Space **Secrets** referenced by each target's `token_env`. For example, `AI_RECORD_STORAGE_TOKEN_HF_PRIMARY` and `AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR`.

> **Recommended:** configure storage using `RECORD_STORAGE_TARGETS`; keep the legacy variables only when you need a simple one-HF-repo deployment or a rollback path. Full recipes and security notes are in [DATASET_COLLECTION_GUIDANCE.md](./DATASET_COLLECTION_GUIDANCE.md).

### Timeouts

All values are in seconds.  Non-integer values silently fall back to the default.

| Variable | Default | Applies to | Description |
|---|---|---|---|
| `PROXY_TIMEOUT` | `600` | Path 1 | Read timeout for custom `BACKEND_URL`. Covers local model cold starts. |
| `PATH2_TIMEOUT` | `600` | Path 2 | Read timeout for custom ai-model Space.  CPU 7B inference takes 4–5 min. |
| `PATH3_TIMEOUT` | `120` | Path 3 | Read timeout for HF Serverless API.  GPU inference resolves in 30–90 s. |
| `PROXY_CONNECT_TIMEOUT` | `10` | All | TCP handshake timeout. |
| `PROXY_WRITE_TIMEOUT` | `30` | All | Request body upload timeout. |
| `PROXY_POOL_TIMEOUT` | `10` | All | HTTP connection-pool acquire timeout. |
| `PROXY_PROTOCOL_RETRIES` | `1` | Streaming pre-output only | Retries remote protocol/read failures before any browser-visible output. Bounded to `0..2`. Local protocol errors are never retried. |

### Other

| Variable | Default | Description |
|---|---|---|
| `ALLOWED_ORIGINS` | empty | Comma-separated exact browser origins. In `additive` mode they are appended to the two bundled defaults (`https://scikit-plots.github.io`, `https://scikit-plots-learn.readthedocs.io`). In `replace` mode they are the complete browser allow-list. Missing `Origin` remains valid for server-to-server clients. `*` is an explicit insecure compatibility escape hatch and is not authentication. |
| `ALLOWED_ORIGINS_MODE` | `additive` | `additive` retains the bundled project origins; `replace` starts from an empty allow-list and trusts only valid entries from `ALLOWED_ORIGINS`. Invalid values fail safely back to `additive`. |
| `DEPLOYMENT_PROFILE` | `compat` | `compat` preserves explicit legacy deployment choices. The hardened Dockerfile sets `strict`, which fails startup as root, rejects wildcard origins and opaque-origin **writes**, and requires verified TLS for all configured Redis authorities. Read-only opaque Share compatibility remains a separate explicit opt-in because local `file://` viewers may need it. |
| `REDIS_REQUIRE_TLS` | `false` (`true` in strict) | Requires `rediss://`; URL query parameters are rejected so callers cannot disable certificate verification through redis-py URL options. |
| `REQUIRE_NON_ROOT` | `false` (`true` in strict) | Fails application startup when the process is UID 0 on POSIX. |
| `SHARE_ALLOW_OPAQUE_ORIGIN` | `false` | **Read-only** Share compatibility for browser `Origin: null`: viewer/read and legacy GET/HEAD only. This is not authentication because sandboxed hostile documents can also serialize to `null`. |
| `SHARE_ALLOW_OPAQUE_ORIGIN_WRITE` | `false` | Additional high-risk opt-in for Share create/update/revoke/status-capability mutations from `Origin: null`. It has no effect unless read compatibility is also enabled, and `DEPLOYMENT_PROFILE=strict` refuses it. |
| `SHARE_PUBLIC_BASE_URL` | empty | Optional explicit public Share base. Production remote values must be HTTPS. On HF Spaces, `SPACE_HOST` is used automatically when this is unset, and an accidental `http://<same SPACE_HOST>` value is upgraded safely. |
| `SHARE_STORE_BACKEND` | `memory` | Global Share lifecycle authority: `memory` (process-local compatibility), `sqlite` (restart-durable local authority), or `redis` (shared atomic authority across replicas). |
| `SHARE_STORE_SQLITE_PATH` | `/tmp/scikitplot-ai-global-share.sqlite3` | SQLite file used only with `SHARE_STORE_BACKEND=sqlite`. Put it on deployment-owned persistent storage when restart durability is required. |
| `SHARE_STORE_REDIS_URL` | empty | Redis connection URL used only with `SHARE_STORE_BACKEND=redis`. Treat the URL as a secret. `DEPLOYMENT_PROFILE=strict` requires `rediss://` with certificate and hostname verification. |
| `SHARE_STORE_KEY_PREFIX` | `sphinx-ai-assistant` | Non-secret Redis namespace prefix. Public Share locators are SHA-256 pseudonymized before becoming Redis/SQLite keys. |
| `SHARE_STORE_REDIS_TIMEOUT_SECONDS` | `2` | Redis Share-store timeout, clamped to `0.25..10` seconds. |
| `SHARE_REDIS_DURABILITY_CONFIRMED` | `false` | Operator assertion that the selected Redis service has the persistence/durability contract required for Global Share lifecycle. Shared coordination alone does not make this true. |
| `SHARE_REQUIRE_DURABLE` | `false` | When true, Global Share writes fail closed unless the active store reports durable lifecycle storage. |
| `SHARE_REQUIRE_SHARED` | `false` | When true, Global Share writes fail closed unless the active store is shared and authoritative. Use for horizontally scaled replicas. |
| `MAX_BODY_BYTES` | `10485760` | Maximum accepted chat request body size. Enforced while streaming and hard-clamped to 16 MiB. |
| `MAX_UPSTREAM_RESPONSE_BYTES` | `8388608` | Maximum decoded upstream response body accepted by the proxy. Enforced while streaming before whole-body buffering and hard-clamped to 32 MiB. Oversize or malformed declared lengths fail closed. |
| `CHAT_RATE_LIMIT_PER_HOUR` | `30` | Chat requests per resolved client identity. Enforced by the selected local or Redis backend. |
| `SHARE_RATE_LIMIT_PER_HOUR` | `10` | Global Share creates/updates per resolved client identity. Enforced by the selected backend. |
| `FEEDBACK_RATE_LIMIT_PER_HOUR` | `30` | Feedback writes, including retractions, per resolved client identity. |
| `CONTRIBUTION_RATE_LIMIT_PER_HOUR` | `5` | Contribution writes per resolved client identity. |
| `RATE_LIMIT_BACKEND` | `local` | `local` keeps the bounded per-process abuse gate; `redis` uses one shared atomic fixed-window consistency domain across replicas. |
| `RATE_LIMIT_REDIS_URL` | empty | Redis connection URL used only with `RATE_LIMIT_BACKEND=redis`. Treat it as a secret. `DEPLOYMENT_PROFILE=strict` requires `rediss://` with certificate and hostname verification. |
| `RATE_LIMIT_IDENTITY_SECRET` | empty | Server-only HMAC-SHA256 key, at least 32 UTF-8 bytes. Required by Redis mode so raw client identities are not externalized in rate-limit keys. |
| `RATE_LIMIT_KEY_PREFIX` | `sphinx-ai-assistant` | Non-secret namespace prefix for shared Redis rate-limit keys. |
| `RATE_LIMIT_REDIS_TIMEOUT_SECONDS` | `2` | Redis operation timeout, clamped to `0.25..10` seconds. |
| `RATE_LIMIT_REQUIRE_SHARED` | `false` | When `true`, local limiting is rejected and unavailable/misconfigured Redis fails closed with HTTP 503 rather than silently splitting quota across replicas. |
| `RATE_LIMIT_MAX_IDENTITIES` | `10000` | Hard bound for each local in-memory identity table only (clamped to `128..50000`). New identities fail closed when the table is full. |
| `TRUST_X_FORWARDED_FOR` | `false` | Use leftmost `X-Forwarded-For` only behind a known ingress that overwrites it. Default identity is the direct peer address. |
| `STUB_ENABLED` | `false` | Enable the deterministic `stub/*` diagnostic responder. Put this non-sensitive boolean in a Space **Variable**. Disabled stub requests fail closed locally with HTTP 503. |

For a horizontally scaled HF deployment that requires one quota across replicas, set `RATE_LIMIT_BACKEND=redis`, provide `RATE_LIMIT_REDIS_URL` and a dedicated `RATE_LIMIT_IDENTITY_SECRET`, and set `RATE_LIMIT_REQUIRE_SHARED=true`. The Redis backend uses one atomic Lua fixed-window operation; if that configured shared backend is unavailable, requests fail closed instead of falling back to independent process-local counters. This guarantee is scoped to a single Redis consistency domain; it is an abuse-control quota, not billing-grade accounting across unrelated/Active-Active stores.

Do not reuse API/provider tokens as `RATE_LIMIT_IDENTITY_SECRET`. The HMAC key exists only to pseudonymize rate-limit identities before shared storage. Health/discovery responses report the selected backend and readiness, never the Redis URL or HMAC secret.

---

## Token setup guide

### Create tokens on HuggingFace

Go to **https://huggingface.co/settings/tokens** → **New token** → choose
**Fine-grained** (recommended over classic tokens).

#### Read token → `HF_TOKEN`

```
Type:        Fine-grained
Name:        sphinx-ai-proxy-read   (any descriptive name)
Permissions: ✅ Make calls to the serverless Inference API
             ✅ Read access to contents of all repos under your namespace
             ❌ No write permissions of any kind
```

#### Dataset token → `HF_DATASET_TOKEN`

```
Type:        Fine-grained
Name:        sphinx-ai-proxy-dataset-write  (any descriptive name)
Permissions: ✅ Write access — scoped to ONE repo only:
                scikit-plots/ai-assistant-contributions
             ❌ No Inference API access
             ❌ No write access to any other repo
```

Scoping the write token to a single repo means a leaked token can only append
JSONL files to your training dataset — it cannot modify model weights, code, or
any other repository.

#### Classic tokens (legacy, if fine-grained are unavailable)

```
read  role  →  HF_TOKEN         (inference only; set in Space secrets)
write role  →  HF_DATASET_TOKEN (dataset persistence; broader than fine-grained)
```

### Declare token types in Space secrets

After creating the tokens, set the corresponding type variables so the proxy
can validate least-privilege at startup without network calls:

```
# Fine-grained tokens (recommended):
HF_TOKEN_TYPE         = fine-grained
HF_DATASET_TOKEN_TYPE = fine-grained

# Classic read/write tokens (legacy):
HF_TOKEN_TYPE         = read
HF_DATASET_TOKEN_TYPE = write
```

If you omit these variables the proxy applies a length-based heuristic
(tokens ≥ 52 chars → `fine-grained`; shorter → `unknown`).  The startup log
and status page will show `"unknown"` for the type, which suppresses
least-privilege warnings — always set explicit types in production.

---

## Record storage targets (primary + mirrors)

The proxy can persist the same canonical feedback/contribution record to more
than one repository provider.  The browser never receives write credentials;
all tokens remain server-side.

`RECORD_STORAGE_TARGETS` is a JSON array with exactly one `primary` and zero or
more `mirror` targets (maximum 8).  Supported providers are `huggingface`,
`github`, `gitlab`, and `bitbucket`.

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
    "token_type": "fine-grained"
  },
  {
    "id": "github-mirror",
    "label": "GitHub Mirror",
    "provider": "github",
    "role": "mirror",
    "repo": "scikit-plots/ai-assistant-records",
    "branch": "main",
    "token_env": "AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR"
  }
]
```

Set the referenced tokens as independent secrets, for example:

```text
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY=hf_...
AI_RECORD_STORAGE_TOKEN_HF_PRIMARY_TYPE=fine-grained
AI_RECORD_STORAGE_TOKEN_GITHUB_MIRROR=<github-repo-token>
```

Only environment names beginning with `AI_RECORD_STORAGE_TOKEN_` are accepted
in target configuration.  This prevents a storage target from accidentally
referencing an unrelated process secret.

Use **private repositories** for feedback/contribution records unless you have an
explicit public-data policy. Set `"expose_links": false` on a target when its
repository identity should not be published in `GET /`; the manifest will still
report provider/role/health but omits the repo identifier and browser links.

### Hugging Face token capability

Hugging Face has three User Access Token roles:

| Type | Dataset persistence | Guidance |
|---|---|---|
| `fine-grained` | **Allowed when the token has write access to the target dataset repo** | Recommended for production; scope it only to the dataset that receives records. |
| `read` | **Blocked** | Cannot create/push dataset files. |
| `write` | **Allowed** | Works, but grants broader repository write permission than a repo-scoped fine-grained token. |

The proxy therefore does **not** require the token label `write`.  A
`fine-grained` token is preferred.  On versions of `huggingface_hub` exposing
`auth_check(..., write=True)`, startup verifies repo-specific write capability
without mutating the repo.  On older compatible versions the capability is
reported as `unverified` and the first successful commit promotes it to
`verified`.  A declared `read` token is rejected before any commit attempt.

### Persistence semantics

One canonical JSONL payload and record ID are created per accepted write.  The
primary target is written first.  Only primary success defines acceptance;
mirrors are replication and may degrade independently.  Records use the same
logical path on every provider:

```text
feedback/YYYY/MM/DD/fb_<content-hash>.jsonl
contributions/YYYY/MM/DD/ct_<content-hash>.jsonl
```

This makes retries idempotent and gives later reconciliation a stable key.
Provider writes are serialized per target, bounded to two attempts, and protected
by a small circuit breaker after repeated failures.  Server-owned tasks are
shielded from browser disconnects/broken pipes once persistence has started.

Public discovery (`GET /`) exposes only sanitized metadata under `storage`:
provider, role, repo, branch, record paths, status, public links, and token
**type/capability**.  It never returns token values or Authorization headers.

### Backward compatibility

If `RECORD_STORAGE_TARGETS` is absent, the existing configuration is synthesized
as one Hugging Face primary target:

```text
TRAINING_DATASET_REPO
HF_DATASET_TOKEN (preferred) or HF_WRITE_TOKEN (legacy alias) or HF_TOKEN (fallback)
HF_DATASET_TOKEN_TYPE / HF_WRITE_TOKEN_TYPE / HF_TOKEN_TYPE
```

No migration is required for existing deployments.

Provider credential minimums for mirror targets:

| Provider | Minimum write capability |
|---|---|
| Hugging Face | Fine-grained token with write access to the specific dataset repo (preferred), or classic `write` |
| GitHub | Fine-grained token / GitHub App token with repository **Contents: write**; `provider-pr` additionally needs **Pull requests: write** |
| GitLab | Token with `api` scope and write access to the target project (branch + merge-request creation/merge) |
| Bitbucket Cloud | Repository/project/workspace access token with repository write permission plus pull-request write permission (`repository:write` + `pullrequest:write`, or corresponding API-token scopes) |


## Dataset / record-storage setup

The proxy supports a legacy single-Hugging-Face dataset and provider-neutral storage with Hugging Face, GitHub, GitLab, and Bitbucket targets. New multi-store deployments should use `RECORD_STORAGE_TARGETS`; the browser still sends one `/v1/feedback` or `/v1/contribute` request and `app.py` performs Primary → Mirror persistence server-side.

For a complete setup and operations walkthrough, use:

**[DATASET_COLLECTION_GUIDANCE.md](./DATASET_COLLECTION_GUIDANCE.md)**

It includes:

- Space **Secret vs Variable** classification;
- HF Fine-grained / Read / Write capability rules;
- HF-only, GitHub-only, HF→GitHub and GitHub→HF examples;
- exact live-write and mirror-failure tests;
- folder/path tuning and sanitized diagnostics;
- `deduplicate_dataset.py` legacy and provider-neutral commands;
- mirror-aware audit/recovery and conflict handling;
- migration/rollback checklists and official provider references.

Minimal legacy HF example:

```text
# Variable
TRAINING_DATASET_REPO=scikit-plots/ai-assistant-contributions
HF_DATASET_TOKEN_TYPE=fine-grained

# Secret
HF_DATASET_TOKEN=hf_<repo-scoped-token>
```

Recommended HF Primary + GitHub Mirror example uses two provider Secrets plus one `RECORD_STORAGE_TARGETS` Variable; see Recipe D in the guide rather than duplicating the topology here.

## Quick deployment recipes

### Path 3 — HF Serverless API (simplest, standard provider models)

Use this for models registered with a HuggingFace Inference Provider
(Qwen/\*, mistralai/\*, etc.).

```
# Secret:
HF_TOKEN      = hf_<your-read-token>
# Variables:
DEFAULT_MODEL = Qwen/Qwen2.5-Coder-32B-Instruct
PATH3_TIMEOUT = 120
```

### Path 2 — Custom ZeroGPU Space (mirror repos, free GPU)

Use this for `scikit-plots/*` mirror repos that are not registered with any
Inference Provider.  See [Why mirror repos fail](#why-mirror-repos-fail)
below.

```
# Variables:
HF_SPACES_MODEL_URL        = https://scikit-plots-ai-model.hf.space/v1/chat/completions
HF_SPACES_MODEL_NAMESPACES = scikit-plots
PATH2_TIMEOUT              = 600   ← CPU cold start takes 4–5 minutes
# Path 2 preserves scikitplot-chat-v1 to the model Space; the model service
# independently constructs the same server-owned system policy.
DEFAULT_MODEL              = scikit-plots/Qwen2.5-Coder-7B-Instruct
```

### Path 1 — Local / custom backend (Docker Model Runner, Ollama)

Set in your shell or CI environment — do **not** put a localhost URL in Space
secrets (the Space container cannot reach your local machine).

```bash
export BACKEND_URL=http://localhost:12434/engines/llama.cpp/v1/chat/completions
# Optional dedicated credential for that exact backend only:
export BACKEND_AUTH_TOKEN=<backend-specific-token>
```

### Full production setup (all features enabled)

```
# Inference
HF_TOKEN                   = hf_<read-token>
HF_TOKEN_TYPE              = fine-grained
HF_SPACES_MODEL_URL        = https://scikit-plots-ai-model.hf.space/v1/chat/completions
HF_SPACES_MODEL_NAMESPACES = scikit-plots
DEFAULT_MODEL              = scikit-plots/Qwen2.5-Coder-7B-Instruct
PATH2_TIMEOUT              = 600
PATH3_TIMEOUT              = 120

# Record storage — legacy single-HF example
# Put repo/type in Variables; put the token in Secrets.
TRAINING_DATASET_REPO      = scikit-plots/ai-assistant-contributions
HF_DATASET_TOKEN_TYPE      = fine-grained
HF_DATASET_TOKEN           = hf_<fine-grained-dataset-token>
# For Primary + Mirrors, use RECORD_STORAGE_TARGETS instead; see DATASET_COLLECTION_GUIDANCE.md.

# Security
# Both current Scikit-plots origins are built in:
#   https://scikit-plots.github.io
#   https://scikit-plots-learn.readthedocs.io
# Leave empty to use only those defaults, or add exact custom origins.
ALLOWED_ORIGINS            =
ALLOWED_ORIGINS_MODE       = additive
# Fork/downstream alternative:
# ALLOWED_ORIGINS=https://docs.example.org,https://learn.example.org
# ALLOWED_ORIGINS_MODE=replace
# Optional read-only local-file (Origin:null) Share compatibility.
# Keep false unless you explicitly need Global Share viewing from file:// pages.
SHARE_ALLOW_OPAQUE_ORIGIN       = false
# Separate high-risk mutation authority; strict deployments refuse this.
SHARE_ALLOW_OPAQUE_ORIGIN_WRITE = false
```

---

## Verify the deployment

```bash
BASE=https://scikit-plots-ai.hf.space

# 1. Liveness probe
curl $BASE/health
# {"status":"ok","version":"7.3.0"}

# Optional deterministic stub rig status
curl -s $BASE/health | python3 -m json.tool
# capabilities.stub.enabled is true only when STUB_ENABLED=true.

# 2. Full status — check routing, token slots, and record-storage readiness
curl $BASE/ | python3 -m json.tool
# Look for:
#   "storage":  { "policy": "primary_then_mirrors", "targets": [...] }
#   "training": { "contribute_ready": true }

# 3. Test a chat completion (Path 3 — HF Serverless)
curl $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-Coder-32B-Instruct","messages":[{"role":"user","content":"hi"}]}'

# 4. Test a chat completion (Path 2 — custom Space)
curl $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"scikit-plots/Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"hi"}]}'

# 4b. Test the same Path-2 request with stream:true.  The proxy now inspects
# the upstream Content-Type: a JSON-only model Space stays JSON; a true SSE
# backend stays SSE.  Both are valid and the browser handles both.
curl -i $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"scikit-plots/Qwen2.5-Coder-7B-Instruct","stream":true,"messages":[{"role":"user","content":"hi"}]}'

# 4c. Deterministic wire test (requires Space Variable STUB_ENABLED=true).
curl -N $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"stub/qa","stream":true,"messages":[{"role":"user","content":"ping"}]}'
# Must contain an SSE delta with "pong" and finish with: data: [DONE]

# 5. Test contribution persistence (requires a write-ready Primary target)
curl $BASE/v1/contribute \
  -H "Content-Type: application/json" \
  -d '{
    "schemaVersion": 4,
    "consentFlag": true,
    "consentVersion": "2.0.0",
    "page": "https://example.com",
    "model": {"id":"test-model","provider":"test"},
    "records": [{
      "recordType": "qa",
      "answerIndex": 0,
      "query": "test query",
      "answer": "test answer",
      "ratingValue": 2,
      "ratingLabel": "helpful",
      "message": "",
      "ts": 1781002584724
    }]
  }'
# {"contributed": true, "status": "quarantined", "rows": 1, "receiptId": "...", "deleteToken": "..."}
```

---

## Chat streaming contract (v6.4+)

The browser may request ``"stream": true``, but that is an intent, not proof that
the selected upstream actually speaks Server-Sent Events.  The proxy therefore
opens the upstream response before committing its own response and chooses the
downstream mode from the upstream status and ``Content-Type``.

```text
browser stream:true
        |
        v
proxy opens upstream first
        |
        +-- non-2xx / connect / local protocol failure -> real HTTP 4xx/5xx
        |
        +-- application/json ---------------------------> JSON unchanged
        |
        `-- text/event-stream --------------------------> SSE passthrough
                                                            |
                                                            `-- mid-stream error
                                                                -> event: error
```

This is especially important for the custom ``scikit-plots/ai-model`` Space:
that backend currently produces a complete OpenAI-compatible JSON response
rather than incremental SSE.  Older proxy builds labelled those JSON bytes as
``text/event-stream`` when the panel requested streaming; an SSE parser then had
no valid ``data:`` frames to render.  v6.4 preserves JSON as JSON, and the panel's
existing streaming fallback parser displays the completion normally.

``stub/*`` is also a reserved namespace.  When ``STUB_ENABLED=false`` the proxy
returns local HTTP 503 ``stub_disabled`` and never sends the diagnostic model id
to Path 1/2/3.  Use a Space **Variable**, not a Secret, for this non-sensitive
boolean.

Remote protocol/read failures are retried only before output is visible and only
up to ``PROXY_PROTOCOL_RETRIES`` (default 1, max 2).  A
``LocalProtocolError`` is not retried because HTTPX defines it as a protocol
violation by the local client/request; repeating the same invalid request is not
a recovery strategy.

For ``LocalProtocolError`` the proxy never logs the raw exception message because
HTTP libraries may include a rejected header value in it.  Instead it emits only
a fixed reason label such as ``illegal-header``, ``content-length-overrun``,
``content-length-underrun``, ``missing-host``, ``request-line``, or
``unspecified``.

---

## Why mirror repos fail — and how Path 2 solves it

`scikit-plots/gpt-oss-20b` and `scikit-plots/Qwen2.5-Coder-7B-Instruct` are
**mirror repositories** — weights copied from the originals but **not
registered with any HF Inference Provider**.

| Request | Result |
|---|---|
| `model: "scikit-plots/..."` → `router.huggingface.co` (no Path-2) | ❌ 404 or 503 |
| `model: "Qwen/Qwen2.5-Coder-32B-Instruct"` → `router.huggingface.co` | ✅ works |
| `model: "scikit-plots/..."` → this proxy with `HF_SPACES_MODEL_NAMESPACES=scikit-plots` | ✅ intercepted by Path 2, forwarded to ZeroGPU Space |

Path 2 intercepts requests whose `model` field starts with a configured
namespace **before** they reach the HF Serverless router, and forwards them to
the custom ZeroGPU Space that actually has the weights loaded.  This is why
`DEFAULT_MODEL = scikit-plots/Qwen2.5-Coder-7B-Instruct` works when
`HF_SPACES_MODEL_NAMESPACES = scikit-plots` is set.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `POST /v1/contribute` → 503 / storage not ready | No write-ready Primary target | If using `RECORD_STORAGE_TARGETS`, inspect `GET /` → `storage.targets` and the Primary token capability. In legacy HF mode, set `TRAINING_DATASET_REPO` as a Variable plus a write-capable dataset-token Secret. |
| Primary target shows `missing-token` | Target `token_env` does not resolve to a Secret | Create the exact Space Secret named by `token_env`; do not place the token value inside `RECORD_STORAGE_TARGETS`. |
| HF target shows `denied` / `denied-read-token` | Token cannot write the selected Dataset repo | Prefer a repo-scoped Fine-grained token with write permission; classic Write also works but is broader. A Read token is intentionally blocked. |
| `POST /v1/contribute` → 422 `"consentVersion … is not current"` | Browser widget cached an old page with an outdated consent version string | Hard-refresh the docs page (`Ctrl+Shift+R`) |
| `POST /v1/contribute` → 429 | Rate limit exceeded (5 contributions per IP per hour) | Wait 1 hour or test from a different IP |
| `POST /v1/chat/completions` logs `LocalProtocolError` and the panel is empty | Proxy-side HTTP request violated protocol semantics, or an old build forwarded a disabled `stub/*` model upstream | Deploy proxy v6.4+; `stub/*` now fails closed locally when disabled, blank Bearer headers are never constructed, and pre-header protocol failures return a real 502 instead of an empty 200. |
| Path-2 request returns HTTP 200 but the assistant bubble is empty | Backend returned ordinary JSON while the old proxy advertised it downstream as `text/event-stream` | Deploy proxy v6.4+. The proxy now negotiates by actual upstream `Content-Type`: JSON remains JSON and true SSE remains SSE. |
| Stream starts and then closes with no useful message | Mid-stream transport/protocol failure | Proxy v6.4 emits an explicit `event: error` SSE frame. Check the server-side fixed diagnostic code (`UPSTREAM_PROTOCOL_REMOTE`, `UPSTREAM_TIMEOUT`, etc.); provider bodies and request content are not logged. |
| `Stub · canned answers` does not return `pong` | Proxy stub rig is disabled | Add Space **Variable** `STUB_ENABLED=true`, restart, then confirm `GET /health` reports `capabilities.stub.enabled: true`. |
| `POST /v1/chat/completions` → 400 with `PROXY_MODEL_NOT_ALLOWED` | The selected public model is not in the proxy allow-list, so the request is rejected locally before any provider call | Set `ALLOWED_MODELS` to the exact public model IDs used by the docs, or deploy the current defaults. Do not broaden to `*`. |
| `POST /v1/chat/completions` → 401 | `HF_TOKEN` not set or expired | Regenerate token at `huggingface.co/settings/tokens` |
| `POST /v1/chat/completions` → 503 / 404 for `scikit-plots/*` | Path-2 not configured | Set `HF_SPACES_MODEL_URL` and `HF_SPACES_MODEL_NAMESPACES=scikit-plots` |
| Status page shows `contribute_ready: false` | Primary storage is not write-ready | Inspect `storage.targets` first. `RECORD_STORAGE_TARGETS`, when non-empty, is authoritative over legacy `TRAINING_DATASET_REPO`. |
| Legacy status shows `least_privilege_mode: false` | Legacy HF mode is falling back to the inference token | Set a dedicated dataset-token Secret. In provider-neutral mode, use each target's `token_env` Secret instead. |
| Container crashes: `ModuleNotFoundError` for `_utils` or a helper module | `_utils/` is missing/incomplete in the Space repo or image | Commit the complete `_utils/` directory alongside `app.py`; Dockerfile v3.2 copies it as one package |
| 413 on large chat request | Body exceeds `MAX_BODY_BYTES` (default 10 MiB; hard ceiling 16 MiB) | Reduce the request. The application ceiling is intentionally bounded rather than arbitrarily raised. |
| Startup log: `WARNING: Startup token-config check: HF_TOKEN has type 'write' …` | A write token is being used for inference — violates least-privilege | Replace `HF_TOKEN` with a read or fine-grained token scoped to Inference only; set `HF_TOKEN_TYPE=read` or `HF_TOKEN_TYPE=fine-grained` as a Space Variable |
| Startup diagnostic reports dataset token type `read` | A read-only token is configured for persistence | Replace it with a fine-grained token scoped to write the dataset repo (preferred), or a classic Write token; set `HF_DATASET_TOKEN_TYPE=fine-grained` or `write` as a Space Variable |
| Status page shows `hf_token_type: "unknown"` | `HF_TOKEN_TYPE` not set; length-based heuristic could not determine token type | Set `HF_TOKEN_TYPE=fine-grained` or `HF_TOKEN_TYPE=read` as a Space Variable to enable least-privilege startup checks |
| Status page shows `hf_token_type: "write"` with a startup WARNING | `HF_TOKEN` is a write token — too many permissions for inference | Create a separate read or fine-grained token for `HF_TOKEN` and set `HF_TOKEN_TYPE=read`; keep the dataset token only in `HF_DATASET_TOKEN` |

---

## References

- [DATASET_COLLECTION_GUIDANCE.md](./DATASET_COLLECTION_GUIDANCE.md) — End-to-end single/multi-store setup, testing, deduplication, migration, and provider references
- [HuggingFace fine-grained tokens](https://huggingface.co/docs/hub/security-tokens)
- [HTTPX exception hierarchy](https://www.python-httpx.org/exceptions/) — `LocalProtocolError` vs `RemoteProtocolError` semantics used by the stream bridge
- [Hugging Face streaming](https://huggingface.co/docs/text-generation-inference/conceptual/streaming) — OpenAI-compatible `stream=True` / SSE behavior
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference/)
- [HuggingFace Datasets — `huggingface_hub`](https://huggingface.co/docs/huggingface_hub/guides/repository)
- [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu)


### Provider control-response boundary (Run 25 / B44)

Record-storage provider control responses are bounded independently from intentional model/dataset downloads. The default ceiling is **4 MiB** and `AI_RECORD_STORAGE_CONTROL_RESPONSE_MAX_BYTES` is clamped to **16 MiB**. Provider mutations that do not require response content are streamed and closed without buffering a body; metadata JSON is byte-counted before parsing. Custom `api_base` is supported only for GitLab and must be HTTPS with a valid host and no userinfo, query, fragment, control characters, or traversal.

### Native provider review quarantine (`CONTRIBUTION_REVIEW_MODE=provider-pr`)

This mode deliberately separates **durable review presence** from **training eligibility**. The submitted record is written to its final canonical path on a provider-native review ref, never directly to the canonical branch. Review refs use an opaque receipt-derived key such as `ai-contrib-<24 hex>`; user text, page titles, e-mail addresses, and other contribution content are never placed in branch names or review titles.

| Primary provider | Quarantine/review surface | Accept | Reject |
|---|---|---|---|
| Hugging Face Dataset | Hub Pull Request (`refs/pr/*`) | Merge PR | Close PR |
| GitHub | Temporary branch + Pull Request | Merge PR | Close PR |
| GitLab | Temporary branch + Merge Request | Merge MR | Close MR |
| Bitbucket Cloud | Temporary branch + Pull Request | Merge PR | Decline PR |

The configured target `branch` (normally `main`) is the only eligible branch. Participant deletion while review is open closes the provider review first and then clears the active local receipt copy. If a maintainer merged through the provider UI before the participant checks status or withdraws, the proxy detects the merged review and monotonically ratchets the receipt to `eligible`; withdrawal then follows the existing post-promotion removal/tombstone workflow.

Recommended HF Space variable:

```text
CONTRIBUTION_REVIEW_MODE=provider-pr
```

Keep `CONTRIBUTION_REVIEW_TOKEN` only if you also want the authenticated `/promote` endpoint as an automation/fallback merge mechanism. Native provider UI review does not require exposing that token to browsers.


> **Mirror boundary:** native review/merge authority belongs only to the Primary. A maintainer merge updates that canonical branch; it is not synchronously replicated to Mirrors by the review path. Use explicit reconciliation when immediate mirror convergence is required.
