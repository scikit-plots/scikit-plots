# B18 — Privacy, Secrets, Identity & Abuse Threat Model

Status: **IN PROGRESS — Runs 1–3 LANDED; prompt/logging/contribution/privacy runs remain**
Source baseline: **B16 overlay SHA-256 37228bb8fece0e181494a0a27f44d7689dcba03f60371293bf82b8e53c5e928c**

## Landed implementation increments

### Run 1 — client/build-time secret lifecycle

- endpoint-profile storage schema bumped to v3;
- v1/v2/raw legacy profile storage is re-sanitized and rewritten;
- `shareToken` / `feedbackToken` are absent from persistent profile payloads;
- runtime-entered tokens remain usable in memory for the current page only;
- non-empty build-time endpoint-profile token values are ignored and never serialized;
- legacy flat feedback/share token config values are ignored and never serialized;
- example configuration now states that `os.environ` protects source control, not generated HTML;
- executable gates: `test_endpoint_secret_lifecycle.mjs`, `test_client_secret_boundary.py`, mutation positives.

B18 overall remains open: prompt authority, destination binding, capability-safe logging, contribution provenance, retention/deletion semantics, personal-data minimization, and supply-chain gates are not closed by Run 1.

### Run 2 — export / Share active-content isolation

- one canonical privacy-sanitized snapshot feeds JSON/HTML/TXT and Share;
- embedded JSON is HTML raw-text safe and round-trips hostile data;
- exported HTML denies scripts/network/object/form execution via CSP;
- self-contained `c2` carries validated structured snapshots, never rendered HTML;
- legacy `c1.html` and legacy IndexedDB HTML are inert text only;
- local filesystem source paths and URL credentials/query/hash are excluded from
  exported snapshot metadata;
- hostile-content and positive-control mutation gates are executable.

B18 overall remains open after Run 2: Global Share server representation/auth,
prompt authority, destination binding, capability-safe logging, contribution
provenance, retention/deletion semantics, personal-data minimization, and
supply-chain gates are not closed by browser/export isolation.

### Run 3 — Global Share server authority / capability separation

- client sends structured canonical snapshot + allowlisted format, never rendered HTML/MIME authority;
- HF and Cloudflare canonicalize again and own representation;
- public read ID is separate from private edit capability;
- PATCH/DELETE require the edit capability; server stores only its digest;
- browser keeps edit capability in memory only;
- Share responses are no-store/noindex/nosniff with sandboxed HTML CSP;
- explicit body/count/aggregate store budgets are enforced (Cloudflare aggregate check is conservative/eventually-consistent, not atomic);
- Share application logs omit public/private capabilities; bundled Uvicorn access logs are disabled;
- forwarded IP identity is opt-in on HF instead of blindly trusting caller X-Forwarded-For.

B18 remains open after Run 3: prompt authority/destination binding, centralized
logging + traceback minimization, contribution provenance/retention/deletion,
local sensitive-data preflight, and supply-chain/operational closure remain.

---


Status: **DESIGN REVIEW / CROSS-CUTTING SECURITY CONTRACT — production code not yet changed**
Date: **2026-08-29**
Depends on: **B16 Share Conversation UX**, **B17 Share Sheet / Export Content Isolation**
Feeds existing checkpoints: **B02 Security P0 Revalidation**, **B03 Client/Server Config Authority**, **B04 Prompt Authority**, **B05 Proxy Routing/CORS**, **B06 Auth/Identity/Limits**, **B07 Feedback Provenance**, **B17 Export/Share Content Isolation**

---

## 1. Decision

The assistant is open-source browser software backed by optional public proxy services. Therefore:

> **Assume every client-side implementation detail is public and bypassable.**

This includes:

- source code;
- DOM IDs and storage keys;
- endpoint routes;
- regexes and detection thresholds;
- system-prompt text present in the static bundle;
- format schemas;
- capability names;
- client-side validation;
- client-generated IDs;
- hidden/debug modes;
- build-time browser configuration.

No security property may depend on an attacker failing to discover one of those details.

Client-side controls remain valuable for **user protection, local hygiene, transparency, early warning, and defense in depth**, but every security-relevant authorization, identity, destination-binding, storage, and policy decision must be enforced again by the trusted server boundary.

---

## 2. Primary user-protection goal

The design target is not merely “prevent XSS.” It is:

> **Collect the least possible user data, keep it for the shortest useful period, avoid linking it to identity unless necessary, never place secrets in client-visible or model-visible contexts, and make every retained copy auditable, purpose-bound, revocable where technically possible, and difficult to abuse after compromise.**

This lowers impact from:

- malicious users;
- malicious page/document content;
- malicious model/provider output;
- malicious or compromised endpoint operators;
- direct API callers bypassing the browser;
- compromised logs or storage;
- malicious dependencies or contributors;
- accidental user disclosure;
- impersonation;
- data poisoning;
- coercion/extortion using retained sensitive conversations;
- bulk profiling or dossier construction.

---

## 3. Threat actors

### T1 — Direct API attacker

Calls `/v1/chat/completions`, `/v1/share`, `/v1/feedback`, or `/v1/contribute` directly without using the UI.

Assume this actor can:

- choose arbitrary JSON;
- choose arbitrary roles unless server rejects them;
- spoof every client-generated field;
- replay requests;
- bypass JavaScript regexes and warnings;
- send malformed Unicode and oversized/nested payloads;
- vary source IP/proxy headers where infrastructure permits.

### T2 — Malicious documentation/page author

Controls text that enters page context through documentation, docstrings, examples, generated API docs, third-party embeds, or compromised documentation content.

Goals may include:

- prompt injection;
- secret extraction;
- hidden instructions;
- Unicode/bidi deception;
- forcing model/tool behavior;
- poisoning exported/shared transcripts.

### T3 — Malicious model/provider/upstream

Returns attacker-controlled text, metadata, malformed streaming frames, HTML-looking payloads, secrets copied from context, fake model identity, or instructions intended to attack later consumers/exporters.

Model output is **untrusted input**, even if it comes from a configured provider.

### T4 — Malicious Share creator or recipient

Crafts Share fragments/URLs, uploads hostile content, or forwards links to exploit a viewer, phish recipients, or expose data.

### T5 — Same-origin code attacker

Any XSS, unsafe Blob/data viewer, compromised same-origin dependency, or other JavaScript running in the documentation origin can read browser storage available to that origin.

### T6 — Malicious/compromised operator or log reader

Has access to logs, dataset repositories, mirrors, proxy environment, or administrative dashboards.

The product cannot fully protect data from an operator who intentionally controls the server, but it **can minimize what the operator ever receives or retains**.

### T7 — Malicious open-source contributor / supply-chain dependency

Introduces a patch or dependency that exfiltrates data, weakens sanitizers, broadens CORS, changes endpoints, logs secrets, or disables regression tests.

### T8 — Accidental user disclosure

User pastes an API key, private URL, personal data, internal code, medical/legal/financial material, or confidential conversation without realizing it will leave the browser or be persisted.

---

## 4. Assets and data classes

### Class 0 — Public / expected-to-be-discoverable

Examples:

- open-source code;
- endpoint paths;
- client-side prompt templates;
- schemas;
- client detection patterns;
- public model catalog metadata.

**Rule:** never treat Class 0 secrecy as a control.

### Class 1 — Low-sensitivity operational metadata

Examples:

- status class;
- latency bucket;
- payload-size bucket;
- route name;
- static event code;
- feature enabled/disabled state.

May be logged with retention limits.

### Class 2 — Conversation content

Examples:

- user query;
- assistant answer;
- page/retrieved context;
- feedback comment;
- model output;
- error text copied into transcript.

Default policy: **transient unless user explicitly chooses a storage/share/contribution action.**

### Class 3 — Personal / sensitive / confidential content

Potentially includes:

- name, email, phone, address;
- private organization/project data;
- health, financial, legal, employment, relationship, or identity information;
- access patterns and exact local filesystem paths;
- private repository/document URLs;
- embarrassing or coercion-sensitive statements;
- confidential source code or business information.

The system often cannot perfectly identify Class 3 automatically. Therefore the primary control is **minimization**, not an overconfident PII detector.

### Class 4 — Secrets and authorization capabilities

Examples:

- API keys;
- OAuth/access/refresh tokens;
- session/authentication cookies;
- private keys;
- database credentials;
- Share edit/delete capability;
- any bearer locator that grants access to non-public content.

Rules:

1. never put in static browser configuration;
2. never put in prompts;
3. never log;
4. never persist in localStorage;
5. least privilege;
6. short lifetime where possible;
7. revocable;
8. destination-bound.

---

## 5. Existing maintenance security families already cover much of this

The current maintenance index already defines these durable security families:

- `SEC-P0-01` — server credential attached to configurable/unbound backend destination;
- `SEC-P0-02` — wildcard/permissive CORS;
- `SEC-P0-03` — Share locator doubles as edit authority;
- `SEC-P0-04` — alternate/direct inference relay bypasses controls;
- `SEC-P0-05` — caller/browser can supply authoritative `system` content;
- `SEC-P0-06` — feedback/training poisoning;
- `SEC-P0-07` — consent/version provenance insufficient/disabled;
- `SEC-P0-08` — untrusted forwarded client identity;
- `SEC-P0-09` — body/resource limits;
- `SEC-P0-10` — supply-chain/container hardening;
- `SEC-P0-11` — credential/token values reach browser/generated configuration.

B18 is an **umbrella user-protection contract**. It must not mark B04/B06/B07 complete automatically; their direct service-level bypass tests are still required.

---

# Part I — Confirmed source findings

## 6. CONFIRMED — client endpoint tokens are persisted despite a comment saying they are not

Source:

`_static/ai-assistant.js`

Observed behavior:

- `_sanitizeStoredProfile()` intentionally returns loaded profiles with `shareToken: ''` and `feedbackToken: ''`.
- `_persistCustom()` contains a comment saying token values are omitted and survive only the current page session.
- the actual object written to `localStorage` includes:

```js
shareToken:    p.shareToken    || '',
feedbackToken: p.feedbackToken || '',
```

and persists the object under:

```text
ai-assistant-ep-custom
```

This is a direct contradiction between intended and actual behavior.

### Impact

Any JavaScript executing in the same documentation origin can read those token values. B17 already discovered an arbitrary-HTML Blob execution path that makes this especially important.

### Decision

**P0/P1 depending token privilege. Fix before release.**

The persisted object must contain **no token fields at all** (or empty values produced before serialization). Add a regression test that searches the raw serialized localStorage value.

If a browser-entered token is supported at all, keep it in memory for the page lifetime only and label it accordingly.

---

## 7. CONFIRMED — endpoint URL userinfo is correctly rejected, but feature endpoint query strings remain possible

Good current control in `_sanitizeRuntimeUrl()`:

```js
if (parsed.username || parsed.password) {
    return ... 'Credentials must not be embedded in endpoint URLs.';
}
```

Base endpoints also reject query strings.

However, absolute feature endpoints call `_sanitizeRuntimeUrl(value, true)`, and relative feature routes may contain a query component.

### Risk

A user/operator can accidentally place credentials in:

```text
/v1/chat?api_key=...
https://service.example/api?token=...
```

Those URLs may then be displayed, persisted as profile configuration, copied, exposed in browser history/tooling, or leak through diagnostics/referrers in another implementation.

### Decision

Default endpoint policy should reject query-based credentials and preferably reject **all endpoint query parameters** unless a feature explicitly requires them.

If query parameters are allowed for a custom endpoint:

- reject secret-shaped parameter names such as `token`, `access_token`, `api_key`, `key`, `secret`, `sig`, `signature`, `auth`, `password`;
- never persist a secret-bearing query;
- never log it;
- never include it in export/share metadata;
- display origin/path without query in routine UI.

---

## 8. CONFIRMED — browser prompt containment is thoughtful, but server prompt authority is still bypassable

Good current client controls:

- explicit comment that prompt injection cannot be reliably detected;
- `_scanInjection()` used as a signal;
- invisible-character stripping;
- high-confidence secret redaction on page context;
- `_fenceUntrusted()` containment with per-request nonce;
- page context treated as untrusted reference text.

But `/v1/chat/completions` currently validates only body size before `_forward(body)` forwards the raw request upstream.

A direct caller can therefore bypass the browser and supply arbitrary OpenAI-compatible messages, including an authoritative-looking `system` role.

### Decision

This confirms the existing `SEC-P0-05` / B04 concern.

The service must not accept browser/client `system` or `developer` authority.

Recommended server request contract:

```json
{
  "model": "allowlisted-model-id",
  "user_message": "...",
  "context": {
    "page_text": "...",
    "page_descriptor": "..."
  },
  "reasoning": { "level": "low" },
  "stream": true
}
```

The server then constructs the actual provider request and authoritative system policy.

For backward OpenAI compatibility, a transitional route may accept `messages`, but it must:

- reject client `system` / `developer` roles;
- allow only bounded `user` content plus explicitly typed context;
- reject tools/function authority unless separately implemented;
- ignore/reject security-sensitive extra keys.

---

## 9. System-prompt extraction / “distillation” must not be solved by hiding the prompt

For this open-source client, any static client-side system prompt is discoverable directly from source. Even a server-owned hidden prompt should be assumed behaviorally inferable over time.

Therefore:

- system prompts contain **no credentials, tokens, private URLs, access-control rules that are themselves relied upon, or sensitive user data**;
- authorization is deterministic outside the model;
- destination binding is deterministic outside the model;
- contribution/share permissions are deterministic outside the model;
- “do not reveal this prompt” is not a security boundary.

Repeated-query prompt extraction and behavioral distillation may still be rate-limited as abuse, but confidentiality of a prompt must not protect any asset.

---

## 10. CONFIRMED — page-context secret scanner does not protect intentional user query input

`_redactSecrets()` is currently applied to page/document context before model submission. It is not applied to the user's typed question.

This is appropriate in one sense: silently rewriting a user's question can change meaning.

But it leaves accidental paste risk:

> user pastes an API key/private token into the chat and immediately sends it to the configured provider.

### Recommendation

Add a **local pre-send secret warning** for user-authored text:

```text
Possible secret detected
This message appears to contain an API key/token.
It will be sent to <provider hostname>.

[Remove detected secret]  [Send anyway]
```

Rules:

- high-confidence secret patterns may block until explicit override;
- do not send the text to another service merely to classify it;
- do not log the matched value;
- warning reports only type/count, e.g. `Possible OpenAI key ×1`;
- ordinary PII detection remains advisory because false positives/negatives are unavoidable.

---

## 11. CONFIRMED — model/provider identity metadata is not authentication

Conversation records contain values such as `model_id`, `model_provider`, and `model_name`. The hostile fixture also demonstrates a custom endpoint can label itself `stub/hostile`.

Client-supplied or endpoint-reported model identity must be treated as **claimed metadata**, not proof of origin.

### Recommendation

Distinguish:

- `requested_model` — what the client asked for;
- `resolved_model` — what the trusted proxy routed to;
- `provider` — server-observed provider route;
- `identity_assurance` — `server-observed | endpoint-reported | client-configured`.

UI should never use a provider logo alone to imply verified identity.

Custom endpoints should visibly show:

```text
Custom endpoint · example.org
Identity reported by endpoint
```

A server-owned response header/body field may attest the actual resolved route for the built-in proxy.

---

# Part II — Secrets and capability architecture

## 12. CONFIRMED — build-time Share/Feedback tokens are baked into rendered HTML

This is not only a general open-source warning; the current source explicitly implements it.

`__init__.py` reads `ai_assistant_panel_feedback_token` and `ai_assistant_global_share_token` into `AI_ASSISTANT_CONFIG`, then injects that configuration into a `<script>` on every rendered page. Endpoint profiles likewise accept `shareToken` / `feedbackToken`, and the source comments state that **all profiles are baked into rendered HTML at build time**. The example configuration recommends values such as `os.environ.get("SHARE_WRITE_TOKEN")`.

Reading a secret from `os.environ` during a **static documentation build** protects it from source control, but it does not keep it server-side: the resulting value is serialized into the generated site and is readable by every visitor.

This confirms the existing `SEC-P0-11` family.

### Decision

Remove secret-bearing token fields from Sphinx/browser configuration entirely. A static site may contain an endpoint URL and public feature metadata, but no reusable bearer credential.

Any token embedded by Sphinx into static JavaScript can be read by every visitor who can load the page.

Therefore build-time values such as Share/Feedback bearer tokens cannot be treated as confidential authentication secrets.

### Allowed uses

A public/static value may be:

- an application identifier;
- a low-value routing hint;
- a public anti-CSRF nonce only when paired with a server-side session boundary;
- a feature flag.

It must not authorize access to private data or privileged storage merely because users are unlikely to inspect JavaScript.

### Preferred design

Sensitive upstream tokens:

```text
Browser
   |
   | no upstream secret
   v
Trusted proxy
   |
   +-- server env / secret manager --> provider
```

Share mutation:

```text
POST share
   |
   +--> public read ID
   +--> separate high-entropy edit/delete capability

server stores hash(edit capability)
client keeps edit capability session-only unless user explicitly exports it
public URL never contains edit capability
logs never contain either capability in full
```

---

## 13. CONFIRMED — inference token can be injected into an arbitrary configured BACKEND_URL

In `_shared_logic._resolve_upstream_url()` Path 1:

```python
if backend_url:
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
    return backend_url, headers, proxy_timeout
```

This is the existing `SEC-P0-01` family.

### Decision

A provider credential must be destination-bound.

Do not attach `HF_TOKEN` to an arbitrary generic `BACKEND_URL` merely because both are configured.

Use separate configuration such as:

```text
BACKEND_URL=https://custom.example/v1/chat
BACKEND_AUTH_ENV=AI_BACKEND_TOKEN_CUSTOM
BACKEND_ALLOWED_HOSTS=custom.example
```

Or define explicit provider profiles server-side with bound secret slots.

A Hugging Face token should be sent only to an allowlisted Hugging Face destination set.

---

## 14. Never log partial secrets

`_token_log_fragment()` currently returns first 8 + last 4 characters of a token.

Although current search found no normal production call site using the helper, this API encourages partial secret disclosure.

### Decision

Replace with one of:

```text
<not-set>
<set:read>
<set:fine-grained>
```

If correlation between token configurations is genuinely needed, use a **keyed server-side HMAC fingerprint**, never a visible token substring.

---

# Part III — Logging and record keeping

## 15. Logging rule: metadata, not content

Operational/security logs must not contain:

- user query;
- assistant answer;
- page context;
- feedback free text;
- raw request/response bodies;
- Authorization/Cookie headers;
- API keys/tokens;
- full Share IDs when those IDs grant read access;
- edit/delete capabilities;
- raw session/conversation IDs;
- exact local file paths;
- URL query/fragment;
- email/phone/government identifiers unless there is a documented legal/operational need.

### Recommended operational event schema

```json
{
  "event": "share.create",
  "request_id": "random-non-capability-id",
  "ts_bucket": "2026-08-29T02:00Z",
  "route": "share",
  "status_class": "2xx",
  "bytes_bucket": "16-32KiB",
  "latency_bucket": "100-250ms",
  "rate_limit": "allowed"
}
```

No content is needed to answer “is the service healthy or under abuse?”

---

## 16. CONFIRMED — Global Share read capabilities are logged in full

HF proxy currently logs:

```json
{"event":"share.create","id":"<share_id>", ...}
{"event":"share.update","id":"<share_id>", ...}
```

Cloudflare Worker currently logs full Share UUID for write/read/miss events.

The Share UUID is presently a bearer read locator. Therefore logs can become an alternate directory of private shared conversations.

### Decision

**Do not log full Share IDs.**

Use:

- independent random request/event ID; or
- keyed HMAC of share ID for correlation, truncated to a non-reversible operational fingerprint.

Never use an unkeyed ordinary hash for a bearer capability.

---

## 17. CONFIRMED — Cloudflare logs feedback session ID and uses a weak deterministic IP hash

Cloudflare currently logs feedback events with `sessionId` and a non-cryptographic `_ipHash()` based on a 31-multiplication string hash.

The source correctly states `_ipHash` is not cryptographic and is for rate-limit keys, not security. However, the same deterministic value is also written to logs.

### Risk

IP addresses have a small/structured search space. An unkeyed deterministic hash is pseudonymization, not anonymization, and can be brute-forced/correlated.

### Decision

For rate-limit KV keys, prefer:

```text
HMAC-SHA-256(rotating_server_pepper, canonical_client_ip)
```

with a rotation window matching the abuse-control need.

For logs, omit even the pseudonym unless correlation is necessary.

Feedback session/conversation IDs should not be logged directly.

---

## 18. CONFIRMED — HF log redaction does not sanitize separately rendered traceback text

Current pipeline:

1. `_RedactingFilter` sanitizes `record.getMessage()` and writes it back to `record.msg`.
2. `_StructuredFormatter` later adds:

```python
payload["exc_info"] = self.formatException(record.exc_info)
```

The separately formatted traceback does not pass through `_RedactingFilter`'s string substitutions.

### Impact

A third-party exception/traceback containing credentials, URLs, PII, or provider response text may bypass the current redaction filter.

### Decision

Create one central:

```python
sanitize_log_text(text: str) -> str
```

and apply it **after formatting** to:

- event/message;
- traceback/exception text;
- any structured string field that may contain external data.

Better still, custom application paths should log fixed error codes and exception class names rather than exception bodies.

---

## 19. CONFIRMED — server log-redaction vocabulary is narrower than browser secret detection

Server `_REDACT_PATTERNS` currently covers:

- Hugging Face `hf_...` tokens;
- IPv4 addresses.

It does not generally cover other token families, JWTs, Authorization header values, cookies, email addresses, URL query secrets, private-key blocks, etc.

The browser already contains a broader high-confidence secret detector for several provider/key families.

### Decision

Maintain a **shared adversarial secret corpus** across JS and Python tests.

The implementations need not use identical regexes, but tests should cover at least:

- Hugging Face;
- OpenAI;
- Anthropic;
- GitHub;
- Slack;
- Google API keys;
- AWS-style access IDs;
- JWTs;
- private-key headers;
- `Authorization: Bearer ...`;
- cookie/session-token examples;
- secret-bearing URL query parameters.

Logging sanitizer should replace values with reason codes/types, never the original.

---

## 20. Protect logs against injection and resource abuse

User-controlled text should not become event names or unbounded log fields.

Rules:

- event names come from server constants;
- CR/LF/control characters removed from any externally derived diagnostic string;
- every string field length-bounded;
- no arbitrary JSON object from caller merged into log event;
- no raw provider response body on error;
- rate-limit repeated malformed/log-generating requests;
- logging must not be usable to exhaust disk/ingestion quota.

---

# Part IV — Feedback, contribution, training and data poisoning

## 21. CONFIRMED — feedback persistence default and documentation contradict each other

Configuration currently sets:

```python
FEEDBACK_PERSIST_ENABLED = ... default "true"
```

and comments explicitly describe the opt-out default.

But `/v1/feedback` docstring states that `false` is the default.

### User-protection issue

When storage credentials are configured, a routine feedback action may persist content under an opt-out default, while documentation suggests persistence is opt-in.

### Decision

For an open-source documentation assistant, default to **least data**:

```text
FEEDBACK_PERSIST_ENABLED=false
```

A simple thumbs-up/down should not automatically persist full query/answer text.

If an operator enables feedback persistence, the UI and deployment documentation must state exactly what fields are stored and for how long.

---

## 22. CONFIRMED — feedback records can persist full conversation content and metadata

Canonical feedback storage includes fields for:

- `conversationId`;
- `feedbackId`;
- `answerIndex`;
- rating fields;
- free-text `message`;
- full `query`;
- full `answer`;
- `model`;
- `page`;
- `consentVersion`;
- timestamps/provenance fields.

This is a valuable research record, but it is also a high-value sensitive corpus if the user discussed private information.

### Decision: split feedback from content contribution

#### Rating telemetry

Default feedback payload/storage:

```json
{
  "rating": "helpful",
  "answer_index": 3,
  "resolved_model": "server-observed-id",
  "page_category": "optional-coarse-path",
  "server_ts": "..."
}
```

No query, answer, comment, raw page URL, or stable conversation identifier by default.

#### Content contribution

Full query/answer may be stored only after a separate explicit contribution action with field preview and versioned consent.

---

## 23. CONFIRMED — contribution consent is client-asserted and version enforcement is disabled

`POST /v1/contribute` currently rejects when `consentFlag` is false, but a direct API caller can set it to true.

`CONSENT_VERSION_ENABLED` is currently false, so the stored `consentVersion` resolves to `null` and no live consent-text version is bound to the record.

### Important distinction

A boolean supplied by an unauthenticated client proves only:

> “this HTTP request contained `consentFlag=true`.”

It does **not** prove which user saw which consent text or whether the record is authentic training data.

### Decision

Do not label the endpoint “GDPR-gated” as a security/compliance guarantee until the deployment has a reviewed consent/data-governance design.

Recommended consent receipt:

```json
{
  "receipt_id": "server-generated-random-id",
  "policy_version": "2026-08-29.1",
  "purpose": "model-quality-research",
  "selected_fields": ["query", "answer", "rating"],
  "accepted_at": "server timestamp",
  "source": "web-widget",
  "deletion_capability_hash": "..."
}
```

The server remains unable to prove a human identity without authentication, so records must be labeled **unauthenticated user contribution**, not “verified user data.”

---

## 24. CONFIRMED — dataset poisoning remains possible through direct unauthenticated submissions

A direct caller can fabricate:

- query;
- answer;
- rating;
- model label;
- page URL;
- conversation ID;
- feedback ID;
- contribution records.

Rate limiting does not establish authenticity.

### Decision

Contributions enter a **quarantine/staging tier**, never an automatic training-ready dataset.

Recommended pipeline:

```text
untrusted contribution
       |
       v
bounded intake / consent receipt
       |
       v
quarantine store (short TTL)
       |
       +--> secret/PII warning scan
       +--> schema validation
       +--> provenance normalization
       +--> duplicate/sybil/anomaly signals
       +--> moderation/review as required
       |
       v
sanitized approved example
       |
       v
training dataset
```

Each stored row must retain provenance state such as:

```text
source = unauthenticated_web
server_received_at
requested_model
resolved_model (if server-observed)
validation_state
consent_policy_version
review_state
```

Do not transform “client said model X” into “model X produced this” without server evidence.

---

## 25. Retraction tombstones are not physical deletion

Current guidance uses retraction tombstones to suppress older ratings from the **clean training output**.

Raw records are written through Git-like provider commits (Hugging Face/GitHub/GitLab/Bitbucket storage adapters). A tombstone can make downstream training ignore an earlier record while the earlier content remains present in raw repository history/mirrors.

### Consequence

A UI action named “remove”, “retract”, or “delete my contribution” must not imply physical erasure unless all persisted copies/history/mirrors are actually handled.

### Decision

Separate concepts:

- **Withdraw from training** — logical tombstone; stops future training selection.
- **Delete stored personal content** — physical deletion/erasure workflow, if supported.

For sensitive raw user conversations, a Git-history-backed repository is a poor **intake store** because deletion is difficult and replication is easy.

Recommended architecture:

1. short-lived mutable quarantine store for raw contributions;
2. user-held deletion capability/receipt;
3. sanitization/review;
4. only sanitized/de-identified approved examples promoted to append-only training repositories;
5. raw quarantine automatically expires.

---

# Part V — Personal information and coercion/extortion risk

## 26. Design against breach impact, not only unauthorized access

No detector can guarantee that a conversation contains no personal or sensitive information.

Therefore the best defense against later coercion/extortion (“blackmail”) from a compromised corpus is:

- do not collect unnecessary content;
- do not retain it indefinitely;
- do not join it to stable identity;
- do not copy it into logs;
- do not mirror raw content widely;
- do not expose searchable operator dashboards by default;
- do not retain exact URLs/local paths;
- make sharing/training explicit and purpose-specific;
- use short retention for raw intake;
- support real revocation/deletion where promised.

The product should never create a hidden cross-session dossier merely because identifiers are convenient for analytics.

---

## 27. No stable identity by default

This assistant does not need a real-world user identity for ordinary question answering.

Therefore:

- no name/email/account field should be introduced merely for analytics;
- IP address is an abuse signal, not identity;
- `conversationId` is client-generated state, not identity or authorization;
- `sessionId` is an idempotency/correlation value, not identity;
- model labels are metadata, not identity proof;
- share ID is a read capability, not identity;
- edit/delete token is a capability, not identity.

If a future authenticated account mode is added, keep account identity out of training/share payloads unless specifically required and consented.

---

## 28. PII/sensitive-content detection is a warning system, not a guarantee

Optional local detection can help users notice likely:

- email addresses;
- phone numbers;
- private keys/API tokens;
- government-identifier-like patterns;
- URLs containing query secrets;
- filesystem paths;
- invisible/bidi characters.

But copy must say:

```text
Possible sensitive information detected
```

not:

```text
This conversation is safe / contains no PII
```

High-confidence secrets can be deterministically redacted or blocked with user override. General PII detection should normally warn and let the user inspect/select fields.

---

## 29. Add a “Privacy preflight” before Global Share and Contribution

Before a network persistence action:

```text
Privacy check

Destination      Global Share · scikit-plots-ai.hf.space
Retention        30 days
Messages         16
Possible secrets 0
Possible personal-data patterns 2
Invisible/bidi   4
Source URL        sanitized
Session ID        excluded

[Review data]  [Create global link]
```

For training contribution:

```text
Data contribution

Purpose           model-quality research
Retention         operator policy link
Fields            query, answer, rating
Identity          not requested
Source page        excluded/sanitized
Deletion receipt  provided after submit

[Preview exact data] [Contribute]
```

Never send the data to a remote “PII scanner” before the user consents to the original destination.

---

## 30. Privacy modes

Provide a small product-level mode, not dozens of obscure toggles.

### Local-only

- inference only if explicitly initiated;
- no Global Share;
- no feedback content persistence;
- no training contribution;
- local preview/download available;
- transcript persistence optional/local.

### Standard — recommended default

- inference to selected endpoint;
- feedback rating may be sent as minimal metadata;
- no query/answer persistence from feedback;
- Global Share only when user explicitly creates it;
- contribution separate and explicit.

### Research contribution

- exposes contribution action;
- exact data preview;
- versioned consent;
- deletion/withdrawal receipt;
- never silently enabled by selecting Share.

---

# Part VI — Share and export privacy

## 31. Preserve B17 Content & Privacy architecture, strengthen defaults

B17 already defines Content & Privacy as a first-class Share decision axis and requires filters to run before serialization.

Strengthen the default **Standard** preset:

Recommended default Share fields:

- Messages: on;
- Timestamps: **off by default unless useful**;
- Model/provider: on, but label assurance;
- Ratings/feedback: off by default for ordinary Share;
- Error messages: on only when user wants debugging context;
- Page title: optional;
- Safe source page: off by default or origin/path only;
- Session identifier: always off unless explicit diagnostic export;
- raw page URL/query/hash/file path: never.

“Complete” must still exclude credentials, query secrets, local paths, edit tokens, browser storage, and hidden security state.

---

## 32. Share links must not become surveillance identifiers

Server should not build analytics keyed forever by Share UUID.

Operational counters should be aggregate where possible.

For `GET /v1/share/{id}`:

- `Cache-Control: private, no-store` unless public archival is explicitly intended;
- `Referrer-Policy: no-referrer`;
- `X-Robots-Tag: noindex, nofollow, noarchive`;
- restrictive CSP / sandboxed viewer;
- no tracking pixels/remote dependencies;
- no third-party analytics inside shared HTML.

---

# Part VII — Network identity, impersonation and abuse

## 33. CONFIRMED — HF proxy trusts leftmost X-Forwarded-For

Current `_client_ip()` uses the first value of `X-Forwarded-For` with a comment that it represents the original client.

That is only safe when the trusted ingress guarantees/sanitizes the header. A direct or differently proxied deployment can allow caller-controlled leftmost values.

### Decision

Make trusted-proxy behavior explicit:

```text
TRUSTED_PROXY_MODE=hf-space | direct | custom
TRUSTED_PROXY_HOPS=N
```

Prefer a platform-provided trusted client-IP header when documented.

If the trusted boundary cannot be established, treat IP as `unknown` and use non-IP abuse controls rather than trusting arbitrary forwarding headers.

Never use IP address to make authorization decisions.

---

## 34. Layer abuse controls; do not invent identity from rate limiting

Rate limits are useful against bulk abuse but do not prove a user.

Use independent controls:

- body-size limits before expensive parsing;
- maximum nesting/record count;
- per-route payload limits;
- global memory/store quotas;
- request concurrency limit;
- rate limits based on trusted edge signal;
- anonymous proof-of-work/challenge only if abuse warrants it;
- authenticated quotas only if a real account system exists.

Log only reason codes/buckets.

---

# Part VIII — Health/debug/admin exposure

## 35. Public health endpoint should expose less deployment detail

Current health/config response includes routing destinations, namespace configuration, token-presence/type flags, dataset/storage readiness, and CORS origins.

None of those values is itself equivalent to a secret, but the combined response gives unnecessary deployment reconnaissance.

### Recommendation

Public:

```json
{
  "status": "ok",
  "service": "sphinx-ai-assistant-proxy",
  "version": "..."
}
```

Detailed diagnostics:

- disabled by default; or
- protected by operator/admin access; or
- available only in local/dev mode.

Do not expose dataset repository paths or backend topology without a real operational need.

---

# Part IX — Browser storage

## 36. Storage classification

### localStorage

Allowed:

- harmless UI preferences;
- selected format;
- theme;
- endpoint profile IDs/labels and non-secret URLs after sanitization.

Forbidden:

- API/auth tokens;
- share edit/delete capabilities;
- session/authentication tokens;
- raw sensitive conversations unless a clearly separate user-controlled local-history feature explicitly promises it.

### sessionStorage / in-memory

May hold short-lived browser-only capability state, but XSS can still read it. It reduces persistence, not script-origin risk.

### IndexedDB

Treat as persistent local data. Do not call it “private” merely because it is local. Same-origin scripts can access it under the web security model.

Add “Clear AI Assistant local data” that deterministically clears transcript/history/share state/preferences while preserving only explicitly chosen configuration if desired.

---

# Part X — Supply chain / open-source contributor threat

## 37. Open source increases auditability but not automatic safety

Attackers can study the code, and malicious contributions/dependencies can alter trusted behavior.

Security-sensitive files should have stronger review/test gates:

- `_static/ai-assistant.js` security helpers;
- proxy routing/auth/CORS;
- storage adapters;
- Share serializer/viewer;
- dataset schema/contribution logic;
- Docker/requirements/workflow files.

### Recommended repository gates

- secret scanning in commits/PRs;
- dependency vulnerability scanning;
- pinned/locked deploy dependencies with hashes where practical;
- SBOM for release/service image;
- minimal container user/permissions;
- digest-pinned base image for production reproducibility where operationally feasible;
- no remote runtime scripts in exported/shared viewer;
- security regression fixtures required for changes to sanitizer/auth/storage code;
- signed release/checksum artifacts where release process supports it.

The existing `SEC-P0-10` checkpoint should own the exact implementation details.

---

# Part XI — Detection strategy

## 38. Detection matrix

| Problem | Detection reliability | Product action |
|---|---:|---|
| known API-key syntax | relatively high | local redact/block+override |
| private-key header | high | local redact/block+override |
| URL userinfo | deterministic | reject |
| secret-bearing query key | high | reject/warn |
| bidi/invisible control | deterministic | warn/inspect; context may strip |
| raw HTML/script breakout | deterministic | serializer must make impossible |
| prompt injection | fundamentally incomplete | signal only; isolate authority |
| general PII | incomplete | advisory warning, not guarantee |
| malicious model output | impossible to classify perfectly | treat all output as untrusted data |
| forged client identity | deterministic architectural issue | server ignores unauthenticated claims |
| data poisoning | probabilistic + provenance | quarantine/review, never auto-train |
| abusive request volume | measurable | rate/resource controls |

### Critical principle

**Detection does not replace isolation.**

Do not attempt to make HTML safe by detecting `<script>`. Encode by context so every string is safe regardless of content.

Do not attempt to make prompt authority safe by detecting “ignore previous instructions.” Keep untrusted content out of the authoritative role.

Do not attempt to make client identity trustworthy by validating UUID shape. Use server capabilities/authentication when authority is required.

---

# Part XII — Recommended data-flow architecture

## 39. Inference

```text
page/document content [UNTRUSTED]
      |
      +--> local normalize / secret redaction / bidi cleanup
      |
user query [INTENTIONAL USER DATA]
      |
      +--> local high-confidence secret preflight
      |
      v
structured client request
      |
      v
trusted proxy
      |
      +--> schema/role/model validation
      +--> authoritative system policy construction
      +--> destination-bound server secret
      |
      v
provider
      |
      v
untrusted model output
      |
      +--> safe renderer
      +--> no raw HTML authority
```

## 40. Share

```text
conversation
   |
   v
canonical privacy-filtered snapshot
   |
   +--> Local preview  (no upload)
   +--> Self-contained structured envelope (not executable HTML payload)
   +--> Global Share
            |
            v
      trusted server renderer/storage
            |
      public read capability
      separate edit/delete capability
```

## 41. Feedback

```text
rating click
   |
   v
minimal non-content telemetry  ----> optional operational store

"Contribute content" action
   |
   v
exact data preview + versioned consent
   |
   v
quarantine + provenance + TTL
   |
   v
sanitized approved training example
```

Operational telemetry and research content must be separate stores with separate purposes/access/retention.

---

# Part XIII — User-facing copy rules

## 42. Never overclaim privacy/security

Avoid:

- “private” when a bearer URL exists;
- “only you can access” for Blob/localStorage state;
- “secure link” without defining the property;
- “anonymous” when IP/session metadata is retained;
- “GDPR compliant/gated” based only on a client boolean;
- “deleted” when only a tombstone/local cache is removed;
- “verified model” when identity is only client/endpoint supplied;
- “no PII detected” as proof of safety.

Prefer:

- “Local preview — nothing uploaded”;
- “Self-contained — not encrypted, not revocable after sharing”;
- “Global link — stored for N days”;
- “Possible sensitive information detected”;
- “Withdraw from training” versus “Delete stored data”;
- “Custom endpoint — identity not verified by scikit-plots”.

---

# Part XIV — New/updated security gates

## 43. Gate G1 — Client secret persistence

A generated profile containing recognizable test tokens must leave **zero token bytes** in:

- localStorage;
- IndexedDB;
- exported profile JSON;
- URL/hash;
- console/log output.

## 44. Gate G2 — Direct system-role bypass

Raw HTTP request to the service containing caller `system` / `developer` authority must be rejected or deterministically demoted to untrusted user/context data.

Browser behavior is not evidence for this gate.

## 45. Gate G3 — Secret destination binding

A server token configured for Provider A must never be attached to arbitrary Provider B/custom backend.

Mutation test changes target host and verifies `Authorization` is absent.

## 46. Gate G4 — Log capability leakage

Seed logs/exceptions with:

- API keys;
- JWT;
- Authorization header;
- IPv4/IPv6;
- email;
- secret query URL;
- Share ID/edit token;
- traceback containing secret.

Assert emitted structured log contains no sensitive raw value.

## 47. Gate G5 — Feedback minimal-default

Default feedback action must not persist query/answer/message/page.

Explicit contribution is a separate tested flow.

## 48. Gate G6 — Consent provenance

Contribution persistence requires active policy version and server-created consent receipt/provenance state.

A direct `{"consentFlag":true}` request alone cannot be labeled verified consent.

## 49. Gate G7 — Identity spoofing

Direct caller supplied:

- conversationId;
- model/provider;
- forwarded IP;
- source label;

must never become server-trusted identity/provenance without an independent server observation.

## 50. Gate G8 — Real deletion semantics

Any UI action named `Delete` must have a test proving which stores/copies become inaccessible or removed.

If raw Git history/mirrors remain, call the operation `Withdraw from training` instead.

## 51. Gate G9 — Personal-data minimization

Standard Share/Feedback outputs must exclude:

- local filesystem path;
- query/hash;
- session identifier;
- auth/query secret;
- raw endpoint token;

by construction across JSON/HTML/TXT/YAML/TOML and Global Share.

## 52. Gate G10 — No bearer capability in logs

Full Share IDs, edit/delete tokens, session auth tokens and password-reset-like capabilities must never be present in log fixtures.

## 53. Gate G11 — Supply-chain regression

Security-test mutation suite must fail when:

- sanitizer call removed;
- server role validation removed;
- token persistence restored;
- CORS broadened to `*` in production fixture;
- Share UUID logged;
- feedback content persistence enabled as default;
- consent version check disabled in a production policy fixture.

---

# Part XV — Suggested implementation sequence

## 54. Phase A — immediate P0/P1 containment

1. Fix `localStorage` token persistence contradiction.
2. Remove full Share IDs/capabilities from logs.
3. Fix HTML export breakout and disable arbitrary `c1.html` execution (B17).
4. Make Global Share server-owned representation / non-arbitrary MIME (B17).
5. Separate Share read and edit capabilities.
6. Stop forwarding caller-owned system authority; server constructs policy.
7. Bind provider secrets to explicit destinations.
8. Change production CORS default away from wildcard.

## 55. Phase B — privacy-by-default

9. Set feedback content persistence default false.
10. Split minimal rating telemetry from explicit content contribution.
11. Enable real versioned consent receipt/provenance.
12. Sanitize page/source URLs for every storage/export path.
13. Add local user-query secret preflight.
14. Add Share/Contribution privacy preflight.
15. Stop logging stable session/feedback IDs.
16. Centralize post-formatting log sanitizer.

## 56. Phase C — retention and data governance

17. Define raw contribution TTL.
18. Add withdrawal versus physical deletion semantics.
19. Move raw contribution intake away from append-only Git history when real erasure is required.
20. Define mirror deletion/retention policy.
21. Add user-held deletion/withdrawal receipt/capability.
22. Make operational logs and research datasets separate stores.

## 57. Phase D — provenance / anti-poisoning

23. Server-attest resolved model/provider route.
24. Treat all client source/model/session claims as untrusted.
25. Quarantine contributions before training promotion.
26. Add anomaly/sybil/dedup signals without constructing durable user identity.
27. Add moderation/review state and provenance fields.

## 58. Phase E — supply chain and production hardening

28. Lock/pin deploy dependencies reproducibly.
29. SBOM + secret/dependency scanning.
30. least-privilege container/service configuration.
31. real-browser security E2E.
32. periodic retention/deletion/log-redaction tests.

---

# Part XVI — Final non-negotiable invariants

1. **Open-source client code is public; no secret depends on obscurity.**
2. **Client validation is never the sole service security boundary.**
3. **No reusable server credential is embedded in static documentation.**
4. **No browser credential/capability is persisted in localStorage.**
5. **No bearer capability is logged in full.**
6. **No secret is placed in a system prompt.**
7. **Client cannot assert authoritative system/developer role to the trusted proxy.**
8. **Untrusted page content and model output remain data, never authority.**
9. **Provider credentials are destination-bound and least-privilege.**
10. **IP/session/conversation IDs are not user identity.**
11. **Client model/provider labels are not identity proof.**
12. **Feedback rating does not silently become full-content training collection.**
13. **Content contribution is explicit, previewable, purpose-bound and versioned.**
14. **Contributed data is untrusted/quarantined until reviewed/validated.**
15. **Standard Share excludes local paths, query strings, fragments and session IDs.**
16. **PII detection is advisory; minimization is the primary protection.**
17. **Operational logs contain metadata, not conversation content.**
18. **Tracebacks/errors pass through the same redaction boundary as messages.**
19. **Share read and edit/delete capabilities are distinct.**
20. **A UI action called Delete must have real deletion semantics.**
21. **Withdrawal from training is not misrepresented as physical erasure.**
22. **Raw sensitive contribution storage has bounded retention.**
23. **Mirrors/history are included in retention/deletion design.**
24. **No remote analytics/scripts are embedded in shared/exported conversation viewers.**
25. **Every security-sensitive change is exercised by direct-service and adversarial mutation tests.**

---

## 59. External security baseline used for recommendations

These are external references, not claims about the current source:

- OWASP Logging Cheat Sheet — avoid directly logging session IDs, access tokens, sensitive PII, passwords, connection strings and keys; protect logs as sensitive assets.
  https://cheatsheetseries.owasp.org/cheatsheets/Logging_Cheat_Sheet.html

- OWASP Secrets Management Cheat Sheet — secrets should be short-lived where practical, least-privilege, revocable and never logged.
  https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html

- OWASP LLM Prompt Injection Prevention Cheat Sheet — prompt injection includes indirect content, obfuscation/Unicode, system-prompt extraction and data exfiltration; isolation/least privilege are required beyond pattern detection.
  https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html

- OWASP GenAI LLM07 System Prompt Leakage — system prompts should not be treated as secrets/security controls and should not contain credentials or other sensitive authorization material.
  https://genai.owasp.org/llmrisk/llm072025-system-prompt-leakage/

- OWASP Session Management / HTML5 Security guidance — authentication/session credentials should not be stored in browser Web Storage because same-origin JavaScript can read them.
  https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html
  https://cheatsheetseries.owasp.org/cheatsheets/HTML5_Security_Cheat_Sheet.html

---

## 60. Closure criterion

B18 is not closed by adding more regexes.

It closes only when the architecture proves:

```text
least data
+ explicit purpose
+ server-owned authority
+ destination-bound secrets
+ no credential persistence/logging
+ bounded retention
+ honest identity/provenance labels
+ real capability separation
+ direct API bypass tests
+ adversarial client/export tests
```

The desired outcome is that compromise of one layer exposes **as little useful user data as possible**, rather than relying on every layer remaining perfect forever.
