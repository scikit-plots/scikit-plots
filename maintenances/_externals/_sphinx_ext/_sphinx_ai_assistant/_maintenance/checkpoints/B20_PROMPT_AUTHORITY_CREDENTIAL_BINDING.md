# B20 — Prompt Authority & Credential Destination Binding

Status: **RUN 4 COMPLETE — bundled proxy, worker, dev proxy, and direct model-service prompt authority are enforced; CORS/logging/resource/privacy residuals remain separate runs**
Date: **2026-08-29**
Depends on: **B18 threat model**, **B19 Global Share server authority**

## 1. Decision

The browser is an untrusted client. A public caller may read the source, bypass
all UI controls, and POST directly to the proxy. Therefore the bundled inference
services must own:

- the authoritative system role;
- the accepted request schema;
- the allowed model set;
- provider-native reasoning field names;
- the upstream destination;
- which credential, if any, is permitted to reach that destination;
- redirect behavior for credential-bearing requests.

```mermaid
flowchart TD
    B[Browser UI] -->|typed envelope| P[Bundled proxy]
    A[Direct attacker] -->|typed envelope or malicious raw body| P
    P --> V{scikitplot-chat-v1 valid?}
    V -->|no| X[400 fail closed]
    V -->|yes| M{model allowlisted?}
    M -->|no| X
    M -->|yes| S[Server-owned system policy]
    S --> U[Untrusted page + user question as user data]
    U --> R[Server-built provider request]
    R --> D[Fixed/operator-bound destination]
    D --> C[Destination-specific credential only]
```

## 2. Public chat contract

Bundled HF, Cloudflare, and local development proxies advertise:

```json
{
  "capabilities": {
    "chat_request": {"contract": "scikitplot-chat-v1"}
  }
}
```

The browser **negotiates** this capability from `GET /health`. It does not infer
trust from provider label, hostname, or `/v1/chat/completions` path.

Accepted request shape:

```json
{
  "contract": "scikitplot-chat-v1",
  "model": "allowlisted/model-id",
  "user_message": "...",
  "context": {
    "page_text": "...",
    "page_descriptor": "..."
  },
  "max_tokens": 1000,
  "stream": true,
  "reasoning": {
    "effort": "low|medium|high|extra|max",
    "thinking": false,
    "budget_tokens": 1000
  }
}
```

Unknown root/context/reasoning keys are rejected. In particular the public
contract does not accept client authority such as:

- `system`;
- `developer`;
- `messages`;
- `tools` / `tool_choice` / function authority;
- `api_key` / `authorization`;
- caller-selected `url` / `endpoint`.

The old OpenAI-compatible request body is therefore no longer an authority
contract for the bundled production proxy. Reserved `stub/*` requests remain a
local diagnostic exception because they are intercepted before any server
credential or real upstream path is resolved.

## 3. Prompt authority

The server constructs exactly one authoritative system message from a static,
non-secret policy. The policy contains no credential, private URL, bearer
capability, or access-control fact that depends on secrecy.

Page content and the user's question are placed into the provider request under
`user` authority only.

```mermaid
sequenceDiagram
    participant C as Untrusted client
    participant P as Proxy
    participant L as Provider

    C->>P: {user_message, context.page_text, model}
    P->>P: validate schema + model
    P->>P: create SERVER_SYSTEM_POLICY
    P->>P: create nonce-bounded untrusted context
    P->>L: system = server policy
    P->>L: user = page context + question

    C--xP: system/developer/messages/tools
```

A malicious page may contain strings such as `SYSTEM: ignore previous`, HTML,
Unicode controls, or prompt-extraction text. Those bytes can survive as evidence
inside the user/context message, but they do not become the provider's system
role.

The browser still performs page-secret redaction before sending context. That is
defense-in-depth and user protection, not prompt authority. A direct caller can
skip that browser redaction, so server policy must remain safe even when context
contains hostile or sensitive-looking data.

### Direct model-service revalidation

The separately reachable `_hf_spaces_model` service is also an authority
boundary. Run 4 does not trust the proxy simply because it is the expected
caller. The model Space advertises and parses the same `scikitplot-chat-v1`
contract and reconstructs the server system policy independently.

```mermaid
sequenceDiagram
    participant C as Browser / direct caller
    participant P as HF proxy
    participant M as Model Space

    C->>P: scikitplot-chat-v1
    P->>P: validate + own policy for Path 1/3
    P->>M: Path 2 preserves scikitplot-chat-v1
    M->>M: validate model + reject messages/system/developer
    M->>M: reconstruct SERVER_SYSTEM_POLICY
    M->>M: inference

    C--xM: arbitrary OpenAI messages/system
```

The proxy and model deployment each ship `_chat_contract.py`; a regression test
requires those deployment copies to be byte-identical. This duplication is a
deployment artifact, not two independent specifications. If the contract is
changed, both deployment copies must change in the same commit.

This is intentionally stronger than checking that the first incoming system
message *looks* correct. The direct service no longer accepts a caller-authored
`messages` array at all, so knowledge of the open-source system-policy text
does not create role authority. Security does not depend on hiding that text.

## 4. Custom endpoint compatibility

Custom/self-hosted endpoints that do **not** advertise `scikitplot-chat-v1`
continue receiving their existing Anthropic/OpenAI-compatible client body.

This is deliberate:

```mermaid
flowchart LR
    E[Configured endpoint] --> H[GET origin/health]
    H -->|advertises scikitplot-chat-v1| S[Structured server-authority body]
    H -->|absent / unsupported| L[Legacy provider-compatible body]
```

A custom endpoint is outside the bundled server-secret trust boundary. It may
choose its own prompt policy, but the project must not describe its provider/model
identity as cryptographically verified merely because the UI configuration says
so.

## 5. Model authority

A direct caller may not choose arbitrary provider spend merely by changing the
`model` string.

HF proxy:

- exact provider models come from `ALLOWED_MODELS`;
- Path-2 project models may additionally be accepted by the explicitly configured
  `HF_SPACES_MODEL_NAMESPACES`;
- `stub/*` remains separately reserved and intercepted locally.

Cloudflare Worker:

- exact models come from `ALLOWED_MODELS` (default: `DEFAULT_MODEL`);
- optional namespaces come from `ALLOWED_MODEL_NAMESPACES`.

Local `dev_proxy.py`:

- exact models come from `ALLOWED_MODELS`.

A rejected model produces 4xx before a provider credential is spent.

## 6. Reasoning authority

The browser sends only provider-neutral intent to a negotiated bundled proxy:

```json
{"reasoning": {"effort": "high", "thinking": true, "budget_tokens": 1200}}
```

The HF server owns whether reasoning is enabled and the provider-native field
names/modes (`REASONING_EFFORT_PARAM`, `REASONING_THINKING_PARAM`, budget bounds).
A client cannot choose an arbitrary top-level provider field through the trusted
contract.

Custom endpoints retain their legacy provider-specific adapter behavior because
they are not using the bundled trusted contract.

## 7. Credential destination matrix

Run 4 removes the old `HF_TOKEN` reuse across three unrelated destinations.

| Route | Destination authority | Credential allowed |
|---|---|---|
| Path 1 | exact operator `BACKEND_URL` | `BACKEND_AUTH_TOKEN` only |
| Path 2 | exact operator `HF_SPACES_MODEL_URL` | `HF_SPACES_AUTH_TOKEN` only |
| Path 3 | official Hugging Face HTTPS origin | `HF_TOKEN` only |
| Cloudflare | constant `https://router.huggingface.co/v1/chat/completions` | Worker `HF_TOKEN` only |
| local dev proxy | validated official HF base | local `HF_TOKEN` only |

```mermaid
flowchart TD
    HFT[HF_TOKEN] --> HFG{Official Hugging Face HTTPS origin?}
    HFG -->|yes| HF[HF router]
    HFG -->|no| X[Startup failure]

    BT[BACKEND_AUTH_TOKEN] --> BU[Exact BACKEND_URL]
    ST[HF_SPACES_AUTH_TOKEN] --> SU[Exact HF_SPACES_MODEL_URL]

    HFT -. never .-> BU
    HFT -. never .-> SU
    BT -. never .-> HF
    ST -. never .-> HF
```

Credential-bearing destination validation rejects:

- URL userinfo;
- query strings;
- fragments;
- insecure HTTP except explicit localhost-only custom-backend development;
- `HF_TOKEN` destinations outside official Hugging Face HTTPS origins;
- non-standard HF token ports.

The dedicated custom/Space tokens are bound by configuration to their exact
operator-selected target instead of reusing the broader HF inference token.

## 8. Redirect rule

Credential-bearing requests do not automatically follow redirects.

- HF `httpx.AsyncClient(follow_redirects=False)`;
- local `httpx.post(..., follow_redirects=False)`;
- Worker `fetch(..., redirect: "manual")`.

```mermaid
sequenceDiagram
    participant P as Proxy
    participant A as Approved upstream
    participant E as Redirect target

    P->>A: Authorization + request
    A-->>P: 30x Location: E
    P--xE: credential is NOT auto-forwarded
```

A future redirect policy, if ever needed, must revalidate destination and
credential binding before issuing a second request; it must not rely on library
defaults.

## 9. Compatibility consequences

1. Direct clients of the bundled `/v1/chat/completions` route must migrate from
   OpenAI `messages` bodies to `scikitplot-chat-v1`.
2. Existing Sphinx browser clients negotiate automatically after `/health`
   advertises the contract.
3. Custom endpoints that do not advertise the contract keep legacy provider
   request shapes.
4. Site-authored `panelSystemPrompt` is **not** authoritative when the negotiated
   bundled proxy contract is active. The server owns system policy.
5. Operators using Path 1 authentication must set `BACKEND_AUTH_TOKEN`; `HF_TOKEN`
   is intentionally no longer reused.
6. Private/authenticated Path 2 deployments must set `HF_SPACES_AUTH_TOKEN`;
   `HF_TOKEN` is intentionally no longer reused.
7. Provider models beyond the default must be added to the server-side
   `ALLOWED_MODELS` policy.
8. Direct callers of `_hf_spaces_model/v1/chat/completions` must also use
   `scikitplot-chat-v1`; arbitrary OpenAI `messages` are intentionally rejected.
9. Path 2 preserves the structured contract across proxy → model-service rather
   than treating proxy-generated provider messages as trusted input.

These are security-breaking changes, not silent compatibility shims.

## 10. Verification gates

Run 4 adds:

### `tests/test_chat_authority.py`

- direct legacy `messages/system` body rejected;
- `system`, `developer`, tools, API key, URL and other authority-smuggling keys rejected;
- server policy is the only system role;
- hostile page text remains under user authority;
- server model allowlist enforced;
- provider-neutral reasoning mapped to server-owned wire fields;
- Path 1 receives only `BACKEND_AUTH_TOKEN`;
- Path 2 receives only `HF_SPACES_AUTH_TOKEN`;
- Path 3 receives only `HF_TOKEN`;
- unsafe HF credential destinations rejected;
- explicit localhost custom-backend development remains possible;
- bundled Docker image copies the chat contract module.

### `tests/test_chat_authority.mjs`

- browser negotiates `scikitplot-chat-v1` rather than guessing trust;
- discovery sends no browser credentials;
- structured body carries `user_message` + redacted context, not client system role;
- Worker advertises the contract;
- direct legacy/system request receives 400 and makes zero upstream calls;
- server-generated Worker system role excludes attacker page text;
- Worker model allowlist blocks provider spend;
- HF destination is fixed and redirect mode is manual.

### `tests/test_model_service_authority.py`

- proxy/model deployment contract modules must be byte-identical;
- direct model service rejects legacy OpenAI `messages/system`;
- hostile context/question remains under user authority;
- model app route calls the typed parser and server-owned builder;
- Path 2 preserves structured contract proxy → model service;
- model deployment documentation no longer advertises caller-authored messages.

### Mutation positives

- `chat-contract-negotiation-bypassed`;
- `chat-structured-user-message-replaced-by-system`;
- `chat-structured-context-skips-redaction`.

### Final working-tree gate summary

- client/Worker authority harness: **23/23**;
- Python proxy + direct-model authority: **27 passed**;
- executable Node harness integration: **31 passed**;
- mutation suite: **129 passed**;
- complete runnable non-Sphinx suite: **479 passed, 3 skipped**;
- full tree: **945 passed, 3 skipped, 5 failed, 62 errors**, with every
  failure/error confined to the missing-Sphinx `test___init__.py` fixture path;
- maintenance drift checker: **GREEN**;
- clean-extracted package copy: **GREEN** with the same 23/27/31/129/479+3 gates.

## 11. Closure statement

Run 4 closes the **bundled inference code-layer** prompt-authority and credential-
destination defects across the browser-negotiated HF proxy, Cloudflare Worker,
local dev proxy, **and the independently reachable model Space**:

- public callers cannot choose authoritative system/developer/tool roles;
- public callers cannot choose arbitrary model spend outside server policy;
- server provider requests are constructed from a typed envelope;
- HF inference credentials are no longer reused for custom backend/Space routes;
- redirect defaults cannot silently move credentials to a new origin;
- Path 2 preserves the typed contract and the model Space independently rebuilds
  system authority, so bypassing the relay does not restore client system roles.

It does **not** close:

- production CORS least-privilege parity (`AIA-005` / B05-B06);
- pre-buffer chat resource limits (`AIA-010`, `AIA-C11`) — subsequently closed by Run 11 / B27;
- centralized log/traceback minimization and infrastructure logging (Run 5);
- feedback/contribution provenance/retention/deletion (Run 6);
- local user-secret/sensitive-data send preflight (Run 7);
- cryptographic attestation of third-party custom endpoint/model identity;
- YAML/TOML/final Share information architecture (Run 8).
