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

Thin OpenAI-compatible reverse proxy for the **sphinx-ai-assistant** Sphinx
extension.  Runs as a free CPU Docker Space on HuggingFace.  Accepts
unauthenticated requests from the browser widget, resolves the upstream model
backend, injects the required auth header server-side, and returns the
response — keeping all tokens out of the browser.

---

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
                   HF_TOKEN injected    no auth needed    HF_TOKEN injected
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
| `app.py` | FastAPI proxy application — all route handlers |
| `_shared_logic.py` | Pure Python helpers imported by `app.py` (stdlib only, no pip deps) |
| `Dockerfile` | Container build — copies both Python files into the image |
| `requirements.txt` | `fastapi`, `uvicorn`, `httpx`, `huggingface_hub` |
| `README.md` | This file — HF Space metadata + full documentation |

> **Critical** — `_shared_logic.py` **must** be committed alongside `app.py`.
> The Dockerfile `COPY`s it explicitly.  If it is absent the container crashes
> at startup with `ModuleNotFoundError: No module named '_shared_logic'`.

---

## Endpoints

| Method | Path | Purpose | Notes |
|---|---|---|---|
| `GET`  | `/`                    | Status page — routing config, token states, `contribute_ready` flag | Always 200 while running |
| `GET`  | `/health`              | Minimal liveness probe for container orchestrators | Always `{"status":"ok"}` |
| `HEAD` | `/`                    | Health-monitor probe (no body) | Required by HF uptime monitor |
| `HEAD` | `/health`              | Health-monitor probe (no body) | Required by HF uptime monitor |
| `POST` | `/`                    | Backward-compat alias for `/v1/chat/completions` | Identical behaviour |
| `POST` | `/v1/chat/completions` | Primary proxy — routes to Path 1 / 2 / 3 | SSE streaming preserved |
| `POST` | `/v1/feedback`         | Receive 👍/👎 rating from the widget; logged, not persisted | Rate-limited: 30/IP/hour |
| `POST` | `/v1/share`            | Store a conversation snapshot; returns a shareable link | In-memory; clears on restart |
| `GET`  | `/v1/share/{uuid}`     | Retrieve a stored snapshot by UUID | — |
| `POST` | `/v1/contribute`       | GDPR-gated — push rated Q&A pairs to `TRAINING_DATASET_REPO` | Rate-limited: 5/IP/hour |

### `POST /v1/contribute` — payload schema

```json
{
  "schemaVersion":  1,
  "consentFlag":    true,
  "consentVersion": "v1.0",
  "sessionId":      "uuid-string",
  "page":           "https://your-docs-site/index.html",
  "model":          { "id": "...", "provider": "...", "model": "..." },
  "records": [
    {
      "answerIndex":  0,
      "query":        "What is a confusion matrix?",
      "answer":       "A confusion matrix is...",
      "ratingValue":  2,
      "ratingLabel":  "👍",
      "message":      "optional free-text from user",
      "ts":           1781002584724
    }
  ]
}
```

Each record is written to `contributions/<timestamp_ms>.jsonl` in
`TRAINING_DATASET_REPO`.  Only answers the user has explicitly rated are
included — unrated answers are skipped by the browser widget before the
request is sent.

---

## Configuration

All variables are set in **Space → Settings → Repository secrets**.

### Tokens

| Variable | Required? | Scope | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (for Path 3) | Read / inference-api | Injected as `Bearer` into Path 1 and Path 3 upstream requests.  Use a **read-only** fine-grained token — it never needs write permission. |
| `HF_TOKEN_TYPE` | No | — | Declares the type of `HF_TOKEN`.  Accepted: `fine-grained` \| `read` \| `write`.  When set, startup validation enforces least-privilege without network calls.  When absent, a length-based heuristic is used and type shows as `"unknown"` on the status page.  Setting `"write"` triggers a startup `WARNING`. |
| `HF_WRITE_TOKEN` | No | Write (dataset repo only) | Dedicated write token used **only** by `POST /v1/contribute` to push JSONL files to `TRAINING_DATASET_REPO`.  When set, `HF_TOKEN` remains strictly read-only.  When absent, the proxy falls back to `HF_TOKEN` for writes (single-token mode, backward compatible but less secure). |
| `HF_WRITE_TOKEN_TYPE` | No | — | Declares the type of `HF_WRITE_TOKEN`.  Accepted: `fine-grained` \| `write`.  Enables startup validation to detect a read token used for writes (which would cause every `/v1/contribute` call to fail with 403). |

> **Why two tokens?** Principle of least privilege.  `HF_TOKEN` is forwarded
> to external model backends on every inference request — if it carried write
> permission a compromised upstream could modify your repos.  `HF_WRITE_TOKEN`
> never leaves the proxy and is scoped to one dataset repo.

The `/` status page shows all token states without exposing values:

```json
"tokens": {
  "hf_token_set":         true,
  "hf_token_type":        "fine-grained",
  "hf_write_token_set":   true,
  "hf_write_token_type":  "fine-grained",
  "least_privilege_mode": true
},
"training": {
  "dataset_repo":    "scikit-plots/ai-assistant-contributions",
  "contribute_ready": true
}
```

`hf_token_type: "unknown"` means `HF_TOKEN_TYPE` is not set and the length
heuristic could not determine the type — set `HF_TOKEN_TYPE=fine-grained` or
`HF_TOKEN_TYPE=read` in Space secrets to resolve.

`least_privilege_mode: false` means `HF_WRITE_TOKEN` is absent and writes fall
back to `HF_TOKEN` — functional but not recommended for production.

### Routing

| Variable | Required? | Default | Description |
|---|---|---|---|
| `BACKEND_URL` | No | `""` | **Path 1** — Explicit upstream URL.  All requests forwarded here when set.  Use for Docker Model Runner, Ollama, or any custom backend. |
| `HF_SPACES_MODEL_URL` | No | — | **Path 2** — Custom ZeroGPU Space URL (e.g. `https://scikit-plots-ai-model.hf.space/v1/chat/completions`). Receives requests whose `model` matches `HF_SPACES_MODEL_NAMESPACES`. |
| `HF_SPACES_MODEL_NAMESPACES` | No | `scikit-plots` | Comma-separated model owner prefixes routed to Path 2 (e.g. `scikit-plots,my-org`). |
| `HF_BASE` | No | `https://router.huggingface.co` | **Path 3** — HF Serverless API base URL.  Only used when Path 1 and Path 2 do not match. |
| `DEFAULT_MODEL` | No | `scikit-plots/Qwen2.5-Coder-7B-Instruct` | Fallback model when the request body omits `"model"`. |

### Training data collection

| Variable | Required? | Default | Description |
|---|---|---|---|
| `TRAINING_DATASET_REPO` | No | `""` | HF dataset repo ID where `/v1/contribute` writes JSONL files (e.g. `scikit-plots/ai-assistant-contributions`). Must exist before the proxy starts. See **Training data setup** below. |

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

### Other

| Variable | Default | Description |
|---|---|---|
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins.  Production: `https://scikit-plots.github.io`. |
| `MAX_BODY_BYTES` | `10485760` | Maximum accepted request body size (bytes). |

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

#### Write token → `HF_WRITE_TOKEN`

```
Type:        Fine-grained
Name:        sphinx-ai-proxy-write  (any descriptive name)
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
read  role  →  HF_TOKEN       (inference only; set in Space secrets)
write role  →  HF_WRITE_TOKEN (dataset push;   set in Space secrets)
```

### Declare token types in Space secrets

After creating the tokens, set the corresponding type variables so the proxy
can validate least-privilege at startup without network calls:

```
# Fine-grained tokens (recommended):
HF_TOKEN_TYPE       = fine-grained
HF_WRITE_TOKEN_TYPE = fine-grained

# Classic read/write tokens (legacy):
HF_TOKEN_TYPE       = read
HF_WRITE_TOKEN_TYPE = write
```

If you omit these variables the proxy applies a length-based heuristic
(tokens ≥ 52 chars → `fine-grained`; shorter → `unknown`).  The startup log
and status page will show `"unknown"` for the type, which suppresses
least-privilege warnings — always set explicit types in production.

---

## Training data setup

### 1. Create the dataset repository

```
https://huggingface.co/new-dataset
  Owner:      scikit-plots
  Name:       ai-assistant-contributions
  Visibility: Private  ← training data should never be public
  License:    BSD-3-Clause (or your choice)
```

The proxy writes files to `contributions/<timestamp_ms>.jsonl` inside this
repo.  Each file contains one JSON object per line (one per rated answer):

```jsonl
{"answerIndex":0,"query":"...","answer":"...","ratingValue":2,"ratingLabel":"👍","message":"","ts":1781002584724,"_sessionId":"...","_page":"https://...","_model":{...},"_consentVersion":"v1.0","_ts":1781003000000}
{"answerIndex":2,"query":"...","answer":"...","ratingValue":-2,"ratingLabel":"👎","message":"wrong output","ts":...}
```

### 2. Set Space secrets

In **Space → Settings → Repository secrets**, set all three:

```
TRAINING_DATASET_REPO = scikit-plots/ai-assistant-contributions
HF_TOKEN              = hf_<your-read-token>
HF_WRITE_TOKEN        = hf_<your-write-token>
```

### 3. Verify the endpoint is ready

```bash
curl https://scikit-plots-ai.hf.space/ | python3 -m json.tool | grep -A5 '"training"'
# Expected:
# "training": {
#   "dataset_repo":    "scikit-plots/ai-assistant-contributions",
#   "contribute_ready": true
# }
```

`contribute_ready: false` means either `TRAINING_DATASET_REPO` or the write
token is missing — check the Space logs for the startup warning.

### 4. Load the data with pandas

```python
import json
from pathlib import Path
import pandas as pd

# After cloning or downloading the dataset repo:
records = []
for f in Path("contributions").glob("*.jsonl"):
    records.extend(json.loads(line) for line in f.read_text().splitlines() if line)

df = pd.DataFrame(records)
df["ts"] = pd.to_datetime(df["ts"], unit="ms")
print(df[["ts", "query", "answer", "ratingValue", "ratingLabel"]].head())
```

---

## Quick deployment recipes

### Path 3 — HF Serverless API (simplest, standard provider models)

Use this for models registered with a HuggingFace Inference Provider
(Qwen/\*, mistralai/\*, etc.).

```
# Space → Settings → Repository secrets:
HF_TOKEN      = hf_<your-read-token>
DEFAULT_MODEL = Qwen/Qwen2.5-Coder-32B-Instruct
PATH3_TIMEOUT = 120
```

### Path 2 — Custom ZeroGPU Space (mirror repos, free GPU)

Use this for `scikit-plots/*` mirror repos that are not registered with any
Inference Provider.  See [Why mirror repos fail](#why-mirror-repos-fail)
below.

```
# Space → Settings → Repository secrets:
HF_SPACES_MODEL_URL        = https://scikit-plots-ai-model.hf.space/v1/chat/completions
HF_SPACES_MODEL_NAMESPACES = scikit-plots
PATH2_TIMEOUT              = 600   ← CPU cold start takes 4–5 minutes
DEFAULT_MODEL              = scikit-plots/Qwen2.5-Coder-7B-Instruct
```

### Path 1 — Local / custom backend (Docker Model Runner, Ollama)

Set in your shell or CI environment — do **not** put a localhost URL in Space
secrets (the Space container cannot reach your local machine).

```bash
export BACKEND_URL=http://localhost:12434/engines/llama.cpp/v1/chat/completions
export HF_TOKEN=hf_<optional-for-path-1>
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

# Training data collection
HF_WRITE_TOKEN             = hf_<write-token>
HF_WRITE_TOKEN_TYPE        = fine-grained
TRAINING_DATASET_REPO      = scikit-plots/ai-assistant-contributions

# Security
ALLOWED_ORIGINS            = https://scikit-plots.github.io
```

---

## Verify the deployment

```bash
BASE=https://scikit-plots-ai.hf.space

# 1. Liveness probe
curl $BASE/health
# {"status":"ok","version":"6.1.0"}

# 2. Full status — check routing, token slots, and training readiness
curl $BASE/ | python3 -m json.tool
# Look for:
#   "tokens":   { "hf_token_set": true, "hf_write_token_set": true, "least_privilege_mode": true }
#   "training": { "contribute_ready": true }

# 3. Test a chat completion (Path 3 — HF Serverless)
curl $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-Coder-32B-Instruct","messages":[{"role":"user","content":"hi"}]}'

# 4. Test a chat completion (Path 2 — custom Space)
curl $BASE/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"scikit-plots/Qwen2.5-Coder-7B-Instruct","messages":[{"role":"user","content":"hi"}]}'

# 5. Test training contribution (requires TRAINING_DATASET_REPO + write token)
curl $BASE/v1/contribute \
  -H "Content-Type: application/json" \
  -d '{
    "schemaVersion":  1,
    "consentFlag":    true,
    "consentVersion": "v1.0",
    "sessionId":      "test-session-001",
    "page":           "https://example.com",
    "model":          null,
    "records": [{
      "answerIndex":  0,
      "query":        "test query",
      "answer":       "test answer",
      "ratingValue":  2,
      "ratingLabel":  "👍",
      "message":      "",
      "ts":           1781002584724
    }]
  }'
# {"contributed": true, "rows": 1}
```

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
| `POST /v1/contribute` → 503 `"TRAINING_DATASET_REPO not set"` | Env var missing | Set `TRAINING_DATASET_REPO` in Space secrets and restart |
| `POST /v1/contribute` → 503 `"No write token configured"` | Both `HF_WRITE_TOKEN` and `HF_TOKEN` are empty | Set at least one write-capable token |
| `POST /v1/contribute` → 503 `"Failed to store contribution"` | `HfApi.create_commit` failed — wrong token scope or repo does not exist | Verify the write token has write access to the dataset repo; verify the repo exists |
| `POST /v1/contribute` → 422 `"consentVersion … is not current"` | Browser widget cached an old page with an outdated consent version string | Hard-refresh the docs page (`Ctrl+Shift+R`) |
| `POST /v1/contribute` → 429 | Rate limit exceeded (5 contributions per IP per hour) | Wait 1 hour or test from a different IP |
| `POST /v1/chat/completions` → 401 | `HF_TOKEN` not set or expired | Regenerate token at `huggingface.co/settings/tokens` |
| `POST /v1/chat/completions` → 503 / 404 for `scikit-plots/*` | Path-2 not configured | Set `HF_SPACES_MODEL_URL` and `HF_SPACES_MODEL_NAMESPACES=scikit-plots` |
| Space status page shows `contribute_ready: false` | `TRAINING_DATASET_REPO` or write token missing | Check both; see Space logs for the startup warning identifying which is missing |
| `least_privilege_mode: false` on status page | `HF_WRITE_TOKEN` not set | Set `HF_WRITE_TOKEN`; proxy works but uses `HF_TOKEN` for writes (less secure) |
| Container crashes: `ModuleNotFoundError: No module named '_shared_logic'` | `_shared_logic.py` missing from the Space repo | Commit `_shared_logic.py` alongside `app.py` |
| 413 on large conversation export | Body exceeds `MAX_BODY_BYTES` (default 10 MB) | Increase `MAX_BODY_BYTES` in Space secrets |
| Startup log: `WARNING: Startup token-config check: HF_TOKEN has type 'write' …` | A write token is being used for inference — violates least-privilege | Replace `HF_TOKEN` with a read or fine-grained token scoped to Inference only; set `HF_TOKEN_TYPE=read` or `HF_TOKEN_TYPE=fine-grained` in Space secrets |
| Startup log: `WARNING: Startup token-config check: HF_WRITE_TOKEN has type 'read' …` | A read-only token is configured as the write token | Replace `HF_WRITE_TOKEN` with a write or fine-grained token that has dataset write scope; set `HF_WRITE_TOKEN_TYPE=write` or `HF_WRITE_TOKEN_TYPE=fine-grained` |
| Status page shows `hf_token_type: "unknown"` | `HF_TOKEN_TYPE` not set; length-based heuristic could not determine token type | Set `HF_TOKEN_TYPE=fine-grained` or `HF_TOKEN_TYPE=read` in Space secrets to enable least-privilege startup checks |
| Status page shows `hf_token_type: "write"` with a startup WARNING | `HF_TOKEN` is a write token — too many permissions for inference | Create a separate read or fine-grained token for `HF_TOKEN` and set `HF_TOKEN_TYPE=read`; keep the write token only in `HF_WRITE_TOKEN` |

---

## References

- [FREE_PROXY_SOLUTIONS.md](./FREE_PROXY_SOLUTIONS.md) — Full routing path decision tree
- [HuggingFace fine-grained tokens](https://huggingface.co/docs/hub/security-tokens)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference/)
- [HuggingFace Datasets — `huggingface_hub`](https://huggingface.co/docs/huggingface_hub/guides/repository)
- [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu)
