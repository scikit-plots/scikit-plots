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
short_description: ai
---

# sphinx-ai-assistant proxy

Thin OpenAI-compatible reverse proxy for the **sphinx-ai-assistant** Sphinx
extension.  Runs as a free CPU Docker Space on HuggingFace.

## What it does

1. Accepts `POST /v1/chat/completions` from the browser (no auth header).
2. Resolves the upstream backend via env vars (see Configuration below).
3. Injects `Authorization: Bearer $HF_TOKEN` when required.
4. Forwards the request body verbatim to the model backend.
5. Returns the response (JSON or SSE stream) with CORS headers.

## Files committed to this Space

| File | Purpose |
|---|---|
| `app.py` | FastAPI proxy application |
| `_shared_logic.py` | Pure helpers imported by `app.py` (no pip deps) |
| `Dockerfile` | Container build — copies both Python files |
| `requirements.txt` | FastAPI, uvicorn, httpx |
| `README.md` | This file (HF Space metadata + documentation) |

> **Important** — `_shared_logic.py` **must** be committed alongside `app.py`.
> The Dockerfile copies it into the container.  If it is absent, the container
> will crash on startup with `ModuleNotFoundError: No module named 'shared_logic'`.

## Endpoints

| Method | Path | Purpose | Link |
|---|---|---|---|
| `GET`  | `/`                    | Status page / HF health-check | https://scikit-plots-ai.hf.space |
| `GET`  | `/health`              | Liveness probe | https://scikit-plots-ai.hf.space/health |
| `POST` | `/`                    | Backward-compat alias | https://scikit-plots-ai.hf.space |
| `POST` | `/v1/chat/completions` | Primary proxy endpoint | https://scikit-plots-ai.hf.space/v1/chat/completions |

## Configuration

Set in **Space → Settings → Repository secrets**:

| Variable | Required? | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (unless `BACKEND_URL` set) | — | HuggingFace API token. Requires "Make calls to Inference API" permission. |
| `BACKEND_URL` | No | `""` | Custom backend URL. Bypasses HF Serverless API. Use for DMR (Path A) or ZeroGPU Space (Path C). |
| `DEFAULT_MODEL` | No | `openai/gpt-oss-20b` | Fallback model when request body omits `model`. Must have Inference Provider if `BACKEND_URL` unset. |
| `HF_BASE` | No | `https://api-inference.huggingface.co/models` | HF API base URL. Only used when `BACKEND_URL` is empty. |
| `PROXY_TIMEOUT` | No | `120` | Upstream read timeout (seconds). Set `600` for ZeroGPU cold start. Non-integer values fall back to `120`. |
| `ALLOWED_ORIGINS` | No | `*` | Comma-separated CORS origins. Production: `https://scikit-plots.github.io` |
| `MAX_BODY_BYTES` | No | `10485760` | Maximum request body size (bytes). Non-integer values fall back to `10485760`. |

## Quick path reference

```
Path B — HF Serverless API (original provider models):
  HF_TOKEN      = hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
  DEFAULT_MODEL = openai/gpt-oss-20b
  PROXY_TIMEOUT = 120

Path C — ZeroGPU Space (any model weights, free GPU):
  BACKEND_URL   = https://scikit-plots-ai-model.hf.space/v1/chat/completions
  PROXY_TIMEOUT = 600   ← cold start for 20B model takes 2–10 minutes

Path A — Local Docker Model Runner (set in your shell, not Space secrets):
  export BACKEND_URL=http://localhost:12434/engines/llama.cpp/v1/chat/completions
```

## Verify the deployment

```bash
# Liveness check
curl https://scikit-plots-ai.hf.space/health
# Expected: {"status":"ok","version":"3.1.0"}

# Status page (shows active backend)
curl https://scikit-plots-ai.hf.space/
# Expected: {"status":"ok","service":"sphinx-ai-assistant proxy v3.1.0",...}

# Test a real completion (Path B — replace with your Space URL)
curl https://scikit-plots-ai.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"hi"}]}'
```

## Why mirror repos fail with HF Serverless API

`scikit-plots/gpt-oss-20b` and `scikit-plots/Qwen2.5-Coder-32B-Instruct` are
**mirror repositories** — they contain weights copied from the original repos
but are **not registered with any HF Inference Provider**.

- `POST` to `api-inference.huggingface.co/models/scikit-plots/gpt-oss-20b/...` → **404 / 503**
- `POST` to `api-inference.huggingface.co/models/openai/gpt-oss-20b/...` → **✓ works**

Use `DEFAULT_MODEL = openai/gpt-oss-20b` for Path B.
Use Path C (ZeroGPU) to run `scikit-plots/*` weights directly.

## References

- [FREE_PROXY_SOLUTIONS.md](./FREE_PROXY_SOLUTIONS.md) — Full path decision tree
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference/)
- [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu)
