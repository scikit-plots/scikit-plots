---
title: scikit-plots AI Model Endpoint
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.15.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - inference-api
license: bsd-3-clause
short_description: ai-model
---

# scikit-plots AI Model Endpoint

ZeroGPU Space that serves scikit-plots model weights via an
OpenAI-compatible REST endpoint.  Called by the proxy Space
(`scikit-plots/ai`) via its `BACKEND_URL` environment variable.

## Primary endpoint

```
POST /v1/chat/completions
```

Request body (OpenAI Chat Completions format):

```json
{
  "model": "scikit-plots/Qwen2.5-Coder-7B-Instruct",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 512
}
```

## Other endpoints

| Method | Path | Purpose | Link |
|---|---|---|---|
| `GET` | `/` | Status page | https://scikit-plots-ai-model.hf.space |
| `GET` | `/health` | Liveness probe (model loaded check) | https://scikit-plots-ai-model.hf.space/health |
| `GET` | `/ui` | Gradio test UI | https://scikit-plots-ai-model.hf.space/ui |
| `POST` | `/v1/chat/completions` | Primary inference endpoint | https://scikit-plots-ai-model.hf.space/v1/chat/completions |

## ⚠️ Cold start

The **first request** in a new ZeroGPU session triggers:

1. GPU allocation from the shared pool (1–5 minutes)
2. Model loading from storage to GPU VRAM (for models 14GB of RAM under 16GiB hard limit: 3–8 minutes)

**Total cold start: 2–10 minutes for a model.**

Set `PROXY_TIMEOUT=600` in the proxy Space (`scikit-plots/ai`) secrets.
Subsequent requests in the same active session complete in seconds.

## Configuration

Set in **Space → Settings → Repository secrets**:

| Variable | Required? | Default | Description |
|---|---|---|---|
| `MODEL_ID` | No | `scikit-plots/Qwen2.5-Coder-7B-Instruct` | Model weights to load. Supports `scikit-plots/*` mirrors. |
| `ALLOWED_ORIGINS` | No | `https://scikit-plots-ai.hf.space` | Comma-separated CORS origins. Add `http://localhost:7860` for local dev. Do not set to `*` in production. |
| `MAX_BODY_BYTES` | No | `10485760` | Maximum request body size (bytes). Non-integer values fall back to default. |

## Why this works with scikit-plots/* mirrors

This ZeroGPU Space downloads raw model weights directly from HuggingFace
storage (Git LFS), bypassing the HuggingFace Serverless Inference API.
Mirror repos (`scikit-plots/*`) have weights but no registered Inference
Provider — so they work here but fail with the HF Serverless API.

## Wire the proxy Space

Add these to `scikit-plots/ai` → Settings → Repository secrets:

```
BACKEND_URL   = https://scikit-plots-ai-model.hf.space/v1/chat/completions
PROXY_TIMEOUT = 600
```

## References

- [FREE_PROXY_SOLUTIONS.md Path C](./FREE_PROXY_SOLUTIONS.md#path-c--new-zerogpu-space-completely-free-gpu)
- [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu)
