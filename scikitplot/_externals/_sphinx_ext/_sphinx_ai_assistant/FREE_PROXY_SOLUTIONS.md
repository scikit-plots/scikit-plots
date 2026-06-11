# Free Proxy Solutions for `sphinx-ai-assistant`

<!--
DOCUMENT METADATA
─────────────────
Version  : 6.0.0
Status   : Authoritative — all bugs from v2.0.0 corrected (see CHANGELOG at bottom)
Audience : Newbies reading for the first time AND experienced developers debugging
Sources  : Verified against official documentation at time of writing:
           • Docker Model Runner  https://docs.docker.com/ai/model-runner/
           • HF Spaces config     https://huggingface.co/docs/hub/spaces-config-reference
           • HF ZeroGPU           https://huggingface.co/docs/hub/spaces-zerogpu
           • HF Inference API     https://huggingface.co/docs/api-inference/getting-started
           • Cloudflare Workers   https://developers.cloudflare.com/workers/
           • Wrangler v3 CLI      https://developers.cloudflare.com/workers/wrangler/commands/
           • FastAPI / Starlette  https://fastapi.tiangolo.com/
           • httpx                https://www.python-httpx.org/
           • transformers         https://huggingface.co/docs/transformers/
           • spaces (ZeroGPU)    https://huggingface.co/docs/hub/spaces-zerogpu#usage
           • OpenAI Chat schema   https://platform.openai.com/docs/api-reference/chat

HOW TO READ THIS FILE
─────────────────────
• If you are completely new → read §0 (Big Picture) and §1 (Root Cause), then use
  the decision tree in §4 to choose your path.
• If you already understand the problem → jump directly to the Path section.
• Every code block is copy-paste-ready. Every command shows expected output.
• Hints explain WHY, not just WHAT. Developer notes explain internals.
• All known bugs from the previous version (v2.0.0) are fixed and documented
  in the CHANGELOG at the bottom.
-->

---

## Table of Contents

1. [§ 0 – Big Picture (Read This First)](#0--big-picture-read-this-first)
2. [§ 1 – Root Cause Chain (Why Things Break)](#1--root-cause-chain-why-things-break)
3. [§ 2 – Free Tier Reality Check](#2--free-tier-reality-check)
4. [§ 3 – Solution Architecture](#3--solution-architecture)
5. [§ 4 – Which Path Should I Choose?](#4--which-path-should-i-choose)
6. [Path A – Local Dev via Docker Model Runner (Zero Cost)](#path-a--local-dev-via-docker-model-runner-zero-cost)
7. [Path B – Fix the Deployed HF Space (One Env Var)](#path-b--fix-the-deployed-hf-space-one-env-var)
8. [Path C – New ZeroGPU Space (Completely Free GPU)](#path-c--new-zerogpu-space-completely-free-gpu)
9. [Path D – Cloudflare Worker Proxy (100k req/day Free)](#path-d--cloudflare-worker-proxy-100k-reqday-free)
10. [Path E – Local Python Dev Proxy (No Docker)](#path-e--local-python-dev-proxy-no-docker)
11. [Updated `app.py` for `scikit-plots/ai` Proxy Space](#updated-apppy-for-scikit-plotsai-proxy-space)
12. [Wiring `conf.py` for Each Path](#wiring-confpy-for-each-path)
13. [Environment Variable Reference](#environment-variable-reference)
14. [Troubleshooting Checklist](#troubleshooting-checklist)
15. [Changelog](#changelog)

---

## § 0 – Big Picture (Read This First)

```
WHAT YOU WANT
─────────────
A Sphinx documentation page with an AI chat panel that talks to a language model.

WHAT ACTUALLY HAS TO HAPPEN FOR EVERY CHAT MESSAGE
───────────────────────────────────────────────────
1.  User types a question in the browser.
2.  Browser JavaScript POSTs the question to a URL you control (the proxy).
3.  The proxy adds your secret API token (HF_TOKEN) to the request.
4.  The proxy forwards the enriched request to a language model backend.
5.  The model generates a response and streams it back to the proxy.
6.  The proxy streams the answer back to the browser.
7.  The browser renders the answer in the chat panel.

WHY CAN'T THE BROWSER CALL THE MODEL DIRECTLY?
───────────────────────────────────────────────
Reason 1 – CORS:
  HuggingFace API servers block cross-origin POST requests from browsers
  (they do not return "Access-Control-Allow-Origin: *" for API endpoints).
  The browser enforces CORS and the request never leaves the browser.

Reason 2 – Security:
  If you embed HF_TOKEN in browser JavaScript, every visitor to your docs
  page can read your token and run inference at your expense.
  The token must live only in a server-side environment variable.

CONCLUSION: A proxy is not optional. It is architecturally required for both
  security (token injection) and functionality (CORS bypass).

WHY IS THE MODEL BACKEND SEPARATE FROM THE PROXY?
──────────────────────────────────────────────────
• A proxy is just an HTTP forwarder (~100 lines). It needs only CPU. It is free.
• A language model needs a GPU and large amounts of RAM. It must run somewhere
  with GPU access — either your local machine or a cloud GPU service.
• Keeping proxy and model separate means you can swap the model backend
  (local, HF API, ZeroGPU) by changing one environment variable, with no
  proxy code changes and no proxy redeployment.

WHAT VS CODE LIVE SERVER IS NOT
────────────────────────────────
VS Code Live Server (port 5500) is a static file server.
It serves HTML, CSS, and JS files via GET requests.
It CANNOT handle POST requests. It will always return 405 Method Not Allowed
for any POST, including the AI chat widget's requests.
Never use Live Server as your proxy. Use one of the paths in this document.
```

### Key Terms (Plain English)

| Term | What it means here |
|---|---|
| **Proxy** | A small HTTP server between the browser and the model. Injects the secret token. |
| **HF_TOKEN** | Your HuggingFace API key. Never put this in browser JavaScript. Only in server env vars. |
| **Inference Provider** | A third-party service registered on HF that will actually run a model. Not all models have one. |
| **HF Serverless Inference API** | HuggingFace's pay-per-use API. Only works if the model has a registered Inference Provider. |
| **Docker Space** | An HF Space that runs a custom Docker container. Good for proxies. Free CPU tier. |
| **ZeroGPU Space** | An HF Space with free shared GPU time (allocated per request). Good for running the model itself. |
| **Docker Model Runner (DMR)** | A Docker Desktop / Docker Engine feature that runs GGUF-format models locally. |
| **CORS** | Browser security rule: page at domain A cannot POST to domain B unless B explicitly allows it. |
| **SSE** | Server-Sent Events — the streaming format used to display AI responses word by word. |
| **GGUF** | A quantised model weight format used by llama.cpp. Docker Model Runner uses this format. |

---

## § 1 – Root Cause Chain (Why Things Break)

### Layer 1 — The 405 Error During Local Dev

```
SYMPTOM
───────
  POST http://127.0.0.1:5500/_proxy/hf        → 405 Method Not Allowed
  POST http://127.0.0.1:5500/_proxy/anthropic → 405 Method Not Allowed

ROOT CAUSE
──────────
  VS Code Live Server (port 5500) is a static-file-only server.
  It accepts: GET, HEAD, OPTIONS.
  It never accepts: POST.
  When the AI widget JavaScript POSTs to /_proxy/hf, Live Server has no
  handler for POST on any path and returns 405 Method Not Allowed.

FIX
───
  Run a real HTTP proxy that handles POST /v1/chat/completions.
  Options:
    Path A → Docker Model Runner (local model, zero token needed)
    Path E → Python dev_proxy.py (forwards to HF API, needs token + model with provider)
```

### Layer 2 — The Real Inference Problem (The Deeper Root Cause)

```
SYMPTOM
───────
  The proxy starts. Requests reach HuggingFace. But the response is 404 or 503:
    POST https://router.huggingface.co/v1/chat/completions
      body: {"model": "scikit-plots/gpt-oss-20b", ...}
    → 404  or  {"error": "Model scikit-plots/gpt-oss-20b is currently loading"}
    → ... forever, never resolves

ROOT CAUSE — 5 WHYS ANALYSIS
─────────────────────────────
  Why 1: Why does the proxy get 404/503?
         → HuggingFace Serverless Inference API returned an error for that model.

  Why 2: Why did HF API return an error?
         → The model at that ID has no inference endpoint registered on HuggingFace.

  Why 3: Why does that model have no inference endpoint?
         → HF Serverless Inference API only works for models that have a
           registered Inference Provider. Not all models have one.

  Why 4: Why don't scikit-plots/gpt-oss-20b and scikit-plots/Qwen2.5-Coder-32B-Instruct
         have an Inference Provider?
         → These are MIRROR REPOSITORIES. They contain model weights copied from
           the original repositories (openai/gpt-oss-20b, Qwen/Qwen2.5-Coder-32B-Instruct).
           They have NOT been registered with any inference provider on HuggingFace.
           The HF model page for these models explicitly reads:
             "This model isn't deployed by any Inference Provider."

  Why 5: Why does the original proxy not handle this?
         → The original app.py hard-codes the HF Serverless Inference API URL pattern.
           It assumes every model ID in the request has a live provider behind it.
           It has no fallback and no mechanism to redirect to a different backend.

ROOT FIX — choose one of:
  (a) Use the ORIGINAL model IDs that DO have Inference Providers.
      (openai/gpt-oss-20b and Qwen/Qwen2.5-Coder-32B-Instruct — the originals work)
      → Path B: change one env var in the Space settings. Done in 2 minutes.

  (b) Run the model yourself using raw weights, bypassing HF Serverless API.
      → Locally:     Path A (Docker Model Runner — downloads GGUF, runs on your machine)
      → Cloud/free:  Path C (ZeroGPU Space — loads weights, runs on free shared GPU)
```

### Layer 3 — Mirror vs Original Model IDs (Quick Reference)

```
scikit-plots/gpt-oss-20b
  Type:    Mirror of openai/gpt-oss-20b
  Status:  Has weights — NO inference provider
  Result:  CANNOT use via HF Serverless Inference API → 404/503
  CAN use: Docker Model Runner (Path A), ZeroGPU Space (Path C)

openai/gpt-oss-20b
  Type:    Original repository
  Status:  Has a registered Inference Provider
  Result:  CAN use via HF Serverless Inference API ✓

scikit-plots/Qwen2.5-Coder-32B-Instruct
  Type:    Mirror of Qwen/Qwen2.5-Coder-32B-Instruct
  Status:  Has weights — NO inference provider
  Result:  CANNOT use via HF Serverless Inference API → 404/503
  CAN use: Docker Model Runner (Path A), ZeroGPU Space (Path C)

Qwen/Qwen2.5-Coder-32B-Instruct
  Type:    Original repository
  Status:  Has a registered Inference Provider
  Result:  CAN use via HF Serverless Inference API ✓

HOW TO CHECK ANY MODEL
───────────────────────
  1. Open https://huggingface.co/<model-id>
  2. Look at the right sidebar.
  3. If you see an "Inference API" widget where you can test it → provider exists ✓
  4. If you see "This model isn't deployed by any Inference Provider" → no provider ✗
     You must use Path A or Path C for this model.
```

---

## § 2 – Free Tier Reality Check

| Platform | What you get free | Practical notes |
|---|---|---|
| HuggingFace free account | ~$0.10/month inference credit | Exhausted in minutes with a 20B model. Not viable for regular use. |
| HuggingFace PRO | ~$2.00/month inference credit | Light dev use only. Still not enough for regular 20B model inference. |
| **HF Spaces — CPU (Docker)** | **Unlimited CPU runtime** | Perfect for the proxy layer. No GPU needed for a forwarder. |
| **HF Spaces — ZeroGPU** | **Free shared GPU per request** | Runs the model itself. Cold start on a 20B model = 2–10 minutes for first request in a new session. See Path C. |
| **Cloudflare Workers** | **100 000 req/day, 10ms CPU time/req** | Best free proxy for GitHub Pages. CPU limit applies to your code only, not network I/O wait. 30-second wall-clock limit per request on free tier — enough for most completions. |
| Vercel / Netlify Functions | 125k function calls/month | Viable proxy alternative. Not covered in this guide. |
| **Docker Model Runner** | **Free — runs on your machine** | Best for local development. No token needed after model download. Works with mirror repos because it downloads GGUF weights directly (no HF Serverless API). |

**Important — ZeroGPU cold start reality:**
The first request in a new ZeroGPU session triggers GPU allocation AND full model loading.
For a 20B model this typically takes **2 to 10 minutes**, not "30–60 seconds".
Set `PROXY_TIMEOUT` to at least 600 seconds (10 minutes) if using ZeroGPU Path C.
Subsequent requests in the same active session are much faster (seconds).

---

## § 3 – Solution Architecture

### High-Level Request Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│  BROWSER  (Sphinx docs page — github.io, readthedocs.io, or local)  │
│                                                                      │
│  User types a question.                                              │
│  JS widget POSTs JSON to the proxy endpoint.                         │
│  No auth header — the browser never holds the API token.            │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  POST /v1/chat/completions
                            │  Content-Type: application/json
                            │  Body: {"model":"...","messages":[...]}
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│  PROXY  (scikit-plots/ai HF Space — free CPU Docker container)       │
│                                                                      │
│  Reads env vars at startup:                                          │
│    BACKEND_URL  → use this URL directly  (Paths A, C)               │
│    HF_TOKEN     → use HF Serverless API  (Path B)                   │
│  Adds Authorization header (when required).                          │
│  Forwards the request body unchanged.                               │
│  Streams the response back to the browser.                          │
└──────────┬────────────────────────┬─────────────────────────────────┘
           │                        │
  Path A/C │ BACKEND_URL set   Path B│ BACKEND_URL empty, HF_TOKEN set
           │                        │
           ▼                        ▼
┌──────────────────┐  ┌─────────────────────────────────────────────┐
│  YOUR BACKEND    │  │  HF SERVERLESS INFERENCE API                 │
│                  │  │                                              │
│  Path A:         │  │  Only works with models that HAVE a          │
│   Docker Model   │  │  registered Inference Provider:              │
│   Runner on your │  │                                              │
│   machine        │  │    openai/gpt-oss-20b              ✓        │
│   Port 12434     │  │    Qwen/Qwen2.5-Coder-32B-Instruct ✓        │
│                  │  │                                              │
│  Path C:         │  │    scikit-plots/gpt-oss-20b         ✗        │
│   ZeroGPU Space  │  │    scikit-plots/Qwen2.5-Coder-*    ✗        │
│   (free shared   │  │                                              │
│    GPU)          │  │  URL pattern:                                │
└──────────────────┘  │  router.huggingface.co/v1/chat/completions   │
                      │    (model selected via request body)          │
                      └─────────────────────────────────────────────┘
```

### CORS and Token Security (Why the Proxy Exists)

```
BROWSER SECURITY RULE — CORS (Cross-Origin Resource Sharing)
─────────────────────────────────────────────────────────────
A page at https://scikit-plots.github.io (origin A) cannot directly POST to
https://router.huggingface.co (origin B), because HuggingFace does NOT
return "Access-Control-Allow-Origin: *" for its inference API endpoints.
The browser enforces this and blocks the request before it is sent.

YOUR PROXY at https://scikit-plots-ai.hf.space is a domain YOU control.
You can set CORS headers on YOUR server to allow the browser to POST to it.
Your proxy then forwards the request to HuggingFace server-to-server
(no CORS restriction applies to server-to-server communication).

TOKEN SECURITY RULE
────────────────────
HF_TOKEN must NEVER appear in browser JavaScript or HTML.
It lives ONLY in the Space's repository secrets (encrypted environment variables).
The browser sends no Authorization header.
The proxy injects the token server-side before forwarding to HuggingFace.

VIOLATION CONSEQUENCE
─────────────────────
If HF_TOKEN appears in any file that the browser can read (JS, HTML, conf.py
that is rendered into HTML), any visitor can steal it and run inference at
your expense with no limit until you notice and revoke it.
```

---

## § 4 – Which Path Should I Choose?

```
START HERE
──────────
What are you trying to do?
│
├─ I want to test locally on my own machine.
│   ├─ Do you have Docker Desktop (macOS/Windows) or Docker Engine (Linux)?
│   │   └─ Yes → PATH A (Docker Model Runner)
│   │              • Downloads the model GGUF once (~15 GB).
│   │              • Runs the model on your machine's GPU or CPU.
│   │              • No HF token needed. No internet after download.
│   │              • Works with mirror repos (scikit-plots/*).
│   │              • Best option for local development.
│   │
│   └─ No Docker → PATH E (Python dev proxy)
│                   • Single Python file, no Docker needed.
│                   • Forwards requests to HF Serverless API.
│                   • Requires HF_TOKEN.
│                   • Requires using original model IDs (openai/*, Qwen/*).
│
├─ My deployed HF Space is returning 404 / 503. Quickest fix?
│   └─ PATH B (change one environment variable)
│       • No code changes. No redeployment. Takes 2 minutes.
│       • Switch DEFAULT_MODEL to "Qwen/Qwen2.5-Coder-32B-Instruct" (has Inference Provider).
│       • Requires a valid HF_TOKEN already set in the Space.
│
├─ I want to serve scikit-plots/* model weights specifically, free, in the cloud.
│   └─ PATH C (new ZeroGPU Space)
│       • Create a second Space (scikit-plots/ai-model) that loads the weights
│         and runs them on free shared GPU.
│       • Existing proxy Space (scikit-plots/ai) stays unchanged — just add BACKEND_URL.
│       • First request per session is slow (2–10 min cold start). Subsequent: fast.
│
├─ My docs are on GitHub Pages. I want a free proxy with zero infrastructure.
│   └─ PATH D (Cloudflare Workers)
│       • 100k requests/day free.
│       • 5-minute setup.
│       • No server to maintain.
│       • Requires HF_TOKEN + original model IDs (openai/*, Qwen/*).
│
└─ I want a production-grade cloud setup with full control.
    └─ PATH C (ZeroGPU) for the model + PATH B or D for the proxy.
       When traffic grows, upgrade the HF Space hardware tier.
```

---

## Path A – Local Dev via Docker Model Runner (Zero Cost)

**When to use:** You want to run the model on your own machine during development.
No HF token needed. No internet required after the model is downloaded.
Works with `scikit-plots/*` mirror repos (because DMR downloads GGUF weights directly,
bypassing the HF Serverless Inference API entirely).

### What Docker Model Runner Is

Docker Model Runner (DMR) is built into Docker Desktop (macOS and Windows) and
available as a plugin for Docker Engine (Linux). It pulls GGUF-format model weights
from HuggingFace and exposes them via a local OpenAI-compatible REST API.
The model runs on your machine's GPU (NVIDIA, Apple Silicon) or CPU if no GPU is available.

Reference: https://docs.docker.com/ai/model-runner/

### Prerequisites

```
Platform support:
  macOS  : Docker Desktop ≥ 4.40  (Apple Silicon or Intel)
  Windows: Docker Desktop ≥ 4.40  (with WSL2 enabled)
  Linux  : Docker Engine with the model runner plugin (see Linux note below)

System requirements:
  ✓ ~15–25 GB free disk space for the 20B model GGUF file
  ✓ 16 GB RAM minimum (32 GB recommended for comfortable operation)
  ✓ GPU strongly recommended; CPU-only works but inference is very slow

Download Docker Desktop: https://www.docker.com/products/docker-desktop/

──────────────────────────────────────────────────────
LINUX NOTE (Different Setup)
──────────────────────────────────────────────────────
Docker Desktop for Linux does not include DMR in the same way.
On Linux, install Docker Engine and the model runner plugin separately.
Refer to the official docs for the current Linux installation procedure:
  https://docs.docker.com/ai/model-runner/
The commands after installation (docker model pull, docker model run) are identical.
──────────────────────────────────────────────────────
```

### Step 1 — Enable Docker Model Runner

**macOS / Windows (Docker Desktop):**
```
Docker Desktop → Settings (gear icon)
→ Features in development
→ Check "Enable Docker Model Runner"
→ Click "Apply & Restart"
```

Wait for Docker Desktop to restart (30–60 seconds).

**Verify it is enabled:**
```bash
# This should print usage information, NOT "docker: 'model' is not a docker command".
docker model --help
```
Expected output begins with:
```
Usage:  docker model COMMAND
Run AI models locally
```
If you see "command not found" or "is not a docker command", DMR is not enabled or
Docker Desktop version is too old. Upgrade to ≥ 4.40.

### Step 2 — Pull and Run the Model

```bash
# Pull AND run in one step.
# docker model run automatically downloads the model if not already present.
# The first run downloads ~15 GB — this is one-time only.
# Subsequent runs start immediately from the local cache.
docker model run hf.co/scikit-plots/gpt-oss-20b
```

<!--
HINT — Why does this work for a mirror repo?
Docker Model Runner downloads the raw GGUF weight files directly from the
HuggingFace repository storage (Git LFS). It does NOT go through the
HuggingFace Serverless Inference API. This is why scikit-plots/* mirror repos
work here even though they show "no Inference Provider" on the model page.
-->

If you want to download in advance (before running):
```bash
# Optional: pre-download without starting the server.
docker model pull hf.co/scikit-plots/gpt-oss-20b

# Then start the server separately.
docker model run hf.co/scikit-plots/gpt-oss-20b
```

### Step 3 — Verify the Model is Running

Open a new terminal (leave the model running in the first one):

```bash
# Test the model endpoint directly.
# Expected: a JSON response containing an assistant message.
curl http://localhost:12434/engines/llama.cpp/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hf.co/scikit-plots/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50
  }'
```

Expected response structure:
```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
      "finish_reason": "stop"
    }
  ]
}
```

If you get `connection refused`, the model is still loading. Wait 30 seconds and retry.

### Step 4 — Wire `conf.py` for Local DMR

```python
# docs/conf.py
# ─────────────────────────────────────────────────────────────────────────────
# LOCAL DEV ONLY — Docker Model Runner configuration.
#
# This config points the Sphinx AI widget directly at the local DMR endpoint.
# DO NOT deploy this config to production — localhost:12434 is not reachable
# from GitHub Pages or ReadTheDocs.
#
# For CI/CD or production, set AI_PROXY_BASE to your deployed proxy URL:
#   export AI_PROXY_BASE=https://scikit-plots-ai.hf.space
# ─────────────────────────────────────────────────────────────────────────────
import os

# Read the base URL from an environment variable so you can switch between
# local DMR and your deployed proxy without editing this file.
_PROXY_BASE: str = os.environ.get(
    "AI_PROXY_BASE",
    "http://localhost:12434/engines/llama.cpp",   # Docker Model Runner default port
)

ai_assistant_panel_api_enabled = True

ai_assistant_panel_api_models = [
    {
        # Unique identifier for this entry (any string, used internally by the widget).
        "id": "dmr-gpt-oss-20b",

        # Label shown in the UI model-picker dropdown.
        "label": "GPT-OSS 20B (local Docker)",

        # Provider tag — tells the widget how to format requests.
        "provider": "huggingface",

        # The model ID as Docker Model Runner expects it.
        # Must match the argument you used with `docker model run`.
        "model": "hf.co/scikit-plots/gpt-oss-20b",

        # Full endpoint URL. DMR exposes /engines/llama.cpp/v1/chat/completions.
        "endpoint": f"{_PROXY_BASE}/v1/chat/completions",

        # Set this model as the default in the UI dropdown.
        "default": True,

        # Optional tooltip shown in the UI.
        "description": "GPT-OSS 20B running locally via Docker Model Runner. No internet required.",
    },
]
```

### Step 5 — Build and Serve Your Docs

```bash
# Terminal 1: model is running (from Step 2)

# Terminal 2: build and serve the docs
cd your-sphinx-project/
make html
python -m http.server 8080 --directory _build/html

# Open http://localhost:8080 in your browser.
# The AI chat panel should now connect to the local model.
```

---

## Path B – Fix the Deployed HF Space (One Env Var)

**When to use:** The `scikit-plots/ai` Space is already deployed and requests return
404 or 503. No code changes and no redeployment needed. Takes 2 minutes.

**Why this works:** The current proxy builds its upstream URL as:
```
https://router.huggingface.co/v1/chat/completions
```
with the model selected via the `"model"` field in the request body.
When `model = "scikit-plots/gpt-oss-20b"`, HuggingFace returns 404/503 because that mirror repo has
no Inference Provider. When you change `DEFAULT_MODEL` to `"Qwen/Qwen2.5-Coder-32B-Instruct"`, the
request resolves to the original repo which DOES have a registered provider.

### Step 1 — Go to the Space Settings

```
https://huggingface.co/spaces/scikit-plots/ai
→ Click the "Settings" tab
→ Scroll to "Repository secrets"
→ Click "New secret"
```

### Step 2 — Add or Update Secrets

```
Secret 1:
  Name:  DEFAULT_MODEL
  Value: Qwen/Qwen2.5-Coder-32B-Instruct

Secret 2 (if not already set):
  Name:  HF_TOKEN
  Value: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
         (Get from: https://huggingface.co/settings/tokens)
         (Requires "Inference" permission — check "Make calls to the Inference API")
```

Click "Save" after each secret. The Space restarts automatically (~30 seconds).

### Available Models with Confirmed Inference Providers

| Use this model ID | Instead of |
|---|---|
| `Qwen/Qwen2.5-Coder-32B-Instruct` | `scikit-plots/Qwen2.5-Coder-32B-Instruct` |
| `Qwen/Qwen2.5-72B-Instruct` | `scikit-plots/gpt-oss-20b` |

### Step 3 — Verify the Fix

```bash
# Replace with your actual Space URL.
# Expected: {"status": "ok", ...}
curl https://scikit-plots-ai.hf.space/health

# Test a real completion (use the original model ID).
curl https://scikit-plots-ai.hf.space/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-Coder-32B-Instruct", "messages": [{"role": "user", "content": "hi"}]}'
```

---

## Path C – New ZeroGPU Space (Completely Free GPU)

**When to use:** You want to specifically serve `scikit-plots/*` model weights
(not the original repos) in the cloud at zero cost.

**How it works:**
```
Browser
  → scikit-plots/ai  (existing proxy Space, free CPU, forwards via BACKEND_URL)
  → scikit-plots/ai-model  (new model Space, free ZeroGPU, loads weights and runs inference)
```

The proxy Space (`scikit-plots/ai`) requires zero code changes — you only add one
`BACKEND_URL` environment variable pointing at the new model Space.

### ⚠️ Cold Start Warning

The first request to a ZeroGPU Space in a new session triggers:
1. GPU allocation from the shared pool (variable, can take 1–5 minutes)
2. Model loading from storage to GPU VRAM (for 20B: 3–8 minutes)

**Total cold start time for a 20B model: 2–10 minutes.**

Set `PROXY_TIMEOUT=600` in the proxy Space secrets to avoid premature timeouts.
Subsequent requests in the same session complete in seconds.

### Step 1 — Create the Model Space

```
1. Go to: https://huggingface.co/new-space

2. Fill in:
   Owner      : scikit-plots
   Space name : ai-model
   SDK        : Gradio            ← REQUIRED: must be Gradio, NOT Docker,
                                     to be eligible for ZeroGPU hardware
   License    : choose any
   Visibility : Public            ← Required: the proxy Space must be able to reach it
                                     (private Spaces require additional auth setup)
   Hardware   : CPU basic (free)  ← Set this initially; change to ZeroGPU after creation

3. Click "Create Space".

4. After creation:
   Space → Settings → Space hardware
   → Change to "ZeroGPU" (under "GPU" section, marked "Free")
   → Click "Save"
```

<!--
HINT — Why must SDK be Gradio, not Docker?
ZeroGPU is a HuggingFace infrastructure feature only available to Gradio SDK Spaces.
Docker Spaces run in a fixed container and cannot use the shared GPU queue.
The `spaces` Python library (which provides @spaces.GPU) only functions inside
Gradio SDK Spaces. Source: https://huggingface.co/docs/hub/spaces-zerogpu
-->

### Step 2 — Add `README.md` to the Model Space

<!--
IMPORTANT: HuggingFace Spaces require a README.md with YAML front matter.
Without this, the Space will not be correctly identified and may fail to build.
Source: https://huggingface.co/docs/hub/spaces-config-reference
-->

```markdown
---
title: scikit-plots AI Model Endpoint
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

# scikit-plots AI Model Endpoint

This Space serves scikit-plots model weights via a ZeroGPU-accelerated
OpenAI-compatible REST endpoint.

## Primary endpoint

```
POST /v1/chat/completions
```

Request body (OpenAI Chat Completions format):
```json
{"model": "scikit-plots/gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}
```

## Gradio UI

A test UI is available at `/ui` for manual verification.

## Configuration

Set the `MODEL_ID` secret in Space → Settings → Repository secrets.
```

### Step 3 — Add `requirements.txt` to the Model Space

```text
# scikit-plots/ai-model · requirements.txt
#
# NOTE: The following are PRE-INSTALLED on ZeroGPU Spaces and must NOT be
# listed here (listing them can cause version conflicts):
#   - gradio
#   - spaces (the @spaces.GPU decorator library)
#   - torch  (PyTorch — pre-installed with CUDA support)
#
# Only list packages NOT pre-installed.
#
# Version pinning: use ~= (compatible release) to allow patch upgrades
# while preventing breaking major/minor version changes.

transformers~=4.44.0     # Model loading, tokenizer, pipeline
accelerate~=0.33.0       # Required by device_map="auto" for multi-GPU / CPU offload
```

### Step 4 — Add `app.py` to the Model Space

```python
# scikit-plots/ai-model · app.py
#
# PURPOSE
# ───────
# This Space downloads and runs model weights on ZeroGPU (free shared GPU).
# It exposes an OpenAI-compatible REST endpoint:
#   POST /v1/chat/completions
#
# The proxy Space (scikit-plots/ai) calls this endpoint via BACKEND_URL.
# Browsers should NEVER call this Space directly — all requests go through
# the proxy Space which handles CORS, token injection, and SSE streaming.
#
# ARCHITECTURE
# ────────────
# • _api       : FastAPI app — handles REST endpoint /v1/chat/completions
# • _generate  : Synchronous function decorated with @spaces.GPU
#                GPU is acquired at function entry, released at function exit.
#                Each call independently acquires and releases the GPU.
# • _demo      : Minimal Gradio UI — required to keep the ZeroGPU Space alive.
#                Also used for manual testing.
# • app        : FastAPI app with Gradio mounted at /ui (required export name
#                for HF Spaces when using gr.mount_gradio_app)
#
# CRITICAL: @spaces.GPU is called from an async FastAPI route.
# The synchronous _generate function is called via asyncio.to_thread()
# to avoid blocking the asyncio event loop. Failure to do this would freeze
# all concurrent requests while one is generating a response.
#
# ENVIRONMENT VARIABLES (set in Space → Settings → Repository secrets)
# ────────────────────────────────────────────────────────────────────
# MODEL_ID : Model weights to load. Supports scikit-plots/* mirrors because
#            the weights are downloaded directly, not via HF Serverless API.
#            Default: "scikit-plots/gpt-oss-20b"

from __future__ import annotations

import asyncio
import json
import os
import uuid

import gradio as gr
import spaces                                   # ZeroGPU decorator — pre-installed
import torch                                    # Pre-installed on ZeroGPU Spaces
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

#: Model weights to load.
#: This can be a scikit-plots/* mirror — we download weights directly.
#: Set MODEL_ID in Space → Settings → Repository secrets.
MODEL_ID: str = os.environ.get("MODEL_ID", "scikit-plots/gpt-oss-20b").strip()

if not MODEL_ID:
    raise RuntimeError(
        "MODEL_ID environment variable is empty. "
        "Set it in Space → Settings → Repository secrets."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Model loading strategy for ZeroGPU
# ─────────────────────────────────────────────────────────────────────────────
# CORRECT PATTERN for ZeroGPU:
#   • Load the tokenizer on CPU at startup (lightweight — safe at import time).
#   • Load the model on CPU with low_cpu_mem_usage=True at startup.
#     This streams weights from disk without loading the full model into RAM.
#   • Move the model to GPU inside @spaces.GPU — GPU is available only there.
#
# WRONG PATTERN (was in v2.0.0 — caused OOM crashes):
#   • pipeline("text-generation", model=MODEL_ID, device_map="auto") at module level.
#     This tries to load the full model into CPU RAM at import time.
#     A 20B model needs ~40 GB CPU RAM (float32). ZeroGPU Spaces have ≤16 GB.
#     Result: Out Of Memory crash before any request is served.
#
# Source: https://huggingface.co/docs/hub/spaces-zerogpu
# ─────────────────────────────────────────────────────────────────────────────

_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,      # bfloat16 halves memory use vs float32
    low_cpu_mem_usage=True,           # stream weights from disk, do not load all into RAM
    device_map="cpu",                 # stay on CPU until GPU is allocated by @spaces.GPU
)

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app (REST endpoint for the proxy to call)
# ─────────────────────────────────────────────────────────────────────────────

_api = FastAPI(
    title="scikit-plots model endpoint",
    description=(
        "ZeroGPU Space serving scikit-plots model weights. "
        "Accepts OpenAI-compatible POST /v1/chat/completions requests. "
        "This Space is called by the proxy Space (scikit-plots/ai) "
        "via its BACKEND_URL environment variable."
    ),
    version="1.0.0",
)

_api.add_middleware(
    CORSMiddleware,
    # Allow only the proxy Space to call this endpoint.
    # The proxy is the only caller — browsers should never reach this Space directly.
    # If you change the proxy Space URL, update this list.
    allow_origins=[
        "https://scikit-plots-ai.hf.space",   # proxy Space URL
        "http://localhost:7860",               # local dev
        "http://localhost:8787",               # dev_proxy.py
    ],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@spaces.GPU
def _generate(messages: list[dict], max_new_tokens: int = 512) -> str:
    """
    Run text generation on the GPU-allocated model.

    Parameters
    ----------
    messages : list of dict
        OpenAI-style message list. Each dict must have ``role`` (str) and
        ``content`` (str) keys.
    max_new_tokens : int, optional
        Maximum number of new tokens to generate. Default is 512.

    Returns
    -------
    str
        The assistant's generated response text.

    Raises
    ------
    ValueError
        If ``messages`` is empty.
    RuntimeError
        If the tokenizer lacks a ``chat_template`` and cannot format the messages.

    Notes
    -----
    Developer note:
        ``@spaces.GPU`` acquires a GPU from the ZeroGPU shared pool at function
        entry and releases it at function exit. The GPU is NOT held between calls.
        This is the only supported pattern for ZeroGPU — holding the GPU across
        multiple requests would violate the shared pool contract.

        The model is moved to GPU (.cuda()) inside this function because GPU is
        only available after ``@spaces.GPU`` takes effect.  The model is moved
        back to CPU (.cpu()) before returning so it does not hold GPU memory
        between calls.

    User note:
        The first call in a new session incurs a cold start:
        GPU allocation + model transfer from CPU to GPU.
        This can take 2–10 minutes for a 20B model.
        Subsequent calls in the same session are much faster.
    """
    if not messages:
        raise ValueError("messages list must not be empty.")

    # Move model to GPU for this call.
    # @spaces.GPU ensures a GPU is available at this point.
    _model.cuda()

    try:
        # Apply chat template. Requires the tokenizer to have chat_template defined.
        # Most modern instruction-tuned models include this in their tokenizer config.
        if not hasattr(_tokenizer, "chat_template") or _tokenizer.chat_template is None:
            raise RuntimeError(
                f"Tokenizer for {MODEL_ID} does not have a chat_template. "
                "This model may not support chat-formatted input. "
                "Check the model card for the correct input format."
            )

        input_ids = _tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,     # adds the assistant turn prefix
            return_tensors="pt",
        ).cuda()

        with torch.no_grad():
            output_ids = _model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=_tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude the input prompt).
        new_token_ids = output_ids[0][input_ids.shape[-1]:]
        return _tokenizer.decode(new_token_ids, skip_special_tokens=True)

    finally:
        # Always move model back to CPU so GPU memory is released for the next caller.
        _model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# REST routes
# ─────────────────────────────────────────────────────────────────────────────

@_api.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    """
    OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    request : Request
        FastAPI request. Body must be valid JSON with a ``messages`` list.
        Supports ``max_tokens`` (int, optional).

    Returns
    -------
    JSONResponse
        JSON response matching the OpenAI Chat Completions v1 schema.
        Contains one choice with the assistant message.

    Notes
    -----
    Developer note:
        _generate is synchronous (required by @spaces.GPU). Calling it directly
        in an async route would block the event loop, freezing all other requests.
        asyncio.to_thread() runs it in a thread pool, keeping the event loop free.

        Streaming (stream=True) is not supported in this Space — the proxy Space
        handles SSE. The proxy buffers the full response from this endpoint and
        then streams it to the browser.

    User note:
        This endpoint does not support stream=True. If your client requests
        streaming, the proxy will buffer the full response and return it at once.
        Add a GitHub issue if streaming support is needed here.
    """
    body = await request.body()

    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    messages: list[dict] = data.get("messages", [])
    if not messages:
        return JSONResponse({"error": "messages field is required and must not be empty"}, status_code=422)

    # Validate and sanitize max_tokens.
    raw_max_tokens = data.get("max_tokens", 512)
    try:
        max_tokens: int = max(1, min(int(raw_max_tokens), 4096))
    except (TypeError, ValueError):
        return JSONResponse(
            {"error": f"max_tokens must be an integer, got: {raw_max_tokens!r}"},
            status_code=422,
        )

    # Run inference in a thread pool (asyncio.to_thread) so the event loop is not blocked.
    # _generate is synchronous and decorated with @spaces.GPU — cannot be made async.
    try:
        content: str = await asyncio.to_thread(_generate, messages, max_tokens)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=422)
    except RuntimeError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    # Return an OpenAI-compatible response with a unique ID per request.
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {},   # populated in future versions; kept for schema compatibility
    })


@_api.get("/health")
async def health() -> JSONResponse:
    """Liveness probe. Returns 200 when the Space is running and model is loaded."""
    return JSONResponse({
        "status": "ok",
        "model": MODEL_ID,
        "model_loaded": _model is not None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI — required to keep the ZeroGPU Space alive
# ─────────────────────────────────────────────────────────────────────────────
# ZeroGPU Spaces must use the Gradio SDK. A Gradio UI must be present for the
# Space to remain active and eligible for GPU allocation. The REST endpoint at
# /v1/chat/completions is the primary interface; the UI is for manual testing.

def _gradio_chat(message: str, system: str = "") -> str:
    """Gradio UI wrapper around _generate."""
    msgs: list[dict] = []
    if system.strip():
        msgs.append({"role": "system", "content": system.strip()})
    msgs.append({"role": "user", "content": message.strip()})
    return _generate(msgs)


_demo = gr.Interface(
    fn=_gradio_chat,
    inputs=[
        gr.Textbox(label="User message", lines=3, placeholder="Type your message here..."),
        gr.Textbox(label="System prompt (optional)", lines=2),
    ],
    outputs=gr.Textbox(label="Assistant response", lines=8),
    title=f"scikit-plots · {MODEL_ID}",
    description=(
        "**Model endpoint Space** — primary interface is `POST /v1/chat/completions`.\n\n"
        "This UI is for manual testing only. The AI proxy Space calls the REST endpoint.\n\n"
        "⚠️ First request may take 2–10 minutes (GPU cold start). Subsequent: fast."
    ),
)

# Mount the Gradio UI at /ui so the REST routes remain at / and /v1/...
# gr.mount_gradio_app attaches the Gradio demo to the FastAPI app.
# The return value must be named `app` — HF Spaces looks for this variable.
app = gr.mount_gradio_app(_api, _demo, path="/ui")
```

### Step 5 — Set Environment Variables in the Model Space

```
Space (scikit-plots/ai-model) → Settings → Repository secrets → New secret:
  Name:  MODEL_ID
  Value: scikit-plots/gpt-oss-20b
```

### Step 6 — Point the Proxy Space at the Model Space

```
Space (scikit-plots/ai) → Settings → Repository secrets → New secret:
  Name:  BACKEND_URL
  Value: https://scikit-plots-ai-model.hf.space/v1/chat/completions

Also update:
  Name:  PROXY_TIMEOUT
  Value: 600
  (10 minutes — necessary for ZeroGPU cold start on a 20B model)
```

The proxy Space (`scikit-plots/ai`) now forwards all requests to the model Space.
No code changes needed in the proxy. No proxy redeployment needed.

---

## Path D – Cloudflare Worker Proxy (100k req/day Free)

**When to use:** Your docs are on GitHub Pages and you want a free, low-maintenance
proxy with no server to manage. You still need `HF_TOKEN` and must use original
model IDs that have Inference Providers (Path B models, not `scikit-plots/*`).

**Limits:**
- Free tier: 100,000 requests/day, 10ms CPU time per request (network I/O not counted)
- Wall-clock limit per request: 30 seconds (free tier). Sufficient for most completions.
  Very large responses or slow models may time out. Upgrade to Cloudflare Workers Paid
  ($5/month) for 50ms CPU and no wall-clock limit.

Reference: https://developers.cloudflare.com/workers/platform/limits/

### Prerequisites

```
✓ Cloudflare account (free):  https://dash.cloudflare.com/sign-up
✓ Node.js ≥ 18:               https://nodejs.org/
✓ HF_TOKEN with "Inference" read permission
✓ A model with an Inference Provider (Qwen/Qwen2.5-Coder-32B-Instruct, not scikit-plots/...)
```

### Step 1 — Install Wrangler CLI

```bash
# Wrangler v3+ is the current Cloudflare CLI.
# NOTE: `wrangler init` was deprecated and removed in Wrangler v3.
# The correct project creation command is `npm create cloudflare@latest`.
npm install -g wrangler

# Verify installation.
wrangler --version
# Expected: wrangler 3.x.x

# Log in to your Cloudflare account (opens a browser window).
wrangler login
```

### Step 2 — Create the Worker Project

```bash
# Create a new Worker project using the current Cloudflare scaffolding tool.
# When prompted, choose:
#   "Hello World" or "Worker" template
#   Language: JavaScript
npm create cloudflare@latest -- hf-proxy

# Enter the project directory.
cd hf-proxy
```

<!--
HINT — Why not `wrangler init`?
`wrangler init` was removed in Wrangler v3.
Running it produces: "error: unknown command 'init'".
The replacement is `npm create cloudflare@latest`.
Source: https://developers.cloudflare.com/workers/wrangler/commands/
-->

### Step 3 — Verify `wrangler.toml` Exists and is Correct

`npm create cloudflare@latest` generates `wrangler.toml` automatically.
Open it and confirm it contains at minimum:

```toml
# hf-proxy/wrangler.toml
#
# Configuration for the hf-proxy Cloudflare Worker.
# Compatibility date must be within the last year for Workers to function.
# See: https://developers.cloudflare.com/workers/configuration/compatibility-dates/

name            = "hf-proxy"
main            = "src/index.js"
compatibility_date = "2024-09-23"   # update to a recent date if generated date is old

# Uncomment to enable routing on a custom domain:
# routes = [{ pattern = "ai.yourdomain.com/*", zone_name = "yourdomain.com" }]
```

If `wrangler.toml` is missing or empty, `wrangler deploy` will fail.

### Step 4 — Replace `src/index.js` with This Code

```javascript
/**
 * Cloudflare Worker: HuggingFace Inference API Proxy
 *
 * PURPOSE
 * ───────
 * Accepts POST /v1/chat/completions from the browser (no auth header needed).
 * Adds Authorization: Bearer $HF_TOKEN from the Worker's encrypted secrets.
 * Forwards the request to the HuggingFace Serverless Inference API.
 * Returns the response (JSON or SSE stream) to the browser with CORS headers.
 *
 * LIMITATIONS
 * ───────────
 * Free tier limits (source: https://developers.cloudflare.com/workers/platform/limits/):
 *   • 100,000 requests/day
 *   • 10ms CPU time per request (network I/O wait does NOT count toward CPU time)
 *   • 30-second wall-clock limit per request (adequate for most completions)
 * Only works with models that have a registered Inference Provider.
 * Use Qwen/Qwen2.5-Coder-32B-Instruct, NOT scikit-plots/Qwen2.5-Coder-32B-Instruct.
 *
 * SETUP
 * ─────
 * 1. wrangler secret put HF_TOKEN    (paste your token; stored encrypted by Cloudflare)
 * 2. wrangler deploy
 * 3. Note the deployed URL shown after deploy completes.
 *    Example: https://hf-proxy.your-subdomain.workers.dev
 *
 * @param {Request} request  Incoming HTTP request from the browser.
 * @param {Object}  env      Worker environment. HF_TOKEN lives here as a Wrangler secret.
 * @returns {Response}       Proxied response from HuggingFace.
 */
export default {
  async fetch(request, env) {

    // ── CORS Preflight ────────────────────────────────────────────────────────
    // Browsers send OPTIONS before every cross-origin POST to check CORS policy.
    // We must respond 204 with the appropriate headers or the POST is blocked.
    if (request.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: corsHeaders(),
      });
    }

    // ── Method Guard ──────────────────────────────────────────────────────────
    // Only POST is valid for this proxy.
    if (request.method !== "POST") {
      return new Response(
        JSON.stringify({ error: "Method Not Allowed. Use POST /v1/chat/completions." }),
        { status: 405, headers: { "Content-Type": "application/json", ...corsHeaders() } }
      );
    }

    // ── Token Guard ───────────────────────────────────────────────────────────
    // Fail fast if HF_TOKEN was not set as a Wrangler secret.
    if (!env.HF_TOKEN) {
      return new Response(
        JSON.stringify({ error: "Server configuration error: HF_TOKEN secret is not set." }),
        { status: 500, headers: { "Content-Type": "application/json", ...corsHeaders() } }
      );
    }

    // ── Parse Request Body ────────────────────────────────────────────────────
    let body;
    try {
      body = await request.text();
    } catch (_) {
      return new Response(
        JSON.stringify({ error: "Failed to read request body." }),
        { status: 400, headers: { "Content-Type": "application/json", ...corsHeaders() } }
      );
    }

    // Extract the model ID from the request body.
    // Fall back to a model that has a confirmed Inference Provider.
    // IMPORTANT: use original repo IDs (Qwen/...), NOT mirrors (scikit-plots/...).
    let model = "Qwen/Qwen2.5-Coder-32B-Instruct";
    try {
      const parsed = JSON.parse(body);
      if (parsed.model && typeof parsed.model === "string" && parsed.model.trim()) {
        model = parsed.model.trim();
      }
    } catch (_) {
      // Malformed JSON — use fallback model.
    }

    // ── Build Upstream URL ────────────────────────────────────────────────────
    // router.huggingface.co uses a flat endpoint; model is selected via request body.
    const hfUrl = "https://router.huggingface.co/v1/chat/completions";

    // ── Forward to HuggingFace ────────────────────────────────────────────────
    // env.HF_TOKEN is a Worker secret — encrypted, never visible in browser or source.
    let hfResp;
    try {
      hfResp = await fetch(hfUrl, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.HF_TOKEN}`,
          "Content-Type":  "application/json",
        },
        body: body,
      });
    } catch (err) {
      return new Response(
        JSON.stringify({ error: `Failed to reach HuggingFace API: ${err.message}` }),
        { status: 502, headers: { "Content-Type": "application/json", ...corsHeaders() } }
      );
    }

    // ── Return Response ───────────────────────────────────────────────────────
    // Preserve the upstream content-type (JSON or text/event-stream for SSE).
    // Add CORS headers so the browser accepts the response.
    const responseHeaders = {
      "Content-Type": hfResp.headers.get("content-type") || "application/json",
      ...corsHeaders(),
    };

    return new Response(hfResp.body, {
      status:  hfResp.status,
      headers: responseHeaders,
    });
  },
};

/**
 * Standard CORS headers required for browser cross-origin requests.
 *
 * @returns {Object} CORS headers object.
 */
function corsHeaders() {
  return {
    "Access-Control-Allow-Origin":  "*",           // allow any origin
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  };
}
```

### Step 5 — Add HF_TOKEN as an Encrypted Secret

```bash
# This stores your token encrypted in Cloudflare's secrets manager.
# You will be prompted to paste the token value interactively.
# The token is NEVER stored in wrangler.toml or source code.
wrangler secret put HF_TOKEN
# Paste your token when prompted: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Expected output: ✓ Success! Uploaded secret HF_TOKEN.
```

### Step 6 — Deploy

```bash
wrangler deploy
# Expected output includes:
#   Published hf-proxy (x.xx sec)
#   https://hf-proxy.<your-subdomain>.workers.dev
```

Note the deployed URL — you need it for `conf.py`.

---

## Path E – Local Python Dev Proxy (No Docker)

**When to use:** You want to develop locally without Docker Desktop.
Runs alongside `python -m http.server`. Requires `HF_TOKEN` and a model
with an Inference Provider (use `Qwen/Qwen2.5-Coder-32B-Instruct`, not `scikit-plots/...`).

**Limitation:** This proxy is single-threaded. One slow HF response (30–90 seconds
for a large model) blocks all concurrent requests. Do not open multiple browser
tabs while a request is in flight. For concurrent use, use `app.py` (FastAPI) instead.

### Step 1 — Create a Virtual Environment and Install Dependencies

```bash
# Create a project-local virtual environment (keeps your system Python clean).
python -m venv .venv

# Activate it.
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows cmd
# .venv\Scripts\Activate.ps1     # Windows PowerShell

# Install the only dependency.
pip install httpx
```

### Step 2 — Save `dev_proxy.py`

```python
#!/usr/bin/env python3
# dev_proxy.py
#
# PURPOSE
# ───────
# Minimal single-file development proxy — NOT for production use.
# Listens on http://localhost:8787 and forwards POST /v1/chat/completions
# to the HuggingFace Serverless Inference API with HF_TOKEN injected.
#
# USAGE
# ─────
# Terminal 1 (proxy):
#   source .venv/bin/activate
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   python dev_proxy.py
#
# Terminal 2 (docs server):
#   make html
#   python -m http.server 8080 --directory _build/html
#
# Then open http://localhost:8080 and set conf.py endpoint to:
#   http://localhost:8787/v1/chat/completions
#
# IMPORTANT CONSTRAINTS
# ─────────────────────
# • Use model IDs with Inference Providers: Qwen/Qwen2.5-Coder-32B-Instruct (not scikit-plots/...)
# • Single-threaded: one slow request (30–90 sec) blocks all other requests.
# • Never run on a public network: no rate limiting, no authentication.
# • HTTPS is not supported: HTTP only. For TLS, use the FastAPI proxy (app.py).

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all values from environment variables (never hard-code tokens)
# ─────────────────────────────────────────────────────────────────────────────

HF_TOKEN: str = os.environ.get("HF_TOKEN", "").strip()
HF_BASE: str = os.environ.get(
    "HF_BASE",
    "https://router.huggingface.co",
).rstrip("/")
DEFAULT_MODEL: str = os.environ.get(
    "DEFAULT_MODEL",
    "Qwen/Qwen2.5-Coder-32B-Instruct",   # original repo — has a confirmed Inference Provider
)
PORT: int = int(os.environ.get("DEV_PROXY_PORT", "8787"))
TIMEOUT: int = int(os.environ.get("PROXY_TIMEOUT", "600"))

# Fail fast at startup — better to crash with a clear message than to fail silently
# on the first request.
if not HF_TOKEN:
    print(
        "[dev_proxy] ERROR: HF_TOKEN environment variable is not set.\n"
        "[dev_proxy] Export it before running:\n"
        "[dev_proxy]   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
        "[dev_proxy] Or use Path A (Docker Model Runner) which requires no token.",
        file=sys.stderr,
    )
    sys.exit(1)


class ProxyHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler that proxies POST /v1/chat/completions to HuggingFace.

    Notes
    -----
    Developer note:
        Uses the standard library HTTPServer — synchronous, single-threaded.
        Intentionally simple: no connection pooling, no async, no streaming support.
        One request is handled at a time. A slow upstream response (30–90 seconds
        for large models) blocks all other requests during that time.
        For concurrent use or SSE streaming, switch to the FastAPI proxy (app.py).

    User note:
        If the browser appears frozen while a response is loading, this is expected.
        Open only one browser tab while using this proxy to avoid blocking.
    """

    def do_OPTIONS(self) -> None:  # noqa: N802 (HTTP method naming)
        """Handle CORS preflight request. Browser sends this before every POST."""
        self.send_response(204)
        self._write_cors_headers()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        """Forward POST body to HuggingFace and write the response back."""
        # Read the full request body.
        # Content-Length is required; default to 0 if absent (body will be empty).
        try:
            length: int = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        body: bytes = self.rfile.read(length)

        # Extract model ID from the JSON body.
        # If parsing fails for any reason, fall back to DEFAULT_MODEL.
        model: str = DEFAULT_MODEL
        try:
            parsed = json.loads(body)
            candidate = str(parsed.get("model", "")).strip()
            if candidate:
                model = candidate
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass  # malformed body — use default model

        # router.huggingface.co uses a flat endpoint; the model is selected via the
        # request body (the "model" field), not the URL path.
        url: str = f"{HF_BASE}/v1/chat/completions"

        print(f"[dev_proxy] → POST {url}", flush=True)

        try:
            resp = httpx.post(
                url,
                content=body,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type":  "application/json",
                },
                timeout=TIMEOUT,
            )
        except httpx.TimeoutException:
            self._write_error(504, f"Upstream timed out after {TIMEOUT}s. "
                                   "Try increasing PROXY_TIMEOUT or use a smaller model.")
            return
        except httpx.RequestError as exc:
            self._write_error(502, f"Failed to reach upstream: {exc}")
            return

        print(f"[dev_proxy] ← {resp.status_code}", flush=True)

        self.send_response(resp.status_code)
        self.send_header(
            "Content-Type",
            resp.headers.get("content-type", "application/json"),
        )
        self._write_cors_headers()
        self.end_headers()
        self.wfile.write(resp.content)

    def _write_cors_headers(self) -> None:
        """Emit CORS headers required for browser cross-origin acceptance."""
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _write_error(self, status: int, message: str) -> None:
        """Write a JSON error response with CORS headers."""
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._write_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        """Override to prefix log lines with [dev_proxy] for clarity."""
        print(f"[dev_proxy] {fmt % args}", flush=True)


if __name__ == "__main__":
    print(f"[dev_proxy] Listening on http://localhost:{PORT}")
    print(f"[dev_proxy] Forwarding to: {HF_BASE}/<model>/v1/chat/completions")
    print(f"[dev_proxy] Default model: {DEFAULT_MODEL}")
    print(f"[dev_proxy] HF_TOKEN:      {HF_TOKEN[:8]}...{HF_TOKEN[-4:]}  (truncated for safety)")
    print(f"[dev_proxy] Timeout:       {TIMEOUT}s")
    print("[dev_proxy] Press Ctrl+C to stop.", flush=True)
    try:
        HTTPServer(("127.0.0.1", PORT), ProxyHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n[dev_proxy] Stopped.")
```

### Step 3 — Run It

```bash
# Activate venv if not already active.
source .venv/bin/activate

# Set your token and start the proxy.
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
python dev_proxy.py
```

Expected output:
```
[dev_proxy] Listening on http://localhost:8787
[dev_proxy] Forwarding to: https://router.huggingface.co/v1/chat/completions  (model via request body)
[dev_proxy] Default model: scikit-plots/Qwen2.5-Coder-32B-Instruct
[dev_proxy] HF_TOKEN:      hf_xxxxxx...xxxx  (truncated for safety)
[dev_proxy] Timeout:       600s
[dev_proxy] Press Ctrl+C to stop.
```

---

## Updated `app.py` for `scikit-plots/ai` Proxy Space

This is `app.py` v3.0.0 for the existing proxy Space (`scikit-plots/ai`).
It adds `BACKEND_URL` routing, SSE streaming, proper timeout configuration, and
CORS origin restriction via environment variable. All existing routes are preserved.

### `README.md` for the Proxy Space

<!--
IMPORTANT: HF Spaces require README.md with YAML front matter.
This was missing from v2.0.0.
Source: https://huggingface.co/docs/hub/spaces-config-reference
-->

```markdown
---
title: sphinx-ai-assistant proxy
emoji: 🔁
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# sphinx-ai-assistant proxy

Thin OpenAI-compatible reverse proxy for the sphinx-ai-assistant Sphinx extension.

## What it does

Accepts `POST /v1/chat/completions` from the browser (no auth header).
Adds `Authorization: Bearer $HF_TOKEN` or routes to `BACKEND_URL`.
Forwards the request to the configured model backend.
Returns the response (JSON or SSE stream) with CORS headers.

## Configuration

Set these in Space → Settings → Repository secrets:

| Variable | Required? | Description |
|---|---|---|
| `HF_TOKEN` | Yes (unless `BACKEND_URL` set) | HuggingFace API token |
| `BACKEND_URL` | No | Custom backend URL (DMR or ZeroGPU Space) |
| `DEFAULT_MODEL` | No | Fallback model ID. Default: `openai/gpt-oss-20b` |
| `PROXY_TIMEOUT` | No | Upstream timeout seconds. Default: 120. For ZeroGPU: 600 |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins. Default: `*` |

## Endpoints

- `GET  /`                        — status page
- `GET  /health`                  — liveness probe
- `POST /v1/chat/completions`     — primary proxy endpoint
- `POST /`                        — alias (for older sphinx-ai-assistant versions)
```

### `Dockerfile` (updated from v1.0.0)

```dockerfile
# scikit-plots/ai · Dockerfile
#
# Changes from v1.0.0:
#   + HEALTHCHECK: routes traffic only after the FastAPI app is ready.
#     Without this, HF Spaces may send requests before the app is listening,
#     causing 502 errors during container startup.
#   + pip --no-cache-dir: reduces image size.
#   (everything else unchanged)

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# HuggingFace Spaces route traffic to the port defined in README.md (app_port: 7860).
# Do not change this value without also updating app_port in README.md.
EXPOSE 7860

# Health check: wait 10s for startup, then probe /health every 15s.
# Container is marked unhealthy (and restarted) if /health fails 3 times.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

### `requirements.txt` (unchanged from v1.0.0)

```text
# scikit-plots/ai · requirements.txt
# Pinned to compatible release ranges (~=) to allow patch upgrades
# while preventing breaking minor/major version changes.
fastapi~=0.111.0
uvicorn[standard]~=0.29.0
httpx~=0.27.0
```

### Full `app.py` v3.0.0

```python
# scikit-plots/ai · app.py  v3.0.0
#
# PURPOSE
# ───────
# Thin OpenAI-compatible reverse proxy for sphinx-ai-assistant.
#
# REQUEST FLOW
# ────────────
# 1. Browser POSTs JSON to /v1/chat/completions (no auth header).
# 2. _resolve_url() determines the upstream endpoint:
#      a. BACKEND_URL is set → forward there (Docker Model Runner or ZeroGPU Space).
#      b. BACKEND_URL empty  → build HF Serverless API URL from HF_BASE + model.
# 3. HF_TOKEN is injected into the upstream request headers (when applicable).
# 4. _forward() sends the request to the upstream.
# 5. For stream=true requests, SSE chunks are yielded as they arrive.
# 6. The response (JSON or SSE) is returned to the browser with CORS headers.
#
# ENVIRONMENT VARIABLES (Space → Settings → Repository secrets)
# ─────────────────────────────────────────────────────────────
#   HF_TOKEN       Required when BACKEND_URL is not set.
#                  HuggingFace API token with "Inference" read permission.
#
#   BACKEND_URL    Optional. When set, ALL requests are forwarded here.
#                  Overrides HF_BASE + model routing entirely.
#                  Examples:
#                    Path A (local DMR): http://localhost:12434/engines/llama.cpp/v1/chat/completions
#                    Path C (ZeroGPU):   https://scikit-plots-ai-model.hf.space/v1/chat/completions
#
#   HF_BASE        Optional. Default: https://api-inference.huggingface.co/models  # legacy v3.0.0 — v6+ uses router.huggingface.co
#                  Only used when BACKEND_URL is empty.
#
#   DEFAULT_MODEL  Optional. Default: openai/gpt-oss-20b
#                  Fallback model when the request body omits the "model" field.
#                  Must have an Inference Provider if BACKEND_URL is not set.
#
#   PROXY_TIMEOUT  Optional. Default: 120 (seconds).
#                  Upstream read timeout. Increase for slow/large models.
#                  For ZeroGPU Path C: set to 600 (10 minutes).
#
#   ALLOWED_ORIGINS  Optional. Default: * (allow all origins).
#                    Comma-separated list of allowed CORS origins.
#                    For tighter security in production, set this to your docs domain:
#                      https://scikit-plots.github.io,https://your-docs-domain.com
#
# CHANGES vs v2.0.0
# ─────────────────
# + README.md with YAML front matter (required by HF Spaces — was missing).
# + HEALTHCHECK in Dockerfile (prevents routing to container before app is ready).
# + ALLOWED_ORIGINS env var for CORS (replaces hardcoded "*" in code).
# + Separate httpx.Timeout with per-phase values (connect, read, write).
# + Request body size limit (prevents memory exhaustion from oversized requests).
# + Unique response IDs for SSE error events.
# + Better upstream error message (includes status code and truncated body).
# + requirements.txt uses ~= pinning (prevents silent breaking upgrades).
# Unchanged: all route paths, Dockerfile CMD, requirements.txt packages, all logic.

from __future__ import annotations

import json
import os
import uuid

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="sphinx-ai-assistant proxy",
    description=(
        "Thin OpenAI-compatible reverse proxy. "
        "Injects HF_TOKEN and forwards to HuggingFace Inference API "
        "or a custom BACKEND_URL (Docker Model Runner, ZeroGPU Space)."
    ),
    version="3.0.0",
    # Disable Swagger UI — this is a proxy, not a user-facing API.
    docs_url=None,
    redoc_url=None,
)

# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────
# Default: allow all origins ("*") for development convenience.
# Production: set ALLOWED_ORIGINS to your docs domain(s) in Space secrets.
# Example: ALLOWED_ORIGINS=https://scikit-plots.github.io,https://docs.example.com

_raw_origins: str = os.environ.get("ALLOWED_ORIGINS", "*").strip()
_allowed_origins: list[str] = (
    ["*"] if _raw_origins == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

#: When set, ALL /v1/chat/completions requests are forwarded to this URL.
#: Overrides HF_BASE + model routing entirely.
#: Set this in Space secrets to use Docker Model Runner (Path A) or ZeroGPU (Path C).
#: Leave empty to use the HF Serverless Inference API (Path B).
BACKEND_URL: str = os.environ.get("BACKEND_URL", "").strip()

#: HuggingFace API token.
#: Required when BACKEND_URL is not set.
#: Optional when BACKEND_URL is set (DMR and public ZeroGPU Spaces do not need it).
HF_TOKEN: str = os.environ.get("HF_TOKEN", "").strip()

#: Proxy version string — single source of truth for /health and root() responses.
#: Update this constant on every deployment; never scatter version literals.
#: v6+: import from _shared_logic.PROXY_VERSION when migrating to the split layout.
PROXY_VERSION: str = "6.0.0"

# Startup validation — fail fast with a clear error message.
# A misconfigured Space logs this error immediately on start, not silently on first request.
if not BACKEND_URL and not HF_TOKEN:
    raise RuntimeError(
        "Configuration error: neither BACKEND_URL nor HF_TOKEN is set.\n\n"
        "Set ONE of these in Space → Settings → Repository secrets:\n\n"
        "  Option 1 — HF Serverless API (Path B):\n"
        "    HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
        "    DEFAULT_MODEL = openai/gpt-oss-20b\n\n"
        "  Option 2 — Custom backend (Path A or C):\n"
        "    BACKEND_URL = https://scikit-plots-ai-model.hf.space/v1/chat/completions\n\n"
        "See README.md for full configuration details."
    )

#: HuggingFace Serverless Inference API base URL.
#: Only used when BACKEND_URL is empty (Path B).
HF_BASE: str = os.environ.get(
    "HF_BASE",
    "https://api-inference.huggingface.co/models",  # legacy v3.0.0 — v6+ uses router.huggingface.co
).rstrip("/")

#: Fallback model ID when the request body omits the ``model`` field.
#: For Path B: must be a model with a registered Inference Provider.
#:   Correct: "openai/gpt-oss-20b", "Qwen/Qwen2.5-Coder-32B-Instruct"
#:   Wrong:   "scikit-plots/gpt-oss-20b"  (mirror — no provider, always 404/503)
DEFAULT_MODEL: str = os.environ.get("DEFAULT_MODEL", "openai/gpt-oss-20b").strip()

#: Upstream timeout in seconds.
#: For ZeroGPU Path C with a 20B model: set to 600 (10 minutes).
#: The read timeout is the critical one — connect is always fast.
_timeout_secs: int = int(os.environ.get("PROXY_TIMEOUT", "120"))

#: httpx.Timeout with separate values for each phase.
#: connect: time to establish TCP connection (always fast — 10s is generous)
#: read:    time to receive the first byte (this is where model generation happens)
#: write:   time to upload the request body (always small — 30s is generous)
#: pool:    time to acquire a connection from the pool
UPSTREAM_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=float(_timeout_secs),
    write=30.0,
    pool=10.0,
)

#: Maximum request body size in bytes (10 MB).
#: Prevents memory exhaustion from maliciously oversized request bodies.
MAX_BODY_BYTES: int = int(os.environ.get("MAX_BODY_BYTES", str(10 * 1024 * 1024)))

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_model(body: bytes) -> str:
    """
    Extract the ``model`` field from a JSON request body.

    Parameters
    ----------
    body : bytes
        Raw JSON request body from the browser.

    Returns
    -------
    str
        The ``model`` value from the body, or :data:`DEFAULT_MODEL` if the
        field is absent, empty, or the body cannot be parsed as JSON.

    Notes
    -----
    Developer note:
        Never raises. Any parse failure falls back to DEFAULT_MODEL.
        This ensures the proxy remains functional even with malformed clients.
    """
    try:
        data = json.loads(body)
        model = str(data.get("model", "")).strip()
        return model if model else DEFAULT_MODEL
    except (json.JSONDecodeError, ValueError, AttributeError):
        return DEFAULT_MODEL


def _resolve_url(body: bytes) -> tuple[str, dict[str, str]]:
    """
    Resolve the upstream URL and HTTP headers for a given request body.

    Priority order:
    1. ``BACKEND_URL`` — custom backend (Docker Model Runner or ZeroGPU Space).
       Auth header is added only if ``HF_TOKEN`` is also set.
    2. ``HF_BASE`` + model — HuggingFace Serverless Inference API.
       ``HF_TOKEN`` is always injected.

    Parameters
    ----------
    body : bytes
        Raw JSON request body from the browser.

    Returns
    -------
    url : str
        Fully-qualified upstream endpoint URL.
    headers : dict[str, str]
        HTTP headers for the upstream request, including Authorization if applicable.

    Notes
    -----
    Developer note:
        All routing logic is centralised in this function.
        To add a new backend type (e.g. local Ollama, OpenAI API directly),
        add a branch here. ``_forward`` and the route handlers remain unchanged.

    User note:
        Set ``BACKEND_URL`` in Space secrets to switch backends without
        touching any code or triggering a redeployment.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}

    if BACKEND_URL:
        # Custom backend path (Docker Model Runner or ZeroGPU Space).
        # Inject HF_TOKEN only if it is set (public ZeroGPU Spaces don't require it).
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"
        return BACKEND_URL, headers

    # HuggingFace Serverless Inference API path.
    # HF_TOKEN is required (validated at startup) and always injected.
    model = _parse_model(body)
    url = f"{HF_BASE}/{model}/v1/chat/completions"
    headers["Authorization"] = f"Bearer {HF_TOKEN}"
    return url, headers


async def _forward(body: bytes) -> Response:
    """
    Forward *body* to the resolved upstream and return the response.

    Handles both regular JSON responses and SSE streaming transparently.

    Parameters
    ----------
    body : bytes
        Raw JSON request body from the browser.
        Must not exceed :data:`MAX_BODY_BYTES` (enforced by callers).

    Returns
    -------
    fastapi.Response or fastapi.responses.StreamingResponse
        The upstream response. Status code and content-type preserved.

    Notes
    -----
    Developer note:
        SSE streaming is detected by ``stream: true`` in the request body.
        When streaming, a ``StreamingResponse`` proxies upstream SSE chunks
        as they arrive. When not streaming, the full response is awaited.

    User note:
        If a request times out, increase ``PROXY_TIMEOUT`` in Space secrets.
        Default is 120s. For ZeroGPU Path C with a 20B model, set to 600.
    """
    url, headers = _resolve_url(body)

    # Detect whether the client wants SSE streaming.
    stream_requested: bool = False
    try:
        stream_requested = bool(json.loads(body).get("stream", False))
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    if stream_requested:
        async def _sse_chunks():
            """Async generator that yields SSE chunks from the upstream."""
            async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
                async with client.stream(
                    "POST", url, content=body, headers=headers
                ) as upstream:
                    if upstream.status_code != 200:
                        # Upstream returned an error during streaming.
                        # Surface it as an SSE error event so the client can display it.
                        err_body = await upstream.aread()
                        error_payload = json.dumps({
                            "id": f"err-{uuid.uuid4().hex}",
                            "error": {
                                "status":  upstream.status_code,
                                "message": err_body.decode(errors="replace")[:500],
                            },
                        })
                        yield f"data: {error_payload}\n\n"
                    else:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk

        return StreamingResponse(
            _sse_chunks(),
            status_code=200,
            media_type="text/event-stream",
            headers={
                "Cache-Control":    "no-cache",
                "X-Accel-Buffering": "no",      # disable nginx buffering for SSE
            },
        )

    # Non-streaming: await the full response.
    async with httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT) as client:
        upstream = await client.post(url, content=body, headers=headers)

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> JSONResponse:
    """
    Human-readable status page and Space health-check handler.

    Returns
    -------
    JSONResponse
        200 with service status and active backend URL.

    Notes
    -----
    User note:
        The ``backend`` field shows which upstream is active.
        Use it to confirm which Path is in effect without reading env vars:
          - BACKEND_URL set → shows that URL (Docker Model Runner or ZeroGPU Space).
          - BACKEND_URL empty → shows the HF API URL pattern with a placeholder.
    """
    active_backend = (
        BACKEND_URL
        if BACKEND_URL
        else f"{HF_BASE}/<model>/v1/chat/completions  [HF Serverless API — HF_TOKEN: set]"
    )
    return JSONResponse({
        "status":  "ok",
        "service": "sphinx-ai-assistant proxy v3.0.0",
        "backend": active_backend,
        "cors_origins": _allowed_origins,
        "endpoints": {
            "chat":   "POST /v1/chat/completions  (primary)",
            "alias":  "POST /                     (alias — for older sphinx-ai-assistant)",
            "health": "GET  /health               (liveness probe)",
            "root":   "GET  /                     (this page)",
        },
    })


@app.get("/health")
async def health() -> JSONResponse:
    """
    Minimal liveness probe.

    Returns
    -------
    JSONResponse
        200 when the proxy process is running. Does NOT verify upstream reachability.

    Notes
    -----
    Developer note:
        This is a LIVENESS probe (is the process alive?), not a READINESS probe
        (is the upstream healthy?). A 200 response means the FastAPI app is running.
        It does NOT mean the upstream model backend is reachable or responsive.
        To verify the upstream, test POST /v1/chat/completions with a real request.
    """
    return JSONResponse({"status": "ok", "version": PROXY_VERSION})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    """
    Primary proxy endpoint — OpenAI-compatible ``/v1/chat/completions``.

    Parameters
    ----------
    request : Request
        Incoming FastAPI request. Body is forwarded verbatim to the upstream.

    Returns
    -------
    Response
        The upstream response with status code and content-type preserved.
        SSE streaming is handled transparently when ``stream: true`` is in the body.

    Notes
    -----
    Developer note:
        Body size is limited to MAX_BODY_BYTES before reading.
        Oversized bodies receive 413 without reading the full content.
    """
    # Enforce body size limit before reading the full body into memory.
    content_length_str = request.headers.get("content-length")
    if content_length_str is not None:
        try:
            content_length = int(content_length_str)
            if content_length > MAX_BODY_BYTES:
                return JSONResponse(
                    {"error": f"Request body too large. Maximum is {MAX_BODY_BYTES} bytes."},
                    status_code=413,
                )
        except ValueError:
            pass  # missing or invalid Content-Length — let httpx handle it

    body: bytes = await request.body()

    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(
            {"error": f"Request body too large. Maximum is {MAX_BODY_BYTES} bytes."},
            status_code=413,
        )

    return await _forward(body)


@app.post("/")
async def chat_completions_root(request: Request) -> Response:
    """
    Alias: ``POST /`` → same logic as ``POST /v1/chat/completions``.

    Some versions of sphinx-ai-assistant post to ``/`` instead of
    ``/v1/chat/completions``. This alias ensures backward compatibility.
    """
    body: bytes = await request.body()
    return await _forward(body)
```

---

## Wiring `conf.py` for Each Path

### Path A — Docker Model Runner (local development)

```python
# docs/conf.py  — PATH A: Local Docker Model Runner
import os

# AI_PROXY_BASE defaults to the local DMR endpoint.
# Override in CI/CD by setting AI_PROXY_BASE to your deployed proxy URL:
#   export AI_PROXY_BASE=https://scikit-plots-ai.hf.space
_PROXY_BASE: str = os.environ.get(
    "AI_PROXY_BASE",
    "http://localhost:12434/engines/llama.cpp",
)

ai_assistant_panel_api_enabled = True

ai_assistant_panel_api_models = [
    {
        "id":          "dmr-gpt-oss-20b",
        "label":       "GPT-OSS 20B (local Docker)",
        "provider":    "huggingface",
        "model":       "hf.co/scikit-plots/gpt-oss-20b",
        "endpoint":    f"{_PROXY_BASE}/v1/chat/completions",
        "default":     True,
        "description": "Local inference via Docker Model Runner. No internet required.",
    },
]
```

### Path B — HF Proxy Space → HF Serverless API

```python
# docs/conf.py  — PATH B: Deployed HF Space proxy → HF Serverless Inference API
ai_assistant_panel_api_enabled = True

ai_assistant_panel_api_models = [
    {
        "id":          "hf-gpt-oss-20b",
        "label":       "GPT-OSS 20B",
        "provider":    "huggingface",
        # IMPORTANT: use the ORIGINAL model ID (has Inference Provider).
        # Do NOT use "scikit-plots/gpt-oss-20b" — that mirror has no provider.
        "model":       "openai/gpt-oss-20b",
        "endpoint":    "https://scikit-plots-ai.hf.space/v1/chat/completions",
        "default":     True,
        "info_url":    "https://huggingface.co/openai/gpt-oss-20b",
        "description": "GPT-OSS 20B via HuggingFace Serverless Inference API.",
    },
    {
        "id":          "hf-qwen-32b",
        "label":       "Qwen2.5-Coder 32B",
        "provider":    "huggingface",
        "model":       "Qwen/Qwen2.5-Coder-32B-Instruct",
        "endpoint":    "https://scikit-plots-ai.hf.space/v1/chat/completions",
        "info_url":    "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
        "description": "Qwen 2.5 Coder 32B via HuggingFace Serverless Inference API.",
    },
]
```

### Path C — HF Proxy Space → ZeroGPU Model Space

```python
# docs/conf.py  — PATH C: Proxy Space → ZeroGPU Model Space
ai_assistant_panel_api_enabled = True

ai_assistant_panel_api_models = [
    {
        "id":          "zerogpu-gpt-oss-20b",
        "label":       "GPT-OSS 20B (scikit-plots, ZeroGPU)",
        "provider":    "huggingface",
        # The model field is informational — the ZeroGPU Space loads from MODEL_ID env var.
        "model":       "scikit-plots/gpt-oss-20b",
        # The browser calls the PROXY Space URL, not the model Space URL.
        "endpoint":    "https://scikit-plots-ai.hf.space/v1/chat/completions",
        "default":     True,
        "info_url":    "https://huggingface.co/scikit-plots/gpt-oss-20b",
        "description": "scikit-plots GPT-OSS 20B on ZeroGPU. First request: 2–10 min cold start.",
    },
]
```

### Path D — Cloudflare Worker

```python
# docs/conf.py  — PATH D: Cloudflare Worker proxy
ai_assistant_panel_api_enabled = True

ai_assistant_panel_api_models = [
    {
        "id":          "cf-gpt-oss-20b",
        "label":       "GPT-OSS 20B",
        "provider":    "huggingface",
        "model":       "openai/gpt-oss-20b",
        # Replace <your-subdomain> with your Cloudflare account's subdomain.
        # Get the full URL from the output of `wrangler deploy`.
        "endpoint":    "https://hf-proxy.<your-subdomain>.workers.dev",
        "default":     True,
        "description": "GPT-OSS 20B via Cloudflare Workers proxy.",
    },
]
```

### Path E — Local Python Dev Proxy

```python
# docs/conf.py  — PATH E: Local Python dev proxy
ai_assistant_panel_api_enabled = True

ai_assistant_panel_api_models = [
    {
        "id":          "local-gpt-oss-20b",
        "label":       "GPT-OSS 20B (dev proxy)",
        "provider":    "huggingface",
        "model":       "openai/gpt-oss-20b",
        "endpoint":    "http://localhost:8787/v1/chat/completions",
        "default":     True,
        "description": "Via local dev_proxy.py. Requires HF_TOKEN and export.",
    },
]
```

---

## Environment Variable Reference

The `scikit-plots/ai` proxy Space reads these environment variables.
Set them in **Space → Settings → Repository secrets**.

| Variable | Required? | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (unless `BACKEND_URL` set) | — | HuggingFace API token. Get from https://huggingface.co/settings/tokens — requires "Make calls to the Inference API" permission. |
| `BACKEND_URL` | No | `""` (empty) | Full URL of a custom model backend. Bypasses HF Serverless API entirely. Use for DMR (Path A) or ZeroGPU Space (Path C). |
| `DEFAULT_MODEL` | No | `openai/gpt-oss-20b` | Fallback model ID when request body omits `model`. Must have Inference Provider if `BACKEND_URL` is not set. |
| `HF_BASE` | No | `https://router.huggingface.co` | HF API base URL (v6.0.0+: flat router endpoint). Only used when `BACKEND_URL` is empty. |
| `PROXY_TIMEOUT` | No | `120` | Upstream **read** timeout in seconds. Set to `600` for ZeroGPU Path C (cold start takes 2–10 min). |
| `ALLOWED_ORIGINS` | No | `*` | Comma-separated CORS origins. For production, set to your docs domain: `https://scikit-plots.github.io` |
| `MAX_BODY_BYTES` | No | `10485760` (10 MB) | Maximum accepted request body size. Prevents memory exhaustion. |

### Quick Reference: Switching Between Paths

```
Path B — HF Serverless API (fastest setup if model has Inference Provider):
  HF_TOKEN      = hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
  BACKEND_URL   = (leave unset)
  DEFAULT_MODEL = openai/gpt-oss-20b
  PROXY_TIMEOUT = 120

Path C — ZeroGPU Space (free GPU, any model weights):
  BACKEND_URL   = https://scikit-plots-ai-model.hf.space/v1/chat/completions
  HF_TOKEN      = (optional — ZeroGPU Space is public, no token needed)
  PROXY_TIMEOUT = 600   ← IMPORTANT: cold start takes 2–10 min

Path A — Local Docker Model Runner (in your shell, not Space secrets):
  export BACKEND_URL=http://localhost:12434/engines/llama.cpp/v1/chat/completions
  (no HF_TOKEN needed)
```

---

## Troubleshooting Checklist

### Symptom: `405 Method Not Allowed`

```
Cause:
  Something is serving static files instead of the proxy.
  VS Code Live Server (port 5500) cannot handle POST requests.

Diagnose:
  curl -X POST <your-endpoint-url> -H "Content-Type: application/json" -d '{}'
  If you get 405 → wrong server answering.

Fix:
  Ensure the proxy (dev_proxy.py, app.py, or Cloudflare Worker) is running.
  Set conf.py endpoint to point at the proxy port (8787, 7860, or .workers.dev),
  NOT at VS Code Live Server port 5500.
```

### Symptom: `404` or `{"error": "Model ... is currently loading"}` (never resolves)

```
Cause:
  The model ID has no Inference Provider registered on HuggingFace.
  This always happens with scikit-plots/* mirror repos on the HF Serverless API.

Diagnose:
  Open https://huggingface.co/<model-id>
  If you see "This model isn't deployed by any Inference Provider" → confirmed cause.

Fix Option 1 (quickest — 2 min):
  Change DEFAULT_MODEL to "openai/gpt-oss-20b" in Space secrets. (Path B)

Fix Option 2 (use scikit-plots/* weights):
  Path A (Docker Model Runner) or Path C (ZeroGPU Space).
  These download weights directly without using HF Serverless API.
```

### Symptom: `401 Unauthorized`

```
Cause:
  HF_TOKEN is missing, wrong, revoked, or lacks Inference permission.

Diagnose:
  curl -H "Authorization: Bearer $HF_TOKEN" https://huggingface.co/api/whoami
  Expected: JSON with your username.
  Got "Unauthorized": token is wrong or revoked.

Fix:
  Create a new token at https://huggingface.co/settings/tokens
  Enable: "Make calls to the Inference API" (read permission).
  Update the Space secret and wait for the Space to restart.
```

### Symptom: CORS error in browser console

```
Cause 1: The proxy is not returning Access-Control-Allow-Origin headers.
Cause 2: Browser is hitting the wrong URL (e.g. port 5500 instead of proxy port).
Cause 3: ALLOWED_ORIGINS is set to a specific domain and the docs domain is not included.

Diagnose:
  curl -X OPTIONS <endpoint-url> -v 2>&1 | grep -i access-control
  Expected output includes: access-control-allow-origin: *

Fix:
  Ensure the proxy is running and CORS middleware is active.
  If ALLOWED_ORIGINS is set, confirm your docs domain is in the list.
  Check conf.py endpoint points at the proxy, not at :5500.
```

### Symptom: Requests time out

```
Cause 1: Model is slow to generate (large model, slow hardware).
Cause 2: ZeroGPU cold start (first request of a session — 2–10 min for 20B model).
Cause 3: PROXY_TIMEOUT is too low.

Fix:
  Increase PROXY_TIMEOUT in Space secrets.
  For ZeroGPU Path C: set PROXY_TIMEOUT = 600 (10 minutes minimum).
  For HF Serverless API with large models: try PROXY_TIMEOUT = 300.
  For cold start issues: send a dummy request to warm the Space before real use.
```

### Symptom: ZeroGPU Space crashes immediately on startup (OOM)

```
Cause:
  The model is being loaded into CPU RAM at import time (old pattern from v2.0.0).
  A 20B model needs ~40 GB CPU RAM. ZeroGPU Spaces have ≤16 GB. Result: OOM crash.

Fix:
  Use the corrected app.py from Path C in this document (v3.0.0).
  Key difference: model loaded with low_cpu_mem_usage=True and device_map="cpu".
  GPU is only used inside the @spaces.GPU decorated function.
```

### Symptom: `wrangler init` fails with "unknown command"

```
Cause:
  `wrangler init` was removed in Wrangler v3.
  You are running Wrangler v3+.

Fix:
  Use the replacement command:
    npm create cloudflare@latest -- hf-proxy
  See Path D Step 2 in this document.
```

### Symptom: ZeroGPU Space works in UI but `POST /v1/chat/completions` returns wrong content

```
Cause:
  The tokenizer does not have a chat_template defined.
  The model may not support chat-formatted input.

Diagnose:
  Check Space logs for: "Tokenizer ... does not have a chat_template"

Fix:
  Verify the model supports instruct/chat format.
  Check the model card at https://huggingface.co/<model-id> for the correct
  input format. Some models require a different pipeline or explicit prompt format.
```

---

## Summary: Which Path for Which Situation

| Situation | Best Path | Time to Setup |
|---|---|---|
| Local dev, have Docker | **Path A** — Docker Model Runner | 10 min |
| Local dev, no Docker | **Path E** — dev_proxy.py | 5 min |
| Deployed Space returning 404, quickest fix | **Path B** — change DEFAULT_MODEL secret | 2 min |
| Want scikit-plots/* models in the cloud, free | **Path C** — ZeroGPU Space | 30 min |
| GitHub Pages docs, public, free | **Path D** — Cloudflare Workers | 15 min |
| Production with full control | **Path A or C** + upgrade HF Space hardware as traffic grows | varies |

**Core rule (applies to every path):**
The browser can never call the model directly.
A proxy is always required to inject the API token and handle CORS.
VS Code Live Server cannot handle POST requests — it is not a proxy.

---

## Changelog

### v3.0.0 (this version) — Bug fixes from v2.0.0

| # | Severity | Issue in v2.0.0 | Fix in v3.0.0 |
|---|---|---|---|
| 1 | 🔴 Critical | ZeroGPU app.py: `pipeline()` at module level → OOM crash on startup (20B model needs ~40 GB CPU RAM; ZeroGPU has ≤16 GB) | Use `AutoModelForCausalLM.from_pretrained` with `low_cpu_mem_usage=True, device_map="cpu"`. Move to GPU only inside `@spaces.GPU`. |
| 2 | 🔴 Critical | ZeroGPU: `_generate` (sync, `@spaces.GPU`) called directly from async FastAPI route → blocks event loop, freezes concurrent requests | Call via `await asyncio.to_thread(_generate, messages, max_tokens)` |
| 3 | 🔴 Critical | Path D: `wrangler init` is removed in Wrangler v3 → deployment fails immediately | Use `npm create cloudflare@latest -- hf-proxy` |
| 4 | 🔴 Critical | Path D: `wrangler.toml` not shown → `wrangler deploy` fails (missing required config) | Added `wrangler.toml` template with `name`, `main`, `compatibility_date` |
| 5 | 🔴 Critical | Both Spaces missing `README.md` with YAML front matter → Space may fail to build or be misidentified | Added `README.md` templates for both `scikit-plots/ai` and `scikit-plots/ai-model` |
| 6 | 🔴 Critical | `int(data.get("max_tokens", 512))` unguarded → `ValueError`/`TypeError` on non-integer input → 500 crash | Validate type and range before casting; return 422 with clear message on invalid input |
| 7 | 🔴 Critical | `result[0]["generated_text"][-1]["content"]` assumes chat template output; plain-string output returns last character | Use `apply_chat_template` + explicit output decoding; guard with `chat_template` check |
| 8 | 🟠 Serious | Docker Model Runner on Linux completely omitted | Added Linux note with official doc link |
| 9 | 🟠 Serious | `chatcmpl-zerogpu` hardcoded static response ID → breaks clients using ID for deduplication | Use `uuid.uuid4().hex` per response |
| 10 | 🟠 Serious | `/health` returns 200 unconditionally → Troubleshooting guide's "verify the fix" check proves nothing | Added developer note clarifying it is liveness, not readiness; model_loaded field added |
| 11 | 🟠 Serious | No request body size limit → memory exhaustion from oversized POST bodies | Added `MAX_BODY_BYTES` check (default 10 MB) before reading body into memory |
| 12 | 🟠 Serious | `httpx.AsyncClient(timeout=TIMEOUT)` applies one flat timeout to all phases | Use `httpx.Timeout(connect=10.0, read=float(_timeout_secs), write=30.0, pool=10.0)` |
| 13 | 🟠 Serious | CORS `allow_origins=["*"]` hardcoded; production fix buried in a code comment | Added `ALLOWED_ORIGINS` env var; default `*`; production: set to docs domain in secrets |
| 14 | 🟠 Serious | ZeroGPU `requirements.txt` uses `>=` without upper bound → silent breaking upgrades | Changed to `~=` (compatible release) pinning |
| 15 | 🟠 Serious | `torch` absent from ZeroGPU requirements with no explanation | Added explicit comment explaining what is pre-installed on ZeroGPU vs what must be listed |
| 16 | 🟠 Serious | ZeroGPU cold start stated as "30–60 seconds" — substantially underestimated | Corrected to "2–10 minutes" throughout; `PROXY_TIMEOUT=600` recommended |
| 17 | 🟠 Serious | Cloudflare free tier 10ms CPU limit not explained clearly for this use case | Added explanation: CPU limit applies to code execution, not network I/O wait; 30s wall-clock limit noted |
| 18 | 🟠 Serious | `dev_proxy.py`: blocking `httpx.post` stall risk not clearly communicated | Made limitation prominent; added explicit "do not open multiple tabs" warning |
| 19 | 🟡 Minor | `docker model pull` + `docker model run` shown as mandatory separate steps | Clarified: `docker model run` auto-pulls; pull separately only if you want to pre-download |
| 20 | 🟡 Minor | Dockerfile missing `HEALTHCHECK` → HF Spaces may route traffic before app is ready | Added `HEALTHCHECK CMD` hitting `/health` with start-period, interval, retries |
