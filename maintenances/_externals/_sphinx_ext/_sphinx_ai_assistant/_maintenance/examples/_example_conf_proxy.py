# path_a_conf_snippets.py
#
# conf.py ai_assistant_panel_api_models entries for all 5 proxy paths.
#
# Copy the relevant block into your Sphinx docs/conf.py.
# Only ONE path should be active at a time.
#
# SPDX-License-Identifier: BSD-3-Clause
# Authors: The scikit-plots developers

"""
Sphinx ``conf.py`` snippets wiring the AI assistant panel to each proxy path.

Notes
-----
**User note** — Each path requires a different deployment step before the
browser can reach the model.  Refer to FREE_PROXY_SOLUTIONS.md for the
complete setup procedure for each path.

**Developer note** — The ``AI_PROXY_BASE`` environment variable pattern
allows CI/CD to inject the correct endpoint URL without editing ``conf.py``:

    export AI_PROXY_BASE=https://scikit-plots-ai.hf.space

This means the same ``conf.py`` can serve local dev (Path A / E) and
production (Path B / C / D) by changing one env var.
"""

from __future__ import annotations

import os

# ─────────────────────────────────────────────────────────────────────────────
# Shared base — read from environment so CI/CD can override without edits
# ─────────────────────────────────────────────────────────────────────────────

# Override with the deployed proxy URL in CI/CD / ReadTheDocs environment:
#   export AI_PROXY_BASE=https://scikit-plots-ai.hf.space
_AI_PROXY_BASE: str = os.environ.get("AI_PROXY_BASE", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# PATH A — Local Docker Model Runner (zero cost, no token needed)
# ─────────────────────────────────────────────────────────────────────────────
#
# Prerequisites:
#   1. Docker Desktop ≥ 4.40 (macOS/Windows) or Docker Engine with plugin (Linux).
#   2. Enable Docker Model Runner in Docker Desktop → Settings → Features.
#   3. Run: docker model run hf.co/scikit-plots/Qwen2.5-Coder-7B-Instruct
#      (first run downloads ~15 GB; subsequent runs start immediately)
#
# Works with scikit-plots/* mirror repos — DMR downloads weights directly
# from HF storage (Git LFS), bypassing the HF Serverless Inference API.

_PATH_A_BASE: str = _AI_PROXY_BASE or "http://localhost:12434/engines/llama.cpp"

PATH_A_CONF: dict = {
    "ai_assistant_panel_api_enabled": True,
    "ai_assistant_panel_api_models": [
        {
            "default": True,
            # Unique identifier (alphanumeric + hyphen/underscore, no spaces).
            "id": "Qwen2.5-Coder-7B-Instruct-local-dmr",
            # Label shown in the UI model-picker dropdown.
            "label": "Qwen2.5-Coder-7B-Instruct (Qwen/local Docker)",
            # Provider tag — controls UI accent colour and feedback routing.
            # "huggingface" is correct for any HF-hosted or DMR model.
            "provider": "huggingface",
            # Model ID as Docker Model Runner expects it (hf.co/<org>/<repo>).
            "model": "hf.co/scikit-plots/Qwen2.5-Coder-7B-Instruct",
            # DMR endpoint — consistent with DEFAULT in docker model run output.
            "endpoint": f"{_PATH_A_BASE}/v1/chat/completions",
            "description": (
                "Qwen2.5-Coder-7B-Instruct running locally via Docker Model Runner.  "
                "No internet required after model download."
            ),
        },
    ],
}

# Paste this into conf.py for Path A:
# ai_assistant_panel_api_enabled = True
# ai_assistant_panel_api_models  = PATH_A_CONF["ai_assistant_panel_api_models"]


# ─────────────────────────────────────────────────────────────────────────────
# PATH B — Deployed HF Proxy Space → HF Serverless Inference API
# ─────────────────────────────────────────────────────────────────────────────
#
# Prerequisites:
#   1. scikit-plots/ai Space deployed with path_b/app.py v3.0.0.
#   2. HF_TOKEN set in Space → Settings → Repository secrets.
#   3. DEFAULT_MODEL = Qwen/Qwen2.5-Coder-7B-Instruct (ORIGINAL repo — has Inference Provider).
#      Do NOT use "scikit-plots/Qwen2.5-Coder-7B-Instruct" — mirror has no provider → 404/503.

_PATH_B_BASE: str = _AI_PROXY_BASE or "https://scikit-plots-ai.hf.space"

PATH_B_CONF: dict = {
    "ai_assistant_panel_api_enabled": True,
    # IMPORTANT: original repo — has a registered Inference Provider.
    # Swapping to "scikit-plots/gpt-oss-20b" causes 404/503.
    "ai_assistant_panel_api_models": [
        {
            "id": "gpt-oss-20b-hf",
            "label": "GPT-OSS 20B (OpenAI/HuggingFace)",
            "provider": "huggingface",
            "model": "openai/gpt-oss-20b",
            "endpoint": f"{_PATH_B_BASE}/v1/chat/completions",
            "info_url": "https://huggingface.co/openai/gpt-oss-20b",
            "description": (
                "OpenAI open-source 20B via HuggingFace Inference API — "
                "OpenAI-compat /v1/chat/completions, SSE streaming enabled."
            ),
        },
        {
            "default": True,
            "id": "Qwen2.5-Coder-7B-Instruct-hf",
            "label": "Qwen2.5-Coder-7B-Instruct (Qwen/HuggingFace)",
            "provider": "huggingface",
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "endpoint": f"{_PATH_B_BASE}/v1/chat/completions",
            "info_url": "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
            "description": (
                "Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). "
                "For more:https://github.com/QwenLM"
            ),
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# PATH C — HF Proxy Space → ZeroGPU Model Space
# ─────────────────────────────────────────────────────────────────────────────
#
# Prerequisites:
#   1. scikit-plots/ai-model Space deployed with path_c/app.py.
#   2. scikit-plots/ai Space has:
#      BACKEND_URL = https://scikit-plots-ai-model.hf.space/v1/chat/completions
#      PROXY_TIMEOUT = 600  (cold start takes 2-10 minutes for 20B model)
#
# The browser calls the PROXY Space URL — NOT the model Space URL.
# The proxy forwards to the ZeroGPU Space via BACKEND_URL.

_PATH_C_BASE: str = _AI_PROXY_BASE or "https://scikit-plots-ai.hf.space"

PATH_C_CONF: dict = {
    "ai_assistant_panel_api_enabled": True,
    "ai_assistant_panel_api_models": [
        {
            "default": True,
            "id": "Qwen2.5-Coder-7B-Instruct-skplt",
            "label": "Qwen2.5-Coder-7B-Instruct (scikit-plots/ZeroGPU)",
            "provider": "huggingface",
            # The model field is informational — the ZeroGPU Space loads from MODEL_ID.
            "model": "scikit-plots/Qwen2.5-Coder-7B-Instruct",
            # Browser calls the PROXY Space, not the model Space directly.
            "endpoint": f"{_PATH_C_BASE}/v1/chat/completions",
            "info_url": "https://huggingface.co/scikit-plots/Qwen2.5-Coder-7B-Instruct",
            "description": (
                "scikit-plots Qwen2.5-Coder-7B-Instruct on ZeroGPU (free shared GPU).  "
                "First request may take 2-10 min (cold start).  Subsequent: fast."
            ),
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# PATH D — Cloudflare Worker Proxy
# ─────────────────────────────────────────────────────────────────────────────
#
# Prerequisites:
#   1. npm create cloudflare@latest -- hf-proxy
#   2. Replace src/index.js with path_d/src/index.js.
#   3. wrangler secret put HF_TOKEN
#   4. wrangler deploy
#   5. Note the deployed URL (e.g. https://hf-proxy.<subdomain>.workers.dev).
#
# Replace <your-subdomain> below with your Cloudflare Workers subdomain.

_PATH_D_WORKER_URL: str = _AI_PROXY_BASE or (
    "https://hf-proxy.<your-subdomain>.workers.dev"
)

PATH_D_CONF: dict = {
    "ai_assistant_panel_api_enabled": True,
    "ai_assistant_panel_api_models": [
        {
            "default": True,
            "id": "Qwen2.5-Coder-7B-Instruct-cf",
            "label": "Qwen2.5-Coder-7B-Instruct (Qwen/Cloudflare)",
            "provider": "huggingface",
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            # The Worker URL is the proxy endpoint — no path suffix needed
            # because the Worker handles all POST requests on any path.
            "endpoint": _PATH_D_WORKER_URL,
            "description": (
                "Qwen/Qwen2.5-Coder-7B-Instruct via Cloudflare Workers proxy "
                "(100k req/day free tier)."
            ),
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# PATH E — Local Python Dev Proxy (no Docker required)
# ─────────────────────────────────────────────────────────────────────────────
#
# Prerequisites:
#   1. pip install httpx
#   2. export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   3. python path_e/dev_proxy.py        (Terminal 1)
#   4. make html && python -m http.server 8080 --directory _build/html  (Terminal 2)
#
# IMPORTANT: Use original repo model IDs (openai/*, Qwen/*).
#            Mirror repos (scikit-plots/*) have no Inference Provider → 404/503.

_PATH_E_BASE: str = _AI_PROXY_BASE or "http://localhost:8787"

PATH_E_CONF: dict = {
    "ai_assistant_panel_api_enabled": True,
    "ai_assistant_panel_api_models": [
        {
            "default": True,
            "id": "Qwen2.5-Coder-7B-Instruct-local",
            "label": "Qwen2.5-Coder-7B-Instruct (dev proxy)",
            "provider": "huggingface",
            "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "endpoint": f"{_PATH_E_BASE}/v1/chat/completions",
            "description": (
                "Via local dev_proxy.py.  Requires HF_TOKEN and openai/* model IDs."
            ),
        },
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Active path selection — set AI_PROXY_PATH to pick (default: B)
# ─────────────────────────────────────────────────────────────────────────────

_ACTIVE_PATH: str = os.environ.get("AI_PROXY_PATH", "B").upper().strip()

_PATH_MAP: dict[str, dict] = {
    "A": PATH_A_CONF,
    "B": PATH_B_CONF,
    "C": PATH_C_CONF,
    "D": PATH_D_CONF,
    "E": PATH_E_CONF,
}

if _ACTIVE_PATH not in _PATH_MAP:
    raise ValueError(
        f"AI_PROXY_PATH={_ACTIVE_PATH!r} is not valid.  "
        f"Choose one of: {', '.join(_PATH_MAP)}."
    )

_active_conf: dict = _PATH_MAP[_ACTIVE_PATH]

# ── Paste these two lines into your actual docs/conf.py ──────────────────────
ai_assistant_panel_api_enabled = _active_conf["ai_assistant_panel_api_enabled"]
ai_assistant_panel_api_models = _active_conf["ai_assistant_panel_api_models"]
# ─────────────────────────────────────────────────────────────────────────────
