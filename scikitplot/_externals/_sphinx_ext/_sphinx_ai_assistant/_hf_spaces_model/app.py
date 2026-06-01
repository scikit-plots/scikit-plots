# scikit-plots/ai-model  ·  app.py  v4.0.0
#
# PURPOSE
# ───────
# Dual-mode Space that downloads model weights and serves them via an
# OpenAI-compatible REST endpoint:
#
#     POST /v1/chat/completions
#
# Supports two hardware tiers transparently:
#
#   ZeroGPU Spaces  — @spaces.GPU scope; model moves CPU ↔ CUDA per request.
#   CPU basic Space — inference runs on CPU; spaces is not required.
#
# The proxy Space (scikit-plots/ai) calls this endpoint via its
# BACKEND_URL environment variable.
#
# Browsers should NEVER call this Space directly — all requests go
# through the proxy which handles CORS, token injection, and SSE
# streaming.
#
# DESIGN PRINCIPLES
# ─────────────────
# + HuggingFace Spaces lifecycle compatibility (CPU basic + ZeroGPU)
# + Explicit hardware-tier detection; no silent fallback to wrong device
# + ZeroGPU-safe model lifecycle when spaces is available
# + Explicit configuration validation
# + Async-safe inference dispatch
# + Deterministic request validation
# + Exception-safe GPU cleanup
# + Long-term maintainability
# + Minimal hidden behavior
#
# CRITICAL: SINGLE-WORKER REQUIREMENT
# ────────────────────────────────────
# This Space MUST run with a single uvicorn worker.
# The model (7B params, bfloat16) consumes ~14 GB of RAM.
# The HuggingFace free CPU basic tier provides 16 GB total RAM.
#
# Two workers x 14 GB = 28 GB → OOM → the OS kills the second process
# with a clean exit code (0), which HuggingFace reports as "runtime error".
#
# The ``if __name__ == "__main__"`` block at the bottom calls
# ``uvicorn.run(app, ...)`` with a single worker (the default when passing
# an app object rather than an import string).  HuggingFace Spaces with
# sdk: gradio executes ``python app.py`` directly, so this block runs
# and the process stays alive serving requests.
#
# CRITICAL: WHY THE PROCESS WAS EXITING (v2.2.0 → v3.0.0 root cause)
# ─────────────────────────────────────────────────────────────────────
# v2.2.0 exported ``app = _GradioApp.create_app(_gradio_ui)`` at module
# level but never called ``uvicorn.run()`` or ``demo.launch()``.
#
# On ZeroGPU Spaces, HuggingFace's internal ZeroGPU session manager
# hosted the ASGI app automatically.  After migrating to CPU basic,
# HuggingFace runs ``python app.py`` directly.  The module-level code
# completed (logging "initialized successfully") and the Python process
# fell off the end of the script and exited with code 0.  HuggingFace
# then restarted the process ~11 seconds later (the constant-gap restart
# timer visible in both old and new logs), producing two identical
# initialization sequences followed by exit code 0 → "runtime error".
#
# Fix: add ``if __name__ == "__main__": uvicorn.run(app, ...)`` at the
# bottom.  The process now stays alive serving requests.
#
# CRITICAL: SPACES / ZEROGPU IMPORT GUARD
# ────────────────────────────────────────
# ``spaces`` is pre-installed on ZeroGPU Spaces; it is NOT available on
# CPU basic.  The try/except ImportError at the spaces import site sets
# ``_ZEROGPU = False`` when running on CPU basic, enabling CPU-only
# inference without any code changes.
#
# CRITICAL: ZEROGPU ARCHITECTURE REQUIREMENT (when _ZEROGPU = True)
# ──────────────────────────────────────────────────────────────────
# On HuggingFace Spaces with sdk: gradio, ZeroGPU hooks attach to the
# Gradio server lifecycle. Gradio MUST be the ASGI root application.
#
# CORRECT (v2.x / v3.x):
#   Gradio (gr.Blocks) is the ASGI root.
#   REST routes are registered on Gradio's internal FastAPI instance
#   via gradio.routes.App.create_app(demo).
#   @spaces.GPU is active on _generate (applied conditionally).
#
# WRONG (v1.x):
#   FastAPI was the ASGI root; Gradio was a child via gr.mount_gradio_app.
#   @spaces.GPU was commented out.
#   ZeroGPU hooks never activated.
#
# CRITICAL: MODEL LIFECYCLE PATTERN
# ──────────────────────────────────
# CORRECT:
#   * Tokenizer loaded on CPU at first inference request.
#   * Model loaded on CPU with low_cpu_mem_usage=True.
#   * On ZeroGPU: model moved to GPU ONLY inside @spaces.GPU scope.
#   * On CPU basic: model stays on CPU throughout.
#   * Model returned to CPU in finally block after GPU inference.
#   * torch.cuda.empty_cache() called only when CUDA is available.
#   * _MODEL_LOCK serialises all model device transitions.
#
# WRONG:
#   * pipeline(... device_map="auto") at module level.
#   * model.to("cuda") outside @spaces.GPU scope.
#   * Holding GPU between requests.
#   * Blocking asyncio event loop with synchronous inference.
#   * Concurrent model.cuda() / model.cpu() without a lock.
#
# ASSEMBLY DIAGRAM (v3.0.0)
# ─────────────────────────
#
#   HuggingFace Spaces
#       └── app                          ← exported ASGI application
#             └── Gradio (_gradio_ui)    ← ASGI root (ZeroGPU-compatible)
#                   ├── GET  /           ← Gradio test UI (developer only)
#                   ├── GET  /health     ← liveness probe
#                   └── POST /v1/chat/completions
#
# ENVIRONMENT VARIABLES
# ─────────────────────
# MODEL_ID
#     Model weights to load.
#
# ALLOWED_ORIGINS
#     Comma-separated CORS origins.
#
# MAX_BODY_BYTES
#     Maximum accepted request size.
#
# CHANGES v2.0.0 → v2.1.0
# ─────────────────────────
# [CRITICAL] Add _MODEL_LOCK (threading.Lock) to serialise all model
#            device transitions (cuda/cpu) across concurrent inference
#            calls. Without this, concurrent @spaces.GPU activations
#            can corrupt model device state.
#
# [CRITICAL] Explicit GPU tensor cleanup (del input_ids, output_ids,
#            new_token_ids) in the success path before _model.cpu() and
#            torch.cuda.empty_cache(). Ensures VRAM is fully released
#            before the ZeroGPU scope exits.
#
# [HIGH]     Add `except RuntimeError: raise` to _generate exception
#            chain so that RuntimeErrors (including the empty-response
#            guard below) are not accidentally double-wrapped.
#
# [HIGH]     Guard against empty model output: raise RuntimeError if
#            the decoded string is empty after skip_special_tokens.
#
# [MEDIUM]   temperature and top_p are now configurable from the
#            request body (REST) and from sliders (Gradio UI).
#            Defaults: temperature=0.7, top_p=1.0.
#            temperature=0.0 → greedy decoding (do_sample=False).
#
# [MEDIUM]   Log the requested model field from the request body for
#            proxy-routing diagnostics.
#
# [MEDIUM]   Fix chat_completions docstring: JSONResponse error cases
#            moved from the incorrect Raises section to Notes, because
#            they are returned values, not raised exceptions.
#
# [LOW]      _parse_request_body and _build_completion_response now
#            carry precise dict[str, Any] return type annotations.
#
# [LOW]      system_fingerprint field added to completion response for
#            improved OpenAI SDK compatibility.
#
# [LOW]      Explicit allow_credentials=False in CORS middleware.
#
# [DOC]      Prominent single-worker warning added to module header
#            (see above) explaining the double-startup / exit-0 OOM
#            failure mode observed in the container log.
#
# CHANGES v2.1.0 → v2.2.0
# ─────────────────────────
# ROOT CAUSE: Container logs showed two "Starting ... initialization..."
# sequences separated by ~14 seconds (first at 20:29:15, second at
# 20:30:06). Each eager model load consumes ~14 GB RAM. Two concurrent
# loads exceed the ZeroGPU 16 GB hard limit → OS SIGKILL → exit 0 →
# HuggingFace reports "runtime error". Gradio 6.x or ZeroGPU session
# management can spawn a second Python worker process under certain
# conditions even when sdk: gradio should default to single-worker.
#
# [CRITICAL] Lazy model loading: _model is now None at module import
#            and is loaded exactly once on the first inference request
#            via _ensure_model_loaded(). Secondary processes that never
#            receive an inference request never load the model, so
#            RAM stays under 16 GB. If both workers receive requests
#            simultaneously _INIT_LOCK serialises the load within each
#            process; cross-process OOM is prevented by ensuring only
#            one process handles requests (configure GRADIO_NUM_WORKERS=1
#            in Space secrets as an additional guard).
#
# [CRITICAL] Add _INIT_LOCK (threading.Lock) as a dedicated one-time
#            initialisation guard, separate from _MODEL_LOCK which
#            serialises GPU device transitions. The two locks have
#            disjoint scopes and are never held simultaneously, so
#            there is no deadlock risk.
#
# [CRITICAL] Add _model_is_loaded (threading.Event) for a lock-free
#            fast path in _ensure_model_loaded() and to expose model
#            readiness in the /health endpoint.
#
# [CRITICAL] _ensure_model_loaded() must be called by callers BEFORE
#            the @spaces.GPU scope so that model loading (a CPU-only
#            operation) does not consume ZeroGPU quota. Updated callers:
#            _generate_async (REST path) and _gradio_respond (UI path).
#
# [HIGH]     Add _VERSION: Final[str] = "2.2.0" module-level constant.
#            Eliminates the hardcoded "2.1.0" literal in health() and
#            the startup summary log.
#
# [HIGH]     Add _SYSTEM_FINGERPRINT: Final[str] computed once from
#            MODEL_ID at module load. Eliminates repeated string
#            transformation on every call to _build_completion_response().
#
# [HIGH]     Add _VALID_ROLES: Final[frozenset] and role validation in
#            _validate_messages(). Unknown roles previously produced a
#            cryptic chat-template error; they now produce a clear 400
#            with the exact invalid role name.
#
# [MEDIUM]   health() now includes model_ready: bool reflecting
#            _model_is_loaded.is_set() so the proxy can distinguish
#            "server up" from "model loaded and ready for inference".
#
# CHANGES v2.2.0 → v3.0.0
# ─────────────────────────
# ROOT CAUSE: After migrating from ZeroGPU to CPU basic (free tier),
# the Python process exited immediately after module initialization
# because no server was started.  HuggingFace restarts the process
# ~11 seconds later (confirmed constant-gap restart timer in logs),
# producing two identical "initialized successfully" sequences followed
# by exit code 0 → "runtime error".  Additionally, ``import spaces``
# raises ImportError on CPU basic because the ZeroGPU library is only
# pre-installed on GPU Spaces.
#
# [CRITICAL] Add server entry point: ``if __name__ == "__main__":
#            uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning")``.
#            HuggingFace Spaces with sdk: gradio executes ``python app.py``
#            directly.  This block keeps the process alive serving requests
#            and is the single source of truth for the server lifecycle.
#            Passing the app object (not an import string) enforces
#            single-worker mode — no forking, no second model load.
#
# [CRITICAL] Guard ``import spaces`` with try/except ImportError.
#            Sets ``_ZEROGPU: Final[bool] = False`` on CPU basic.
#            All GPU-specific code paths are gated on ``_ZEROGPU`` and
#            ``torch.cuda.is_available()``, ensuring a clean fall-through
#            to CPU-only inference with no runtime errors.
#
# [CRITICAL] Add ``_ZEROGPU: Final[bool]`` — single source of truth for
#            hardware-tier detection.  Derived from spaces import success.
#
# [CRITICAL] Add ``_DEVICE: Final[str]`` — "cuda" on ZeroGPU with CUDA
#            present, "cpu" otherwise.  Used uniformly throughout
#            _generate to select the correct device path.
#
# [CRITICAL] Refactor ``_generate`` for dual-mode operation.  Remove the
#            ``@spaces.GPU(duration=120)`` decorator from the function
#            definition; apply it conditionally after the function body
#            (``if _ZEROGPU: _generate = spaces.GPU(duration=120)(_generate)``).
#            Inside the function, all CUDA calls (model.cuda(), tensor.cuda(),
#            model.cpu(), empty_cache()) are guarded by ``_DEVICE == "cuda"``.
#            On CPU basic the inference path is free of any CUDA calls.
#
# [HIGH]     Update _VERSION to "3.0.0".  Major version bump reflects
#            dual-mode hardware support and server entry-point addition.
#
# [MEDIUM]   Inference log prefix updated to "GPU" or "CPU" based on
#            _DEVICE to make per-request hardware tier immediately visible
#            in container logs.
#
# CHANGES v3.0.0 → v4.0.0
# ─────────────────────────
# ROOT CAUSE 1: Gradio GET / crash under Python 3.13 + Gradio 5.x.
# ``gradio/routes.py:673`` calls ``app.get_blocks().config``.  Under
# this runtime combination ``blocks.config`` evaluates to ``None``
# (a known library incompatibility), causing:
#
#     jinja2.exceptions.UndefinedError: 'None' has no attribute 'get'
#
# This crash only affects the developer GET / test UI; the REST API
# (``/health``, ``/v1/chat/completions``) is unaffected.  The crash
# fires on every page load, flooding the container log with full
# tracebacks.
#
# [CRITICAL] Introduce ``_GradioRootFix`` — a thin ASGI wrapper that
#            intercepts ``GET /`` at the ASGI protocol level, BEFORE any
#            Gradio routing, and returns a static HTML developer UI page.
#            All other paths (``/health``, ``/v1/chat/completions``,
#            Gradio internal routes) are forwarded unchanged to the inner
#            Gradio ASGI app.
#
#            Architecture change:
#              ``_app_inner``: internal Gradio ASGI app (``_GradioApp``).
#                              All FastAPI routes and CORS middleware
#                              remain on ``_app_inner``.
#              ``app``:        exported ASGI application.
#                              Now a ``_GradioRootFix`` wrapper around
#                              ``_app_inner``.  HuggingFace Spaces and
#                              ``uvicorn.run`` receive this wrapper.
#
#            No Gradio internals are modified; no fragile route removal
#            or monkey-patching is required.
#
# ROOT CAUSE 2: Missing ``attention_mask`` in ``_model.generate()``.
# Transformers emits:
#
#     The attention mask is not set and cannot be inferred from input
#     because pad token is same as eos token.
#
# The Qwen2.5 tokenizer sets ``pad_token_id == eos_token_id``.
# Transformers cannot auto-infer the mask in this case and warns of
# potential unexpected behavior.
#
# [HIGH] Pass explicit ``attention_mask`` to ``_model.generate()``.
#        Since ``apply_chat_template`` produces a single fully-real
#        token sequence (no padding), the correct mask is all-ones:
#        ``torch.ones_like(input_ids)``.  The mask is moved to CUDA
#        alongside ``input_ids`` on the ZeroGPU path and deleted in
#        the same ``del`` statement as ``input_ids`` to free memory
#        before ``_model.cpu()`` / ``torch.cuda.empty_cache()``.
#
# [HIGH]  Update ``_VERSION`` to ``"4.0.0"``.  Major version bump
#         reflects the ASGI architecture change and attention-mask fix.
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Dual-mode model Space for scikit-plots AI endpoint.

Downloads and runs model weights on HuggingFace ZeroGPU (GPU) or
CPU basic (CPU), exposing an OpenAI-compatible REST endpoint consumed
by the proxy Space.

Notes
-----
Developer note
    Gradio (gr.Blocks) is the ASGI root application. Custom REST routes
    (``/health``, ``/v1/chat/completions``) are registered on Gradio's
    internal FastAPI instance after ``App.create_app(demo)`` is called.
    This is the only architecture that activates ZeroGPU on HuggingFace
    Spaces with ``sdk: gradio``.

    HuggingFace Spaces exports the ``app`` variable. It must be the
    Gradio-rooted ASGI application returned by ``App.create_app``.

    The server is started by ``uvicorn.run(app, ...)`` in the
    ``if __name__ == "__main__"`` block at the bottom of this file.
    HuggingFace Spaces with ``sdk: gradio`` executes ``python app.py``
    directly, so this block runs and keeps the process alive.

    **Hardware tier detection**: ``_ZEROGPU`` is set to ``True`` only
    when the ``spaces`` package imports successfully (pre-installed on
    ZeroGPU Spaces, absent on CPU basic).  ``_DEVICE`` is ``"cuda"``
    on ZeroGPU with CUDA present, and ``"cpu"`` everywhere else.

    **Model loading is lazy**: ``_model`` is ``None`` at import time
    and is loaded on the first inference request by ``_ensure_model_loaded()``.
    This prevents OOM when a secondary process (restart probe or health
    check) starts the module but issues no inference request.

    Two locks are used with strictly disjoint scopes (never held
    simultaneously, no deadlock risk):

    * ``_INIT_LOCK`` — guards the one-time model initialisation inside
      ``_ensure_model_loaded()``. Held only during CPU-side model loading,
      never inside ``@spaces.GPU``.

    * ``_MODEL_LOCK`` — serialises ``_model.cuda()`` and ``_model.cpu()``
      transitions during GPU inference. Held only inside ``@spaces.GPU``
      scope (when ``_ZEROGPU = True``). On CPU basic, serialises CPU
      inference to prevent concurrent model access.

    ``_ensure_model_loaded()`` must be called by callers **before** the
    ``@spaces.GPU`` scope (when active) so that model loading (a CPU-only
    operation) does not consume ZeroGPU GPU quota.

User note
    The Gradio UI at ``/`` is for manual testing only.
    Production traffic routes through the proxy Space.
    The first request after a cold start may take minutes while
    the model downloads and loads to CPU (then optionally to GPU).
    CPU basic inference (7B model) is significantly slower than GPU —
    allow 3-10 minutes per response depending on output length.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import traceback
import uuid
from typing import Any, Final

import gradio as gr  # type: ignore[]
import torch  # type: ignore[import-untyped]
from fastapi import Request  # FastAPI is a Gradio dependency; no extra install needed.
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from gradio.routes import (  # type: ignore[]
    App as _GradioApp,  # Gradio's internal FastAPI subclass
)
from starlette.types import ASGIApp, Receive, Scope, Send
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[]

# ─────────────────────────────────────────────────────────────────────────────
# ZeroGPU detection
# ─────────────────────────────────────────────────────────────────────────────
# ``spaces`` is pre-installed on ZeroGPU Spaces; it is NOT available on
# CPU basic hardware.  The try/except below makes this module work on
# both platforms without any configuration change:
#
#   ZeroGPU Spaces (T4 GPU):   spaces imports OK → _ZEROGPU = True
#   CPU basic (free tier):     ImportError caught → _ZEROGPU = False
#
# All GPU-specific code paths (model.cuda(), input.cuda(), empty_cache(),
# @spaces.GPU decorator) are guarded by ``_ZEROGPU`` or ``_DEVICE == "cuda"``.

try:
    import spaces  # type: ignore[]  # ZeroGPU — pre-installed on GPU Spaces only

    _ZEROGPU: Final[bool] = True
except ImportError:
    spaces = None  # type: ignore[]  # Not available on CPU basic — handled below
    _ZEROGPU: Final[bool] = False

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
)

logger = logging.getLogger(__name__)

logger.info("Starting scikit-plots ai-model Space initialization...")


# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_int(
    value: str | None,
    default: int,
) -> int:
    """
    Return int(value) or default.

    Parameters
    ----------
    value : str or None
        Input value.

    default : int
        Fallback integer.

    Returns
    -------
    int
        Parsed integer or fallback.
    """
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _require_non_empty_env(
    value: str,
    env_name: str,
) -> str:
    """
    Validate non-empty environment value.

    Parameters
    ----------
    value : str
        Candidate value.

    env_name : str
        Environment variable name.

    Returns
    -------
    str
        Stripped validated value.

    Raises
    ------
    RuntimeError
        If empty after stripping.
    """
    cleaned = value.strip()

    if not cleaned:
        raise RuntimeError(
            f"{env_name} environment variable is empty. "
            "Configure it in HuggingFace Space secrets."
        )

    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_ID: Final[str] = "scikit-plots/Qwen2.5-Coder-7B-Instruct"

MODEL_ID: Final[str] = _require_non_empty_env(
    os.environ.get(
        "MODEL_ID",
        DEFAULT_MODEL_ID,
    ),
    "MODEL_ID",
)

_MAX_NEW_TOKENS_FLOOR: Final[int] = 1
_MAX_NEW_TOKENS_CEIL: Final[int] = 4096
_MAX_NEW_TOKENS_DEFAULT: Final[int] = 512

# Generation defaults — match OpenAI API defaults where applicable.
_DEFAULT_TEMPERATURE: Final[float] = 0.7
_DEFAULT_TOP_P: Final[float] = 1.0

DEFAULT_MAX_BODY_BYTES: Final[int] = 10 * 1024 * 1024

MAX_BODY_BYTES: Final[int] = _safe_int(
    os.environ.get("MAX_BODY_BYTES"),
    DEFAULT_MAX_BODY_BYTES,
)

_raw_cors: str = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://scikit-plots-ai.hf.space",
).strip()

CORS_ORIGINS: Final[list[str]] = (
    ["*"]
    if _raw_cors == "*"
    else [origin.strip() for origin in _raw_cors.split(",") if origin.strip()]
)

logger.info(
    "Configuration loaded | MODEL_ID=%s | MAX_BODY_BYTES=%s | CORS=%s | ZeroGPU=%s",
    MODEL_ID,
    MAX_BODY_BYTES,
    CORS_ORIGINS,
    _ZEROGPU,
)

# ─────────────────────────────────────────────────────────────────────────────
# Derived constants
# ─────────────────────────────────────────────────────────────────────────────
# Computed once from immutable configuration; never recomputed at request time.

_VERSION: Final[str] = "4.0.0"
"""Service version string. Single source of truth for health() and startup log."""

_DEVICE: Final[str] = "cuda" if (_ZEROGPU and torch.cuda.is_available()) else "cpu"
"""
Inference device for this process.

``"cuda"`` on ZeroGPU Spaces when CUDA is present inside the
``@spaces.GPU`` scope.  ``"cpu"`` on CPU basic or when CUDA is absent.
All CUDA calls inside ``_generate`` are guarded by ``_DEVICE == "cuda"``.
"""

_VALID_ROLES: Final[frozenset[str]] = frozenset({"system", "user", "assistant"})
"""Accepted OpenAI message roles for the Qwen2.5 chat template.

Unknown roles produce a clear 400 validation error instead of a cryptic
chat-template exception buried inside _generate().
"""

_SYSTEM_FINGERPRINT: Final[str] = "fp-" + MODEL_ID.lower().replace("/", "-").replace(
    ".", "-"
).replace("_", "-")
"""Pre-computed OpenAI-compatible system_fingerprint derived from MODEL_ID.

Computed at module load to avoid repeated string transformation on every
call to _build_completion_response().
"""


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _clamp_max_tokens(
    value: object,
) -> int:
    """
    Clamp generation token count within configured bounds.

    Parameters
    ----------
    value : object
        Requested max token count.

    Returns
    -------
    int
        Validated token count clamped to
        [``_MAX_NEW_TOKENS_FLOOR``, ``_MAX_NEW_TOKENS_CEIL``].

    Raises
    ------
    ValueError
        If conversion to int fails.
    """
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"max_tokens must be an integer, got {value!r}") from exc

    return max(
        _MAX_NEW_TOKENS_FLOOR,
        min(parsed, _MAX_NEW_TOKENS_CEIL),
    )


def _validate_messages(
    messages: object,
) -> list[dict[str, str]]:
    """
    Validate OpenAI-style messages payload.

    Parameters
    ----------
    messages : object
        Candidate message list.

    Returns
    -------
    list of dict[str, str]
        Validated messages, each with ``"role"`` and ``"content"`` keys.

    Raises
    ------
    ValueError
        On invalid structure, type, empty list, or unknown role value.

    Notes
    -----
    Developer note
        Role values are checked against ``_VALID_ROLES``.  An unknown
        role previously surfaced as a cryptic ``jinja2`` template error
        deep inside ``_generate``; it now produces a clear 400 response
        at the validation boundary.
    """
    if not isinstance(messages, list):
        raise ValueError("messages must be a list.")  # noqa: TRY004

    if not messages:
        raise ValueError("messages must not be empty.")

    validated: list[dict[str, str]] = []

    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            raise ValueError(f"messages[{index}] must be object.")  # noqa: TRY004

        role = item.get("role")
        content = item.get("content")

        if not isinstance(role, str):
            raise ValueError(f"messages[{index}].role must be string.")  # noqa: TRY004

        if role not in _VALID_ROLES:
            raise ValueError(
                f"messages[{index}].role {role!r} is not valid. "
                f"Must be one of {sorted(_VALID_ROLES)}."
            )

        if not isinstance(content, str):
            raise ValueError(  # noqa: TRY004
                f"messages[{index}].content must be string."
            )

        validated.append(
            {
                "role": role,
                "content": content,
            }
        )

    return validated


def _validate_temperature(
    value: object,
) -> float:
    """
    Validate and return a generation temperature value.

    Parameters
    ----------
    value : object
        Candidate temperature.

    Returns
    -------
    float
        Validated temperature in [0.0, 2.0].

    Raises
    ------
    ValueError
        If conversion fails or value is out of range.

    Notes
    -----
    Developer note
        ``temperature=0.0`` selects greedy decoding (``do_sample=False``).
        The upper bound 2.0 matches the OpenAI API specification.

    References
    ----------
    .. [1] OpenAI API reference: temperature parameter
           https://platform.openai.com/docs/api-reference/chat/create#temperature
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"temperature must be a number, got {value!r}") from exc

    if not (0.0 <= parsed <= 2.0):  # noqa: PLR2004
        raise ValueError(f"temperature must be in [0.0, 2.0], got {parsed!r}")

    return parsed


def _validate_top_p(
    value: object,
) -> float:
    """
    Validate and return a nucleus-sampling top_p value.

    Parameters
    ----------
    value : object
        Candidate top_p.

    Returns
    -------
    float
        Validated top_p in (0.0, 1.0].

    Raises
    ------
    ValueError
        If conversion fails or value is out of range.

    Notes
    -----
    Developer note
        ``top_p=1.0`` effectively disables nucleus sampling.
        OpenAI recommends altering temperature or top_p but not both.

    References
    ----------
    .. [1] OpenAI API reference: top_p parameter
           https://platform.openai.com/docs/api-reference/chat/create#top_p
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"top_p must be a number, got {value!r}") from exc

    if not (0.0 < parsed <= 1.0):
        raise ValueError(f"top_p must be in (0.0, 1.0], got {parsed!r}")

    return parsed


logger.info("Validation helpers initialized successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Model lock
# ─────────────────────────────────────────────────────────────────────────────
# Serialises all _model.cuda() / _model.cpu() transitions and CPU inference.
#
# On ZeroGPU: a single _model object must not be moved to different devices
# by two threads simultaneously. @spaces.GPU does not prevent concurrent calls
# by itself (the Gradio queue or multiple in-flight async requests can
# dispatch _generate from multiple threads at the same time).
#
# On CPU basic: serialises concurrent CPU inference calls to prevent
# simultaneous access to the shared _model object from multiple threads.
#
# Holding _MODEL_LOCK for the duration of the entire inference is correct
# and safe: we are single-model, single-device (GPU or CPU).

_MODEL_LOCK: Final[threading.Lock] = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Initialization lock and readiness event
# ─────────────────────────────────────────────────────────────────────────────
# _INIT_LOCK       — guards the one-time model initialisation inside
#                    _ensure_model_loaded().  Held only during CPU-side
#                    loading, never inside @spaces.GPU.  Separate from
#                    _MODEL_LOCK which serialises inference.
#                    The two locks have strictly disjoint scopes and are
#                    never held simultaneously: no deadlock risk.
#
# _model_is_loaded — threading.Event set exactly once after a successful
#                    load.  Provides a lock-free fast path on every
#                    subsequent call to _ensure_model_loaded() and
#                    exposes model readiness in /health.

_INIT_LOCK: Final[threading.Lock] = threading.Lock()
_model_is_loaded: Final[threading.Event] = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
# Both are None at module import; loaded exactly once on the first
# inference request via _ensure_model_loaded().  This prevents OOM when
# a restart probe or health-check process starts the module but issues no
# inference request — a process that loads no model stays well within the
# 16 GB RAM limit on CPU basic.
#
# Never call .to("cuda") or device_map="auto" at module level —
# CUDA is not available outside @spaces.GPU on ZeroGPU Spaces,
# and is absent entirely on CPU basic.

_tokenizer: AutoTokenizer | None = None
_model: AutoModelForCausalLM | None = None


def _ensure_model_loaded() -> None:
    """
    Load tokenizer and model exactly once; no-op on subsequent calls.

    Uses double-checked locking (``_INIT_LOCK``) to guarantee that
    tokenizer and model loading occur at most once across all threads in
    the process.  After the first successful load, all subsequent calls
    return immediately via a lock-free check on ``_model_is_loaded``.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If ``AutoTokenizer.from_pretrained`` or
        ``AutoModelForCausalLM.from_pretrained`` raises.  If the
        tokenizer loads but the model fails, ``_model_is_loaded`` is
        never set so the next call retries the full sequence from
        the tokenizer step.

    Notes
    -----
    Developer note
        Must be called by callers **before** the ``@spaces.GPU`` scope
        (when ``_ZEROGPU = True``) so that model loading (a CPU-only
        operation) does not consume ZeroGPU GPU quota.

        From sync callers (e.g. ``_gradio_respond``): call directly.

        From async callers (e.g. ``_generate_async``,
        ``chat_completions``): call via
        ``await asyncio.to_thread(_ensure_model_loaded)`` to prevent
        blocking the asyncio event loop during the first load.

        Lock scope: ``_INIT_LOCK`` is held only during CPU-side loading,
        never inside ``@spaces.GPU``.  ``_MODEL_LOCK`` serialises
        inference inside ``_generate``.  The two locks have strictly
        disjoint scopes — no deadlock risk.

    User note
        The first inference request after a cold start may take several
        minutes while the model downloads (~14 GB) and loads to CPU.
        On CPU basic, subsequent requests are also slow (2-5 tokens/s).
        On ZeroGPU, GPU inference is significantly faster after the
        initial load.
    """
    # Fast path — lock-free check on the threading.Event.
    if _model_is_loaded.is_set():
        return

    with _INIT_LOCK:
        # Double-checked locking: re-test inside the mutex in case
        # another thread completed loading between the fast-path check
        # above and lock acquisition.
        if _model_is_loaded.is_set():
            return

        global _tokenizer, _model  # noqa: PLW0603

        logger.info("Loading tokenizer for MODEL_ID=%s", MODEL_ID)

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        logger.info("Tokenizer loaded successfully.")

        logger.info(
            "Loading model on CPU (low_cpu_mem_usage=True, torch_dtype=bfloat16)..."
        )

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )

        logger.info("Model loaded on CPU successfully.")

        # Set the event last, only after both loads succeed.
        # Any exception above leaves _model_is_loaded unset so the
        # next request retries the full load sequence.
        _model_is_loaded.set()


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────
# Device dispatch:
#
#   ZeroGPU (_ZEROGPU = True, _DEVICE = "cuda"):
#       @spaces.GPU scope allocates a GPU for the duration of _generate.
#       Model moves CPU → CUDA at entry; CUDA → CPU in finally.
#       VRAM fully released after every request.
#       _MODEL_LOCK held for the full inference duration (cuda → generate → cpu).
#
#   CPU basic (_ZEROGPU = False, _DEVICE = "cpu"):
#       Model stays on CPU throughout. No device moves.
#       @spaces.GPU decorator is NOT applied.
#       _MODEL_LOCK held for the full inference duration.
#       Inference is slower (2-5 tokens/s) but functionally identical.
#
# This function is called from both:
#   - Gradio event handlers (direct sync call via _gradio_respond)
#   - FastAPI route handlers (via asyncio.to_thread in _generate_async)


def _generate(  # noqa: PLR0912
    messages: list[dict[str, str]],
    max_new_tokens: int = _MAX_NEW_TOKENS_DEFAULT,
    temperature: float = _DEFAULT_TEMPERATURE,
    top_p: float = _DEFAULT_TOP_P,
) -> str:
    """
    Run inference on the configured device (GPU or CPU).

    Parameters
    ----------
    messages : list of dict[str, str]
        OpenAI chat messages.

    max_new_tokens : int, default=512
        Maximum generated tokens.

    temperature : float, default=0.7
        Sampling temperature in [0.0, 2.0].
        ``0.0`` selects greedy decoding (do_sample=False).

    top_p : float, default=1.0
        Nucleus sampling cutoff in (0.0, 1.0].
        ``1.0`` disables nucleus sampling.

    Returns
    -------
    str
        Assistant response text.

    Raises
    ------
    ValueError
        On invalid inputs or missing chat template.

    RuntimeError
        On inference failure or empty model output.

    Notes
    -----
    Developer note
        Callers must invoke ``_ensure_model_loaded()`` **before** the
        ``@spaces.GPU`` scope (when ``_ZEROGPU = True``).  A guard at
        the start of this function raises ``RuntimeError`` immediately
        if ``_tokenizer`` or ``_model`` is ``None``.

        **ZeroGPU path** (``_DEVICE == "cuda"``):
        GPU is acquired automatically by ``@spaces.GPU`` (applied
        conditionally after this function's definition).
        ``_MODEL_LOCK`` is held for the entire inference duration
        (cuda → generate → cpu) to prevent concurrent device transitions.
        GPU tensors are explicitly deleted in the success path before
        ``_model.cpu()`` and ``torch.cuda.empty_cache()`` to ensure
        VRAM is fully reclaimed before the ``@spaces.GPU`` scope exits.

        **CPU basic path** (``_DEVICE == "cpu"``):
        No device moves — the model is already on CPU.
        ``_MODEL_LOCK`` is held for the inference duration to serialise
        concurrent requests against the shared ``_model`` object.
        No CUDA calls are made; ``torch.cuda.empty_cache()`` is skipped.

        ``finally`` block ensures resource cleanup even if inference
        raises.  On the GPU path, the inner ``try/except`` around
        ``_model.cpu()`` logs and absorbs a potential CPU-move failure
        so that the original inference exception is not masked.

        This function is intentionally synchronous.  Async routes call
        it via ``_generate_async`` which wraps it with
        ``asyncio.to_thread``.  Gradio event handlers call it directly
        because Gradio dispatches handlers in its own thread pool,
        outside the asyncio event loop.

    User note
        Do not call this function directly from async code.
        Use ``_generate_async`` from FastAPI routes.
        On CPU basic, expect 2-10 minutes per response for a 7B model.
    """
    # Guard: callers must invoke _ensure_model_loaded() before inference.
    # This check makes the contract explicit and produces a clear
    # RuntimeError instead of an AttributeError on None.
    if _tokenizer is None or _model is None:
        raise RuntimeError(
            "_ensure_model_loaded() must be called by the caller "
            "before entering inference. "
            "This is a programming error, not a user error."
        )

    validated_messages = _validate_messages(messages)
    max_new_tokens = _clamp_max_tokens(max_new_tokens)

    if not getattr(_tokenizer, "chat_template", None):
        raise ValueError(f"Tokenizer for {MODEL_ID!r} does not define chat_template.")

    logger.info(
        "%s inference starting | "
        "messages=%d | "
        "max_new_tokens=%d | "
        "temperature=%.2f | "
        "top_p=%.2f",
        "GPU" if _DEVICE == "cuda" else "CPU",
        len(validated_messages),
        max_new_tokens,
        temperature,
        top_p,
    )

    with _MODEL_LOCK:
        try:
            # ── ZeroGPU path: move model and inputs to CUDA ───────────────────
            # ── CPU basic path: model is already on CPU; skip CUDA calls ──────
            if _DEVICE == "cuda":
                logger.info("Moving model to GPU...")
                _model.cuda()

            input_ids = _tokenizer.apply_chat_template(
                validated_messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

            # Explicit attention mask: all ones (no padding in a single-sequence
            # prompt produced by apply_chat_template).  Required because
            # Qwen2.5 sets pad_token_id == eos_token_id; transformers cannot
            # auto-infer the mask in that case and emits a runtime warning
            # about unexpected behavior without it.
            attention_mask = torch.ones_like(input_ids)

            if _DEVICE == "cuda":
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()

            logger.info("Generation started.")

            # Build generation kwargs.
            # temperature=0.0 → greedy (do_sample=False, no temperature/top_p).
            # temperature>0.0 → sampling; top_p applied only when < 1.0.
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": _tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = temperature
                if top_p < 1.0:
                    generate_kwargs["top_p"] = top_p

            with torch.no_grad():
                output_ids = _model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **generate_kwargs,
                )

            new_token_ids = output_ids[0][input_ids.shape[-1] :]
            decoded = _tokenizer.decode(
                new_token_ids,
                skip_special_tokens=True,
            )

            # Release tensors before CPU move and cache clear.
            # new_token_ids is a view of output_ids; deleting both
            # drops all references, freeing the underlying storage.
            # attention_mask is also freed here to release VRAM early.
            del input_ids, output_ids, new_token_ids, attention_mask

            if not decoded.strip():
                raise RuntimeError(
                    "Model returned an empty response. Retry or reduce prompt length."
                )

            logger.info("Generation completed successfully.")

            return decoded

        except ValueError:
            raise

        except RuntimeError:
            raise

        except Exception as exc:
            logger.exception("Inference failure.")
            raise RuntimeError(f"Inference failed: {exc}") from exc

        finally:
            logger.info("Releasing inference resources...")

            # ── ZeroGPU path: return model to CPU; clear VRAM ─────────────────
            if _DEVICE == "cuda":
                try:
                    _model.cpu()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed to move model back to CPU. "
                        "VRAM may not be fully released."
                    )
                finally:
                    torch.cuda.empty_cache()

            # ── CPU basic path: no device moves or cache operations needed ─────

            logger.info("Inference resources released.")


# Apply @spaces.GPU on ZeroGPU Spaces for GPU session management.
# On CPU basic (_ZEROGPU = False), _generate is used as-is — no decorator.
# Applying conditionally after the function definition preserves the
# original type signature and docstring while allowing the decorator
# to be optional at runtime.
if _ZEROGPU:
    _generate = spaces.GPU(duration=120)(_generate)  # type: ignore[]


# ─────────────────────────────────────────────────────────────────────────────
# Async wrapper
# ─────────────────────────────────────────────────────────────────────────────
# @spaces.GPU (when active) requires sync execution.
# FastAPI routes are async.
#
# Therefore:
#
#   async route
#       -> asyncio.to_thread()
#           -> sync _generate function (optionally wrapped by @spaces.GPU)
#
# This prevents event-loop blocking.


async def _generate_async(
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float = _DEFAULT_TEMPERATURE,
    top_p: float = _DEFAULT_TOP_P,
) -> str:
    """
    Async wrapper for inference.

    Parameters
    ----------
    messages : list of dict[str, str]
        OpenAI chat messages.

    max_new_tokens : int
        Generation token limit.

    temperature : float, default=0.7
        Sampling temperature forwarded to ``_generate``.

    top_p : float, default=1.0
        Nucleus sampling cutoff forwarded to ``_generate``.

    Returns
    -------
    str
        Generated response text.

    Notes
    -----
    Developer note
        Calls ``_ensure_model_loaded()`` via ``asyncio.to_thread``
        before dispatching ``_generate``, so the CPU-only model load
        does not block the asyncio event loop and does not consume
        ZeroGPU GPU quota (when applicable).  Subsequent calls hit the
        lock-free fast path (``_model_is_loaded.is_set()``) immediately.

        Offloads the synchronous ``_generate`` call to a thread via
        ``asyncio.to_thread`` so the asyncio event loop is not blocked
        during inference (GPU or CPU).

        Must NOT be called from Gradio event handlers — use ``_generate``
        directly from Gradio since it runs in its own thread pool.
    """
    # Load tokenizer and model on first call only (CPU-only operation).
    # Called before asyncio.to_thread(_generate) so loading completes
    # before @spaces.GPU activates (when _ZEROGPU = True) —
    # ZeroGPU GPU quota is not consumed during model loading.
    await asyncio.to_thread(_ensure_model_loaded)

    return await asyncio.to_thread(
        _generate,
        messages,
        max_new_tokens,
        temperature,
        top_p,
    )


logger.info(
    "Inference subsystem initialized | device=%s | ZeroGPU=%s",
    _DEVICE,
    _ZEROGPU,
)


# ─────────────────────────────────────────────────────────────────────────────
# Request helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _read_bounded_body(
    request: Request,
) -> bytes:
    """
    Read raw request body with size enforcement.

    Parameters
    ----------
    request : Request
        Incoming FastAPI request.

    Returns
    -------
    bytes
        Raw request body.

    Raises
    ------
    ValueError
        If body length exceeds ``MAX_BODY_BYTES``.

    Notes
    -----
    Developer note
        Body is read once in full before JSON parsing.
        Enforcing size here prevents unbounded memory growth from
        malformed or adversarial payloads.

    User note
        Maximum body size is controlled by the ``MAX_BODY_BYTES``
        environment variable (default: 10 MiB).
    """
    body = await request.body()

    if len(body) > MAX_BODY_BYTES:
        raise ValueError(
            f"Request body size {len(body):,} bytes "
            f"exceeds maximum of {MAX_BODY_BYTES:,} bytes."
        )

    return body


def _parse_request_body(
    raw: bytes,
) -> dict[str, Any]:
    """
    Decode and parse a UTF-8 JSON request body.

    Parameters
    ----------
    raw : bytes
        Raw body bytes from the request.

    Returns
    -------
    dict[str, Any]
        Parsed JSON payload.

    Raises
    ------
    ValueError
        If UTF-8 decoding or JSON parsing fails.

    Notes
    -----
    Developer note
        Raised ``ValueError`` messages are safe to propagate directly
        into 400 error responses — they contain no internal state.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Request body is not valid UTF-8: {exc}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Request body is not valid JSON: {exc}") from exc


def _count_prompt_tokens(
    messages: list[dict[str, str]],
) -> int:
    """
    Count prompt tokens using the chat template.

    Parameters
    ----------
    messages : list of dict[str, str]
        Validated OpenAI chat messages.

    Returns
    -------
    int
        Number of tokens in the formatted prompt.

    Notes
    -----
    Developer note
        Tokenization runs on CPU with no gradient tracking.
        The resulting tensor is deleted immediately after the count
        is extracted to release memory before ``_generate_async``
        performs its own tokenization.

        This explicit double-tokenization is an accepted trade-off for
        keeping ``_generate``'s return type as ``str`` and avoiding
        interface entanglement between the inference and routing layers.
    """
    prompt_ids = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    count: int = int(prompt_ids.shape[-1])

    del prompt_ids  # Free CPU tensor before inference dispatch.

    return count


def _count_completion_tokens(
    text: str,
) -> int:
    """
    Count completion tokens from decoded output text.

    Parameters
    ----------
    text : str
        Decoded model output string.

    Returns
    -------
    int
        Token count of the completion string.

    Notes
    -----
    Developer note
        ``add_special_tokens=False`` is required here.
        Special tokens are already accounted for in the prompt count.
    """
    return len(
        _tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def _build_completion_response(
    content: str,
    model_id: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict[str, Any]:
    """
    Build an OpenAI-compatible chat completion response payload.

    Parameters
    ----------
    content : str
        Generated assistant response text.

    model_id : str
        Model identifier string, included verbatim in the response.

    prompt_tokens : int
        Token count for the formatted input prompt.

    completion_tokens : int
        Token count for the generated completion.

    Returns
    -------
    dict[str, Any]
        OpenAI-compatible ``chat.completion`` object.

    Notes
    -----
    Developer note
        ``id`` uses a ``chatcmpl-`` prefix followed by a random UUID
        hex string, matching the format used by the OpenAI API.
        ``created`` is a Unix epoch integer, as required by the spec.
        ``finish_reason`` is hardcoded to ``"stop"`` because the
        current ``_generate`` implementation does not expose partial
        stop conditions. Extend this if streaming or early stopping
        is added.
        ``system_fingerprint`` uses the pre-computed module-level
        constant ``_SYSTEM_FINGERPRINT`` (derived from ``MODEL_ID`` at
        import time) to avoid repeated string transformation per call.

    User note
        The returned dict is compatible with OpenAI Python SDK
        response parsing via ``client.chat.completions.create``.

    References
    ----------
    .. [1] OpenAI API reference: Chat completions object
           https://platform.openai.com/docs/api-reference/chat/object
    """
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "system_fingerprint": _SYSTEM_FINGERPRINT,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _error_response(
    message: str,
    error_type: str,
    code: str,
    status_code: int,
) -> JSONResponse:
    """
    Build a structured OpenAI-compatible error response.

    Parameters
    ----------
    message : str
        Human-readable error description.

    error_type : str
        OpenAI error type string
        (e.g. ``"invalid_request_error"``, ``"server_error"``).

    code : str
        Machine-readable error code string
        (e.g. ``"invalid_json"``, ``"inference_error"``).

    status_code : int
        HTTP status code for the response.

    Returns
    -------
    JSONResponse
        Structured error response.

    Notes
    -----
    Developer note
        Error shape mirrors the OpenAI API error envelope so that
        proxy and client code can handle upstream and downstream
        errors with a single code path.

        Internal exception text is never forwarded — only safe static
        strings and validated ``ValueError`` messages appear here.

    References
    ----------
    .. [1] OpenAI API reference: Error codes
           https://platform.openai.com/docs/guides/error-codes
    """
    # Log traceback internally for server-side errors only.
    if error_type == "server_error":
        logger.error(
            "Server error response | code=%s | traceback=\n%s",
            code,
            traceback.format_exc(),
        )

        safe_server_messages: dict[str, str] = {
            "model_load_error": "Model loading failed. Please retry in a few minutes.",
            "inference_error": "Inference failed. Please retry.",
            "internal_error": "An unexpected server error occurred.",
        }

        message = safe_server_messages.get(
            code,
            "An unexpected server error occurred.",
        )

    return JSONResponse(
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            },
        },
        status_code=status_code,
    )


logger.info("Request helpers initialized successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Gradio history normalizer
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_gradio_history(
    history: list,
) -> list[dict[str, str]]:
    """
    Normalize Gradio chat history to OpenAI message format.

    Parameters
    ----------
    history : list
        Gradio chat history in either supported format:

        * **Gradio 4** — ``list[list[str | None]]``
          where each inner list is ``[user_message, assistant_message]``.

        * **Gradio 5+** — ``list[dict]``
          where each dict has ``{"role": str, "content": str}``.

    Returns
    -------
    list of dict[str, str]
        OpenAI-compatible messages with ``"role"`` and ``"content"`` keys.
        Empty or ``None`` turns are silently skipped.

    Notes
    -----
    Developer note
        Defensive normalization is required because Gradio changed the
        default history format between major versions. Handling both
        formats here insulates ``_gradio_respond`` from any Gradio
        upgrade.

        ``None`` assistant messages appear in Gradio 4 when a turn is
        pending; they are deliberately excluded from the output.

    User note
        Converted history is passed directly to ``_generate`` as the
        OpenAI messages list. The current user message is appended
        afterwards by ``_gradio_respond``.
    """
    messages: list[dict[str, str]] = []

    for turn in history:
        # ── Gradio 5+: list of dicts ──────────────────────────────────────────
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content")

            if (
                isinstance(role, str)
                and isinstance(content, str)
                and role.strip()
                and content.strip()
            ):
                messages.append(
                    {
                        "role": role.strip(),
                        "content": content.strip(),
                    }
                )

        # ── Gradio 4: list of [user, assistant] pairs ─────────────────────────
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:  # noqa: PLR2004
            user_msg, assistant_msg = turn

            if user_msg is not None:
                user_text = str(user_msg).strip()
                if user_text:
                    messages.append(
                        {
                            "role": "user",
                            "content": user_text,
                        }
                    )

            if assistant_msg is not None:
                assistant_text = str(assistant_msg).strip()
                if assistant_text:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_text,
                        }
                    )

    return messages


# ─────────────────────────────────────────────────────────────────────────────
# Gradio event handler
# ─────────────────────────────────────────────────────────────────────────────


def _gradio_respond(
    message: str,
    history: list,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """
    Gradio ``ChatInterface`` event handler.

    Parameters
    ----------
    message : str
        Current user message from the chat input box.

    history : list
        Preceding conversation turns from Gradio, in either
        Gradio 4 or Gradio 5+ history format.

    max_new_tokens : int
        Maximum tokens to generate, sourced from the UI slider.

    temperature : float
        Sampling temperature sourced from the UI slider.

    top_p : float
        Nucleus sampling cutoff sourced from the UI slider.

    Returns
    -------
    str
        Model-generated assistant response.

    Raises
    ------
    ValueError
        If ``message`` is empty after stripping.

    RuntimeError
        Propagated from ``_generate`` on inference failure or empty
        model output.

    Notes
    -----
    Developer note
        Calls ``_ensure_model_loaded()`` directly (synchronous) before
        ``_generate`` so the CPU-only model load does not consume
        ZeroGPU GPU quota (when ``_ZEROGPU = True``).  This is correct:
        Gradio dispatches event handlers in its own thread pool, so
        calling a blocking function here does not block the asyncio
        event loop.

        Calls ``_generate`` (sync) directly.
        Must NOT call ``_generate_async`` (async) because Gradio
        dispatches event handlers via its own thread pool, completely
        outside the asyncio event loop.

        Token clamping and message validation are delegated to
        ``_generate`` — no duplicate logic here.

    User note
        The current message is validated then appended to the
        normalized history before being passed to the model.
    """
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Message must be a non-empty string.")

    # Load tokenizer and model on first call only (CPU-only operation).
    # Called before _generate/@spaces.GPU so loading does not consume
    # ZeroGPU GPU quota.  Gradio's thread pool makes this blocking call safe.
    _ensure_model_loaded()

    messages = _normalize_gradio_history(history)

    messages.append(
        {
            "role": "user",
            "content": message.strip(),
        }
    )

    logger.info(
        "Gradio inference | "
        "history_turns=%d | "
        "max_new_tokens=%d | "
        "temperature=%.2f | "
        "top_p=%.2f",
        len(messages) - 1,
        max_new_tokens,
        temperature,
        top_p,
    )

    return _generate(
        messages,
        max_new_tokens,
        temperature,
        top_p,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
# v2.x / v3.x: Gradio is the ASGI ROOT — not a child sub-app mounted on FastAPI.
# This is required for ZeroGPU to activate on HuggingFace Spaces.
#
# The Gradio UI is served at / (root).
# Custom REST routes are added to Gradio's internal FastAPI instance below.

_UI_WARNING = """\
## scikit-plots · model Space — test UI

> ⚠️ **Developer testing only.**
> Production traffic must route through the proxy Space at
> `https://scikit-plots-ai.hf.space`.
> Browsers must never call this Space directly.
"""

_gradio_ui = gr.Blocks(
    title="scikit-plots model endpoint — test UI",
    analytics_enabled=False,
)

with _gradio_ui:
    gr.Markdown(_UI_WARNING)

    gr.ChatInterface(
        fn=_gradio_respond,
        additional_inputs=[
            gr.Slider(
                minimum=_MAX_NEW_TOKENS_FLOOR,
                maximum=_MAX_NEW_TOKENS_CEIL,
                value=_MAX_NEW_TOKENS_DEFAULT,
                step=1,
                label="max_tokens",
                info=(
                    f"Range: {_MAX_NEW_TOKENS_FLOOR}-{_MAX_NEW_TOKENS_CEIL}. "
                    f"Default: {_MAX_NEW_TOKENS_DEFAULT}."
                ),
            ),
            gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=_DEFAULT_TEMPERATURE,
                step=0.05,
                label="temperature",
                info="0.0 = greedy, 0.7 = default, 2.0 = very random.",
            ),
            gr.Slider(
                minimum=0.01,
                maximum=1.0,
                value=_DEFAULT_TOP_P,
                step=0.01,
                label="top_p",
                info="Nucleus sampling cutoff. 1.0 = disabled.",
            ),
        ],
        additional_inputs_accordion="Generation settings",
    )

logger.info("Gradio root UI defined. Will be served at / (root).")


# ─────────────────────────────────────────────────────────────────────────────
# App assembly — HuggingFace Space export
# ─────────────────────────────────────────────────────────────────────────────
# v2.x / v3.x ARCHITECTURE:
#
#   app = _GradioApp.create_app(_gradio_ui)   ← Gradio is ASGI root
#   @app.get/post(...)                        ← routes on Gradio's FastAPI
#   ZeroGPU activates correctly.
#
# ``_GradioApp`` is ``gradio.routes.App``, a FastAPI subclass that Gradio
# uses internally. ``create_app(blocks)`` builds the full ASGI app with all
# Gradio routes pre-registered. We then augment it with our REST routes.
# No separate ``FastAPI()`` instance is needed or created.

_gradio_ui.queue()  # Enable request queue for concurrent scheduling.

_app_inner: _GradioApp = _GradioApp.create_app(_gradio_ui)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Add CORS to Gradio's internal FastAPI, not a standalone FastAPI instance.
# Default: only the proxy Space origin is allowed.
# Local dev: set ALLOWED_ORIGINS=https://scikit-plots-ai.hf.space,http://localhost:7860

_app_inner.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
    ],
    allow_credentials=False,  # This Space does not use credential-bearing requests.
)

logger.info(
    "Gradio ASGI app created. CORS applied. Allowed origins=%s",
    CORS_ORIGINS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────────────────────────────────


@_app_inner.get("/health")
async def health() -> JSONResponse:
    """
    Liveness check endpoint.

    Returns
    -------
    JSONResponse
        HTTP 200 with status metadata.

    Notes
    -----
    Developer note
        Called by the proxy Space and HuggingFace infrastructure to
        confirm the backend is reachable before routing requests.
        Does not perform an inference round-trip.
        ``device`` field exposes the active hardware tier for
        diagnostics: ``"cuda"`` on ZeroGPU, ``"cpu"`` on CPU basic.

    User note
        Returns model identity, service version, and hardware tier.

    Examples
    --------
    >>> # curl http://localhost:7860/health
    ... # {"status": "ok", "model": "...", "version": "4.0.0",
    ... #  "model_ready": true, "device": "cpu"}
    """
    logger.info("GET /health")

    return JSONResponse(
        content={
            "status": "ok",
            "model": MODEL_ID,
            "version": _VERSION,
            "model_ready": _model_is_loaded.is_set(),
            "device": _DEVICE,
        },
        status_code=200,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Chat completions endpoint
# ─────────────────────────────────────────────────────────────────────────────


@_app_inner.post("/v1/chat/completions")
async def chat_completions(  # noqa: PLR0911
    request: Request,
) -> JSONResponse:
    """
    OpenAI-compatible chat completions endpoint.

    Parameters
    ----------
    request : Request
        FastAPI request carrying a JSON body.

    Returns
    -------
    JSONResponse
        HTTP 200 with an OpenAI-compatible completion payload on success.

        HTTP 413 if the body exceeds ``MAX_BODY_BYTES``.

        HTTP 400 if the body is not valid UTF-8 JSON, or if
        ``messages``, ``max_tokens``, ``temperature``, or ``top_p``
        fail validation.

        HTTP 500 on inference failure or unexpected server error.

    Notes
    -----
    Developer note
        Request pipeline:

        1. Read and bound-check raw body bytes (413 guard).
        2. Decode and parse JSON (400 guard).
        3. Extract ``messages``, ``max_tokens``, ``temperature``,
           ``top_p``, and ``model`` fields.
        4. Validate with field-specific validators (400 guard).
        4b. Lazy model load — ``_ensure_model_loaded()`` via
            ``asyncio.to_thread`` (500 on failure).
        5. Count prompt tokens on CPU (no GPU needed).
        6. Dispatch to ``_generate_async`` which offloads to
           the inference function via ``asyncio.to_thread``.
        7. Count completion tokens on CPU from decoded output.
        8. Return structured OpenAI-compatible JSON response.

        Exception hierarchy in inference block:

        * ``ValueError``  → 400 (bad input that slipped past step 4)
        * ``RuntimeError`` → 500 (wrapped inference failure from ``_generate``)
        * ``Exception``   → 500 (unexpected catch-all, never leaks internals)

        The requested ``model`` field is logged for proxy-routing
        diagnostics but does not affect which model is used; this Space
        always serves ``MODEL_ID``.

    User note
        Compatible with the OpenAI Python SDK:

        .. code-block:: python

            import openai

            client = openai.OpenAI(
                base_url="https://<space>.hf.space",
                api_key="unused",
            )
            response = client.chat.completions.create(
                model="any",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                top_p=1.0,
            )
    """
    request_id = uuid.uuid4().hex

    logger.info(
        "POST /v1/chat/completions | request_id=%s",
        request_id,
    )

    # ── 1. Body size guard ────────────────────────────────────────────────────

    try:
        raw_body = await _read_bounded_body(request)
    except ValueError as exc:
        logger.warning(
            "Body size exceeded | request_id=%s | error=%s",
            request_id,
            exc,
        )
        return _error_response(
            message=str(exc),
            error_type="invalid_request_error",
            code="request_too_large",
            status_code=413,
        )

    # ── 2. JSON parse ─────────────────────────────────────────────────────────

    try:
        payload = _parse_request_body(raw_body)
    except ValueError as exc:
        logger.warning(
            "JSON parse error | request_id=%s | error=%s",
            request_id,
            exc,
        )
        return _error_response(
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_json",
            status_code=400,
        )

    # ── 3. Field extraction ───────────────────────────────────────────────────

    messages_raw: object = payload.get("messages")
    max_tokens_raw: object = payload.get(
        "max_tokens",
        _MAX_NEW_TOKENS_DEFAULT,
    )
    temperature_raw: object = payload.get(
        "temperature",
        _DEFAULT_TEMPERATURE,
    )
    top_p_raw: object = payload.get(
        "top_p",
        _DEFAULT_TOP_P,
    )
    # Log requested model for proxy-routing diagnostics only.
    # This Space always serves MODEL_ID regardless of the field value.
    model_requested: object = payload.get("model", MODEL_ID)

    # ── 4. Input validation ───────────────────────────────────────────────────

    try:
        messages = _validate_messages(messages_raw)
        max_new_tokens = _clamp_max_tokens(max_tokens_raw)
        temperature = _validate_temperature(temperature_raw)
        top_p = _validate_top_p(top_p_raw)
    except ValueError as exc:
        logger.warning(
            "Validation error | request_id=%s | error=%s",
            request_id,
            exc,
        )
        return _error_response(
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_value",
            status_code=400,
        )

    logger.info(
        "Dispatching inference | "
        "request_id=%s | "
        "model_requested=%s | "
        "messages=%d | "
        "max_new_tokens=%d | "
        "temperature=%.2f | "
        "top_p=%.2f",
        request_id,
        model_requested,
        len(messages),
        max_new_tokens,
        temperature,
        top_p,
    )

    # ── 4b. Lazy model loading (CPU only, before GPU dispatch) ───────────────
    # _ensure_model_loaded() must complete before _count_prompt_tokens
    # (which needs _tokenizer) and before the @spaces.GPU scope inside
    # _generate (when _ZEROGPU = True).  asyncio.to_thread prevents
    # blocking the event loop on the first load (downloading ~14 GB).

    try:
        await asyncio.to_thread(_ensure_model_loaded)
    except Exception:  # noqa: BLE001
        logger.exception(
            "Model loading failed | request_id=%s",
            request_id,
        )
        return _error_response(
            message=("Model loading failed. Please retry in a few minutes."),
            error_type="server_error",
            code="model_load_error",
            status_code=500,
        )

    # ── 5. Prompt token count (CPU, pre-dispatch) ─────────────────────────────

    prompt_tokens: int = _count_prompt_tokens(messages)

    # ── 6. Inference ──────────────────────────────────────────────────────────

    try:
        content = await _generate_async(
            messages,
            max_new_tokens,
            temperature,
            top_p,
        )

    except ValueError as exc:
        logger.warning(
            "Inference validation error | request_id=%s | error=%s",
            request_id,
            exc,
        )
        return _error_response(
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_value",
            status_code=400,
        )

    except RuntimeError:
        logger.exception(
            "Inference runtime error | request_id=%s",
            request_id,
        )
        return _error_response(
            message="Inference failed. Please retry.",
            error_type="server_error",
            code="inference_error",
            status_code=500,
        )

    except Exception:  # noqa: BLE001
        logger.exception(
            "Unexpected error during inference | request_id=%s",
            request_id,
        )
        return _error_response(
            message="An unexpected server error occurred.",
            error_type="server_error",
            code="internal_error",
            status_code=500,
        )

    # ── 7. Completion token count (CPU, post-dispatch) ────────────────────────

    completion_tokens: int = _count_completion_tokens(content)

    # ── 8. Response assembly ──────────────────────────────────────────────────

    response_body = _build_completion_response(
        content=content,
        model_id=MODEL_ID,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    logger.info(
        "POST /v1/chat/completions completed | "
        "request_id=%s | "
        "prompt_tokens=%d | "
        "completion_tokens=%d | "
        "total_tokens=%d",
        request_id,
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    )

    return JSONResponse(
        content=response_body,
        status_code=200,
    )


logger.info(
    "REST routes registered on Gradio ASGI app: GET /health | POST /v1/chat/completions"
)


# ─────────────────────────────────────────────────────────────────────────────
# Developer UI HTML page (static, served at GET /)
# ─────────────────────────────────────────────────────────────────────────────
# Replaces Gradio's broken Jinja2-rendered root page (Python 3.13 / Gradio 5.x
# incompatibility where ``blocks.config`` is ``None`` at template render time).
# This is the only content change: REST API behaviour is unchanged.

_DEVELOPER_UI_HTML: Final[str] = (
    "<!DOCTYPE html>\n"
    '<html lang="en">\n'
    "<head>\n"
    '  <meta charset="utf-8">\n'
    '  <meta name="viewport" content="width=device-width, initial-scale=1">\n'
    f"  <title>scikit-plots ai-model — test UI</title>\n"
    "  <style>\n"
    "    body{font-family:system-ui,sans-serif;max-width:720px;"
    "margin:40px auto;padding:0 20px;color:#222}\n"
    "    .warn{background:#fff3cd;border:1px solid #ffc107;"
    "border-radius:4px;padding:12px 16px;margin:16px 0}\n"
    "    code{background:#f0f0f0;padding:2px 6px;border-radius:3px;"
    "font-size:.9em}\n"
    "    table{border-collapse:collapse;width:100%;margin-top:16px}\n"
    "    th,td{border:1px solid #ddd;padding:8px 12px;text-align:left}\n"
    "    th{background:#f5f5f5}\n"
    "  </style>\n"
    "</head>\n"
    "<body>\n"
    "  <h1>scikit-plots ai-model — test UI</h1>\n"
    '  <div class="warn">\n'
    "    <strong>&#9888;&#65039; Developer testing only.</strong><br>\n"
    "    Production traffic must route through the proxy Space at\n"
    "    <code>https://scikit-plots-ai.hf.space</code>.\n"
    "    Browsers must never call this Space directly.\n"
    "  </div>\n"
    "  <table>\n"
    "    <tr><th>Method</th><th>Path</th><th>Description</th></tr>\n"
    "    <tr><td>GET</td><td><code>/health</code></td>"
    "<td>Liveness check. Returns model identity, version, and readiness.</td></tr>\n"
    "    <tr><td>POST</td><td><code>/v1/chat/completions</code></td>"
    "<td>OpenAI-compatible chat completions endpoint.</td></tr>\n"
    "  </table>\n"
    f"  <p>Model: <code>{MODEL_ID}</code></p>\n"
    f"  <p>Version: <code>{_VERSION}</code></p>\n"
    "</body>\n"
    "</html>\n"
)


# ─────────────────────────────────────────────────────────────────────────────
# ASGI wrapper — Gradio GET / fix
# ─────────────────────────────────────────────────────────────────────────────
# Root cause: Gradio's internal route for ``GET /`` calls
#   ``app.get_blocks().config``
# and passes the result to a Jinja2 template.  Under Python 3.13 + Gradio 5.x,
# ``blocks.config`` evaluates to ``None``, which makes Jinja2 raise:
#
#   jinja2.exceptions.UndefinedError: 'None' has no attribute 'get'
#
# This crash occurs on every page load of the developer test UI and fills the
# container log with full tracebacks.  The REST API is completely unaffected —
# only the GET / developer page is broken.
#
# Fix strategy: wrap the inner Gradio ASGI app with ``_GradioRootFix``.
# The wrapper intercepts GET / at the ASGI protocol level (before any routing
# or middleware processing by the inner app) and returns a static HTML page.
# All other paths are forwarded unchanged.
#
# This avoids:
#   * Removing or replacing routes from ``_app_inner.router.routes``
#     (fragile, depends on Gradio route ordering internals)
#   * Monkey-patching Gradio's route handler or template context
#   * Overriding ``blocks.config`` with a dummy value
#
# Exported ``app``:
#   _GradioRootFix(_app_inner)
#     ├── GET /                    → static HTML (this wrapper)
#     └── everything else          → _app_inner (Gradio FastAPI)
#           ├── GET  /health       → health()
#           ├── POST /v1/chat/completions → chat_completions()
#           └── Gradio internal routes (websocket, static assets, etc.)


class _GradioRootFix:
    """
    ASGI wrapper that intercepts ``GET /`` before Gradio's broken renderer.

    Gradio's internal route for ``GET /`` crashes under Python 3.13 +
    Gradio 5.x because ``blocks.config`` evaluates to ``None`` at template
    render time, causing ``jinja2.exceptions.UndefinedError``.

    This wrapper intercepts the problematic request at the ASGI level and
    returns a lightweight static HTML developer UI page.  All other requests
    pass through to the inner Gradio application unchanged.

    Parameters
    ----------
    inner : ASGIApp
        Inner Gradio ASGI application (``_app_inner``).  All non-root
        requests are forwarded to this app.

    Notes
    -----
    Developer note
        Intercepting at the ASGI level (before Starlette routing) is the
        only approach that reliably prevents Gradio's GET / handler from
        running, without relying on Gradio internals or route ordering.

        The inner app (``_app_inner``) retains all REST route registrations
        (``/health``, ``/v1/chat/completions``) and handles all non-root
        requests.  CORS middleware added to ``_app_inner`` applies to those
        routes normally.

        The module-level ``app`` export is this wrapper instance.
        HuggingFace Spaces and ``uvicorn.run`` receive the wrapper as the
        ASGI application; its ``__call__`` signature satisfies the ASGI 3.0
        protocol.

    User note
        ``GET /`` returns a static developer information page.  No Gradio
        UI is rendered because of the Python 3.13 + Gradio 5.x
        incompatibility.  The REST API (``/health``,
        ``/v1/chat/completions``) is fully functional.

    See Also
    --------
    _DEVELOPER_UI_HTML : Static HTML content returned for ``GET /``.
    _app_inner : Inner Gradio ASGI application.
    """

    def __init__(self, inner: ASGIApp) -> None:
        self._inner = inner

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        """
        Dispatch an ASGI request.

        Parameters
        ----------
        scope : dict
            ASGI connection scope.

        receive : callable
            ASGI receive channel.

        send : callable
            ASGI send channel.

        Notes
        -----
        Developer note
            Only HTTP ``GET /`` requests are intercepted.  WebSocket,
            lifespan, and all other HTTP paths are forwarded to the inner
            app without modification.
        """
        if (
            scope.get("type") == "http"
            and scope.get("method") == "GET"
            and scope.get("path") == "/"
        ):
            response = HTMLResponse(
                content=_DEVELOPER_UI_HTML,
                status_code=200,
            )
            await response(scope, receive, send)
            return

        await self._inner(scope, receive, send)


# ─────────────────────────────────────────────────────────────────────────────
# Exported ASGI application
# ─────────────────────────────────────────────────────────────────────────────
# ``app`` is the module-level name that HuggingFace Spaces expects.
# It wraps ``_app_inner`` (Gradio) with the GET / fix.

app: _GradioRootFix = _GradioRootFix(_app_inner)

logger.info(
    "ASGI app assembled: "
    "_GradioRootFix(_app_inner). "
    "GET / served by static HTML wrapper. "
    "All other paths forwarded to Gradio inner app."
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup summary
# ─────────────────────────────────────────────────────────────────────────────

logger.info(
    "scikit-plots ai-model Space initialized successfully.\n"
    "  version   : %s\n"
    "  model     : %s\n"
    "  device    : %s\n"
    "  ZeroGPU   : %s\n"
    "  CORS      : %s\n"
    "  max_body  : %s bytes\n"
    "  ASGI root : _GradioRootFix(_app_inner)\n"
    "  routes    : GET /health | POST /v1/chat/completions\n"
    "  test UI   : / (root, static HTML — Gradio renderer bypassed)",
    _VERSION,
    MODEL_ID,
    _DEVICE,
    _ZEROGPU,
    CORS_ORIGINS,
    MAX_BODY_BYTES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Server entry point
# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace Spaces with sdk: gradio executes ``python app.py`` directly
# (``__name__ == "__main__"``).  Without a blocking server call the Python
# process exits immediately after module initialization, causing the
# "runtime error / exit code 0" failure observed in container logs.
#
# ``app`` is now a ``_GradioRootFix`` instance (ASGI wrapper around
# ``_app_inner``).  Passing an app OBJECT (not an import string) to
# ``uvicorn.run`` starts uvicorn in single-process mode.  This is critical:
#
#   * Prevents a second worker process from loading a second 14 GB copy
#     of the model → OOM on 16 GB RAM hardware.
#   * Eliminates the constant-interval restart-probe pattern that was
#     appearing as two identical initialization sequences in the logs.
#
# ``log_level="warning"`` suppresses uvicorn's access log lines; the
# application logger above provides structured per-request logging.
#
# To run locally:
#   python app.py
#   # Server starts at http://0.0.0.0:7860

if __name__ == "__main__":
    import uvicorn  # already in requirements.txt

    uvicorn.run(
        app,
        host="0.0.0.0",  # noqa: S104
        port=7860,
        log_level="warning",  # Application logger handles structured logging.
    )
