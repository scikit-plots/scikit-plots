# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/app.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# scikit-plots/ai  ·  _hf_spaces_proxy/app.py  v6.2.0
#
# Thin OpenAI-compatible reverse proxy for sphinx-ai-assistant.
#
# THREE-PATH ROUTING  (evaluated in order)
# ─────────────────────────────────────────
#   Path 1 — BACKEND_URL set  (explicit custom backend override)
#     Forward verbatim to BACKEND_URL.
#     HF_TOKEN injected as Bearer token when also set.
#     Read timeout: PROXY_TIMEOUT (default 600 s).
#
#   Path 2 — Model namespace in HF_SPACES_MODEL_NAMESPACES
#     Model owner matches a custom namespace (default: "scikit-plots").
#     Forward to HF_SPACES_MODEL_URL (the scikit-plots/ai-model HF Space).
#     CPU inference on a 7B model takes 4-5 minutes.
#     Read timeout: PATH2_TIMEOUT (default 600 s).
#
#   Path 3 — HF Serverless Inference API (default fallback)
#     Model has a registered HF Inference Provider (openai/*, Qwen/*, etc.).
#     Build {HF_BASE}/{model}/v1/chat/completions and inject HF_TOKEN.
#     Read timeout: PATH3_TIMEOUT (default 120 s).
#
# WHY PER-PATH TIMEOUTS MATTER  (root cause of the "network error" in v4)
# ─────────────────────────────────────────────────────────────────────────
# v4.0.0 used a single PROXY_TIMEOUT=120 s applied to ALL paths.
# The ai-model Space runs a 7B model on CPU basic hardware.  Cold-start
# inference requires ~50 s tokenizer load + ~50 s model load + ~4.5 min
# generation.  Every request to the ai-model Space timed out at 120 s and
# the browser reported "Sorry, something went wrong: network error".
# v5.0.0 fixes this by:
#   1. Raising DEFAULT_PROXY_TIMEOUT from 120 to 600 s.
#   2. Adding per-path timeouts so Path 3 (fast GPU) stays at 120 s
#      while Path 2 (slow CPU) gets the full 600 s.
#   3. Using httpx per-request timeouts so a single shared client
#      serves both fast and slow paths without interference.
#
# ENVIRONMENT VARIABLES  (Space → Settings → Repository secrets)
# ─────────────────────────────────────────────────────────────
#   HF_TOKEN            Required for Path 3 (read/inference).  Optional for
#                       write operations when HF_WRITE_TOKEN is set.
#   HF_WRITE_TOKEN      Write-scoped token for POST /v1/contribute (dataset
#                       push).  When set, inference uses HF_TOKEN (read-only)
#                       and dataset writes use HF_WRITE_TOKEN — principle of
#                       least privilege.  Falls back to HF_TOKEN when absent
#                       so existing single-token deployments keep working.
#   HF_SPACES_MODEL_URL Path 2 destination URL.
#                       Default: https://scikit-plots-ai-model.hf.space/v1/chat/completions
#   HF_SPACES_MODEL_NAMESPACES  Comma-separated owner namespaces for Path 2.
#                       Default: scikit-plots
#   BACKEND_URL         Path 1 override (all requests go here when set).
#   HF_BASE             HF Serverless API base URL.
#                       Default: https://router.huggingface.co
#   DEFAULT_MODEL       Fallback model when request body omits "model".
#                       Default: scikit-plots/Qwen2.5-Coder-7B-Instruct
#   PROXY_TIMEOUT       Path 1 read timeout in seconds.  Default: 600.
#   PATH2_TIMEOUT       Path 2 read timeout in seconds.  Default: 600.
#   PATH3_TIMEOUT       Path 3 read timeout in seconds.  Default: 120.
#   PROXY_CONNECT_TIMEOUT TCP handshake timeout.  Default: 10.
#   PROXY_WRITE_TIMEOUT   Request body upload timeout.  Default: 30.
#   PROXY_POOL_TIMEOUT    Connection pool acquire timeout.  Default: 10.
#   HF_TOKEN_TYPE       Declare the type of HF_TOKEN so startup validation can
#                       enforce principle-of-least-privilege without network calls.
#                       Accepted values: fine-grained | read | write
#                       When absent the proxy applies a length-based heuristic.
#                       Set to "read" when using a classic read token; set to
#                       "fine-grained" for new-style scoped tokens.  Never set
#                       "write" — that triggers a startup WARNING.
#   HF_WRITE_TOKEN_TYPE Declare the type of HF_WRITE_TOKEN.
#                       Accepted values: fine-grained | write
#                       Set to match the token created in HF Settings → Tokens.
#   ALLOWED_ORIGINS     Comma-separated CORS origins.  Default: *.
#   MAX_BODY_BYTES      Maximum accepted body size.  Default: 10485760.

"""
FastAPI reverse proxy for sphinx-ai-assistant (scikit-plots/ai HF Space).

Routes browser POST requests through three ordered paths with independent
per-path read timeouts:

* **Path 1** — ``BACKEND_URL`` set: explicit custom backend.
* **Path 2** — Model namespace in ``HF_SPACES_MODEL_NAMESPACES``:
  forward to ``HF_SPACES_MODEL_URL`` (the ``scikit-plots/ai-model`` Space,
  CPU inference — read timeout 600 s by default).
* **Path 3** — Default: HF Serverless Inference API (GPU, read timeout 120 s).

Notes
-----
Developer note — per-path timeouts
    ``_resolve_upstream_url`` returns ``(url, headers, read_timeout_s)``.
    ``_forward`` builds an ``httpx.Timeout`` from *read_timeout_s* and
    the shared connect/write/pool values, then passes it **per-request**
    so the shared client never imposes a global ceiling.  This means
    concurrent slow (Path 2) and fast (Path 3) requests never block each
    other.

Developer note — shared HTTP client
    A single :class:`httpx.AsyncClient` is created during lifespan and
    shared across all requests.  It is created with ``timeout=None`` so
    all timeout control lives in each request's own ``httpx.Timeout``
    object.  Streaming uses ``client.stream()`` which closes the response
    body (not the client) on context exit, so concurrent SSE requests are
    safe.

Developer note — explicit error handling
    ``_forward`` catches ``httpx.ReadTimeout``, ``httpx.ConnectTimeout``,
    and ``httpx.RequestError`` individually and returns meaningful JSON
    errors with appropriate HTTP status codes so the browser widget can
    display a useful message instead of a generic "network error".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time as _time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# _shared_logic.py and _dataset_schema.py must live alongside this file.
try:
    from ._dataset_schema import (  # type: ignore[import]
        CONSENT_VERSION_ENABLED,
        RESERVED_CONSENT_VERSION,
        normalize_contribution_record,
        normalize_feedback_record,
    )
except Exception:  # noqa: BLE001
    from _dataset_schema import (  # type: ignore[import]
        CONSENT_VERSION_ENABLED,
        RESERVED_CONSENT_VERSION,
        normalize_contribution_record,
        normalize_feedback_record,
    )

try:
    from ._shared_logic import (  # type: ignore[import]
        DEFAULT_HF_BASE,
        DEFAULT_HF_SPACES_MODEL_NAMESPACES,
        DEFAULT_HF_SPACES_MODEL_URL,
        DEFAULT_MAX_BODY_BYTES,
        DEFAULT_MODEL,
        DEFAULT_PATH2_READ_TIMEOUT,
        DEFAULT_PATH3_READ_TIMEOUT,
        DEFAULT_PROXY_TIMEOUT,
        PROXY_VERSION,
        _classify_token_type,
        _mask_ip,
        _RedactingFilter,
        _resolve_upstream_url,
        _safe_float,
        _safe_int,
        _token_log_fragment,
        _validate_env,
        _validate_token_config,
    )
except Exception:  # noqa: BLE001
    from _shared_logic import (  # type: ignore[import]
        DEFAULT_HF_BASE,
        DEFAULT_HF_SPACES_MODEL_NAMESPACES,
        DEFAULT_HF_SPACES_MODEL_URL,
        DEFAULT_MAX_BODY_BYTES,
        DEFAULT_MODEL,
        DEFAULT_PATH2_READ_TIMEOUT,
        DEFAULT_PATH3_READ_TIMEOUT,
        DEFAULT_PROXY_TIMEOUT,
        PROXY_VERSION,
        _classify_token_type,
        _mask_ip,
        _RedactingFilter,
        _resolve_upstream_url,
        _safe_float,
        _safe_int,
        _token_log_fragment,
        _validate_env,
        _validate_token_config,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — placed before configuration so they are available at module scope
# ─────────────────────────────────────────────────────────────────────────────


def _client_ip(request: Request) -> str:
    """Extract the real client IP from the request.

    Parameters
    ----------
    request : fastapi.Request
        Incoming HTTP request.

    Returns
    -------
    str
        Best-effort client IP string; ``"unknown"`` when unavailable.

    Notes
    -----
    Developer: HF Spaces sits behind a proxy so ``request.client.host``
    is the proxy IP, not the user IP.  ``X-Forwarded-For`` is the correct
    source — take the FIRST value only (leftmost = original client;
    rightmost values can be spoofed by intermediaries).
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


class _StructuredFormatter(logging.Formatter):
    """Emit one JSON object per log record to stdout.

    Parameters
    ----------
    *args, **kwargs
        Forwarded to :class:`logging.Formatter`.

    Notes
    -----
    Developer: JSON format is required for machine-parseable log ingestion
    (HF Spaces log export, Datadog, etc.).  Text-format lines require regex
    in log queries; JSON fields are natively queryable.

    The ``exc_info`` key is omitted entirely when no exception is attached
    so log consumers do not need to handle a null field on every record.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_handler = logging.StreamHandler()
_handler.setFormatter(_StructuredFormatter())
_handler.addFilter(_RedactingFilter())  # scrub tokens/IPs before emission
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — read once at module import, never at request time
# ─────────────────────────────────────────────────────────────────────────────

#: Explicit custom backend URL (Path 1).
BACKEND_URL: str = os.environ.get("BACKEND_URL", "").strip()

#: HuggingFace API token for **inference** (read-only).
#: Required for Path 3 (HF Serverless API); optional for Path 2.
#: Use a read-only / ``inference-api``-scoped fine-grained token here.
#: This token is forwarded to upstream model backends and must not carry
#: unnecessary write permission — see ``HF_WRITE_TOKEN`` below.
HF_TOKEN: str = os.environ.get("HF_TOKEN", "").strip()

#: HuggingFace write-scoped token for dataset push operations.
#:
#: Separation of concerns (principle of least privilege):
#:
#: * ``HF_TOKEN``       — read / inference-api scope.
#:                        Injected into every upstream inference request.
#: * ``HF_WRITE_TOKEN`` — write scope, scoped to the training dataset repo only.
#:                        Used exclusively by ``HfApi.create_commit`` inside
#:                        ``POST /v1/contribute``.  Never forwarded to models.
#:
#: How to create each token on HuggingFace
#: ----------------------------------------
#: Go to ``https://huggingface.co/settings/tokens`` → **New token**:
#:
#: Read token (``HF_TOKEN``)
#:   Type: Fine-grained
#:   Permissions: ``Make calls to the serverless Inference API``
#:   (Do NOT grant any repo write access)
#:
#: Write token (``HF_WRITE_TOKEN``)
#:   Type: Fine-grained
#:   Permissions: ``Write access`` scoped to
#:   ``scikit-plots/ai-assistant-contributions`` (or your dataset repo)
#:   (Do NOT grant Inference API access — not needed for writes)
#:
#: Classic tokens (legacy, less secure than fine-grained):
#:   ``read`` role  → ``HF_TOKEN``       (inference only)
#:   ``write`` role → ``HF_WRITE_TOKEN`` (dataset push)
#:
#: Fallback behaviour
#: ------------------
#: When ``HF_WRITE_TOKEN`` is not set, ``HF_DATASET_TOKEN`` falls back to
#: ``HF_TOKEN`` so single-token deployments keep working without changes.
HF_WRITE_TOKEN: str = os.environ.get("HF_WRITE_TOKEN", "").strip()

#: Effective token used **only** for HuggingFace dataset write operations.
#: Resolves to ``HF_WRITE_TOKEN`` when set; falls back to ``HF_TOKEN`` when not.
#: Never forward this token to model inference backends.
HF_DATASET_TOKEN: str = HF_WRITE_TOKEN or HF_TOKEN

#: Classified type for HF_TOKEN — used by startup validation and log output.
#:
#: Source priority:
#:   1. Explicit ``HF_TOKEN_TYPE`` env var (``fine-grained`` | ``read`` | ``write``).
#:   2. Length-based heuristic: tokens ≥ 52 chars are classified ``"fine-grained"``;
#:      shorter classic tokens that cannot be distinguished return ``"unknown"``.
#:
#: Set ``HF_TOKEN_TYPE=read`` or ``HF_TOKEN_TYPE=fine-grained`` in Space secrets
#: to enable accurate least-privilege startup warnings.
HF_TOKEN_TYPE: str = _classify_token_type(
    HF_TOKEN,
    declared_type=os.environ.get("HF_TOKEN_TYPE"),
)

#: Classified type for HF_WRITE_TOKEN.
#: Source priority mirrors HF_TOKEN_TYPE above.
#: Accepted values: ``"fine-grained"``, ``"write"``.
#: Returns ``"unknown"`` when the token is absent or type cannot be inferred.
HF_WRITE_TOKEN_TYPE: str = _classify_token_type(
    HF_WRITE_TOKEN,
    declared_type=os.environ.get("HF_WRITE_TOKEN_TYPE"),
)

#: HF Serverless Inference API base URL (no trailing slash).
HF_BASE: str = os.environ.get("HF_BASE", DEFAULT_HF_BASE).rstrip("/")

#: Fallback model when request body omits ``model``.
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

#: Path 2 destination URL — the custom ai-model HF Space.
HF_SPACES_MODEL_URL: str = os.environ.get(
    "HF_SPACES_MODEL_URL", DEFAULT_HF_SPACES_MODEL_URL
).strip()

#: Parsed model owner namespaces routed to HF_SPACES_MODEL_URL (Path 2).
_raw_namespaces: str = os.environ.get(
    "HF_SPACES_MODEL_NAMESPACES",
    ",".join(DEFAULT_HF_SPACES_MODEL_NAMESPACES),
)
HF_SPACES_MODEL_NAMESPACES: tuple[str, ...] = (
    tuple(ns.strip() for ns in _raw_namespaces.split(",") if ns.strip())
    or DEFAULT_HF_SPACES_MODEL_NAMESPACES
)

#: Maximum accepted request body size (bytes).
MAX_BODY_BYTES: int = _safe_int(
    os.environ.get("MAX_BODY_BYTES"),
    DEFAULT_MAX_BODY_BYTES,
)

# ── Per-path read timeouts ────────────────────────────────────────────────────
#: Path 1 (BACKEND_URL) read timeout in seconds.
_proxy_timeout_secs: float = float(
    _safe_int(
        os.environ.get("PROXY_TIMEOUT"),
        DEFAULT_PROXY_TIMEOUT,
    )
)

#: Path 2 (ai-model Space, CPU inference) read timeout in seconds.
#: Default 600 s — covers 4-5 min CPU inference with 1 min headroom.
_path2_timeout_secs: float = _safe_float(
    os.environ.get("PATH2_TIMEOUT"),
    DEFAULT_PATH2_READ_TIMEOUT,
)

#: Path 3 (HF Serverless API, GPU) read timeout in seconds.
#: Default 120 s — generous margin for GPU-backed inference (30-90 s typical).
_path3_timeout_secs: float = _safe_float(
    os.environ.get("PATH3_TIMEOUT"),
    DEFAULT_PATH3_READ_TIMEOUT,
)

# ── Shared phase timeouts (apply to all paths) ────────────────────────────────
#: TCP handshake timeout in seconds.
#: Uses ``_safe_float`` — a non-numeric env var logs a warning and falls back
#: to the default rather than raising ``ValueError`` at startup.
_connect_timeout_secs: float = _safe_float(
    os.environ.get("PROXY_CONNECT_TIMEOUT"), 10.0
)
#: Request body upload timeout in seconds.
_write_timeout_secs: float = _safe_float(os.environ.get("PROXY_WRITE_TIMEOUT"), 30.0)
#: Connection pool acquire timeout in seconds.
_pool_timeout_secs: float = _safe_float(os.environ.get("PROXY_POOL_TIMEOUT"), 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────

_raw_origins: str = os.environ.get("ALLOWED_ORIGINS", "*").strip()
_allowed_origins: list[str] = (
    ["*"]
    if _raw_origins == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)

#: HuggingFace Dataset repo for training contributions.
#: Must be set if POST /v1/contribute is expected to succeed.
TRAINING_DATASET_REPO: str = os.environ.get("TRAINING_DATASET_REPO", "").strip()

#: Consent-version enforcement is controlled by a SINGLE flag,
#: :data:`_dataset_schema.CONSENT_VERSION_ENABLED`, imported above as
#: ``CONSENT_VERSION_ENABLED``.  While ``False`` (current state), the
#: ``consentVersion`` field in ``POST /v1/contribute`` is **not validated** —
#: any value (including ``null``, sent by the current JS) is accepted, and
#: every normalised record stores ``consentVersion: null`` regardless of what
#: the client sent (see ``_dataset_schema._resolve_consent_version``).
#:
#: Previously this module had its OWN ``TRAINING_CONSENT_VERSION = "v1.0"``
#: constant and a hard ``payload["consentVersion"] != "v1.0"`` check — when
#: the JS was updated to send ``consentVersion: null`` (matching
#: ``_dataset_schema``'s new convention), every ``/v1/contribute`` call
#: started failing with 422 "consentVersion ... is not current", because the
#: two modules' notions of "the current consent version" had silently
#: diverged.  Importing the flag (and :data:`RESERVED_CONSENT_VERSION` below)
#: directly from ``_dataset_schema`` makes that divergence structurally
#: impossible: flipping ``CONSENT_VERSION_ENABLED`` to ``True`` in ONE place
#: (``_dataset_schema.py``) simultaneously activates the 422 guard below AND
#: the write-side normalisation — see ``_dataset_schema.py`` for the full
#: activation checklist (also requires an ``ai-assistant.js`` update to send
#: ``consentVersion: CONSENT_VERSION`` again instead of ``null``).

#: When ``True``, ``POST /v1/feedback`` is persisted to ``TRAINING_DATASET_REPO``
#: under the ``feedback/`` folder alongside ``contributions/``.
#:
#: Default ``True`` — feedback is written to the HuggingFace dataset (durable,
#: survives Space restarts) whenever ``TRAINING_DATASET_REPO`` and
#: ``HF_DATASET_TOKEN`` are configured.  Set ``FEEDBACK_PERSIST_ENABLED=false``
#: (or ``0`` / ``no``) to revert to log-only / in-memory behaviour.
#:
#: **Rationale for opt-out default** — the previous opt-in default (``False``)
#: silently discarded all feedback on fresh or restarted deployments even when
#: dataset credentials were present.  Opt-out is safer for operators who intend
#: to collect training data: a misconfiguration now causes visible log noise
#: rather than silent data loss.
#:
#: **Deduplication responsibility** — enabling this flag means the same
#: ``(conversationId, answerIndex)`` pair *may* appear in both
#: ``contributions/*.jsonl`` (when the user also clicks Contribute) and
#: ``feedback/*.jsonl`` (from the immediate rating button).  Every stored
#: record carries ``_source`` and ``_dedup_key`` fields so that downstream
#: consumers can deterministically resolve duplicates before training.
#: See ``DATASET_COLLECTION_GUIDANCE.md`` for the canonical dedup procedure.
#:
#: How to disable
#: --------------
#: HF Spaces → Settings → Repository secrets → add
#: ``FEEDBACK_PERSIST_ENABLED`` = ``false``  (also accepts ``0``, ``no``)
#: Without ``TRAINING_DATASET_REPO`` and ``HF_WRITE_TOKEN`` / ``HF_TOKEN``
#: the flag has no effect regardless of its value.
FEEDBACK_PERSIST_ENABLED: bool = os.environ.get(
    "FEEDBACK_PERSIST_ENABLED", "true"
).strip().lower() not in ("false", "0", "no")

#: Maximum records per contribution POST.
MAX_CONTRIBUTION_RECORDS: int = 100

#: Maximum number of distinct IP entries kept in each in-memory rate-limit dict.
#: When the dict would exceed this size a full sweep removes all entries whose
#: sliding window has expired before the new entry is stored.  This bounds
#: memory to O(_MAX_RL_ENTRIES) even under a slow-drip unique-IP scan that
#: never reuses the same address.  Value chosen to be generous for legitimate
#: traffic (thousands of real users) while preventing unbounded growth.
_MAX_RL_ENTRIES: int = 10_000

#: In-memory per-IP rate-limit store for contribution endpoint.
#: Keys: IP string.  Values: (count, window_start_timestamp).
_contrib_rl: dict[str, tuple[int, float]] = {}
_contrib_rl_lock = asyncio.Lock()

#: In-memory per-IP rate-limit store for share endpoint.
_share_rl: dict[str, tuple[int, float]] = {}
_share_rl_lock = asyncio.Lock()

#: In-memory conversation share store.
#: Keys: UUID hex string.  Values: share metadata dict including ``expiresAt_ts``.
#: Ephemeral — clears on process restart.  Stale entries are evicted lazily on write.
_share_store: dict[str, dict] = {}
_share_store_lock = asyncio.Lock()

#: In-memory per-IP rate-limit store for feedback endpoint.
_feedback_rl: dict[str, tuple[int, float]] = {}
_feedback_rl_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Startup validation — fail fast with actionable messages
# ─────────────────────────────────────────────────────────────────────────────

_validate_env(BACKEND_URL, HF_TOKEN, HF_SPACES_MODEL_URL)

# Token-type least-privilege validation.  Runs after _validate_env confirms
# that routing is viable.  Issues are logged at WARNING or ERROR level but
# never block startup — the proxy starts in degraded mode so operators can
# read the log and fix the configuration without a redeploy loop.
_token_config_issues: list[str] = _validate_token_config(
    HF_TOKEN,
    HF_WRITE_TOKEN,
    TRAINING_DATASET_REPO,
    hf_token_type=HF_TOKEN_TYPE,
    hf_write_token_type=HF_WRITE_TOKEN_TYPE,
)
for _issue in _token_config_issues:
    _log_level = logging.ERROR if _issue.startswith("ERROR:") else logging.WARNING
    logger.log(_log_level, "Startup token-config check: %s", _issue)

if not BACKEND_URL and not HF_TOKEN:
    logger.warning(
        "HF_TOKEN is not set. Requests to standard HF Inference API models "
        "(e.g. openai/gpt-oss-20b, Qwen/*) will fail with 401 Unauthorized. "
        "Only models in namespaces %s will be served via %s.",
        list(HF_SPACES_MODEL_NAMESPACES),
        HF_SPACES_MODEL_URL or "<HF_SPACES_MODEL_URL not set>",
    )

if TRAINING_DATASET_REPO and not HF_DATASET_TOKEN:
    logger.warning(
        "TRAINING_DATASET_REPO is set (%r) but neither HF_WRITE_TOKEN nor "
        "HF_TOKEN is configured. POST /v1/contribute will return 503. "
        "Set HF_WRITE_TOKEN (preferred, write-scoped) or HF_TOKEN as fallback.",
        TRAINING_DATASET_REPO,
    )
elif TRAINING_DATASET_REPO and HF_WRITE_TOKEN:
    logger.info(
        "Training contributions enabled: repo=%r, using dedicated "
        "HF_WRITE_TOKEN (hf_token=%s, write_token=%s).",
        TRAINING_DATASET_REPO,
        _token_log_fragment(HF_TOKEN),
        _token_log_fragment(HF_WRITE_TOKEN),
    )
elif TRAINING_DATASET_REPO:
    logger.warning(
        "Training contributions enabled: repo=%r. Using HF_TOKEN for dataset "
        "writes (HF_WRITE_TOKEN not set). Recommend setting a dedicated "
        "write-scoped HF_WRITE_TOKEN — see HF Settings > Tokens.",
        TRAINING_DATASET_REPO,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP client — lifecycle managed by FastAPI lifespan
# ─────────────────────────────────────────────────────────────────────────────

#: Module-level reference to the shared httpx client.
#: Created with ``timeout=None`` so all timeout control is per-request.
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Create and close the shared HTTP client on application startup / shutdown.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance.

    Notes
    -----
    **Developer note** — The client is created with ``timeout=None`` so
    that every request supplies its own :class:`httpx.Timeout` object.
    This allows concurrent Path 2 requests (600 s) and Path 3 requests
    (120 s) to coexist on the same client without either blocking the other.
    """
    global _http_client  # noqa: PLW0603
    _http_client = httpx.AsyncClient()
    logger.info(
        "Proxy v%s started. HTTP client ready (timeout=per-request).",
        PROXY_VERSION,
    )
    logger.info(
        "Routing: backend_url=%r | hf_spaces_model_url=%r | "
        "hf_spaces_namespaces=%r | hf_token=%s | write_token=%s | default_model=%r",
        BACKEND_URL or None,
        HF_SPACES_MODEL_URL or None,
        list(HF_SPACES_MODEL_NAMESPACES),
        _token_log_fragment(HF_TOKEN, token_type=HF_TOKEN_TYPE),
        (
            _token_log_fragment(HF_WRITE_TOKEN, token_type=HF_WRITE_TOKEN_TYPE)
            if HF_WRITE_TOKEN
            else "<using-hf-token-fallback>"
        ),
        DEFAULT_MODEL,
    )
    logger.info(
        "Timeouts (seconds): path1=%s | path2=%s | path3=%s | "
        "connect=%s | write=%s | pool=%s",
        _proxy_timeout_secs,
        _path2_timeout_secs,
        _path3_timeout_secs,
        _connect_timeout_secs,
        _write_timeout_secs,
        _pool_timeout_secs,
    )
    try:
        yield
    finally:
        await _http_client.aclose()
        _http_client = None
        logger.info("Proxy shutdown. HTTP client closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="sphinx-ai-assistant proxy",
    description=(
        "Thin OpenAI-compatible reverse proxy for sphinx-ai-assistant. "
        "Routes to HF Serverless Inference API, a custom ai-model Space, "
        "or an explicit backend URL based on the model namespace."
    ),
    version=PROXY_VERSION,
    lifespan=_lifespan,
    docs_url=None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["GET", "HEAD", "POST", "PATCH", "OPTIONS"],
    # Authorization added for write endpoints (POST /v1/feedback, POST /v1/contribute)
    # that validate a Bearer token.  Without this the browser preflight rejects
    # requests containing Authorization headers before the handler runs.
    # HEAD added so the JS _pingUrl health-check and the HF Space internal health
    # monitor can send cross-origin HEAD / and HEAD /health without a CORS error.
    # PATCH added for PATCH /v1/share/{uuid} (content update, URL preserved).
    allow_headers=["Content-Type", "Authorization"],
    allow_credentials=False,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_url(body: bytes) -> tuple[str, dict[str, str], float]:
    """
    Thin wrapper around :func:`_resolve_upstream_url`.

    Closes over module-level config globals so route handlers need not pass
    environment variables explicitly.

    Parameters
    ----------
    body : bytes
        Raw JSON request body from the browser.

    Returns
    -------
    url : str
        Fully-qualified upstream endpoint URL.
    headers : dict[str, str]
        HTTP headers for the upstream POST request.
    read_timeout_s : float
        Per-path read timeout in seconds.

    See Also
    --------
    _shared_logic._resolve_upstream_url : Full three-path routing logic.
    """
    return _resolve_upstream_url(
        body,
        backend_url=BACKEND_URL,
        hf_token=HF_TOKEN,
        hf_base=HF_BASE,
        default_model=DEFAULT_MODEL,
        hf_spaces_model_url=HF_SPACES_MODEL_URL,
        hf_spaces_model_namespaces=HF_SPACES_MODEL_NAMESPACES,
        proxy_timeout=_proxy_timeout_secs,
        path2_read_timeout=_path2_timeout_secs,
        path3_read_timeout=_path3_timeout_secs,
    )


def _make_timeout(read_s: float) -> httpx.Timeout:
    """
    Build a per-request :class:`httpx.Timeout` with the given read timeout.

    Parameters
    ----------
    read_s : float
        Read timeout in seconds for this specific request.

    Returns
    -------
    httpx.Timeout
        Fully specified timeout with connect, read, write, and pool phases.

    Notes
    -----
    **Developer note** — connect, write, and pool timeouts are shared
    across all paths because they do not vary by inference speed.  Only
    the read timeout varies: long (600 s) for CPU inference (Path 2),
    short (120 s) for GPU inference (Path 3).
    """
    return httpx.Timeout(
        connect=_connect_timeout_secs,
        read=read_s,
        write=_write_timeout_secs,
        pool=_pool_timeout_secs,
    )


async def _validated_body(request: Request) -> bytes:
    """
    FastAPI dependency: read and validate the request body size.

    Parameters
    ----------
    request : Request
        The incoming FastAPI request.

    Returns
    -------
    bytes
        The raw request body.

    Raises
    ------
    HTTPException
        HTTP 413 when the body exceeds :data:`MAX_BODY_BYTES`.
    """
    cl = _safe_int(request.headers.get("content-length"), -1)
    if cl > MAX_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Request body too large (Content-Length: {cl:,} bytes). "
                f"Maximum allowed: {MAX_BODY_BYTES:,} bytes."
            ),
        )
    body: bytes = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Request body too large ({len(body):,} bytes). "
                f"Maximum allowed: {MAX_BODY_BYTES:,} bytes."
            ),
        )
    return body


async def _forward(body: bytes) -> Response:
    """
    Forward *body* to the resolved upstream and return the response.

    Handles both non-streaming (JSON) and streaming (SSE) responses
    transparently by detecting ``"stream": true`` in the request body.

    Per-path timeouts are applied at the individual request level so slow
    CPU (Path 2) and fast GPU (Path 3) requests do not interfere with each
    other on the shared HTTP client.

    Parameters
    ----------
    body : bytes
        Raw JSON request body from the browser.

    Returns
    -------
    fastapi.Response
        Upstream response with original status code and content-type.
        SSE streaming is preserved via :class:`~fastapi.responses.StreamingResponse`.

    Notes
    -----
    **Developer note** — Error handling is explicit and specific:

    * ``httpx.ReadTimeout``    → HTTP 504 with actionable timeout message.
    * ``httpx.ConnectTimeout`` → HTTP 504 with connect-specific message.
    * ``httpx.RequestError``   → HTTP 502 Bad Gateway.

    These map to the correct HTTP semantics and allow the browser widget
    to display a useful message rather than a generic "network error".

    **Developer note** — SSE error events include a UUID so log aggregators
    can correlate browser-visible errors to specific upstream failure events.
    """
    if _http_client is None:
        raise RuntimeError(
            "HTTP client is not initialised. "
            "FastAPI lifespan may not have started correctly."
        )

    url, headers, read_timeout_s = _resolve_url(body)
    req_timeout = _make_timeout(read_timeout_s)

    # Detect streaming intent before opening the upstream connection.
    stream_requested: bool = False
    try:
        payload: Any = json.loads(body)
        stream_requested = bool(payload.get("stream", False))
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        pass

    if stream_requested:

        async def _sse_chunks() -> AsyncGenerator[bytes, None]:
            """Async generator proxying upstream SSE frames to the browser."""
            try:
                async with _http_client.stream(  # type: ignore[union-attr]
                    "POST", url, content=body, headers=headers, timeout=req_timeout
                ) as upstream:
                    if upstream.status_code != 200:  # noqa: PLR2004
                        err_body = await upstream.aread()
                        error_payload = json.dumps(
                            {
                                "id": f"err-{uuid.uuid4().hex}",
                                "error": {
                                    "status": upstream.status_code,
                                    "message": err_body.decode(errors="replace")[:500],
                                },
                            }
                        )
                        yield f"data: {error_payload}\n\n".encode()
                    else:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk

            except httpx.ReadTimeout:
                err_id = uuid.uuid4().hex
                logger.warning(
                    "ReadTimeout after %.0f s on streaming request to %s [%s]",
                    read_timeout_s,
                    url,
                    err_id,
                )
                yield f'data: {{"id":"err-{err_id}","error":{{"status":504,"message":'
                yield (
                    f'"Upstream timed out after {read_timeout_s:.0f} s. '
                    f"CPU inference can take 4-5 minutes. "
                    f'If using the ai-model Space, the model may still be loading."}}}}\n\n'
                ).encode()

            except httpx.ConnectTimeout:
                err_id = uuid.uuid4().hex
                logger.warning(
                    "ConnectTimeout on streaming request to %s [%s]", url, err_id
                )
                yield (
                    f'data: {{"id":"err-{err_id}","error":{{"status":504,"message":'
                    f'"Connection timed out reaching {url}. '
                    f'The HF Space may be starting up."}}}}\n\n'
                ).encode()

            except httpx.RequestError as exc:
                err_id = uuid.uuid4().hex
                logger.warning(
                    "RequestError on streaming request to %s: %s [%s]",
                    url,
                    exc,
                    err_id,
                )
                yield (
                    f'data: {{"id":"err-{err_id}","error":{{"status":502,"message":'
                    f'"Failed to reach upstream: {type(exc).__name__}"}}}}\n\n'
                ).encode()

        return StreamingResponse(
            _sse_chunks(),
            status_code=200,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming path: await the full upstream response.
    try:
        upstream = await _http_client.post(
            url, content=body, headers=headers, timeout=req_timeout
        )
    except httpx.ReadTimeout:
        logger.warning(
            "ReadTimeout after %.0f s on non-streaming request to %s",
            read_timeout_s,
            url,
        )
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "type": "timeout_error",
                    "message": (
                        f"Upstream timed out after {read_timeout_s:.0f} s. "
                        "CPU inference on the ai-model Space can take 4-5 minutes. "
                        "The model may still be loading — retry in a few minutes."
                    ),
                }
            },
        )
    except httpx.ConnectTimeout:
        logger.warning("ConnectTimeout on non-streaming request to %s", url)
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "type": "timeout_error",
                    "message": (
                        f"Connection timed out reaching {url}. "
                        "The HF Space may be cold-starting — retry in 30 seconds."
                    ),
                }
            },
        )
    except httpx.RequestError as exc:
        logger.warning("RequestError on non-streaming request to %s: %s", url, exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "upstream_error",
                    "message": f"Failed to reach upstream: {type(exc).__name__}",
                }
            },
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────


@app.head("/")
async def root_head() -> Response:
    """HEAD probe for ``/`` — returns 200 with no body.

    Notes
    -----
    **Developer note** — FastAPI 0.111 + Starlette 0.37.2 do **not** automatically
    handle HEAD requests for ``@app.get`` routes; the response is 405 unless an
    explicit ``@app.head`` handler is registered.

    Two callers depend on this handler:

    * **HF Space internal health monitor** — probes ``HEAD /`` to determine whether
      the Space is healthy enough to serve traffic.  A 405 causes the Space to be
      marked as unhealthy and removed from the routing pool.
    * **JS ``_pingUrl``** — the configuration panel "Test All Connectivity" button
      sends ``HEAD {base_url}`` (i.e. ``HEAD /``) for every feature whose base URL
      is this proxy.  A 405 produces noisy log lines; a 200 is silent and correct.
    """
    return Response(status_code=200)


@app.head("/health")
async def health_head() -> Response:
    """HEAD probe for ``/health`` — returns 200 with no body.

    Notes
    -----
    **Developer note** — Mirrors the reasoning for :func:`root_head`.
    Container orchestrators and uptime monitors that prefer ``HEAD /health``
    over ``GET /health`` (to avoid parsing the JSON body) receive 200.
    """
    return Response(status_code=200)


@app.get("/")
async def root() -> JSONResponse:
    """
    Human-readable status page and HF Space health-check handler.

    Returns
    -------
    JSONResponse
        HTTP 200 with service status and the active routing configuration.

    Notes
    -----
    **User note** — The ``timeouts`` field shows read timeouts in seconds
    per path.  ``path2`` corresponds to the ai-model Space (CPU inference,
    default 600 s).  ``path3`` corresponds to the HF Serverless API (GPU,
    default 120 s).
    """
    return JSONResponse(
        {
            "status": "ok",
            "service": f"sphinx-ai-assistant proxy v{PROXY_VERSION}",
            "routing": {
                "path_1_backend_url": BACKEND_URL or None,
                "path_2_model_space_url": HF_SPACES_MODEL_URL or None,
                "path_2_namespaces": list(HF_SPACES_MODEL_NAMESPACES),
                "path_3_hf_api_base": HF_BASE,
                "path_3_hf_token_set": bool(HF_TOKEN),
            },
            "tokens": {
                # Never expose token values — only whether each slot is filled.
                # hf_token_type / hf_write_token_type reflect the classified token
                # class so operators can verify least-privilege configuration without
                # reading Space logs.  Possible values: fine-grained | read | write |
                # unknown.  "unknown" means the type could not be inferred — set
                # HF_TOKEN_TYPE / HF_WRITE_TOKEN_TYPE in Space secrets to resolve.
                "hf_token_set": bool(HF_TOKEN),
                "hf_token_type": HF_TOKEN_TYPE,
                "hf_write_token_set": bool(HF_WRITE_TOKEN),
                "hf_write_token_type": HF_WRITE_TOKEN_TYPE,
                # True when HF_WRITE_TOKEN is set; False means writes fall
                # back to HF_TOKEN (single-token mode, less secure).
                "least_privilege_mode": bool(HF_WRITE_TOKEN),
            },
            "training": {
                "dataset_repo": TRAINING_DATASET_REPO or None,
                "contribute_ready": bool(TRAINING_DATASET_REPO and HF_DATASET_TOKEN),
                "feedback_persist_enabled": FEEDBACK_PERSIST_ENABLED,
            },
            "timeouts": {
                "path1_s": _proxy_timeout_secs,
                "path2_s": _path2_timeout_secs,
                "path3_s": _path3_timeout_secs,
                "connect_s": _connect_timeout_secs,
                "write_s": _write_timeout_secs,
            },
            "cors_origins": _allowed_origins,
            "endpoints": {
                "chat": "POST /v1/chat/completions (primary)",
                "share": "POST /v1/share            (conversation share)",
                "share_get": "GET  /v1/share/{uuid}     (retrieve shared snapshot)",
                "share_patch": (
                    "PATCH /v1/share/{uuid}    (update snapshot, URL preserved)"
                ),
                "feedback": "POST /v1/feedback         (rating persistence)",
                "training": "POST /v1/contribute       (GDPR-gated training data)",
                "alias": "POST /                    (path-agnostic alias)",
                "health": "GET  /health              (liveness probe)",
                "head_root": "HEAD /                    (health-monitor probe)",
                "head_health": "HEAD /health              (health-monitor probe)",
            },
        }
    )


@app.get("/health")
async def health() -> JSONResponse:
    """
    Minimal liveness probe for container orchestrators and uptime monitors.

    Returns
    -------
    JSONResponse
        Always HTTP 200 while the process is running.
    """
    return JSONResponse({"status": "ok", "version": PROXY_VERSION})


@app.post("/v1/chat/completions")
async def chat_completions(body: bytes = Depends(_validated_body)) -> Response:
    """
    Primary proxy endpoint — OpenAI-compatible ``/v1/chat/completions``.

    Parameters
    ----------
    body : bytes
        Raw request body, pre-validated by :func:`_validated_body`.

    Returns
    -------
    fastapi.Response
        Upstream response.  SSE streaming preserved when ``"stream": true``.

    Notes
    -----
    **User note** — Set ``endpoint`` in ``conf.py`` to::

        "https://scikit-plots-ai.hf.space/v1/chat/completions"

    **User note** — Model routing:

    * ``scikit-plots/Qwen2.5-Coder-7B-Instruct`` → ai-model Space (Path 2,
      CPU inference, up to 5 minutes per response).
    * ``openai/gpt-oss-20b``, ``Qwen/Qwen2.5-Coder-7B-Instruct`` →
      HF Serverless Inference API (Path 3, GPU, typically 30-90 s).

    See Also
    --------
    chat_completions_alias : ``POST /`` path-agnostic alias.
    """
    return await _forward(body)


@app.post("/")
async def chat_completions_alias(body: bytes = Depends(_validated_body)) -> Response:
    """
    Path-agnostic alias: ``POST /`` → identical to ``POST /v1/chat/completions``.

    Parameters
    ----------
    body : bytes
        Raw request body, pre-validated by :func:`_validated_body`.

    Returns
    -------
    fastapi.Response
        Identical to :func:`chat_completions`.

    Notes
    -----
    **User note** — Prefer the explicit ``/v1/chat/completions`` path.
    This alias handles ``conf.py`` configurations that set ``endpoint``
    to the bare Space URL without the path suffix.
    """
    return await _forward(body)


@app.post("/v1/contribute")
async def contribute(request: Request) -> JSONResponse:  # noqa: PLR0912
    """Accept a training data contribution from the AI assistant browser widget.

    Parameters
    ----------
    request : fastapi.Request
        HTTP request.  Body must be JSON conforming to the contribution schema.

    Returns
    -------
    fastapi.responses.JSONResponse
        ``{"contributed": true, "rows": N}`` on success.

    Raises
    ------
    fastapi.HTTPException
        422 when consent is absent/false, schemaVersion is unsupported, or
        records exceed the maximum allowed count.
        429 when the IP rate limit is exceeded.
        503 when the HF Dataset push fails.

    Notes
    -----
    Developer: ``consentFlag`` and ``consentVersion`` are checked before any
    other validation.  A missing or mismatched consent version is a hard
    rejection — the UI must always send the current version string so that
    old cached pages cannot submit records that would silently bypass a
    consent-text update.

    Developer: The in-memory rate-limit store (``_contrib_rl``) is per-process
    only.  HF Spaces may run multiple replicas.  The limit (5 per hour) is
    intentionally loose; the primary defence is the GDPR consent gate.

    Developer: ``huggingface_hub.HfApi.create_commit`` is called synchronously
    inside an async handler.  This blocks the event loop for the duration of
    the HTTP round-trip to HF (~200 ms on a warm connection).  For the current
    traffic level this is acceptable; if throughput grows, wrap in
    ``asyncio.get_event_loop().run_in_executor(None, ...)`` instead.
    """
    # Body size guard
    raw = await request.body()
    if len(raw) > DEFAULT_MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    # Consent guard — GDPR Article 7: explicit consent required
    if not payload.get("consentFlag"):
        raise HTTPException(
            status_code=422,
            detail="consentFlag must be true.  Contribution requires explicit user consent.",
        )
    # Consent-version guard (reserved for future enforcement).
    # CONSENT_VERSION_ENABLED (imported from _dataset_schema) is False in the
    # current state, so any consentVersion value — including null, sent by
    # the current JS — is accepted.  This is the single flag that previously
    # diverged from this module's own (now-removed) TRAINING_CONSENT_VERSION
    # check, causing the 422 "consentVersion not current" error.
    # When enforcement is activated: flip CONSENT_VERSION_ENABLED to True in
    # _dataset_schema.py, set RESERVED_CONSENT_VERSION there to the live
    # consent-banner version, and update ai-assistant.js to send
    # `consentVersion: CONSENT_VERSION` again instead of `null` — then this
    # block rejects mismatched/missing versions.
    if CONSENT_VERSION_ENABLED:  # noqa: SIM102
        if payload.get("consentVersion") != RESERVED_CONSENT_VERSION:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"consentVersion {payload.get('consentVersion')!r} is not current. "
                    f"Expected {RESERVED_CONSENT_VERSION!r}. Reload the page and try again."
                ),
            )

    # Schema version guard.
    # Accepts schemaVersion 1 (JS clients pre-v2 or not yet updated) and 2
    # (JS clients sending the new canonical version).  The server always
    # normalizes stored records to the current SCHEMA_VERSION (2) via
    # normalize_contribution_record regardless of which version the client
    # declares — so both values are safe to accept.
    supported_versions: frozenset[int] = frozenset({1, 2})
    if payload.get("schemaVersion") not in supported_versions:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported schemaVersion {payload.get('schemaVersion')!r}. "
                f"Supported: {sorted(supported_versions)}"
            ),
        )

    records = payload.get("records", [])
    if not isinstance(records, list):
        raise HTTPException(status_code=422, detail="records must be a list.")
    if len(records) > MAX_CONTRIBUTION_RECORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Too many records. Maximum {MAX_CONTRIBUTION_RECORDS} per request.",
        )

    # Per-IP rate limit: 5 contributions per hour
    client_ip = _client_ip(request)
    async with _contrib_rl_lock:
        now = _time.time()
        count, window_start = _contrib_rl.get(client_ip, (0, now))
        if now - window_start > 3600:  # noqa: PLR2004
            count, window_start = 0, now
        count += 1
        _contrib_rl[client_ip] = (count, window_start)
        if count > 5:  # noqa: PLR2004
            logger.warning(
                json.dumps({"event": "contribute.ratelimit", "ip": _mask_ip(client_ip)})
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 5 contributions per hour.",
                headers={"Retry-After": "3600"},
            )

    if not TRAINING_DATASET_REPO:
        raise HTTPException(
            status_code=503,
            detail="Training endpoint not configured (TRAINING_DATASET_REPO not set).",
        )
    if not HF_DATASET_TOKEN:
        # HF_DATASET_TOKEN = HF_WRITE_TOKEN or HF_TOKEN (resolved at startup).
        # Provide an actionable error that names the preferred variable first.
        raise HTTPException(
            status_code=503,
            detail=(
                "No write token configured. "
                "Set HF_WRITE_TOKEN (recommended, write-scoped) or HF_TOKEN "
                "(legacy fallback) in Space → Settings → Repository secrets."
            ),
        )

    # Push to HF Dataset using the write-scoped token.
    # HF_DATASET_TOKEN resolves to HF_WRITE_TOKEN when set, else HF_TOKEN.
    # This ensures the inference token (HF_TOKEN, read-only) is never used
    # for dataset write operations when a dedicated write token is available.
    from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415

    api = HfApi(token=HF_DATASET_TOKEN)

    # Compute server receive timestamp once so all rows in the batch share the
    # same ``_ts`` value.  The previous inline ``int(_time.time() * 1000)``
    # inside the list comprehension produced slightly different values per row.
    _server_ts_ms: int = int(_time.time() * 1000)

    # normalize_contribution_record maps legacy underscore-prefixed field names
    # (_sessionId → conversationId, _page → page, _model → model,
    # _consentVersion → consentVersion) to the canonical schema shared with the
    # feedback endpoint.  It also normalises ratingLabel (slug vs Title Case),
    # expands the model object to the full 8-key shape, and enforces key order.
    # See _dataset_schema.py for the full canonical column list.
    rows_jsonl = "\n".join(
        json.dumps(
            normalize_contribution_record(
                rec,
                envelope=payload,
                server_ts_ms=_server_ts_ms,
            ),
            ensure_ascii=False,
        )
        for rec in records
        if isinstance(rec, dict)
    )
    filename = f"contributions/{int(_time.time() * 1000)}.jsonl"
    try:
        # Fix D: Guard against empty JSONL — an empty file would silently corrupt
        # the dataset; fail fast with a clear error before touching HuggingFace.
        if not rows_jsonl.strip():
            logger.error(
                json.dumps(
                    {
                        "event": "contribute.empty_jsonl",
                        "records": len(records),
                        "detail": (
                            "rows_jsonl is empty after serialising records; "
                            "aborting HF commit to prevent corrupt dataset file."
                        ),
                    }
                )
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    "No JSONL rows were produced from the submitted records. "
                    "Verify that each record contains at least one non-empty field."
                ),
            )
        # BUG-03 FIX: HfApi has no .commit() in any released version.
        # Correct method is .create_commit() (stable since huggingface_hub 0.14,
        # fully supported through the declared ~=0.23.0 dependency range).
        # Without this fix, every POST would AttributeError → silent 503.
        #
        # Fix 3: api.create_commit is a synchronous HTTP call.  Calling it
        # directly inside an async handler blocks the uvicorn event loop for the
        # entire HuggingFace round-trip.  asyncio.to_thread() offloads the call
        # to a worker thread, keeping the loop free for other requests.
        await asyncio.to_thread(
            api.create_commit,
            repo_id=TRAINING_DATASET_REPO,
            repo_type="dataset",
            operations=[
                CommitOperationAdd(
                    path_in_repo=filename,
                    path_or_fileobj=rows_jsonl.encode(),
                )
            ],
            commit_message=f"Add {len(records)} contribution record(s)",
        )
    except Exception as exc:
        logger.error(json.dumps({"event": "contribute.hf_fail", "error": str(exc)}))
        raise HTTPException(
            status_code=503,
            detail="Failed to store contribution. Try again later.",
        ) from exc

    logger.info(
        json.dumps(
            {
                "event": "contribute.write",
                "rows": len(records),
                "ip": _mask_ip(client_ip),
            }
        )
    )
    return JSONResponse({"contributed": True, "rows": len(records)})


@app.post("/v1/share")
async def share(request: Request) -> JSONResponse:
    """Accept a conversation snapshot for global sharing.

    Parameters
    ----------
    request : fastapi.Request
        HTTP request.  Body must be JSON with at minimum a ``content`` field.

    Returns
    -------
    fastapi.responses.JSONResponse
        ``{"uuid": "<hex>", "url": "<share_url>", "expiresAt": "<ISO-8601>"}``
        on success.

    Raises
    ------
    fastapi.HTTPException
        400 when the body is not valid JSON.
        413 when the body exceeds :data:`~_shared_logic.DEFAULT_MAX_BODY_BYTES`.
        422 when ``content`` is missing or empty.
        429 when the IP rate limit (10/hour) is exceeded.

    Notes
    -----
    **User note** — Stored shares are ephemeral: they live only for the lifetime
    of the process.  A Space restart (or scale-to-zero cold start) clears all
    shares.  For persistent storage, set ``TRAINING_DATASET_REPO`` and use the
    ``/v1/contribute`` endpoint instead.

    **User note** — The returned ``url`` points to
    ``GET /v1/share/{uuid}`` on this proxy.  Distribute that link to share the
    conversation.

    **Developer note** — ``ttlDays`` is clamped to ``[1, 365]``.  Expired
    entries are evicted lazily: on every write, entries whose
    ``expiresAt_ts`` has passed are removed before the new entry is stored.
    This keeps memory bounded without a background task.

    **Developer note** — Rate limit (10/hour per IP) is intentionally more
    permissive than ``/v1/contribute`` because share payloads are user-facing
    outputs, not GDPR-sensitive training data.
    """
    # Body size guard
    raw = await request.body()
    if len(raw) > DEFAULT_MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    # Required field: content
    content = payload.get("content", "")
    if not content or not isinstance(content, str):
        raise HTTPException(
            status_code=422,
            detail="content is required and must be a non-empty string.",
        )

    mime_type: str = payload.get("mimeType") or "text/html;charset=utf-8"
    ext: str = payload.get("ext") or ".html"
    title: str = payload.get("title") or "Shared conversation"
    ttl_days: int = max(1, min(_safe_int(payload.get("ttlDays"), 30), 365))

    # Per-IP rate limit: 10 shares per hour
    client_ip = _client_ip(request)
    async with _share_rl_lock:
        now = _time.time()
        count, window_start = _share_rl.get(client_ip, (0, now))
        if now - window_start > 3600:  # noqa: PLR2004
            count, window_start = 0, now
        count += 1
        _share_rl[client_ip] = (count, window_start)
        if count > 10:  # noqa: PLR2004
            logger.warning(
                json.dumps({"event": "share.ratelimit", "ip": _mask_ip(client_ip)})
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 10 shares per hour.",
                headers={"Retry-After": "3600"},
            )

    share_id: str = uuid.uuid4().hex
    now_ts = _time.time()
    expires_ts = now_ts + ttl_days * 86400
    expires_iso = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(expires_ts))

    # Store in memory; evict expired entries lazily on each write.
    async with _share_store_lock:
        expired_keys = [
            k for k, v in _share_store.items() if v["expiresAt_ts"] < now_ts
        ]
        for k in expired_keys:
            del _share_store[k]
        _share_store[share_id] = {
            "content": content,
            "mimeType": mime_type,
            "ext": ext,
            "title": title,
            "expiresAt_ts": expires_ts,
            "expiresAt": expires_iso,
        }

    logger.info(
        json.dumps(
            {
                "event": "share.create",
                "id": share_id,
                "ip": _mask_ip(client_ip),
                "ttl_days": ttl_days,
            }
        )
    )

    base_url = str(request.base_url).rstrip("/")
    share_url = f"{base_url}/v1/share/{share_id}"

    return JSONResponse({"uuid": share_id, "url": share_url, "expiresAt": expires_iso})


@app.get("/v1/share/{share_id}")
async def share_get(share_id: str) -> Response:
    """Retrieve a shared conversation snapshot by UUID.

    Parameters
    ----------
    share_id : str
        UUID hex string returned by ``POST /v1/share``.

    Returns
    -------
    fastapi.Response
        The stored content with its original MIME type and a
        ``Content-Disposition: inline`` header.

    Raises
    ------
    fastapi.HTTPException
        404 when the UUID is not found.
        410 when the share exists but has expired.

    Notes
    -----
    **User note** — Shares are stored in process memory only.  A Space
    restart clears all shares, returning 404 for previously valid links.
    """
    async with _share_store_lock:
        entry = _share_store.get(share_id)

    if entry is None:
        raise HTTPException(status_code=404, detail="Share not found or expired.")

    if entry["expiresAt_ts"] < _time.time():
        async with _share_store_lock:
            _share_store.pop(share_id, None)
        raise HTTPException(status_code=410, detail="Share has expired.")

    return Response(
        content=entry["content"],
        status_code=200,
        media_type=entry.get("mimeType", "text/html;charset=utf-8"),
        headers={
            "Content-Disposition": (
                f'inline; filename="share{entry.get("ext", ".html")}"'
            ),
        },
    )


@app.patch("/v1/share/{share_id}")
async def share_patch(share_id: str, request: Request) -> JSONResponse:
    """Update an existing shared conversation snapshot in-place.

    PATCH replaces the stored content while preserving the UUID and therefore
    the public URL.  Callers — typically the browser JS after the user edits
    a conversation — can refresh a previously shared link without distributing
    a new URL.

    Parameters
    ----------
    share_id : str
        UUID hex string returned by the original ``POST /v1/share``.
    request : fastapi.Request
        HTTP request.  Body must be JSON with at minimum a ``content`` field.
        Optional fields ``mimeType``, ``ext``, ``title``, and ``ttlDays``
        override the stored values; omitted fields retain their stored value.

    Returns
    -------
    fastapi.responses.JSONResponse
        ``{"uuid": "<hex>", "url": "<share_url>", "expiresAt": "<ISO-8601>"}``
        on success — identical shape to ``POST /v1/share``.

    Raises
    ------
    fastapi.HTTPException
        400 when the body is not valid JSON.
        404 when the UUID is not found (never existed or evicted on a prior
            lazy sweep).
        410 when the entry exists but has already expired.
        413 when the body exceeds :data:`~_shared_logic.DEFAULT_MAX_BODY_BYTES`.
        422 when ``content`` is missing or empty.
        429 when the IP rate limit (shared with POST, 10/hour) is exceeded.

    Notes
    -----
    **User note** — TTL is reset from the time of the PATCH.  Patching a
    link that would have expired tomorrow extends it by the full ``ttlDays``
    value (default 30 days from now).

    **Developer note** — Rate-limited via the same ``_share_rl`` store as
    ``POST /v1/share``.  CREATE and UPDATE together count toward the
    10 requests/hour ceiling, preventing PATCH from being used as a bypass.

    **Developer note** — No ownership token is required; any caller who
    knows the UUID may PATCH it.  This is intentional: the feature targets
    closed documentation environments (internal Sphinx docs) where possession
    of the UUID already implies authorisation.

    **Developer note** — The lazy-eviction sweep that runs on ``POST /v1/share``
    does NOT run here to avoid the overhead of a full store scan on every
    update.  Expired entries are discovered on the existence check
    (``expiresAt_ts < now``) and returned as 410, which the JS client treats
    as a signal to fall back to a fresh POST.
    """
    # Body size guard
    raw = await request.body()
    if len(raw) > DEFAULT_MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    content = payload.get("content", "")
    if not content or not isinstance(content, str):
        raise HTTPException(
            status_code=422,
            detail="content is required and must be a non-empty string.",
        )

    # Per-IP rate limit — shared with POST so CREATE+UPDATE count together.
    client_ip = _client_ip(request)
    async with _share_rl_lock:
        now = _time.time()
        count, window_start = _share_rl.get(client_ip, (0, now))
        if now - window_start > 3600:  # noqa: PLR2004
            count, window_start = 0, now
        count += 1
        _share_rl[client_ip] = (count, window_start)
        if count > 10:  # noqa: PLR2004
            logger.warning(
                json.dumps({"event": "share.ratelimit", "ip": _mask_ip(client_ip)})
            )
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 10 shares per hour.",
                headers={"Retry-After": "3600"},
            )

    # Existence and expiry check (read lock only — no sweep needed here).
    async with _share_store_lock:
        entry = _share_store.get(share_id)

    if entry is None:
        raise HTTPException(status_code=404, detail="Share not found or expired.")

    if entry["expiresAt_ts"] < _time.time():
        async with _share_store_lock:
            _share_store.pop(share_id, None)
        raise HTTPException(status_code=410, detail="Share has expired.")

    # Merge: caller-supplied fields override stored values; omitted fields
    # keep their existing values so a minimal {"content": "..."} body works.
    mime_type: str = payload.get("mimeType") or entry.get(
        "mimeType", "text/html;charset=utf-8"
    )
    ext: str = payload.get("ext") or entry.get("ext", ".html")
    title: str = payload.get("title") or entry.get("title", "Shared conversation")
    ttl_days: int = max(1, min(_safe_int(payload.get("ttlDays"), 30), 365))

    now_ts = _time.time()
    expires_ts = now_ts + ttl_days * 86400
    expires_iso = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(expires_ts))

    async with _share_store_lock:
        _share_store[share_id] = {
            "content": content,
            "mimeType": mime_type,
            "ext": ext,
            "title": title,
            "expiresAt_ts": expires_ts,
            "expiresAt": expires_iso,
        }

    logger.info(
        json.dumps(
            {
                "event": "share.update",
                "id": share_id,
                "ip": _mask_ip(client_ip),
                "ttl_days": ttl_days,
            }
        )
    )

    base_url = str(request.base_url).rstrip("/")
    share_url = f"{base_url}/v1/share/{share_id}"

    return JSONResponse({"uuid": share_id, "url": share_url, "expiresAt": expires_iso})


@app.post("/v1/feedback")
async def feedback(request: Request) -> JSONResponse:
    """Accept a thumbs-up/down feedback record from the AI assistant widget.

    Parameters
    ----------
    request : fastapi.Request
        HTTP request.  Body must be valid JSON; all fields are optional.
        Typical payload fields from the JS widget:

        * ``ratingValue``  — ``"thumbs-up"`` or ``"thumbs-down"``
        * ``ratingLabel``  — human-readable label (e.g. ``"Helpful"``)
        * ``message``      — optional free-text comment from the user
        * ``query``        — the question that was asked
        * ``answer``       — the response that was rated
        * ``model``        — model ID that produced the answer
        * ``sessionId``    — browser session UUID
        * ``page``         — originating documentation page URL
        * ``ts``           — client-side timestamp (ms since epoch)

    Returns
    -------
    fastapi.responses.JSONResponse
        ``{"ok": true}`` on success.

    Raises
    ------
    fastapi.HTTPException
        400 when the body is not valid JSON.
        413 when the body exceeds :data:`~_shared_logic.DEFAULT_MAX_BODY_BYTES`.
        429 when the IP rate limit (30/hour) is exceeded.

    Notes
    -----
    **User note** — When ``FEEDBACK_PERSIST_ENABLED`` is ``false`` (the
    default), feedback is logged to the Space log stream only and is NOT
    stored in the dataset.  To enable dataset persistence set the environment
    variable ``FEEDBACK_PERSIST_ENABLED=true`` together with
    ``TRAINING_DATASET_REPO`` and ``HF_WRITE_TOKEN`` (or ``HF_TOKEN``).
    Persisted records are written to ``feedback/{timestamp}.jsonl`` in the
    dataset repo.  See ``DATASET_COLLECTION_GUIDANCE.md`` for the full
    data-collection contract and the canonical dedup procedure that eliminates
    cross-source duplicates before training.

    **Developer note** — The JS widget sends feedback with ``keepalive: true``
    so the request survives page unload.  The handler therefore needs no
    response body beyond ``{"ok": true}``; the widget does not read it.

    **Developer note** — Rate limit (30/hour per IP) is set high because
    users commonly rate multiple answers in a session.  The limit prevents
    programmatic flooding while allowing normal interactive use.

    **Developer note** — Deduplication contract: every persisted feedback record
    carries ``_source="feedback"`` and ``_dedup_key="{conversationId}:{answerIndex}"``.
    The ``conversationId`` is the stable per-page-load session UUID that the JS
    widget adds as ``detail.conversationId`` (distinct from the per-click
    ``detail.sessionId`` idempotency key).  Contribution records (from
    ``POST /v1/contribute``) carry the same ``_dedup_key`` format so that a
    simple equality comparison across both ``feedback/`` and ``contributions/``
    folders identifies cross-source duplicates.  Priority rule: when both sources
    carry the same ``_dedup_key``, keep the ``contribution`` record and discard
    the ``feedback`` record — the GDPR-gated explicit contribution is the
    higher-quality signal.
    """
    # Body size guard
    raw = await request.body()
    if len(raw) > DEFAULT_MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Payload too large.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422, detail="Feedback body must be a JSON object."
        )

    # ── Distinguish retraction tombstones from regular ratings ───────────────
    # Retractions are system-generated housekeeping records that invalidate a
    # previous rating in the training dataset.  They carry action="retract" and
    # prevSessionId (pointing to the original record) but NO ratingValue.
    # Key behavioural differences vs regular feedback:
    #   1. NOT counted against the per-IP rate limit — a user who edits 10
    #      answers would otherwise exhaust their 30-slot hourly budget on cleanup
    #      records alone (10 originals + 10 retractions + 10 new ratings = 30).
    #   2. Validated differently — prevSessionId is required; ratingValue is absent.
    #   3. Logged with event "feedback.retract" so operators can distinguish
    #      retraction volume from new-rating volume in log dashboards.
    #   4. Committed with a distinct commit_message so the HF repo history is legible.
    is_retract: bool = payload.get("action") == "retract"

    client_ip = _client_ip(request)

    if is_retract:
        # Validate required retraction fields before touching any rate-limit state.
        if not payload.get("prevSessionId"):
            raise HTTPException(
                status_code=422,
                detail="Retraction records must include a non-empty prevSessionId.",
            )
        logger.info(
            json.dumps(
                {
                    "event": "feedback.retract",
                    "ip": _mask_ip(client_ip),
                    "prevSessionId": payload.get("prevSessionId"),
                    "conversationId": payload.get("conversationId"),
                    "answerIndex": payload.get("answerIndex"),
                    "ts": payload.get("ts"),
                    "persist": FEEDBACK_PERSIST_ENABLED,
                }
            )
        )
    else:
        # ── Per-IP rate limit: 30 regular ratings per hour ───────────────────
        # Retractions are exempt (see above).  The limit guards against
        # programmatic flooding; normal interactive use stays well under it.
        async with _feedback_rl_lock:
            now = _time.time()
            # Sweep expired entries to bound memory under a unique-IP flood.
            # _MAX_RL_ENTRIES is the safety ceiling declared at module level.
            if len(_feedback_rl) >= _MAX_RL_ENTRIES:
                cutoff = now - 3600
                expired = [k for k, (_, ws) in _feedback_rl.items() if ws < cutoff]
                for k in expired:
                    del _feedback_rl[k]
            count, window_start = _feedback_rl.get(client_ip, (0, now))
            if now - window_start > 3600:  # noqa: PLR2004
                count, window_start = 0, now
            count += 1
            _feedback_rl[client_ip] = (count, window_start)
            if count > 30:  # noqa: PLR2004
                logger.warning(
                    json.dumps(
                        {"event": "feedback.ratelimit", "ip": _mask_ip(client_ip)}
                    )
                )
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Maximum 30 feedback submissions per hour.",
                    headers={"Retry-After": "3600"},
                )

        logger.info(
            json.dumps(
                {
                    "event": "feedback.receive",
                    "ip": _mask_ip(client_ip),
                    "ratingValue": payload.get("ratingValue"),
                    "model": payload.get("model"),
                    "conversationId": payload.get("conversationId"),
                    "answerIndex": payload.get("answerIndex"),
                    "page": payload.get("page"),
                    "ts": payload.get("ts"),
                    "persist": FEEDBACK_PERSIST_ENABLED,
                }
            )
        )

    # ── Optional HF dataset persistence ─────────────────────────────────────
    # Activated only when FEEDBACK_PERSIST_ENABLED=true AND the dataset repo
    # and write token are both configured.  Failures are logged and swallowed
    # so that a dataset-write error never breaks the user's rating experience
    # (the keepalive fire-and-forget model means the user won't see a retry UI
    # anyway).  Operators should monitor "feedback.persist_fail" log events.
    #
    # Retraction records ARE persisted (when persistence is enabled) because
    # the training pipeline needs to see them to suppress the original rating.
    # A retraction that arrives without persistence is a no-op — the original
    # rating also wasn't persisted, so there is nothing to suppress.
    if FEEDBACK_PERSIST_ENABLED and TRAINING_DATASET_REPO and HF_DATASET_TOKEN:
        try:
            from huggingface_hub import CommitOperationAdd, HfApi  # noqa: PLC0415

            # ``conversationId`` is the stable per-page-load session UUID that
            # the JS widget sets as ``detail.conversationId = _sessionId``.
            # It is *different* from ``detail.sessionId`` which is a per-click
            # idempotency key.  We need ``conversationId`` here because the
            # contribution endpoint uses ``payload.sessionId`` (= _sessionId)
            # as the first component of the dedup key.
            # normalize_feedback_record uses payload.get("conversationId") and
            # payload.get("answerIndex") internally to build the canonical record.
            # The variables below are derived from the serialised record so the
            # log event and filename are always consistent with what is stored.

            # normalize_feedback_record whitelists only known fields (discards
            # arbitrary client-supplied extras and the legacy ``rating`` alias),
            # normalises ratingLabel to a canonical slug, expands the model object
            # to the full 8-key shape, renames sessionId → feedbackId and
            # prevSessionId → prevFeedbackId, and enforces canonical key order.
            # The resulting record is structurally identical to a contribution row
            # so both sources concatenate directly into one pandas DataFrame.
            # See _dataset_schema.py for the full canonical column list.
            _rec_dict: dict = normalize_feedback_record(
                payload,
                server_ts_ms=int(_time.time() * 1000),
            )
            record: str = json.dumps(_rec_dict, ensure_ascii=False)
            conversation_id: str = str(_rec_dict.get("conversationId") or "")
            answer_index: Any = _rec_dict.get("answerIndex", "")
            filename = f"feedback/{int(_time.time() * 1000)}.jsonl"
            api = HfApi(token=HF_DATASET_TOKEN)
            # Fix 3 (feedback): same event-loop fix as /v1/contribute — offload
            # the synchronous HF HTTP call to a worker thread so the uvicorn
            # loop stays unblocked during the HuggingFace round-trip.
            commit_msg = (
                "Retract 1 feedback record" if is_retract else "Add 1 feedback record"
            )
            await asyncio.to_thread(
                api.create_commit,
                repo_id=TRAINING_DATASET_REPO,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(
                        path_in_repo=filename,
                        path_or_fileobj=record.encode(),
                    )
                ],
                commit_message=commit_msg,
            )
            logger.info(
                json.dumps(
                    {
                        "event": "feedback.persist_ok",
                        "retract": is_retract,
                        "dedup_key": f"{conversation_id}:{answer_index}",
                        "ip": _mask_ip(client_ip),
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Never propagate — the JS widget does not read the response body
            # and the user's rating must always be acknowledged with 200.
            logger.error(
                json.dumps({"event": "feedback.persist_fail", "error": str(exc)})
            )

    return JSONResponse({"ok": True})
