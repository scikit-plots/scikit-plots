# scikit-plots/ai  ·  _hf_spaces_proxy/app.py  v5.0.0
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
#   HF_TOKEN            Required for Path 3; optional for Path 2.
#   HF_SPACES_MODEL_URL Path 2 destination URL.
#                       Default: https://scikit-plots-ai-model.hf.space/v1/chat/completions
#   HF_SPACES_MODEL_NAMESPACES  Comma-separated owner namespaces for Path 2.
#                       Default: scikit-plots
#   BACKEND_URL         Path 1 override (all requests go here when set).
#   HF_BASE             HF Serverless API base URL.
#                       Default: https://api-inference.huggingface.co/models
#   DEFAULT_MODEL       Fallback model when request body omits "model".
#                       Default: scikit-plots/Qwen2.5-Coder-32B-Instruct
#   PROXY_TIMEOUT       Path 1 read timeout in seconds.  Default: 600.
#   PATH2_TIMEOUT       Path 2 read timeout in seconds.  Default: 600.
#   PATH3_TIMEOUT       Path 3 read timeout in seconds.  Default: 120.
#   PROXY_CONNECT_TIMEOUT TCP handshake timeout.  Default: 10.
#   PROXY_WRITE_TIMEOUT   Request body upload timeout.  Default: 30.
#   PROXY_POOL_TIMEOUT    Connection pool acquire timeout.  Default: 10.
#   ALLOWED_ORIGINS     Comma-separated CORS origins.  Default: *.
#   MAX_BODY_BYTES      Maximum accepted body size.  Default: 10485760.
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

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

import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# _shared_logic.py must live alongside this file.
try:
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
        _resolve_upstream_url,
        _safe_float,
        _safe_int,
        _token_log_fragment,
        _validate_env,
    )
except ImportError:
    from .._shared_logic import (  # type: ignore[import]
        DEFAULT_HF_BASE,
        DEFAULT_HF_SPACES_MODEL_NAMESPACES,
        DEFAULT_HF_SPACES_MODEL_URL,
        DEFAULT_MAX_BODY_BYTES,
        DEFAULT_MODEL,
        DEFAULT_PATH2_READ_TIMEOUT,
        DEFAULT_PATH3_READ_TIMEOUT,
        DEFAULT_PROXY_TIMEOUT,
        PROXY_VERSION,
        _resolve_upstream_url,
        _safe_float,
        _safe_int,
        _token_log_fragment,
        _validate_env,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — read once at module import, never at request time
# ─────────────────────────────────────────────────────────────────────────────

#: Explicit custom backend URL (Path 1).
BACKEND_URL: str = os.environ.get("BACKEND_URL", "").strip()

#: HuggingFace API token.  Required for Path 3; optional for Path 2.
HF_TOKEN: str = os.environ.get("HF_TOKEN", "").strip()

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
_connect_timeout_secs: float = float(os.environ.get("PROXY_CONNECT_TIMEOUT", "10"))
#: Request body upload timeout in seconds.
_write_timeout_secs: float = float(os.environ.get("PROXY_WRITE_TIMEOUT", "30"))
#: Connection pool acquire timeout in seconds.
_pool_timeout_secs: float = float(os.environ.get("PROXY_POOL_TIMEOUT", "10"))


# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────

_raw_origins: str = os.environ.get("ALLOWED_ORIGINS", "*").strip()
_allowed_origins: list[str] = (
    ["*"]
    if _raw_origins == "*"
    else [o.strip() for o in _raw_origins.split(",") if o.strip()]
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup validation — fail fast with actionable messages
# ─────────────────────────────────────────────────────────────────────────────

_validate_env(BACKEND_URL, HF_TOKEN, HF_SPACES_MODEL_URL)

if not BACKEND_URL and not HF_TOKEN:
    logger.warning(
        "HF_TOKEN is not set. Requests to standard HF Inference API models "
        "(e.g. openai/gpt-oss-20b, Qwen/*) will fail with 401 Unauthorized. "
        "Only models in namespaces %s will be served via %s.",
        list(HF_SPACES_MODEL_NAMESPACES),
        HF_SPACES_MODEL_URL or "<HF_SPACES_MODEL_URL not set>",
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
        "hf_spaces_namespaces=%r | hf_token=%s | default_model=%r",
        BACKEND_URL or None,
        HF_SPACES_MODEL_URL or None,
        list(HF_SPACES_MODEL_NAMESPACES),
        _token_log_fragment(HF_TOKEN),
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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
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
            "timeouts": {
                "path1_s": _proxy_timeout_secs,
                "path2_s": _path2_timeout_secs,
                "path3_s": _path3_timeout_secs,
                "connect_s": _connect_timeout_secs,
                "write_s": _write_timeout_secs,
            },
            "cors_origins": _allowed_origins,
            "endpoints": {
                "chat": "POST /v1/chat/completions  (primary)",
                "alias": "POST /                     (path-agnostic alias)",
                "health": "GET  /health               (liveness probe)",
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
