# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_hf_spaces_proxy/app.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
#
# scikit-plots/ai  ·  _hf_spaces_proxy/app.py  v7.4.0
#
# Server-authoritative chat proxy for sphinx-ai-assistant.
#
# THREE-PATH ROUTING  (evaluated in order)
# ─────────────────────────────────────────
#   Path 1 — BACKEND_URL set  (explicit custom backend override)
#     Forward the server-constructed provider request to BACKEND_URL.
#     Only BACKEND_AUTH_TOKEN may be attached; HF_TOKEN is never reused here.
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
#   HF_TOKEN            Required for Path 3 (read/inference). Keep this token
#                       inference-only whenever a separate dataset token exists.
#   HF_DATASET_TOKEN    Preferred persistence token for feedback/contributions.
#                       A fine-grained token scoped to write only the target
#                       dataset repo is recommended; classic Write also works.
#   HF_WRITE_TOKEN      Historical alias for HF_DATASET_TOKEN. Despite the name,
#                       a repo-scoped fine-grained token is valid. Falls back to
#                       HF_TOKEN only for backward-compatible single-token setups.
#   HF_SPACES_MODEL_URL Path 2 destination URL.
#                       Default: https://scikit-plots-ai-model.hf.space/v1/chat/completions
#   HF_SPACES_MODEL_NAMESPACES  Comma-separated owner namespaces for Path 2.
#                       Default: scikit-plots
#   BACKEND_URL         Path 1 override (server-constructed requests go here).
#   BACKEND_AUTH_TOKEN  Optional bearer token bound only to BACKEND_URL.
#   HF_SPACES_AUTH_TOKEN Optional bearer token bound only to HF_SPACES_MODEL_URL.
#   ALLOWED_MODELS      Comma-separated exact provider model IDs accepted by chat.
#                       Default matches the public models in _example_conf.py.
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
#   PROXY_PROTOCOL_RETRIES  Retry a pre-output remote protocol/read failure.
#                       Bounded to 0..2; default: 1. LocalProtocolError is not
#                       retried because it indicates a client/request defect.
#   HF_TOKEN_TYPE       Declare the type of HF_TOKEN so startup validation can
#                       enforce principle-of-least-privilege without network calls.
#                       Accepted values: fine-grained | read | write
#                       When absent the proxy applies a length-based heuristic.
#                       Set to "read" when using a classic read token; set to
#                       "fine-grained" for new-style scoped tokens.  Never set
#                       "write" — that triggers a startup WARNING.
#   HF_DATASET_TOKEN    Preferred dataset-persistence token. A fine-grained
#                       token scoped to write the target dataset repo is
#                       recommended. HF_WRITE_TOKEN remains a legacy alias.
#   HF_DATASET_TOKEN_TYPE  fine-grained | read | write. Read is rejected for
#                       persistence; fine-grained repo-write is preferred.
#   HF_WRITE_TOKEN_TYPE Legacy type declaration for HF_WRITE_TOKEN.
#                       Accepted values: fine-grained | read | write.
#   ALLOWED_ORIGINS     Comma-separated exact CORS origins. Additive by default; use ALLOWED_ORIGINS_MODE=replace for downstream sites.
#   ALLOWED_ORIGINS_MODE additive (default) keeps built-in Scikit-plots origins; replace trusts only ALLOWED_ORIGINS.
#   RECORD_STORAGE_TARGETS  Optional JSON array defining one primary record
#                       store plus mirrors (huggingface/github/gitlab/bitbucket).
#                       Credentials are referenced only through env names with
#                       prefix AI_RECORD_STORAGE_TOKEN_. When absent, legacy
#                       TRAINING_DATASET_REPO + HF_* token settings are used.
#   MAX_BODY_BYTES      Maximum accepted body size.  Default: 10485760.
#   MAX_UPSTREAM_RESPONSE_BYTES Maximum decoded upstream response bytes. Default: 8388608.
#   SHARE_MAX_BODY_BYTES Global Share request/canonical snapshot limit. Default: 512000.
#   SHARE_MAX_ENTRIES   Maximum live in-memory Global Share entries. Default: 256.
#   SHARE_MAX_TOTAL_BYTES Aggregate live Global Share byte budget. Default: 16777216.
#   SHARE_WRITE_TOKEN   Optional server-only create gate for POST /v1/share.
#                       Never bake this into static Sphinx HTML.
#   SHARE_PUBLIC_BASE_URL Optional externally-visible HTTPS base for Share links.
#   SHARE_STORE_BACKEND memory (compatibility), sqlite (restart-durable local),
#                       or redis (shared multi-replica authority).
#   SHARE_STORE_SQLITE_PATH SQLite file used when backend=sqlite.
#   SHARE_STORE_REDIS_URL Redis/rediss URL used when backend=redis.
#   DEPLOYMENT_PROFILE   compat (default) or strict. The hardened container sets
#                       strict, which requires non-root execution, exact CORS,
#                       and verified TLS for every configured Redis control plane.
#   REDIS_REQUIRE_TLS    Require rediss:// + certificate/hostname verification.
#                       Forced true by DEPLOYMENT_PROFILE=strict.
#   REQUIRE_NON_ROOT     Reject startup as UID 0. Forced true by strict profile.
#   SHARE_REQUIRE_DURABLE Fail Share writes closed unless backend reports durable.
#   SHARE_REQUIRE_SHARED Fail Share writes closed unless backend is shared+authoritative.
#   TRUST_X_FORWARDED_FOR Trust leftmost X-Forwarded-For only when a known
#                       ingress proxy strips/overwrites caller-supplied values.
#   RATE_LIMIT_BACKEND local (default) or redis. Redis is the shared atomic
#                       consistency domain for horizontally scaled HF replicas.
#   RATE_LIMIT_REDIS_URL Redis/rediss connection URL used only when backend=redis.
#   RATE_LIMIT_IDENTITY_SECRET >=32-byte server-only HMAC key. Raw client
#                       identities are never written to shared Redis keys.
#   RATE_LIMIT_KEY_PREFIX Optional shared Redis namespace. Default: sphinx-ai-assistant.
#   RATE_LIMIT_REQUIRE_SHARED When true, local rate limiting is rejected and
#                       Redis initialization/runtime failure returns HTTP 503.
#   RATE_LIMIT_REDIS_TIMEOUT_SECONDS Redis operation timeout; clamped 0.25..10 s.
#   FEEDBACK_PERSIST_ENABLED Persist privacy-minimal consent-gated rating telemetry only. Default false.
#   CONTRIBUTION_REVIEW_MODE ledger (compatibility) or provider-pr. provider-pr
#                       stores consented submissions in a native Git/HF review ref;
#                       only merge into the canonical branch makes them eligible.
#   CONTRIBUTION_REVIEW_TOKEN Optional operator token for API-driven merge/promotion.
#                       Native provider UI merge/close remains the preferred review path.
#   CONTRIBUTION_QUARANTINE_TTL_SECONDS Pending contribution lifetime. Default 86400.
#   CONTRIBUTION_LEDGER_BACKEND memory (default compatibility) or sqlite.
#   CONTRIBUTION_LEDGER_SQLITE_PATH SQLite control-plane file when backend=sqlite.
#   CONTRIBUTION_REQUIRE_DURABLE When true, /v1/contribute fails closed unless a
#                       durable receipt ledger (currently sqlite) is active.

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
    object.  Streaming opens the upstream response with
    ``build_request(..., timeout=...)`` + ``send(..., stream=True)`` before the
    downstream status is committed.  The returned response is always closed
    explicitly, so concurrent SSE requests are safe.

Developer note — explicit error handling
    ``_forward`` catches ``httpx.ReadTimeout``, ``httpx.ConnectTimeout``,
    and ``httpx.RequestError`` individually and returns meaningful JSON
    errors with appropriate HTTP status codes so the browser widget can
    display a useful message instead of a generic "network error".
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time as _time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Private service helpers live in the sibling _utils package.
try:
    from ._utils._dataset_schema import (  # type: ignore[import]
        FEEDBACK_TELEMETRY_CONSENT_VERSION,
        FEEDBACK_TELEMETRY_SCHEMA_VERSION,
        LEGACY_CONSENT_VERSIONS,
        MAX_CONTRIBUTION_NOTE_CHARS,
        MAX_CONVERSATION_MESSAGE_CHARS,
        MAX_CONVERSATION_MESSAGES,
        RESERVED_CONSENT_VERSION,
        normalize_contribution_record,
        normalize_contribution_withdrawal_record,
        normalize_feedback_record,
        normalize_feedback_review_record,
    )
except Exception:  # noqa: BLE001
    from _utils._dataset_schema import (  # type: ignore[import]
        FEEDBACK_TELEMETRY_CONSENT_VERSION,
        FEEDBACK_TELEMETRY_SCHEMA_VERSION,
        LEGACY_CONSENT_VERSIONS,
        MAX_CONTRIBUTION_NOTE_CHARS,
        MAX_CONVERSATION_MESSAGE_CHARS,
        MAX_CONVERSATION_MESSAGES,
        RESERVED_CONSENT_VERSION,
        normalize_contribution_record,
        normalize_contribution_withdrawal_record,
        normalize_feedback_record,
        normalize_feedback_review_record,
    )

try:
    from ._utils._storage import (  # type: ignore[import]
        StorageConfigError,
        StorageCoordinator,
        StorageWriteError,
        load_storage_targets,
    )
except Exception:  # noqa: BLE001
    from _utils._storage import (  # type: ignore[import]
        StorageConfigError,
        StorageCoordinator,
        StorageWriteError,
        load_storage_targets,
    )

try:
    from ._utils._contribution_ledger import (  # type: ignore[import]
        ContributionLedgerError,
        build_contribution_ledger,
    )
except Exception:  # noqa: BLE001
    from _utils._contribution_ledger import (  # type: ignore[import]
        ContributionLedgerError,
        build_contribution_ledger,
    )

try:
    from ._utils._share_store import (  # type: ignore[import]
        ShareStoreError,
        build_share_store,
    )
except Exception:  # noqa: BLE001
    from _utils._share_store import (  # type: ignore[import]
        ShareStoreError,
        build_share_store,
    )

try:
    from ._utils._rate_limit import (  # type: ignore[import]
        RateLimitBackendError,
        RedisRateLimiter,
    )
except Exception:  # noqa: BLE001
    from _utils._rate_limit import (  # type: ignore[import]
        RateLimitBackendError,
        RedisRateLimiter,
    )

try:
    from ._utils._telemetry import PrivacyJsonFormatter
except ImportError:  # standalone HF Space deployment
    from _utils._telemetry import PrivacyJsonFormatter

try:
    from ._utils._chat_contract import (  # type: ignore[import]
        CHAT_CONTRACT,
        ChatContractError,
        encode_upstream_payload,
        parse_chat_request,
    )
except Exception:  # noqa: BLE001
    from _utils._chat_contract import (  # type: ignore[import]
        CHAT_CONTRACT,
        ChatContractError,
        encode_upstream_payload,
        parse_chat_request,
    )

try:
    from ._utils._share_contract import (  # type: ignore[import]
        ShareValidationError,
        canonicalize_share_snapshot,
        generate_edit_token,
        hash_edit_token,
        render_share,
        render_share_viewer_shell,
        sanitize_share_page_url,
        valid_share_id,
        validate_share_format,
        verify_edit_token,
    )
except Exception:  # noqa: BLE001
    from _utils._share_contract import (  # type: ignore[import]
        ShareValidationError,
        canonicalize_share_snapshot,
        generate_edit_token,
        hash_edit_token,
        render_share,
        render_share_viewer_shell,
        sanitize_share_page_url,
        valid_share_id,
        validate_share_format,
        verify_edit_token,
    )

try:
    from ._utils._shared_logic import (  # type: ignore[import]
        DEFAULT_HF_BASE,
        DEFAULT_HF_PROVIDER_MODELS,
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
        _validate_credential_destination,
        _validate_env,
        _validate_token_config,
    )
    from ._utils._stub_model import (  # type: ignore[import]
        is_stub_model,
        parse_stub_mode,
        stub_delay_ms,
        stub_modes,
        stub_payload,
        stub_sse_frames,
    )
except Exception:  # noqa: BLE001
    from _utils._shared_logic import (  # type: ignore[import]
        DEFAULT_HF_BASE,
        DEFAULT_HF_PROVIDER_MODELS,
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
        _validate_credential_destination,
        _validate_env,
        _validate_token_config,
    )
    from _utils._stub_model import (  # type: ignore[import]
        is_stub_model,
        parse_stub_mode,
        stub_delay_ms,
        stub_modes,
        stub_payload,
        stub_sse_frames,
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
    Developer: default to the direct ASGI peer address.  A deployment may opt
    into leftmost ``X-Forwarded-For`` only when a known ingress strips and
    overwrites caller-supplied forwarding headers.  The boolean is a trust
    declaration, not a way to make an arbitrary forwarded header trustworthy.
    """
    # Forwarded identity is accepted only when the deployment explicitly says
    # its ingress overwrites the header.  Otherwise a direct caller can forge
    # X-Forwarded-For and rotate rate-limit identities at will.
    if globals().get("TRUST_X_FORWARDED_FOR", False):
        xff = request.headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[0].strip()
    if request.client:
        return request.client.host or "unknown"
    return "unknown"


def _opaque_share_request_mode(request: Request) -> str:
    """Classify an opaque-origin Share request as ``read``, ``write`` or ``none``.

    ``Origin: null`` is shared by legitimate ``file://`` pages and hostile
    sandboxed opaque documents, so compatibility is deliberately route/method
    scoped.  Browser preflights are classified using
    ``Access-Control-Request-Method`` rather than the OPTIONS method itself.
    """
    path = request.url.path
    if path != "/v1/share" and not path.startswith("/v1/share/"):
        return "none"
    method = request.method.upper()
    if method == "OPTIONS":
        method = (
            request.headers.get("access-control-request-method", "").strip().upper()
        )
    if path == "/v1/share" and method in {"GET", "HEAD"}:
        return "read"
    if path == "/v1/share/read" and method == "POST":
        return "read"
    if path.startswith("/v1/share/") and method in {"GET", "HEAD"}:
        return "read"
    return "write"


def _origin_allowed(  # ruff: ignore[too-many-return-statements]
    request: Request,
) -> bool:
    """Return whether a browser-supplied Origin is allowed for this service.

    Missing Origin is accepted for server-to-server/curl traffic.  Origin is a
    browser abuse boundary only; it is never treated as authentication.

    ``Origin: null`` read compatibility and mutation authority are separate.
    The broad write opt-in never works by itself: read compatibility must also
    be explicitly enabled, and strict deployments reject opaque-origin writes.
    """
    raw_origin = request.headers.get("origin", "").strip()
    if not raw_origin:
        return True
    if raw_origin == "null":
        mode = _opaque_share_request_mode(request)
        if mode == "read":
            return SHARE_ALLOW_OPAQUE_ORIGIN
        if mode == "write":
            return bool(SHARE_ALLOW_OPAQUE_ORIGIN and SHARE_ALLOW_OPAQUE_ORIGIN_WRITE)
        return False
    origin = _normalise_browser_origin(raw_origin)
    if not origin:
        return False
    if _allowed_origins == ["*"] or origin in _allowed_origins:
        return True
    # Same-origin browser requests remain usable even when the public host is
    # reverse-proxied and therefore is not listed in ALLOWED_ORIGINS.
    host = request.headers.get("host", "").strip().lower()
    if host:
        try:
            from urllib.parse import urlsplit  # ruff: ignore[import-outside-top-level]

            parsed = urlsplit(origin)
            if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == host:
                return True
        except ValueError:
            pass
    return False


class _UpstreamResponseTooLarge(  # ruff: ignore[error-suffix-on-exception-name]
    RuntimeError,
):
    """Raised when decoded upstream response bytes exceed the hard application cap."""


def _upstream_declared_length(response: httpx.Response) -> int | None:
    raw = (response.headers.get("content-length") or "").strip()
    if not raw:
        return None
    if not raw.isdigit():
        raise _UpstreamResponseTooLarge("invalid upstream content-length")
    value = int(raw)
    if value < 0:
        raise _UpstreamResponseTooLarge("invalid upstream content-length")
    return value


def _check_upstream_declared_length(
    response: httpx.Response, max_bytes: int | None = None
) -> None:
    limit = MAX_UPSTREAM_RESPONSE_BYTES if max_bytes is None else int(max_bytes)
    declared = _upstream_declared_length(response)
    if declared is not None and declared > limit:
        raise _UpstreamResponseTooLarge("declared upstream response too large")


async def _read_upstream_limited(
    response: httpx.Response, max_bytes: int | None = None
) -> bytes:
    """Read a streamed upstream response with a decoded-byte hard ceiling."""
    limit = MAX_UPSTREAM_RESPONSE_BYTES if max_bytes is None else int(max_bytes)
    _check_upstream_declared_length(response, limit)
    out = bytearray()
    total = 0
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise _UpstreamResponseTooLarge("upstream response too large")
        out.extend(chunk)
    return bytes(out)


async def _read_limited_body(
    request: Request, max_bytes: int, label: str = "Request"
) -> bytes:
    """Read an ASGI request incrementally and stop once *max_bytes* is crossed."""
    raw_cl = request.headers.get("content-length")
    if raw_cl not in (None, ""):
        try:
            content_length = int(raw_cl)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail="Invalid Content-Length header."
            ) from exc
        if content_length < 0:
            raise HTTPException(
                status_code=400, detail="Invalid Content-Length header."
            )
        if content_length > max_bytes:
            raise HTTPException(status_code=413, detail=f"{label} body too large.")

    body = bytearray()
    async for chunk in request.stream():
        if not chunk:
            continue
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{label} body too large.")
        body.extend(chunk)
    return bytes(body)


async def _consume_rate_limit(
    store: dict[str, tuple[int, float]],
    lock: asyncio.Lock,
    identity: str,
    *,
    limit: int,
    window_seconds: int = 3600,
    scope: str = "generic",
) -> tuple[bool, int]:
    """Consume one rate slot from the configured control plane.

    ``local`` mode preserves the bounded process-local abuse gate. ``redis``
    mode is fail-closed: a configured shared limiter that is unavailable never
    silently falls back to per-process counters, because that would split one
    advertised quota into independent replica-local windows.
    """
    if RATE_LIMIT_REQUIRE_SHARED and RATE_LIMIT_BACKEND == "local":
        raise HTTPException(
            status_code=503, detail="Shared rate limiter required by deployment policy."
        )
    if RATE_LIMIT_BACKEND != "local":
        if (
            _RATE_LIMIT_CONFIG_ERROR
            or _SHARED_RATE_LIMITER is None
            or not _SHARED_RATE_LIMITER_READY
        ):
            raise HTTPException(
                status_code=503, detail="Shared rate limiter unavailable."
            )
        try:
            allowed, count, _retry_after = await _SHARED_RATE_LIMITER.consume(
                identity, scope=scope, limit=limit, window_seconds=window_seconds
            )
            return allowed, count
        except RateLimitBackendError as exc:
            logger.error(
                "Shared rate limiter failure: code=%s scope=%s", exc.code, scope
            )
            raise HTTPException(
                status_code=503, detail="Shared rate limiter unavailable."
            ) from exc

    now = _time.time()
    async with lock:
        cutoff = now - window_seconds
        stale = [key for key, (_count, started) in store.items() if started <= cutoff]
        for key in stale:
            store.pop(key, None)

        if identity not in store and len(store) >= _MAX_RL_ENTRIES:
            # Fail closed rather than adding another attacker-controlled key.
            return False, 0

        count, window_start = store.get(identity, (0, now))
        if now - window_start >= window_seconds:
            count, window_start = 0, now
        count += 1
        store[identity] = (count, window_start)
        return count <= limit, count


def _hf_space_public_base() -> str:
    """Return the deployment-owned public HF Space origin, if available.

    Hugging Face injects ``SPACE_HOST`` into Spaces.  Treat it as a hostname,
    never as a free-form URL, and accept only the platform ``*.hf.space``
    namespace.  This lets a Docker Space whose ASGI scope is internally HTTP
    mint the externally correct HTTPS Share URL without trusting caller-supplied
    forwarding headers.
    """
    raw = str(SPACE_HOST or "").strip().lower().rstrip(".")
    if not raw or any(ch in raw for ch in ("/", "\\", "?", "#", "@")) or ":" in raw:
        return ""
    if not raw.endswith(".hf.space") or raw == ".hf.space":
        return ""
    return f"https://{raw}"


def _share_public_base(request: Request) -> str:
    """Return the externally safe base URL used in public Share links.

    Precedence is explicit operator configuration, then Hugging Face's
    deployment-owned ``SPACE_HOST``, then the ASGI request base.  An explicit
    HTTP URL is upgraded only when it names this same HF Space; arbitrary
    remote HTTP bases remain an error.
    """
    space_base = _hf_space_public_base()
    candidate = SHARE_PUBLIC_BASE_URL or space_base or str(request.base_url)
    safe = sanitize_share_page_url(candidate).rstrip("/")
    if not safe:
        raise HTTPException(status_code=500, detail="Share public base URL is invalid.")

    if safe.startswith("http://"):
        from urllib.parse import (  # ruff: ignore[import-outside-top-level]
            urlsplit,
            urlunsplit,
        )

        parsed = urlsplit(safe)
        host = (parsed.hostname or "").lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            return safe

        # Common HF Docker deployment mistake: SHARE_PUBLIC_BASE_URL was set to
        # the public Space hostname but with http:// because the internal
        # container listener is HTTP.  SPACE_HOST is deployment-owned evidence
        # that the same hostname is externally HTTPS, so upgrade only this exact
        # case instead of trusting X-Forwarded-* from arbitrary callers.
        if space_base and host == space_base.removeprefix("https://"):
            return urlunsplit(("https", parsed.netloc, parsed.path, "", "")).rstrip("/")

        raise HTTPException(
            status_code=500, detail="Share public base URL must use HTTPS."
        )
    return safe


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────


# Privacy-safe structured formatter shared with the direct model service.
# Exception tracebacks are reduced to bounded filename/function/line metadata
# and all messages pass the same high-confidence secret/URL redactor.
_handler = logging.StreamHandler()
_handler.setFormatter(PrivacyJsonFormatter())
_handler.addFilter(_RedactingFilter())
logging.root.handlers = [_handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration — read once at module import, never at request time
# ─────────────────────────────────────────────────────────────────────────────

#: Explicit custom backend URL (Path 1).
BACKEND_URL: str = os.environ.get("BACKEND_URL", "").strip()

#: Dedicated Path-1 bearer token. Never reuse HF_TOKEN for custom backends.
BACKEND_AUTH_TOKEN: str = os.environ.get("BACKEND_AUTH_TOKEN", "").strip()

#: Dedicated Path-2 bearer token bound to HF_SPACES_MODEL_URL.
HF_SPACES_AUTH_TOKEN: str = os.environ.get("HF_SPACES_AUTH_TOKEN", "").strip()

#: HuggingFace API token for **inference** (read-only).
#: Required for Path 3 (HF Serverless API). Never used for Paths 1 or 2.
#: Use a read-only / ``inference-api``-scoped fine-grained token here.
#: This token is forwarded to upstream model backends and must not carry
#: unnecessary write permission — see ``HF_WRITE_TOKEN`` below.
HF_TOKEN: str = os.environ.get("HF_TOKEN", "").strip()

#: Historical HuggingFace dataset-token alias. New deployments should prefer
#: ``HF_DATASET_TOKEN`` below. Despite this variable's old name, a repo-scoped
#: fine-grained token is valid and preferred; a classic ``write`` token is not
#: required.
#:
#: Separation of concerns (principle of least privilege):
#:
#: * ``HF_TOKEN``         — read / inference-api scope.
#:                          Injected into upstream inference requests.
#: * ``HF_DATASET_TOKEN`` — dataset-repo write capability only (preferred).
#: * ``HF_WRITE_TOKEN``   — backward-compatible alias for dataset writes.
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
#: Dataset token (``HF_DATASET_TOKEN``; preferred)
#:   Type: Fine-grained
#:   Permissions: ``Write access`` scoped to
#:   ``scikit-plots/ai-assistant-contributions`` (or your dataset repo)
#:   (Do NOT grant Inference API access — not needed for writes)
#:
#: Classic tokens (legacy, less secure than fine-grained):
#:   ``read`` role  → ``HF_TOKEN``       (inference only)
#:   ``write`` role → ``HF_DATASET_TOKEN`` or legacy ``HF_WRITE_TOKEN``
#:
#: Fallback behaviour
#: ------------------
#: If the preferred ``HF_DATASET_TOKEN`` is absent, the proxy tries legacy
#: ``HF_WRITE_TOKEN`` and finally ``HF_TOKEN`` so existing deployments remain
#: compatible.
HF_WRITE_TOKEN: str = os.environ.get("HF_WRITE_TOKEN", "").strip()

#: Preferred neutral dataset-persistence secret. Unlike the historical name
#: ``HF_WRITE_TOKEN``, this does not imply the classic Hugging Face ``write``
#: role: a repo-scoped fine-grained token is preferred for production.
HF_DATASET_TOKEN_EXPLICIT: str = os.environ.get("HF_DATASET_TOKEN", "").strip()

#: Effective token used **only** for HuggingFace dataset write operations.
#: Priority: HF_DATASET_TOKEN (preferred) -> HF_WRITE_TOKEN (legacy alias) ->
#: HF_TOKEN (single-token compatibility fallback). Never forward this token to
#: model inference backends.
HF_DATASET_TOKEN: str = HF_DATASET_TOKEN_EXPLICIT or HF_WRITE_TOKEN or HF_TOKEN

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

#: Classified type for the effective dataset token.
#: ``HF_DATASET_TOKEN_TYPE`` is preferred when HF_DATASET_TOKEN is set; legacy
#: aliases inherit their existing type declaration.
HF_DATASET_TOKEN_TYPE: str = (
    _classify_token_type(
        HF_DATASET_TOKEN_EXPLICIT,
        declared_type=os.environ.get("HF_DATASET_TOKEN_TYPE"),
    )
    if HF_DATASET_TOKEN_EXPLICIT
    else (HF_WRITE_TOKEN_TYPE if HF_WRITE_TOKEN else HF_TOKEN_TYPE)
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

#: Exact provider models this proxy will route. Namespaced Path-2 models are
#: separately allowed by HF_SPACES_MODEL_NAMESPACES.
_DEFAULT_ALLOWED_MODELS: tuple[str, ...] = tuple(
    dict.fromkeys((DEFAULT_MODEL, *DEFAULT_HF_PROVIDER_MODELS))
)
_raw_allowed_models = os.environ.get(
    "ALLOWED_MODELS", ",".join(_DEFAULT_ALLOWED_MODELS)
)
ALLOWED_MODELS: tuple[str, ...] = (
    tuple(dict.fromkeys(m.strip() for m in _raw_allowed_models.split(",") if m.strip()))
    or _DEFAULT_ALLOWED_MODELS
)

#: Maximum accepted request body size (bytes).
MAX_BODY_BYTES: int = max(
    16_384,
    min(
        _safe_int(os.environ.get("MAX_BODY_BYTES"), DEFAULT_MAX_BODY_BYTES),
        16 * 1024 * 1024,
    ),
)

#: Maximum decoded upstream response bytes accepted before the proxy aborts the
#: provider stream. This is independent from request-body limits because a
#: malicious/misconfigured upstream controls response size.
MAX_UPSTREAM_RESPONSE_BYTES: int = max(
    64 * 1024,
    min(
        _safe_int(os.environ.get("MAX_UPSTREAM_RESPONSE_BYTES"), 8 * 1024 * 1024),
        32 * 1024 * 1024,
    ),
)

#: Maximum JSON request size accepted by Global Share.  Kept independent from
#: chat bodies so a public share service cannot consume 10 MiB per request.
SHARE_MAX_BODY_BYTES: int = max(
    16_384,
    min(
        _safe_int(os.environ.get("SHARE_MAX_BODY_BYTES"), 512_000),
        2_000_000,
    ),
)

#: Hard in-memory share-store bounds.  A deployment can lower these values but
#: cannot raise them beyond conservative process-safety ceilings here.
SHARE_MAX_ENTRIES: int = max(
    1,
    min(
        _safe_int(os.environ.get("SHARE_MAX_ENTRIES"), 256),
        2048,
    ),
)
SHARE_MAX_TOTAL_BYTES: int = max(
    SHARE_MAX_BODY_BYTES,
    min(
        _safe_int(os.environ.get("SHARE_MAX_TOTAL_BYTES"), 16 * 1024 * 1024),
        64 * 1024 * 1024,
    ),
)

#: Deployment hardening profile. ``compat`` preserves existing deployments.
#: ``strict`` is used by the release container and cannot be weakened by a
#: per-feature setting: it requires non-root execution, exact-origin CORS, and
#: verified TLS for every configured Redis control plane.
DEPLOYMENT_PROFILE: str = (
    os.environ.get("DEPLOYMENT_PROFILE", "compat").strip().lower() or "compat"
)
_DEPLOYMENT_PROFILE_VALID: bool = DEPLOYMENT_PROFILE in {"compat", "strict"}
DEPLOYMENT_STRICT: bool = DEPLOYMENT_PROFILE == "strict"
REDIS_REQUIRE_TLS: bool = DEPLOYMENT_STRICT or os.environ.get(
    "REDIS_REQUIRE_TLS", "false"
).strip().lower() in {"1", "true", "yes", "on"}
REQUIRE_NON_ROOT: bool = DEPLOYMENT_STRICT or os.environ.get(
    "REQUIRE_NON_ROOT", "false"
).strip().lower() in {"1", "true", "yes", "on"}

#: Optional deployment-level write gate for creating new Global Shares.  This
#: secret must live only on the server.  Static Sphinx builds intentionally do
#: not serialize it; browser users may supply a short-lived runtime credential
#: only when an operator chooses this deployment mode.
SHARE_WRITE_TOKEN: str = os.environ.get("SHARE_WRITE_TOKEN", "").strip()

#: Optional externally-visible base URL for Share links when the app sits behind
#: a reverse proxy that does not preserve the public scheme/host in ASGI scope.
#: Must be HTTP(S); production non-local bases must be HTTPS.  Hugging Face
#: Spaces normally do not need this because SPACE_HOST provides the public host.
SHARE_PUBLIC_BASE_URL: str = os.environ.get("SHARE_PUBLIC_BASE_URL", "").strip()

#: Global Share lifecycle storage.  ``memory`` is compatibility only; it must
#: never be represented as durable/shared.  Requirements fail writes closed.
SHARE_STORE_BACKEND: str = (
    os.environ.get("SHARE_STORE_BACKEND", "memory").strip().lower() or "memory"
)
SHARE_STORE_SQLITE_PATH: str = os.environ.get(
    "SHARE_STORE_SQLITE_PATH",
    "/tmp/scikitplot-ai-global-share.sqlite3",  # ruff: ignore[hardcoded-temp-file]
).strip()
SHARE_STORE_REDIS_URL: str = os.environ.get("SHARE_STORE_REDIS_URL", "").strip()
SHARE_STORE_KEY_PREFIX: str = (
    os.environ.get("SHARE_STORE_KEY_PREFIX", "sphinx-ai-assistant").strip()
    or "sphinx-ai-assistant"
)
try:
    _raw_share_redis_timeout = float(
        os.environ.get("SHARE_STORE_REDIS_TIMEOUT_SECONDS", "2") or 2
    )
except (TypeError, ValueError):
    _raw_share_redis_timeout = 2.0
SHARE_STORE_REDIS_TIMEOUT_SECONDS: float = max(
    0.25, min(10.0, _raw_share_redis_timeout)
)
SHARE_REDIS_DURABILITY_CONFIRMED: bool = os.environ.get(
    "SHARE_REDIS_DURABILITY_CONFIRMED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
SHARE_REQUIRE_DURABLE: bool = os.environ.get(
    "SHARE_REQUIRE_DURABLE", "false"
).strip().lower() in {"1", "true", "yes", "on"}
SHARE_REQUIRE_SHARED: bool = os.environ.get(
    "SHARE_REQUIRE_SHARED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

#: Hugging Face-provided public Space hostname (for example,
#: ``owner-space.hf.space``).  This is deployment metadata, not caller input.
SPACE_HOST: str = os.environ.get("SPACE_HOST", "").strip()

#: Server-owned Share transport generation. Generation 2 means the public locator
#: belongs in URL fragment/body transport and MUST NOT be served by capability-bearing
#: legacy ``/v1/share/{id}`` request paths. Missing/1 marks pre-Run-14 objects only.
SHARE_TRANSPORT_VERSION: int = 2

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

#: Number of retries for a *remote* protocol/read failure before any response
#: bytes have been exposed to the browser.  A chat completion has no persistence
#: side effect, so one bounded retry is a useful recovery from a stale upstream
#: keep-alive connection.  ``LocalProtocolError`` is deliberately never retried:
#: it means our own request violated HTTP semantics and repeating it is pointless.
_protocol_retries: int = max(
    0, min(_safe_int(os.environ.get("PROXY_PROTOCOL_RETRIES"), 1), 2)
)


# ─────────────────────────────────────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_ALLOWED_ORIGINS: tuple[str, ...] = (
    "https://scikit-plots.github.io",
    "https://scikit-plots-learn.readthedocs.io",
)
_ALLOWED_ORIGIN_MODES: frozenset[str] = frozenset({"additive", "replace"})


def _normalise_browser_origin(value: str) -> str:
    """Return a canonical HTTP(S) origin, or ``""`` for malformed entries."""
    candidate = str(value or "").strip().rstrip("/")
    if not candidate:
        return ""
    try:
        from urllib.parse import urlsplit  # ruff: ignore[import-outside-top-level]

        parsed = urlsplit(candidate)
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return ""
    if parsed.path not in {"", "/"}:
        return ""
    return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}"


def _normalise_allowed_origins_mode(value: str) -> str:
    """Return the supported CORS composition mode without echoing bad input."""
    mode = str(value or "additive").strip().lower() or "additive"
    if mode not in _ALLOWED_ORIGIN_MODES:
        logger.warning(
            "AI proxy [CORS_MODE_INVALID]: invalid ALLOWED_ORIGINS_MODE; using additive defaults."
        )
        return "additive"
    return mode


def _build_allowed_origins(raw: str, *, mode: str = "additive") -> list[str]:
    """Build the exact browser-origin allowlist.

    ``additive`` retains the package defaults and appends deployment-specific
    exact origins. ``replace`` starts empty so downstream/open-source deployments
    can own their complete browser-origin trust boundary without editing source.
    Only an explicit ``*`` selects wildcard compatibility mode.
    """
    text = str(raw or "").strip()
    if text == "*":
        return ["*"]
    selected_mode = _normalise_allowed_origins_mode(mode)
    merged: list[str] = (
        list(_DEFAULT_ALLOWED_ORIGINS) if selected_mode == "additive" else []
    )
    for item in text.split(","):
        item = item.strip()  # ruff: ignore[redefined-loop-name]
        if not item:
            continue
        normalised = _normalise_browser_origin(item)
        if not normalised:
            logger.warning(
                "AI proxy [CORS_ORIGIN_IGNORED]: malformed ALLOWED_ORIGINS entry ignored."
            )
            continue
        if normalised not in merged:
            merged.append(normalised)
    return merged


_raw_origins: str = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS_MODE: str = _normalise_allowed_origins_mode(
    os.environ.get("ALLOWED_ORIGINS_MODE", "additive")
)
_allowed_origins: list[str] = _build_allowed_origins(
    _raw_origins, mode=ALLOWED_ORIGINS_MODE
)

#: Optional Share-only compatibility for browser opaque origins such as
#: ``file://`` documents, which browsers serialize as ``Origin: null``.  This is
#: deliberately disabled by default because sandboxed/opaque hostile documents
#: can also produce ``Origin: null``; enabling it is an operator abuse-surface
#: decision, not a way to authenticate local files.
SHARE_ALLOW_OPAQUE_ORIGIN: bool = os.environ.get(
    "SHARE_ALLOW_OPAQUE_ORIGIN", "false"
).strip().lower() in {"1", "true", "yes", "on"}

#: Additional high-risk opt-in for mutation/capability-management requests from
#: ``Origin: null``.  It never grants anything unless read compatibility above
#: is also enabled.  Strict deployments reject this flag entirely because a
#: sandboxed hostile document is indistinguishable from a local file by Origin.
SHARE_ALLOW_OPAQUE_ORIGIN_WRITE: bool = os.environ.get(
    "SHARE_ALLOW_OPAQUE_ORIGIN_WRITE", "false"
).strip().lower() in {"1", "true", "yes", "on"}

# Starlette's CORS middleware must know about the literal serialized opaque
# origin for the browser preflight to succeed.  The application guard below
# still limits that opt-in to /v1/share routes only.
_cors_allowed_origins: list[str] = list(_allowed_origins)
if SHARE_ALLOW_OPAQUE_ORIGIN and _cors_allowed_origins != ["*"]:
    _cors_allowed_origins.append("null")

if _allowed_origins == ["*"]:
    logger.warning(
        "AI proxy [CORS_WILDCARD]: ALLOWED_ORIGINS=* explicitly enables every browser origin; "
        "use an exact comma-separated origin list in production."
    )
elif ALLOWED_ORIGINS_MODE == "replace":
    logger.info(
        "AI proxy CORS: replacement mode active; only exact ALLOWED_ORIGINS entries are trusted."
    )
elif _raw_origins:
    logger.info(
        "AI proxy CORS: built-in documentation origins retained; ALLOWED_ORIGINS contributes additional exact origins."
    )

#: Trust ``X-Forwarded-For`` only when a known ingress proxy strips/overwrites
#: caller-supplied values.  Default false prevents rate-limit identity spoofing.
TRUST_X_FORWARDED_FOR: bool = os.environ.get(
    "TRUST_X_FORWARDED_FOR", "false"
).strip().lower() in {"1", "true", "yes", "on"}

#: HuggingFace Dataset repo for training contributions.
#: Must be set if POST /v1/contribute is expected to succeed.
TRAINING_DATASET_REPO: str = os.environ.get("TRAINING_DATASET_REPO", "").strip()

#: Provider-neutral record storage targets. JSON array; see README.  When unset,
#: TRAINING_DATASET_REPO + HF_WRITE_TOKEN/HF_TOKEN are synthesized as one
#: Hugging Face primary target for backwards compatibility.
RECORD_STORAGE_TARGETS: str = (
    os.environ.get("RECORD_STORAGE_TARGETS", "").strip()
    or os.environ.get("DATASET_TARGETS_JSON", "").strip()
)


#: Contribution consent version is owned by ``_dataset_schema`` and enforced
#: on every new intake.  Browser and server versions must change together.
#: Ordinary feedback is privacy-minimal telemetry, never training content.
#: Persistence is opt-in at both browser and server.  Even when enabled the
#: server normalizer discards query/answer/comment/model/page/session fields.
FEEDBACK_PERSIST_ENABLED: bool = os.environ.get(
    "FEEDBACK_PERSIST_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

#: Independent small body limit for rating telemetry.
FEEDBACK_MAX_BODY_BYTES: int = 16 * 1024

#: Content-bearing maintainer feedback is a separate authority from telemetry.
#: ``provider-pr`` opens/updates one native review per participant receipt;
#: ``disabled`` keeps all ratings local/telemetry-only even if a stale client
#: attempts the review route.  Browser review permission is independently
#: versioned and required on every review create/update.
FEEDBACK_REVIEW_MODE: str = (
    os.environ.get("FEEDBACK_REVIEW_MODE", "provider-pr").strip().lower()
    or "provider-pr"
)
if FEEDBACK_REVIEW_MODE not in {"disabled", "provider-pr"}:
    FEEDBACK_REVIEW_MODE = "disabled"
FEEDBACK_REVIEW_CONSENT_VERSION: str = "2.0.0"
FEEDBACK_TRAINING_CONSENT_VERSION: str = "1.0.0"
FEEDBACK_REVIEW_MAX_BODY_BYTES: int = 256 * 1024
FEEDBACK_REVIEW_TTL_SECONDS: int = max(
    3600,
    min(
        30 * 86400, _safe_int(os.environ.get("FEEDBACK_REVIEW_TTL_SECONDS"), 7 * 86400)
    ),
)

#: Pending contribution review configuration.
#: ``ledger`` keeps the historical process/DB quarantine and requires explicit
#: promotion. ``provider-pr`` writes the future eligible record to a native
#: provider review ref (GitHub PR, GitLab MR, Bitbucket PR, or Hugging Face PR);
#: only merging into the configured canonical branch makes it training eligible.
CONTRIBUTION_REVIEW_MODE: str = (
    os.environ.get("CONTRIBUTION_REVIEW_MODE", "ledger").strip().lower() or "ledger"
)
if CONTRIBUTION_REVIEW_MODE not in {"ledger", "provider-pr"}:
    CONTRIBUTION_REVIEW_MODE = "ledger"
CONTRIBUTION_REVIEW_TOKEN: str = os.environ.get("CONTRIBUTION_REVIEW_TOKEN", "").strip()
CONTRIBUTION_MAX_BODY_BYTES: int = 256 * 1024
CONTRIBUTION_QUARANTINE_TTL_SECONDS: int = max(
    300,
    min(
        604800,
        int(os.environ.get("CONTRIBUTION_QUARANTINE_TTL_SECONDS", "86400") or 86400),
    ),
)
CONTRIBUTION_QUARANTINE_MAX_ENTRIES: int = max(
    1,
    min(1024, int(os.environ.get("CONTRIBUTION_QUARANTINE_MAX_ENTRIES", "128") or 128)),
)
CONTRIBUTION_QUARANTINE_MAX_TOTAL_BYTES: int = max(
    65536,
    min(
        64 * 1024 * 1024,
        int(
            os.environ.get(
                "CONTRIBUTION_QUARANTINE_MAX_TOTAL_BYTES", str(4 * 1024 * 1024)
            )
            or (4 * 1024 * 1024)
        ),
    ),
)
CONTRIBUTION_LEDGER_BACKEND: str = (
    os.environ.get("CONTRIBUTION_LEDGER_BACKEND", "memory").strip().lower() or "memory"
)
CONTRIBUTION_LEDGER_SQLITE_PATH: str = os.environ.get(
    "CONTRIBUTION_LEDGER_SQLITE_PATH",
    "/tmp/scikitplot-ai-contribution-lifecycle.sqlite3",  # ruff: ignore[hardcoded-temp-file]
).strip()
CONTRIBUTION_REQUIRE_DURABLE: bool = os.environ.get(
    "CONTRIBUTION_REQUIRE_DURABLE", "false"
).strip().lower() in {"1", "true", "yes", "on"}
CONTRIBUTION_REQUIRE_SHARED: bool = os.environ.get(
    "CONTRIBUTION_REQUIRE_SHARED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
CONTRIBUTION_LEDGER_REDIS_URL: str = os.environ.get(
    "CONTRIBUTION_LEDGER_REDIS_URL", ""
).strip()
CONTRIBUTION_LEDGER_KEY_SECRET: str = os.environ.get(
    "CONTRIBUTION_LEDGER_KEY_SECRET", ""
)
CONTRIBUTION_LEDGER_KEY_PREFIX: str = (
    os.environ.get("CONTRIBUTION_LEDGER_KEY_PREFIX", "sphinx-ai-assistant").strip()
    or "sphinx-ai-assistant"
)
CONTRIBUTION_OPERATION_LEASE_SECONDS: int = max(
    30, min(900, _safe_int(os.environ.get("CONTRIBUTION_OPERATION_LEASE_SECONDS"), 120))
)
try:
    _raw_contribution_redis_timeout = float(
        os.environ.get("CONTRIBUTION_LEDGER_REDIS_TIMEOUT_SECONDS", "2") or 2
    )
except (TypeError, ValueError):
    _raw_contribution_redis_timeout = 2.0
CONTRIBUTION_LEDGER_REDIS_TIMEOUT_SECONDS: float = max(
    0.25, min(10.0, _raw_contribution_redis_timeout)
)
CONTRIBUTION_LEDGER_MAX_RECEIPTS: int = max(
    CONTRIBUTION_QUARANTINE_MAX_ENTRIES,
    min(100_000, _safe_int(os.environ.get("CONTRIBUTION_LEDGER_MAX_RECEIPTS"), 10_000)),
)
CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS: int = max(
    300,
    min(
        30 * 86400,
        _safe_int(
            os.environ.get("CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS"), 86400
        ),
    ),
)


#: Feedback-review lifecycle uses the same hardened ledger implementation but
#: a distinct namespace/file so its participant capabilities and capacity never
#: collide with dataset-contribution receipts. Defaults inherit the contribution
#: backend topology for operator simplicity.
FEEDBACK_REVIEW_LEDGER_BACKEND: str = (
    os.environ.get("FEEDBACK_REVIEW_LEDGER_BACKEND", CONTRIBUTION_LEDGER_BACKEND)
    .strip()
    .lower()
    or CONTRIBUTION_LEDGER_BACKEND
)
FEEDBACK_REVIEW_LEDGER_SQLITE_PATH: str = os.environ.get(
    "FEEDBACK_REVIEW_LEDGER_SQLITE_PATH",
    CONTRIBUTION_LEDGER_SQLITE_PATH + ".feedback-review",
).strip()
FEEDBACK_REVIEW_LEDGER_REDIS_URL: str = os.environ.get(
    "FEEDBACK_REVIEW_LEDGER_REDIS_URL", CONTRIBUTION_LEDGER_REDIS_URL
).strip()
FEEDBACK_REVIEW_LEDGER_KEY_SECRET: str = os.environ.get(
    "FEEDBACK_REVIEW_LEDGER_KEY_SECRET", CONTRIBUTION_LEDGER_KEY_SECRET
)
FEEDBACK_REVIEW_LEDGER_KEY_PREFIX: str = os.environ.get(
    "FEEDBACK_REVIEW_LEDGER_KEY_PREFIX",
    CONTRIBUTION_LEDGER_KEY_PREFIX + ":feedback-review",
).strip() or (CONTRIBUTION_LEDGER_KEY_PREFIX + ":feedback-review")
FEEDBACK_REVIEW_REQUIRE_DURABLE: bool = os.environ.get(
    "FEEDBACK_REVIEW_REQUIRE_DURABLE",
    "true" if CONTRIBUTION_REQUIRE_DURABLE else "false",
).strip().lower() in {"1", "true", "yes", "on"}
FEEDBACK_REVIEW_REQUIRE_SHARED: bool = os.environ.get(
    "FEEDBACK_REVIEW_REQUIRE_SHARED",
    "true" if CONTRIBUTION_REQUIRE_SHARED else "false",
).strip().lower() in {"1", "true", "yes", "on"}
FEEDBACK_REVIEW_LEDGER_MAX_RECEIPTS: int = max(
    128,
    min(
        100_000,
        _safe_int(os.environ.get("FEEDBACK_REVIEW_LEDGER_MAX_RECEIPTS"), 10_000),
    ),
)


#: Rate-limit backend. ``local`` is the bounded compatibility abuse gate.
#: ``redis`` is a shared atomic fixed-window decision domain suitable for
#: horizontally scaled proxy replicas. Redis mode requires a >=32-byte secret
#: so raw client identities are HMACed before they enter external storage.
RATE_LIMIT_BACKEND: str = (
    os.environ.get("RATE_LIMIT_BACKEND", "local").strip().lower() or "local"
)
RATE_LIMIT_REDIS_URL: str = os.environ.get("RATE_LIMIT_REDIS_URL", "").strip()
RATE_LIMIT_IDENTITY_SECRET: str = os.environ.get("RATE_LIMIT_IDENTITY_SECRET", "")
RATE_LIMIT_KEY_PREFIX: str = (
    os.environ.get("RATE_LIMIT_KEY_PREFIX", "sphinx-ai-assistant").strip()
    or "sphinx-ai-assistant"
)
RATE_LIMIT_REQUIRE_SHARED: bool = os.environ.get(
    "RATE_LIMIT_REQUIRE_SHARED", "false"
).strip().lower() in {"1", "true", "yes", "on"}
try:
    _raw_rate_limit_timeout = float(
        os.environ.get("RATE_LIMIT_REDIS_TIMEOUT_SECONDS", "2") or 2
    )
except (TypeError, ValueError):
    _raw_rate_limit_timeout = 2.0
RATE_LIMIT_REDIS_TIMEOUT_SECONDS: float = max(0.25, min(10.0, _raw_rate_limit_timeout))

CHAT_RATE_LIMIT_PER_HOUR: int = max(
    1, min(_safe_int(os.environ.get("CHAT_RATE_LIMIT_PER_HOUR"), 30), 10_000)
)
SHARE_RATE_LIMIT_PER_HOUR: int = max(
    1, min(_safe_int(os.environ.get("SHARE_RATE_LIMIT_PER_HOUR"), 10), 10_000)
)
FEEDBACK_RATE_LIMIT_PER_HOUR: int = max(
    1, min(_safe_int(os.environ.get("FEEDBACK_RATE_LIMIT_PER_HOUR"), 30), 10_000)
)
FEEDBACK_REVIEW_RATE_LIMIT_PER_HOUR: int = max(
    1, min(_safe_int(os.environ.get("FEEDBACK_REVIEW_RATE_LIMIT_PER_HOUR"), 20), 10_000)
)
CONTRIBUTION_RATE_LIMIT_PER_HOUR: int = max(
    1, min(_safe_int(os.environ.get("CONTRIBUTION_RATE_LIMIT_PER_HOUR"), 5), 10_000)
)

#: Whether this proxy forwards the reasoning-control parameters that the
#: documentation panel can send (``reasoning_effort`` for OpenAI-compatible
#: upstreams, ``thinking`` for Anthropic).
#:
#: Configuration
#: -------------
#: HF Spaces -> Settings -> Repository secrets -> add
#: ``REASONING_ENABLED`` = ``true``  (also accepts ``1``, ``yes``)
#:
#: **Operator note** -- Default ``false``, and deliberately so.  A strict
#: upstream rejects a request carrying an unknown top-level field with a 400,
#: so advertising support the upstream does not have would break every chat
#: request rather than degrade one setting.  Turn this on once you know the
#: model behind this proxy accepts the fields.
#:
#: **Developer note** -- This flag only controls what ``/health`` ADVERTISES.
#: The proxy forwards request bodies verbatim either way; it is a transparent
#: pipe, and making it strip or inject fields would break that contract and
#: hide upstream errors from the operator.  The flag exists so the panel can
#: stop relying on a hand-written declaration in a different repository.
#: Whether the deterministic stub responder ("Path 0") is reachable.
#:
#: Configuration
#: -------------
#: HF Spaces -> Settings -> Variables -> add
#: ``STUB_ENABLED`` = ``true``
#:
#: **Operator note** -- Default ``false``.  The stub never forwards upstream and
#: never reads a credential, but it does report what a request contained, and an
#: endpoint that describes incoming requests is not something to leave reachable
#: without a decision.  Every stub request is logged.
#:
#: **Developer note** -- ``stub/*`` is a reserved fail-closed namespace. With
#: this off, the proxy returns a local HTTP 503 ``stub_disabled`` response and
#: never forwards the diagnostic request to a real inference provider.
STUB_ENABLED: bool = os.environ.get(
    "STUB_ENABLED",
    "true",  # "false"
).strip().lower() in ("true", "1", "yes")

REASONING_ENABLED: bool = os.environ.get(
    "REASONING_ENABLED", "false"
).strip().lower() in ("true", "1", "yes")

#: Wire field name for the effort level, when ``REASONING_ENABLED`` is set.
#: Override only for an upstream that renames it.
REASONING_EFFORT_PARAM: str = os.environ.get(
    "REASONING_EFFORT_PARAM", "reasoning_effort"
).strip()

#: Wire field name for the extended-reasoning object.  Set to an empty string
#: for an upstream that offers effort but not a thinking budget.
REASONING_THINKING_PARAM: str = os.environ.get("REASONING_THINKING_PARAM", "").strip()

#: Thinking payload mode advertised to the panel when a Thinking field is
#: configured.  The proxy remains a transparent forwarder; this tells the UI
#: only which validated adapter shape the upstream is known to accept.
_raw_reasoning_thinking_mode = (
    os.environ.get("REASONING_THINKING_MODE", "budget").strip().lower()
)
if _raw_reasoning_thinking_mode not in {"boolean", "adaptive", "budget"}:
    logger.error(
        "AI proxy [REASONING_CONFIG_INVALID]: invalid REASONING_THINKING_MODE; "
        "safe budget mode will be advertised."
    )
    _raw_reasoning_thinking_mode = "budget"
REASONING_THINKING_MODE: str = _raw_reasoning_thinking_mode


def _reasoning_env_int(name: str, default: int) -> int:
    """Return a bounded reasoning integer without logging its raw value."""
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw or default)
    except (TypeError, ValueError):
        logger.error(
            "AI proxy [REASONING_CONFIG_INVALID]: invalid %s; safe default used.",
            name,
        )
        value = default
    return max(500, min(16000, value))


#: Token-budget bounds advertised to the panel. Invalid environment values are
#: fail-soft and privacy-safe: only the fixed option name is logged, never the
#: supplied value or request data.
REASONING_BUDGET_MIN: int = _reasoning_env_int("REASONING_BUDGET_MIN", 500)
REASONING_BUDGET_MAX: int = max(
    REASONING_BUDGET_MIN,
    _reasoning_env_int("REASONING_BUDGET_MAX", 16000),
)

#: Maximum records per contribution POST.
MAX_CONTRIBUTION_RECORDS: int = 100

#: Maximum number of distinct identities kept in each in-memory rate-limit dict.
#: Expired windows are swept before admission; when all remaining entries are
#: live, a new identity fails closed instead of growing the map.  This keeps
#: attacker-controlled identity cardinality strictly O(_MAX_RL_ENTRIES).
_MAX_RL_ENTRIES: int = max(
    128,
    min(
        _safe_int(os.environ.get("RATE_LIMIT_MAX_IDENTITIES"), 10_000),
        50_000,
    ),
)

#: In-memory per-IP rate-limit stores. Raw identities are process-memory only;
#: application logs receive only the existing masked representation.
_chat_rl: dict[str, tuple[int, float]] = {}
_chat_rl_lock = asyncio.Lock()
_contrib_rl: dict[str, tuple[int, float]] = {}
_contrib_rl_lock = asyncio.Lock()
_feedback_review_rl: dict[str, tuple[int, float]] = {}
_feedback_review_rl_lock = asyncio.Lock()

#: Optional shared rate-limit authority. Configuration errors are retained as
#: bounded codes and cause every rate-limited route to fail closed in redis mode.
_RATE_LIMIT_CONFIG_ERROR: str = ""
_SHARED_RATE_LIMITER: RedisRateLimiter | None = None
_SHARED_RATE_LIMITER_READY: bool = False
if RATE_LIMIT_BACKEND == "redis":
    try:
        _SHARED_RATE_LIMITER = RedisRateLimiter(
            RATE_LIMIT_REDIS_URL,
            identity_secret=RATE_LIMIT_IDENTITY_SECRET,
            key_prefix=RATE_LIMIT_KEY_PREFIX,
            socket_timeout_seconds=RATE_LIMIT_REDIS_TIMEOUT_SECONDS,
            require_tls=REDIS_REQUIRE_TLS,
        )
    except RateLimitBackendError as _rl_exc:
        _RATE_LIMIT_CONFIG_ERROR = _rl_exc.code
elif RATE_LIMIT_BACKEND != "local":
    _RATE_LIMIT_CONFIG_ERROR = "UNSUPPORTED_RATE_LIMIT_BACKEND"

#: Mutable contribution receipt control plane.  The default ``memory`` mode
#: preserves compatibility and intentionally remains process-local.  SQLite is
#: the local transactional/restart-durable option.  Redis is the shared atomic
#: cross-replica authority when configured with a dedicated HMAC key; deployments
#: may require either property explicitly and fail closed when it is unavailable.
#: External Redis persistence/durability is a separate operator guarantee.
_CONTRIBUTION_LEDGER_CONFIG_ERROR: str = ""
_CONTRIBUTION_LEDGER_READY: bool = False
try:
    _CONTRIBUTION_LEDGER = build_contribution_ledger(
        CONTRIBUTION_LEDGER_BACKEND,
        sqlite_path=CONTRIBUTION_LEDGER_SQLITE_PATH,
        redis_url=CONTRIBUTION_LEDGER_REDIS_URL,
        redis_key_secret=CONTRIBUTION_LEDGER_KEY_SECRET,
        redis_key_prefix=CONTRIBUTION_LEDGER_KEY_PREFIX,
        redis_timeout_seconds=CONTRIBUTION_LEDGER_REDIS_TIMEOUT_SECONDS,
        operation_lease_seconds=CONTRIBUTION_OPERATION_LEASE_SECONDS,
        require_redis_tls=REDIS_REQUIRE_TLS,
        max_pending_entries=CONTRIBUTION_QUARANTINE_MAX_ENTRIES,
        max_pending_bytes=CONTRIBUTION_QUARANTINE_MAX_TOTAL_BYTES,
        max_receipts=CONTRIBUTION_LEDGER_MAX_RECEIPTS,
        terminal_retention_seconds=CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS,
    )
except ContributionLedgerError as _ledger_exc:
    _CONTRIBUTION_LEDGER_CONFIG_ERROR = _ledger_exc.code
    _CONTRIBUTION_LEDGER = build_contribution_ledger(
        "memory",
        sqlite_path=CONTRIBUTION_LEDGER_SQLITE_PATH,
        max_pending_entries=CONTRIBUTION_QUARANTINE_MAX_ENTRIES,
        max_pending_bytes=CONTRIBUTION_QUARANTINE_MAX_TOTAL_BYTES,
        max_receipts=CONTRIBUTION_LEDGER_MAX_RECEIPTS,
        terminal_retention_seconds=CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS,
    )

# Compatibility-only inspection alias used by older tests/debug tooling. Routes
# never use this mapping as authority. It is populated only by memory backend.
_contrib_quarantine: dict[str, dict[str, Any]] = getattr(
    _CONTRIBUTION_LEDGER, "entries", {}
)

_FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR: str = ""
_FEEDBACK_REVIEW_LEDGER_READY: bool = False
try:
    _FEEDBACK_REVIEW_LEDGER = build_contribution_ledger(
        FEEDBACK_REVIEW_LEDGER_BACKEND,
        sqlite_path=FEEDBACK_REVIEW_LEDGER_SQLITE_PATH,
        redis_url=FEEDBACK_REVIEW_LEDGER_REDIS_URL,
        redis_key_secret=FEEDBACK_REVIEW_LEDGER_KEY_SECRET,
        redis_key_prefix=FEEDBACK_REVIEW_LEDGER_KEY_PREFIX,
        redis_timeout_seconds=CONTRIBUTION_LEDGER_REDIS_TIMEOUT_SECONDS,
        operation_lease_seconds=CONTRIBUTION_OPERATION_LEASE_SECONDS,
        require_redis_tls=REDIS_REQUIRE_TLS,
        max_pending_entries=max(128, CONTRIBUTION_QUARANTINE_MAX_ENTRIES),
        max_pending_bytes=max(4 * 1024 * 1024, CONTRIBUTION_QUARANTINE_MAX_TOTAL_BYTES),
        max_receipts=FEEDBACK_REVIEW_LEDGER_MAX_RECEIPTS,
        terminal_retention_seconds=CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS,
    )
except ContributionLedgerError as _feedback_ledger_exc:
    _FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR = _feedback_ledger_exc.code
    _FEEDBACK_REVIEW_LEDGER = build_contribution_ledger(
        "memory",
        sqlite_path=FEEDBACK_REVIEW_LEDGER_SQLITE_PATH,
        max_pending_entries=max(128, CONTRIBUTION_QUARANTINE_MAX_ENTRIES),
        max_pending_bytes=max(4 * 1024 * 1024, CONTRIBUTION_QUARANTINE_MAX_TOTAL_BYTES),
        max_receipts=FEEDBACK_REVIEW_LEDGER_MAX_RECEIPTS,
        terminal_retention_seconds=CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS,
    )

#: In-memory per-IP rate-limit store for share endpoint.
_share_rl: dict[str, tuple[int, float]] = {}
_share_rl_lock = asyncio.Lock()

#: Global Share lifecycle control plane.  The compatibility alias below is
#: populated only by the memory backend; routes never use it as authority.
_SHARE_STORE_CONFIG_ERROR: str = ""
_SHARE_STORE_READY: bool = False
try:
    _SHARE_STORE = build_share_store(
        SHARE_STORE_BACKEND,
        sqlite_path=SHARE_STORE_SQLITE_PATH,
        redis_url=SHARE_STORE_REDIS_URL,
        redis_key_prefix=SHARE_STORE_KEY_PREFIX,
        redis_timeout_seconds=SHARE_STORE_REDIS_TIMEOUT_SECONDS,
        redis_durable_confirmed=SHARE_REDIS_DURABILITY_CONFIRMED,
        require_redis_tls=REDIS_REQUIRE_TLS,
        max_entries=SHARE_MAX_ENTRIES,
        max_total_bytes=SHARE_MAX_TOTAL_BYTES,
    )
except ShareStoreError as _share_store_exc:
    _SHARE_STORE_CONFIG_ERROR = _share_store_exc.code
    _SHARE_STORE = build_share_store(
        "memory",
        sqlite_path=SHARE_STORE_SQLITE_PATH,
        max_entries=SHARE_MAX_ENTRIES,
        max_total_bytes=SHARE_MAX_TOTAL_BYTES,
    )
_share_store: dict[str, dict[str, Any]] = getattr(_SHARE_STORE, "entries", {})
_share_store_lock = asyncio.Lock()  # compatibility-only test/debug symbol

#: In-memory per-IP rate-limit store for feedback endpoint.
_feedback_rl: dict[str, tuple[int, float]] = {}
_feedback_rl_lock = asyncio.Lock()

#: Strong references for storage tasks shielded from client disconnects.
#: asyncio.shield() alone protects cancellation propagation, but retaining the
#: Task explicitly guarantees the event loop can keep it alive until completion.
_storage_tasks: set[asyncio.Task[Any]] = set()


async def _persist_storage_record(
    *,
    kind: str,
    content: bytes,
    commit_message: str,
    path_timestamp: float | None = None,
):
    """Run one storage write independently of the request socket lifetime."""
    task = asyncio.create_task(
        _STORAGE.write(
            kind=kind,
            content=content,
            commit_message=commit_message,
            path_timestamp=path_timestamp,
        )
    )
    _storage_tasks.add(task)
    task.add_done_callback(_storage_tasks.discard)
    return await asyncio.shield(task)


# ─────────────────────────────────────────────────────────────────────────────
# Storage target normalization. Invalid multi-target configuration degrades to
# the legacy Hugging Face target rather than preventing inference from starting.
try:
    _storage_targets = load_storage_targets(
        RECORD_STORAGE_TARGETS,
        legacy_repo=TRAINING_DATASET_REPO,
        legacy_token=HF_DATASET_TOKEN,
        legacy_token_type=HF_DATASET_TOKEN_TYPE,
    )
except StorageConfigError as _storage_exc:
    logger.error("Record storage configuration rejected: code=%s", str(_storage_exc))
    _storage_targets = load_storage_targets(
        "",
        legacy_repo=TRAINING_DATASET_REPO,
        legacy_token=HF_DATASET_TOKEN,
        legacy_token_type=HF_DATASET_TOKEN_TYPE,
    )
_STORAGE = StorageCoordinator(_storage_targets)

# Startup validation — fail fast with actionable messages
# ─────────────────────────────────────────────────────────────────────────────

_validate_env(BACKEND_URL, HF_TOKEN, HF_SPACES_MODEL_URL)
if BACKEND_AUTH_TOKEN:
    _validate_credential_destination(
        BACKEND_URL, credential_kind="BACKEND_AUTH_TOKEN", allow_local_http=True
    )
if HF_SPACES_AUTH_TOKEN:
    _validate_credential_destination(
        HF_SPACES_MODEL_URL, credential_kind="HF_SPACES_AUTH_TOKEN"
    )
if HF_TOKEN:
    _validate_credential_destination(HF_BASE, credential_kind="HF_TOKEN")

# Token-type least-privilege validation.  Runs after _validate_env confirms
# that routing is viable.  Issues are logged at WARNING or ERROR level but
# never block startup — the proxy starts in degraded mode so operators can
# read the log and fix the configuration without a redeploy loop.
# RECORD_STORAGE_TARGETS is authoritative when present.  Keep
# TRAINING_DATASET_REPO available for legacy discovery/rollback, but do not run
# the legacy HF token-consistency checks against it when provider-neutral
# storage is active: those writes use each target's token_env instead.
_LEGACY_HF_STORAGE_ACTIVE: bool = bool(
    TRAINING_DATASET_REPO and not RECORD_STORAGE_TARGETS
)
_token_config_issues: list[str] = _validate_token_config(
    HF_TOKEN,
    HF_DATASET_TOKEN,
    TRAINING_DATASET_REPO if _LEGACY_HF_STORAGE_ACTIVE else "",
    hf_token_type=HF_TOKEN_TYPE,
    hf_write_token_type=HF_DATASET_TOKEN_TYPE,
)
for _issue in _token_config_issues:
    _log_level = logging.ERROR if _issue.startswith("ERROR:") else logging.WARNING
    logger.log(_log_level, "Startup token-config check: %s", _issue)

if not BACKEND_URL and not HF_TOKEN:
    logger.warning(
        "HF_TOKEN is not set. Standard provider inference is unavailable; "
        "custom model-space routing remains configured=%s namespace_count=%d.",
        bool(HF_SPACES_MODEL_URL),
        len(HF_SPACES_MODEL_NAMESPACES),
    )


def is_read_only(token_type: str = HF_DATASET_TOKEN_TYPE):
    """is_read_only."""
    return token_type == "read"  # ruff: ignore[hardcoded-password-string]


if (
    _LEGACY_HF_STORAGE_ACTIVE
    and HF_DATASET_TOKEN
    and is_read_only(HF_DATASET_TOKEN_TYPE)
):
    logger.error(
        "Dataset persistence token is read-only; configure a repo-scoped "
        "fine-grained token or a classic write token."
    )

if _LEGACY_HF_STORAGE_ACTIVE and not HF_DATASET_TOKEN:
    logger.warning(
        "Legacy HF record storage is configured but no dataset token is present. "
        "POST /v1/contribute will return 503 until a write-capable token is configured."
    )
elif _LEGACY_HF_STORAGE_ACTIVE and (HF_DATASET_TOKEN_EXPLICIT or HF_WRITE_TOKEN):
    logger.info(
        "Legacy HF record storage enabled with a dedicated dataset token "
        "(inference_type=%s dataset_type=%s).",
        HF_TOKEN_TYPE,
        HF_DATASET_TOKEN_TYPE,
    )
elif _LEGACY_HF_STORAGE_ACTIVE:
    logger.warning(
        "Legacy HF record storage is using the inference-token fallback "
        "(token_type=%s). Prefer a repo-scoped fine-grained dataset token.",
        HF_TOKEN_TYPE,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP client — lifecycle managed by FastAPI lifespan
# ─────────────────────────────────────────────────────────────────────────────

#: Module-level reference to the shared httpx client.
#: Created with ``timeout=None`` so all timeout control is per-request.
_http_client: httpx.AsyncClient | None = None


def _is_non_root_process() -> bool:
    """Return a coarse least-privilege process fact without exposing the UID."""
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:  # pragma: no cover - Windows/non-POSIX deployment
        return True
    try:
        return int(geteuid()) != 0
    # defensive platform fallback
    except Exception:  # pragma: no cover # ruff: ignore[blind-except]
        return False


def _deployment_policy_error() -> str:
    """Return one bounded hardening-policy code, or ``""`` when satisfied."""
    if not _DEPLOYMENT_PROFILE_VALID:
        return "DEPLOYMENT_PROFILE_INVALID"
    if REQUIRE_NON_ROOT and not _is_non_root_process():
        return "ROOT_PROCESS_FORBIDDEN"
    if DEPLOYMENT_STRICT and _allowed_origins == ["*"]:
        return "CORS_WILDCARD_FORBIDDEN"
    if DEPLOYMENT_STRICT and SHARE_ALLOW_OPAQUE_ORIGIN_WRITE:
        return "OPAQUE_ORIGIN_WRITE_FORBIDDEN"
    return ""


def _deployment_public_status() -> dict[str, Any]:
    """Expose coarse hardening facts only; never paths, hosts, credentials or UIDs."""
    code = _deployment_policy_error()
    return {
        "profile": DEPLOYMENT_PROFILE if _DEPLOYMENT_PROFILE_VALID else "invalid",
        "strict": DEPLOYMENT_STRICT,
        "non_root": _is_non_root_process(),
        "non_root_required": REQUIRE_NON_ROOT,
        "redis_tls_required": REDIS_REQUIRE_TLS,
        "policy_ready": not bool(code),
    }


@asynccontextmanager
async def _lifespan(  # ruff: ignore[too-many-branches]
    app: FastAPI,
) -> AsyncGenerator[None, None]:
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
    global _http_client, _SHARED_RATE_LIMITER_READY, _CONTRIBUTION_LEDGER_READY, _FEEDBACK_REVIEW_LEDGER_READY, _SHARE_STORE_READY  # noqa: PLW0603
    _deployment_error = _deployment_policy_error()
    if _deployment_error:
        logger.critical(
            "Deployment hardening policy rejected startup: code=%s", _deployment_error
        )
        raise RuntimeError(_deployment_error)
    _http_client = httpx.AsyncClient(follow_redirects=False)
    _STORAGE.set_client(_http_client)
    if (
        RATE_LIMIT_BACKEND == "redis"
        and _SHARED_RATE_LIMITER is not None
        and not _RATE_LIMIT_CONFIG_ERROR
    ):
        try:
            await _SHARED_RATE_LIMITER.initialize()
            _SHARED_RATE_LIMITER_READY = True
        except RateLimitBackendError as exc:
            _SHARED_RATE_LIMITER_READY = False
            logger.error(
                "Shared rate limiter unavailable at startup: code=%s", exc.code
            )
    try:
        await _CONTRIBUTION_LEDGER.initialize()
        _CONTRIBUTION_LEDGER_READY = True
    except ContributionLedgerError as exc:
        _CONTRIBUTION_LEDGER_READY = False
        logger.error("Contribution lifecycle backend unavailable: code=%s", exc.code)
    try:
        await _FEEDBACK_REVIEW_LEDGER.initialize()
        _FEEDBACK_REVIEW_LEDGER_READY = True
    except ContributionLedgerError as exc:
        _FEEDBACK_REVIEW_LEDGER_READY = False
        logger.error("Feedback review lifecycle backend unavailable: code=%s", exc.code)
    try:
        await _SHARE_STORE.initialize()
        _SHARE_STORE_READY = True
    except ShareStoreError as exc:
        _SHARE_STORE_READY = False
        logger.error("Global Share lifecycle backend unavailable: code=%s", exc.code)
    await _STORAGE.initialize()
    for _storage_target in _STORAGE.manifest().get("targets", []):
        _cap = (_storage_target.get("token") or {}).get("write_capability") or "unknown"
        if _cap in {"missing-token", "denied", "denied-read-token"}:
            logger.error(
                "Record storage target unavailable: provider=%s code=%s",
                _storage_target.get("provider") or "unknown",
                _cap,
            )
        elif _cap in {"unverified", "legacy-unverified"}:
            logger.warning(
                "Record storage capability not pre-verified: provider=%s code=%s",
                _storage_target.get("provider") or "unknown",
                _cap,
            )
    _ledger_manifest = _CONTRIBUTION_LEDGER.manifest()
    if _CONTRIBUTION_LEDGER_CONFIG_ERROR:
        logger.error(
            "Contribution lifecycle backend invalid: code=%s",
            _CONTRIBUTION_LEDGER_CONFIG_ERROR,
        )
    if CONTRIBUTION_REQUIRE_DURABLE and not bool(_ledger_manifest.get("durable")):
        logger.error(
            "Contribution intake requires durable lifecycle storage but configured backend is non-durable."
        )
    if CONTRIBUTION_REQUIRE_SHARED and not (
        bool(_ledger_manifest.get("shared"))
        and bool(_ledger_manifest.get("authoritative"))
        and _CONTRIBUTION_LEDGER_READY
    ):
        logger.error(
            "Contribution intake requires shared authoritative lifecycle storage but it is unavailable."
        )
    logger.info(
        "Contribution lifecycle ready: backend=%s durability=%s shared=%s",
        _ledger_manifest.get("backend"),
        _ledger_manifest.get("durability"),
        bool(_ledger_manifest.get("shared")),
    )
    _feedback_review_manifest = _FEEDBACK_REVIEW_LEDGER.manifest()
    if _FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR:
        logger.error(
            "Feedback review lifecycle backend invalid: code=%s",
            _FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR,
        )
    if FEEDBACK_REVIEW_REQUIRE_DURABLE and not bool(
        _feedback_review_manifest.get("durable")
    ):
        logger.error(
            "Feedback review requires durable lifecycle storage but configured backend is non-durable."
        )
    if FEEDBACK_REVIEW_REQUIRE_SHARED and not (
        bool(_feedback_review_manifest.get("shared"))
        and bool(_feedback_review_manifest.get("authoritative"))
        and _FEEDBACK_REVIEW_LEDGER_READY
    ):
        logger.error(
            "Feedback review requires shared authoritative lifecycle storage but it is unavailable."
        )
    logger.info(
        "Feedback review lifecycle ready: mode=%s backend=%s durability=%s shared=%s",
        FEEDBACK_REVIEW_MODE,
        _feedback_review_manifest.get("backend"),
        _feedback_review_manifest.get("durability"),
        bool(_feedback_review_manifest.get("shared")),
    )
    _share_manifest = _SHARE_STORE.manifest()
    if _SHARE_STORE_CONFIG_ERROR:
        logger.error(
            "Global Share lifecycle backend invalid: code=%s", _SHARE_STORE_CONFIG_ERROR
        )
    if SHARE_REQUIRE_DURABLE and not bool(_share_manifest.get("durable")):
        logger.error(
            "Global Share requires durable storage but configured backend is non-durable."
        )
    if SHARE_REQUIRE_SHARED and not (
        bool(_share_manifest.get("shared"))
        and bool(_share_manifest.get("authoritative"))
        and _SHARE_STORE_READY
    ):
        logger.error(
            "Global Share requires shared authoritative storage but it is unavailable."
        )
    logger.info(
        "Global Share lifecycle ready: backend=%s durability=%s shared=%s",
        _share_manifest.get("backend"),
        _share_manifest.get("durability"),
        bool(_share_manifest.get("shared")),
    )
    _rl_manifest = (
        _SHARED_RATE_LIMITER.manifest()
        if _SHARED_RATE_LIMITER is not None
        else {
            "backend": "local",
            "shared": False,
            "authoritative": False,
            "consistency_scope": "process_local",
        }
    )
    logger.info(
        "Rate limit control plane: backend=%s shared=%s authoritative=%s ready=%s",
        _rl_manifest.get("backend"),
        bool(_rl_manifest.get("shared")),
        bool(_rl_manifest.get("authoritative")),
        RATE_LIMIT_BACKEND == "local" or _SHARED_RATE_LIMITER_READY,
    )
    logger.info(
        "Proxy v%s started. HTTP client ready (timeout=per-request).",
        PROXY_VERSION,
    )
    logger.info(
        "Contribution review workflow: mode=%s provider=%s canonical_branch=%s",
        CONTRIBUTION_REVIEW_MODE,
        getattr(getattr(_STORAGE, "primary", None), "provider", "none"),
        getattr(getattr(_STORAGE, "primary", None), "branch", "none"),
    )
    logger.info(
        "Routing initialized: backend_configured=%s | model_space_configured=%s | "
        "hf_token_set=%s hf_token_type=%s | dataset_token_set=%s dataset_token_type=%s | "
        "allowed_model_count=%d namespace_count=%d",
        bool(BACKEND_URL),
        bool(HF_SPACES_MODEL_URL),
        bool(HF_TOKEN),
        HF_TOKEN_TYPE,
        bool(HF_DATASET_TOKEN),
        HF_DATASET_TOKEN_TYPE,
        len(ALLOWED_MODELS),
        len(HF_SPACES_MODEL_NAMESPACES),
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
    logger.info(
        "HTTP transport ready: httpx=%s | protocol_retries=%d",
        getattr(httpx, "__version__", "unknown"),
        _protocol_retries,
    )
    try:
        yield
    finally:
        await _STORAGE.close()
        await _CONTRIBUTION_LEDGER.close()
        _CONTRIBUTION_LEDGER_READY = False
        await _FEEDBACK_REVIEW_LEDGER.close()
        _FEEDBACK_REVIEW_LEDGER_READY = False
        await _SHARE_STORE.close()
        _SHARE_STORE_READY = False
        if _SHARED_RATE_LIMITER is not None:
            await _SHARED_RATE_LIMITER.close()
        _SHARED_RATE_LIMITER_READY = False
        await _http_client.aclose()
        _STORAGE.set_client(None)
        _http_client = None
        logger.info("Proxy shutdown. HTTP client closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="sphinx-ai-assistant proxy",
    description=(
        "Server-authoritative chat proxy for sphinx-ai-assistant. "
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
    allow_origins=_cors_allowed_origins,
    allow_methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    # Authorization added for write endpoints (POST /v1/feedback, POST /v1/contribute)
    # that validate a Bearer token.  Without this the browser preflight rejects
    # requests containing Authorization headers before the handler runs.
    # HEAD added so the JS _pingUrl health-check and the HF Space internal health
    # monitor can send cross-origin HEAD / and HEAD /health without a CORS error.
    # PATCH added for PATCH /v1/share/{uuid} (content update, URL preserved).
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Share-Edit-Token",
        "X-Contribution-Delete-Token",
        "X-Feedback-Review-Token",
        "X-AI-Operation-Id",
        "X-AI-Resource-Id",
        "X-AI-Management-Token-Hash",
        "X-AI-Operation-Created-At",
    ],
    allow_credentials=False,
)


@app.middleware("http")
async def _browser_origin_guard(request: Request, call_next):
    """Reject disallowed browser origins before handlers perform work/writes."""
    if not _origin_allowed(request):
        raw_origin = request.headers.get("origin", "").strip()
        if raw_origin == "null":
            mode = _opaque_share_request_mode(request)
            detail = (
                "Opaque browser origin not allowed. Read-only local-file Share "
                "compatibility requires SHARE_ALLOW_OPAQUE_ORIGIN=true. "
                + (
                    "Opaque-origin mutation additionally requires the high-risk "
                    "SHARE_ALLOW_OPAQUE_ORIGIN_WRITE=true opt-in, which strict deployments forbid."
                    if mode == "write"
                    else "This compatibility also admits sandboxed opaque documents, so keep it off unless required."
                )
            )
        else:
            detail = (
                "Origin not allowed. Official Scikit-Plots documentation origin "
                "is always permitted by proxy v6.5.1+."
            )
        return JSONResponse(
            {"detail": detail},
            status_code=403,
            headers={"Cache-Control": "no-store"},
        )
    return await call_next(request)


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
        backend_auth_token=BACKEND_AUTH_TOKEN,
        hf_spaces_auth_token=HF_SPACES_AUTH_TOKEN,
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
    return await _read_limited_body(request, MAX_BODY_BYTES, "Request")


def _local_protocol_reason(exc: BaseException) -> str:
    """Map a HTTPX local protocol exception to a privacy-safe fixed reason.

    HTTP protocol exception messages may contain a rejected header value, so
    callers must never log the raw exception text.  This helper inspects it
    only to return one of a small fixed set of diagnostic labels.
    """
    text = str(exc).lower()
    if "illegal header value" in text:
        return "illegal-header"
    if "content-length" in text and "too much data" in text:
        return "content-length-overrun"
    if "content-length" in text and "too little data" in text:
        return "content-length-underrun"
    if "host" in text and ("mandatory" in text or "missing" in text):
        return "missing-host"
    if "request line" in text:
        return "request-line"
    return "unspecified"


async def _stub_intercept(body: bytes, headers: Any) -> Response | None:
    """Answer a reserved ``stub/*`` request locally, or return ``None``.

    ``stub/*`` is a fail-closed namespace: once a request selects it, the
    request is never forwarded to any real inference provider.  When the test
    rig is disabled we return HTTP 503 locally.  This prevents a disabled stub
    from accidentally entering Path 3 and turning a diagnostic request into an
    upstream request.

    The slow-stub delay uses :func:`asyncio.sleep` so one diagnostic request
    cannot block every other request on the FastAPI event loop.
    """
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or not is_stub_model(payload.get("model")):
        return None

    if not STUB_ENABLED:
        logger.warning("AI proxy [STUB_DISABLED]: reserved stub model requested.")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "type": "stub_disabled",
                    "code": "stub_disabled",
                    "message": (
                        "The deterministic stub responder is disabled on this proxy."
                    ),
                }
            },
            headers={"X-Stub-Model": "disabled"},
        )

    model = payload.get("model")
    mode, _arg = parse_stub_mode(model)
    # Do not log the model/request body; mode is a fixed registry value and the
    # body size is sufficient for diagnostics without exposing user content.
    logger.info("stub request: mode=%s bytes=%d", mode, len(body))

    delay_ms = stub_delay_ms(model)
    if delay_ms:
        await asyncio.sleep(delay_ms / 1000.0)

    if payload.get("stream"):
        frames = stub_sse_frames(model, payload, headers)

        async def _gen() -> AsyncGenerator[bytes, None]:
            for frame in frames:
                yield frame.encode()

        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Stub-Model": "true",
            },
        )

    status, doc = stub_payload(model, payload, headers, created=int(_time.time()))
    return JSONResponse(doc, status_code=status, headers={"X-Stub-Model": "true"})


async def _forward(  # ruff: ignore[too-many-branches, too-many-return-statements]
    body: bytes,
    *,
    structured_body: bytes | None = None,
) -> Response:
    """Forward one OpenAI-compatible request with mode-aware stream bridging.

    Streaming is treated as a negotiated transport, not an assumption.  The
    proxy opens the upstream request *before* returning a downstream 200, then
    inspects the actual upstream ``Content-Type``:

    * true ``text/event-stream`` responses are proxied incrementally;
    * ordinary JSON responses stay ordinary JSON, even when the caller asked
      for streaming (the panel already has a JSON fallback parser);
    * pre-header transport/protocol failures become real HTTP 502/504 responses;
    * mid-stream failures become explicit ``event: error`` SSE frames so the
      panel cannot silently finalize an empty assistant bubble.

    A remote protocol/read failure may be retried a bounded number of times
    before any output is exposed.  Local protocol failures are never retried.
    """
    if _http_client is None:
        raise RuntimeError(
            "HTTP client is not initialised. "
            "FastAPI lifespan may not have started correctly."
        )

    url, headers, read_timeout_s = _resolve_url(body)
    req_timeout = _make_timeout(read_timeout_s)

    # Path 2 is another bundled server-authoritative service.  Preserve the
    # typed public contract across that hop so the model Space independently
    # constructs its system policy instead of trusting proxy-authored messages.
    # Path 1 custom backends and Path 3 provider APIs still receive the
    # server-constructed provider body.
    _path2_contract = bool(
        structured_body is not None
        and not BACKEND_URL
        and HF_SPACES_MODEL_URL
        and url.rstrip("/") == HF_SPACES_MODEL_URL.rstrip("/")
    )
    wire_body = structured_body if _path2_contract else body

    stream_requested = False
    try:
        payload: Any = json.loads(wire_body)
        stream_requested = bool(payload.get("stream", False))
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        payload = None

    def _error_response(status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={
                "code": code,
                "error": {"type": "upstream_error", "code": code, "message": message},
            },
            headers={"Cache-Control": "no-store"},
        )

    def _sse_error_frame(status: int, code: str, message: str) -> bytes:
        err_id = uuid.uuid4().hex
        doc = json.dumps(
            {
                "id": f"err-{err_id}",
                "error": {"status": status, "code": code, "message": message},
            },
            separators=(",", ":"),
        )
        return f"event: error\ndata: {doc}\n\n".encode()

    def _body_with_stream(value: bool) -> bytes:
        if not isinstance(payload, dict):
            return wire_body
        clone = dict(payload)
        clone["stream"] = bool(value)
        return json.dumps(clone, ensure_ascii=False, separators=(",", ":")).encode()

    async def _post_buffered(  # ruff: ignore[too-many-return-statements]
        request_body: bytes,
    ) -> Response:
        upstream: httpx.Response | None = None
        try:
            request = _http_client.build_request(
                "POST", url, content=request_body, headers=headers, timeout=req_timeout
            )
            upstream = await _http_client.send(request, stream=True)
            status = upstream.status_code
            if status < 200 or status >= 300:  # ruff: ignore[magic-value-comparison]
                code = _upstream_public_error_code(status)
                logger.warning(
                    "AI proxy [UPSTREAM_STATUS]: upstream rejected non-streaming request status=%d code=%s",
                    status,
                    code,
                )
                return _error_response(
                    status, code, "The upstream AI provider rejected the request."
                )
            media_type = upstream.headers.get("content-type", "application/json")
            content = await _read_upstream_limited(upstream)
        except _UpstreamResponseTooLarge:
            logger.warning(
                "AI proxy [UPSTREAM_RESPONSE_LIMIT]: upstream response exceeded the application byte ceiling"
            )
            return _error_response(
                502,
                "upstream_response_too_large",
                "The upstream response exceeded the proxy safety limit.",
            )
        except httpx.ReadTimeout:
            logger.warning(
                "AI proxy [UPSTREAM_TIMEOUT]: non-streaming request timed out after %.0f s",
                read_timeout_s,
            )
            return _error_response(
                504,
                "upstream_timeout",
                f"Upstream timed out after {read_timeout_s:.0f} s.",
            )
        except httpx.ConnectTimeout:
            logger.warning(
                "AI proxy [UPSTREAM_CONNECT_TIMEOUT]: non-streaming connect timed out"
            )
            return _error_response(
                504,
                "upstream_connect_timeout",
                "Connection to the upstream timed out.",
            )
        except httpx.LocalProtocolError as exc:
            logger.error(
                "AI proxy [UPSTREAM_PROTOCOL_LOCAL]: local HTTP protocol validation failed reason=%s.",
                _local_protocol_reason(exc),
            )
            return _error_response(
                502,
                "upstream_local_protocol_error",
                "The proxy could not construct a valid upstream HTTP request.",
            )
        except httpx.RemoteProtocolError:
            logger.warning(
                "AI proxy [UPSTREAM_PROTOCOL_REMOTE]: upstream returned invalid/incomplete HTTP."
            )
            return _error_response(
                502,
                "upstream_remote_protocol_error",
                "The upstream service returned an invalid or incomplete HTTP response.",
            )
        except httpx.RequestError as exc:
            logger.warning(
                "AI proxy [UPSTREAM_REQUEST_ERROR]: non-streaming upstream request failed (%s)",
                type(exc).__name__,
            )
            return _error_response(
                502,
                "upstream_request_error",
                "Failed to reach the upstream service.",
            )
        finally:
            if upstream is not None and not upstream.is_closed:
                await upstream.aclose()

        return Response(
            content=content,
            status_code=status,
            media_type=media_type,
        )

    if not stream_requested:
        return await _post_buffered(wire_body)

    # Streaming intent is advisory: ask upstream for SSE, but accept a normal
    # JSON completion from a backend that does not implement streaming.
    stream_headers = dict(headers)
    stream_headers.setdefault("Accept", "text/event-stream, application/json")

    upstream: httpx.Response | None = None
    attempt = 0
    while True:
        try:
            request = _http_client.build_request(
                "POST",
                url,
                content=wire_body,
                headers=stream_headers,
                timeout=req_timeout,
            )
            upstream = await _http_client.send(request, stream=True)
            break
        except httpx.ReadTimeout:
            logger.warning(
                "AI proxy [UPSTREAM_TIMEOUT]: streaming request timed out before headers after %.0f s",
                read_timeout_s,
            )
            return _error_response(
                504,
                "upstream_timeout",
                f"Upstream timed out after {read_timeout_s:.0f} s.",
            )
        except httpx.ConnectTimeout:
            logger.warning(
                "AI proxy [UPSTREAM_CONNECT_TIMEOUT]: streaming connect timed out before headers"
            )
            return _error_response(
                504,
                "upstream_connect_timeout",
                "Connection to the upstream timed out.",
            )
        except httpx.LocalProtocolError as exc:
            logger.error(
                "AI proxy [UPSTREAM_PROTOCOL_LOCAL]: local HTTP protocol validation failed before headers reason=%s.",
                _local_protocol_reason(exc),
            )
            return _error_response(
                502,
                "upstream_local_protocol_error",
                "The proxy could not construct a valid upstream HTTP request.",
            )
        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            if attempt < _protocol_retries:
                attempt += 1
                logger.warning(
                    "AI proxy [UPSTREAM_PROTOCOL_RETRY]: retrying pre-output upstream stream (%s attempt=%d).",
                    type(exc).__name__,
                    attempt,
                )
                continue
            logger.warning(
                "AI proxy [UPSTREAM_PROTOCOL_REMOTE]: streaming upstream failed before headers (%s).",
                type(exc).__name__,
            )
            return _error_response(
                502,
                "upstream_remote_protocol_error",
                "The upstream service returned an invalid or incomplete HTTP response.",
            )
        except httpx.RequestError as exc:
            logger.warning(
                "AI proxy [UPSTREAM_REQUEST_ERROR]: streaming upstream request failed before headers (%s)",
                type(exc).__name__,
            )
            return _error_response(
                502,
                "upstream_request_error",
                "Failed to reach the upstream service.",
            )

    assert upstream is not None  # ruff: ignore[assert]

    if upstream.status_code != 200:  # noqa: PLR2004
        status = upstream.status_code
        await upstream.aclose()
        logger.warning(
            "AI proxy [UPSTREAM_STATUS]: upstream rejected streaming request status=%d",
            status,
        )
        return _error_response(
            status,
            _upstream_public_error_code(status),
            "The upstream AI provider rejected the request.",
        )

    try:
        _check_upstream_declared_length(upstream)
    except _UpstreamResponseTooLarge:
        await upstream.aclose()
        logger.warning(
            "AI proxy [UPSTREAM_RESPONSE_LIMIT]: declared upstream response exceeded the application byte ceiling"
        )
        return _error_response(
            502,
            "upstream_response_too_large",
            "The upstream response exceeded the proxy safety limit.",
        )

    content_type = (upstream.headers.get("content-type") or "").lower()
    if "text/event-stream" not in content_type:
        # The custom ai-model Space currently returns an ordinary OpenAI JSON
        # response even when ``stream:true`` was requested.  Preserve that mode
        # instead of falsely labelling raw JSON bytes as SSE (which the browser
        # would correctly ignore, producing an empty answer).
        try:
            content = await _read_upstream_limited(upstream)
        except _UpstreamResponseTooLarge:
            await upstream.aclose()
            logger.warning(
                "AI proxy [UPSTREAM_RESPONSE_LIMIT]: buffered JSON fallback exceeded the application byte ceiling"
            )
            return _error_response(
                502,
                "upstream_response_too_large",
                "The upstream response exceeded the proxy safety limit.",
            )
        except httpx.ReadTimeout:
            await upstream.aclose()
            logger.warning(
                "AI proxy [UPSTREAM_TIMEOUT]: buffered JSON fallback timed out after %.0f s",
                read_timeout_s,
            )
            return _error_response(
                504,
                "upstream_timeout",
                "The upstream response timed out before it was complete.",
            )
        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            await upstream.aclose()
            logger.warning(
                "AI proxy [UPSTREAM_JSON_STREAM_FALLBACK]: buffered stream read failed (%s); retrying once without streaming.",
                type(exc).__name__,
            )
            # No downstream bytes have been committed yet, so a single
            # non-streaming retry cannot duplicate visible output.
            return await _post_buffered(_body_with_stream(False))
        except httpx.LocalProtocolError as exc:
            await upstream.aclose()
            logger.error(
                "AI proxy [UPSTREAM_PROTOCOL_LOCAL]: local protocol failure while buffering JSON fallback reason=%s.",
                _local_protocol_reason(exc),
            )
            return _error_response(
                502,
                "upstream_local_protocol_error",
                "The proxy could not complete a valid upstream HTTP exchange.",
            )
        finally:
            if not upstream.is_closed:
                await upstream.aclose()

        if not content.strip():
            logger.warning(
                "AI proxy [UPSTREAM_EMPTY_RESPONSE]: upstream returned HTTP 200 with an empty body."
            )
            return _error_response(
                502,
                "upstream_empty_response",
                "The upstream service returned an empty response.",
            )

        # Some reverse proxies buffer a valid SSE response but lose or rewrite
        # its Content-Type.  A bounded prefix sniff lets us recover that format
        # without interpreting arbitrary provider payloads.
        prefix = content.lstrip()[:32].lower()
        if prefix.startswith((b"data:", b"event:")):
            return Response(
                content=content,
                status_code=200,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                },
            )

        return Response(
            content=content,
            status_code=200,
            media_type=upstream.headers.get("content-type", "application/json"),
        )

    async def _sse_chunks() -> AsyncGenerator[bytes, None]:
        """Proxy a confirmed SSE upstream and make terminal failures explicit."""
        emitted = False
        saw_data_field = False
        probe_tail = b""
        total_bytes = 0
        try:
            async for chunk in upstream.aiter_bytes():
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > MAX_UPSTREAM_RESPONSE_BYTES:
                    logger.warning(
                        "AI proxy [UPSTREAM_RESPONSE_LIMIT]: SSE response exceeded the application byte ceiling"
                    )
                    yield _sse_error_frame(
                        502,
                        "upstream_response_too_large",
                        "The upstream response exceeded the proxy safety limit.",
                    )
                    return
                emitted = True
                probe = probe_tail + chunk.lower()
                if b"data:" in probe:
                    saw_data_field = True
                probe_tail = probe[-5:]
                yield chunk
            if not saw_data_field:
                logger.warning(
                    "AI proxy [UPSTREAM_EMPTY_STREAM]: SSE upstream ended without a data field."
                )
                yield _sse_error_frame(
                    502,
                    "upstream_empty_stream",
                    "The upstream stream ended without a completion. Please retry.",
                )
        except asyncio.CancelledError:
            logger.info(
                "AI proxy [STREAM_CLOSED]: downstream streaming connection closed; upstream stream cancelled."
            )
            raise
        except (BrokenPipeError, ConnectionResetError):
            logger.warning(
                "AI proxy [STREAM_CLOSED]: downstream pipe closed; stream ended safely."
            )
            return
        except httpx.ReadTimeout:
            logger.warning(
                "AI proxy [UPSTREAM_TIMEOUT]: SSE body timed out after %.0f s",
                read_timeout_s,
            )
            yield _sse_error_frame(
                504,
                "upstream_timeout",
                "The upstream stream timed out. Please retry.",
            )
        except httpx.LocalProtocolError as exc:
            logger.error(
                "AI proxy [UPSTREAM_PROTOCOL_LOCAL]: local protocol failure during SSE body reason=%s.",
                _local_protocol_reason(exc),
            )
            yield _sse_error_frame(
                502,
                "upstream_local_protocol_error",
                "The proxy encountered an HTTP protocol error. Please retry.",
            )
        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            logger.warning(
                "AI proxy [UPSTREAM_PROTOCOL_REMOTE]: SSE body ended unexpectedly (%s emitted=%s).",
                type(exc).__name__,
                emitted,
            )
            yield _sse_error_frame(
                502,
                "upstream_remote_protocol_error",
                "The upstream stream closed unexpectedly. Please retry.",
            )
        except httpx.RequestError as exc:
            logger.warning(
                "AI proxy [UPSTREAM_REQUEST_ERROR]: SSE body failed (%s emitted=%s).",
                type(exc).__name__,
                emitted,
            )
            yield _sse_error_frame(
                502,
                "upstream_request_error",
                "The upstream streaming connection failed. Please retry.",
            )
        finally:
            await upstream.aclose()

    return StreamingResponse(
        _sse_chunks(),
        status_code=200,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
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


def _contribution_pipeline_ready() -> bool:
    """Return coarse contribution readiness without exposing ledger topology."""
    if _CONTRIBUTION_LEDGER_CONFIG_ERROR:
        return False
    ledger = _CONTRIBUTION_LEDGER.manifest()
    if not _CONTRIBUTION_LEDGER_READY:
        return False
    if CONTRIBUTION_REQUIRE_DURABLE and not bool(ledger.get("durable")):
        return False
    if CONTRIBUTION_REQUIRE_SHARED and not (
        bool(ledger.get("shared")) and bool(ledger.get("authoritative"))
    ):
        return False
    return _STORAGE.primary_ready()


def _share_store_public_status() -> dict[str, Any]:
    """Expose only coarse Share lifecycle properties; never connection details."""
    manifest = _SHARE_STORE.manifest()
    ready = bool(_SHARE_STORE_READY and not _SHARE_STORE_CONFIG_ERROR)
    if SHARE_REQUIRE_DURABLE and not bool(manifest.get("durable")):
        ready = False
    if SHARE_REQUIRE_SHARED and not (
        bool(manifest.get("shared")) and bool(manifest.get("authoritative"))
    ):
        ready = False
    return {
        "ready": ready,
        "backend": manifest.get("backend"),
        "durability": manifest.get("durability"),
        "durable": bool(manifest.get("durable")),
        "shared": bool(manifest.get("shared")),
        "authoritative": bool(manifest.get("authoritative")),
        "consistency_scope": manifest.get("consistency_scope"),
        "required_durable": SHARE_REQUIRE_DURABLE,
        "required_shared": SHARE_REQUIRE_SHARED,
    }


def _require_share_store_ready() -> dict[str, Any]:
    status = _share_store_public_status()
    if not status["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Global Share lifecycle storage is unavailable for this deployment.",
        )
    return status


def _sync_share_store_runtime_limits() -> None:
    """Keep backend limits aligned with runtime config/test overrides."""
    if hasattr(_SHARE_STORE, "max_entries"):
        _SHARE_STORE.max_entries = int(SHARE_MAX_ENTRIES)
    if hasattr(_SHARE_STORE, "max_total_bytes"):
        _SHARE_STORE.max_total_bytes = int(SHARE_MAX_TOTAL_BYTES)


def _share_store_http_error(exc: ShareStoreError) -> HTTPException:
    mapping = {
        "NOT_FOUND": (404, "Share not found or expired."),
        "EXPIRED": (410, "Share has expired."),
        "AUTH": (403, "Invalid share edit capability."),
        "DUPLICATE_SHARE": (409, "Share operation already exists."),
        "ENTRY_CAPACITY": (507, "Share storage entry capacity reached."),
        "BYTE_CAPACITY": (507, "Share storage byte capacity reached."),
    }
    status, detail = mapping.get(exc.code, (503, "Share storage unavailable."))
    return HTTPException(status_code=status, detail=detail)


def _operation_envelope(
    request: Request, payload: dict[str, Any], purpose: str, *, max_age_ms: int
) -> tuple[str, str, str] | None:
    """Resolve a create-once envelope without receiving the raw management capability.

    The browser creates the public resource locator and the management token
    before the request, but transmits only ``SHA-256(management_token)``.  The
    server therefore has enough information to authorize later revoke/delete
    operations without ever seeing the create-time bearer capability itself.
    ``purpose`` is retained in the signature to keep call sites explicit.
    """
    del payload, purpose  # operation metadata is header-only; never content JSON
    op_id = request.headers.get("X-AI-Operation-Id")
    resource_id = request.headers.get("X-AI-Resource-Id")
    token_hash = request.headers.get("X-AI-Management-Token-Hash")
    created_raw = request.headers.get("X-AI-Operation-Created-At")
    if (
        op_id is None
        and resource_id is None
        and token_hash is None
        and created_raw is None
    ):
        return None
    if (
        not isinstance(op_id, str)
        or not (16 <= len(op_id) <= 128)  # ruff: ignore[magic-value-comparison]
        or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for ch in op_id
        )
    ):
        raise HTTPException(status_code=422, detail="Invalid operationId.")
    if (
        not isinstance(resource_id, str)
        or len(resource_id) != 32  # ruff: ignore[magic-value-comparison]
        or any(ch not in "0123456789abcdefABCDEF" for ch in resource_id)
    ):
        raise HTTPException(
            status_code=422, detail="Invalid operation resource locator."
        )
    if (
        not isinstance(token_hash, str)
        or len(token_hash) != 64  # ruff: ignore[magic-value-comparison]
        or any(ch not in "0123456789abcdefABCDEF" for ch in token_hash)
    ):
        raise HTTPException(
            status_code=422, detail="Invalid management capability digest."
        )
    try:
        created_at = int(created_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="Invalid operationCreatedAt."
        ) from exc
    now_ms = int(_time.time() * 1000)
    if (
        created_at > now_ms + 300_000
        or created_at <= 0
        or now_ms - created_at > max(60_000, int(max_age_ms))
    ):
        raise HTTPException(
            status_code=409,
            detail="Operation recovery window has expired; start a new reviewed operation.",
        )
    return resource_id.lower(), token_hash.lower(), op_id


def _canonical_payload_digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cors_public_status() -> dict[str, Any]:
    """Return public CORS diagnostics without exposing custom/private origins."""
    wildcard = _allowed_origins == ["*"]
    primary_default = _DEFAULT_ALLOWED_ORIGINS[0]
    return {
        # Backward-compatible singular fields refer to the primary project origin.
        "official_docs_origin": primary_default,
        "official_docs_origin_allowed": wildcard or primary_default in _allowed_origins,
        "default_allowed_origin_count": len(_DEFAULT_ALLOWED_ORIGINS),
        "default_allowed_origins_allowed": (
            wildcard
            or all(origin in _allowed_origins for origin in _DEFAULT_ALLOWED_ORIGINS)
        ),
        "wildcard": wildcard,
        "allowed_origin_count": None if wildcard else len(_allowed_origins),
        "env_semantics": ALLOWED_ORIGINS_MODE,
        "share_opaque_origin_allowed": SHARE_ALLOW_OPAQUE_ORIGIN,
        "share_opaque_origin_write_allowed": bool(
            SHARE_ALLOW_OPAQUE_ORIGIN and SHARE_ALLOW_OPAQUE_ORIGIN_WRITE
        ),
    }


@app.get("/")
async def root() -> JSONResponse:
    """Public status/discovery with privacy-minimized collection capabilities."""
    storage_manifest = _STORAGE.manifest()
    storage_targets = (
        storage_manifest.get("targets", [])
        if isinstance(storage_manifest, dict)
        else []
    )
    return JSONResponse(
        {
            "status": "ok",
            "service": "sphinx-ai-assistant proxy",
            "version": PROXY_VERSION,
            "deployment": _deployment_public_status(),
            "capabilities": _reasoning_capability(),
            "training": {
                "dataset_repo": None,
                "contribute_ready": _contribution_pipeline_ready(),
                "feedback_persist_enabled": FEEDBACK_PERSIST_ENABLED,
                "feedback_telemetry_schema_version": FEEDBACK_TELEMETRY_SCHEMA_VERSION,
                "feedback_telemetry_consent_version": (
                    FEEDBACK_TELEMETRY_CONSENT_VERSION
                ),
                "feedback_review_mode": FEEDBACK_REVIEW_MODE,
                "feedback_review_ready": _feedback_review_pipeline_ready(),
                "feedback_review_consent_version": FEEDBACK_REVIEW_CONSENT_VERSION,
                "feedback_training_consent_version": FEEDBACK_TRAINING_CONSENT_VERSION,
                "feedback_review_updates": True,
                "contribution_review_mode": CONTRIBUTION_REVIEW_MODE,
                "pending_review_updates": True,
                "duplicate_resubmit_policy": "same-receipt-noop-or-update",
                "canonical_branch": getattr(
                    getattr(_STORAGE, "primary", None), "branch", None
                ),
            },
            "tokens": {
                "hf_token_type": "unknown",
                "hf_dataset_token_type": "unknown",
                "hf_write_token_type": "unknown",
                "least_privilege_mode": bool(
                    HF_DATASET_TOKEN_EXPLICIT or HF_WRITE_TOKEN
                ),
            },
            "storage": {
                "configured": bool(storage_targets),
                "target_count": len(storage_targets),
                "primary_ready": _STORAGE.primary_ready(),
            },
            "share": _share_store_public_status(),
            "rate_limit": {
                "backend": RATE_LIMIT_BACKEND,
                "shared": RATE_LIMIT_BACKEND == "redis",
                "authoritative": RATE_LIMIT_BACKEND == "redis",
                "ready": (
                    (RATE_LIMIT_BACKEND == "redis" and _SHARED_RATE_LIMITER_READY)
                    or (RATE_LIMIT_BACKEND == "local" and not RATE_LIMIT_REQUIRE_SHARED)
                ),
                "required_shared": RATE_LIMIT_REQUIRE_SHARED,
                "scope": (
                    "single_redis_consistency_domain"
                    if RATE_LIMIT_BACKEND == "redis"
                    else "process_local"
                ),
            },
            "limits": {"max_upstream_response_bytes": MAX_UPSTREAM_RESPONSE_BYTES},
            "cors": _cors_public_status(),
        },
        headers={"Cache-Control": "no-store"},
    )


def _reasoning_capability() -> dict:
    """Describe the reasoning parameters this proxy forwards.

    Returned inside ``/health`` so the documentation panel can discover what
    to send instead of relying on a hand-written declaration in the site's
    ``conf.py`` -- which states a fact about this proxy from another
    repository and goes stale the moment this one changes.

    Returns
    -------
    dict
        ``{"reasoning": {...}}``.  ``enabled`` is ``False`` unless
        ``REASONING_ENABLED`` is set, and an explicit ``False`` is a useful
        answer in its own right: it pins the panel's controls to the
        provider's defaults even where the site optimistically enabled them.

    Notes
    -----
    **Developer note** -- The effort map is sent in full for all five panel
    levels.  A partial map would leave the unmapped levels silently sending
    nothing, so the panel rejects one; sending every level keeps this endpoint
    the single place where the mapping is decided.  ``extra`` and ``max``
    collapse onto the top value of the three-value OpenAI-compatible scale.

    **Security note** -- This response is untrusted input from the panel's
    point of view, and the panel validates it accordingly: field names must
    match a strict pattern and miss a reserved list, so nothing declared here
    can override ``model``, ``messages``, or any other field that decides what
    is sent or to whom.  Widening what this returns will not widen what the
    panel accepts.
    """
    if not REASONING_ENABLED:
        return {
            "reasoning": {"enabled": False},
            "chat_request": {"contract": CHAT_CONTRACT},
            "stub": _stub_capability(),
        }

    caps: dict = {
        "enabled": True,
        "budget_min": REASONING_BUDGET_MIN,
        "budget_max": REASONING_BUDGET_MAX,
    }
    caps["effort_enabled"] = bool(REASONING_EFFORT_PARAM)
    caps["thinking_enabled"] = bool(REASONING_THINKING_PARAM)
    if REASONING_EFFORT_PARAM:
        caps["effort_param"] = REASONING_EFFORT_PARAM
        caps["effort_values"] = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "extra": "high",
            "max": "high",
        }
    if REASONING_THINKING_PARAM:
        caps["thinking_param"] = REASONING_THINKING_PARAM
        caps["thinking_mode"] = REASONING_THINKING_MODE
    return {
        "reasoning": caps,
        "chat_request": {"contract": CHAT_CONTRACT},
        "stub": _stub_capability(),
    }


def _stub_capability() -> dict:
    """Advertise the stub rig and the modes this deployment actually supports.

    Read from the mode registry rather than a literal list, so a deployment
    that registers an extra mode advertises it without a second edit -- and a
    client can discover the real set instead of assuming one that may be older
    than the server.

    Returns
    -------
    dict
        ``{"enabled": bool, "prefix": "stub/", "modes": {name: summary}}``.
        ``modes`` is omitted when disabled: a disabled rig should not publish
        a menu of scenarios it will not run.
    """
    if not STUB_ENABLED:
        return {"enabled": False}
    return {"enabled": True, "prefix": "stub/", "modes": stub_modes()}


@app.get("/health")
async def health() -> JSONResponse:
    """
    Minimal liveness probe for container orchestrators and uptime monitors.

    Returns
    -------
    JSONResponse
        Always HTTP 200 while the process is running.
    """
    return JSONResponse(
        {
            "status": "ok",
            "version": PROXY_VERSION,
            "deployment": _deployment_public_status(),
            "capabilities": _reasoning_capability(),
            "share": _share_store_public_status(),
            "rate_limit": {
                "backend": RATE_LIMIT_BACKEND,
                "shared": RATE_LIMIT_BACKEND == "redis",
                "authoritative": RATE_LIMIT_BACKEND == "redis",
                "ready": (
                    (RATE_LIMIT_BACKEND == "redis" and _SHARED_RATE_LIMITER_READY)
                    or (RATE_LIMIT_BACKEND == "local" and not RATE_LIMIT_REQUIRE_SHARED)
                ),
                "required_shared": RATE_LIMIT_REQUIRE_SHARED,
            },
            "limits": {"max_upstream_response_bytes": MAX_UPSTREAM_RESPONSE_BYTES},
            "cors": _cors_public_status(),
        },
        headers={"Cache-Control": "no-store"},
    )


def _chat_contract_error_code(exc: ChatContractError) -> str:
    """Map a local chat-contract rejection to a bounded public reason code."""
    text = str(exc)
    if text == "requested model is not allowed by this proxy" or text.startswith(
        "model "
    ):
        return "PROXY_MODEL_NOT_ALLOWED"
    if text.startswith("context") or "context field" in text:
        return "PROXY_CONTEXT_INVALID"
    if text.startswith("reasoning") or "reasoning field" in text:
        return "PROXY_REASONING_INVALID"
    if text.startswith("user_message"):
        return "PROXY_USER_MESSAGE_INVALID"
    return "PROXY_CHAT_CONTRACT_INVALID"


def _chat_contract_error_response(exc: ChatContractError) -> JSONResponse:
    """Return a privacy-safe machine-readable local 400 response."""
    code = _chat_contract_error_code(exc)
    logger.warning("AI proxy [%s]: chat request rejected locally.", code)
    return JSONResponse(
        status_code=400,
        content={
            "code": code,
            "error": {
                "type": "proxy_request_rejected",
                "code": code,
                "message": "The AI proxy rejected the request before provider routing.",
            },
        },
        headers={"Cache-Control": "no-store"},
    )


def _upstream_public_error_code(status: int) -> str:
    """Map provider HTTP status to the browser's bounded diagnostic vocabulary."""
    if status in {401, 403}:
        return "UPSTREAM_AUTH_OR_ACCESS_REJECTED"
    if status == 404:  # ruff: ignore[magic-value-comparison]
        return "UPSTREAM_MODEL_OR_ROUTE_NOT_FOUND"
    if status == 429:  # ruff: ignore[magic-value-comparison]
        return "UPSTREAM_RATE_LIMITED"
    if 500 <= status <= 599:  # ruff: ignore[magic-value-comparison]
        return "UPSTREAM_SERVICE_ERROR"
    return "UPSTREAM_REQUEST_REJECTED"


def _server_owned_chat_body(body: bytes) -> bytes:
    """Validate the public chat envelope and build the provider request."""
    req = parse_chat_request(
        body,
        allowed_models=ALLOWED_MODELS,
        allowed_namespaces=HF_SPACES_MODEL_NAMESPACES,
    )
    return encode_upstream_payload(
        req,
        reasoning_enabled=REASONING_ENABLED,
        effort_param=REASONING_EFFORT_PARAM if REASONING_ENABLED else "",
        thinking_param=REASONING_THINKING_PARAM if REASONING_ENABLED else "",
        thinking_mode=REASONING_THINKING_MODE,
        budget_min=REASONING_BUDGET_MIN,
        budget_max=REASONING_BUDGET_MAX,
    )


@app.post("/v1/chat/completions")
async def chat_completions(  # ruff: ignore[undocumented-param]
    request: Request, body: bytes = Depends(_validated_body)
) -> Response:
    """
    Primary proxy endpoint using the ``scikitplot-chat-v1`` client contract.

    Parameters
    ----------
    body : bytes
        ``scikitplot-chat-v1`` request body, pre-validated for size by :func:`_validated_body`.

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
    client_ip = _client_ip(request)
    allowed, _count = await _consume_rate_limit(
        _chat_rl,
        _chat_rl_lock,
        client_ip,
        limit=CHAT_RATE_LIMIT_PER_HOUR,
        scope="chat",
    )
    if not allowed:
        logger.warning(
            json.dumps({"event": "chat.ratelimit", "ip": _mask_ip(client_ip)})
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for chat requests.",
            headers={"Retry-After": "3600"},
        )
    stubbed = await _stub_intercept(body, request.headers)
    if stubbed is not None:
        return stubbed
    try:
        upstream_body = _server_owned_chat_body(body)
    except ChatContractError as exc:
        return _chat_contract_error_response(exc)
    return await _forward(upstream_body, structured_body=body)


@app.post("/")
async def chat_completions_alias(  # ruff: ignore[undocumented-param]
    request: Request, body: bytes = Depends(_validated_body)
) -> Response:
    """
    Path-agnostic alias: ``POST /`` → identical to ``POST /v1/chat/completions``.

    Parameters
    ----------
    body : bytes
        ``scikitplot-chat-v1`` request body, pre-validated for size by :func:`_validated_body`.

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
    client_ip = _client_ip(request)
    allowed, _count = await _consume_rate_limit(
        _chat_rl,
        _chat_rl_lock,
        client_ip,
        limit=CHAT_RATE_LIMIT_PER_HOUR,
        scope="chat",
    )
    if not allowed:
        logger.warning(
            json.dumps({"event": "chat.ratelimit", "ip": _mask_ip(client_ip)})
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for chat requests.",
            headers={"Retry-After": "3600"},
        )
    stubbed = await _stub_intercept(body, request.headers)
    if stubbed is not None:
        return stubbed
    try:
        upstream_body = _server_owned_chat_body(body)
    except ChatContractError as exc:
        return _chat_contract_error_response(exc)
    return await _forward(upstream_body, structured_body=body)


def _contribution_ledger_http_error(  # ruff: ignore[too-many-return-statements]
    exc: ContributionLedgerError,
) -> HTTPException:
    """Map private ledger state codes to bounded public lifecycle semantics."""
    code = exc.code
    if code == "NOT_FOUND":
        return HTTPException(status_code=404, detail="Contribution receipt not found.")
    if code == "EXPIRED":
        return HTTPException(status_code=410, detail="Pending contribution expired.")
    if code in {"PROMOTION_IN_PROGRESS", "WITHDRAWAL_IN_PROGRESS", "BUSY"}:
        return HTTPException(
            status_code=409,
            detail="Contribution lifecycle operation already in progress.",
        )
    if code in {"RECONCILIATION_REQUIRED", "STALE_CLAIM"}:
        return HTTPException(
            status_code=409,
            detail="Contribution lifecycle outcome requires reconciliation.",
        )
    if code in {"PENDING_CAPACITY", "PENDING_BYTE_CAPACITY", "RECEIPT_CAPACITY"}:
        return HTTPException(
            status_code=507, detail="Contribution lifecycle capacity reached."
        )
    if code in {"NOT_PENDING", "NOT_ELIGIBLE", "PROMOTION_STATE", "WITHDRAWAL_STATE"}:
        return HTTPException(
            status_code=409,
            detail="Contribution is not in the required lifecycle state.",
        )
    return HTTPException(
        status_code=503, detail="Contribution lifecycle control plane unavailable."
    )


def _storage_receipt_metadata(receipt: Any) -> dict[str, Any]:
    """Return only non-secret provider control metadata for the receipt ledger."""
    return {
        "recordId": str(getattr(receipt, "record_id", "") or ""),
        "primary": getattr(receipt, "primary", None),
        "mirrors": dict(getattr(receipt, "mirrors", {}) or {}),
        "paths": dict(getattr(receipt, "paths", {}) or {}),
    }


def _contribution_review_reference(
    entry: dict[str, Any], review: Any | None = None
) -> dict[str, Any]:
    """Return non-secret review locator metadata safe for participant support.

    This intentionally excludes provider URLs, repository tokens, management
    capabilities, and raw contribution content.  The reference is useful when
    a participant no longer has a working management capability and needs a
    maintainer to locate the native PR/MR or its stable review file.
    """
    storage = entry.get("storage") if isinstance(entry.get("storage"), dict) else {}
    raw_review = (
        storage.get("review") if isinstance(storage.get("review"), dict) else {}
    )
    paths = storage.get("paths") if isinstance(storage.get("paths"), dict) else {}

    provider = str(getattr(review, "provider", "") or raw_review.get("provider") or "")[
        :32
    ]
    review_id = str(
        getattr(review, "review_id", "") or raw_review.get("reviewId") or ""
    )[:32]
    path = str(getattr(review, "path", "") or "")
    if not path:
        target_id = str(raw_review.get("targetId") or "")
        if target_id and target_id in paths:
            path = str(paths.get(target_id) or "")
        elif len(paths) == 1:
            path = str(next(iter(paths.values())) or "")
    # Storage paths are validated before entering the ledger.  Re-bound here so
    # malformed legacy metadata cannot turn a support reference into log/markup
    # injection material.
    path = path.replace("\\", "/").replace("\r", "").replace("\n", "")[:512]
    if not path or path.startswith("/") or ".." in path.split("/"):
        path = ""

    out: dict[str, Any] = {}
    if provider:
        out["reviewProvider"] = provider
    if review_id and review_id.isdigit():
        out["reviewId"] = review_id
    if path:
        out["reviewPath"] = path
    return out


def _contribution_status_payload(entry: dict[str, Any]) -> dict[str, Any]:
    state = str(entry.get("state") or "unknown")
    return {
        "status": state,
        "trainingEligible": state == "eligible",
        "trainingWithdrawn": state == "withdrawn",
        "lifecycleUncertain": state in {"promotion_uncertain", "withdrawal_uncertain"},
        "reconciliationRequired": (
            state in {"promotion_uncertain", "withdrawal_uncertain"}
        ),
        "pendingPhysicalDeleteAvailable": state == "quarantined",
        # Even a mutable-store deletion is not represented as forensic/global
        # physical erasure: database pages/WAL, filesystem snapshots, backups,
        # or provider history may exist.  What we can prove is removal of the
        # active pending payload from this lifecycle ledger.
        "contentRemovedFromActiveLedger": (
            state in {"deleted", "expired", "eligible", "withdrawn"}
            or (
                state in {"withdrawing", "withdrawal_uncertain"}
                and not bool(entry.get("records"))
            )
        ),
        "physicalErasureGuaranteed": False,
        "physicalErasureScope": "not-guaranteed",
        "reviewRevision": max(
            1, int((entry.get("operation") or {}).get("reviewRevision") or 1)
        ),
        "expiresAt": (
            int(float(entry.get("expiresAt") or 0) * 1000)
            if entry.get("expiresAt")
            else None
        ),
        "promotedAt": (
            int(float(entry.get("promotedAt") or 0) * 1000)
            if entry.get("promotedAt")
            else None
        ),
        "withdrawnAt": (
            int(float(entry.get("withdrawnAt") or 0) * 1000)
            if entry.get("withdrawnAt")
            else None
        ),
    }


async def _persist_contribution_withdrawal(
    dedup_keys: list[str],
    *,
    commit_message: str,
    server_ts_ms: int | None = None,
    path_timestamp: float | None = None,
):
    """Persist privacy-minimal withdrawal tombstones to authoritative storage.

    A caller may supply receipt-stable time values so crash/retry replays produce
    byte-identical tombstones at the same provider path instead of accumulating
    multiple logical withdrawal files.
    """
    server_ts_ms = int(
        server_ts_ms if server_ts_ms is not None else _time.time() * 1000
    )
    rows = [
        normalize_contribution_withdrawal_record(key, server_ts_ms=server_ts_ms)
        for key in dedup_keys
        if isinstance(key, str) and key
    ]
    if not rows:
        raise StorageWriteError("WITHDRAWAL_KEYS")
    content = ("\n".join(json.dumps(row, ensure_ascii=False) for row in rows)).encode(
        "utf-8"
    )
    return await _persist_storage_record(
        kind="contributions",
        content=content,
        commit_message=commit_message,
        path_timestamp=path_timestamp,
    )


def _provider_review_enabled() -> bool:
    return CONTRIBUTION_REVIEW_MODE == "provider-pr"


def _provider_review_content(entry: dict[str, Any]) -> bytes:
    """Render the exact future-main bytes for a quarantined contribution."""
    reviewed: list[dict[str, Any]] = []
    for row in entry.get("records", []):
        if not isinstance(row, dict):
            continue
        promoted = dict(row)
        promoted["trainingStatus"] = "eligible"
        reviewed.append(promoted)
    if not reviewed:
        raise StorageWriteError("EMPTY_REVIEW")
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in reviewed)).encode(
        "utf-8"
    )


async def _ensure_provider_review(entry: dict[str, Any]):
    """Resolve one native review, opening it only for legacy/unbound receipts."""
    if not _provider_review_enabled():
        return None
    if _STORAGE.primary is None:
        raise StorageWriteError("NO_PRIMARY_TARGET")
    receipt_id = str(entry.get("receiptId") or "")
    storage_hint = (
        entry.get("storage") if isinstance(entry.get("storage"), dict) else {}
    )
    if isinstance(storage_hint.get("review"), dict):
        review = await _STORAGE.get_contribution_review(
            receipt_id, review_hint=storage_hint
        )
        if review is None:
            raise StorageWriteError("REVIEW_NOT_FOUND")
        return review
    review = await _STORAGE.open_contribution_review(
        receipt_id=receipt_id,
        content=_provider_review_content(entry),
        commit_message=(
            "Automated dataset contribution review. "
            "Merge to make this record training-eligible; close/decline to reject it."
        ),
        path_timestamp=float(entry.get("receivedAt") or _time.time()),
    )
    if str(entry.get("state") or "") == "quarantined":
        try:
            bound = await _CONTRIBUTION_LEDGER.set_pending_storage(
                receipt_id, storage=review.storage_metadata()
            )
            entry["storage"] = bound.get("storage") or review.storage_metadata()
        except ContributionLedgerError as exc:
            raise StorageWriteError("REVIEW_BIND_LEDGER", transient=True) from exc
    return review


async def _sync_provider_review_merge(
    entry: dict[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    """Observe manual provider-UI merges and ratchet the local lifecycle forward."""
    if not _provider_review_enabled() or str(entry.get("state") or "") != "quarantined":
        return entry, None
    review = await _ensure_provider_review(entry)
    if review is None or review.status != "merged":
        return entry, review
    try:
        claimed = await _CONTRIBUTION_LEDGER.begin_promotion(
            str(entry.get("receiptId") or "")
        )
        claim = str(claimed.get("operationClaim") or "") or None
        promoted = await _CONTRIBUTION_LEDGER.mark_promoted(
            str(entry.get("receiptId") or ""),
            storage=review.storage_metadata(),
            claim_token=claim,
        )
        logger.info(
            json.dumps(
                {"event": "contribute.review_merged", "provider": review.provider}
            )
        )
        return promoted, review
    except ContributionLedgerError as exc:
        # Another replica/request may have observed the same merge first. Re-read
        # rather than interpreting an already-ratcheted state as a merge failure.
        current = await _CONTRIBUTION_LEDGER.get(str(entry.get("receiptId") or ""))
        if current is not None and str(current.get("state") or "") == "eligible":
            return current, review
        raise _contribution_ledger_http_error(exc) from exc


def _strict_current_contribution_records(  # ruff: ignore[too-many-branches]
    records: list[Any],
) -> None:
    """Reject schema-v4 reviewed content that normalization would truncate/drop."""
    for index, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise HTTPException(
                status_code=422, detail=f"records[{index}] must be an object."
            )
        record_type = rec.get("recordType")
        if record_type not in {"qa", "conversation"}:
            raise HTTPException(
                status_code=422,
                detail=f"records[{index}].recordType must be qa or conversation.",
            )
        note = rec.get("message", "")
        if note is not None and not isinstance(note, str):
            raise HTTPException(
                status_code=422, detail=f"records[{index}].message must be text."
            )
        if isinstance(note, str) and len(note) > MAX_CONTRIBUTION_NOTE_CHARS:
            raise HTTPException(
                status_code=422,
                detail=f"records[{index}].message exceeds {MAX_CONTRIBUTION_NOTE_CHARS} characters.",
            )
        if record_type == "qa":
            for field in ("query", "answer"):
                value = rec.get(field, "")
                if not isinstance(value, str):
                    raise HTTPException(
                        status_code=422,
                        detail=f"records[{index}].{field} must be text.",
                    )
                if len(value) > MAX_CONVERSATION_MESSAGE_CHARS:
                    raise HTTPException(
                        status_code=422,
                        detail=f"records[{index}].{field} exceeds {MAX_CONVERSATION_MESSAGE_CHARS} characters.",
                    )
            if not rec.get("query") and not rec.get("answer"):
                raise HTTPException(
                    status_code=422, detail=f"records[{index}] contains no Q&A content."
                )
            continue
        messages = rec.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=422,
                detail=f"records[{index}].messages must be a non-empty list.",
            )
        if len(messages) > MAX_CONVERSATION_MESSAGES:
            raise HTTPException(
                status_code=422,
                detail=f"records[{index}].messages exceeds {MAX_CONVERSATION_MESSAGES} messages.",
            )
        for msg_index, message in enumerate(messages):
            if not isinstance(message, dict):
                # Legacy/current normalization deliberately ignores non-object
                # rows. They are not retained, so they cannot be silently
                # truncated retained content.
                continue
            role = message.get("role")
            if role not in {"user", "assistant"}:
                # error/tool/system rows are a documented non-training family
                # and remain filtered for Run-17 compatibility.
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content:
                # Empty retained-role rows are filtered by the canonical
                # normalizer; if all rows are filtered the existing
                # "No valid contribution records" error remains authoritative.
                continue
            if len(content) > MAX_CONVERSATION_MESSAGE_CHARS:
                raise HTTPException(
                    status_code=422,
                    detail=f"records[{index}].messages[{msg_index}].content exceeds {MAX_CONVERSATION_MESSAGE_CHARS} characters.",
                )
            feedback = message.get("feedback")
            if feedback is not None:
                if not isinstance(feedback, dict):
                    raise HTTPException(
                        status_code=422,
                        detail=f"records[{index}].messages[{msg_index}].feedback must be an object.",
                    )
                fb_note = feedback.get("note", "")
                if not isinstance(fb_note, str):
                    raise HTTPException(
                        status_code=422,
                        detail=f"records[{index}].messages[{msg_index}].feedback.note must be text.",
                    )
                if len(fb_note) > MAX_CONTRIBUTION_NOTE_CHARS:
                    raise HTTPException(
                        status_code=422,
                        detail=f"records[{index}].messages[{msg_index}].feedback.note exceeds {MAX_CONTRIBUTION_NOTE_CHARS} characters.",
                    )


def _contribution_replay_response(
    entry: dict[str, Any],
    *,
    receipt_id: str,
    delete_token: str,
    ledger_manifest: dict[str, Any],
    replay: bool,
    review: Any | None = None,
    review_update: str = "",
) -> JSONResponse:
    body = {
        "accepted": True,
        "status": str(entry.get("state") or "quarantined"),
        "rows": int(entry.get("rowCount") or len(entry.get("records") or [])),
        "receiptId": receipt_id,
        "expiresAt": int(float(entry.get("expiresAt") or 0) * 1000),
        "consentVersion": RESERVED_CONSENT_VERSION,
        "receiptDurability": ledger_manifest.get("durability"),
        "idempotentReplay": bool(replay),
        "reviewMode": CONTRIBUTION_REVIEW_MODE,
        "trainingEligible": str(entry.get("state") or "") == "eligible",
        "reviewRevision": max(
            1, int((entry.get("operation") or {}).get("reviewRevision") or 1)
        ),
    }
    if review_update:
        body["reviewUpdate"] = review_update
    if review is not None:
        body["reviewProvider"] = str(getattr(review, "provider", "") or "")
        body["reviewStatus"] = str(getattr(review, "status", "") or "")
    elif _provider_review_enabled() and _STORAGE.primary is not None:
        body["reviewProvider"] = _STORAGE.primary.provider
    body.update(_contribution_review_reference(entry, review))
    # Legacy/non-envelope API clients need the generated capability once.
    # Current browser clients already hold it locally, so never echo it.
    if delete_token:
        body["deleteToken"] = delete_token
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


@app.post("/v1/contribute")
async def contribute(  # ruff: ignore[too-many-branches]
    request: Request,
) -> JSONResponse:
    """Accept explicit content into quarantine with create-once recovery.

    Schema-v4 intake validates the exact reviewed content before normalization;
    it rejects, rather than silently truncates, records/messages that exceed the
    canonical limits. Current browser clients also send a pre-request operation
    envelope so a lost response can be retried without duplicating or orphaning
    a receipt.
    """
    ledger_manifest = _CONTRIBUTION_LEDGER.manifest()
    if _CONTRIBUTION_LEDGER_CONFIG_ERROR:
        raise HTTPException(
            status_code=503, detail="Contribution lifecycle storage is misconfigured."
        )
    if not _CONTRIBUTION_LEDGER_READY:
        raise HTTPException(
            status_code=503, detail="Contribution lifecycle storage is unavailable."
        )
    if CONTRIBUTION_REQUIRE_DURABLE and not bool(ledger_manifest.get("durable")):
        raise HTTPException(
            status_code=503,
            detail="Durable contribution lifecycle storage is required.",
        )
    if CONTRIBUTION_REQUIRE_SHARED and not (
        bool(ledger_manifest.get("shared"))
        and bool(ledger_manifest.get("authoritative"))
    ):
        raise HTTPException(
            status_code=503,
            detail="Shared contribution lifecycle authority is required.",
        )

    raw = await _read_limited_body(request, CONTRIBUTION_MAX_BODY_BYTES, "Contribution")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422, detail="Contribution body must be a JSON object."
        )
    if payload.get("consentFlag") is not True:
        raise HTTPException(
            status_code=422, detail="Explicit contribution consent is required."
        )
    schema_version = payload.get("schemaVersion")
    if schema_version not in {2, 3, 4}:
        raise HTTPException(
            status_code=422, detail="Unsupported contribution schemaVersion."
        )
    consent_version = payload.get("consentVersion")
    consent_ok = (
        consent_version == RESERVED_CONSENT_VERSION
        if schema_version == 4  # ruff: ignore[magic-value-comparison]
        else consent_version in (LEGACY_CONSENT_VERSIONS | {RESERVED_CONSENT_VERSION})
    )
    if not consent_ok:
        raise HTTPException(
            status_code=422,
            detail="Consent text changed or is missing. Reload the page and review consent again.",
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=422, detail="records must be a non-empty list.")
    if len(records) > MAX_CONTRIBUTION_RECORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Too many records. Maximum {MAX_CONTRIBUTION_RECORDS} per request.",
        )
    if schema_version == 4:  # ruff: ignore[magic-value-comparison]
        _strict_current_contribution_records(records)

    envelope = _operation_envelope(
        request,
        payload,
        "contribution",
        max_age_ms=min(
            CONTRIBUTION_QUARANTINE_TTL_SECONDS,
            CONTRIBUTION_LEDGER_TERMINAL_RETENTION_SECONDS,
        )
        * 1000,
    )
    if envelope:
        receipt_id, delete_hash, operation_id = envelope
        delete_token = ""  # raw capability never crosses the create request boundary
    else:
        receipt_id, delete_token, operation_id = (
            uuid.uuid4().hex,
            generate_edit_token(),
            "",
        )
        delete_hash = hash_edit_token(delete_token)
    payload_digest = _canonical_payload_digest(payload)

    async def _replay_if_same() -> JSONResponse | None:
        if not envelope:
            return None
        try:
            existing = await _CONTRIBUTION_LEDGER.get(receipt_id)
        except ContributionLedgerError as exc:
            raise _contribution_ledger_http_error(exc) from exc
        if existing is None:
            return None
        operation = (
            existing.get("operation")
            if isinstance(existing.get("operation"), dict)
            else {}
        )
        if (
            not secrets.compare_digest(
                str(operation.get("payloadDigest") or ""), payload_digest
            )
            or not secrets.compare_digest(
                str(operation.get("operationId") or ""), operation_id
            )
            or not secrets.compare_digest(
                str(existing.get("deleteTokenHash") or ""), delete_hash
            )
        ):
            raise HTTPException(
                status_code=409,
                detail="Operation identity was already used with different contribution content.",
            )
        review = None
        if (
            _provider_review_enabled()
            and str(existing.get("state") or "") == "quarantined"
        ):
            try:
                review = await _ensure_provider_review(existing)
            except StorageWriteError as exc:
                logger.error(
                    json.dumps(
                        {"event": "contribute.review_recover_fail", "code": exc.code}
                    )
                )
                raise HTTPException(
                    status_code=503,
                    detail="Contribution review could not be recovered.",
                ) from exc
        return _contribution_replay_response(
            existing,
            receipt_id=receipt_id,
            delete_token=delete_token,
            ledger_manifest=ledger_manifest,
            replay=True,
            review=review,
        )

    replay = await _replay_if_same()
    if replay is not None:
        logger.info(json.dumps({"event": "contribute.create_replay"}))
        return replay

    client_ip = _client_ip(request)
    allowed, _count = await _consume_rate_limit(
        _contrib_rl,
        _contrib_rl_lock,
        client_ip,
        limit=CONTRIBUTION_RATE_LIMIT_PER_HOUR,
        scope="contribution",
    )
    if not allowed:
        logger.warning(
            json.dumps({"event": "contribute.ratelimit", "ip": _mask_ip(client_ip)})
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded.",
            headers={"Retry-After": "3600"},
        )

    server_ts_ms = int(_time.time() * 1000)
    normalized = [
        normalize_contribution_record(
            rec,
            envelope=payload,
            server_ts_ms=server_ts_ms,
            training_status="quarantined",
            submission_id=receipt_id,
        )
        for rec in records
    ]
    normalized = [
        row
        for row in normalized
        if row.get("recordType") != "conversation" or bool(row.get("messages"))
    ]
    if not normalized:
        raise HTTPException(
            status_code=422, detail="No valid contribution records were supplied."
        )
    encoded = ("\n".join(json.dumps(r, ensure_ascii=False) for r in normalized)).encode(
        "utf-8"
    )
    expires_at = _time.time() + CONTRIBUTION_QUARANTINE_TTL_SECONDS
    entry = {
        "receiptId": receipt_id,
        "state": "quarantined",
        "records": normalized,
        "bytes": len(encoded),
        "deleteTokenHash": delete_hash,
        "expiresAt": expires_at,
        "receivedAt": _time.time(),
        "dedupKeys": [
            str(row.get("_dedup_key") or "")
            for row in normalized
            if row.get("_dedup_key")
        ],
        "storage": {},
        "withdrawalStorage": {},
        "currentViewRemoval": {},
        "lastError": "",
        "operation": {
            "payloadDigest": payload_digest,
            "operationId": operation_id if envelope else "",
            "reviewRevision": 1,
        },
        "rowCount": len(normalized),
    }
    try:
        await _CONTRIBUTION_LEDGER.create(entry)
    except ContributionLedgerError as exc:
        if envelope and exc.code == "DUPLICATE_RECEIPT":
            replay = await _replay_if_same()
            if replay is not None:
                return replay
        raise _contribution_ledger_http_error(exc) from exc

    review = None
    if _provider_review_enabled():
        try:
            review = await _ensure_provider_review(entry)
        except StorageWriteError as exc:
            # Leave the receipt in quarantine on ambiguous provider outcomes so
            # the create-once replay can recover the deterministic native review.
            # Definite failures are also retained for bounded TTL rather than
            # pretending the submission never existed after consented intake.
            logger.error(
                json.dumps({"event": "contribute.review_open_fail", "code": exc.code})
            )
            raise HTTPException(
                status_code=503,
                detail="Contribution was quarantined but its provider review could not be opened. Retry the same submission.",
            ) from exc

    logger.info(
        json.dumps(
            {
                "event": "contribute.quarantined",
                "rows": len(normalized),
                "idempotent": bool(envelope),
                "review_mode": CONTRIBUTION_REVIEW_MODE,
                "review_provider": (
                    getattr(review, "provider", None) if review is not None else None
                ),
            }
        )
    )
    return _contribution_replay_response(
        entry,
        receipt_id=receipt_id,
        delete_token=delete_token,
        ledger_manifest=ledger_manifest,
        replay=False,
        review=review,
    )


def _validate_contribution_update_payload(payload: Any) -> list[Any]:
    """Validate the same public contribution contract used by intake."""
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422, detail="Contribution body must be a JSON object."
        )
    if payload.get("consentFlag") is not True:
        raise HTTPException(
            status_code=422, detail="Explicit contribution consent is required."
        )
    schema_version = payload.get("schemaVersion")
    if schema_version not in {2, 3, 4}:
        raise HTTPException(
            status_code=422, detail="Unsupported contribution schemaVersion."
        )
    consent_version = payload.get("consentVersion")
    consent_ok = (
        consent_version == RESERVED_CONSENT_VERSION
        if schema_version == 4  # ruff: ignore[magic-value-comparison]
        else consent_version in (LEGACY_CONSENT_VERSIONS | {RESERVED_CONSENT_VERSION})
    )
    if not consent_ok:
        raise HTTPException(
            status_code=422,
            detail="Consent text changed or is missing. Reload the page and review consent again.",
        )
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise HTTPException(status_code=422, detail="records must be a non-empty list.")
    if len(records) > MAX_CONTRIBUTION_RECORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Too many records. Maximum {MAX_CONTRIBUTION_RECORDS} per request.",
        )
    if schema_version == 4:  # ruff: ignore[magic-value-comparison]
        _strict_current_contribution_records(records)
    return records


@app.put("/v1/contribute/{receipt_id}")
async def update_pending_contribution(
    receipt_id: str, request: Request
) -> JSONResponse:
    """Replace a pending review instead of opening a duplicate PR/MR."""
    ledger_manifest = _CONTRIBUTION_LEDGER.manifest()
    entry = await _authorized_contribution_entry(receipt_id, request)
    current_review = None
    if str(entry.get("state") or "") == "quarantined" and _provider_review_enabled():
        try:
            entry, current_review = await _sync_provider_review_merge(entry)
        except StorageWriteError as exc:
            logger.warning(
                json.dumps(
                    {"event": "contribute.review_update_status_fail", "code": exc.code}
                )
            )
            raise HTTPException(
                status_code=503,
                detail="Contribution review status could not be verified.",
            ) from exc
    if str(entry.get("state") or "") != "quarantined":
        raise HTTPException(
            status_code=409, detail="Only a pending contribution review can be updated."
        )

    raw = await _read_limited_body(request, CONTRIBUTION_MAX_BODY_BYTES, "Contribution")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
    records = _validate_contribution_update_payload(payload)
    payload_digest = _canonical_payload_digest(payload)
    operation = (
        entry.get("operation") if isinstance(entry.get("operation"), dict) else {}
    )
    current_digest = str(operation.get("payloadDigest") or "")

    if current_digest and secrets.compare_digest(current_digest, payload_digest):
        review = current_review
        if review is None and _provider_review_enabled():
            try:
                review = await _ensure_provider_review(entry)
            except StorageWriteError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Contribution review could not be recovered.",
                ) from exc
        logger.info(json.dumps({"event": "contribute.review_duplicate_suppressed"}))
        return _contribution_replay_response(
            entry,
            receipt_id=receipt_id,
            delete_token="",
            ledger_manifest=ledger_manifest,
            replay=True,
            review=review,
            review_update="unchanged",
        )

    server_ts_ms = int(_time.time() * 1000)
    normalized = [
        normalize_contribution_record(
            rec,
            envelope=payload,
            server_ts_ms=server_ts_ms,
            training_status="quarantined",
            submission_id=receipt_id,
        )
        for rec in records
    ]
    normalized = [
        row
        for row in normalized
        if row.get("recordType") != "conversation" or bool(row.get("messages"))
    ]
    if not normalized:
        raise HTTPException(
            status_code=422, detail="No valid contribution records were supplied."
        )
    encoded = ("\n".join(json.dumps(r, ensure_ascii=False) for r in normalized)).encode(
        "utf-8"
    )
    next_revision = max(1, int(operation.get("reviewRevision") or 1)) + 1
    review = current_review
    if _provider_review_enabled():
        eligible_bytes = _provider_review_content({"records": normalized})
        try:
            review = await _STORAGE.update_contribution_review(
                receipt_id=receipt_id,
                content=eligible_bytes,
                commit_message=f"Update dataset contribution review (revision {next_revision})",
                path_timestamp=float(entry.get("receivedAt") or _time.time()),
                review_hint=(
                    entry.get("storage")
                    if isinstance(entry.get("storage"), dict)
                    else None
                ),
            )
        except StorageWriteError as exc:
            if exc.code in {"REVIEW_CLOSED", "REVIEW_MERGED", "REVIEW_NOT_FOUND"}:
                raise HTTPException(
                    status_code=409,
                    detail="The existing repository review is no longer open for updates.",
                ) from exc
            logger.error(
                json.dumps({"event": "contribute.review_update_fail", "code": exc.code})
            )
            raise HTTPException(
                status_code=503,
                detail="The existing repository review could not be updated safely.",
            ) from exc

    try:
        updated = await _CONTRIBUTION_LEDGER.replace_pending_payload(
            receipt_id,
            records=normalized,
            byte_count=len(encoded),
            dedup_keys=[
                str(row.get("_dedup_key") or "")
                for row in normalized
                if row.get("_dedup_key")
            ],
            payload_digest=payload_digest,
            row_count=len(normalized),
            storage=(
                review.storage_metadata()
                if review is not None
                else (
                    entry.get("storage")
                    if isinstance(entry.get("storage"), dict)
                    else {}
                )
            ),
        )
    except ContributionLedgerError as exc:
        logger.error(
            json.dumps(
                {"event": "contribute.review_update_ledger_fail", "code": exc.code}
            )
        )
        raise _contribution_ledger_http_error(exc) from exc

    logger.info(
        json.dumps(
            {
                "event": "contribute.review_updated",
                "rows": len(normalized),
                "revision": int(
                    (updated.get("operation") or {}).get("reviewRevision")
                    or next_revision
                ),
                "review_provider": (
                    getattr(review, "provider", None) if review else None
                ),
            }
        )
    )
    return _contribution_replay_response(
        updated,
        receipt_id=receipt_id,
        delete_token="",
        ledger_manifest=ledger_manifest,
        replay=False,
        review=review,
        review_update="updated",
    )


async def _authorized_contribution_entry(
    receipt_id: str, request: Request
) -> dict[str, Any]:
    try:
        entry = await _CONTRIBUTION_LEDGER.get(receipt_id)
    except ContributionLedgerError as exc:
        raise _contribution_ledger_http_error(exc) from exc
    if entry is None:
        raise HTTPException(status_code=404, detail="Contribution receipt not found.")
    supplied = request.headers.get("X-Contribution-Delete-Token", "")
    if not supplied or not verify_edit_token(
        supplied, str(entry.get("deleteTokenHash") or "")
    ):
        raise HTTPException(
            status_code=403, detail="Invalid contribution management capability."
        )
    return entry


@app.get("/v1/contribute/{receipt_id}")
async def contribution_status(receipt_id: str, request: Request) -> JSONResponse:
    """Return capability-protected lifecycle status without returning content."""
    entry = await _authorized_contribution_entry(receipt_id, request)
    review = None
    if _provider_review_enabled() and str(entry.get("state") or "") == "quarantined":
        try:
            entry, review = await _sync_provider_review_merge(entry)
        except StorageWriteError as exc:
            logger.warning(
                json.dumps({"event": "contribute.review_status_fail", "code": exc.code})
            )
    body = _contribution_status_payload(entry)
    body["reviewMode"] = CONTRIBUTION_REVIEW_MODE
    if review is not None:
        body["reviewProvider"] = review.provider
        body["reviewStatus"] = review.status
    body.update(_contribution_review_reference(entry, review))
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


@app.delete("/v1/contribute/{receipt_id}")
async def delete_or_withdraw_contribution(  # ruff: ignore[too-many-branches]
    receipt_id: str, request: Request
) -> JSONResponse:
    """Delete pending data or withdraw a previously promoted contribution.

    Pending deletion removes content from the active mutable ledger before it
    leaves for training storage.  It is not represented as forensic filesystem
    erasure. After promotion the same capability means *withdraw
    from training*: a privacy-minimal tombstone is durably written and the
    current provider branch view is removed best-effort.  Versioned provider/Git
    history is explicitly not represented as physically erased.
    """
    entry = await _authorized_contribution_entry(receipt_id, request)
    state = str(entry.get("state") or "")

    if state == "quarantined" and _provider_review_enabled():
        try:
            entry, _review = await _sync_provider_review_merge(entry)
            state = str(entry.get("state") or "")
            if state == "quarantined":
                close_result = await _STORAGE.close_contribution_review(
                    receipt_id,
                    review_hint=(
                        entry.get("storage")
                        if isinstance(entry.get("storage"), dict)
                        else None
                    ),
                )
                logger.info(
                    json.dumps(
                        {"event": "contribute.review_closed", "result": close_result}
                    )
                )
        except StorageWriteError as exc:
            logger.error(
                json.dumps({"event": "contribute.review_close_fail", "code": exc.code})
            )
            raise HTTPException(
                status_code=503,
                detail="Contribution review could not be closed safely.",
            ) from exc

    if state == "quarantined":
        try:
            deleted = await _CONTRIBUTION_LEDGER.delete_pending(receipt_id)
        except ContributionLedgerError as exc:
            raise _contribution_ledger_http_error(exc) from exc
        logger.info(json.dumps({"event": "contribute.pending_deleted"}))
        return JSONResponse(
            {
                "deleted": True,
                **_contribution_status_payload(deleted),
            }
        )
    if state == "deleted":
        return JSONResponse({"deleted": True, **_contribution_status_payload(entry)})
    if state == "expired":
        raise HTTPException(status_code=410, detail="Pending contribution expired.")
    if state in {"promoting", "withdrawing"}:
        raise HTTPException(
            status_code=409,
            detail="Contribution lifecycle operation already in progress.",
        )
    if state == "withdrawn":
        return JSONResponse(
            {
                "withdrawn": True,
                **_contribution_status_payload(entry),
                "currentViewRemoval": entry.get("currentViewRemoval") or {},
            }
        )
    if state not in {"eligible", "promotion_uncertain", "withdrawal_uncertain"}:
        raise HTTPException(
            status_code=409,
            detail="Contribution cannot be managed in its current state.",
        )

    try:
        claimed = await _CONTRIBUTION_LEDGER.begin_withdrawal(receipt_id)
        operation_claim = str(claimed.get("operationClaim") or "") or None
    except ContributionLedgerError as exc:
        raise _contribution_ledger_http_error(exc) from exc

    dedup_keys = [
        str(v) for v in claimed.get("dedupKeys", []) if isinstance(v, str) and v
    ]
    try:
        promoted_at = float(
            claimed.get("promotedAt") or claimed.get("receivedAt") or _time.time()
        )
        withdrawal_receipt = await _persist_contribution_withdrawal(
            dedup_keys,
            commit_message=f"Withdraw reviewed contribution receipt ({len(dedup_keys)} record(s))",
            server_ts_ms=int(promoted_at * 1000) + 1,
            path_timestamp=promoted_at,
        )
    except StorageWriteError as exc:
        await _CONTRIBUTION_LEDGER.withdrawal_failed(
            receipt_id, exc.code, claim_token=operation_claim
        )
        logger.error(
            json.dumps({"event": "contribute.withdraw_fail", "code": exc.code})
        )
        raise HTTPException(
            status_code=503, detail="Failed to persist training withdrawal."
        ) from exc

    original_storage = (
        claimed.get("storage") if isinstance(claimed.get("storage"), dict) else {}
    )
    removal = await _STORAGE.remove_current_view(
        dict(original_storage.get("paths") or {}),
        record_id=str(original_storage.get("recordId") or "") or None,
    )
    withdrawal_meta = _storage_receipt_metadata(withdrawal_receipt)
    try:
        withdrawn = await _CONTRIBUTION_LEDGER.mark_withdrawn(
            receipt_id,
            withdrawal_storage=withdrawal_meta,
            current_view_removal=removal,
            claim_token=operation_claim,
        )
    except ContributionLedgerError as exc:
        # The authoritative training withdrawal already exists. Do not retry the
        # eligible write or falsely report erasure; surface control-plane failure.
        logger.error(
            json.dumps({"event": "contribute.withdraw_ledger_fail", "code": exc.code})
        )
        raise HTTPException(
            status_code=503,
            detail="Training withdrawal persisted but lifecycle confirmation failed.",
        ) from exc

    logger.info(json.dumps({"event": "contribute.withdrawn", "rows": len(dedup_keys)}))
    return JSONResponse(
        {
            "withdrawn": True,
            **_contribution_status_payload(withdrawn),
            "currentViewRemoval": removal,
            "trainingWithdrawalRecordId": withdrawal_meta.get("recordId"),
        },
        status_code=202,
    )


@app.post("/v1/contribute/{receipt_id}/promote")
async def promote_contribution(  # ruff: ignore[too-many-branches]
    receipt_id: str,
    request: Request,
) -> JSONResponse:
    """Atomically claim review, then promote only once to training-eligible storage."""
    auth = request.headers.get("Authorization", "")
    token = auth[7:] if auth.lower().startswith("bearer ") else ""
    if not CONTRIBUTION_REVIEW_TOKEN or not secrets.compare_digest(
        token, CONTRIBUTION_REVIEW_TOKEN
    ):
        raise HTTPException(
            status_code=401, detail="Contribution review authorization required."
        )
    if _STORAGE.primary is None:
        raise HTTPException(
            status_code=503, detail="Training storage is not configured."
        )
    try:
        entry = await _CONTRIBUTION_LEDGER.begin_promotion(receipt_id)
        operation_claim = str(entry.get("operationClaim") or "") or None
    except ContributionLedgerError as exc:
        raise _contribution_ledger_http_error(exc) from exc

    reviewed = []
    for row in entry.get("records", []):
        if not isinstance(row, dict):
            continue
        promoted = dict(row)
        promoted["trainingStatus"] = "eligible"
        reviewed.append(promoted)
    if not reviewed:
        await _CONTRIBUTION_LEDGER.promotion_failed(
            receipt_id, "EMPTY_REVIEW", claim_token=operation_claim
        )
        raise HTTPException(
            status_code=409, detail="Contribution contains no reviewable records."
        )

    content = ("\n".join(json.dumps(r, ensure_ascii=False) for r in reviewed)).encode(
        "utf-8"
    )
    try:
        if _provider_review_enabled():
            # The native provider review already contains these exact eligible
            # bytes. Merge the PR/MR instead of creating a second direct commit.
            prepared_review = await _ensure_provider_review(entry)
            merged_review = await _STORAGE.merge_contribution_review(
                receipt_id,
                review_hint=(
                    entry.get("storage")
                    if isinstance(entry.get("storage"), dict)
                    else None
                ),
            )
            if prepared_review is None:
                raise StorageWriteError("REVIEW_NOT_FOUND")
            receipt = replace(prepared_review, status=merged_review.status)
        else:
            receipt = await _persist_storage_record(
                kind="contributions",
                content=content,
                commit_message=f"Promote {len(reviewed)} reviewed contribution record(s)",
                path_timestamp=float(entry.get("receivedAt") or _time.time()),
            )
    except StorageWriteError as exc:
        if bool(getattr(exc, "transient", False)):
            try:
                await _CONTRIBUTION_LEDGER.mark_promotion_uncertain(
                    receipt_id, exc.code, claim_token=operation_claim
                )
            except ContributionLedgerError as ledger_exc:
                logger.error(
                    json.dumps(
                        {
                            "event": "contribute.promote_uncertain_ledger_fail",
                            "code": ledger_exc.code,
                        }
                    )
                )
            logger.error(
                json.dumps(
                    {"event": "contribute.promote_outcome_uncertain", "code": exc.code}
                )
            )
            raise HTTPException(
                status_code=503,
                detail="Training storage outcome is uncertain; re-promotion is blocked. Use the contribution management capability to withdraw it or reconcile operationally.",
            ) from exc
        await _CONTRIBUTION_LEDGER.promotion_failed(
            receipt_id, exc.code, claim_token=operation_claim
        )
        logger.error(json.dumps({"event": "contribute.promote_fail", "code": exc.code}))
        raise HTTPException(
            status_code=503, detail="Failed to store reviewed contribution."
        ) from exc

    storage_meta = (
        receipt.storage_metadata()
        if _provider_review_enabled()
        else _storage_receipt_metadata(receipt)
    )
    try:
        await _CONTRIBUTION_LEDGER.mark_promoted(
            receipt_id, storage=storage_meta, claim_token=operation_claim
        )
    except ContributionLedgerError as exc:
        # Durable eligible bytes now exist but the mutable ledger could not
        # confirm them. Fail safe by writing withdrawal tombstones immediately;
        # never reopen the receipt for a second promotion race.
        dedup_keys = [
            str(v) for v in entry.get("dedupKeys", []) if isinstance(v, str) and v
        ]
        compensation = "not-attempted"
        try:
            compensation_ts = (
                max(
                    [
                        int(r.get("_ts") or 0)
                        for r in entry.get("records", [])
                        if isinstance(r, dict)
                    ]
                    or [0]
                )
                + 1
            )
            received_at = float(entry.get("receivedAt") or _time.time())
            await _persist_contribution_withdrawal(
                dedup_keys,
                commit_message="Compensate unconfirmed contribution promotion",
                server_ts_ms=compensation_ts,
                path_timestamp=received_at,
            )
            compensation = "withdrawal-persisted"
            await _STORAGE.remove_current_view(
                dict(storage_meta.get("paths") or {}),
                record_id=str(storage_meta.get("recordId") or "") or None,
            )
        except StorageWriteError:
            compensation = "withdrawal-failed"
        logger.error(
            json.dumps(
                {
                    "event": "contribute.promote_ledger_fail",
                    "code": exc.code,
                    "compensation": compensation,
                }
            )
        )
        raise HTTPException(
            status_code=503, detail="Promotion could not be safely finalized."
        ) from exc

    logger.info(json.dumps({"event": "contribute.promoted", "rows": len(reviewed)}))
    return JSONResponse(
        {
            "promoted": True,
            "status": "eligible",
            "rows": len(reviewed),
            "recordId": receipt.record_id,
            "primary": (
                receipt.review_url if _provider_review_enabled() else receipt.primary
            ),
            "mirrors": {} if _provider_review_enabled() else receipt.mirrors,
            "reviewMode": CONTRIBUTION_REVIEW_MODE,
        }
    )


@app.post("/v1/share")
async def share(  # ruff: ignore[too-many-branches]
    request: Request,
) -> JSONResponse:
    """Create a Global Share with optional create-once recovery semantics.

    Current browser clients create the public locator and revoke capability
    before the request, then send only ``operationId``, the locator, and
    ``SHA-256(revoke_capability)``. If the response is lost, retrying the *same*
    reviewed payload resolves to the same object instead of creating an orphan.
    The raw management capability never crosses the create-request boundary.
    """
    store_status = _require_share_store_ready()
    raw = await _read_limited_body(request, SHARE_MAX_BODY_BYTES, "Share")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Share body must be an object.")

    if SHARE_WRITE_TOKEN:
        auth = request.headers.get("authorization", "")
        provided = auth[7:] if auth.lower().startswith("bearer ") else ""
        if not provided or not secrets.compare_digest(provided, SHARE_WRITE_TOKEN):
            logger.warning(json.dumps({"event": "share.auth_fail"}))
            raise HTTPException(
                status_code=401, detail="Not authorized to create shares."
            )

    try:
        snapshot = canonicalize_share_snapshot(payload.get("snapshot"))
        fmt = validate_share_format(payload.get("format"))
    except ShareValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    ttl_days = max(1, min(_safe_int(payload.get("ttlDays"), 30), 365))
    canonical_bytes = len(
        json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if canonical_bytes > SHARE_MAX_BODY_BYTES:
        raise HTTPException(
            status_code=413, detail="Canonical share payload too large."
        )

    envelope = _operation_envelope(
        request, payload, "share", max_age_ms=ttl_days * 86400 * 1000
    )
    if envelope:
        share_id, edit_hash, operation_id = envelope
        edit_token = ""  # raw revoke capability remains browser-only on create
    else:
        share_id, edit_token, operation_id = uuid.uuid4().hex, generate_edit_token(), ""
        edit_hash = hash_edit_token(edit_token)
    payload_digest = _canonical_payload_digest(
        {"snapshot": snapshot, "format": fmt, "ttlDays": ttl_days}
    )
    base_url = _share_public_base(request)

    async def _replay_if_same() -> JSONResponse | None:
        if not envelope:
            return None
        try:
            existing = await _SHARE_STORE.get(share_id)
        except ShareStoreError as exc:
            if exc.code == "EXPIRED":
                raise _share_store_http_error(exc) from exc
            raise _share_store_http_error(exc) from exc
        if existing is None:
            return None
        if (
            not secrets.compare_digest(
                str(existing.get("operation_payload_digest") or ""), payload_digest
            )
            or not secrets.compare_digest(
                str(existing.get("operation_id") or ""), operation_id
            )
            or not secrets.compare_digest(
                str(existing.get("edit_hash") or ""), edit_hash
            )
        ):
            raise HTTPException(
                status_code=409,
                detail="Operation identity was already used with different Share content.",
            )
        body = {
            "uuid": share_id,
            "url": f"{base_url}/v1/share#share={share_id}",
            "expiresAt": existing.get("expiresAt"),
            "idempotentReplay": True,
            "storage": store_status,
        }
        if edit_token:
            body["editToken"] = edit_token
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    replay = await _replay_if_same()
    if replay is not None:
        logger.info(json.dumps({"event": "share.create_replay", "format": fmt}))
        return replay

    client_ip = _client_ip(request)
    allowed, _count = await _consume_rate_limit(
        _share_rl,
        _share_rl_lock,
        client_ip,
        limit=SHARE_RATE_LIMIT_PER_HOUR,
        scope="share",
    )
    if not allowed:
        logger.warning(
            json.dumps({"event": "share.ratelimit", "ip": _mask_ip(client_ip)})
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for Global Share writes.",
            headers={"Retry-After": "3600"},
        )

    now_ts = _time.time()
    expires_ts = now_ts + ttl_days * 86400
    expires_iso = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(expires_ts))
    entry = {
        "snapshot": snapshot,
        "format": fmt,
        "edit_hash": edit_hash,
        "bytes": canonical_bytes,
        "expiresAt_ts": expires_ts,
        "expiresAt": expires_iso,
        "transport_version": SHARE_TRANSPORT_VERSION,
        "operation_payload_digest": payload_digest if envelope else "",
        "operation_id": operation_id if envelope else "",
    }
    _sync_share_store_runtime_limits()
    try:
        await _SHARE_STORE.create(share_id, entry)
    except ShareStoreError as exc:
        # Concurrent safe retries race on the same deterministic locator. Re-read
        # and return the existing object when the reviewed payload is identical.
        if envelope and exc.code == "DUPLICATE_SHARE":
            replay = await _replay_if_same()
            if replay is not None:
                return replay
        raise _share_store_http_error(exc) from exc

    logger.info(
        json.dumps(
            {
                "event": "share.create",
                "ip": _mask_ip(client_ip),
                "bytes": canonical_bytes,
                "format": fmt,
                "ttl_days": ttl_days,
                "idempotent": bool(envelope),
            }
        )
    )
    body = {
        "uuid": share_id,
        "url": f"{base_url}/v1/share#share={share_id}",
        "expiresAt": expires_iso,
        "idempotentReplay": False,
        "storage": store_status,
    }
    if edit_token:
        body["editToken"] = edit_token
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


_SHARE_LOCATOR_BODY_BYTES = 4096


def _parse_share_locator_payload(raw: bytes) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Share body must be an object.")
    share_id = payload.get("shareId")
    if not isinstance(share_id, str) or not valid_share_id(share_id):
        raise HTTPException(status_code=404, detail="Share not found or expired.")
    return share_id


async def _share_lookup_live(share_id: str) -> dict[str, Any]:
    _require_share_store_ready()
    try:
        entry = await _SHARE_STORE.get(share_id)
    except ShareStoreError as exc:
        raise _share_store_http_error(exc) from exc
    if entry is None:
        raise HTTPException(status_code=404, detail="Share not found or expired.")
    return entry


def _legacy_share_entry_allowed(entry: dict[str, Any]) -> bool:
    """Allow capability-bearing paths only for genuinely pre-generation entries.

    Missing generation metadata is the compatibility marker for objects created
    before Run 14. Unknown/tampered values fail closed. New or fixed-path-updated
    entries are generation 2 and cannot be read, probed, updated or revoked via
    ``/v1/share/{id}``.
    """
    version = entry.get("transport_version")
    return version is None or version == 1


def _require_legacy_share_entry(entry: dict[str, Any]) -> None:
    if not _legacy_share_entry_allowed(entry):
        raise HTTPException(status_code=404, detail="Share not found or expired.")


def _legacy_share_headers(entry: dict[str, Any]) -> dict[str, str]:
    expires_ts = float(entry.get("expiresAt_ts") or 0)
    sunset = (
        _time.strftime("%a, %d %b %Y %H:%M:%S GMT", _time.gmtime(expires_ts))
        if expires_ts > 0
        else ""
    )
    headers = {
        # RFC 9745 Structured Field Date: Run 14 deprecation effective
        # 2026-08-29T00:00:00Z.
        "Deprecation": "@1787961600",
        "Link": '</v1/share>; rel="successor-version"',
    }
    if sunset:
        headers["Sunset"] = sunset
    return headers


@app.get("/v1/share")
async def share_viewer() -> Response:
    """Serve a fixed-path viewer whose public capability stays in URL fragment."""
    return Response(
        content=render_share_viewer_shell("/v1/share/read"),
        status_code=200,
        media_type="text/html",
        headers={
            "Cache-Control": "private, no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Permissions-Policy": (
                "camera=(), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=()"
            ),
            "X-Robots-Tag": "noindex, nofollow, noarchive",
            "Content-Security-Policy": (
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            ),
        },
    )


@app.post("/v1/share/read")
async def share_read_fixed(request: Request) -> JSONResponse:
    """Resolve a public Share from a fixed path with its locator in the body."""
    raw = await _read_limited_body(request, _SHARE_LOCATOR_BODY_BYTES, "Share locator")
    share_id = _parse_share_locator_payload(raw)
    try:
        entry = await _share_lookup_live(share_id)
    except HTTPException as exc:
        if exc.status_code == 404:  # ruff: ignore[magic-value-comparison]
            logger.info(json.dumps({"event": "share.miss"}))
        raise
    try:
        content, _mime_type, _ext = render_share(entry["snapshot"], entry["format"])
    except ShareValidationError as exc:
        logger.error(json.dumps({"event": "share.corrupt_entry"}))
        raise HTTPException(status_code=500, detail="Stored share is invalid.") from exc
    logger.info(json.dumps({"event": "share.read", "format": entry["format"]}))
    payload: dict[str, Any] = {
        "format": entry["format"],
        "expiresAt": entry["expiresAt"],
    }
    if entry["format"] == "html":
        payload["snapshot"] = entry["snapshot"]
    else:
        payload["content"] = content
    return JSONResponse(
        payload,
        headers={
            "Cache-Control": "private, no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Robots-Tag": "noindex, nofollow, noarchive",
        },
    )


@app.post("/v1/share/status")
async def share_status_fixed(request: Request) -> Response:
    """Probe lifecycle without putting the read capability in the request path."""
    raw = await _read_limited_body(request, _SHARE_LOCATOR_BODY_BYTES, "Share locator")
    share_id = _parse_share_locator_payload(raw)
    entry = await _share_lookup_live(share_id)
    logger.info(json.dumps({"event": "share.status", "format": entry["format"]}))
    return Response(
        status_code=200,
        headers={
            "Cache-Control": "private, no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Robots-Tag": "noindex, nofollow, noarchive",
        },
    )


@app.post("/v1/share/update")
async def share_update_fixed(request: Request) -> JSONResponse:
    """Replace a Share using a fixed request path and private edit capability."""
    raw = await _read_limited_body(
        request, SHARE_MAX_BODY_BYTES + _SHARE_LOCATOR_BODY_BYTES, "Share"
    )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Share body must be an object.")
    share_id = payload.get("shareId")
    if not isinstance(share_id, str) or not valid_share_id(share_id):
        raise HTTPException(status_code=404, detail="Share not found or expired.")
    entry = await _share_lookup_live(share_id)
    edit_token = request.headers.get("x-share-edit-token", "")
    if not verify_edit_token(edit_token, entry.get("edit_hash", "")):
        logger.warning(json.dumps({"event": "share.edit_auth_fail"}))
        raise HTTPException(status_code=403, detail="Invalid share edit capability.")
    try:
        snapshot = canonicalize_share_snapshot(payload.get("snapshot"))
        fmt = validate_share_format(payload.get("format"))
    except ShareValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    ttl_days = max(1, min(_safe_int(payload.get("ttlDays"), 30), 365))
    canonical_bytes = len(
        json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    if canonical_bytes > SHARE_MAX_BODY_BYTES:
        raise HTTPException(
            status_code=413, detail="Canonical share payload too large."
        )
    client_ip = _client_ip(request)
    allowed, _count = await _consume_rate_limit(
        _share_rl,
        _share_rl_lock,
        client_ip,
        limit=SHARE_RATE_LIMIT_PER_HOUR,
        scope="share",
    )
    if not allowed:
        logger.warning(
            json.dumps({"event": "share.ratelimit", "ip": _mask_ip(client_ip)})
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for Global Share writes.",
            headers={"Retry-After": "3600"},
        )
    now_ts = _time.time()
    expires_ts = now_ts + ttl_days * 86400
    expires_iso = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(expires_ts))
    next_entry = {
        "snapshot": snapshot,
        "format": fmt,
        "edit_hash": entry["edit_hash"],
        "bytes": canonical_bytes,
        "expiresAt_ts": expires_ts,
        "expiresAt": expires_iso,
        "transport_version": SHARE_TRANSPORT_VERSION,
        # Updating the object consumes the create-operation digest; a later
        # replay of the original create must not silently overwrite edits.
        "operation_payload_digest": "",
    }
    try:
        await _SHARE_STORE.replace_authorized(
            share_id, hash_edit_token(edit_token), next_entry
        )
    except ShareStoreError as exc:
        raise _share_store_http_error(exc) from exc
    logger.info(
        json.dumps(
            {
                "event": "share.update",
                "ip": _mask_ip(client_ip),
                "bytes": canonical_bytes,
                "format": fmt,
                "ttl_days": ttl_days,
            }
        )
    )
    base_url = _share_public_base(request)
    return JSONResponse(
        {
            "uuid": share_id,
            "url": f"{base_url}/v1/share#share={share_id}",
            "expiresAt": expires_iso,
        }
    )


@app.post("/v1/share/revoke")
async def share_revoke_fixed(request: Request) -> JSONResponse:
    """Revoke a Share using a fixed request path and memory-only edit token."""
    raw = await _read_limited_body(request, _SHARE_LOCATOR_BODY_BYTES, "Share locator")
    share_id = _parse_share_locator_payload(raw)
    entry = await _share_lookup_live(share_id)
    edit_token = request.headers.get("x-share-edit-token", "")
    if not verify_edit_token(edit_token, entry.get("edit_hash", "")):
        logger.warning(json.dumps({"event": "share.edit_auth_fail"}))
        raise HTTPException(status_code=403, detail="Invalid share edit capability.")
    try:
        await _SHARE_STORE.delete_authorized(share_id, hash_edit_token(edit_token))
    except ShareStoreError as exc:
        raise _share_store_http_error(exc) from exc
    logger.info(json.dumps({"event": "share.revoke"}))
    return JSONResponse({"revoked": True}, headers={"Cache-Control": "no-store"})


@app.head("/v1/share/{share_id}")
async def share_head(share_id: str) -> Response:
    """Deprecate lifecycle probe for pre-generation Share objects only."""
    if not valid_share_id(share_id):
        raise HTTPException(status_code=404, detail="Share not found or expired.")
    entry = await _share_lookup_live(share_id)
    _require_legacy_share_entry(entry)
    logger.info(
        json.dumps(
            {
                "event": "share.status",
                "format": entry["format"],
                "legacy_transport": True,
            }
        )
    )
    return Response(
        status_code=200,
        headers={
            "Cache-Control": "private, no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Robots-Tag": "noindex, nofollow, noarchive",
            **_legacy_share_headers(entry),
        },
    )


@app.get("/v1/share/{share_id}")
async def share_get(share_id: str) -> Response:
    """Deprecate direct renderer for pre-generation Share objects only."""
    if not valid_share_id(share_id):
        raise HTTPException(status_code=404, detail="Share not found or expired.")
    try:
        entry = await _share_lookup_live(share_id)
    except HTTPException as exc:
        if exc.status_code == 404:  # ruff: ignore[magic-value-comparison]
            logger.info(json.dumps({"event": "share.miss"}))
        raise
    _require_legacy_share_entry(entry)

    try:
        content, mime_type, ext = render_share(entry["snapshot"], entry["format"])
    except ShareValidationError as exc:
        logger.error(json.dumps({"event": "share.corrupt_entry"}))
        raise HTTPException(status_code=500, detail="Stored share is invalid.") from exc

    headers = {
        "Content-Disposition": f'inline; filename="ai-conversation{ext}"',
        "Cache-Control": "private, no-store",
        "Pragma": "no-cache",
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": (
            "camera=(), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=()"
        ),
        "X-Robots-Tag": "noindex, nofollow, noarchive",
        **_legacy_share_headers(entry),
    }
    if entry["format"] == "html":
        headers["Content-Security-Policy"] = (
            "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; "
            "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
        )
    logger.info(json.dumps({"event": "share.read", "format": entry["format"]}))
    return Response(
        content=content, status_code=200, media_type=mime_type, headers=headers
    )


@app.patch("/v1/share/{share_id}")
async def share_patch(share_id: str, request: Request) -> JSONResponse:
    """Retired capability-bearing mutation path for pre-generation objects.

    Legacy public links remain readable during their bounded TTL window, but
    extending/updating them through a capability-bearing request path would
    perpetuate that transport generation and create cross-deployment races.
    Current clients must use ``POST /v1/share/update`` with ``shareId`` in the
    bounded body; that route atomically upgrades the stored generation marker.
    """
    if not valid_share_id(share_id):
        raise HTTPException(status_code=404, detail="Share not found or expired.")
    entry = await _share_lookup_live(share_id)
    _require_legacy_share_entry(entry)
    logger.info(json.dumps({"event": "share.legacy_update_retired"}))
    return JSONResponse(
        {"error": "Legacy Share update path is retired. Use POST /v1/share/update."},
        status_code=410,
        headers={"Cache-Control": "no-store", **_legacy_share_headers(entry)},
    )


@app.delete("/v1/share/{share_id}")
async def share_delete(share_id: str, request: Request) -> JSONResponse:
    """Revoke a Global Share; public read capability alone is insufficient."""
    if not valid_share_id(share_id):
        raise HTTPException(status_code=404, detail="Share not found or expired.")
    entry = await _share_lookup_live(share_id)
    _require_legacy_share_entry(entry)
    edit_token = request.headers.get("x-share-edit-token", "")
    if not verify_edit_token(edit_token, entry.get("edit_hash", "")):
        logger.warning(json.dumps({"event": "share.edit_auth_fail"}))
        raise HTTPException(status_code=403, detail="Invalid share edit capability.")
    try:
        await _SHARE_STORE.delete_authorized(share_id, hash_edit_token(edit_token))
    except ShareStoreError as exc:
        raise _share_store_http_error(exc) from exc
    logger.info(json.dumps({"event": "share.revoke", "legacy_transport": True}))
    return JSONResponse(
        {"revoked": True},
        headers={"Cache-Control": "no-store", **_legacy_share_headers(entry)},
    )


def _feedback_review_enabled() -> bool:
    return FEEDBACK_REVIEW_MODE == "provider-pr"


def _feedback_review_pipeline_ready() -> bool:
    if not _feedback_review_enabled():
        return False
    if _FEEDBACK_REVIEW_LEDGER_CONFIG_ERROR or not _FEEDBACK_REVIEW_LEDGER_READY:
        return False
    manifest = _FEEDBACK_REVIEW_LEDGER.manifest()
    if FEEDBACK_REVIEW_REQUIRE_DURABLE and not bool(manifest.get("durable")):
        return False
    if FEEDBACK_REVIEW_REQUIRE_SHARED and not (
        bool(manifest.get("shared")) and bool(manifest.get("authoritative"))
    ):
        return False
    return _STORAGE.primary is not None and _STORAGE.primary_ready()


def _validate_feedback_review_payload(  # ruff: ignore[too-many-branches]
    payload: Any,
) -> dict[str, Any]:
    """Validate explicit one-Q&A maintainer feedback without silent truncation."""
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422, detail="Feedback review body must be an object."
        )
    if payload.get("schemaVersion") != 1:
        raise HTTPException(
            status_code=422, detail="Unsupported feedback review schemaVersion."
        )
    if (
        payload.get("consentFlag") is not True
        or payload.get("consentVersion") != FEEDBACK_REVIEW_CONSENT_VERSION
    ):
        raise HTTPException(
            status_code=403, detail="Explicit feedback review permission is required."
        )
    if (
        payload.get("trainingConsentFlag") is not True
        or payload.get("trainingConsentVersion") != FEEDBACK_TRAINING_CONSENT_VERSION
    ):
        raise HTTPException(
            status_code=403, detail="Explicit feedback training permission is required."
        )
    query = payload.get("query")
    answer = payload.get("answer")
    message = payload.get("message", "")
    if not isinstance(query, str) or not query.strip():
        raise HTTPException(
            status_code=422, detail="Feedback review requires the question text."
        )
    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(
            status_code=422, detail="Feedback review requires the answer text."
        )
    if (
        len(query) > MAX_CONVERSATION_MESSAGE_CHARS
        or len(answer) > MAX_CONVERSATION_MESSAGE_CHARS
    ):
        raise HTTPException(
            status_code=422, detail="Feedback review Q&A exceeds the supported size."
        )
    if not isinstance(message, str) or len(message) > MAX_CONTRIBUTION_NOTE_CHARS:
        raise HTTPException(
            status_code=422, detail="Feedback review note exceeds the supported size."
        )
    value = payload.get("ratingValue")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise HTTPException(
            status_code=422, detail="Feedback review requires a numeric rating."
        )
    scale_min = payload.get("ratingScaleMin")
    scale_max = payload.get("ratingScaleMax")
    if (
        not isinstance(scale_min, (int, float))
        or isinstance(scale_min, bool)
        or not isinstance(scale_max, (int, float))
        or isinstance(scale_max, bool)
        or scale_min >= scale_max
        or value < scale_min
        or value > scale_max
    ):
        raise HTTPException(
            status_code=422, detail="Feedback review rating scale is invalid."
        )
    mode = payload.get("ratingMode")
    if mode not in {"quick", "panel"}:
        raise HTTPException(
            status_code=422, detail="Feedback review ratingMode must be quick or panel."
        )
    label = payload.get("ratingLabel")
    title = payload.get("ratingTitle")

    def is_above_thr(
        label: str = label,
        thr: int = 64,
    ):
        return len(label) > thr

    if label is not None and (not isinstance(label, str) or is_above_thr(label, 64)):
        raise HTTPException(
            status_code=422, detail="Feedback review ratingLabel is invalid."
        )
    if title is not None and (not isinstance(title, str) or is_above_thr(label, 128)):
        raise HTTPException(
            status_code=422, detail="Feedback review ratingTitle is invalid."
        )
    return payload


def _feedback_review_reference(
    entry: dict[str, Any], review: Any | None = None
) -> dict[str, Any]:
    storage = entry.get("storage") if isinstance(entry.get("storage"), dict) else {}
    raw_review = (
        storage.get("review") if isinstance(storage.get("review"), dict) else {}
    )
    paths = storage.get("paths") if isinstance(storage.get("paths"), dict) else {}
    provider = str(getattr(review, "provider", "") or raw_review.get("provider") or "")[
        :32
    ]
    review_id = str(
        getattr(review, "review_id", "") or raw_review.get("reviewId") or ""
    )[:32]
    path = str(getattr(review, "path", "") or "")
    if not path:
        target_id = str(raw_review.get("targetId") or "")
        if target_id and target_id in paths:
            path = str(paths.get(target_id) or "")
        elif len(paths) == 1:
            path = str(next(iter(paths.values())) or "")
    path = path.replace("\\", "/").replace("\r", "").replace("\n", "")[:512]
    if not path or path.startswith("/") or ".." in path.split("/"):
        path = ""
    out: dict[str, Any] = {}
    if provider:
        out["reviewProvider"] = provider
    if review_id.isdigit():
        out["reviewId"] = review_id
    if path:
        out["reviewPath"] = path
    return out


def _feedback_review_public_status(
    entry: dict[str, Any], review: Any | None = None
) -> dict[str, Any]:
    state = str(entry.get("state") or "unknown")
    review_state = str(getattr(review, "status", "") or "").lower()
    if state == "eligible":
        status = "reviewed"
    elif state == "withdrawn":
        status = "withdrawn"
    elif state in {"deleted", "expired"}:
        status = state
    elif review_state in {"closed", "rejected"}:
        status = "rejected"
    else:
        status = "in_review"
    body = {
        "status": status,
        "reviewMode": FEEDBACK_REVIEW_MODE,
        "trainingEligible": state == "eligible",
        "feedbackReview": True,
        "reviewRevision": max(
            1, int((entry.get("operation") or {}).get("reviewRevision") or 1)
        ),
        "expiresAt": (
            int(float(entry.get("expiresAt") or 0) * 1000)
            if entry.get("expiresAt")
            else None
        ),
    }
    if review is not None:
        body["reviewStatus"] = review.status
    body.update(_feedback_review_reference(entry, review))
    return body


def _feedback_review_response(
    entry: dict[str, Any],
    *,
    receipt_id: str,
    delete_token: str,
    review: Any | None,
    replay: bool = False,
    review_update: str = "",
) -> JSONResponse:
    body = {
        "accepted": True,
        "receiptId": receipt_id,
        "idempotentReplay": bool(replay),
        **_feedback_review_public_status(entry, review),
    }
    if review_update:
        body["reviewUpdate"] = review_update
    if delete_token:
        body["deleteToken"] = delete_token
    return JSONResponse(body, headers={"Cache-Control": "no-store"})


async def _authorized_feedback_review_entry(
    receipt_id: str, request: Request
) -> dict[str, Any]:
    try:
        entry = await _FEEDBACK_REVIEW_LEDGER.get(receipt_id)
    except ContributionLedgerError as exc:
        raise _contribution_ledger_http_error(exc) from exc
    if entry is None:
        raise HTTPException(
            status_code=404, detail="Feedback review receipt not found."
        )
    supplied = request.headers.get("X-Feedback-Review-Token", "")
    if not supplied or not verify_edit_token(
        supplied, str(entry.get("deleteTokenHash") or "")
    ):
        raise HTTPException(
            status_code=403, detail="Invalid feedback review management capability."
        )
    return entry


async def _ensure_feedback_provider_review(entry: dict[str, Any]):
    if not _feedback_review_enabled():
        return None
    receipt_id = str(entry.get("receiptId") or "")
    storage_hint = (
        entry.get("storage") if isinstance(entry.get("storage"), dict) else {}
    )
    if isinstance(storage_hint.get("review"), dict):
        review = await _STORAGE.get_feedback_review(
            receipt_id, review_hint=storage_hint
        )
        if review is None:
            raise StorageWriteError("REVIEW_NOT_FOUND")
        return review
    records = [row for row in entry.get("records", []) if isinstance(row, dict)]
    if not records:
        raise StorageWriteError("EMPTY_REVIEW")
    content = ("\n".join(json.dumps(r, ensure_ascii=False) for r in records)).encode(
        "utf-8"
    )
    review = await _STORAGE.open_feedback_review(
        receipt_id=receipt_id,
        content=content,
        commit_message="Open maintainer feedback review · revision 1",
        path_timestamp=float(entry.get("receivedAt") or _time.time()),
    )
    try:
        bound = await _FEEDBACK_REVIEW_LEDGER.set_pending_storage(
            receipt_id, storage=review.storage_metadata()
        )
        entry["storage"] = bound.get("storage") or review.storage_metadata()
    except ContributionLedgerError as exc:
        raise StorageWriteError("REVIEW_BIND_LEDGER", transient=True) from exc
    return review


async def _sync_feedback_review_merge(
    entry: dict[str, Any],
) -> tuple[dict[str, Any], Any | None]:
    if not _feedback_review_enabled() or str(entry.get("state") or "") != "quarantined":
        return entry, None
    review = await _ensure_feedback_provider_review(entry)
    if review is None or review.status != "merged":
        return entry, review
    try:
        claimed = await _FEEDBACK_REVIEW_LEDGER.begin_promotion(
            str(entry.get("receiptId") or "")
        )
        claim = str(claimed.get("operationClaim") or "") or None
        reviewed = await _FEEDBACK_REVIEW_LEDGER.mark_promoted(
            str(entry.get("receiptId") or ""),
            storage=review.storage_metadata(),
            claim_token=claim,
        )
        logger.info(
            json.dumps({"event": "feedback_review.merged", "provider": review.provider})
        )
        return reviewed, review
    except ContributionLedgerError:
        current = await _FEEDBACK_REVIEW_LEDGER.get(str(entry.get("receiptId") or ""))
        if current is not None and str(current.get("state") or "") == "eligible":
            return current, review
        raise


@app.post("/v1/feedback/review")
async def create_feedback_review(request: Request) -> JSONResponse:
    """Create one explicit content-bearing maintainer feedback review."""
    if not _feedback_review_pipeline_ready():
        raise HTTPException(
            status_code=503, detail="Feedback review workflow is unavailable."
        )
    raw = await _read_limited_body(
        request, FEEDBACK_REVIEW_MAX_BODY_BYTES, "Feedback review"
    )
    try:
        payload = _validate_feedback_review_payload(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
    envelope = _operation_envelope(
        request,
        payload,
        "feedback-review",
        max_age_ms=FEEDBACK_REVIEW_TTL_SECONDS * 1000,
    )
    if envelope:
        receipt_id, delete_hash, operation_id = envelope
        delete_token = ""
    else:
        receipt_id, delete_token, operation_id = (
            uuid.uuid4().hex,
            generate_edit_token(),
            "",
        )
        delete_hash = hash_edit_token(delete_token)
    payload_digest = _canonical_payload_digest(payload)

    if envelope:
        try:
            existing = await _FEEDBACK_REVIEW_LEDGER.get(receipt_id)
        except ContributionLedgerError as exc:
            raise _contribution_ledger_http_error(exc) from exc
        if existing is not None:
            op = (
                existing.get("operation")
                if isinstance(existing.get("operation"), dict)
                else {}
            )
            if not (
                secrets.compare_digest(
                    str(op.get("payloadDigest") or ""), payload_digest
                )
                and secrets.compare_digest(
                    str(op.get("operationId") or ""), operation_id
                )
                and secrets.compare_digest(
                    str(existing.get("deleteTokenHash") or ""), delete_hash
                )
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Feedback review operation identity was reused with different content.",
                )
            review = await _ensure_feedback_provider_review(existing)
            return _feedback_review_response(
                existing,
                receipt_id=receipt_id,
                delete_token=delete_token,
                review=review,
                replay=True,
                review_update="unchanged",
            )

    client_ip = _client_ip(request)
    allowed, _count = await _consume_rate_limit(
        _feedback_review_rl,
        _feedback_review_rl_lock,
        client_ip,
        limit=FEEDBACK_REVIEW_RATE_LIMIT_PER_HOUR,
        scope="feedback-review",
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for feedback reviews.",
            headers={"Retry-After": "3600"},
        )

    server_ts_ms = int(_time.time() * 1000)
    record = normalize_feedback_review_record(
        payload, server_ts_ms=server_ts_ms, receipt_id=receipt_id
    )
    encoded = json.dumps(record, ensure_ascii=False).encode("utf-8")
    entry = {
        "receiptId": receipt_id,
        "kind": "feedback-review",
        "state": "quarantined",
        "records": [record],
        "bytes": len(encoded),
        "deleteTokenHash": delete_hash,
        "expiresAt": _time.time() + FEEDBACK_REVIEW_TTL_SECONDS,
        "receivedAt": _time.time(),
        "dedupKeys": [str(record.get("_dedup_key") or "")],
        "storage": {},
        "withdrawalStorage": {},
        "currentViewRemoval": {},
        "lastError": "",
        "operation": {
            "payloadDigest": payload_digest,
            "operationId": operation_id,
            "reviewRevision": 1,
        },
        "rowCount": 1,
    }
    try:
        await _FEEDBACK_REVIEW_LEDGER.create(entry)
    except ContributionLedgerError as exc:
        raise _contribution_ledger_http_error(exc) from exc
    try:
        review = await _ensure_feedback_provider_review(entry)
    except StorageWriteError as exc:
        logger.error(
            json.dumps({"event": "feedback_review.open_fail", "code": exc.code})
        )
        raise HTTPException(
            status_code=503,
            detail="Feedback was accepted into review lifecycle but its repository review could not be opened. Retry the same action.",
        ) from exc
    logger.info(
        json.dumps({"event": "feedback_review.opened", "provider": review.provider})
    )
    return _feedback_review_response(
        entry, receipt_id=receipt_id, delete_token=delete_token, review=review
    )


@app.put("/v1/feedback/review/{receipt_id}")
async def update_feedback_review(receipt_id: str, request: Request) -> JSONResponse:
    """Update one open feedback review; identical content is a no-op."""
    entry = await _authorized_feedback_review_entry(receipt_id, request)
    if str(entry.get("state") or "") == "quarantined":
        try:
            entry, current_review = await _sync_feedback_review_merge(entry)
        except (StorageWriteError, ContributionLedgerError) as exc:
            raise HTTPException(
                status_code=503, detail="Feedback review state could not be refreshed."
            ) from exc
    else:
        current_review = None
    if str(entry.get("state") or "") != "quarantined":
        raise HTTPException(
            status_code=409, detail="Feedback review is no longer open for updates."
        )
    if current_review is not None and current_review.status in {"closed", "rejected"}:
        raise HTTPException(
            status_code=409, detail="Feedback review was closed and cannot be updated."
        )
    raw = await _read_limited_body(
        request, FEEDBACK_REVIEW_MAX_BODY_BYTES, "Feedback review"
    )
    try:
        payload = _validate_feedback_review_payload(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body.") from exc
    digest = _canonical_payload_digest(payload)
    operation = (
        entry.get("operation") if isinstance(entry.get("operation"), dict) else {}
    )
    if str(operation.get("payloadDigest") or "") and secrets.compare_digest(
        str(operation.get("payloadDigest") or ""), digest
    ):
        review = current_review or await _ensure_feedback_provider_review(entry)
        return _feedback_review_response(
            entry,
            receipt_id=receipt_id,
            delete_token="",
            review=review,
            replay=True,
            review_update="unchanged",
        )
    record = normalize_feedback_review_record(
        payload, server_ts_ms=int(_time.time() * 1000), receipt_id=receipt_id
    )
    encoded = json.dumps(record, ensure_ascii=False).encode("utf-8")
    next_revision = max(1, int(operation.get("reviewRevision") or 1)) + 1
    try:
        review = await _STORAGE.update_feedback_review(
            receipt_id=receipt_id,
            content=encoded,
            commit_message=f"Update maintainer feedback review · revision {next_revision}",
            path_timestamp=float(entry.get("receivedAt") or _time.time()),
            review_hint=(
                entry.get("storage") if isinstance(entry.get("storage"), dict) else None
            ),
        )
    except StorageWriteError as exc:
        if exc.code in {"REVIEW_CLOSED", "REVIEW_MERGED", "REVIEW_NOT_FOUND"}:
            raise HTTPException(
                status_code=409, detail="Feedback review is no longer open for updates."
            ) from exc
        raise HTTPException(
            status_code=503, detail="Feedback review could not be updated safely."
        ) from exc
    try:
        updated = await _FEEDBACK_REVIEW_LEDGER.replace_pending_payload(
            receipt_id,
            records=[record],
            byte_count=len(encoded),
            dedup_keys=[str(record.get("_dedup_key") or "")],
            payload_digest=digest,
            row_count=1,
            storage=review.storage_metadata(),
        )
    except ContributionLedgerError as exc:
        raise _contribution_ledger_http_error(exc) from exc
    logger.info(
        json.dumps(
            {
                "event": "feedback_review.updated",
                "revision": next_revision,
                "provider": review.provider,
            }
        )
    )
    return _feedback_review_response(
        updated,
        receipt_id=receipt_id,
        delete_token="",
        review=review,
        review_update="updated",
    )


@app.get("/v1/feedback/review/{receipt_id}")
async def feedback_review_status(  # ruff: ignore[undocumented-public-function]
    receipt_id: str,
    request: Request,
) -> JSONResponse:
    entry = await _authorized_feedback_review_entry(receipt_id, request)
    review = None
    if str(entry.get("state") or "") == "quarantined" and _feedback_review_enabled():
        try:
            entry, review = await _sync_feedback_review_merge(entry)
        except (StorageWriteError, ContributionLedgerError) as exc:
            logger.warning(
                json.dumps(
                    {
                        "event": "feedback_review.status_fail",
                        "error_type": type(exc).__name__,
                    }
                )
            )
    return JSONResponse(
        _feedback_review_public_status(entry, review),
        headers={"Cache-Control": "no-store"},
    )


@app.delete("/v1/feedback/review/{receipt_id}")
async def withdraw_feedback_review(receipt_id: str, request: Request) -> JSONResponse:
    """Withdraw pending or merged maintainer feedback using participant authority."""
    entry = await _authorized_feedback_review_entry(receipt_id, request)
    state = str(entry.get("state") or "")
    if state == "quarantined":
        try:
            entry, review = await _sync_feedback_review_merge(entry)
            state = str(entry.get("state") or "")
        except (StorageWriteError, ContributionLedgerError) as exc:
            raise HTTPException(
                status_code=503, detail="Feedback review state could not be refreshed."
            ) from exc
        if state == "quarantined":
            if review is not None and review.status not in {"closed", "rejected"}:
                try:
                    await _STORAGE.close_feedback_review(
                        receipt_id,
                        review_hint=(
                            entry.get("storage")
                            if isinstance(entry.get("storage"), dict)
                            else None
                        ),
                    )
                except StorageWriteError as exc:
                    raise HTTPException(
                        status_code=503,
                        detail="Feedback review could not be closed safely.",
                    ) from exc
            try:
                deleted = await _FEEDBACK_REVIEW_LEDGER.delete_pending(receipt_id)
            except ContributionLedgerError as exc:
                raise _contribution_ledger_http_error(exc) from exc
            logger.info(json.dumps({"event": "feedback_review.withdrawn_pending"}))
            body = _feedback_review_public_status(deleted, review)
            body["status"] = "withdrawn"
            return JSONResponse(body, headers={"Cache-Control": "no-store"})
    if state in {"deleted", "withdrawn"}:
        body = _feedback_review_public_status(entry, None)
        body["status"] = "withdrawn"
        return JSONResponse(body, headers={"Cache-Control": "no-store"})
    if state == "expired":
        raise HTTPException(status_code=410, detail="Feedback review receipt expired.")
    if state not in {"eligible", "promotion_uncertain", "withdrawal_uncertain"}:
        raise HTTPException(
            status_code=409,
            detail="Feedback review cannot be withdrawn in its current state.",
        )
    try:
        claimed = await _FEEDBACK_REVIEW_LEDGER.begin_withdrawal(receipt_id)
        claim = str(claimed.get("operationClaim") or "") or None
    except ContributionLedgerError as exc:
        raise _contribution_ledger_http_error(exc) from exc
    storage = claimed.get("storage") if isinstance(claimed.get("storage"), dict) else {}
    removal = await _STORAGE.remove_current_view(
        dict(storage.get("paths") or {}),
        record_id=str(storage.get("recordId") or "") or None,
    )
    try:
        withdrawn = await _FEEDBACK_REVIEW_LEDGER.mark_withdrawn(
            receipt_id,
            withdrawal_storage={},
            current_view_removal=removal,
            claim_token=claim,
        )
    except ContributionLedgerError as exc:
        raise HTTPException(
            status_code=503,
            detail="Feedback removal completed but lifecycle confirmation failed.",
        ) from exc
    logger.info(json.dumps({"event": "feedback_review.withdrawn_reviewed"}))
    body = _feedback_review_public_status(withdrawn, None)
    body["status"] = "withdrawn"
    body["currentViewRemoval"] = removal
    return JSONResponse(body, status_code=202, headers={"Cache-Control": "no-store"})


@app.post("/v1/feedback")
async def feedback(request: Request) -> JSONResponse:
    """Accept privacy-minimal rating telemetry.

    Every request must carry the current explicit telemetry-consent marker.
    Direct callers may still submit query, answer, comment, model, page or
    conversation identifiers, but the server normalizer discards them before
    optional persistence. Persistence is server opt-in
    (``FEEDBACK_PERSIST_ENABLED=false`` by default) and persisted rows are
    ``trainingStatus=telemetry`` so the training builder excludes them.
    """
    # Body size guard — enforced while streaming, before full allocation.
    raw = await _read_limited_body(request, FEEDBACK_MAX_BODY_BYTES, "Feedback")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=422, detail="Feedback body must be a JSON object."
        )

    # Network feedback is user-consented telemetry, not an implicit side effect
    # of clicking a local rating button. Reject legacy/misbehaving clients that
    # do not carry the current explicit consent marker. This check occurs before
    # rate-limit/storage work so denied telemetry consumes minimal server state.
    if (
        payload.get("telemetryConsent") is not True
        or payload.get("telemetryConsentVersion") != FEEDBACK_TELEMETRY_CONSENT_VERSION
    ):
        logger.warning(json.dumps({"event": "feedback.consent_required"}))
        raise HTTPException(
            status_code=403,
            detail="Explicit feedback telemetry permission is required.",
        )
    if payload.get("schemaVersion") != FEEDBACK_TELEMETRY_SCHEMA_VERSION:
        raise HTTPException(
            status_code=422, detail="Unsupported feedback telemetry schema version."
        )
    consent_at = payload.get("telemetryConsentAt")
    if (
        not isinstance(consent_at, (int, float))
        or isinstance(consent_at, bool)
        or consent_at <= 0
    ):
        raise HTTPException(
            status_code=422, detail="Feedback telemetry consent timestamp is required."
        )

    # ── Distinguish retraction tombstones from regular ratings ───────────────
    # Retractions are system-generated housekeeping records that invalidate a
    # previous rating in the training dataset.  They carry action="retract" and
    # prevSessionId (pointing to the original record) but NO ratingValue.
    # Key behavioural differences vs regular feedback:
    #   1. Counted against the same bounded abuse gate as every feedback write;
    #      retractions must not create an unlimited write path.
    #   2. Validated differently — prevSessionId is required; ratingValue is absent.
    #   3. Logged with event "feedback.retract" so operators can distinguish
    #      retraction volume from new-rating volume in log dashboards.
    #   4. Committed with a distinct commit_message so the HF repo history is legible.
    is_retract: bool = payload.get("action") == "retract"

    client_ip = _client_ip(request)
    allowed, _count = await _consume_rate_limit(
        _feedback_rl,
        _feedback_rl_lock,
        client_ip,
        limit=FEEDBACK_RATE_LIMIT_PER_HOUR,
        scope="feedback",
    )
    if not allowed:
        logger.warning(
            json.dumps({"event": "feedback.ratelimit", "ip": _mask_ip(client_ip)})
        )
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for feedback submissions.",
            headers={"Retry-After": "3600"},
        )

    if is_retract:
        if not payload.get("prevSessionId"):
            raise HTTPException(
                status_code=422,
                detail="Retraction records must include a non-empty prevSessionId.",
            )
        logger.info(
            json.dumps(
                {"event": "feedback.retract", "persist": FEEDBACK_PERSIST_ENABLED}
            )
        )
    else:
        logger.info(
            json.dumps(
                {"event": "feedback.receive", "persist": FEEDBACK_PERSIST_ENABLED}
            )
        )

    # ── Optional provider-neutral record persistence ─────────────────────────
    # Activated only when FEEDBACK_PERSIST_ENABLED=true and a primary storage
    # target is configured. Failures are logged and swallowed
    # so that a dataset-write error never breaks the user's rating experience
    # (the keepalive fire-and-forget model means the user won't see a retry UI
    # anyway).  Operators should monitor "feedback.persist_fail" log events.
    #
    # Persisted feedback remains telemetry-only and is excluded from training.
    if FEEDBACK_PERSIST_ENABLED and _STORAGE.primary is not None:
        try:
            _rec_dict: dict = normalize_feedback_record(
                payload,
                server_ts_ms=int(_time.time() * 1000),
            )
            record: str = json.dumps(_rec_dict, ensure_ascii=False)
            commit_msg = (
                "Retract 1 feedback record" if is_retract else "Add 1 feedback record"
            )
            receipt = await _persist_storage_record(
                kind="feedback",
                content=record.encode("utf-8"),
                commit_message=commit_msg,
            )
            logger.info(
                json.dumps(
                    {
                        "event": "feedback.persist_ok",
                        "retract": is_retract,
                        "mirror_count": len(receipt.mirrors),
                        "mirror_degraded": sum(
                            1 for _v in receipt.mirrors.values() if _v != "ok"
                        ),
                    }
                )
            )
        except StorageWriteError as exc:
            # Never propagate: rating UX remains successful even when storage is
            # degraded. Log only the sanitized adapter code, never provider body.
            logger.error(
                json.dumps({"event": "feedback.persist_fail", "code": exc.code})
            )

    return JSONResponse({"ok": True})
