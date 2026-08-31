#!/usr/bin/env python3
# ruff: noqa: EXE001
# dev_proxy.py  —  Path E: Local Python Development Proxy
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
#   pip install httpx
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   python dev_proxy.py
#
# Terminal 2 (docs server):
#   make html  &&  python -m http.server 8080 --directory _build/html
#
# Then open http://localhost:8080 and set conf.py endpoint to:
#   http://localhost:8787/v1/chat/completions
#
# CONSTRAINTS
# ───────────
# • Single-threaded: one slow request (30-90 s for large models) blocks all
#   other requests.  Open only one browser tab while a request is in flight.
# • Requires model IDs with Inference Providers: Qwen/Qwen2.5-Coder-32B-Instruct works;
#   scikit-plots/Qwen2.5-Coder-32B-Instruct does NOT (mirror — no provider on router).
# • HTTP only — no TLS.  For HTTPS, use path_b/app.py (FastAPI + uvicorn).
# • Never expose on a public network: no rate limiting, no authentication.
#
# DEPENDENCIES
# ────────────
# stdlib only + httpx (pip install httpx).  No FastAPI, no asyncio.
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal blocking HTTP proxy for local sphinx-ai-assistant development.

Accepts the ``scikitplot-chat-v1`` browser envelope, constructs the
authoritative provider request locally, and forwards it to the HuggingFace
Inference API with :data:`HF_TOKEN` injected server-side.

Notes
-----
**Developer note** — This server is intentionally synchronous and
single-threaded.  One slow upstream response blocks all other requests.
This is an acceptable trade-off for a local development tool used in a
single-browser-tab workflow.  For concurrent use or SSE streaming, use
``path_b/app.py`` (FastAPI + uvicorn).

**User note** — If the browser appears frozen while a response is loading,
this is expected behaviour.  Avoid opening multiple browser tabs while this
proxy is handling a request.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import NoReturn

import httpx

# ── Path 0 stub responder ─────────────────────────────────────────────────────
#
# The implementation lives in _hf_spaces_proxy/ because that directory is the
# deployable unit the Dockerfile copies; importing it from here rather than
# re-implementing keeps ONE responder, so a rig verified locally is the same
# code that answers in production.
#
# The import is guarded because dev_proxy.py must stay runnable from a checkout
# that has the file missing or moved.  When it fails the stub is simply
# unavailable and a stub/* model id falls through to normal routing and errors
# there visibly -- never silently succeeding, which would make a disabled rig
# indistinguishable from a working one.
try:  # pragma: no cover - exercised by presence/absence of the sibling package
    sys.path.insert(
        0, str(pathlib.Path(__file__).resolve().parent / "_hf_spaces_proxy")
    )
    from _chat_contract import (  # type: ignore[import]
        CHAT_CONTRACT,
        ChatContractError,
        encode_upstream_payload,
        parse_chat_request,
    )
    from _shared_logic import _validate_credential_destination  # type: ignore[import]
    from _stub_model import (  # type: ignore[import]  # ruff: ignore[unused-import]
        is_stub_model,
        parse_stub_mode,
        stub_delay_ms,
        stub_modes,
        stub_payload,
        stub_sse_frames,
    )

    _STUB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _STUB_AVAILABLE = False
    CHAT_CONTRACT = "scikitplot-chat-v1"

#: Whether the stub responder answers.  Off unless explicitly enabled, matching
#: the HF Space proxy so the two behave identically by default.
STUB_ENABLED: bool = os.environ.get("STUB_ENABLED", "false").strip().lower() in (
    "true",
    "1",
    "yes",
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="[dev_proxy] %(levelname)s %(message)s",
    level=logging.INFO,
)
_LOG = logging.getLogger("dev_proxy")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — read once at startup from environment variables
# ─────────────────────────────────────────────────────────────────────────────

#: HuggingFace API token.  Required — proxy exits with clear message if absent.
HF_TOKEN: str = os.environ.get("HF_TOKEN", "").strip()

#: HuggingFace Inference Providers base URL (no trailing slash).
#: Migrated to router.huggingface.co (v6.0.0); the legacy
#: api-inference.huggingface.co hostname is DNS-unresolvable.
HF_BASE: str = os.environ.get(
    "HF_BASE",
    "https://router.huggingface.co",
).rstrip("/")

#: Fallback model ID when the request body omits the ``model`` field.
#: Must have a registered HF Inference Provider on router.huggingface.co.
#:   ✓  Qwen/Qwen2.5-Coder-32B-Instruct   (original — has provider)
#:   ✓  Qwen/Qwen2.5-72B-Instruct
#:   ✗  scikit-plots/Qwen2.5-Coder-32B-Instruct  (mirror — no provider → 404/503)
#:      To serve scikit-plots/* models use the full proxy with Path-2 routing.
DEFAULT_MODEL: str = os.environ.get(
    "DEFAULT_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct"
).strip()

_raw_allowed_models = os.environ.get("ALLOWED_MODELS", DEFAULT_MODEL)
ALLOWED_MODELS: tuple[str, ...] = tuple(
    dict.fromkeys(x.strip() for x in _raw_allowed_models.split(",") if x.strip())
) or (DEFAULT_MODEL,)

#: Local port the proxy listens on.
PORT: int = int(os.environ.get("DEV_PROXY_PORT", "8787"))

#: Upstream read timeout in seconds.
#: A 20B model on HF Serverless API can take 30-90 seconds to respond.
TIMEOUT: int = int(os.environ.get("PROXY_TIMEOUT", "120"))
MAX_BODY_BYTES: int = max(
    16_384,
    min(
        int(
            os.environ.get("MAX_BODY_BYTES", str(10 * 1024 * 1024))
            or (10 * 1024 * 1024)
        ),
        16 * 1024 * 1024,
    ),
)
MAX_UPSTREAM_RESPONSE_BYTES: int = max(
    64 * 1024,
    min(
        int(
            os.environ.get("MAX_UPSTREAM_RESPONSE_BYTES", str(8 * 1024 * 1024))
            or (8 * 1024 * 1024)
        ),
        32 * 1024 * 1024,
    ),
)
_raw_allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
).strip()
ALLOWED_ORIGINS: tuple[str, ...] = (
    ("*",)
    if _raw_allowed_origins == "*"
    else tuple(
        dict.fromkeys(
            x.strip().rstrip("/") for x in _raw_allowed_origins.split(",") if x.strip()
        )
    )
)


# ─────────────────────────────────────────────────────────────────────────────
# Startup validation — fail fast with actionable message
# ─────────────────────────────────────────────────────────────────────────────


def _fail(message: str) -> NoReturn:
    """Print *message* to stderr and exit with code 1."""
    _LOG.error(message)
    sys.exit(1)


if not HF_TOKEN:
    _fail(
        "HF_TOKEN environment variable is not set.\n"
        "Export it before running:\n"
        "  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
        "Or use Path A (Docker Model Runner) — it requires no token."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_model(body: bytes) -> str:
    """
    Extract the ``model`` field from a raw JSON request body.

    Parameters
    ----------
    body : bytes
        Raw HTTP request body from the browser.

    Returns
    -------
    str
        The ``model`` value, or :data:`DEFAULT_MODEL` when absent or on
        any parse error.

    Notes
    -----
    **Developer note** — Never raises.  Malformed bodies fall back to
    ``DEFAULT_MODEL`` so the upstream error is returned to the browser.
    """
    try:
        data = json.loads(body)
        candidate = str(data.get("model", "")).strip()
        return candidate or DEFAULT_MODEL
    except (json.JSONDecodeError, ValueError, AttributeError):
        return DEFAULT_MODEL


def _origin_allowed(origin: str) -> bool:
    """Return whether a browser Origin is allowed to call the loopback proxy."""
    origin = (origin or "").strip().rstrip("/")
    return not origin or ALLOWED_ORIGINS == ("*",) or origin in ALLOWED_ORIGINS


def _build_cors_headers(origin: str = "") -> dict[str, str]:
    """Return CORS headers only for an explicitly allowed browser origin."""
    origin = (origin or "").strip().rstrip("/")
    headers = {
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    if origin and _origin_allowed(origin):
        headers["Access-Control-Allow-Origin"] = (
            "*" if ALLOWED_ORIGINS == ("*",) else origin
        )
        if ALLOWED_ORIGINS != ("*",):
            headers["Vary"] = "Origin"
    return headers


class _UpstreamResponseTooLarge(  # ruff: ignore[error-suffix-on-exception-name]
    RuntimeError,
):
    """Raised when a dev-proxy upstream response crosses the hard byte cap."""


def _read_upstream_limited(
    response: httpx.Response, max_bytes: int | None = None
) -> bytes:
    """Read a streamed upstream response without ever buffering past *max_bytes*."""
    limit = MAX_UPSTREAM_RESPONSE_BYTES if max_bytes is None else int(max_bytes)
    raw_length = (response.headers.get("content-length") or "").strip()
    if raw_length and (not raw_length.isdigit() or int(raw_length) > limit):
        raise _UpstreamResponseTooLarge("upstream response too large")

    out = bytearray()
    total = 0
    for chunk in response.iter_bytes():
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise _UpstreamResponseTooLarge("upstream response too large")
        out.extend(chunk)
    return bytes(out)


# HF_TOKEN is never sent to an arbitrary operator-supplied origin.
if _STUB_AVAILABLE:
    _validate_credential_destination(HF_BASE, credential_kind="HF_TOKEN")

# ─────────────────────────────────────────────────────────────────────────────
# HTTP handler
# ─────────────────────────────────────────────────────────────────────────────


class ProxyHandler(BaseHTTPRequestHandler):
    """
    Minimal HTTP handler that proxies ``POST /v1/chat/completions`` to HF.

    Notes
    -----
    **Developer note** — Uses the standard-library ``HTTPServer``
    (synchronous, single-threaded).  One request is handled at a time.
    A slow upstream response (30-90 s for large models) blocks all other
    requests.  This is intentional: the tool targets single-tab local dev.

    For concurrent use or SSE streaming, switch to ``path_b/app.py``
    (FastAPI + uvicorn + httpx async).
    """

    def do_OPTIONS(self) -> None:  # noqa: N802  (HTTP method naming)
        """
        Handle CORS preflight request.

        Browsers send ``OPTIONS`` before every cross-origin ``POST``.
        Without a 204 response here, the subsequent ``POST`` is blocked.
        """
        origin = self.headers.get("Origin", "")
        if not _origin_allowed(origin):
            self._write_error(403, "Origin not allowed.", include_cors=False)
            return
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def _write_bytes(self, payload: bytes, *, flush: bool = False) -> bool:
        """Write a response body without surfacing downstream pipe details."""
        try:
            self.wfile.write(payload)
            if flush:
                self.wfile.flush()
            return True
        except (BrokenPipeError, ConnectionResetError):
            # Normal browser/tab disconnect. Never log client address, request
            # body, endpoint, model id or exception text.
            _LOG.warning(
                "AI dev proxy [STREAM_CLOSED]: downstream connection closed; "
                "response ended safely."
            )
            return False

    def do_GET(self) -> None:  # noqa: N802
        """
        Handle ``GET /`` and ``GET /health`` liveness probes.

        Notes
        -----
        Some ``conf.py`` setups probe the proxy URL with ``GET /`` at
        startup to check connectivity.  Without this handler, the probe
        gets a 404 and the page shows a connectivity error even though
        the proxy is working.
        """
        reasoning_enabled = os.environ.get("REASONING_ENABLED", "").strip().lower() in (
            "true",
            "1",
            "yes",
        )
        reasoning_caps: dict[str, object] = {"enabled": False}
        if reasoning_enabled:
            effort_param = os.environ.get(
                "REASONING_EFFORT_PARAM", "reasoning_effort"
            ).strip()
            thinking_param = os.environ.get("REASONING_THINKING_PARAM", "").strip()
            thinking_mode = (
                os.environ.get("REASONING_THINKING_MODE", "budget").strip().lower()
            )
            if thinking_mode not in {"boolean", "adaptive", "budget"}:
                _LOG.error(
                    "AI dev proxy [REASONING_CONFIG_INVALID]: invalid "
                    "REASONING_THINKING_MODE; safe budget mode will be advertised."
                )
                thinking_mode = "budget"

            reasoning_caps = {
                "enabled": True,
                "effort_enabled": bool(effort_param),
                "thinking_enabled": bool(thinking_param),
                "budget_min": 500,
                "budget_max": 16000,
            }
            if effort_param:
                reasoning_caps["effort_param"] = effort_param
                reasoning_caps["effort_values"] = {
                    "low": "low",
                    "medium": "medium",
                    "high": "high",
                    "extra": "high",
                    "max": "high",
                }
            if thinking_param:
                reasoning_caps["thinking_param"] = thinking_param
                reasoning_caps["thinking_mode"] = thinking_mode

        body = json.dumps(
            {
                "status": "ok",
                "service": "sphinx-ai-assistant dev proxy",
                # Same discovery contract as the HF Spaces proxy. It is
                # fail-closed by default and can independently advertise
                # Effort and Thinking when a local adapter is known to forward
                # those fields.
                "capabilities": {
                    "reasoning": reasoning_caps,
                    "chat_request": {"contract": CHAT_CONTRACT},
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self._write_bytes(body)

    def do_POST(  # noqa: N802  # ruff: ignore[too-many-branches, too-many-return-statements]
        self,
    ) -> None:
        """
        Forward ``POST`` body to HuggingFace and write the response back.

        Both ``/v1/chat/completions`` and ``/`` are accepted so that
        ``conf.py`` endpoints with or without the path suffix both work.
        """
        origin = self.headers.get("Origin", "")
        if not _origin_allowed(origin):
            self._write_error(403, "Origin not allowed.", include_cors=False)
            return

        # BaseHTTPRequestHandler does not decode chunked request bodies. Require
        # a valid Content-Length and reject oversize input before rfile.read().
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._write_error(411, "Content-Length is required.")
            return
        try:
            length = int(raw_length)
        except (TypeError, ValueError):
            self._write_error(400, "Invalid Content-Length header.")
            return
        if length < 0:
            self._write_error(400, "Invalid Content-Length header.")
            return
        if length > MAX_BODY_BYTES:
            self._write_error(413, "Request body too large.")
            return
        body: bytes = self.rfile.read(length)
        if len(body) != length:
            self._write_error(400, "Incomplete request body.")
            return

        # ── Path 0: deterministic stub ────────────────────────────────────
        # Resolved BEFORE the token is read and before any upstream request is
        # built, so a stub request cannot reach a credential even by accident.
        # Same module the HF Space proxy uses: one implementation, so a rig
        # that passes locally is testing the same responder that runs in
        # production rather than a look-alike.
        if _STUB_AVAILABLE and STUB_ENABLED:
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, ValueError, TypeError):
                payload = None
            if isinstance(payload, dict) and is_stub_model(payload.get("model")):
                mode, _arg = parse_stub_mode(payload.get("model"))
                _LOG.info("stub request: mode=%s bytes=%d", mode, len(body))
                # Shared clamp — see _stub_model.stub_delay_ms().
                delay_ms = stub_delay_ms(payload.get("model"))
                if delay_ms:
                    time.sleep(delay_ms / 1000.0)
                if payload.get("stream"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Stub-Model", "true")
                    self._send_cors_headers()
                    self.end_headers()
                    for frame in stub_sse_frames(
                        payload.get("model"), payload, self.headers
                    ):
                        if not self._write_bytes(frame.encode(), flush=True):
                            return
                    return
                status, doc = stub_payload(
                    payload.get("model"),
                    payload,
                    self.headers,
                    created=int(time.time()),
                )
                out = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Stub-Model", "true")
                self._send_cors_headers()
                self.end_headers()
                self._write_bytes(out)
                return

        if not _STUB_AVAILABLE:
            self._write_error(500, "Server contract module is unavailable.")
            return
        try:
            chat_req = parse_chat_request(body, allowed_models=ALLOWED_MODELS)
        except ChatContractError as exc:
            self._write_error(400, str(exc))
            return
        body = encode_upstream_payload(chat_req)
        model: str = chat_req.model
        # router.huggingface.co uses a flat endpoint; the model is selected
        # via the server-constructed request body, not the URL path.
        url: str = f"{HF_BASE}/v1/chat/completions"

        _LOG.info("Upstream request started.")

        try:
            with httpx.stream(
                "POST",
                url,
                content=body,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=TIMEOUT,
                follow_redirects=False,
            ) as resp:
                status_code = resp.status_code
                content_type = resp.headers.get("content-type", "application/json")
                content = _read_upstream_limited(resp)
        except _UpstreamResponseTooLarge:
            _LOG.warning(
                "AI dev proxy [UPSTREAM_RESPONSE_TOO_LARGE]: response aborted at safety limit."
            )
            self._write_error(502, "Upstream response exceeded proxy safety limit.")
            return
        except httpx.TimeoutException:
            _LOG.warning("Upstream timed out after %ds.", TIMEOUT)
            self._write_error(
                504,
                f"Upstream timed out after {TIMEOUT}s.  "
                "Try increasing PROXY_TIMEOUT or use a smaller model.",
            )
            return
        except httpx.RequestError as exc:
            _LOG.error(
                "AI dev proxy [UPSTREAM_REQUEST_ERROR]: upstream request failed (%s)",
                type(exc).__name__,
            )
            self._write_error(502, "Failed to reach upstream.")
            return

        _LOG.info("← %d", status_code)

        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self._send_cors_headers()
        self.end_headers()
        self._write_bytes(content)

    # ─── Private helpers ───────────────────────────────────────────────────

    def _send_cors_headers(self) -> None:
        """Emit CORS headers on the current response."""
        for key, value in _build_cors_headers(self.headers.get("Origin", "")).items():
            self.send_header(key, value)

    def _write_error(
        self, status: int, message: str, *, include_cors: bool = True
    ) -> None:
        """Write a JSON error response; denied origins receive no ACAO header."""
        payload = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if include_cors:
            self._send_cors_headers()
        self.end_headers()
        self._write_bytes(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        """
        Override to route server logs through the :mod:`logging` module.

        Parameters
        ----------
        fmt : str
            Format string (standard ``%``-style).
        *args : object
            Arguments for the format string.
        """
        _LOG.debug(fmt, *args)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    """
    Start the development proxy server.

    Binds to ``127.0.0.1:{PORT}`` (loopback only — not reachable from the
    public internet).  Press ``Ctrl+C`` to stop.
    """
    _LOG.info("Listening on http://localhost:%d", PORT)
    _LOG.info(
        "Forwarding to: %s/v1/chat/completions  (model via request body)", HF_BASE
    )
    _LOG.info("Default model: %s", DEFAULT_MODEL)
    _LOG.info("HF_TOKEN configured: %s", bool(HF_TOKEN))
    _LOG.info("Timeout:       %ds", TIMEOUT)
    _LOG.info("Press Ctrl+C to stop.")

    server = HTTPServer(("127.0.0.1", PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _LOG.info("Stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
