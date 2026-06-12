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

Forwards ``POST /v1/chat/completions`` requests from the browser to the
HuggingFace Serverless Inference API with :data:`HF_TOKEN` injected
server-side.

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
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import NoReturn

import httpx

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

#: Local port the proxy listens on.
PORT: int = int(os.environ.get("DEV_PROXY_PORT", "8787"))

#: Upstream read timeout in seconds.
#: A 20B model on HF Serverless API can take 30-90 seconds to respond.
TIMEOUT: int = int(os.environ.get("PROXY_TIMEOUT", "120"))


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


def _token_fragment(token: str) -> str:
    """Return a safely-truncated token string for log output."""
    if not token or len(token) < 12:  # noqa: PLR2004
        return "<not-set>"
    return f"{token[:8]}...{token[-4:]}"


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


def _build_cors_headers() -> dict[str, str]:
    """
    Return standard CORS response headers.

    Returns
    -------
    dict[str, str]
        CORS headers for cross-origin browser acceptance.
    """
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


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
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

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
        body = json.dumps(
            {
                "status": "ok",
                "service": "sphinx-ai-assistant dev proxy (Path E)",
                "endpoint": f"POST http://localhost:{PORT}/v1/chat/completions",
                "default_model": DEFAULT_MODEL,
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        """
        Forward ``POST`` body to HuggingFace and write the response back.

        Both ``/v1/chat/completions`` and ``/`` are accepted so that
        ``conf.py`` endpoints with or without the path suffix both work.
        """
        # Read body — Content-Length required; default to 0 if absent.
        try:
            length: int = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            length = 0
        body: bytes = self.rfile.read(length)

        model: str = _parse_model(body)
        # router.huggingface.co uses a flat endpoint; the model is selected
        # via the request body, not the URL path.
        url: str = f"{HF_BASE}/v1/chat/completions"

        _LOG.info("→ POST %s", url)

        try:
            resp = httpx.post(
                url,
                content=body,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                timeout=TIMEOUT,
            )
        except httpx.TimeoutException:
            _LOG.warning("Upstream timed out after %ds.", TIMEOUT)
            self._write_error(
                504,
                f"Upstream timed out after {TIMEOUT}s.  "
                "Try increasing PROXY_TIMEOUT or use a smaller model.",
            )
            return
        except httpx.RequestError as exc:
            _LOG.error("Failed to reach upstream: %s", exc)
            self._write_error(502, f"Failed to reach upstream: {exc}")
            return

        _LOG.info("← %d", resp.status_code)

        self.send_response(resp.status_code)
        self.send_header(
            "Content-Type",
            resp.headers.get("content-type", "application/json"),
        )
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(resp.content)

    # ─── Private helpers ───────────────────────────────────────────────────

    def _send_cors_headers(self) -> None:
        """Emit CORS headers on the current response."""
        for key, value in _build_cors_headers().items():
            self.send_header(key, value)

    def _write_error(self, status: int, message: str) -> None:
        """Write a JSON error response with CORS headers."""
        payload = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(payload)

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
    _LOG.info("HF_TOKEN:      %s  (truncated for safety)", _token_fragment(HF_TOKEN))
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
