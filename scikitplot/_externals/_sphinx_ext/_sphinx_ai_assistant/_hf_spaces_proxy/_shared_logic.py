# _shared_logic.py  v6.0.0
#
# Single source of truth for shared constants, pure helper functions, and
# type aliases used by the deployed proxy (_hf_spaces_proxy/app.py) and the
# local development proxy (dev_proxy.py).
#
# Import discipline
# -----------------
# Only the Python standard library is imported here.  httpx, fastapi, and
# torch are NOT imported so this module can be sourced by stdlib-only tools
# (dev_proxy) and tested in isolation without any network or GPU environment.
#
# Routing paths (v6.0.0)
# ----------------------
# Three ordered routing paths — each with its own configurable read timeout:
#
#   Path 1 — BACKEND_URL set (explicit override)
#     Forward to BACKEND_URL.  HF_TOKEN injected when also set.
#     Read timeout: proxy_timeout kwarg (env: PROXY_TIMEOUT, default 600 s).
#
#   Path 2 — Model namespace in HF_SPACES_MODEL_NAMESPACES
#     Model owner (e.g. "scikit-plots") matches a custom namespace.
#     Forward to HF_SPACES_MODEL_URL (the ai-model HF Space, CPU inference).
#     These models have no HF Inference Provider → direct HF API returns 404/503.
#     Read timeout: path2_read_timeout kwarg (env: PATH2_TIMEOUT, default 600 s).
#     CPU inference on a 7B model takes 4-5 minutes; 600 s gives safe headroom.
#
#   Path 3 — Standard HF Inference API (default)
#     Model has a registered HF Inference Provider (openai/*, Qwen/*, etc.).
#     Forward to HF_BASE/{model}/v1/chat/completions with HF_TOKEN.
#     Read timeout: path3_read_timeout kwarg (env: PATH3_TIMEOUT, default 120 s).
#     HF Serverless API (GPU-backed) normally responds within 30-90 s.
#
# Breaking changes v4.0.0 → v5.0.0
# ----------------------------------
# + DEFAULT_PROXY_TIMEOUT raised from 120 s to 600 s.
#   Root cause: 120 s was shorter than the 4-5 min CPU inference on the
#   ai-model HF Space, causing every request to return a network error.
# + DEFAULT_PATH2_READ_TIMEOUT added (600 s) — ai-model space per-path timeout.
# + DEFAULT_PATH3_READ_TIMEOUT added (120 s) — HF API per-path timeout.
# + _resolve_upstream_url now accepts path2_read_timeout, path3_read_timeout,
#   and proxy_timeout keyword-only parameters.
# + _resolve_upstream_url return type changed from tuple[str, dict] to
#   tuple[str, dict, float] — the third element is the per-path read timeout.
#   Callers must unpack all three values.
# + load_proxy_env extended with path2_read_timeout and path3_read_timeout.
#
# Breaking changes v5.0.0 → v6.0.0
# ----------------------------------
# + DEFAULT_HF_BASE changed from ``https://api-inference.huggingface.co/models``
#   to ``https://router.huggingface.co``.
#   Root cause: api-inference.huggingface.co was DNS-unresolvable ([Errno -5]
#   EAI_NODATA / EAI_NONAME) from within HF Docker Spaces.
#   router.huggingface.co is the current HF Inference Providers endpoint and
#   resolves correctly in all deployment environments.
#   Callers who hard-code ``HF_BASE`` to the old hostname must migrate to
#   the new router URL.
#
# SPDX-License-Identifier: BSD-3-Clause
# Authors: The scikit-plots developers

"""
Shared utilities for the sphinx-ai-assistant proxy solutions.

This module provides pure, stateless helper functions and typed constants
that are common to all server-side proxy implementations.  It has **no**
runtime dependencies beyond the Python standard library.

Public API:

PROXY_VERSION : str
    Proxy release version string.
DEFAULT_HF_BASE : str
    HuggingFace Serverless Inference API base URL.
DEFAULT_MODEL : str
    Fallback model ID when the request body omits ``model``.
DEFAULT_PROXY_TIMEOUT : int
    Global upstream read timeout in seconds (Path 1 / backward-compat).
DEFAULT_PATH2_READ_TIMEOUT : float
    Per-path read timeout for Path 2 (ai-model space, CPU inference).
DEFAULT_PATH3_READ_TIMEOUT : float
    Per-path read timeout for Path 3 (HF Serverless Inference API).
DEFAULT_MAX_BODY_BYTES : int
    Maximum accepted request body size.
DEFAULT_HF_SPACES_MODEL_URL : str
    Default URL for the custom ai-model HF Space (Path 2).
DEFAULT_HF_SPACES_MODEL_NAMESPACES : tuple[str, ...]
    Default model owner namespaces routed to the model Space (Path 2).
_safe_int : callable
    Parse an integer environment variable with a safe fallback.
_parse_model : callable
    Extract the ``model`` field from a raw JSON request body.
_is_custom_model_namespace : callable
    Return True when a model's owner namespace is in the custom list.
_build_cors_headers : callable
    Return the CORS response-header mapping.
_token_log_fragment : callable
    Produce a safely-truncated token string for log output.
_resolve_upstream_url : callable
    Centralised three-path routing: choose upstream URL, auth headers,
    and per-path read timeout.
_validate_env : callable
    Fail-fast startup check with actionable error messages.
load_proxy_env : callable
    Read all proxy-relevant environment variables into a typed dict.

Notes
-----
**Developer note** — All functions are pure (no side effects, no I/O).
Tests can import this module without a running event loop or any network.
The proxy (FastAPI / asyncio) and dev_proxy (stdlib HTTPServer) both import
from here so that routing and CORS logic are *never* duplicated.

**Breaking change v5.0.0** — ``_resolve_upstream_url`` now returns a
3-tuple ``(url, headers, read_timeout_s: float)`` instead of the previous
2-tuple ``(url, headers)``.  All callers must unpack the third element or
the per-path timeout falls through to the old flat-timeout behaviour.

**Breaking change v6.0.0** — :data:`DEFAULT_HF_BASE` migrated from
``https://api-inference.huggingface.co/models`` to
``https://router.huggingface.co``.  The old hostname was DNS-unresolvable
([Errno -5] EAI_NONAME) from within HF Docker Spaces.  Deployments that
override ``HF_BASE`` to the legacy hostname must update their configuration.

**Security note** — :func:`_token_log_fragment` ensures the full API token
never appears in log output.  Never widen the exposed fragment beyond the
current 8+4 character window without reviewing log-aggregation policy first.

**Versioning note** — Bump :data:`PROXY_VERSION` on every breaking change so
deployed Spaces and log aggregators can correlate errors to a specific release.
"""

from __future__ import annotations

import json
import os
from typing import Any

__all__ = [
    # Constants
    "DEFAULT_HF_BASE",
    "DEFAULT_HF_SPACES_MODEL_NAMESPACES",
    "DEFAULT_HF_SPACES_MODEL_URL",
    "DEFAULT_MAX_BODY_BYTES",
    "DEFAULT_MODEL",
    "DEFAULT_PATH2_READ_TIMEOUT",
    "DEFAULT_PATH3_READ_TIMEOUT",
    "DEFAULT_PROXY_TIMEOUT",
    "PROXY_VERSION",
    # Helpers
    "_build_cors_headers",
    "_is_custom_model_namespace",
    "_parse_model",
    "_resolve_upstream_url",
    "_safe_int",
    "_token_log_fragment",
    "_validate_env",
    "load_proxy_env",
]


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

#: Proxy release version — bump on every breaking change.
PROXY_VERSION: str = "6.0.0"

#: HuggingFace Inference Providers router base URL (no trailing slash).
#: Only used for Path 3 (standard provider models) when ``BACKEND_URL`` is
#: empty and the model namespace is not in ``HF_SPACES_MODEL_NAMESPACES``.
#:
#: Migrated from ``https://api-inference.huggingface.co/models`` (v5.0.0) to
#: ``https://router.huggingface.co`` (v6.0.0).
#: Root cause: api-inference.huggingface.co was DNS-unresolvable ([Errno -5]
#: EAI_NODATA / EAI_NONAME) from within HF Docker Spaces; the router hostname
#: resolves correctly and is the current HF Inference Providers endpoint.
DEFAULT_HF_BASE: str = "https://router.huggingface.co"

#: Fallback model ID when the request body omits the ``model`` field.
#: Must have a registered HF Inference Provider for Path 3.
DEFAULT_MODEL: str = "scikit-plots/Qwen2.5-Coder-32B-Instruct"

#: Global upstream read timeout in seconds (used for Path 1 / backward compat).
#:
#: Raised from 120 s (v4.0.0) to 600 s (v5.0.0).
#:
#: Root cause of the increase: the ai-model HF Space runs a 7B model on CPU
#: basic hardware.  Cold-start inference (model loading + generation) takes
#: 4-5 minutes.  The 120 s ceiling caused every request to the ai-model Space
#: to return ``httpx.ReadTimeout``, which the browser reported as
#: "Sorry, something went wrong: network error".
DEFAULT_PROXY_TIMEOUT: int = 600

#: Per-path read timeout for Path 2 (ai-model HF Space, CPU inference).
#:
#: CPU inference on a 7B model takes 4-5 minutes.  600 s gives 1 minute of
#: additional headroom for cold-start model loading (~50 s tokenizer +
#: ~50 s model load + ~4.5 min generation on the first request).
DEFAULT_PATH2_READ_TIMEOUT: float = 600.0

#: Per-path read timeout for Path 3 (HF Serverless Inference API).
#:
#: The HF Serverless API runs inference on GPU hardware.  Most responses
#: arrive within 30-90 s.  120 s gives a comfortable margin.
DEFAULT_PATH3_READ_TIMEOUT: float = 120.0

#: Maximum accepted request body size in bytes (10 MiB).
#: Prevents memory exhaustion from maliciously oversized POST bodies.
DEFAULT_MAX_BODY_BYTES: int = 10 * 1024 * 1024  # 10 MiB

#: Default URL for the custom ai-model HF Space (Path 2).
#: Requests for models whose namespace is in ``DEFAULT_HF_SPACES_MODEL_NAMESPACES``
#: are forwarded here instead of the HF Serverless Inference API.
#: Overridable via the ``HF_SPACES_MODEL_URL`` environment variable.
DEFAULT_HF_SPACES_MODEL_URL: str = (
    "https://scikit-plots-ai-model.hf.space/v1/chat/completions"
)

#: Default model owner namespaces routed to :data:`DEFAULT_HF_SPACES_MODEL_URL`.
#: Models whose owner (the part before ``/``) matches any entry in this tuple
#: are routed to the ai-model Space (Path 2) rather than the HF API (Path 3).
#: Overridable via the ``HF_SPACES_MODEL_NAMESPACES`` environment variable.
DEFAULT_HF_SPACES_MODEL_NAMESPACES: tuple[str, ...] = ("scikit-plots",)


# ─────────────────────────────────────────────────────────────────────────────
# Pure helper functions
# ─────────────────────────────────────────────────────────────────────────────


def _safe_int(value: str | None, default: int) -> int:
    """
    Parse *value* as an integer, returning *default* on any failure.

    Parameters
    ----------
    value : str or None
        String to parse.  Typically the raw value of an environment variable
        (may be ``None`` when the variable is absent).
    default : int
        Returned when *value* is ``None``, empty, or cannot be converted.

    Returns
    -------
    int
        Parsed integer, or *default* on any ``ValueError`` / ``TypeError``.

    Notes
    -----
    **Developer note** — This function is intentionally never-raise.
    A misconfigured ``PROXY_TIMEOUT`` or ``MAX_BODY_BYTES`` must not prevent
    the proxy from starting — the safe default is better than a crash.

    Examples
    --------
    >>> _safe_int("120", 60)
    120
    >>> _safe_int("not-a-number", 60)
    60
    >>> _safe_int(None, 60)
    60
    >>> _safe_int("", 60)
    60
    """
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    """
    Parse *value* as a float, returning *default* on any failure.

    Parameters
    ----------
    value : str or None
        String to parse.  Typically the raw value of an environment variable.
    default : float
        Returned when *value* is ``None``, empty, or cannot be converted.

    Returns
    -------
    float
        Parsed float, or *default* on any ``ValueError`` / ``TypeError``.

    Notes
    -----
    **Developer note** — Like :func:`_safe_int`, this is intentionally
    never-raise.  A misconfigured ``PATH2_TIMEOUT`` or ``PATH3_TIMEOUT``
    must not crash the proxy at startup.

    Examples
    --------
    >>> _safe_float("600.0", 120.0)
    600.0
    >>> _safe_float("bad", 120.0)
    120.0
    >>> _safe_float(None, 120.0)
    120.0
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_model(body: bytes, default: str = DEFAULT_MODEL) -> str:
    """
    Extract the ``model`` field from a raw JSON request body.

    Parameters
    ----------
    body : bytes
        Raw HTTP request body forwarded from the browser.  Expected to be
        valid JSON but the function never raises on malformed input.
    default : str, optional
        Fallback model ID when the field is absent or the body cannot be
        decoded.  Defaults to :data:`DEFAULT_MODEL`.

    Returns
    -------
    str
        The ``model`` value from the body, or *default* if the field is
        absent, empty, or the body is not valid JSON.

    Notes
    -----
    **Developer note** — This function is intentionally never-raise.
    A malformed body must not crash the proxy; the upstream model backend
    will return a meaningful error that the browser can display.

    Examples
    --------
    >>> _parse_model(b'{"model": "Qwen/Qwen2.5-Coder-7B-Instruct"}')
    'Qwen/Qwen2.5-Coder-7B-Instruct'
    >>> _parse_model(b"{}")
    'scikit-plots/Qwen2.5-Coder-32B-Instruct'
    >>> _parse_model(b"not-json")
    'scikit-plots/Qwen2.5-Coder-32B-Instruct'
    >>> _parse_model(b'{"model": "  "}')
    'scikit-plots/Qwen2.5-Coder-32B-Instruct'
    """
    try:
        data: Any = json.loads(body)
        candidate = str(data.get("model", "")).strip()
        return candidate or default
    except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
        return default


def _is_custom_model_namespace(
    model: str,
    namespaces: tuple[str, ...] | list[str],
) -> bool:
    """
    Return ``True`` when the model owner namespace is in *namespaces*.

    The owner is the portion of the model ID before the first ``/``.
    An optional HF Router variant suffix (e.g. ``:fastest``) is stripped
    before comparison so ``"scikit-plots/Qwen2.5-Coder-7B-Instruct:fastest"``
    is correctly identified as belonging to the ``"scikit-plots"`` namespace.

    Parameters
    ----------
    model : str
        Model ID string, e.g. ``"scikit-plots/Qwen2.5-Coder-7B-Instruct"``
        or ``"openai/gpt-oss-20b:fastest"``.
    namespaces : tuple[str, ...] or list[str]
        Iterable of owner namespace strings to match against (case-insensitive).
        Typically :data:`DEFAULT_HF_SPACES_MODEL_NAMESPACES` or parsed from
        the ``HF_SPACES_MODEL_NAMESPACES`` environment variable.

    Returns
    -------
    bool
        ``True`` when the model owner is in *namespaces*, ``False`` otherwise.

    Notes
    -----
    **Developer note** — Comparison is case-insensitive and strips leading /
    trailing whitespace from both the model owner and each namespace entry.
    A model string without a ``/`` separator (i.e. no namespace component)
    always returns ``False``; such IDs are routed to Path 3 (HF Inference API).

    Examples
    --------
    >>> _is_custom_model_namespace(
    ...     "scikit-plots/Qwen2.5-Coder-7B-Instruct",
    ...     ("scikit-plots",),
    ... )
    True
    >>> _is_custom_model_namespace(
    ...     "scikit-plots/Qwen2.5-Coder-7B-Instruct:fastest",
    ...     ("scikit-plots",),
    ... )
    True
    >>> _is_custom_model_namespace("openai/gpt-oss-20b", ("scikit-plots",))
    False
    >>> _is_custom_model_namespace("no-slash-model", ("scikit-plots",))
    False
    """
    base = model.split(":", maxsplit=1)[0].strip()
    if not base or "/" not in base:
        return False
    owner = base.split("/", 1)[0].lower().strip()
    normalised = {ns.lower().strip() for ns in namespaces if ns.strip()}
    return owner in normalised


def _build_cors_headers(allowed_origin: str = "*") -> dict[str, str]:
    """
    Return the standard CORS response-header mapping.

    Parameters
    ----------
    allowed_origin : str, optional
        Value for the ``Access-Control-Allow-Origin`` header.
        Defaults to ``"*"`` (allow all origins).

    Returns
    -------
    dict[str, str]
        CORS response headers.

    Examples
    --------
    >>> headers = _build_cors_headers()
    >>> headers["Access-Control-Allow-Origin"]
    '*'
    """
    return {
        "Access-Control-Allow-Origin": allowed_origin,
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }


def _token_log_fragment(token: str) -> str:
    """
    Produce a safely-truncated token string for log output.

    Parameters
    ----------
    token : str
        A HuggingFace API token (typically ``hf_xxx...``).

    Returns
    -------
    str
        A truncated representation showing the first 8 and last 4
        characters separated by ``...``.  Returns ``"<not-set>"`` when
        *token* is empty or shorter than 12 characters.

    Notes
    -----
    **Security note** — Never widen the exposed fragment beyond 8+4
    characters without a log-aggregation security review.

    Examples
    --------
    >>> _token_log_fragment("hf_abcdefghij1234")
    'hf_abcde...1234'
    >>> _token_log_fragment("")
    '<not-set>'
    """
    if not token or len(token) < 12:  # noqa: PLR2004
        return "<not-set>"
    return f"{token[:8]}...{token[-4:]}"


def _resolve_upstream_url(
    body: bytes,
    *,
    backend_url: str,
    hf_token: str,
    hf_base: str = DEFAULT_HF_BASE,
    default_model: str = DEFAULT_MODEL,
    hf_spaces_model_url: str = DEFAULT_HF_SPACES_MODEL_URL,
    hf_spaces_model_namespaces: tuple[str, ...] | list[str] = (
        DEFAULT_HF_SPACES_MODEL_NAMESPACES
    ),
    proxy_timeout: float = float(DEFAULT_PROXY_TIMEOUT),
    path2_read_timeout: float = DEFAULT_PATH2_READ_TIMEOUT,
    path3_read_timeout: float = DEFAULT_PATH3_READ_TIMEOUT,
) -> tuple[str, dict[str, str], float]:
    """
    Centralised three-path routing — choose upstream endpoint, auth headers,
    and per-path read timeout.

    Priority
    --------
    1. *backend_url* is non-empty → **Path 1**: explicit custom backend.
       Forward to *backend_url* (Docker Model Runner, Ollama, any backend).
       *hf_token* is injected only when it is also set.
       Read timeout: *proxy_timeout* (env ``PROXY_TIMEOUT``, default 600 s).

    2. Model namespace is in *hf_spaces_model_namespaces* → **Path 2**: HF model Space.
       Forward to *hf_spaces_model_url* (the ``scikit-plots/ai-model`` Space).
       CPU inference on a 7B model takes 4-5 minutes; *path2_read_timeout*
       (env ``PATH2_TIMEOUT``, default 600 s) prevents premature timeout.
       *hf_token* is injected when set (needed for private Spaces).

    3. Otherwise → **Path 3**: HF Serverless Inference API (default).
       Build ``{hf_base}/{model}/v1/chat/completions`` and inject *hf_token*
       (always required for the HF API).
       *path3_read_timeout* (env ``PATH3_TIMEOUT``, default 120 s) is
       appropriate for GPU-backed HF API inference.

    Parameters
    ----------
    body : bytes
        Raw JSON request body.  Used to extract the ``model`` field for
        Paths 2 and 3.
    backend_url : str
        Value of the ``BACKEND_URL`` environment variable.  Non-empty string
        triggers Path 1; empty string means "proceed to Path 2 / 3".
    hf_token : str
        HuggingFace API token.  Required for Path 3; optional for Paths 1 and 2.
    hf_base : str, optional
        HF Serverless Inference API base URL (no trailing slash).
    default_model : str, optional
        Fallback model ID when the body omits the ``model`` field.
    hf_spaces_model_url : str, optional
        URL of the custom ai-model HF Space (Path 2 target).
    hf_spaces_model_namespaces : tuple[str, ...] or list[str], optional
        Model owner namespaces routed to *hf_spaces_model_url*.
    proxy_timeout : float, optional
        Read timeout (seconds) for Path 1.  Default: 600 s.
    path2_read_timeout : float, optional
        Read timeout (seconds) for Path 2 (ai-model Space).  Default: 600 s.
    path3_read_timeout : float, optional
        Read timeout (seconds) for Path 3 (HF Serverless API).  Default: 120 s.

    Returns
    -------
    url : str
        Fully-qualified upstream endpoint URL.
    headers : dict[str, str]
        HTTP headers for the upstream POST request.
    read_timeout_s : float
        Per-path read timeout in seconds.  Pass to ``httpx.Timeout(read=...)``.

    Notes
    -----
    **Breaking change v5.0.0** — Return type changed from
    ``tuple[str, dict]`` to ``tuple[str, dict, float]``.  All callers must
    unpack the third element.

    **Breaking change v6.0.0** — :data:`DEFAULT_HF_BASE` changed from
    ``https://api-inference.huggingface.co/models`` to
    ``https://router.huggingface.co``.  The old hostname was DNS-unresolvable
    from HF Docker Spaces ([Errno -5] EAI_NONAME).

    **Developer note** — All routing logic lives here.  To add a new backend
    type, add a new branch in this function.  Callers (``app.py``,
    ``dev_proxy.py``) remain unchanged when they already unpack 3 values.

    Examples
    --------
    Path 2 — scikit-plots namespace → ai-model Space:

    >>> url, hdrs, t = _resolve_upstream_url(
    ...     b'{"model":"scikit-plots/Qwen2.5-Coder-7B-Instruct","messages":[]}',
    ...     backend_url="",
    ...     hf_token="",
    ... )
    >>> "scikit-plots-ai-model.hf.space" in url
    True
    >>> t
    600.0

    Path 3 — standard HF Inference API:

    >>> url, hdrs, t = _resolve_upstream_url(
    ...     b'{"model":"openai/gpt-oss-20b","messages":[]}',
    ...     backend_url="",
    ...     hf_token="hf_test_token_abc123",
    ... )
    >>> "api-inference.huggingface.co" in url
    True
    >>> t
    120.0

    Path 1 — explicit BACKEND_URL:

    >>> url, hdrs, t = _resolve_upstream_url(
    ...     b"{}",
    ...     backend_url="https://my-model.hf.space/v1/chat/completions",
    ...     hf_token="",
    ... )
    >>> url
    'https://my-model.hf.space/v1/chat/completions'
    >>> t
    600.0
    """  # noqa: D205
    headers: dict[str, str] = {"Content-Type": "application/json"}

    # ── Path 1: explicit custom backend override ──────────────────────────────
    if backend_url:
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        return backend_url, headers, proxy_timeout

    # Extract model ID from request body (needed for Paths 2 and 3).
    model: str = _parse_model(body, default=default_model)

    # ── Path 2: custom model namespace → HF Spaces model backend ─────────────
    if hf_spaces_model_url and _is_custom_model_namespace(
        model, hf_spaces_model_namespaces
    ):
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        return hf_spaces_model_url, headers, path2_read_timeout

    # ── Path 3: HF Serverless Inference API (provider models) ─────────────────
    url = f"{hf_base.rstrip('/')}/{model}/v1/chat/completions"
    headers["Authorization"] = f"Bearer {hf_token}"
    return url, headers, path3_read_timeout


def _validate_env(
    backend_url: str,
    hf_token: str,
    hf_spaces_model_url: str = DEFAULT_HF_SPACES_MODEL_URL,
) -> None:
    """
    Validate the minimum required environment at proxy startup.

    At least one of the three routing paths must be viable:

    * **Path 1** — *backend_url* is non-empty.
    * **Path 2** — *hf_spaces_model_url* is non-empty (serves custom namespace models).
    * **Path 3** — *hf_token* is non-empty (HF Inference API for provider models).

    Parameters
    ----------
    backend_url : str
        Value of the ``BACKEND_URL`` environment variable (may be empty).
    hf_token : str
        Value of the ``HF_TOKEN`` environment variable (may be empty).
    hf_spaces_model_url : str, optional
        Value of the ``HF_SPACES_MODEL_URL`` environment variable.

    Raises
    ------
    RuntimeError
        When all three routing paths are disabled (all parameters are empty).

    Examples
    --------
    >>> _validate_env("https://my-model.hf.space/v1/chat/completions", "", "")
    >>> _validate_env("", "hf_mytoken", "")
    >>> _validate_env(
    ...     "", "", "https://scikit-plots-ai-model.hf.space/v1/chat/completions"
    ... )
    >>> import pytest
    >>> with pytest.raises(RuntimeError, match="no viable routing path"):
    ...     _validate_env("", "", "")
    """
    if not backend_url and not hf_token and not hf_spaces_model_url:
        raise RuntimeError(
            "Proxy configuration error: no viable routing path configured.\n\n"
            "Set at least ONE of the following in Space → Settings → Repository secrets:\n\n"
            "  Option 1 — HF Inference API (standard provider models):\n"
            "    HF_TOKEN      = hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
            "    DEFAULT_MODEL = openai/gpt-oss-20b\n\n"
            "  Option 2 — Custom ai-model Space (scikit-plots/* models):\n"
            "    HF_SPACES_MODEL_URL = "
            "https://scikit-plots-ai-model.hf.space/v1/chat/completions\n\n"
            "  Option 3 — Explicit custom backend (DMR, Ollama, or any backend):\n"
            "    BACKEND_URL = http://localhost:12434/engines/llama.cpp/v1/chat/completions\n\n"
            "See FREE_PROXY_SOLUTIONS.md for the full path decision tree."
        )


def load_proxy_env() -> dict[str, Any]:
    """
    Read all proxy-relevant environment variables and return a typed dict.

    Returns
    -------
    dict[str, Any]
        Keys and types:

        ``backend_url`` : str
        ``hf_token`` : str
        ``hf_base`` : str
        ``default_model`` : str
        ``hf_spaces_model_url`` : str
        ``hf_spaces_model_namespaces`` : tuple[str, ...]
        ``proxy_timeout`` : int
            Global / Path 1 read timeout (env ``PROXY_TIMEOUT``).
        ``path2_read_timeout`` : float
            Path 2 read timeout (env ``PATH2_TIMEOUT``).
        ``path3_read_timeout`` : float
            Path 3 read timeout (env ``PATH3_TIMEOUT``).
        ``max_body_bytes`` : int
        ``allowed_origins`` : str

    Examples
    --------
    >>> import os
    >>> os.environ["PROXY_TIMEOUT"] = "600"
    >>> cfg = load_proxy_env()
    >>> cfg["proxy_timeout"]
    600
    >>> os.environ["PATH2_TIMEOUT"] = "900"
    >>> cfg = load_proxy_env()
    >>> cfg["path2_read_timeout"]
    900.0
    """
    _raw_namespaces: str = os.environ.get(
        "HF_SPACES_MODEL_NAMESPACES",
        ",".join(DEFAULT_HF_SPACES_MODEL_NAMESPACES),
    )
    _parsed_namespaces: tuple[str, ...] = (
        tuple(ns.strip() for ns in _raw_namespaces.split(",") if ns.strip())
        or DEFAULT_HF_SPACES_MODEL_NAMESPACES
    )

    return {
        "backend_url": os.environ.get("BACKEND_URL", "").strip(),
        "hf_token": os.environ.get("HF_TOKEN", "").strip(),
        "hf_base": os.environ.get("HF_BASE", DEFAULT_HF_BASE).rstrip("/"),
        "default_model": (
            os.environ.get("DEFAULT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        ),
        "hf_spaces_model_url": (
            os.environ.get("HF_SPACES_MODEL_URL", DEFAULT_HF_SPACES_MODEL_URL).strip()
        ),
        "hf_spaces_model_namespaces": _parsed_namespaces,
        "proxy_timeout": _safe_int(
            os.environ.get("PROXY_TIMEOUT"),
            DEFAULT_PROXY_TIMEOUT,
        ),
        "path2_read_timeout": _safe_float(
            os.environ.get("PATH2_TIMEOUT"),
            DEFAULT_PATH2_READ_TIMEOUT,
        ),
        "path3_read_timeout": _safe_float(
            os.environ.get("PATH3_TIMEOUT"),
            DEFAULT_PATH3_READ_TIMEOUT,
        ),
        "max_body_bytes": _safe_int(
            os.environ.get("MAX_BODY_BYTES"),
            DEFAULT_MAX_BODY_BYTES,
        ),
        "allowed_origins": os.environ.get("ALLOWED_ORIGINS", "*").strip(),
    }
