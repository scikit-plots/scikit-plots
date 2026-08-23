# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for generating summaries with an OpenAI-compatible API."""

from __future__ import annotations

import hashlib
import ipaddress
import os
from typing import Optional  # ruff: ignore[unused-import]
from urllib.parse import urlparse

from sphinx.errors import ExtensionError

from .version import __version__

DEFAULT_MODEL = ""
DEFAULT_MODEL_ENV = "OPENAI_MODEL"
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_REASONING_EFFORT_ENV = "OPENAI_REASONING_EFFORT"
SYSTEM_PROMPT = "Keep responses concise and focused, avoiding unnecessary elaboration or additional context unless explicitly requested. Do not use bullet points, lists, or nested structures unless specifically asked. If a response requires further detail, prioritize the most relevant information and conclude promptly. Avoid apologies or mentions of limitations; simply deliver the most direct and straightforward answer."
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
SUMMARY_PROMPT_VERSION = 2
SUMMARY_PROMPT = "Respond only with a concise one-sentence summary of the above."
SUMMARY_TEMPERATURE = 0


class MissingGenerationDependenciesError(ExtensionError):
    """Raised when the optional provider dependency is unavailable."""


class MalformedSummaryResponseError(ExtensionError):
    """Raised when a provider response has no usable summary."""


class InsecureEndpointError(ExtensionError):
    """Raised before credentials could be sent over insecure transport."""


def _missing_generation_dependencies(
    error: ImportError,
) -> MissingGenerationDependenciesError:
    """Return a helpful error when the optional generation extra is absent."""
    return MissingGenerationDependenciesError(
        "LLM summarization requires the optional generation dependencies. "
        "Install them with 'pip install sphinx-llm[gen]'.",
        error,
        "sphinx-llm",
    )


def summary_fingerprint(
    text: str,
    model: str,
    *,
    base_url: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> str:
    """
    Return a stable cache key for text and every generation setting.

    Parameters
    ----------
    text : str
        Document text the summary was generated from.
    model : str
        Model identifier used for generation.
    base_url : str, optional
        OpenAI-compatible endpoint the request was sent to.
    api_key_env : str, optional
        **Name of the environment variable** holding the credential, for example
        ``"OPENAI_API_KEY"``. Not the credential itself; see Notes.
    reasoning_effort : str, optional
        Reasoning-effort setting used for generation.

    Returns
    -------
    str
        Hex SHA-256 digest identifying this text/settings combination.

    See Also
    --------
    MarkdownGenerator._summary_fingerprint : the current fingerprint, which
        hashes a settings mapping and is used by ``generate_page_summary``.

    Notes
    -----
    **This is the legacy cache key and its formula is frozen.**
    ``docref.py`` evaluates it inside a set named ``legacy_hashes``, alongside an
    MD5 digest, to recognise cache entries written by earlier versions. Changing
    which fields it hashes, or their order or separator, would not "improve" the
    key — it would stop the function recognising the entries it exists to
    recognise. New fields belong in the current fingerprint instead.

    **On the CodeQL alert** (``py/weak-sensitive-data-hashing``): the query
    matches the identifier ``api_key_env`` with its sensitive-name heuristic and
    reports SHA-256 as unsuitable for password hashing. Neither half applies.

    ``api_key_env`` carries an environment variable *name* --
    ``DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"`` -- not a secret. The credential
    itself is read separately as ``configured_api_key = os.environ.get(...)`` in
    :func:`generate_summary` and never reaches this function; there is no
    dataflow from it to any hash.

    And this is a cache key, not stored authentication material. The query's own
    guidance names SHA-2 as correct outside password hashing. The slow, salted
    alternatives it suggests -- Argon2, scrypt, bcrypt, PBKDF2 -- salt per call
    by design, which would make the digest non-deterministic and every lookup a
    miss.

    The alert is suppressed at the call site rather than worked around, because
    every available workaround would either break the frozen formula or rename a
    public keyword argument to satisfy a heuristic.

    Examples
    --------
    >>> summary_fingerprint("hello world", "gpt-4o")[:16]
    '45548d26938f8ec1'
    """
    # codeql[py/weak-sensitive-data-hashing]: `api_key_env` is the *name* of an
    # environment variable, not a credential, and this is a deterministic cache
    # key rather than stored authentication material. See Notes above.
    return hashlib.sha256(
        (
            f"{SUMMARY_PROMPT_VERSION}\0{model}\0{base_url}\0{api_key_env}"
            f"\0{reasoning_effort}\0{text}"
        ).encode()
    ).hexdigest()


def _extract_summary(response: object) -> str:
    """Validate and extract summary text from a chat completion response."""
    try:
        summary = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        summary = None
    if not isinstance(summary, str) or not summary.strip():
        raise MalformedSummaryResponseError(
            "The OpenAI-compatible endpoint returned a malformed or empty summary"
        )
    return summary.strip()


def _is_loopback_url(url: str) -> bool:
    """Return whether a URL targets a loopback-only hostname or address."""
    hostname = urlparse(url).hostname
    if not hostname:
        return False
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def summarize_text(
    text: str,
    model: str = DEFAULT_MODEL,
    *,
    base_url: str = "",
    api_key_env: str = DEFAULT_API_KEY_ENV,
    reasoning_effort: str | None = None,
    timeout: float | None = None,
    use_environment_defaults: bool = True,
    allow_insecure_auth: bool = False,
) -> str:
    """Generate a concise summary using an OpenAI-compatible chat endpoint."""
    if use_environment_defaults:
        model = model or os.environ.get(DEFAULT_MODEL_ENV, "")
    if not isinstance(model, str) or not model.strip():
        raise ExtensionError(
            "No summary model is configured. Set 'model' in sphinx_llm_options "
            f"or set {DEFAULT_MODEL_ENV}."
        )

    try:
        from openai import OpenAI  # ruff: ignore[import-outside-top-level]
    except ImportError as error:
        raise _missing_generation_dependencies(error) from error

    effective_base_url = base_url
    if use_environment_defaults:
        effective_base_url = effective_base_url or os.environ.get("OPENAI_BASE_URL", "")
    configured_api_key = os.environ.get(api_key_env) if api_key_env else None
    if (
        configured_api_key
        and urlparse(effective_base_url).scheme.lower() == "http"
        and not _is_loopback_url(effective_base_url)
        and not allow_insecure_auth
    ):
        raise InsecureEndpointError(
            "Refusing to send an API key to a non-loopback endpoint over plain HTTP; "
            "use HTTPS, a loopback URL, or explicitly set allow_insecure_auth=True"
        )
    # The OpenAI client requires a non-empty value even when the endpoint does
    # not authenticate requests.
    api_key = configured_api_key or "not-used"
    if reasoning_effort is None:
        if use_environment_defaults:
            reasoning_effort = os.environ.get(
                DEFAULT_REASONING_EFFORT_ENV, DEFAULT_REASONING_EFFORT
            )
        else:
            reasoning_effort = DEFAULT_REASONING_EFFORT

    client_options = {"api_key": api_key}
    if effective_base_url:
        client_options["base_url"] = effective_base_url
    if timeout is not None:
        client_options["timeout"] = timeout
    client = OpenAI(**client_options)
    completion_options = {
        "model": model,
        "temperature": SUMMARY_TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": text + "\n\n" + SUMMARY_PROMPT,
            },
        ],
    }
    if reasoning_effort:
        # extra_body retains compatibility with openai 1.0, while forwarding
        # the standard field understood by newer OpenAI-compatible endpoints.
        completion_options["extra_body"] = {"reasoning_effort": reasoning_effort}
    response = client.chat.completions.create(**completion_options)
    return _extract_summary(response)


def setup(app):
    """Register configuration shared by all summary consumers."""
    app.add_config_value("llms_txt_summary_enabled", False, "env")
    app.add_config_value("llms_txt_summary_provider", "openai-compatible", "env")
    app.add_config_value("llms_txt_summary_model", "", "env")
    app.add_config_value("llms_txt_summary_base_url", "", "env")
    app.add_config_value("llms_txt_summary_api_key_env", DEFAULT_API_KEY_ENV, "env")
    app.add_config_value("llms_txt_summary_allow_insecure_auth", False, "env")
    app.add_config_value("llms_txt_summary_max_input_chars", 12_000, "env")
    app.add_config_value("llms_txt_summary_timeout", 60, "env")
    app.add_config_value("llms_txt_summary_cache_path", "", "env")

    return {
        "version": __version__,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
