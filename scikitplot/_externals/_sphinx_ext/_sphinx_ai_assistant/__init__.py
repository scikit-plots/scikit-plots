# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/__init__.py
#
# flake8: noqa: D213
#
# This module was copied and adapted from the sphinx-ai-assistant project.
# https://github.com/mlazag/sphinx-ai-assistant
#
# Authors: Mladen Zagorac
# SPDX-License-Identifier: MIT
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
A Sphinx extension that adds AI-assistant features to documentation pages,
including one-click Markdown export, AI chat deep-links, MCP tool
integration, and automated ``llms.txt`` generation. [1]_ [2]_ [3]_

The module has **two distinct layers**:

Core layer (Sphinx-free)
    Importable without Sphinx, BeautifulSoup, or markdownify.  All
    security helpers, the HTML→Markdown converter, the multi-process
    HTML walker, the standalone directory processor.

Sphinx layer
    ``setup()``, ``generate_markdown_files()``, ``generate_llms_txt()``,
    and ``add_ai_assistant_context()`` are Sphinx build-event hooks wired
    by :func:`setup`.  They delegate to the core layer internally.

All heavy optional dependencies (``sphinx``, ``bs4``, ``markdownify``,
``IPython``) are imported **lazily** — only when a feature is actually
invoked.  Importing this module at the top level is always safe and has
zero side effects.

Public API (standalone / non-Sphinx):

process_html_directory : callable
    Walk any HTML directory tree, convert pages to Markdown, optionally
    produce ``llms.txt``.  Works with Sphinx, MkDocs, Jekyll, plain HTML,
    or any other static-site generator.
generate_llms_txt_standalone : callable
    Write ``llms.txt`` from an existing set of ``.md`` files without
    requiring a Sphinx build.
html_to_markdown : callable
    Convert an HTML string to Markdown.

Public API (Sphinx extension):

setup : callable
    Sphinx extension entry point.

Notes
-----
**Developer note** — import discipline:

Every import of ``sphinx.*``, ``bs4``, ``markdownify``
lives *inside* the function or class body that needs it, guarded by a
try/except where appropriate.  Nothing is imported at module scope except
the standard library.  This keeps ``import time`` cost near zero and
avoids ``ImportError`` at load time when optional packages are absent.

.. seealso:
  * https://huggingface.co/spaces/scikit-plots/ai/tree/main

**Security notes**:

* :func:`_safe_json_for_script` escapes ``</script>`` sequences to prevent
  script-injection attacks when config is serialised into an HTML page.
* :func:`_is_path_within` prevents path-traversal attacks in the
  multi-process HTML walker.
* :func:`_validate_base_url` rejects non-HTTP(S) schemes in the base URL
  configuration value.
* :func:`_validate_position` rejects unknown widget-position strings.
* :func:`_validate_provider_url_template` rejects non-HTTP(S) schemes in
  AI-provider URL templates (``javascript:``, ``data:``, ``ftp:``, …).
* :func:`_validate_css_selector` rejects selectors containing HTML-injection
  characters (``<`` or ``>``).
* :func:`_validate_provider` checks every required field of a provider dict
  before it is serialised into a page or widget.
* Ollama ``api_base_url`` is validated to allow only ``http://localhost``
  or ``http://127.0.0.1`` origins, preventing exfiltration to remote hosts.

References
----------
.. [1] https://github.com/mlazag/sphinx-ai-assistant
.. [2] https://llmstxt.org/
.. [3] https://ollama.com/

Examples
--------
Sphinx Register Sphinx AI Assistant Extension in ``conf.py``:

.. code-block:: python

    extensions = [
        "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
    ]
    html_theme = "pydata_sphinx_theme"  # scikit-learn / NumPy style
    ai_assistant_enabled = True
    ai_assistant_theme_preset = "pydata_sphinx_theme"  # auto-selects CSS selectors
    ai_assistant_generate_markdown = True
    ai_assistant_generate_llms_txt = True
    html_baseurl = "https://docs.example.com"

Standalone (non-Sphinx):

.. code-block:: python

    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
        process_html_directory,
    )

    stats = process_html_directory(
        "/path/to/site/_site",
        theme_preset="jekyll",
        generate_llms=True,
        base_url="https://example.com",
    )
    print(stats)  # {"generated": 42, "skipped": 3, "errors": 0}
"""  # noqa: D205, D400

from __future__ import annotations

import datetime

# import multiprocessing
import importlib.util
import json
import os
import re
import sys
import time  # noqa: F401
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:  # pragma: no cover — only for type checkers, never at runtime
    from sphinx.application import Sphinx
    from sphinx.builders.html import StandaloneHTMLBuilder  # noqa: F401

# NOTE: ``bs4``, ``markdownify``, and ``IPython`` are intentionally NOT
# imported at module scope.  All callers import them locally when needed.

__all__ = [
    "_resolve_icon",
    "_write_progress_bar",
    "add_ai_assistant_context",
    "generate_llms_txt",
    "generate_llms_txt_standalone",
    "generate_markdown_files",
    "html_to_markdown",
    "html_to_markdown_converter",
    "plot_corpus_knowledge",
    "process_html_directory",
]

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_VERSION: str = "0.2.0"

# ---------------------------------------------------------------------------
# Module-level cached singletons (lazy, private)
# ---------------------------------------------------------------------------

_logger = None  # sphinx.util.logging.getLogger — initialised lazily
_SphinxMarkdownConverter = None  # markdownify subclass — built lazily

# ---------------------------------------------------------------------------
# Internal helpers — resolve max workers ProcessPoolExecutor
# ---------------------------------------------------------------------------


def _resolve_max_workers(cfg: Any) -> int | None:
    """Resolve the ``ai_assistant_max_workers`` config value to a safe integer.

    Parameters
    ----------
    cfg : Any
        Raw config value.  Accepted inputs:

        * ``None`` or ``"auto"`` — return ``None`` (let
          :class:`~concurrent.futures.ProcessPoolExecutor` auto-detect via
          :func:`os.cpu_count`).
        * Any positive integer — returned unchanged.
        * Zero or negative integer — raises :exc:`ValueError`.
        * Non-numeric string — raises :exc:`ValueError`.

    Returns
    -------
    int or None
        Positive integer, or ``None`` for auto-detection.

    Raises
    ------
    ValueError
        When *cfg* is zero, negative, or cannot be cast to ``int``.

    Notes
    -----
    **Developer note** — this is the single authoritative place that
    translates user-facing config into a value safe to pass to
    :class:`~concurrent.futures.ProcessPoolExecutor`.  All callers
    (``generate_markdown_files``, ``process_html_directory``) must use this
    function rather than consuming ``max_workers`` config directly.

    The Python stdlib raises ``ValueError: max_workers must be greater than 0``
    for non-positive integers, but only after acquiring resources.  This
    function rejects those values eagerly — before any worker is spawned —
    so error messages are clear and no process leak can occur.

    Examples
    --------
    >>> _resolve_max_workers(None)  # auto-detect
    >>> _resolve_max_workers("auto")  # auto-detect
    >>> _resolve_max_workers(4)
    4
    >>> _resolve_max_workers(0)  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ValueError: max_workers must be greater than 0; got 0
    """
    if cfg is None or cfg == "auto":
        return None  # let ProcessPoolExecutor use its own default (cpu_count)
    try:
        n = int(cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_workers must be a positive integer or None/'auto'; got {cfg!r}"
        ) from exc
    if n <= 0:
        raise ValueError(f"max_workers must be greater than 0; got {n}")
    return n


# ---------------------------------------------------------------------------
# Internal helpers — dependency detection
# ---------------------------------------------------------------------------


def _has_markdown_deps() -> bool:
    """Return *True* if ``beautifulsoup4`` and ``markdownify`` are importable.

    Uses :func:`importlib.util.find_spec` so the packages are *not* imported
    as a side-effect of the check.

    Returns
    -------
    bool
        ``True`` when both optional packages are available; ``False``
        otherwise.

    Examples
    --------
    >>> _has_markdown_deps()  # doctest: +SKIP
    True
    """
    return (
        importlib.util.find_spec("bs4") is not None
        and importlib.util.find_spec("markdownify") is not None
    )


def _has_ipython() -> bool:
    """Return *True* if ``IPython`` is importable.

    Returns
    -------
    bool
        ``True`` when IPython is available in the current environment.

    Examples
    --------
    >>> _has_ipython()  # doctest: +SKIP
    True
    """
    return importlib.util.find_spec("IPython") is not None


# ---------------------------------------------------------------------------
# Internal helpers — progress reporting
# ---------------------------------------------------------------------------


def _write_progress_bar(
    current: int,
    total: int,
    *,
    label: str = "Converting",
    bar_len: int = 50,
    stream: Any = None,
    part: int = 1,
    total_parts: int = 1,
) -> None:
    r"""Write an ASCII progress bar to *stream* (defaults to :data:`sys.stdout`).

    Parameters
    ----------
    current : int
        Number of items completed so far.
    total : int
        Total number of items.  When ``0`` or negative the function returns
        immediately to avoid division-by-zero.
    label : str, optional
        Short label printed before the bar.
    bar_len : int, optional
        Width of the bar in ``=``/``-`` characters.
    stream : file-like object or None, optional
        Output stream.  Defaults to :data:`sys.stdout`.
    part : int, optional
        Current part number (used only when *total_parts* > 1).
    total_parts : int, optional
        Total number of parts (used to prefix the bar with ``Part N/M``).

    Returns
    -------
    None

    Notes
    -----
    Uses a carriage-return ``\r`` so successive calls overwrite the same
    terminal line.  A newline is written when *current* >= *total* so the
    finalised bar is left visible in the scroll buffer.

    This function follows the ``sys.stdout.write`` / ``sys.stdout.flush``
    pattern — the same approach used by reporthook-style progress tracking
    (see :func:`urllib.request.urlretrieve`).  It is safe in non-TTY
    environments (CI, pipes) — the output simply contains ``\r`` characters.

    Examples
    --------
    >>> _write_progress_bar(3, 10, label="HTML→Markdown")  # doctest: +SKIP
    Converting: [===============-----------------------------------] 30.0% (3/10)
    """
    if total <= 0:
        return
    out = stream if stream is not None else sys.stdout
    percent = round(100.0 * current / total, 1)
    filled = int(bar_len * current // total)
    bar = "=" * filled + "-" * (bar_len - filled)
    if total_parts == 1:
        out.write(f"\r{label}: [{bar}] {percent}% ({current}/{total})")
    else:
        out.write(
            f"\rPart {part}/{total_parts} {label}: [{bar}] {percent}%"
            f" ({current}/{total})"
        )
    out.flush()
    if current >= total:
        out.write("\n")
        out.flush()


# ---------------------------------------------------------------------------
# Internal helpers — type coercion
# ---------------------------------------------------------------------------


def _coerce_to_list(
    val: Any,
    *,
    default: list[str] | None = None,
) -> list[str]:
    """Coerce ``str | list[str] | None`` to a plain ``list[str]``.

    Parameters
    ----------
    val : Any
        Value to coerce.  Accepted types:

        * ``None`` — returns *default* (or ``[]`` when *default* is ``None``).
        * ``str`` — wrapped in a single-element list.
        * Any iterable — each element converted with ``str()``.

    default : list of str or None, optional
        Fallback returned when *val* is ``None``.

    Returns
    -------
    list of str
        Always a newly created list; never a reference to *default*.

    Notes
    -----
    This helper centralises the ``str | list[str] | None`` coercion pattern
    that appears throughout the module's public API so that callers are never
    required to pre-wrap single strings.

    Examples
    --------
    >>> _coerce_to_list(None)
    []
    >>> _coerce_to_list("article")
    ['article']
    >>> _coerce_to_list(["article", "main"])
    ['article', 'main']
    >>> _coerce_to_list(None, default=["main"])
    ['main']
    """
    if val is None:
        return list(default) if default is not None else []
    if isinstance(val, str):
        return [val]
    return [str(v) for v in val]


# ---------------------------------------------------------------------------
# Internal helpers — lazy singletons
# ---------------------------------------------------------------------------


def _get_logger():
    """Return (and lazily initialise) the Sphinx module logger.

    Returns
    -------
    sphinx.util.logging.SphinxLoggerAdapter
        Module-level logger obtained from Sphinx's logging subsystem.

    Notes
    -----
    The logger is stored in the module-level ``_logger`` variable so that
    subsequent calls are O(1) dictionary lookups.
    """
    global _logger  # noqa: PLW0603
    if _logger is None:
        from sphinx.util import logging as _sphinx_logging  # noqa: PLC0415

        _logger = _sphinx_logging.getLogger(__name__)
    return _logger


def _build_converter_class():
    """Build and cache the :class:`SphinxMarkdownConverter` class lazily.

    Returns
    -------
    type
        A subclass of ``markdownify.MarkdownConverter`` customised for
        Sphinx-generated HTML.

    Notes
    -----
    The class is constructed once and cached in ``_SphinxMarkdownConverter``.
    The class definition lives here rather than at module scope so that
    ``markdownify`` is only imported when Markdown conversion is first
    requested.
    """
    global _SphinxMarkdownConverter  # noqa: PLW0603
    if _SphinxMarkdownConverter is not None:
        return _SphinxMarkdownConverter

    from markdownify import MarkdownConverter  # type: ignore[import]  # noqa: PLC0415

    class _SphinxMarkdownConverterImpl(MarkdownConverter):
        """Markdownify converter tuned for Sphinx HTML output.

        Handles Sphinx-specific structural elements such as highlighted code
        blocks, admonition ``div``s, and ``pre`` wrappers.

        Notes
        -----
        Method signatures follow the ``markdownify`` internal convention:
        ``convert_<tag>(el, text, convert_as_inline, **options)``.
        The ``text`` parameter contains the already-converted Markdown of
        the element's children; ``el`` is the original BeautifulSoup element.

        Do **not** mutate ``el`` (e.g. via ``decompose()``) — elements may
        be reused in later passes.
        """

        def convert_code(
            self,
            el: Any,
            text: str,
            convert_as_inline: bool = False,
            **options: Any,
        ) -> str:
            """Convert ``<code>`` elements, preserving language annotations.

            Parameters
            ----------
            el : bs4.element.Tag
                The ``<code>`` BeautifulSoup element.
            text : str
                Already-converted text content of the element's children.
            convert_as_inline : bool, optional
                When ``True`` the element should be rendered inline.
            **options : Any
                Forwarded to the parent converter.

            Returns
            -------
            str
                Fenced code block or inline backtick span.
            """
            content = text or (el.get_text() or "")
            classes: list[str] = list(el.get("class") or [])
            language = ""
            for cls in classes:
                if cls.startswith("highlight-"):
                    language = cls.replace("highlight-", "")
                    break
            if not convert_as_inline and language:
                return f"\n```{language}\n{content}```\n"
            return f"`{content}`" if content else ""

        def convert_div(
            self,
            el: Any,
            text: str,
            convert_as_inline: bool = False,
            **options: Any,
        ) -> str:
            """Convert ``<div>`` elements, with special handling for admonitions.

            Parameters
            ----------
            el : bs4.element.Tag
                The ``<div>`` BeautifulSoup element.
            text : str
                Already-converted Markdown of the element's children.
            convert_as_inline : bool, optional
                Ignored for block-level elements.
            **options : Any
                Forwarded to the parent converter.

            Returns
            -------
            str
                Block-quote style admonition, or the passthrough ``text``.

            Notes
            -----
            This method reads (but never mutates) the original element.
            """
            classes: list[str] = list(el.get("class") or [])
            if "admonition" in classes:
                title_el = el.find("p", class_="admonition-title")
                if title_el:
                    title_text = title_el.get_text(strip=True)
                    content = (text or "").strip()
                    if content.startswith(title_text):
                        content = content[len(title_text) :].strip()
                    return f"\n> **{title_text}**\n> {content}\n"
            return text or ""

        def convert_pre(
            self,
            el: Any,
            text: str,
            convert_as_inline: bool = False,
            **options: Any,
        ) -> str:
            """Convert ``<pre>`` blocks, delegating to :meth:`convert_code`.

            Parameters
            ----------
            el : bs4.element.Tag
                The ``<pre>`` BeautifulSoup element.
            text : str
                Already-converted Markdown of the element's children.
            convert_as_inline : bool, optional
                Ignored for block-level elements.
            **options : Any
                Forwarded to the parent converter.

            Returns
            -------
            str
                Fenced code block.
            """
            code_el = el.find("code")
            if code_el:
                return self.convert_code(code_el, code_el.get_text(), False)
            content = text or (el.get_text() or "")
            return f"\n```\n{content}\n```\n" if content else ""

    _SphinxMarkdownConverter = _SphinxMarkdownConverterImpl
    return _SphinxMarkdownConverter


# ---------------------------------------------------------------------------
# AI Provider registry
# ---------------------------------------------------------------------------

#: Complete default provider registry.
#:
#: Schema per provider (all keys are required unless marked optional):
#:
#: .. code-block:: python
#:
#:     {
#:         "enabled"         : bool,   # shown in widget if True
#:         "label"           : str,    # button text
#:         "description"     : str,    # tooltip / aria-label
#:         "icon"            : str,    # SVG filename in _static/
#:         "url_template"    : str,    # "{prompt}" placeholder; "" for API-only
#:         "prompt_template" : str,    # "{url}" and/or "{content}" placeholders
#:         "model"           : str,    # default model identifier
#:         "type"            : str,    # "web" | "local" | "api"
#:         "api_base_url"    : str,    # (optional) base URL for API-only providers
#:     }
#:
#: Providers with ``type == "local"`` (e.g. Ollama) default to
#: ``enabled = False`` because they require local infrastructure.
# ---------------------------------------------------------------------------
# Ollama recommended open models for fully local/offline AI assistance.
# ---------------------------------------------------------------------------

#: Recommended Ollama model identifiers grouped by capability tier.
#:
#: These models are all freely available via ``ollama pull <model>``.
#:
#: **Anthropic-API-compatible via Ollama** (usable with Claude Code's
#: ``--model`` flag or any Anthropic-compatible client):
#:   - ``qwen3:latest`` — Alibaba Qwen 3 (excellent code + reasoning)
#:   - ``qwen3:8b``, ``qwen3:14b``, ``qwen3:32b`` — size variants
#:   - ``glm4:latest`` — THUDM GLM-4 (strong multilingual)
#:   - ``llama3.3:latest`` — Meta Llama 3.3 70B (strong open model)
#:   - ``llama3.2:latest`` — Meta Llama 3.2 (lightweight, fast)
#:   - ``llama3.2:1b``, ``llama3.2:3b`` — ultra-light variants
#:
#: **Google Gemma 4 (free, Apache-2.0)**:
#:   - ``gemma3:latest`` — Google Gemma 3 (predecessor; Gemma 4 in progress)
#:   - Once Gemma 4 is on Ollama Hub: ``gemma4:latest``
#:   - HuggingFace: ``google/gemma-4-31B`` (pull via ``ollama run`` when available)
#:
#: **Other strong open models**:
#:   - ``mistral:latest`` — Mistral 7B (great for code)
#:   - ``codellama:latest`` — Meta Code Llama (code-specialised)
#:   - ``deepseek-r1:latest`` — DeepSeek R1 reasoning model (locally)
#:   - ``deepseek-coder-v2:latest`` — DeepSeek Coder V2
#:   - ``phi4:latest`` — Microsoft Phi-4 (small, smart)
#:   - ``phi4-mini:latest`` — Phi-4 Mini (very fast)
#:   - ``nomic-embed-text`` — embedding model (not for chat)
#:
#: **Free GPT-compatible models (OpenAI-compatible via Ollama)**:
#:   - Run with: ``OLLAMA_HOST=localhost ollama serve``
#:   - Access via: ``http://localhost:11434/v1`` (OpenAI-compatible endpoint)
#:
#: References
#: ----------
#: .. [1] https://ollama.com/library
#: .. [2] https://ai.google.dev/gemma/docs/core
#: .. [3] https://huggingface.co/google/gemma-4-31B
#: .. [4] https://ollama.com/blog/anthropic-compatible
_OLLAMA_RECOMMENDED_MODELS: tuple[str, ...] = (
    # Anthropic-API-compatible via Ollama
    "qwen3:latest",
    "qwen3:8b",
    "qwen3:14b",
    "qwen3:32b",
    "glm4:latest",
    "llama3.3:latest",
    "llama3.2:latest",
    "llama3.2:3b",
    "llama3.2:1b",
    # Google Gemma (open / Apache-2.0)
    "gemma3:latest",
    "gemma3:12b",
    "gemma3:4b",
    # gemma4:latest — pull once available on Ollama Hub
    # Code-specialised
    "codellama:latest",
    "deepseek-r1:latest",
    "deepseek-coder-v2:latest",
    "phi4:latest",
    "phi4-mini:latest",
    "mistral:latest",
)


# ---------------------------------------------------------------------------
# AI Provider registry — sorted: claude · gemini · chatgpt · ollama (local)
#   → custom (others A-Z) → mcp-backed
# ---------------------------------------------------------------------------

#: Complete default provider registry.
#:
#: **Ordering rationale**:
#:
#: 1. Enabled by default (most capable / popular): ``claude``, ``gemini``,
#:    ``chatgpt``
#: 2. Local / fully-offline: ``ollama`` (disabled by default — requires local
#:    Ollama server; supports Gemma 4, Qwen 3, Llama 3.3, DeepSeek R1, etc.)
#: 3. Custom stub: ``custom`` — user-defined endpoint / private LLM API
#: 4. Others (alphabetical): ``copilot``, ``deepseek``, ``groq``,
#:    ``huggingface``, ``mistral``, ``perplexity``, ``you``
#:
#: Schema per provider (all keys required unless annotated optional):
#:
#: .. code-block:: python
#:
#:     {
#:         "enabled"         : bool,   # shown & clickable when True
#:         "label"           : str,    # button text
#:         "description"     : str,    # tooltip / aria-label
#:         "icon"            : str,    # SVG filename in _static/
#:         "url_template"    : str,    # "{prompt}" placeholder; "" for API-only
#:         "prompt_template" : str,    # "{url}" and/or "{content}" placeholders
#:         "model"           : str,    # default model identifier
#:         "type"            : str,    # "web" | "local" | "api" | "custom"
#:         "api_base_url"    : str,    # (optional) base URL for local/API types
#:     }
#:
#: Setting ``type = "custom"`` marks a user-defined endpoint.  Combine with
#: ``api_base_url`` to point at any OpenAI-compatible local or internal server.
_DEFAULT_PROVIDERS: dict[str, dict[str, Any]] = {
    # ==================================================================
    # ── TIER 1: Enabled by default ────────────────────────────────────
    # ==================================================================
    # ----------------------------------------------------------------- ChatGPT
    "chatgpt": {
        "enabled": True,
        "label": "Ask ChatGPT",
        "description": "Ask OpenAI ChatGPT about this page",
        "icon": "chatgpt.svg",
        "url_template": "https://chatgpt.com/?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "gpt-4o",
        "type": "web",
        # "fetch_mode": "url",
    },
    # ------------------------------------------------------------------ Claude
    "claude": {
        # Enabled: users can click "Ask Claude" without any setup.
        "enabled": True,
        # Button label shown inside the AI-assistant panel.
        "label": "Ask Claude",
        # Tooltip / screen-reader accessible description.
        "description": "Ask Anthropic Claude about this page",
        # SVG icon filename in _static/.  Falls back to a base64 data URI
        # from ``_static/_PROVIDER_META`` if the file is absent on disk.
        "icon": "claude.svg",
        # URL template: {prompt} is substituted with the URL-encoded prompt.
        # Claude's ?q= parameter accepts the full prompt string directly.
        "url_template": "https://claude.ai/new?q={prompt}",
        # Prompt template: {url} → absolute URL of the page's .md companion;
        # {content} → raw Markdown text (can be large for long pages).
        # Using {url} keeps prompts short; Claude fetches and reads the page.
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"
            "I have questions about it."
        ),
        # Model identifier.  Forwarded to the widget for future API-mode use.
        "model": "claude-sonnet-4-6",
        # "web" opens a browser tab; no API key is required from the user.
        "type": "web",
        # "fetch_mode": "url",
    },
    # ------------------------------------------------------------------ Gemini
    "gemini": {
        "enabled": True,
        "label": "Ask Gemini",
        "description": "Ask Google Gemini about this page",
        "icon": "gemini.svg",
        "url_template": "https://gemini.google.com/app?q={prompt}",
        "prompt_template": (
            "Hi! Please review this documentation content: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "gemini-2.5-flash",
        "type": "web",
        # "fetch_mode": "url",
    },
    # ==================================================================
    # ── TIER 2: Local / fully offline ─────────────────────────────────
    # ==================================================================
    # ------------------------------------------------------------------ Ollama
    # Requires local Ollama server (https://ollama.com) + Open WebUI or
    # any compatible front-end.  Set ``enabled: True`` and optionally
    # override ``model`` with any of _OLLAMA_RECOMMENDED_MODELS.
    #
    # Ollama supports an Anthropic-compatible API endpoint — usable with
    # Claude Code (``--model ollama/qwen3:latest``) or any Anthropic SDK:
    #   ANTHROPIC_BASE_URL=http://localhost:11434 python -c "..."
    #
    # Recommended free offline models (``ollama pull <model>``):
    #   - qwen3:latest          — Alibaba Qwen 3 (code + reasoning)
    #   - llama3.3:latest       — Meta Llama 3.3 70B
    #   - llama3.2:latest       — Meta Llama 3.2 (lightweight)
    #   - gemma3:latest         — Google Gemma 3 (Apache-2.0, free)
    #   - deepseek-r1:latest    — DeepSeek R1 (reasoning)
    #   - phi4-mini:latest      — Microsoft Phi-4 Mini (very fast)
    #   - mistral:latest        — Mistral 7B
    #   - codellama:latest      — Meta Code Llama (code-specialised)
    #
    # Gemma 4 note: Google released Gemma 4 (31B) under Apache-2.0.
    #   Pull via HuggingFace: https://huggingface.co/google/gemma-4-31B
    #   Once on Ollama Hub: ``ollama pull gemma4:latest``
    "ollama": {
        "enabled": False,
        "label": "Ask Ollama (Local)",
        "description": (
            "Ask a locally running Ollama model — fully offline, "
            "privacy-preserving. Supports Gemma 4, Qwen 3, Llama 3.3, "
            "DeepSeek R1, Phi-4, Mistral and more."
        ),
        "icon": "ollama.svg",
        "url_template": "http://localhost:3000/?q={prompt}",
        "api_base_url": "http://localhost:11434",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "llama3.2:latest",
        "type": "local",
        # "fetch_mode": "url",
    },
    # ==================================================================
    # ── TIER 3: Custom / user-defined endpoint ─────────────────────────
    # ==================================================================
    # ------------------------------------------------------------------ Custom
    # Stub for any user-defined LLM endpoint (private server, internal
    # corporate LLM, OpenAI-compatible proxy, etc.).
    # Override ``api_base_url``, ``model``, and ``url_template`` in conf.py:
    #
    #   ai_assistant_providers = {
    #       **_DEFAULT_PROVIDERS,
    #       "custom": {
    #           **_DEFAULT_PROVIDERS["custom"],
    #           "enabled": True,
    #           "label": "Ask Internal AI",
    #           "api_base_url": "http://internal-llm.corp/v1",
    #           "model": "company-llm-v2",
    #       }
    #   }
    "custom": {
        "enabled": False,
        "label": "Ask Custom AI",
        "description": "Ask your custom or self-hosted AI endpoint",
        "icon": "custom.svg",
        "url_template": "",
        "api_base_url": "",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "",
        "type": "custom",
        # "fetch_mode": "url",
    },
    # ==================================================================
    # ── TIER 4: Others — alphabetical ─────────────────────────────────
    # ==================================================================
    # ----------------------------------------------------------------- Copilot
    "copilot": {
        "enabled": False,
        "label": "Ask Copilot",
        "description": "Ask Microsoft Copilot about this page",
        "icon": "copilot.svg",
        "url_template": "https://copilot.microsoft.com/?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "gpt-4o",
        "type": "web",
        # "fetch_mode": "url",
    },
    # --------------------------------------------------------------- DeepSeek
    # DeepSeek R1 and V3 are strong open models; also available via Ollama
    # locally (``ollama pull deepseek-r1:latest``).
    "deepseek": {
        "enabled": False,
        "label": "Ask DeepSeek",
        "description": "Ask DeepSeek AI about this page",
        "icon": "deepseek.svg",
        "url_template": "https://chat.deepseek.com/?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "deepseek-reasoner",
        "type": "web",
        # "fetch_mode": "url",
    },
    # -------------------------------------------------------------------- Groq
    # Groq provides extremely fast open-source LLM inference.
    "groq": {
        "enabled": False,
        "label": "Ask Groq",
        "description": "Ask Groq (fast LLM inference) about this page",
        "icon": "groq.svg",
        "url_template": "https://console.groq.com/playground?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "llama-3.3-70b-versatile",
        "type": "web",
        # "fetch_mode": "url",
    },
    # ----------------------------------------------------------- HuggingFace
    # HuggingFace Chat supports many open-source models (Llama, Mistral,
    # Qwen, Gemma, etc.) freely.  Gemma 4 (31B, Apache-2.0) is available at
    # https://huggingface.co/google/gemma-4-31B and via HuggingFace Chat.
    "huggingface": {
        "enabled": False,
        "label": "Ask HuggingFace",
        "description": (
            "Ask HuggingFace Chat about this page — supports Llama, Qwen, "
            "Gemma 4, Mistral and other open models for free"
        ),
        "icon": "huggingface.svg",
        "url_template": "https://huggingface.co/chat/?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "type": "web",
        # "fetch_mode": "url",
    },
    # ----------------------------------------------------------------- Mistral
    "mistral": {
        "enabled": False,
        "label": "Ask Mistral",
        "description": "Ask Mistral AI Le Chat about this page",
        "icon": "mistral.svg",
        "url_template": "https://chat.mistral.ai/chat?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "mistral-large-latest",
        "type": "web",
        # "fetch_mode": "url",
    },
    # --------------------------------------------------------------- Perplexity
    "perplexity": {
        "enabled": False,
        "label": "Ask Perplexity",
        "description": "Ask Perplexity AI about this page",
        "icon": "perplexity.svg",
        "url_template": "https://www.perplexity.ai/?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "sonar-pro",
        "type": "web",
        # "fetch_mode": "url",
    },
    # ----------------------------------------------------------------- You.com
    "you": {
        "enabled": False,
        "label": "Ask You.com",
        "description": "Ask You.com AI about this page",
        "icon": "you.svg",
        "url_template": "https://you.com/?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        "model": "default",
        "type": "web",
        # "fetch_mode": "url",
    },
}

# ---------------------------------------------------------------------------
# Default feature-flag registry
# ---------------------------------------------------------------------------

#: Full default feature-flag dict.
#:
#: **Why this exists as a module-level constant:**
#:
#: ``add_ai_assistant_context`` serialises
#: ``window.AI_ASSISTANT_CONFIG.features`` from
#: ``app.config.ai_assistant_features``.  When a project's ``conf.py``
#: sets only a *partial* dict (e.g. ``{"markdown_export": True, "ai_chat": True}``),
#: the missing keys are absent from the injected JSON.  The JS widget then
#: falls back to ``FEATURE_DEFAULTS``, where ``ai_panel = false``.  That
#: silently hides the AI-panel button even though the Python default for
#: ``ai_assistant_features`` includes ``"ai_panel": True``.
#:
#: The fix is to merge the user's partial dict *over* this constant in
#: ``add_ai_assistant_context`` so every key is always present in the
#: serialised output.  User values override defaults; defaults fill gaps.
#:
#: **Security / forward-compat note:** new feature flags must be added here
#: with a safe default (typically ``True`` for UI features that are on by
#: default, ``False`` for opt-in features like ``mcp_integration``).
_DEFAULT_FEATURES: dict[str, bool] = {
    "markdown_export": True,
    "view_markdown": True,
    "ai_chat": True,
    "mcp_integration": True,  # set False if no MCP tools are configured
    "theme_toggle": True,  # dark / light / system color-scheme toggle
    "pdf_export": True,  # Export as PDF button
    "ai_panel": True,  # Floating AI assistant chat panel
}

#: Required top-level keys for every provider dict.
_PROVIDER_REQUIRED_KEYS: tuple[str, ...] = (
    "enabled",
    "label",
    "description",
    "icon",
    "url_template",
    "prompt_template",
    "model",
    "type",
)

#: Accepted values for ``provider["type"]``.
_PROVIDER_TYPES: frozenset = frozenset({"web", "local", "api", "custom"})

#: Accepted values for the optional ``provider["fetch_mode"]`` field.
#:
#: Semantics:
#:
#: * ``"url"``     — the prompt contains the page URL and the LLM is
#:                   expected to fetch/read it autonomously.  Works reliably
#:                   for Claude and ChatGPT; use only when the provider
#:                   supports URL ingestion.
#: * ``"content"`` — the prompt contains pre-extracted Markdown content
#:                   instead of (or in addition to) the URL.  Required for
#:                   Gemini and any provider that does **not** reliably
#:                   ingest arbitrary external URLs from query params.
#: * ``"both"``    — the prompt includes both URL and content.
#: * ``"paste"``   — the prompt instructs the user to paste content manually.
#:
#: Providers that omit ``fetch_mode`` default to ``"url"`` for backward
#: compatibility.  The widget JS reads this field to decide whether to
#: inject ``{content}`` (Markdown) or ``{url}`` into the prompt template.
_PROVIDER_FETCH_MODES: frozenset = frozenset({"url", "content", "both", "paste"})

# ---------------------------------------------------------------------------
# Feedback rating scales — signed integers centred on zero
# ---------------------------------------------------------------------------

#: Built-in signed-integer scales for ``ai_assistant_panel_feedback_scale``.
#:
#: Why signed integers and not strings:
#:
#: The previous feedback contract emitted ``"positive" / "neutral" / "negative"``.
#: That is fine for a human dashboard but unusable as a training signal —
#: you cannot average strings, you cannot threshold strings, you cannot
#: subtract one rating from another to compute a delta.  Signed integers
#: centred on zero are the canonical Likert-style encoding:
#:
#: * Odd-length scales include the neutral midpoint at ``0``.
#: * Even-length scales drop the midpoint to force a polarised choice.
#: * The sequence is strictly monotonic (i < j ⇒ scale[i] < scale[j]).
#: * For every odd-length scale, ``sum(scale) == 0`` — this invariant is
#:   asserted in :func:`_resolve_feedback_scale`.
#:
#: The values are serialised to the browser as a parallel list aligned with
#: ``ai_assistant_panel_feedback_options``.  The JS widget renders the
#: numeric value as a hover tooltip + ``data-value`` attribute so the final
#: user is always shown what they are submitting.
#:
#: Examples
#: --------
#: >>> _FEEDBACK_SCALES["odd-3"]
#: (-1, 0, 1)
#: >>> _FEEDBACK_SCALES["even-2"]
#: (-1, 1)
#: >>> sum(_FEEDBACK_SCALES["odd-5"])
#: 0
_FEEDBACK_SCALES: dict[str, tuple[int, ...]] = {
    "even-2": (-1, 1),
    "odd-3": (-1, 0, 1),
    "even-4": (-2, -1, 1, 2),
    "odd-5": (-2, -1, 0, 1, 2),
    "even-6": (-3, -2, -1, 1, 2, 3),
    "odd-7": (-3, -2, -1, 0, 1, 2, 3),
    "even-8": (-4, -3, -2, -1, 1, 2, 3, 4),
    "odd-9": (-4, -3, -2, -1, 0, 1, 2, 3, 4),
    "even-10": (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5),
    "odd-11": (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5),
}

#: Accepted values for ``ai_assistant_panel_feedback_scale`` (case-insensitive).
_FEEDBACK_SCALE_NAMES: frozenset = frozenset({"auto", *_FEEDBACK_SCALES.keys()})


def _resolve_feedback_scale(
    scale_name: str,
    num_options: int,
) -> tuple[int, ...]:
    """Resolve the configured scale name to a tuple of signed integers.

    Parameters
    ----------
    scale_name : str
        One of :data:`_FEEDBACK_SCALE_NAMES`.  ``"auto"`` selects by
        *num_options*; anything else picks the matching entry of
        :data:`_FEEDBACK_SCALES` and re-shapes it to *num_options* if needed.
    num_options : int
        Number of emoji options actually configured in
        ``ai_assistant_panel_feedback_options``.  Must be ``>= 2``; the
        built-in named scales cover 2-10.  Values above 10 are accepted and
        produce a generated symmetric scale automatically.

    Returns
    -------
    tuple of int
        A tuple of length *num_options* of signed integers centred on zero.
        For odd *num_options* the midpoint is ``0`` and ``sum == 0``.
        For even *num_options* there is no midpoint.

    Raises
    ------
    ValueError
        If *scale_name* is not in :data:`_FEEDBACK_SCALE_NAMES` or if
        *num_options* is ``< 2``.

    Notes
    -----
    Deductive, deterministic — same input always produces the same output.
    No heuristics, no rounding, no implicit fallback.  For *num_options*
    outside the built-in registry (e.g. 11+) a symmetric range is generated:
    odd N -> ``(-k, ..., -1, 0, 1, ..., k)``;
    even N -> ``(-k, ..., -1, 1, ..., k)``  (no zero).

    Examples
    --------
    >>> _resolve_feedback_scale("odd-3", 3)
    (-1, 0, 1)
    >>> _resolve_feedback_scale("auto", 5)
    (-2, -1, 0, 1, 2)
    >>> _resolve_feedback_scale("auto", 2)
    (-1, 1)
    >>> _resolve_feedback_scale("auto", 10)
    (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5)
    >>> _resolve_feedback_scale("auto", 7)  # generated symmetric range
    (-3, -2, -1, 0, 1, 2, 3)
    """
    if not isinstance(num_options, int) or num_options < 2:  # noqa: PLR2004
        raise ValueError(
            f"feedback scale requires num_options >= 2; got {num_options!r}"
        )
    name = str(scale_name or "auto").lower()
    if name not in _FEEDBACK_SCALE_NAMES:
        raise ValueError(
            f"unknown feedback scale {scale_name!r}; "
            f"expected one of {sorted(_FEEDBACK_SCALE_NAMES)}"
        )

    if name == "auto":
        # Prefer the named registry entry when available so the JS widget
        # gets the canonical shape; otherwise generate a symmetric range.
        key = ("odd" if num_options % 2 else "even") + f"-{num_options}"
        if key in _FEEDBACK_SCALES:
            scale = _FEEDBACK_SCALES[key]
        else:
            scale = _generate_symmetric_scale(num_options)
    else:
        scale = _FEEDBACK_SCALES[name]

    # Reshape if the user picked a named scale whose length disagrees with
    # the number of options — generate a symmetric range matching *num_options*
    # rather than silently truncating or padding.  This makes mismatches
    # explicit and idempotent.
    if len(scale) != num_options:
        scale = _generate_symmetric_scale(num_options)

    # Invariant: odd-length scales sum to 0 (verified, deductive).
    if num_options % 2 == 1 and sum(scale) != 0:
        raise ValueError(f"internal: odd-length scale must sum to 0; got {scale!r}")
    return scale


def _generate_symmetric_scale(num_options: int) -> tuple[int, ...]:
    """Generate a symmetric signed-integer scale of length *num_options*.

    Parameters
    ----------
    num_options : int
        Must be ``>= 2``.

    Returns
    -------
    tuple of int
        Odd  N -> ``(-k, ..., -1, 0, 1, ..., k)`` where ``k = N // 2``
                  (length N, midpoint 0, sum 0).
        Even N -> ``(-k, ..., -1, 1, ..., k)`` where ``k = N // 2``
                  (length N, no midpoint, sum 0).
    """
    if num_options < 2:  # noqa: PLR2004
        raise ValueError(f"num_options must be >= 2; got {num_options!r}")
    half = num_options // 2
    if num_options % 2 == 1:
        # Odd: include 0.  e.g. N=3 -> (-1, 0, 1).
        return tuple(range(-half, half + 1))
    # Even: skip 0.  e.g. N=4 -> (-2, -1, 1, 2).
    return tuple(list(range(-half, 0)) + list(range(1, half + 1)))


# ---------------------------------------------------------------------------
# Phase B — Panel multi-model registry, Terms of Service, Share, Hamburger
# ---------------------------------------------------------------------------

#: Whitelist of accepted ``provider`` values in a panel-model entry.
#:
#: This is the **public-facing** provider name used for UI labelling and for
#: routing the feedback payload's ``model.provider`` field.  It is intentionally
#: a closed set — a typo at config time is an error, not a silent fallback.
#: Adding a new provider here is the only place that needs to change.
#:
#: Notes
#: -----
#: ``"stub"`` is intentionally included so dev-side panels with no proxy can
#: still expose a model picker and emit feedback with a clear "this was a stub
#: answer" attribution.  Production proxies should never advertise the stub
#: provider — its presence in a feedback payload is a strong signal to the
#: training pipeline that the row must be discarded.
_PANEL_MODEL_PROVIDERS: frozenset = frozenset(
    {
        "anthropic",
        "openai",
        "google",
        "mistral",
        "deepseek",
        "huggingface",  # HuggingFace Inference API (OpenAI-compat /v1/chat/completions)
        "ollama",
        "groq",
        "cerebras",  # Cerebras Inference (free tier available)
        "together",  # Together AI (free tier available)
        "fireworks",  # Fireworks AI (free tier available)
        "sambanova",  # SambaNova Cloud (free tier available)
        "cloudflare",  # Cloudflare Workers AI (free 10k tokens/day)
        "perplexity",
        "azure_openai",
        "custom",
        "stub",
    }
)

#: Provider → UI accent colour (hex, used by the JS badge renderer).
#: Kept in Python so the colour map is in one place; serialised into the
#: page config and consumed by ai-assistant.js ``_PROVIDER_COLORS``.
#: ``"custom"`` and ``"stub"`` intentionally have no entry — they render
#: with the neutral grey default.
_PROVIDER_COLORS: dict[str, str] = {
    "anthropic": "#c96442",  # Anthropic brand copper-orange
    "openai": "#74aa9c",  # OpenAI brand teal
    "google": "#4285f4",  # Google blue
    "mistral": "#ff7000",  # Mistral orange
    "deepseek": "#4d6bfe",  # DeepSeek indigo
    "huggingface": "#ff9d00",  # HuggingFace yellow-orange
    "ollama": "#222222",  # Ollama dark
    "groq": "#f55036",  # Groq red
    "cerebras": "#8c52ff",  # Cerebras purple  — mirrors JS _PROVIDER_COLORS_JS
    "together": "#4b5563",  # Together AI grey  — mirrors JS _PROVIDER_COLORS_JS
    "fireworks": "#ef4444",  # Fireworks red     — mirrors JS _PROVIDER_COLORS_JS
    "sambanova": "#e95b2e",  # SambaNova orange  — mirrors JS _PROVIDER_COLORS_JS
    "cloudflare": "#f38020",  # Cloudflare orange — mirrors JS _PROVIDER_COLORS_JS
    "perplexity": "#20b2aa",  # Perplexity teal
    "azure_openai": "#0078d4",  # Azure blue
}

#: Required top-level keys for every entry of ``ai_assistant_panel_api_models``.
#:
#: Why each is required (deductive justification):
#:
#: * ``id`` — stable, unique key persisted in ``sessionStorage`` so the user's
#:   choice survives navigation; also referenced in the feedback payload.
#: * ``provider`` — closed-set whitelist controls UI labelling and ensures the
#:   feedback ``model.provider`` is meaningful for downstream grouping.
#: * ``model`` — the wire model name the proxy must forward to the upstream API.
#: * ``endpoint`` — the proxy URL (http/https or site-relative); the JS hits
#:   this directly, never the upstream provider.
_PANEL_MODEL_REQUIRED_KEYS: tuple[str, ...] = (
    "id",
    "provider",
    "model",
    "endpoint",
)

#: Optional keys recognised on a panel-model entry (passed through verbatim
#: to the browser when present, ignored when absent).  Listed here so a
#: future maintainer has one source of truth.
_PANEL_MODEL_OPTIONAL_KEYS: tuple[str, ...] = (
    "label",  # picker UI text; defaults to id
    "description",  # one-line caption in the sheet
    "info_url",  # public model homepage (e.g. anthropic.com/claude)
    "default",  # bool; at most one entry may set True
    "icon",  # SVG filename in _static/ or absolute URI
)


#: Default share-target registry.  Each entry mirrors the provider schema
#: (``label`` / ``url_template`` / ``icon``) so the existing
#: :func:`_validate_provider_url_template` guard applies unchanged.
#:
#: The placeholder ``{url}`` is the *current* page URL (URL-encoded), and
#: ``{title}`` is ``document.title`` (URL-encoded) — both injected client-side.
#: ``"copy_link"`` is special: an empty ``url_template`` is treated by the JS
#: as "copy the page URL to the clipboard" rather than as an invalid entry.
_DEFAULT_SHARE_TARGETS: dict[str, dict[str, str]] = {
    "copy_link": {
        "label": "Copy link",
        "url_template": "",  # special — handled client-side
        "icon": "copy-to-clipboard.svg",
    },
    "email": {
        "label": "Email",
        "url_template": "mailto:?subject={title}&body={url}",
        "icon": "share.svg",
    },
    "x": {
        "label": "X (Twitter)",
        "url_template": "https://twitter.com/intent/tweet?text={title}&url={url}",
        "icon": "share.svg",
    },
    "linkedin": {
        "label": "LinkedIn",
        "url_template": "https://www.linkedin.com/sharing/share-offsite/?url={url}",
        "icon": "share.svg",
    },
    "reddit": {
        "label": "Reddit",
        "url_template": "https://www.reddit.com/submit?url={url}&title={title}",
        "icon": "share.svg",
    },
    "hacker_news": {
        "label": "Hacker News",
        "url_template": "https://news.ycombinator.com/submitlink?u={url}&t={title}",
        "icon": "share.svg",
    },
}


def _normalize_panel_models(raw: Any) -> list[dict[str, Any]]:
    """Coerce the user-supplied ``ai_assistant_panel_api_models`` value to a list.

    Parameters
    ----------
    raw : Any
        Accepted shapes (forward-compatible):

        * ``None`` or ``""`` or ``[]`` — return ``[]``.  The widget then falls
          back to the legacy single-string ``ai_assistant_panel_api_model``.
        * ``str`` — treated as a single-id list, e.g. ``"gpt-4o"`` →
          ``[{"id": "gpt-4o", "model": "gpt-4o", "provider": "custom",
          "endpoint": ""}]``.  ``endpoint`` empty triggers the same actionable
          error in the JS that the single-model path raises today.
        * ``list[str]`` — each string is normalised to the dict shape above.
        * ``list[dict]`` — passed through (the validator runs next).

    Returns
    -------
    list of dict
        Normalised list (never ``None``).  Entries may still fail
        :func:`_validate_panel_model`; this function only handles *shape*,
        not *correctness*.

    Notes
    -----
    Deductive, deterministic.  Bare strings get ``provider="custom"`` so the
    closed-set whitelist is never silently widened.  The doc-side example in
    ``_example_conf.py`` shows the full-dict form which is what production
    sites should use.

    Examples
    --------
    >>> _normalize_panel_models(None)
    []
    >>> _normalize_panel_models("gpt-4o")  # doctest: +ELLIPSIS
    [{'id': 'gpt-4o', ...}]
    >>> _normalize_panel_models(["a", "b"])  # doctest: +ELLIPSIS
    [{'id': 'a', ...}, {'id': 'b', ...}]
    """
    if raw is None or raw in ("", []):
        return []
    if isinstance(raw, str):
        return [
            {
                "id": raw,
                "model": raw,
                "provider": "custom",
                "endpoint": "",
                "label": raw,
            }
        ]
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            out.append(
                {
                    "id": item,
                    "model": item,
                    "provider": "custom",
                    "endpoint": "",
                    "label": item,
                }
            )
        elif isinstance(item, dict):
            out.append(dict(item))  # shallow copy — never mutate user data
        # else: silently skip non-str / non-dict (validator would reject anyway)
    return out


def _validate_panel_model(
    model: dict[str, Any],
    name: str = "",
) -> list[str]:
    r"""Validate a single panel-model entry.

    Parameters
    ----------
    model : dict
        Already-normalised entry (see :func:`_normalize_panel_models`).
    name : str, optional
        Entry id / index used in error messages.

    Returns
    -------
    list of str
        Empty list means the entry is valid.  Otherwise a list of
        human-readable error strings.  Mirrors :func:`_validate_provider`
        so the existing error-handling shape is reused.

    Notes
    -----
    Checks performed (all deductive, no heuristics):

    * Every :data:`_PANEL_MODEL_REQUIRED_KEYS` is present and non-empty
      (with one explicit exception: ``endpoint`` may be empty when
      ``provider == "stub"`` — the stub provider intentionally has no URL).
    * ``provider`` is in :data:`_PANEL_MODEL_PROVIDERS`.
    * ``endpoint`` is either empty (stub) or passes
      :func:`_validate_provider_url_template` (which rejects ``javascript:``,
      ``data:``, ``ftp:``, etc.).  Site-relative paths (``"/api/proxy"``)
      are allowed because the underlying check already accepts them via
      the leading-slash branch.
    * ``id`` matches a conservative ``[A-Za-z0-9_.:\-]+`` pattern — it is
      used as a ``sessionStorage`` key and a JSON field, so we forbid
      whitespace and HTML-injection characters at config time.
    """
    errors: list[str] = []
    prefix = f"Panel model {name!r}: " if name else "Panel model: "

    for key in _PANEL_MODEL_REQUIRED_KEYS:
        if key not in model:
            errors.append(f"{prefix}missing required key {key!r}")

    provider = str(model.get("provider", ""))
    if provider and provider not in _PANEL_MODEL_PROVIDERS:
        errors.append(
            f"{prefix}provider {provider!r} must be one of "
            f"{sorted(_PANEL_MODEL_PROVIDERS)}"
        )

    endpoint = str(model.get("endpoint", ""))
    if endpoint:
        # Accept:
        #   * http://… and https://… (validated via the cross-module helper)
        #   * site-relative paths beginning with a single "/"
        #     (e.g. "/_proxy/anthropic" — common Vercel / Netlify pattern)
        # Reject everything else (javascript:, data:, ftp:, …) — these are
        # the exact schemes the existing provider-url guard already blocks.
        is_site_relative = (
            endpoint.startswith("/")
            # protocol-relative -> reject
            and not endpoint.startswith("//")
        )
        if not (is_site_relative or _validate_provider_url_template(endpoint)):
            errors.append(
                f"{prefix}endpoint {endpoint!r} must be http://, https://, "
                f"or site-relative (start with a single '/')"
            )
    # NOTE: empty endpoint is INTENTIONALLY accepted for every provider.
    # The JS routing layer falls back to ``cfg.panelApiUrl`` when an entry
    # has no per-model endpoint, which lets the convenient ``list[str]``
    # config shape work against a shared proxy.  See the docstring of
    # ``_normalize_panel_models`` and ``_panelApiCall`` in ai-assistant.js.

    mid = str(model.get("id", ""))
    if mid and not re.match(r"^[A-Za-z0-9_.:\-]+$", mid):
        errors.append(f"{prefix}id {mid!r} must contain only [A-Za-z0-9_.:-]")

    return errors


def _filter_panel_models(
    raw: Any,
) -> list[dict[str, Any]]:
    """Return a copy of *raw* normalised, validated, and de-duplicated.

    Parameters
    ----------
    raw : Any
        Whatever the user wrote in ``ai_assistant_panel_api_models``.

    Returns
    -------
    list of dict
        Each entry is normalised (via :func:`_normalize_panel_models`),
        guaranteed to pass :func:`_validate_panel_model`, and has a unique
        ``id``.  Invalid entries are *dropped* (logged via the Sphinx
        warning channel) rather than aborting the build — config errors
        should never break documentation publishing.

    Notes
    -----
    Default-flag handling: at most one entry may carry ``"default": True``.
    If multiple do, the first wins and the rest are silently demoted (logged
    warning).  If none do, no entry is marked default; the JS treats the
    first valid entry as the active model in that case.
    """
    items = _normalize_panel_models(raw)
    seen_ids: set[str] = set()
    out: list[dict[str, Any]] = []
    seen_default = False
    for idx, entry in enumerate(items):
        errs = _validate_panel_model(entry, name=entry.get("id") or str(idx))
        if errs:
            try:
                for e in errs:
                    _get_logger().warning(f"AI Assistant: {e}")
            except Exception:  # noqa: BLE001 — log path itself never breaks build
                pass
            continue
        mid = str(entry["id"])
        if mid in seen_ids:
            try:  # noqa: SIM105
                _get_logger().warning(
                    f"AI Assistant: duplicate panel-model id {mid!r}; "
                    f"keeping the first occurrence."
                )
            except Exception:  # noqa: BLE001
                pass
            continue
        seen_ids.add(mid)

        # Default-flag arbitration: first True wins; demote later ones.
        if entry.get("default") is True:
            if seen_default:
                entry = {**entry, "default": False}  # noqa: PLW2901
                try:  # noqa: SIM105
                    _get_logger().warning(
                        f"AI Assistant: multiple panel-models marked default; "
                        f"demoting {mid!r}."
                    )
                except Exception:  # noqa: BLE001
                    pass
            else:
                seen_default = True
        out.append(entry)
    return out


def _filter_share_targets(
    raw: Any,
) -> list[dict[str, Any]]:
    """Normalise + URL-validate share-sheet targets.

    Parameters
    ----------
    raw : Any
        Accepted shapes:

        * ``list[tuple[str, dict]]`` — what ``_DEFAULT_SHARE_TARGETS.items()``
          yields when used as the default.
        * ``list[dict]`` — the user-facing config shape.  Each dict must
          contain at minimum ``label`` and ``url_template`` (key ``id`` is
          optional; the dict's own key in the original mapping becomes the
          ``id`` automatically when entering via ``items()``).
        * Anything else → ``[]``.

    Returns
    -------
    list of dict
        Each entry guaranteed to have ``id``, ``label``, ``url_template``
        (validated via :func:`_validate_provider_url_template`), and
        ``icon`` (with sensible default).  Entries whose URL fails the
        scheme allow-list are dropped (logged warning).

    Notes
    -----
    The empty ``url_template`` is intentionally allowed for the
    ``copy_link`` target: the JS handles it by writing the page URL to the
    clipboard instead of opening an intent URL.  Every other empty template
    is treated as a config error and the entry is dropped.
    """
    out: list[dict[str, Any]] = []
    if not isinstance(raw, (list, tuple)):
        return out
    seen_ids: set[str] = set()
    for item in raw:
        if (
            isinstance(item, tuple)
            and len(item) == 2  # noqa: PLR2004
            and isinstance(item[1], dict)
        ):
            tid, tval = item
            entry = {"id": str(tid), **tval}
        elif isinstance(item, dict):
            entry = dict(item)
            if "id" not in entry:
                # Derive a deterministic id from the label as fallback.
                lbl = str(entry.get("label", "")).strip().lower()
                entry["id"] = re.sub(r"[^a-z0-9_-]+", "_", lbl) or "share"
        else:
            continue

        tid = str(entry["id"])
        if tid in seen_ids:
            continue
        url_tpl = str(entry.get("url_template", ""))
        # url_template may be empty ONLY for copy_link; otherwise must pass scheme check.
        if tid != "copy_link":  # noqa: SIM102
            if not url_tpl or not _validate_provider_url_template(  # noqa: SIM102
                url_tpl
            ):
                # mailto: is also a legitimate share scheme — accept it explicitly.
                if not url_tpl.lower().startswith("mailto:"):
                    try:  # noqa: SIM105
                        _get_logger().warning(
                            f"AI Assistant: share target {tid!r} has invalid "
                            f"url_template {url_tpl!r}; dropping."
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    continue
        entry.setdefault("label", tid.replace("_", " ").title())
        entry.setdefault("icon", "share.svg")
        seen_ids.add(tid)
        out.append(entry)
    return out


#: Regex matching localhost or loopback origins; used to validate Ollama URLs.
_LOCALHOST_RE = re.compile(
    r"^https?://(localhost|127\.0\.0\.1)(:\d+)?(/|$)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Default MCP tool registry
# ---------------------------------------------------------------------------

#: Default MCP tool configurations.
#:
#: Each entry describes an MCP server integration that can be surfaced in the
#: AI-assistant widget as a "Connect" button.  Disabled by default; enable and
#: configure via the ``ai_assistant_mcp_tools`` Sphinx config value.
#:
#: Schema per tool:
#:
#: .. code-block:: python
#:
#:     {
#:         "enabled"     : bool,  # shown when True
#:         "type"        : str,   # "vscode" | "claude_desktop" | "cursor" | "windsurf" | "generic"
#:         "label"       : str,   # button text
#:         "description" : str,   # tooltip
#:         "icon"        : str,   # SVG filename
#:         # type-specific fields:
#:         "server_name" : str,   # (vscode/generic) MCP server identifier
#:         "server_url"  : str,   # (vscode/generic) SSE endpoint URL
#:         "transport"   : str,   # "sse" | "stdio"
#:         "mcpb_url"    : str,   # (claude_desktop) mcpb:// deep-link URL
#:     }
_DEFAULT_MCP_TOOLS: dict[str, dict[str, Any]] = {
    "vscode": {
        "enabled": False,
        "type": "vscode",
        "label": "Connect to VS Code",
        "description": "Install and connect the MCP server in VS Code",
        "icon": "vscode.svg",
        "server_name": "",
        "server_url": "",
        "transport": "sse",
    },
    "claude_desktop": {
        "enabled": False,
        "type": "claude_desktop",
        "label": "Connect to Claude Desktop",
        "description": "Connect via Claude Desktop MCP integration",
        "icon": "claude.svg",
        "mcpb_url": "",
    },
    "cursor": {
        "enabled": False,
        "type": "cursor",
        "label": "Connect to Cursor",
        "description": "Connect the MCP server to Cursor IDE",
        "icon": "cursor.svg",
        "server_name": "",
        "server_url": "",
        "transport": "sse",
    },
    "windsurf": {
        "enabled": False,
        "type": "windsurf",
        "label": "Connect to Windsurf",
        "description": "Connect the MCP server to Windsurf IDE",
        "icon": "windsurf.svg",
        "server_name": "",
        "server_url": "",
        "transport": "sse",
    },
    "generic": {
        "enabled": False,
        "type": "generic",
        "label": "Connect MCP Server",
        "description": "Connect any MCP-compatible server",
        "icon": "vscode.svg",
        "server_name": "",
        "server_url": "",
        "transport": "sse",
    },
}


# ---------------------------------------------------------------------------
# Internal helpers — security
# ---------------------------------------------------------------------------

_SCRIPT_CLOSE_RE = re.compile(r"</", re.IGNORECASE)


def _safe_json_for_script(obj: Any) -> str:
    r"""
    Serialise *obj* to a JSON string safe for inline ``<script>`` injection.

    Parameters
    ----------
    obj : Any
        JSON-serialisable Python object.

    Returns
    -------
    str
        JSON string with ``</`` replaced by ``<\/`` to prevent the browser
        from interpreting the literal ``</script>`` tag inside the payload.

    Notes
    -----
    Python's :func:`json.dumps` does **not** escape ``<``, ``>``, or ``&``
    by default.  The ``</script>`` sequence inside a ``<script>`` block
    causes the browser's HTML parser to close the script prematurely, which
    is a common XSS vector.  Replacing every ``</`` with ``<\/`` is the
    canonical defence — the JSON remains semantically identical but the
    HTML parser sees no closing tag.

    References
    ----------
    .. [1] https://html.spec.whatwg.org/multipage/scripting.html

    Examples
    --------
    >>> _safe_json_for_script({"url": "https://example.com/</script>"})
    '{"url": "https://example.com/<\/script>"}'
    """
    # Compact separators: no spaces after ',' or ':'.
    # Rationale: this JSON is embedded in an inline <script> tag served with
    # every page; whitespace has no semantic value there and just adds bytes.
    # Python default separators are (', ', ': ') which adds ~15-30% overhead
    # on large configs.  (',', ':') is the standard compact form.
    raw = json.dumps(obj, ensure_ascii=True, separators=(",", ":"))
    # Replace every '</' to prevent script-injection regardless of tag name
    # re.sub reduces to one literal backslash.
    return _SCRIPT_CLOSE_RE.sub(r"<\\/", raw)


def _is_path_within(path: Path, parent: Path) -> bool:
    """Return *True* if *path* is strictly within *parent*.

    Parameters
    ----------
    path : pathlib.Path
        Candidate path to test.
    parent : pathlib.Path
        Trusted root directory.

    Returns
    -------
    bool
        ``True`` when *path* is the same as or a descendant of *parent*;
        ``False`` for any path outside the tree (path-traversal attempt).

    Examples
    --------
    >>> from pathlib import Path
    >>> _is_path_within(Path("/a/b/c"), Path("/a/b"))
    True
    >>> _is_path_within(Path("/a/../etc/passwd"), Path("/a"))
    False
    """
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


_URL_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)

import ipaddress  # stdlib — safe to add to existing imports block

#: Schema version emitted as ``_schemaV`` in every validated profile dict.
#: Increment when the profile dict shape changes incompatibly so the JS
#: registry can detect and discard stale localStorage caches.
_PROFILE_SCHEMA_VERSION: int = 2

#: Maximum number of profiles before a Sphinx WARNING is emitted.
#: More than this is unusual and may indicate a conf.py loop/bug.
_MAX_PROFILE_COUNT: int = 20

#: Maximum character length for a profile ``label`` field.
#: Enforced during validation; truncated labels get a WARNING.
_MAX_PROFILE_LABEL_LEN: int = 80

#: JS prototype-pollution sentinel keys.  These must *never* appear as
#: top-level keys in ``window.AI_ASSISTANT_ENDPOINTS`` or in a profile dict
#: because they would overwrite Object.prototype / Function properties.
#: See: https://cheatsheetseries.owasp.org/cheatsheets/Prototype_Pollution_Prevention_Cheat_Sheet.html
_DANGEROUS_PROFILE_KEYS: frozenset[str] = frozenset(
    {
        "__proto__",
        "constructor",
        "prototype",
        "toString",
        "valueOf",
        "hasOwnProperty",
        "isPrototypeOf",
        "propertyIsEnumerable",
        "toLocaleString",
        "toJSON",
    }
)

#: Private IPv4 network blocks (RFC 1918, loopback, link-local, CGNAT).
#: URLs that resolve to these hosts are SSRF candidates in production.
#: Note: ``http://localhost`` is intentionally permitted (and warned) so
#: that local-development Ollama / dev-proxy workflows are not broken.
_PRIVATE_NETWORKS: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),  # loopback
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / APIPA
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT (RFC 6598)
    ipaddress.ip_network("192.0.2.0/24"),  # TEST-NET-1 (RFC 5737)
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
    ipaddress.ip_network("0.0.0.0/8"),  # "this" network
    ipaddress.ip_network("240.0.0.0/4"),  # reserved
)

#: Token field characters that indicate XSS / injection attempts.
#: These are rejected regardless of other validation.
_TOKEN_INJECT_RE = re.compile(
    r"(<\s*script|javascript:|data:\s*text/html|</|>)",
    re.IGNORECASE,
)


# ── BLOCK A helper function ──────────────────────────────────────────────────


def _check_private_ip_url(url: str) -> bool:
    """Return True when *url*'s host resolves to a known private IP range.

    Parameters
    ----------
    url : str
        A URL string already confirmed to begin with ``https://`` or
        ``http://`` (i.e. ``_URL_SCHEME_RE`` has already matched).

    Returns
    -------
    bool
        ``True`` when the host literal is a private IPv4 address, a
        loopback address (``127.*`` / ``::1``), or a well-known private
        hostname (``localhost``, ``*.local``).  ``False`` for all public
        addresses and for any hostname that cannot be parsed as an IP.

    Notes
    -----
    Developer: This function performs *only* lexical / literal analysis.
    It does NOT perform DNS resolution — that would be a network side
    effect inside a Sphinx build, which is unacceptable.  A hostname
    like ``my-internal-proxy.corp.example.com`` is NOT flagged; only
    literal private IPs and the reserved hostnames listed below are.

    Developer: Returns ``False`` (not an error) for unrecognised hosts so
    that the strict caller (``_validate_profile``) only warns on
    well-known problematic values.  Operators using split-DNS or internal
    proxies should expect the WARNING and can suppress it by setting a
    ``_allow_private`` flag on the profile dict (future extension).

    Developer: IPv6 loopback ``::1`` is detected.  Full IPv6 private
    range analysis (``fc00::/7``) is omitted — the ULA range is not
    widely used for proxy deployments and false-positives would confuse
    Ollama / WSL2 users.

    Examples
    --------
    >>> _check_private_ip_url("http://127.0.0.1:8080/v1")
    True
    >>> _check_private_ip_url("http://localhost/v1")
    True
    >>> _check_private_ip_url("https://proxy.example.com/v1")
    False
    >>> _check_private_ip_url("http://192.168.1.42/v1")
    True
    >>> _check_private_ip_url("http://10.0.0.1/api")
    True
    >>> _check_private_ip_url("https://1.1.1.1/v1")
    False
    """
    try:
        # Cheap host extraction: strip scheme, split on first "/".
        without_scheme = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host_port = without_scheme.split("/")[0]
        # Remove port if present (handle IPv6 brackets).
        if host_port.startswith("["):
            # IPv6 literal: [::1]:8080 → strip brackets.
            host = host_port.split("]")[0].lstrip("[")
        else:
            host = host_port.split(":")[0]

        host_lower = host.lower().strip()

        # ── Reserved hostnames (lexical check — no DNS) ───────────────
        if host_lower in ("localhost", "localhost.localdomain", "ip6-localhost"):
            return True
        if host_lower.endswith((".local", ".internal")):
            return True
        if host_lower == "::1":
            return True

        # ── Literal IPv4 / IPv6 address check ─────────────────────────
        addr = ipaddress.ip_address(host)
        if addr.version == 6:  # noqa: PLR2004
            return addr.is_loopback or addr.is_private or addr.is_link_local
        # IPv4: check against all private networks.
        return any(addr in network for network in _PRIVATE_NETWORKS)

    except (ValueError, AttributeError, IndexError):
        # host is a hostname or malformed — cannot determine; not flagged.
        return False


#: Regex for the ``mcpb_url`` field in ``claude_desktop`` MCP tools.
#: Only ``mcpb://`` (Claude Desktop deep-link) and ``https://`` (direct
#: download) are safe; ``javascript:``, ``data:``, ``blob:``, ``ftp:``,
#: and all other schemes are rejected to prevent malicious package substitution.
_MCPB_URL_RE = re.compile(r"^(?:mcpb|https)://", re.IGNORECASE)

#: Characters that have no valid place in a CSS selector and signal an
#: HTML-injection attempt.  Note: brackets ``[]`` and quotes ``"'`` are
#: intentionally allowed because attribute selectors legitimately use them
#: (e.g. ``div[role="main"]``).
_DANGEROUS_CSS_CHARS_RE = re.compile(r"[<>]")

#: Widget positions accepted by the JavaScript widget.
_ALLOWED_POSITIONS: frozenset = frozenset({"sidebar", "title", "floating", "none"})


def _validate_base_url(url: str) -> str:
    """Validate and normalise a documentation base URL.

    Parameters
    ----------
    url : str
        The candidate URL.

    Returns
    -------
    str
        The input URL, stripped of trailing slashes.

    Raises
    ------
    ValueError
        If *url* is non-empty and does not begin with ``http://`` or
        ``https://``.

    Examples
    --------
    >>> _validate_base_url("https://docs.example.com/")
    'https://docs.example.com'
    >>> _validate_base_url("")
    ''
    """
    url = url.strip()
    if url and not _URL_SCHEME_RE.match(url):
        raise ValueError(
            f"ai_assistant_base_url must start with http:// or https://; got {url!r}"
        )
    return url.rstrip("/")


def _validate_position(position: str) -> str:
    """Validate and normalise the widget position string.

    Parameters
    ----------
    position : str
        Requested widget position.  Accepted values (case-insensitive):
        ``"sidebar"``, ``"title"``, ``"floating"``, ``"none"``.

    Returns
    -------
    str
        The lower-cased, stripped position value.

    Raises
    ------
    ValueError
        When *position* is not one of the accepted values.

    Examples
    --------
    >>> _validate_position("sidebar")
    'sidebar'
    >>> _validate_position("TITLE")
    'title'
    """
    pos = str(position).strip().lower()
    if pos not in _ALLOWED_POSITIONS:
        raise ValueError(
            f"ai_assistant_position must be one of "
            f"{sorted(_ALLOWED_POSITIONS)}; got {position!r}"
        )
    return pos


def _validate_provider_url_template(url_template: str) -> bool:
    """Return *True* if *url_template* uses a safe ``http``/``https`` scheme.

    Parameters
    ----------
    url_template : str
        The URL template string for an AI provider.

    Returns
    -------
    bool
        ``True`` for empty strings, ``http://``, or ``https://`` prefixes.
        ``False`` for ``javascript:``, ``data:``, ``ftp:``, etc.

    Notes
    -----
    ``http://localhost`` is intentionally **accepted** here because local
    providers (Ollama) legitimately use loopback URLs.  A separate
    :func:`_validate_ollama_url` enforces loopback-only for ``api_base_url``.

    Examples
    --------
    >>> _validate_provider_url_template("https://claude.ai/new?q={prompt}")
    True
    >>> _validate_provider_url_template("javascript:alert(1)")
    False
    """
    stripped = str(url_template).strip()
    if not stripped:
        return True
    return bool(_URL_SCHEME_RE.match(stripped))


def _validate_ollama_url(url: str) -> bool:
    """Return *True* if *url* targets a loopback (localhost) origin only.

    Parameters
    ----------
    url : str
        The ``api_base_url`` for an Ollama-type provider.

    Returns
    -------
    bool
        ``True`` when *url* is empty, ``localhost``, or ``127.0.0.1``.
        ``False`` for any remote host — prevents Ollama config from being
        pointed at an external data-exfiltration endpoint.

    Examples
    --------
    >>> _validate_ollama_url("http://localhost:11434")
    True
    >>> _validate_ollama_url("https://remote.example.com")
    False
    """
    stripped = str(url).strip()
    if not stripped:
        return True
    return bool(_LOCALHOST_RE.match(stripped))


def _validate_css_selector(selector: str) -> bool:
    """Return *True* if *selector* contains no HTML-injection characters.

    Parameters
    ----------
    selector : str
        A CSS selector string.

    Returns
    -------
    bool
        ``False`` when the selector contains ``<`` or ``>``.

    Examples
    --------
    >>> _validate_css_selector('div[role="main"]')
    True
    >>> _validate_css_selector("<script>bad</script>")
    False
    """
    return not bool(_DANGEROUS_CSS_CHARS_RE.search(selector))


def _validate_mcp_tool(tool: dict[str, Any], name: str = "") -> list[str]:
    """Validate a single MCP tool configuration dict.

    Parameters
    ----------
    tool : dict
        MCP tool configuration to validate.
    name : str, optional
        Tool name for use in error messages.

    Returns
    -------
    list of str
        Validation error strings.  Empty list means the tool is valid.

    Notes
    -----
    Checks performed:

    * Required keys: ``"enabled"``, ``"type"``, ``"label"``, ``"description"``.
    * ``"server_url"`` (when present and non-empty) uses a safe HTTP/HTTPS scheme.
    * ``"mcpb_url"`` (when present and non-empty) uses a safe ``mcpb://`` or
      ``https://`` scheme.

    Examples
    --------
    >>> _validate_mcp_tool(
    ...     {"enabled": False, "type": "vscode", "label": "VS Code", "description": "x"}
    ... )
    []
    """
    errors: list[str] = []
    prefix = f"MCP tool {name!r}: " if name else "MCP tool: "
    for key in ("enabled", "type", "label", "description"):
        if key not in tool:
            errors.append(f"{prefix}missing required key {key!r}")
    server_url = str(tool.get("server_url", "")).strip()
    if server_url and not _URL_SCHEME_RE.match(server_url):
        errors.append(f"{prefix}server_url {server_url!r} must use http:// or https://")
    # Validate mcpb_url when present — only mcpb:// or https:// are safe.
    # A javascript: or data: scheme here would reach handleMCPInstall() and
    # trigger a malicious file download.  The JS side also validates, but
    # defence-in-depth at build time is cheaper than a runtime surprise.
    mcpb_url = str(tool.get("mcpb_url", "")).strip()
    if mcpb_url and not _MCPB_URL_RE.match(mcpb_url):
        errors.append(f"{prefix}mcpb_url {mcpb_url!r} must use mcpb:// or https://")
    return errors


def _sanitize_selectors(selectors: list[str]) -> list[str]:
    """Filter out empty or unsafe CSS selectors from a list.

    Parameters
    ----------
    selectors : list of str
        Candidate CSS selectors.

    Returns
    -------
    list of str
        Only selectors that pass :func:`_validate_css_selector` and are
        non-empty after stripping.

    Examples
    --------
    >>> _sanitize_selectors(["article", "<bad>", "  ", "main"])
    ['article', 'main']
    """
    return [s for s in selectors if s.strip() and _validate_css_selector(s)]


def _validate_provider(provider: dict[str, Any], name: str = "") -> list[str]:
    """Validate a provider configuration dict.

    Parameters
    ----------
    provider : dict
        Provider configuration to validate.
    name : str, optional
        Provider name for use in error messages.

    Returns
    -------
    list of str
        List of human-readable validation error strings.  Empty list means
        the provider is valid.

    Notes
    -----
    Checks performed:

    * All :data:`_PROVIDER_REQUIRED_KEYS` are present.
    * ``"type"`` is one of :data:`_PROVIDER_TYPES`.
    * ``"url_template"`` passes :func:`_validate_provider_url_template`.
    * For ``type == "local"``, ``"api_base_url"`` (if present) passes
      :func:`_validate_ollama_url`.
    * ``"fetch_mode"`` (optional) must be one of :data:`_PROVIDER_FETCH_MODES`
      when present.  Omitting it is valid; it defaults to ``"url"`` at
      runtime.

    The ``fetch_mode`` check enforces correctness of provider configurations
    at definition time rather than silently accepting unknown strings that
    the widget JS would later ignore.

    Examples
    --------
    >>> _validate_provider({"enabled": True, "label": "X", ...})  # doctest: +SKIP
    []
    """
    errors: list[str] = []
    prefix = f"Provider {name!r}: " if name else "Provider: "

    for key in _PROVIDER_REQUIRED_KEYS:
        if key not in provider:
            errors.append(f"{prefix}missing required key {key!r}")

    ptype = str(provider.get("type", ""))
    if ptype and ptype not in _PROVIDER_TYPES:
        errors.append(
            f"{prefix}type {ptype!r} must be one of {sorted(_PROVIDER_TYPES)}"
        )

    url_tpl = str(provider.get("url_template", ""))
    if not _validate_provider_url_template(url_tpl):
        errors.append(f"{prefix}url_template {url_tpl!r} must use http:// or https://")

    if ptype == "local":
        api_url = str(provider.get("api_base_url", ""))
        if api_url and not _validate_ollama_url(api_url):
            errors.append(
                f"{prefix}api_base_url {api_url!r} must target localhost / "
                f"127.0.0.1 for local providers"
            )

    # Optional fetch_mode — validate only when explicitly set.
    fetch_mode = provider.get("fetch_mode")
    if fetch_mode is not None and str(fetch_mode) not in _PROVIDER_FETCH_MODES:
        errors.append(
            f"{prefix}fetch_mode {fetch_mode!r} must be one of "
            f"{sorted(_PROVIDER_FETCH_MODES)} (or omitted)"
        )

    return errors


def _filter_providers(
    providers: dict[str, Any],
    *,
    require_enabled: bool = False,
) -> dict[str, Any]:
    """Return a copy of *providers* with unsafe entries removed.

    Parameters
    ----------
    providers : dict
        Mapping of provider name → provider config dict.
    require_enabled : bool, optional
        When ``True``, also drop providers whose ``"enabled"`` field is
        ``False``.

    Returns
    -------
    dict
        Copy of *providers* with invalid URL templates removed (and
        optionally disabled providers removed).

    Notes
    -----
    Validation uses :func:`_validate_provider_url_template`.  Providers
    with dangerous URL schemes (``javascript:``, ``data:``, etc.) are
    always removed regardless of *require_enabled*.
    """
    result: dict[str, Any] = {}
    for name, prov in providers.items():
        url_tpl = str(prov.get("url_template", ""))
        if not _validate_provider_url_template(url_tpl):
            continue
        if require_enabled and not prov.get("enabled", False):
            continue
        result[name] = prov
    return result


# ---------------------------------------------------------------------------
# Inline SVG icon fallback
# ---------------------------------------------------------------------------


def _resolve_icon(
    icon_filename: str,
    entry_name: str,
    static_dir: Path | None = None,
) -> str:
    """Return *icon_filename* if it exists on disk, else a base64 data URI.

    Parameters
    ----------
    icon_filename : str
        Filename of the SVG icon (e.g. ``"claude.svg"``).  Returned as-is
        when the file exists in *static_dir*.
    entry_name : str
        Lower-cased provider or MCP-tool name used to look up the fallback
        icon in :data:`_PROVIDER_META` (e.g. ``"claude"``, ``"ollama"``).
    static_dir : pathlib.Path or None, optional
        Path to the ``_static`` directory.  When ``None``, the module's own
        ``_static/`` subdirectory is used.

    Returns
    -------
    str
        Either *icon_filename* (file found on disk) or a ``data:image/svg+xml``
        base64 URI from :data:`_PROVIDER_META`.  Falls back to *icon_filename*
        unchanged if the ``_static`` subpackage cannot be imported.

    Notes
    -----
    This function is called at Sphinx build time — the ``_static`` subpackage
    is always present in a correct installation.  The ``ImportError`` fallback
    exists only for robustness in incomplete or test environments.

    Examples
    --------
    >>> _resolve_icon("claude.svg", "claude")  # doctest: +SKIP
    'claude.svg'                               # when file exists
    >>> _resolve_icon("missing.svg", "claude")  # doctest: +SKIP
    'data:image/svg+xml;base64,…'             # base64 fallback
    """
    if static_dir is None:
        static_dir = Path(__file__).parent / "_static"

    if (static_dir / icon_filename).is_file():
        return icon_filename  # file present — use normal static serving

    # File absent — look up base64 fallback from _static subpackage.
    try:
        # Single import attempt covers both installed-package and isolated-test
        # environments.  There is no need for a try/except pair with identical
        # import statements — if the first import fails, the identical second one
        # would also fail, making the except branch dead code.
        from ._static import (  # noqa: PLC0415
            _PROVIDER_META as _pm,  # noqa: N811
        )
        from ._static import (  # noqa: PLC0415
            _SVG_DEFAULT as _sd,  # noqa: N811
        )

        key = str(entry_name).lower()
        return _pm.get(key, {}).get("icon", _sd)
    except ImportError:
        # _static subpackage unavailable — return original filename unchanged.
        return icon_filename


# ---------------------------------------------------------------------------
# Theme selector presets
# ---------------------------------------------------------------------------

#: Mapping of ``html_theme`` names to ordered CSS selector tuples.
#: Each tuple lists selectors from most-specific to least-specific.
#: Used by :func:`_resolve_content_selectors` when
#: ``ai_assistant_theme_preset`` is set.
_THEME_SELECTOR_PRESETS: dict[str, tuple[str, ...]] = {
    # scikit-learn / NumPy / SciPy / pandas style
    "pydata_sphinx_theme": (
        "article.bd-article",
        "div.bd-article-container article",
        'div[role="main"]',
        "main",
    ),
    # Furo — https://pradyunsg.me/furo/
    "furo": (
        'article[role="main"]',
        "div.page",
        "main",
    ),
    # sphinx-book-theme (shares markup with pydata)
    "sphinx_book_theme": (
        "article.bd-article",
        'div[role="main"]',
        "main",
    ),
    # Read the Docs — https://sphinx-rtd-theme.readthedocs.io/
    "sphinx_rtd_theme": (
        "div.rst-content",
        'div[role="main"]',
        "main",
    ),
    # Alabaster — Sphinx default
    "alabaster": (
        "div.document",
        'div[role="main"]',
        "div.body",
        "main",
    ),
    # Sphinx Classic
    "classic": (
        "div.body",
        "div.document",
        "main",
    ),
    # Nature
    "nature": (
        "div.body",
        'div[role="main"]',
        "main",
    ),
    # Agogo
    "agogo": (
        "div.document",
        'div[role="main"]',
        "main",
    ),
    # Haiku
    "haiku": (
        "div.content",
        'div[role="main"]',
        "main",
    ),
    # Scrolls
    "scrolls": (
        "div.content",
        "main",
    ),
    # Press
    "press": (
        "div.body",
        "main",
    ),
    # Bootstrap / sphinx-bootstrap-theme
    "bootstrap": (
        "div.section",
        "div.body",
        "main",
    ),
    # Piccolo / sphinx-piccolo-theme
    "piccolo_theme": (
        "div.document",
        'div[role="main"]',
        "main",
    ),
    # Insegel
    "insegel": (
        "div.rst-content",
        'div[role="main"]',
        "main",
    ),
    # Groundwork
    "groundwork": (
        "div.body",
        'div[role="main"]',
        "main",
    ),
    # Non-Sphinx static site generators
    "mkdocs": (
        "div.md-content",
        "article.md-content__inner",
        "main",
    ),
    "mkdocs_material": (
        "article.md-content__inner",
        "div.md-content",
        "main",
    ),
    "jekyll": (
        "article.post-content",
        "div.content",
        "main",
        "article",
    ),
    "hugo": (
        "article",
        "div.content",
        "main",
        "div.post-content",
    ),
    "hexo": (
        "article.article-entry",
        "div.article-entry",
        "main",
        "article",
    ),
    "docusaurus": (
        "article",
        "div.markdown",
        "main",
    ),
    "vitepress": (
        "div.vp-doc",
        "main",
        "article",
    ),
    "gitbook": (
        "section.page-inner",
        "div.book-body",
        "main",
    ),
    "plain_html": (
        "main",
        "article",
        "div.content",
        "div.main",
        "div#content",
        "div#main",
    ),
}


def _resolve_content_selectors(
    preset: str | None,
    custom_selectors: list[str],
) -> tuple[str, ...]:
    """Merge theme-preset and user-defined CSS selectors into one ordered tuple.

    Parameters
    ----------
    preset : str or None
        A key from :data:`_THEME_SELECTOR_PRESETS`, or ``None`` to skip
        preset lookup.
    custom_selectors : list of str
        User-supplied selectors.  These take priority over preset selectors.

    Returns
    -------
    tuple of str
        Deduplicated selectors: custom first, preset second, module defaults
        as final fallback.  All selectors are validated; unsafe entries are
        silently removed.  Never returns an empty tuple.

    Examples
    --------
    >>> _resolve_content_selectors("furo", ["div.custom"])  # doctest: +SKIP
    ('div.custom', 'article[role="main"]', 'div.page', ...)
    """
    preset_sels: tuple[str, ...] = ()
    if preset:
        preset_sels = _THEME_SELECTOR_PRESETS.get(str(preset).strip(), ())

    seen: set = set()
    merged: list[str] = []
    for sel in (
        list(custom_selectors) + list(preset_sels) + list(_DEFAULT_CONTENT_SELECTORS)
    ):
        if sel not in seen:
            seen.add(sel)
            merged.append(sel)

    safe = _sanitize_selectors(merged)
    return tuple(safe) if safe else _DEFAULT_CONTENT_SELECTORS


# ---------------------------------------------------------------------------
# Markdown conversion — public helpers
# ---------------------------------------------------------------------------


def html_to_markdown(
    html_content: str,
    strip_tags: str | list[str] | None = None,
) -> str:
    r"""
    Convert an HTML string to Markdown using the Sphinx-tuned converter.

    Parameters
    ----------
    html_content : str
        Raw HTML string to convert.
    strip_tags : str or list of str or None, optional
        HTML tag names whose elements (including all their content) are
        removed before conversion.  A bare ``str`` is treated as a
        single-element list.  Defaults to ``["script", "style"]`` when
        ``None``.

    Returns
    -------
    str
        Markdown representation of the HTML content.

    Raises
    ------
    ImportError
        If ``markdownify`` is not installed.

    Notes
    -----
    ``bs4`` is imported **lazily** inside this function.  If it is not
    available the stripping step is skipped; ``markdownify``'s own
    ``strip=`` option still removes the listed tags from the output.

    Examples
    --------
    >>> html_to_markdown("<h1>Hello</h1><p>World</p>")  # doctest: +SKIP
    '# Hello\n\nWorld\n\n'
    """
    tags: list[str] = _coerce_to_list(strip_tags, default=["script", "style"])

    # Lazy bs4 import — avoids module-scope ImportError when bs4 is absent.
    # TypeError (BeautifulSoup is None) and ImportError are both handled here.
    try:
        from bs4 import BeautifulSoup as _BS  # noqa: N814, PLC0415

        soup = _BS(html_content, "html.parser")
        for tag in soup(tags):
            tag.decompose()
        html_content = str(soup)
    except ImportError:
        # bs4 not installed; markdownify's strip= option handles tag removal.
        pass

    ConverterClass = _build_converter_class()  # noqa: N806
    return ConverterClass(
        heading_style="ATX",
        bullets="*",
        strong_em_symbol="**",
        strip=tags,
    ).convert(html_content)


# Legacy alias kept for backwards compatibility
html_to_markdown_converter = html_to_markdown


# ---------------------------------------------------------------------------
# Multi-process workers — must be module-level for pickling
# ---------------------------------------------------------------------------

#: CSS selectors tried in order to locate the main page content.
#: Each theme uses a different element; we probe until one matches.
_DEFAULT_CONTENT_SELECTORS: tuple[str, ...] = (
    "article.bd-article",  # pydata-sphinx-theme ≥ 0.13
    'div[role="main"]',  # pydata (older), RTD, generic
    'article[role="main"]',  # Furo theme
    "div.rst-content",  # Read the Docs theme
    "div.document",  # Sphinx Classic / Alabaster
    "div.body",  # Older Sphinx themes
    "div.bd-article-container article",  # pydata nested wrapper
    "div.content",  # Haiku / Scrolls
    "div.section",  # Bootstrap / older Sphinx
    "div.md-content",  # MkDocs / Material
    "article.md-content__inner",  # MkDocs Material inner
    "div.vp-doc",  # VitePress
    "section.page-inner",  # GitBook
    "article.post-content",  # Jekyll
    "article",  # Generic / Hugo / Docusaurus
    "main",  # Generic HTML5
)


def _process_html_file_worker(
    args: tuple[str, str, str, list[str], list[str], list[str]],
) -> tuple[str, str, str]:
    """
    Extend worker: convert one HTML file to Markdown with separate I/O dirs.

    This function is intentionally **at module scope** so that it can be
    serialised by :mod:`multiprocessing`.

    Parameters
    ----------
    args : tuple
        A 6-tuple ``(html_file_path, input_dir_path, output_dir_path,
        exclude_patterns, selectors, strip_tags)``::

            html_file_path : str
                Absolute path of the ``.html`` file to convert.
            input_dir_path : str
                Path-traversal boundary (trusted root).  The HTML file must
                resolve to a path within this directory.
            output_dir_path : str
                Directory where the ``.md`` file is written.  The relative
                path is mirrored from *input_dir_path*.  Set equal to
                *input_dir_path* to write alongside each HTML file (inline
                mode).
            exclude_patterns : list of str
                Substrings; files whose relative path contains any of these
                are skipped without error.
            selectors : list of str
                CSS selectors tried in order to locate the main content.
            strip_tags : list of str
                HTML tag names removed (with content) before conversion.

    Returns
    -------
    tuple of (str, str, str)
        ``(status, relative_path, message)`` where *status* is one of
        ``"success"``, ``"skipped"``, or ``"error"``.

    Notes
    -----
    **Security** — path-traversal guard:
    The function verifies that *html_file_path* resolves to a path within
    *input_dir_path* before reading it.

    **Encoding** — files are read as UTF-8 with ``errors="replace"`` so
    that malformed HTML never crashes a worker process.
    """
    (
        html_file_str,
        input_dir_str,
        output_dir_str,
        exclude_patterns,
        selectors,
        strip_tags,
    ) = args

    html_file = Path(html_file_str)
    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    # ---- Security: path-traversal guard ------------------------------------
    if not _is_path_within(html_file, input_dir):
        return ("error", str(html_file), "Path-traversal attempt blocked")

    try:
        rel_path = html_file.relative_to(input_dir)
    except ValueError:
        return ("error", str(html_file), "File is outside input directory")

    # ---- Exclusion check ---------------------------------------------------
    rel_str = str(rel_path)
    if any(pat in rel_str for pat in exclude_patterns):
        return ("skipped", rel_str, "")

    try:
        from bs4 import BeautifulSoup  # type: ignore[import]  # noqa: PLC0415

        html_content = html_file.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html_content, "html.parser")

        main_content = None
        for selector in selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content is None:
            return ("skipped", rel_str, "No main content element found")

        # ---- OOM fix: single-parse path ------------------------------------
        # Root cause: calling html_to_markdown(str(main_content), ...) caused
        # a *second* BeautifulSoup parse inside html_to_markdown, while the
        # full-page ``soup`` was still alive (main_content is a live reference
        # into it).  With 8 concurrent workers and large Sphinx API pages this
        # doubled per-worker peak memory and triggered the Linux OOM killer.
        #
        # Fix:
        #   1. Strip unwanted tags in-place on the already-parsed tree.
        #   2. Serialise to a plain string (one allocation, no new tree).
        #   3. Null main_content and del soup — CPython refcount hits zero and
        #      the full-page tree is reclaimed *before* markdownify allocates.
        #   4. Call the markdownify converter directly; no second bs4 parse.
        for _tag_name in strip_tags:
            for _el in list(main_content.find_all(_tag_name)):
                _el.decompose()

        html_snippet = str(main_content)
        main_content = None  # release reference into soup
        del soup  # free full-page parse tree immediately

        ConverterClass = _build_converter_class()  # noqa: N806
        markdown_content = ConverterClass(
            heading_style="ATX",
            bullets="*",
            strong_em_symbol="**",
            strip=list(strip_tags),  # belt-and-suspenders for markdownify
        ).convert(html_snippet)

        # Mirror the relative path under output_dir
        md_file = output_dir / rel_path.with_suffix(".md")
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(markdown_content, encoding="utf-8")

        return ("success", rel_str, "")

    except Exception as exc:  # noqa: BLE001
        return ("error", rel_str, str(exc))


def _process_single_html_file(
    args: tuple[str, str, list[str], list[str], list[str]],
) -> tuple[str, str, str]:
    """Process one HTML file and write the companion ``.md`` file.

    This is the original 5-tuple worker kept for backward compatibility.
    It delegates to :func:`_process_html_file_worker` with
    ``output_dir == input_dir`` (inline mode).

    Parameters
    ----------
    args : tuple
        A 5-tuple ``(html_file_path, outdir_path, exclude_patterns,
        selectors, strip_tags)``.

    Returns
    -------
    tuple of (str, str, str)
        ``(status, relative_path, message)``.

    See Also
    --------
    _process_html_file_worker : Extended 6-tuple worker.
    """
    html_file_str, outdir_str, exclude_patterns, selectors, strip_tags = args
    return _process_html_file_worker(
        (
            html_file_str,
            outdir_str,
            outdir_str,  # output_dir == input_dir → inline mode
            exclude_patterns,
            selectors,
            strip_tags,
        )
    )


# ---------------------------------------------------------------------------
# Standalone (non-Sphinx) HTML directory processor  — PUBLIC
# ---------------------------------------------------------------------------


def process_html_directory(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    selectors: list[str] | None = None,
    theme_preset: str | None = None,
    exclude_patterns: list[str] | None = None,
    strip_tags: list[str] | None = None,
    max_workers: int | None = None,
    recursive: bool = True,
    generate_llms: bool = False,
    base_url: str = "",
    llms_txt_max_entries: int | None = None,
    llms_txt_full_content: bool = False,
) -> dict[str, int]:
    """Walk any HTML directory tree and convert pages to Markdown.

    This function is entirely **Sphinx-free** and works with any
    static-site generator: Sphinx, MkDocs, Jekyll, Hugo, Hexo,
    Docusaurus, VitePress, GitBook, or plain HTML.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Root directory containing ``.html`` files.
    output_dir : str or pathlib.Path or None, optional
        Directory where ``.md`` files are written, mirroring the
        directory structure of *input_dir*.  When ``None`` (default),
        ``.md`` files are written alongside each ``.html`` file (inline
        mode).
    selectors : list of str or None, optional
        CSS selectors tried in order to locate the main content element.
        When ``None``, uses the module default combined with *theme_preset*.
    theme_preset : str or None, optional
        Theme name from :data:`_THEME_SELECTOR_PRESETS` (e.g.
        ``"mkdocs_material"``, ``"jekyll"``, ``"plain_html"``).  Merged
        with *selectors*.
    exclude_patterns : list of str or None, optional
        Path substrings to skip.  Defaults to
        ``["genindex", "search", "py-modindex", "_sources", "_static"]``.
    strip_tags : list of str or None, optional
        HTML tag names removed (with content) before conversion.  Defaults
        to ``["script", "style", "nav", "footer", "header"]``.
    max_workers : int or None, optional
        Maximum parallel worker processes.  ``None`` → auto-detect (CPU
        count or 1).
    recursive : bool, optional
        When ``True`` (default), recurse into subdirectories.  When
        ``False``, only the top-level ``.html`` files are processed.
    generate_llms : bool, optional
        When ``True``, write an ``llms.txt`` index file after conversion.
    base_url : str, optional
        Base URL prepended to ``.md`` paths in ``llms.txt``.
    llms_txt_max_entries : int or None, optional
        Cap on the number of entries in ``llms.txt``.
    llms_txt_full_content : bool, optional
        When ``True``, embed full Markdown content inline in ``llms.txt``.

    Returns
    -------
    dict
        ``{"generated": int, "skipped": int, "errors": int}`` — counts of
        files processed, skipped, and errored.

    Raises
    ------
    ValueError
        If *input_dir* does not exist or is not a directory.
    ImportError
        If ``beautifulsoup4`` or ``markdownify`` is not installed.

    Examples
    --------
    >>> stats = process_html_directory(  # doctest: +SKIP
    ...     "/site/_build",
    ...     theme_preset="mkdocs_material",
    ...     generate_llms=True,
    ...     base_url="https://example.com",
    ... )
    >>> print(stats)
    {"generated": 42, "skipped": 3, "errors": 0}
    """
    input_path = Path(input_dir).resolve()
    if not input_path.exists():
        raise ValueError(f"input_dir does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"input_dir is not a directory: {input_path}")

    if not _has_markdown_deps():
        raise ImportError(
            "process_html_directory requires beautifulsoup4 and markdownify. "
            "Install with: pip install beautifulsoup4 markdownify"
        )

    output_path: Path = (
        Path(output_dir).resolve() if output_dir is not None else input_path
    )
    output_path.mkdir(parents=True, exist_ok=True)

    if exclude_patterns is None:
        exclude_patterns = [
            "genindex",
            "search",
            "py-modindex",
            "_sources",
            "_static",
        ]

    if strip_tags is None:
        strip_tags = ["script", "style", "nav", "footer", "header"]

    effective_selectors: list[str] = list(
        _resolve_content_selectors(theme_preset, selectors or [])
    )

    glob_fn = input_path.rglob if recursive else input_path.glob
    html_files = list(glob_fn("*.html"))

    args_list = [
        (
            str(f),
            str(input_path),
            str(output_path),
            list(exclude_patterns),
            effective_selectors,
            list(strip_tags),
        )
        for f in html_files
    ]

    generated = skipped = errors = 0
    total_files = len(args_list)
    processed = 0

    # Validate and resolve max_workers before spawning any processes.
    # Raises ValueError for 0 or negative values; returns None for auto-detect.
    resolved_workers = _resolve_max_workers(max_workers)
    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        futures = {executor.submit(_process_html_file_worker, a): a for a in args_list}
        for future in as_completed(futures):
            try:
                # timeout=120: guards against worker stalls on network-mount
                # filesystems (NFS, SMB) where a file read can block
                # indefinitely.  120 s is generous for any single HTML page.
                # TimeoutError is caught by the broad Exception handler below.
                status, _rel, _msg = future.result(timeout=120)
            except Exception:  # noqa: BLE001
                errors += 1
            else:
                if status == "success":
                    generated += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    errors += 1
            processed += 1
            _write_progress_bar(processed, total_files, label="HTML→Markdown")

    if generate_llms:
        generate_llms_txt_standalone(
            output_path,
            base_url=base_url,
            max_entries=llms_txt_max_entries,
            full_content=llms_txt_full_content,
        )

    return {"generated": generated, "skipped": skipped, "errors": errors}


# ---------------------------------------------------------------------------
# Standalone llms.txt generator — PUBLIC
# ---------------------------------------------------------------------------


def generate_llms_txt_standalone(
    md_root: str | Path,
    *,
    base_url: str = "",
    output_file: str | Path | None = None,
    project_name: str = "Documentation",
    max_entries: int | None = None,
    full_content: bool = False,
) -> Path:
    """Write ``llms.txt`` from an existing set of ``.md`` files.

    This function is entirely **Sphinx-free**.

    Parameters
    ----------
    md_root : str or pathlib.Path
        Root directory containing ``.md`` files (searched recursively).
    base_url : str, optional
        Base URL prepended to each ``.md`` path.  Must be ``http://`` or
        ``https://`` if non-empty.
    output_file : str or pathlib.Path or None, optional
        Explicit path for the output file.  Defaults to
        ``<md_root>/llms.txt``.
    project_name : str, optional
        Project name written in the file header.
    max_entries : int or None, optional
        Cap on the number of entries.  ``None`` means unlimited.
    full_content : bool, optional
        When ``True``, embed each page's Markdown content inline.

    Returns
    -------
    pathlib.Path
        Absolute path of the written ``llms.txt`` file.

    Raises
    ------
    ValueError
        If *base_url* is non-empty and uses a non-HTTP scheme.
    FileNotFoundError
        If *md_root* does not exist.

    Examples
    --------
    >>> generate_llms_txt_standalone(  # doctest: +SKIP
    ...     "/site/_build",
    ...     base_url="https://docs.example.com",
    ...     project_name="MyProject",
    ... )
    PosixPath('/site/_build/llms.txt')
    """
    root = Path(md_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"md_root does not exist: {root}")

    validated_url = _validate_base_url(base_url)

    md_files = sorted(root.rglob("*.md"))
    if max_entries is not None:
        cap = max(0, int(max_entries))
        md_files = md_files[:cap]

    out_path = (
        Path(output_file).resolve() if output_file is not None else root / "llms.txt"
    )

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {project_name} Documentation\n\n")
        fh.write(
            "This file lists all available documentation pages "
            "in Markdown format.\n"
            "Generated by scikitplot._externals._sphinx_ext._sphinx_ai_assistant.\n\n"
        )
        for md_file in md_files:
            rel = md_file.relative_to(root)
            rel_posix = str(rel).replace(os.sep, "/")
            line = f"{validated_url}/{rel_posix}" if validated_url else rel_posix
            if full_content:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                fh.write(f"\n---\n{line}\n\n{content}\n")
            else:
                fh.write(f"{line}\n")

    return out_path


# ---------------------------------------------------------------------------
# Corpus knowledge graph — PUBLIC (Sphinx-free)
# ---------------------------------------------------------------------------


def plot_corpus_knowledge(  # noqa: PLR0912
    root_dir: str | Path,
    *,
    file_ext: str = ".md",
    max_pages: int | None = None,
    include_links: bool = True,
    base_url: str = "",
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    r"""Analyse a documentation corpus and return a knowledge graph.

    Walks *root_dir* for files matching *file_ext*, extracts page titles,
    headings, and internal hyperlinks, and returns a structured graph dict
    suitable for further analysis, serialisation to JSON, or visualisation
    with optional matplotlib / networkx.

    This function is **Sphinx-free** and works on any static-site output
    directory (Sphinx, MkDocs, Jekyll, Hugo, plain Markdown repos).

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root directory to scan recursively.
    file_ext : str, optional
        File extension to collect.  Defaults to ``".md"``.  Use ``".html"``
        to scan raw HTML output (link extraction uses ``href`` attributes in
        that case).
    max_pages : int or None, optional
        Cap on the number of pages to include.  ``None`` means unlimited.
    include_links : bool, optional
        When ``True`` (default), extract internal hyperlinks from each page
        to build the graph edges.  Set to ``False`` for a fast node-only
        summary.
    base_url : str, optional
        Base URL stripped from absolute links to normalise them to relative
        paths.  If non-empty, must start with ``http://`` or ``https://``.
    output_file : str or pathlib.Path or None, optional
        When provided, the returned dict is serialised as JSON and written
        to this path.  The parent directory is created automatically.

    Returns
    -------
    dict
        A knowledge graph dict with the following structure::

            {
                "root": str,          # resolved root_dir path
                "file_ext": str,      # file_ext used
                "pages": {
                    "rel/path.md": {
                        "title": str,        # first heading or filename stem
                        "headings": [str],   # all headings found (h1-h3)
                        "links": [str],      # internal relative links
                        "size_bytes": int,
                    },
                    ...
                },
                "edges": [               # directed link graph
                    {"from": "a.md", "to": "b.md"},
                    ...
                ],
                "stats": {
                    "total_pages": int,
                    "total_edges": int,
                    "total_headings": int,
                    "isolated_pages": int, # pages with no in/out links
                    "avg_links_per_page": float,
                },
            }

    Raises
    ------
    ValueError
        If *root_dir* does not exist, is not a directory, *file_ext* does not
        start with ``"."``, or *base_url* is non-empty and uses a non-HTTP
        scheme.
    TypeError
        If *max_pages* is non-None and cannot be cast to a non-negative int.

    Notes
    -----
    **User note** — the returned dict is always present regardless of whether
    *output_file* is given, so callers can immediately post-process the graph
    without reading the file back.

    **Developer note** — link extraction uses a lightweight regex rather than
    a full Markdown parser so that this function has zero additional
    dependencies.  The regex matches the ``[text](url)`` Markdown link syntax
    and the ``href="..."`` HTML attribute.  Absolute links are kept only when
    they begin with *base_url*; all others are skipped.  Relative links are
    resolved relative to the page's directory using :func:`pathlib.Path`
    semantics so that ``../api/module.md`` from ``docs/guide.md`` correctly
    resolves to ``api/module.md``.

    **Security note** — *root_dir* is resolved with
    :func:`~pathlib.Path.resolve` before every file is checked with
    :func:`_is_path_within` to prevent path-traversal.

    Examples
    --------
    >>> graph = plot_corpus_knowledge("/docs/_build")  # doctest: +SKIP
    >>> print(graph["stats"])
    {"total_pages": 42, "total_edges": 118, ...}

    >>> # Write to JSON for downstream tooling
    >>> plot_corpus_knowledge(
    ...     "/docs/_build",
    ...     output_file="/tmp/corpus_graph.json",
    ... )  # doctest: +SKIP
    """
    root = Path(root_dir).resolve()
    if not root.exists():
        raise ValueError(f"root_dir does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"root_dir is not a directory: {root}")
    if not file_ext.startswith("."):
        raise ValueError(f"file_ext must start with '.'; got {file_ext!r}")

    validated_base = _validate_base_url(base_url)  # raises ValueError for bad schemes

    if max_pages is not None:
        try:
            max_pages = max(0, int(max_pages))
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"max_pages must be a non-negative integer or None; got {max_pages!r}"
            ) from exc

    # --- Regex patterns (stdlib only, no markdown parser) -------------------
    # Markdown link: [text](url) — captures the URL part.
    _MD_LINK_RE = re.compile(r"\[(?:[^\[\]]*)\]\(([^)]+)\)")  # noqa: N806
    # HTML href attribute.
    _HTML_HREF_RE = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)  # noqa: N806
    # Markdown headings: # Title, ## Sub, ### Sub-sub.
    _MD_HEADING_RE = re.compile(r"^#{1,3}\s+(.+)", re.MULTILINE)  # noqa: N806
    # HTML headings <h1>...</h1> through <h3>.
    _HTML_HEADING_RE = re.compile(  # noqa: N806
        r"<h[1-3][^>]*>([^<]+)</h[1-3]>", re.IGNORECASE
    )

    is_md = file_ext.lower() in (".md", ".markdown", ".rst")
    link_re = _MD_LINK_RE if is_md else _HTML_HREF_RE
    heading_re = _MD_HEADING_RE if is_md else _HTML_HEADING_RE

    all_files = sorted(root.rglob(f"*{file_ext}"))
    if max_pages is not None:
        all_files = all_files[:max_pages]

    pages: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    for page_path in all_files:
        # Path-traversal guard.
        if not _is_path_within(page_path, root):
            continue
        try:
            rel = str(page_path.relative_to(root)).replace(os.sep, "/")
            text = page_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # --- Title: first heading, or stem if no heading found. ------------
        headings = [m.group(1).strip() for m in heading_re.finditer(text)]
        title = headings[0] if headings else page_path.stem

        # --- Links: internal relative paths only. --------------------------
        internal_links: list[str] = []
        if include_links:
            for m in link_re.finditer(text):
                raw = m.group(1).strip()
                # Drop anchors, query params, mailto, and mailto-style strings.
                raw = raw.split("#")[0].split("?")[0]
                if not raw:
                    continue
                # Absolute link: keep only if it starts with base_url.
                if raw.startswith(("http://", "https://")):
                    if validated_base and raw.startswith(validated_base):
                        # Strip base to make it relative.
                        raw = raw[len(validated_base) :].lstrip("/")
                    else:
                        continue  # external link — skip
                # Resolve relative to page directory.
                resolved = (page_path.parent / raw).resolve()
                if not _is_path_within(resolved, root):
                    continue
                try:
                    link_rel = str(resolved.relative_to(root)).replace(os.sep, "/")
                except ValueError:
                    continue
                internal_links.append(link_rel)

            for target in internal_links:
                edges.append({"from": rel, "to": target})

        pages[rel] = {
            "title": title,
            "headings": headings,
            "links": internal_links,
            "size_bytes": page_path.stat().st_size,
        }

    # --- Stats --------------------------------------------------------------
    page_keys = set(pages)
    linked_pages: set[str] = set()
    for e in edges:
        linked_pages.add(e["from"])
        linked_pages.add(e["to"])
    isolated = len([p for p in page_keys if p not in linked_pages])
    total_headings = sum(len(v["headings"]) for v in pages.values())
    avg_links = (len(edges) / len(pages)) if pages else 0.0

    graph: dict[str, Any] = {
        "root": str(root),
        "file_ext": file_ext,
        "pages": pages,
        "edges": edges,
        "stats": {
            "total_pages": len(pages),
            "total_edges": len(edges),
            "total_headings": total_headings,
            "isolated_pages": isolated,
            "avg_links_per_page": round(avg_links, 3),
        },
    }

    if output_file is not None:
        out_path = Path(output_file).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(graph, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return graph


# ---------------------------------------------------------------------------
# Build-time hooks (Sphinx layer)
# ---------------------------------------------------------------------------


def generate_markdown_files(app: Sphinx, exception: Exception | None) -> None:
    """Post-build hook: generate ``.md`` companions for every ``.html`` file.

    Registered with Sphinx's ``build-finished`` event in :func:`setup`.
    Processing is parallelised via :class:`concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The active Sphinx application instance.
    exception : Exception or None
        Any exception raised during the build; when not ``None`` this hook
        exits immediately without generating files.

    Returns
    -------
    None

    Raises
    ------
    None
        All per-file errors are logged as warnings; the hook never raises.
    """
    # flake8: noqa: F811
    from sphinx.builders.html import StandaloneHTMLBuilder  # noqa: PLC0415

    if exception is not None:
        return

    log = _get_logger()

    builder = app.builder
    if not isinstance(builder, StandaloneHTMLBuilder):
        return

    if not app.config.ai_assistant_generate_markdown:
        log.info("AI Assistant: Markdown generation disabled")
        return

    if not _has_markdown_deps():
        log.warning(
            "AI Assistant: Markdown generation requires beautifulsoup4 and "
            "markdownify.  Install them with: "
            "pip install beautifulsoup4 markdownify"
        )
        return

    outdir = Path(builder.outdir)
    exclude_patterns: list[str] = list(
        app.config.ai_assistant_markdown_exclude_patterns
    )

    preset: str | None = getattr(app.config, "ai_assistant_theme_preset", None) or None
    selectors: list[str] = list(
        _resolve_content_selectors(
            preset,
            list(app.config.ai_assistant_content_selectors),
        )
    )
    strip_tags: list[str] = list(
        getattr(app.config, "ai_assistant_strip_tags", ["script", "style"])
    )

    html_files = list(outdir.rglob("*.html"))
    log.info(f"AI Assistant: Generating Markdown for {len(html_files)} HTML files…")

    args_list = [
        (str(f), str(outdir), list(exclude_patterns), selectors, strip_tags)
        for f in html_files
    ]

    generated = skipped = errors = 0
    total_files = len(args_list)
    processed = 0
    t0 = time.monotonic()

    # Validate and resolve max_workers before spawning any processes.
    # _resolve_max_workers raises ValueError for 0/negative values and
    # returns None for None/"auto" (ProcessPoolExecutor auto-detects).
    max_workers_cfg = app.config.ai_assistant_max_workers
    resolved_workers = _resolve_max_workers(max_workers_cfg)
    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        futures = {executor.submit(_process_single_html_file, a): a for a in args_list}
        for future in as_completed(futures):
            try:
                # timeout=120: prevents the build from hanging indefinitely
                # when a worker stalls (e.g. NFS stale file handle, OOM kill).
                # concurrent.futures.TimeoutError is caught as a generic
                # Exception below and logged as a worker error.
                status, rel_path, message = future.result(timeout=120)
            except Exception as exc:  # noqa: BLE001
                errors += 1
                log.warning(f"AI Assistant: Worker error: {exc}")
                processed += 1
                _write_progress_bar(processed, total_files, label="HTML→Markdown")
                continue
            if status == "success":
                generated += 1
            elif status == "skipped":
                skipped += 1
                if message:
                    log.debug(f"AI Assistant: Skipped {rel_path}: {message}")
            else:
                errors += 1
                log.warning(f"AI Assistant: Failed to convert {rel_path}: {message}")
            processed += 1
            _write_progress_bar(processed, total_files, label="HTML→Markdown")

    elapsed = time.monotonic() - t0
    log.info(
        f"AI Assistant: {generated} generated, {skipped} skipped, "
        f"{errors} errors — {elapsed:.1f}s ({resolved_workers or 'auto'} workers)"
    )


def generate_llms_txt(  # noqa: PLR0911
    app: Sphinx, exception: Exception | None
) -> None:
    """Post-build hook: write ``llms.txt`` listing all generated ``.md`` URLs.

    Registered with Sphinx's ``build-finished`` event in :func:`setup`.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The active Sphinx application instance.
    exception : Exception or None
        When not ``None``, the hook exits immediately.

    Returns
    -------
    None

    References
    ----------
    .. [1] https://llmstxt.org/
    """
    from sphinx.builders.html import StandaloneHTMLBuilder  # noqa: PLC0415

    if exception is not None:
        return
    if not app.config.ai_assistant_generate_markdown:
        return
    if not app.config.ai_assistant_generate_llms_txt:
        return
    if not isinstance(app.builder, StandaloneHTMLBuilder):
        return

    log = _get_logger()
    outdir = Path(app.builder.outdir)

    raw_base_url: str = (
        getattr(app.config, "html_baseurl", "")
        or app.config.ai_assistant_base_url
        or ""
    )
    try:
        base_url = _validate_base_url(raw_base_url)
    except ValueError as exc:
        log.warning(f"AI Assistant: llms.txt skipped — {exc}")
        return

    md_files = sorted(outdir.rglob("*.md"))
    if not md_files:
        log.debug("AI Assistant: No .md files found; skipping llms.txt")
        return

    max_entries: int | None = getattr(
        app.config, "ai_assistant_llms_txt_max_entries", None
    )
    if max_entries is not None:
        cap = max(0, int(max_entries))
        md_files = md_files[:cap]
        if not md_files:
            # warn (not debug): cap=0 means llms.txt will be empty.
            # This is almost always a conf.py mistake (e.g. a stale 0 left
            # after testing).  A warning surfaces it without breaking the build.
            log.warning(
                "AI Assistant: llms.txt max_entries=%d → no entries written. "
                "Set ai_assistant_llms_txt_max_entries to None or a positive "
                "integer to include entries.",
                cap,
            )
            return

    full_content: bool = bool(
        getattr(app.config, "ai_assistant_llms_txt_full_content", False)
    )
    project_name: str = getattr(app.config, "project", "Documentation")

    llms_txt = outdir / "llms.txt"
    with llms_txt.open("w", encoding="utf-8") as fh:
        fh.write(f"# {project_name} Documentation\n\n")
        fh.write(
            "This file lists all available documentation pages "
            "in Markdown format.\n"
            "Generated by scikitplot._externals._sphinx_ext._sphinx_ai_assistant.\n\n"
        )
        for md_file in md_files:
            rel = md_file.relative_to(outdir)
            rel_posix = str(rel).replace(os.sep, "/")
            line = f"{base_url}/{rel_posix}" if base_url else rel_posix
            if full_content:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                fh.write(f"\n---\n{line}\n\n{content}\n")
            else:
                fh.write(f"{line}\n")

    log.info(f"AI Assistant: llms.txt written with {len(md_files)} entries")


def _cfg_str(config: Any, key: str) -> str | None:
    """Safely read a string config value; returns ``None`` for non-str values.

    Parameters
    ----------
    config : Any
        Sphinx config object or mock.
    key : str
        Configuration key to read.

    Returns
    -------
    str or None
        The string value when present; ``None`` otherwise.

    Notes
    -----
    In test environments where ``config`` is a :class:`unittest.mock.MagicMock`,
    attribute access returns a new ``MagicMock`` rather than a real string.
    This helper returns ``None`` in that case to prevent JSON serialisation errors.
    """
    val = getattr(config, key, None)
    return val if isinstance(val, str) else None


def _cfg_bool(config: Any, key: str, default: bool = False) -> bool:
    """Safely read a boolean config value; returns *default* for non-bool/int values.

    Parameters
    ----------
    config : Any
        Sphinx config object or mock.
    key : str
        Configuration key to read.
    default : bool, optional
        Fallback when the value is not a plain ``bool`` or ``int``.

    Returns
    -------
    bool
    """
    val = getattr(config, key, default)
    return bool(val) if isinstance(val, (bool, int)) else default


def _cfg_str_list(config: Any, key: str) -> list[str]:
    """Safely read a list-of-strings config value.

    Parameters
    ----------
    config : Any
        Sphinx config object or mock.
    key : str
        Configuration key to read.

    Returns
    -------
    list[str]
        Always a plain list of str, never ``None``.  Non-string items in the
        source list are silently dropped.  If the value is a bare string it is
        wrapped in a single-element list.  Any other type returns ``[]``.

    Notes
    -----
    This guard is required because Sphinx config objects may return
    MagicMock instances during test execution, and user conf.py files can
    assign non-list types by mistake.
    """
    val = getattr(config, key, [])
    if isinstance(val, str):
        return [val] if val.strip() else []
    if isinstance(val, (list, tuple)):
        return [str(item) for item in val if isinstance(item, str)]
    return []


def _cfg_list(config: Any, key: str) -> list:
    """Safely read a list config value (e.g. list of option dicts).

    Parameters
    ----------
    config : Any
        Sphinx config object or mock.
    key : str
        Configuration key to read.

    Returns
    -------
    list
        The value when it is a genuine ``list`` or ``tuple`` (converted to
        ``list``); otherwise an empty list.  Never raises and never returns
        ``None`` so the result is always JSON-serialisable.

    Notes
    -----
    Required because Sphinx config objects return :class:`MagicMock` during
    test execution, which is neither a list nor JSON-serialisable.  Unlike
    :func:`_cfg_str_list` this preserves non-string items (e.g. dicts used
    for feedback options) instead of dropping them.
    """
    val = getattr(config, key, [])
    if isinstance(val, (list, tuple)):
        return list(val)
    return []


def _cfg_int(config: Any, key: str, default: int = 0) -> int:
    """Safely read an integer config value; returns *default* for non-int values.

    Parameters
    ----------
    config : Any
        Sphinx config object or mock.
    key : str
        Configuration key to read.
    default : int, optional
        Fallback when the value is absent, non-integer, or a
        :class:`unittest.mock.MagicMock`.  Defaults to ``0``.

    Returns
    -------
    int
        The integer value when present and valid; *default* otherwise.

    Notes
    -----
    Developer: Python ``bool`` is a subclass of ``int``.  A conf.py author
    who accidentally writes ``ai_assistant_global_share_ttl_days = True``
    gets ``1`` (TTL = 1 day) rather than the default of ``0`` (no expiry),
    which is a safer failure mode than silently ignoring the mistake.

    In test environments where ``config`` is a
    :class:`unittest.mock.MagicMock`, attribute access returns a new
    ``MagicMock`` which is not an ``int`` — this helper returns ``default``
    in that case to prevent JSON serialisation errors.

    Examples
    --------
    >>> class _Cfg:
    ...     ai_assistant_global_share_ttl_days = 30
    >>> _cfg_int(_Cfg(), "ai_assistant_global_share_ttl_days", 0)
    30
    >>> _cfg_int(_Cfg(), "missing_key", 7)
    7
    """
    val = getattr(config, key, default)
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    return default


def _cfg_endpoint_url(config: Any, key: str) -> str:
    """Read and validate a remote endpoint URL config value.

    Parameters
    ----------
    config : Any
        Sphinx config object or mock.
    key : str
        Configuration key to read (must be an ``https://`` or local
        ``http://`` URL when set).

    Returns
    -------
    str
        The raw URL string when present and valid.  Empty string ``""``
        when the key is absent, empty, or fails URL validation.

    Notes
    -----
    Developer: Returns ``""`` (not ``None``) so callers can use ``or ""``
    without an additional ``None`` guard.  The JS widget treats ``""`` as
    "feature disabled" consistently.

    Developer: Emits a Sphinx WARNING (not an error) when the value is
    set but invalid — the build still completes so operators are not
    blocked, but the diagnostic is visible in the console and CI logs.

    **SSRF warning**: When the URL targets a private IP range or a
    reserved hostname (``localhost``, ``*.local``, ``127.*``,
    ``10.*``, etc.), an additional WARNING is emitted to alert operators
    that the endpoint is only reachable from the documentation server
    itself, not from end-user browsers.  The URL is still returned so
    that local-development Ollama / dev-proxy setups are not broken.

    Accepted schemes: ``https://`` and ``http://`` (localhost is valid for
    local development).  Rejected: ``javascript:``, ``data:``, ``ftp:``,
    relative paths, and any other non-HTTP scheme.

    Examples
    --------
    >>> class _Cfg:
    ...     ai_assistant_panel_feedback_endpoint = (
    ...         "https://proxy.example.com/v1/feedback"
    ...     )
    >>> _cfg_endpoint_url(_Cfg(), "ai_assistant_panel_feedback_endpoint")
    'https://proxy.example.com/v1/feedback'
    >>> class _Bad:
    ...     ai_assistant_panel_feedback_endpoint = "javascript:alert(1)"
    >>> _cfg_endpoint_url(_Bad(), "ai_assistant_panel_feedback_endpoint")
    ''
    """
    raw = _cfg_str(config, key) or ""
    if not raw:
        return ""
    stripped = raw.strip()
    if not stripped:
        return ""
    if not _URL_SCHEME_RE.match(stripped):
        _get_logger().warning(
            "AI Assistant: %r = %r is not a valid https:// or "
            "http:// URL and will be ignored.  Accepted schemes: https, http.",
            key,
            stripped,
        )
        return ""
    # SSRF / private-IP advisory warning (does NOT reject the URL).
    if _check_private_ip_url(stripped):
        _get_logger().warning(
            "AI Assistant: %r = %r targets a private IP or reserved hostname.  "
            "End-user browsers cannot reach this address.  "
            "This is expected for local-development proxies; set a public URL "
            "for production deployments.",
            key,
            stripped,
        )
    return stripped


def _validate_profile(raw: Any, key: str) -> dict:  # noqa: PLR0912
    r"""Validate and normalise a single endpoint profile dict.

    Parameters
    ----------
    raw : Any
        The raw profile value from ``conf.py``.  Must be a ``dict``; all
        URL fields are validated with the same logic as
        :func:`_cfg_endpoint_url`; token and integer fields are sanitised.
    key : str
        Profile key (e.g. ``"cf"``); used only in warning messages.

    Returns
    -------
    dict
        Normalised profile with every expected key present.  Invalid URL
        fields are replaced with ``""`` and a Sphinx WARNING is emitted so
        the build still completes but the operator sees actionable output.
        Returns an empty ``{}`` when ``raw`` is not a ``dict`` or when the
        profile key itself is a prototype-pollution sentinel.

    Notes
    -----
    Developer: Intentionally lenient — an invalid URL in one field does
    not discard the whole profile.  This allows a profile like::

        "dev": {"chat": "https://ok.example.com", "share": "javascript:bad"}

    to be used with a working ``chat`` endpoint while emitting a warning
    for the invalid ``share`` URL.

    Developer: URL validation reuses the existing ``_URL_SCHEME_RE``
    pattern that is already used by ``_filter_share_targets`` and
    ``_cfg_endpoint_url``.  No new regex is introduced for URL matching.

    **Security hardening (v2)**:

    * ``_DANGEROUS_PROFILE_KEYS`` check prevents a malicious or
      misconfigured ``conf.py`` from injecting ``__proto__`` into the
      ``window.AI_ASSISTANT_ENDPOINTS`` object, which would pollute
      ``Object.prototype`` in every browser tab.

    * Token fields containing ``<``, ``>``, ``script``, or
      ``javascript:`` are rejected (XSS guard) because tokens flow
      into ``Authorization: Bearer`` headers constructed by JS string
      concatenation — an injected ``\n`` or tag could break the header
      or, in some poorly-written proxy code, inject a new HTTP header.

    * Label fields are length-capped at ``_MAX_PROFILE_LABEL_LEN``
      characters to prevent layout overflow attacks in the profile-
      switcher UI.

    * The returned dict includes ``_schemaV`` (integer) so the JS
      registry can detect and discard stale localStorage caches after
      a schema bump.

    * The returned dict includes ``_warn`` (list[str]) when any URL
      field triggered a private-IP SSRF warning.  The JS UI renders a
      ⚠ badge on profiles with a non-empty ``_warn`` list.

    Developer: URL validation reuses the existing ``_URL_SCHEME_RE``
    pattern that is already used by ``_filter_share_targets`` and
    ``_cfg_endpoint_url``.  No new regex is introduced.

    Examples
    --------
    >>> _validate_profile({"chat": "https://proxy.hf.space", "label": "HF"}, "hf")
    {'label': 'HF', 'chat': 'https://proxy.hf.space', 'share': '', ...}
    >>> _validate_profile("not-a-dict", "bad")
    {}
    >>> _validate_profile({"chat": "https://ok.example.com"}, "__proto__")
    {}
    """
    _URL_KEYS = ("chat", "share", "feedback", "training")  # noqa: N806
    _TOKEN_KEYS = ("shareToken", "feedbackToken")  # noqa: N806
    _INT_KEYS = ("ttlDays",)  # noqa: N806

    # ── Prototype-pollution guard ─────────────────────────────────────────
    # A ``conf.py`` author who wrote ``ai_assistant_endpoint_profiles =
    # {"__proto__": {...}}`` would inject a dangerous property into
    # ``window.AI_ASSISTANT_ENDPOINTS``.  Silently reject these keys.
    if key in _DANGEROUS_PROFILE_KEYS:
        _get_logger().warning(
            "AI Assistant: endpoint profile key %r is a reserved JavaScript "
            "prototype property and cannot be used as a profile name — ignored.  "
            "Choose a plain alphanumeric key such as 'cf', 'hf', or 'dev'.",
            key,
        )
        return {}

    if not isinstance(raw, dict):
        _get_logger().warning(
            "AI Assistant: endpoint profile %r must be a dict, got %s — ignored.",
            key,
            type(raw).__name__,
        )
        return {}

    # ── Label sanitisation ────────────────────────────────────────────────
    raw_label = str(raw.get("label", key))
    if len(raw_label) > _MAX_PROFILE_LABEL_LEN:
        _get_logger().warning(
            "AI Assistant: endpoint profile %r label is %d characters; "
            "truncating to %d.  Shorten the label in conf.py.",
            key,
            len(raw_label),
            _MAX_PROFILE_LABEL_LEN,
        )
        raw_label = raw_label[:_MAX_PROFILE_LABEL_LEN]

    result: dict = {
        "label": raw_label,
        "_schemaV": _PROFILE_SCHEMA_VERSION,
    }
    _warnings: list[str] = []

    # ── URL field validation ──────────────────────────────────────────────
    for url_key in _URL_KEYS:
        raw_url = raw.get(url_key, "")
        if not raw_url:
            result[url_key] = ""
            continue
        stripped = str(raw_url).strip().rstrip("/")
        if not stripped:
            result[url_key] = ""
            continue
        if not _URL_SCHEME_RE.match(stripped):
            _get_logger().warning(
                "AI Assistant: endpoint profile %r field %r = %r is not a "
                "valid https:// or http:// URL and will be ignored.  "
                "Accepted schemes: https, http.",
                key,
                url_key,
                stripped,
            )
            result[url_key] = ""
            continue
        # SSRF / private-IP advisory.
        if _check_private_ip_url(stripped):
            _get_logger().warning(
                "AI Assistant: endpoint profile %r field %r = %r targets a "
                "private IP or reserved hostname.  End-user browsers cannot "
                "reach this address outside the documentation server's network.  "
                "This is expected for local development; use a public URL for "
                "production.",
                key,
                url_key,
                stripped,
            )
            _warnings.append(url_key)
        result[url_key] = stripped

    # ── Token field validation ────────────────────────────────────────────
    for tok_key in _TOKEN_KEYS:
        val = raw.get(tok_key, "")
        if val:
            tok_str = str(val)
            # Reject tokens containing HTML/script injection attempts.
            if _TOKEN_INJECT_RE.search(tok_str):
                _get_logger().warning(
                    "AI Assistant: endpoint profile %r token field %r contains "
                    "characters that could enable XSS injection ('<', '>', "
                    "'script', 'javascript:') — field cleared.  "
                    "Tokens must be plain Bearer credential strings.",
                    key,
                    tok_key,
                )
                result[tok_key] = ""
            else:
                result[tok_key] = tok_str
        else:
            result[tok_key] = ""

    # ── Integer field validation ──────────────────────────────────────────
    for int_key in _INT_KEYS:
        raw_int = raw.get(int_key, 0)
        try:
            result[int_key] = max(0, int(float(raw_int)))
        except (TypeError, ValueError):
            result[int_key] = 0

    # ── Inject SSRF warning list (consumed by JS UI) ──────────────────────
    # An empty list serialises as [] and the JS skips badge rendering.
    result["_warn"] = _warnings

    return result


def _serialize_endpoint_profiles(config: Any) -> tuple:
    """Read, validate, and serialise the endpoint profile registry.

    Parameters
    ----------
    config : Any
        Sphinx config object (or mock in tests).

    Returns
    -------
    profiles : dict
        Validated profile registry keyed by profile name, ready for JSON
        serialisation into ``window.AI_ASSISTANT_ENDPOINTS``.  Includes a
        top-level ``_meta`` sub-dict with ``schemaVersion`` and
        ``buildId`` fields for cache-busting.
    default_key : str
        The profile key to activate on first page load.  Empty string
        when no profiles are defined.

    Notes
    -----
    Developer: This function has three code paths:

    **Path 1 — Explicit profiles** (``ai_assistant_endpoint_profiles`` set):
    Profiles are validated via :func:`_validate_profile` and the default
    key is verified.  A mismatch warning is emitted when the configured
    default does not match any key; the first profile is used instead.

    **Path 2 — Auto-profile from legacy flat keys**:
    When ``ai_assistant_endpoint_profiles`` is an empty dict or absent
    *and* at least one legacy flat key is non-empty
    (``ai_assistant_panel_feedback_endpoint`` /
    ``ai_assistant_global_share_endpoint`` /
    ``ai_assistant_training_endpoint``), this function synthesises a
    ``"default"`` profile from those keys.

    This ensures ``_EP.resolve(feature)`` always works in JavaScript
    regardless of whether the operator has migrated to profiles.  The JS
    widget never reads ``cfg.panelFeedbackEndpoint`` directly for
    the live path — it always calls ``_EP.resolve('feedback')``.

    **Path 3 — No configuration**: Returns ``({}, "")``.  The script
    block is NOT injected into the page and the JS widget falls back to
    ``cfg.panel*Endpoint`` reads (full backward compatibility with
    deployments that will never use profiles).

    **Security hardening (v2)**:

    * Dangerous profile keys (``__proto__``, ``constructor``, etc.) are
      filtered out at the registry level even if :func:`_validate_profile`
      somehow returned a result (defence in depth).
    * A WARNING is emitted when the number of profiles exceeds
      ``_MAX_PROFILE_COUNT``, which typically signals a conf.py bug
      (accidental dict comprehension building hundreds of profiles).
    * A ``_meta`` top-level key is injected into the returned dict.
      The JS registry uses ``_meta.schemaVersion`` to detect stale
      ``localStorage`` caches after a schema upgrade.  ``_meta.buildId``
      is the ISO-8601 UTC date of the Sphinx build (seconds precision)
      and acts as a cache-bust signal for rolling deployments.
    * ``_meta`` is serialised as a plain nested dict; it does NOT
      conflict with profile keys because profile names may not begin
      with ``_`` (the validation step rejects leading-underscore keys
      to prevent collisions with internal metadata).

    Examples
    --------
    >>> class _Cfg:
    ...     ai_assistant_endpoint_profiles = {"cf": {"chat": "https://cf.example.com"}}
    ...     ai_assistant_endpoint_default_profile = "cf"
    >>> profiles, default = _serialize_endpoint_profiles(_Cfg())
    >>> default
    'cf'
    >>> profiles["_meta"]["schemaVersion"]
    2
    """
    raw_dict = getattr(config, "ai_assistant_endpoint_profiles", None)
    default_key = _cfg_str(config, "ai_assistant_endpoint_default_profile") or ""

    # ── Build-time metadata injected into the window object ───────────────
    _meta: dict = {
        "schemaVersion": _PROFILE_SCHEMA_VERSION,
        "buildId": (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        ),
    }

    # ── Path 1: explicit profiles ─────────────────────────────────────────
    if isinstance(raw_dict, dict) and raw_dict:
        profiles: dict = {}

        # Warn if an unusually large number of profiles are defined.
        if len(raw_dict) > _MAX_PROFILE_COUNT:
            _get_logger().warning(
                "AI Assistant: ai_assistant_endpoint_profiles defines %d "
                "profiles (limit is %d before this warning).  This is "
                "unusual — verify that conf.py is not constructing profiles "
                "in a loop.  All profiles are baked into every HTML page.",
                len(raw_dict),
                _MAX_PROFILE_COUNT,
            )

        for prof_key, prof_val in raw_dict.items():
            str_key = str(prof_key)

            # Defence-in-depth: strip dangerous keys even if _validate_profile
            # returned a non-empty dict (should not happen, but be safe).
            if str_key in _DANGEROUS_PROFILE_KEYS:
                _get_logger().warning(
                    "AI Assistant: endpoint profile key %r is a reserved "
                    "JavaScript property name and has been removed from the "
                    "registry to prevent prototype-pollution attacks.",
                    str_key,
                )
                continue

            # Reject leading-underscore keys to protect the _meta namespace.
            if str_key.startswith("_"):
                _get_logger().warning(
                    "AI Assistant: endpoint profile key %r begins with '_', "
                    "which is reserved for internal metadata.  Rename the "
                    "profile key to avoid conflicts with _meta and _schemaV.",
                    str_key,
                )
                continue

            validated = _validate_profile(prof_val, str_key)
            if validated:
                profiles[str_key] = validated

        if not profiles:
            return {}, ""

        if default_key and default_key not in profiles:
            _get_logger().warning(
                "AI Assistant: ai_assistant_endpoint_default_profile = %r "
                "does not match any key in ai_assistant_endpoint_profiles "
                "(%s).  Falling back to the first defined profile.",
                default_key,
                ", ".join(repr(k) for k in profiles),
            )
            default_key = next(iter(profiles))
        elif not default_key:
            default_key = next(iter(profiles))

        profiles["_meta"] = _meta
        return profiles, default_key

    # ── Path 2: auto-profile from legacy flat keys ────────────────────────
    fb_url = _cfg_endpoint_url(config, "ai_assistant_panel_feedback_endpoint") or ""
    sh_url = _cfg_endpoint_url(config, "ai_assistant_global_share_endpoint") or ""
    tr_url = _cfg_endpoint_url(config, "ai_assistant_training_endpoint") or ""
    fb_tok = _cfg_str(config, "ai_assistant_panel_feedback_token") or ""
    sh_tok = _cfg_str(config, "ai_assistant_global_share_token") or ""
    ttl = _cfg_int(config, "ai_assistant_global_share_ttl_days", 30)

    if not (fb_url or sh_url or tr_url):
        return {}, ""

    # Derive a chat base from the first non-empty URL (strip known route suffix).
    chat_base = sh_url or fb_url or tr_url
    for suffix in ("/v1/share", "/v1/feedback", "/v1/contribute"):
        if chat_base.endswith(suffix):
            chat_base = chat_base[: -len(suffix)]
            break

    # Validate tokens from legacy flat keys.
    def _scrub_token(tok: str) -> str:
        if _TOKEN_INJECT_RE.search(tok):
            _get_logger().warning(
                "AI Assistant: a legacy token value contains injection "
                "characters and has been cleared."
            )
            return ""
        return tok

    auto_profile: dict = {
        "label": "Default",
        "chat": chat_base,
        "share": sh_url,
        "feedback": fb_url,
        "training": tr_url,
        "shareToken": _scrub_token(sh_tok),
        "feedbackToken": _scrub_token(fb_tok),
        "ttlDays": ttl,
        "_schemaV": _PROFILE_SCHEMA_VERSION,
        "_warn": [],
    }
    result = {"default": auto_profile, "_meta": _meta}
    return result, "default"


def _inspect_profiles(profiles: dict) -> dict:
    """Return a machine-readable capability and risk summary for all profiles.

    Parameters
    ----------
    profiles : dict
        A validated profile registry as returned by
        :func:`_serialize_endpoint_profiles` (the ``profiles`` element of
        the returned tuple, which may include a ``_meta`` key).

    Returns
    -------
    dict
        A ``{profile_key: summary}`` mapping where each summary has the
        shape::

            {
                "label": str,
                "caps": {
                    "chat": bool,
                    "share": bool,
                    "feedback": bool,
                    "training": bool,
                },
                "hasToken": {"share": bool, "feedback": bool},
                "warns": list[str],  # URL fields with SSRF warnings
                "schemaV": int,
            }

        The ``_meta`` key is excluded from the output.

    Notes
    -----
    Developer: This helper is used by the compare-grid in
    ``_buildEndpointConfigSheet`` (via JSON injection) and by the test
    suite to assert capability coverage without parsing the full profile
    dict.

    Developer: ``caps`` values are ``True`` when the URL field is a non-
    empty string after validation.  A ``True`` value does NOT guarantee
    reachability — use the health-check feature in the UI for that.

    Examples
    --------
    >>> profiles = {
    ...     "cf": {
    ...         "label": "CF",
    ...         "chat": "https://cf.example.com",
    ...         "share": "",
    ...         "feedback": "",
    ...         "training": "",
    ...         "shareToken": "",
    ...         "feedbackToken": "",
    ...         "_schemaV": 2,
    ...         "_warn": [],
    ...     },
    ...     "_meta": {"schemaVersion": 2, "buildId": "2025-01-01T00:00:00Z"},
    ... }
    >>> summary = _inspect_profiles(profiles)
    >>> summary["cf"]["caps"]["chat"]
    True
    >>> summary["cf"]["caps"]["share"]
    False
    """
    result: dict = {}
    for key, profile in profiles.items():
        if key == "_meta":
            continue  # skip internal metadata
        if not isinstance(profile, dict):
            continue
        result[key] = {
            "label": str(profile.get("label", key)),
            "caps": {
                "chat": bool(profile.get("chat", "")),
                "share": bool(profile.get("share", "")),
                "feedback": bool(profile.get("feedback", "")),
                "training": bool(profile.get("training", "")),
            },
            "hasToken": {
                "share": bool(profile.get("shareToken", "")),
                "feedback": bool(profile.get("feedbackToken", "")),
            },
            "warns": list(profile.get("_warn", [])),
            "schemaV": int(profile.get("_schemaV", 0)),
        }
    return result


def _resolve_feedback_scale_safe(
    scale_name: str,
    options: list,
) -> tuple[int, ...]:
    """Resolve the feedback scale without raising — never breaks a Sphinx build.

    Parameters
    ----------
    scale_name : str
        Value of ``ai_assistant_panel_feedback_scale``.  ``"auto"`` selects by
        the number of options actually configured.
    options : list
        Value of ``ai_assistant_panel_feedback_options`` (raw, may be empty).
        Empty / shorter-than-2 triggers the documented JS-side 10-emoji
        default, so we resolve the scale against ``num=10`` in that case to
        keep the parallel-list invariant
        (``len(panelFeedbackScale) == len(panelFeedbackOptions)``) holding
        from the widget's point of view.

    Returns
    -------
    tuple of int
        Always returns a usable scale; on any ValueError from the strict
        resolver it logs a warning and falls back to a generated symmetric
        scale of length *n* (so the parallel-list invariant is always
        maintained regardless of the error source).

    Notes
    -----
    The strict resolver :func:`_resolve_feedback_scale` is correct and
    raises on bad input.  Sphinx config validation, however, should fail
    closed for the user-facing widget but never abort the documentation
    build.  This wrapper provides that boundary.
    """
    # Compute n outside the try block so the except clause can use it to
    # generate a correctly-sized fallback scale, maintaining the invariant
    # len(panelFeedbackScale) == len(panelFeedbackOptions) even on error.
    n = (
        len(options)
        if isinstance(options, (list, tuple)) and len(options) >= 2  # noqa: PLR2004
        else 10  # JS default count (changed from 3 → 10)
    )
    try:
        # Mirror the JS-side default: ``< 2`` options ⇒ 10-emoji default,
        # so we must size the scale against 10, not against ``len(options)``.
        return _resolve_feedback_scale(scale_name, n)
    except ValueError as exc:
        _get_logger().warning(
            f"AI Assistant: invalid feedback scale {scale_name!r} "
            f"with {len(options) if hasattr(options, '__len__') else '?'} "
            f"options ({exc}); falling back to generated symmetric scale "
            f"of length {n}."
        )
        return _generate_symmetric_scale(n)


def add_ai_assistant_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    """Inject AI-assistant configuration into each HTML page's template context.

    Registered with Sphinx's ``html-page-context`` event in :func:`setup`.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The active Sphinx application instance.
    pagename : str
        The logical page name (e.g. ``"index"``).
    templatename : str
        The Jinja2 template name (e.g. ``"page.html"``).
    context : dict
        The template rendering context; modified in place.
    doctree : docutils.nodes.document or None
        The parsed document tree for the page.

    Returns
    -------
    None

    Notes
    -----
    **Security**: Serialised via :func:`_safe_json_for_script` (XSS guard).
    Invalid position → fallback to ``"sidebar"`` with a logged warning.
    Providers with dangerous ``url_template`` schemes are filtered out.
    """
    if not app.config.ai_assistant_enabled:
        return

    raw_position = str(app.config.ai_assistant_position)
    try:
        position_val = _validate_position(raw_position)
    except ValueError:
        _get_logger().warning(
            f"AI Assistant: Invalid ai_assistant_position {raw_position!r}; "
            f"falling back to 'sidebar'. "
            f"Accepted values: {sorted(_ALLOWED_POSITIONS)}"
        )
        position_val = "sidebar"

    providers_raw: dict[str, Any] = dict(app.config.ai_assistant_providers)
    providers_safe = _filter_providers(providers_raw)

    # Resolve icons: replace missing SVG filenames with inline base64 data URIs
    # so the widget renders correctly even when optional SVG files are absent.
    _static_dir = Path(__file__).parent / "_static"
    providers_resolved: dict[str, Any] = {
        name: {
            **prov,
            "icon": _resolve_icon(str(prov.get("icon", "")), name, _static_dir),
        }
        for name, prov in providers_safe.items()
    }
    mcp_tools_resolved: dict[str, Any] = {
        name: {
            **tool,
            "icon": _resolve_icon(str(tool.get("icon", "")), name, _static_dir),
        }
        for name, tool in dict(app.config.ai_assistant_mcp_tools).items()
    }

    # ---- Feature-flag merge ------------------------------------------------
    # Merge the user's partial ``ai_assistant_features`` dict OVER the full
    # ``_DEFAULT_FEATURES`` baseline so every key is always present in the
    # serialised JSON.
    #
    # Root cause of BUG-1 (AI panel button invisible):
    #   ``dict(app.config.ai_assistant_features)`` emits only the keys the
    #   user wrote in conf.py.  When ``ai_panel`` is absent the widget JS
    #   falls back to ``FEATURE_DEFAULTS.ai_panel = false``, which means the
    #   AI-panel button is never created, so clicking it is impossible.
    #
    # Fix: start from ``_DEFAULT_FEATURES`` (all keys, correct defaults) and
    # update with user values so user overrides win and gaps are filled.
    features_merged: dict[str, bool] = dict(_DEFAULT_FEATURES)
    features_merged.update(
        {k: bool(v) for k, v in dict(app.config.ai_assistant_features).items()}
    )

    config: dict[str, Any] = {
        "position": position_val,
        "content_selector": app.config.ai_assistant_content_selector,
        # Always include every feature flag — partial conf.py dicts are safe.
        "features": features_merged,
        "providers": providers_resolved,
        "mcp_tools": mcp_tools_resolved,
        "baseUrl": (
            getattr(app.config, "html_baseurl", "")
            or app.config.ai_assistant_base_url
            or ""
        ),
        # Prompt customisation (forwarded to widget JS).
        # _cfg_str / _cfg_bool guard against MagicMock in test environments.
        "intention": _cfg_str(app.config, "ai_assistant_intention"),
        "customContext": _cfg_str(app.config, "ai_assistant_custom_context"),
        "customPromptPrefix": _cfg_str(app.config, "ai_assistant_custom_prompt_prefix"),
        "includeRawImage": _cfg_bool(
            app.config, "ai_assistant_include_raw_image", False
        ),
        # ---- PDF export ---------------------------------------------------
        # Empty string means "use window.print()"; a non-empty value is opened
        # in a new tab as the PDF download URL.
        "pdfExportUrl": _cfg_str(app.config, "ai_assistant_pdf_export_url") or "",
        # Controls whether the URL/Print toggle row is shown below the button.
        # True  → show toggle (default); user can switch between URL and Print.
        # False → hide toggle; button always uses the mode inferred from pdfExportUrl.
        "pdfUrlModeToggle": _cfg_bool(
            app.config, "ai_assistant_pdf_url_mode_toggle", True
        ),
        # ---- AI panel (floating chat drawer) --------------------------------
        # Basic identity
        "panelTitle": (
            _cfg_str(app.config, "ai_assistant_panel_title") or "AI Assistant"
        ),
        "panelPlaceholder": (
            _cfg_str(app.config, "ai_assistant_panel_placeholder")
            or "Ask a question about this page\u2026"
        ),
        "panelApiEnabled": _cfg_bool(
            app.config, "ai_assistant_panel_api_enabled", False
        ),
        # Quick-suggestion chips (list[str], 0-5 items shown).
        # Serialise via _cfg_str_list so the value is always a JSON array
        # in the injected config even when conf.py omits the key.
        "panelQuickQuestions": _cfg_str_list(
            app.config, "ai_assistant_panel_quick_questions"
        ),
        # "Speak with your assistant" microphone banner / button.
        # When True (default) and the browser supports Web Speech API the
        # panel shows a dismissable "Speak with your assistant" pill above
        # the input area, plus a mic icon inside the input group.
        # Set to False to hide all speech UI regardless of browser support.
        "panelSpeakBanner": _cfg_bool(
            app.config, "ai_assistant_panel_speak_banner", True
        ),
        # Trigger pill label (the floating "Ask Us" button shown when minimized).
        "panelTriggerLabel": (
            _cfg_str(app.config, "ai_assistant_panel_trigger_label") or "Ask Us"
        ),
        # Initial panel state: True (default) → show trigger pill on page load
        # (panel starts minimized, pill visible — 1-click access).
        # False → pill hidden; panel opens only via the dropdown button.
        # Controls whether createAIAssistantUI() eagerly creates the trigger
        # pill before the user has ever opened the panel.
        "panelStartMinimized": _cfg_bool(
            app.config, "ai_assistant_panel_start_minimized", True
        ),
        # ---- v0.3 keys ------------------------------------------------------
        # Each maps 1:1 to a window.AI_ASSISTANT_CONFIG.* read in
        # ai-assistant.js.  Keys use the JS camelCase the reader expects.
        #
        # R-persist: sessionStorage transcript (default True).
        "panelPersist": _cfg_bool(app.config, "ai_assistant_panel_persist", True),
        # R7: keyboard shortcut chord (str; "" disables).
        "panelShortcut": (
            _cfg_str(app.config, "ai_assistant_panel_shortcut")
            if _cfg_str(app.config, "ai_assistant_panel_shortcut") is not None
            else "Alt+Shift+A"
        ),
        # C-2: proxy endpoint for API mode (empty → actionable JS error).
        "panelApiUrl": _cfg_str(app.config, "ai_assistant_panel_api_url") or "",
        "panelApiModel": _cfg_str(app.config, "ai_assistant_panel_api_model") or "",
        # ── Phase B: multi-model panel ────────────────────────────────────
        # Always emit a list (possibly empty).  When empty the widget falls
        # back to the legacy single-string panelApiModel above.  Each entry is
        # already shape-normalised, security-validated, and de-duplicated by
        # ``_filter_panel_models`` — the JS does NOT re-validate.
        "panelApiModels": _filter_panel_models(
            _cfg_list(app.config, "ai_assistant_panel_api_models")
            or _cfg_str(app.config, "ai_assistant_panel_api_models")
        ),
        # Provider → accent-colour map forwarded to JS for badge rendering.
        # Merged with JS-side defaults so either side can extend without
        # the other breaking.
        "providerColors": dict(_PROVIDER_COLORS),
        # SSE streaming master switch (bool, default True).
        # When True, the JS enables SSE streaming for every provider in
        # _STREAMING_PROVIDERS.  Set False on PaaS hosts that buffer SSE
        # frames (some coalesce the entire stream into one response, which
        # breaks the incremental render loop).
        # JS read: var streamingEnabled = (cfg.panelApiStreaming !== false)
        "panelApiStreaming": _cfg_bool(
            app.config, "ai_assistant_panel_api_streaming", True
        ),
        # Inline picker beside mic + send (Claude-style bar).
        "panelInlineModelPicker": _cfg_bool(
            app.config, "ai_assistant_panel_inline_model_picker", True
        ),
        # ── Phase B: Terms of Service sheet ────────────────────────────────
        "panelTerms": _cfg_bool(app.config, "ai_assistant_panel_terms", True),
        "panelTermsTitle": (
            _cfg_str(app.config, "ai_assistant_panel_terms_title") or "Terms of Service"
        ),
        "panelTermsLinkText": (
            _cfg_str(app.config, "ai_assistant_panel_terms_link_text")
            or "Terms of Service"
        ),
        # Trusted author HTML (from conf.py, not end-user input).
        "panelTermsHtml": _cfg_str(app.config, "ai_assistant_panel_terms_html") or "",
        # ── Phase B: Share sheet ───────────────────────────────────────────
        "panelShare": _cfg_bool(app.config, "ai_assistant_panel_share", True),
        "panelShareLabel": (
            _cfg_str(app.config, "ai_assistant_panel_share_label") or "Share"
        ),
        # Filter share targets through the same URL allow-list used for
        # AI providers — javascript:/data:/ftp: schemes are rejected.
        # Empty user list ⇒ default registry; we never ship JS without
        # targets so the share button is always functional when enabled.
        "panelShareTargets": _filter_share_targets(
            _cfg_list(app.config, "ai_assistant_panel_share_targets")
            or list(_DEFAULT_SHARE_TARGETS.items())
        ),
        # ── Project Links sheet (source repo + project website) ────────────
        # panelLinks — master switch for the Links slide-over sheet.
        # When False the sheet is not built; sourceBtn/siteBtn fall back to
        # opening their URL directly in a new tab.
        "panelLinks": _cfg_bool(app.config, "ai_assistant_panel_links", True),
        "panelLinksTitle": (
            _cfg_str(app.config, "ai_assistant_panel_links_title") or "Project Links"
        ),
        # Trusted author HTML (from conf.py, not end-user input).  Empty →
        # the built-in two-card layout is rendered.
        "panelLinksHtml": _cfg_str(app.config, "ai_assistant_panel_links_html") or "",
        # Source (GitHub) button — left subbar cluster.
        "panelSource": _cfg_bool(app.config, "ai_assistant_panel_source", True),
        "panelSourceUrl": _cfg_str(app.config, "ai_assistant_panel_source_url") or "",
        "panelSourceLabel": (
            _cfg_str(app.config, "ai_assistant_panel_source_label")
            or "Source Repository"
        ),
        "panelSourceDesc": _cfg_str(app.config, "ai_assistant_panel_source_desc") or "",
        "panelSourceBtnLabel": (
            _cfg_str(app.config, "ai_assistant_panel_source_btn_label") or "Source"
        ),
        # Site (website) button — right subbar cluster, after Share.
        "panelSite": _cfg_bool(app.config, "ai_assistant_panel_site", True),
        "panelSiteUrl": _cfg_str(app.config, "ai_assistant_panel_site_url") or "",
        "panelSiteLabel": (
            _cfg_str(app.config, "ai_assistant_panel_site_label") or "Project Website"
        ),
        "panelSiteDesc": _cfg_str(app.config, "ai_assistant_panel_site_desc") or "",
        "panelSiteBtnLabel": (
            _cfg_str(app.config, "ai_assistant_panel_site_btn_label") or "Website"
        ),
        # ── Phase B: Hamburger overflow menu ──────────────────────────────
        "panelHamburger": _cfg_bool(app.config, "ai_assistant_panel_hamburger", True),
        # R5: feedback block.
        "panelFeedback": _cfg_bool(app.config, "ai_assistant_panel_feedback", True),
        "panelFeedbackQuestion": (
            _cfg_str(app.config, "ai_assistant_panel_feedback_question")
            or "Was this helpful?"
        ),
        # list[dict] preserved via _cfg_list (NOT _cfg_str_list).
        "panelFeedbackOptions": _cfg_list(
            app.config, "ai_assistant_panel_feedback_options"
        ),
        # Signed-integer scale paralleling panelFeedbackOptions.
        # Resolved server-side so the JS widget receives a ready-to-use list
        # of ints (no client-side branching, no risk of mis-derivation).
        # The number of emoji options is computed AFTER applying the JS-side
        # default-of-10 fallback (empty conf list ⇒ 10-emoji default), so the
        # serialised scale length always matches what the user sees.
        "panelFeedbackScale": list(
            _resolve_feedback_scale_safe(
                _cfg_str(app.config, "ai_assistant_panel_feedback_scale") or "auto",
                _cfg_list(app.config, "ai_assistant_panel_feedback_options"),
            )
        ),
        "panelFeedbackScaleName": (
            _cfg_str(app.config, "ai_assistant_panel_feedback_scale") or "auto"
        ),
        "panelFeedbackPlaceholder": (
            _cfg_str(app.config, "ai_assistant_panel_feedback_placeholder") or ""
        ),
        "panelFeedbackSubmit": (
            _cfg_str(app.config, "ai_assistant_panel_feedback_submit")
            or "Send feedback"
        ),
        "panelFeedbackThanks": (
            _cfg_str(app.config, "ai_assistant_panel_feedback_thanks")
            or "Thanks for your feedback!"
        ),
        "panelFeedbackLog": _cfg_bool(
            app.config, "ai_assistant_panel_feedback_log", False
        ),
        # R2: privacy / responsibility sheet.
        "panelPrivacyTitle": (
            _cfg_str(app.config, "ai_assistant_panel_privacy_title")
            or "Privacy & Responsibility"
        ),
        "panelPrivacyLinkText": (
            _cfg_str(app.config, "ai_assistant_panel_privacy_link_text")
            or "Privacy & Responsibility"
        ),
        # Trusted author HTML (from conf.py, not end-user input).
        "panelPrivacyHtml": (
            _cfg_str(app.config, "ai_assistant_panel_privacy_html") or ""
        ),
        # R8: standalone AI search-bar (opt-in; default off).
        "searchBar": _cfg_bool(app.config, "ai_assistant_search_bar", False),
        "searchBarSelector": (
            _cfg_str(app.config, "ai_assistant_search_bar_selector") or ""
        ),
        "searchBarMini": _cfg_bool(app.config, "ai_assistant_search_bar_mini", False),
        # Insertion point inside the host element: "top" → prepend (sidebar
        # top, above navigation links), "bottom" → append (default, current
        # behaviour).  Any value other than "top" is treated as "bottom" so
        # the safe fallback is always the pre-existing behaviour.
        "searchBarPosition": (
            _cfg_str(app.config, "ai_assistant_search_bar_position") or "top"
        ),
        "panelSearchPlaceholder": (
            _cfg_str(app.config, "ai_assistant_panel_search_placeholder")
            or "Ask AI about these docs\u2026"
        ),
        # ── Feedback POST endpoint (P1) ────────────────────────────────────
        # Empty string "" disables the feature — zero behavior change for
        # operators who have not configured an endpoint.
        "panelFeedbackEndpoint": _cfg_endpoint_url(
            app.config, "ai_assistant_panel_feedback_endpoint"
        ),
        # Token sent as Authorization: Bearer <token>.  Empty string = no header.
        # Operators must read this from os.environ — never hardcode in conf.py.
        "panelFeedbackToken": (
            _cfg_str(app.config, "ai_assistant_panel_feedback_token") or ""
        ),
        # ── Global share (P2) ─────────────────────────────────────────────
        "panelGlobalShareEndpoint": _cfg_endpoint_url(
            app.config, "ai_assistant_global_share_endpoint"
        ),
        "panelGlobalShareToken": (
            _cfg_str(app.config, "ai_assistant_global_share_token") or ""
        ),
        "panelGlobalShareTtlDays": _cfg_int(
            app.config, "ai_assistant_global_share_ttl_days", 30
        ),
        # ── Training contribution (P3) ─────────────────────────────────────
        "panelTrainingEndpoint": _cfg_endpoint_url(
            app.config, "ai_assistant_training_endpoint"
        ),
    }

    context["ai_assistant_config"] = config

    if "metatags" not in context:
        context["metatags"] = ""

    safe_json = _safe_json_for_script(config)
    _script = f"\n<script>\nwindow.AI_ASSISTANT_CONFIG = {safe_json};\n</script>\n"
    context["metatags"] += f"{_script}"

    # ── Endpoint Profile Registry (runtime-switchable proxy backends) ────
    # Serialised separately so the profile store can grow without bloating
    # AI_ASSISTANT_CONFIG.  Injected only when profiles are defined.
    _ep_profiles, _ep_default = _serialize_endpoint_profiles(app.config)
    if _ep_profiles:
        _ep_json = _safe_json_for_script(_ep_profiles)
        _ep_default_json = _safe_json_for_script(_ep_default)
        _ep_script = (
            "\n<script>\n"
            f"window.AI_ASSISTANT_ENDPOINTS = {_ep_json};\n"
            f"window.AI_ASSISTANT_ENDPOINT_DEFAULT = {_ep_default_json};\n"
            "</script>\n"
        )
        context["metatags"] += _ep_script


# ---------------------------------------------------------------------------
# Sphinx extension entry point
# ---------------------------------------------------------------------------


def setup(app: Sphinx) -> dict[str, Any]:
    """
    Register the AI-assistant extension with a Sphinx application.

    This is the canonical Sphinx extension entry point.  Sphinx calls it
    automatically when the extension is listed in ``conf.py extensions``.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application being configured.

    Returns
    -------
    dict
        Sphinx extension metadata::

            {
                "version": "0.2.0",
                "parallel_read_safe": True,
                "parallel_write_safe": True,
            }

    Notes
    -----
    **Config values registered** (all are *html*-rebuild-triggering):

    .. list-table::
       :header-rows: 1

       * - Name
         - Default
         - Description
       * - ``ai_assistant_enabled``
         - ``True``
         - Master switch.
       * - ``ai_assistant_position``
         - ``"sidebar"``
         - ``"sidebar"``, ``"title"``, ``"floating"``, or ``"none"``.
       * - ``ai_assistant_content_selector``
         - ``"article"``
         - CSS selector for the page's main content area (client-side).
       * - ``ai_assistant_content_selectors``
         - ``_DEFAULT_CONTENT_SELECTORS``
         - Ordered CSS selectors for server-side Markdown extraction.
       * - ``ai_assistant_theme_preset``
         - ``None``
         - Theme name to auto-select CSS selectors (e.g.
           ``"pydata_sphinx_theme"``).  Merged with
           ``ai_assistant_content_selectors``.
       * - ``ai_assistant_generate_markdown``
         - ``True``
         - Generate ``.md`` files after build.
       * - ``ai_assistant_markdown_exclude_patterns``
         - ``["genindex", "search", "py-modindex"]``
         - Path substrings to exclude.
       * - ``ai_assistant_strip_tags``
         - ``["script", "style", "nav", "footer"]``
         - HTML tags stripped (with content) before Markdown conversion.
       * - ``ai_assistant_generate_llms_txt``
         - ``True``
         - Generate ``llms.txt`` after build.
       * - ``ai_assistant_base_url``
         - ``""``
         - Base URL for ``llms.txt`` entries.
       * - ``ai_assistant_max_workers``
         - ``None``
         - Max parallel workers; ``None`` → auto (CPU count or 1).
       * - ``ai_assistant_llms_txt_max_entries``
         - ``None``
         - Cap on the number of entries in ``llms.txt`` (``None`` → all).
       * - ``ai_assistant_llms_txt_full_content``
         - ``False``
         - Embed full Markdown content in ``llms.txt``.
       * - ``ai_assistant_features``
         - (see source)
         - Feature-flag dict.
       * - ``ai_assistant_providers``
         - (see source)
         - AI provider configuration dict.
       * - ``ai_assistant_mcp_tools``
         - (see source)
         - MCP tool configuration dict.
       * - ``ai_assistant_pdf_export_url``
         - ``None``
         - URL opened by "Export as PDF" button.  ``None`` / ``""`` → calls
           ``window.print()``; any non-empty string → opens that URL in a
           new tab (e.g. a server-side PDF endpoint or GitBook-style path).
       * - ``ai_assistant_pdf_url_mode_toggle``
         - ``True``
         - Show the URL / Print toggle row below the PDF button.  ``True``
           (default) lets users switch mode interactively; ``False`` hides
           the toggle and locks behaviour to the value implied by
           ``ai_assistant_pdf_export_url``.
       * - ``ai_assistant_panel_title``
         - ``"AI Assistant"``
         - Header label shown in the floating AI chat panel.
       * - ``ai_assistant_panel_placeholder``
         - ``"Ask a question about this page…"``
         - Placeholder text for the panel's input field.
       * - ``ai_assistant_panel_api_enabled``
         - ``False``
         - When ``True`` the panel sends page context to the Anthropic API
           and streams a real response.  When ``False`` it renders as a UI
           stub with no network calls.

    Examples
    --------
    In ``conf.py``:

    .. code-block:: python

        extensions = [
            "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
        ]
        html_theme = "pydata_sphinx_theme"
        ai_assistant_enabled = True
        ai_assistant_theme_preset = "pydata_sphinx_theme"
        ai_assistant_position = "sidebar"
        ai_assistant_generate_markdown = True
        ai_assistant_generate_llms_txt = True
        html_baseurl = "https://docs.myproject.org"
    """
    # ---- Core toggles ------------------------------------------------------
    app.add_config_value("ai_assistant_enabled", True, "html")
    app.add_config_value("ai_assistant_position", "sidebar", "html")
    app.add_config_value("ai_assistant_content_selector", "article", "html")

    # ---- Content selectors (server-side Markdown extraction) ---------------
    app.add_config_value(
        "ai_assistant_content_selectors",
        list(_DEFAULT_CONTENT_SELECTORS),
        "html",
    )
    app.add_config_value("ai_assistant_theme_preset", None, "html")

    # ---- Markdown + llms.txt -----------------------------------------------
    app.add_config_value("ai_assistant_generate_markdown", True, "html")
    app.add_config_value(
        "ai_assistant_markdown_exclude_patterns",
        ["genindex", "search", "py-modindex"],
        "html",
    )
    app.add_config_value(
        "ai_assistant_strip_tags",
        ["script", "style", "nav", "footer"],
        "html",
    )
    app.add_config_value("ai_assistant_generate_llms_txt", True, "html")
    app.add_config_value("ai_assistant_base_url", "", "html")
    app.add_config_value(
        "ai_assistant_max_workers", None, "html"
    )  # None = CPU auto-detect
    app.add_config_value("ai_assistant_llms_txt_max_entries", None, "html")
    app.add_config_value("ai_assistant_llms_txt_full_content", False, "html")

    # ---- Feature flags -----------------------------------------------------
    app.add_config_value(
        "ai_assistant_features",
        {
            "markdown_export": True,
            "view_markdown": True,
            "ai_chat": True,
            "mcp_integration": True,
            "theme_toggle": True,  # dark / light / system color-scheme toggle
            "pdf_export": True,  # Export as PDF button (window.print or custom URL)
            "ai_panel": True,  # Floating AI assistant chat panel (stub / API)
        },
        "html",
    )

    # ---- AI provider configuration — defaults to full registry -------------
    app.add_config_value(
        "ai_assistant_providers",
        dict(_DEFAULT_PROVIDERS),
        "html",
    )

    # ---- Ollama / local model config helper --------------------------------
    # Expose the recommended model list so conf.py can reference it:
    #   from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
    #       _OLLAMA_RECOMMENDED_MODELS,
    #   )
    app.add_config_value(
        "ai_assistant_ollama_model",
        "llama3.2:latest",
        "html",
    )

    # ---- MCP tool configuration -------------------------------------------
    # NOTE: Use the full _DEFAULT_MCP_TOOLS registry (vscode, claude_desktop,
    # cursor, windsurf, generic) so conf.py inherits all defaults and only
    # needs to override the entries it customises.
    app.add_config_value(
        "ai_assistant_mcp_tools",
        dict(_DEFAULT_MCP_TOOLS),
        "html",
    )

    # ---- Prompt customisation config values --------------------------------
    app.add_config_value("ai_assistant_intention", None, "html")
    app.add_config_value("ai_assistant_custom_context", None, "html")
    app.add_config_value("ai_assistant_custom_prompt_prefix", None, "html")
    app.add_config_value("ai_assistant_include_raw_image", False, "html")

    # ---- PDF export config -------------------------------------------------
    # When set to a non-empty string the "Export as PDF" button opens that URL
    # (e.g. ``"/_pdf/{pagename}.pdf"`` or a GitBook-style ``"~gitbook/pdf?"``
    # endpoint) in a new tab instead of calling window.print().
    # ``None`` / empty string → trigger browser print dialog (window.print()).
    app.add_config_value("ai_assistant_pdf_export_url", None, "html")

    # When True (default) the URL/Print toggle row is rendered below the
    # "Export as PDF" button so users can switch modes interactively.
    # Set False to hide the toggle and lock to the mode implied by
    # ``ai_assistant_pdf_export_url`` (URL mode when non-empty, Print otherwise).
    app.add_config_value("ai_assistant_pdf_url_mode_toggle", True, "html")

    # ---- AI panel (floating chat drawer) config ----------------------------
    #
    # ``ai_assistant_panel_title``
    #     Header label shown in the floating AI panel.  Default: 'AI Assistant'.
    #
    # ``ai_assistant_panel_placeholder``
    #     Placeholder text for the panel's textarea input field.
    #
    # ``ai_assistant_panel_api_enabled``
    #     When True the panel POSTs to the Anthropic /v1/messages endpoint;
    #     when False the panel renders as a UI stub with no network calls.
    #
    # ``ai_assistant_panel_quick_questions``
    #     List of 0-5 short question strings shown as clickable chip buttons
    #     in the panel welcome screen.  Clicking a chip pre-fills the input.
    #     Example::
    #         ai_assistant_panel_quick_questions = [
    #             "What does this module do?",
    #             "Show me a usage example.",
    #             "What are the main parameters?",
    #         ]
    #
    # ``ai_assistant_panel_speak_banner``
    #     When True (default) and the browser supports Web Speech API, show
    #     a "Speak with your assistant" pill above the input plus a mic icon
    #     inside the input group.  Set False to hide all speech UI.
    #
    # ``ai_assistant_panel_trigger_label``
    #     Label on the floating trigger pill shown when the panel is minimized.
    #     Default: 'Ask Us'.
    #
    # ``ai_assistant_panel_start_minimized``
    #     When True (default) the floating trigger pill is rendered eagerly on
    #     every page load so users can open the panel with a single click.
    #     When False the pill only appears after the user has opened then
    #     minimized the panel, requiring them to first click the expand button
    #     then "AI Assistant".  Set to False only when you prefer the lighter
    #     initial paint at the cost of discoverability.
    app.add_config_value("ai_assistant_panel_title", "AI Assistant", "html")
    app.add_config_value(
        "ai_assistant_panel_placeholder",
        "Ask a question about this page\u2026",
        "html",
    )
    app.add_config_value("ai_assistant_panel_api_enabled", False, "html")
    app.add_config_value("ai_assistant_panel_quick_questions", [], "html")
    app.add_config_value("ai_assistant_panel_speak_banner", True, "html")
    app.add_config_value("ai_assistant_panel_trigger_label", "Ask Us", "html")
    app.add_config_value("ai_assistant_panel_start_minimized", True, "html")

    # -----------------------------------------------------------------------
    # v0.3 config values — resize, persistence, shortcut, proxy, feedback,
    # privacy sheet, AI search-bar.
    #
    # CRITICAL — every key below is read by ai-assistant.js.  This block, the
    # serialisation in ``add_ai_assistant_context`` and the JS reader form
    # the three-layer "config flows one way" invariant.  Changing one layer
    # without the other two silently disables the feature (this is the exact
    # class of defect documented as BUG-1 / C-1).
    # -----------------------------------------------------------------------

    # ``ai_assistant_panel_persist`` (bool, default True)
    #     True  → conversation is stored in sessionStorage so it survives
    #             intra-site navigation; cleared on tab close or "new chat".
    #     False → in-memory only (lost on any navigation).
    app.add_config_value("ai_assistant_panel_persist", True, "html")

    # ``ai_assistant_panel_shortcut`` (str, default "Alt+Shift+A")
    #     Keyboard chord that toggles the panel.  MUST contain at least one
    #     modifier (Ctrl/Alt/Cmd) — a bare key is rejected by the JS parser
    #     so site-wide typing is never hijacked.  Empty string disables the
    #     global listener entirely.  Examples: "Ctrl+Shift+Space", "Cmd+J".
    app.add_config_value("ai_assistant_panel_shortcut", "Alt+Shift+A", "html")

    # ``ai_assistant_panel_api_url`` (str, default "")
    #     Endpoint used in API mode.  A browser CANNOT call Anthropic
    #     directly (no CORS, key would leak), so this MUST point at the doc
    #     owner's own proxy that injects the API key server-side and forwards
    #     the Anthropic /v1/messages-shaped body.  Empty → API mode raises an
    #     actionable error instead of a silent blocked request.
    app.add_config_value("ai_assistant_panel_api_url", "", "html")

    # ``ai_assistant_panel_api_model`` (str, default "")
    #     Optional model name forwarded in the proxy request body.  Empty →
    #     JS default ("claude-sonnet-4-20250514").
    app.add_config_value("ai_assistant_panel_api_model", "", "html")

    # ── Phase B: multi-model panel ────────────────────────────────────────
    # ``ai_assistant_panel_api_models`` (str | list[str] | list[dict],
    # default [])
    #     Optional multi-model registry.  When non-empty the panel shows a
    #     model picker (dedicated sheet + optional inline picker beside the
    #     send button).  Each dict entry MUST have:
    #         id          unique stable key (also persisted in sessionStorage)
    #         provider    one of _PANEL_MODEL_PROVIDERS
    #         model       wire model name forwarded to the proxy
    #         endpoint    proxy URL (http/https or site-relative)
    #     Optional fields: label, description, info_url, default, icon.
    #
    #     SECURITY: API keys MUST NEVER appear here — only proxy endpoint
    #     URLs.  The browser never sees a key.  See _example_conf.py for the
    #     ``os.environ`` + GitHub-Actions-secrets pattern that keeps keys
    #     on the proxy side only.
    app.add_config_value("ai_assistant_panel_api_models", [], "html")

    # ``ai_assistant_panel_api_streaming`` (bool, default True)
    #     Master switch for server-sent event (SSE) streaming in the panel.
    #     When True (default), the JS sends ``stream: true`` to every provider
    #     listed in the JS ``_STREAMING_PROVIDERS`` array and renders tokens
    #     incrementally as they arrive.
    #     Set ``False`` on hosting platforms that buffer SSE frames before
    #     forwarding them to the browser (some PaaS reverse proxies coalesce
    #     the entire stream into a single HTTP response, which causes the UI to
    #     hang until the model finishes generating).
    #     Note: Anthropic-provider models always use the non-streaming path
    #     regardless of this flag because their SSE format differs from the
    #     OpenAI-compat spec that the streaming loop expects.
    app.add_config_value("ai_assistant_panel_api_streaming", True, "html")

    # ``ai_assistant_panel_inline_model_picker`` (bool, default True)
    #     When True and panelApiModels is non-empty the panel footer renders
    #     an inline model picker beside the mic and send buttons (Claude-bar
    #     style).  Set False to keep the picker accessible only through the
    #     dedicated sheet button in the sub-bar.
    app.add_config_value("ai_assistant_panel_inline_model_picker", True, "html")

    # ── Phase B: Terms of Service sheet ───────────────────────────────────
    # ``ai_assistant_panel_terms`` (bool, default True)
    #     Show a "Terms of Service" link in the sub-bar that opens the Terms-of-Service
    #     slide-over sheet (sibling of Privacy & Responsibility).
    app.add_config_value("ai_assistant_panel_terms", True, "html")
    app.add_config_value("ai_assistant_panel_terms_title", "Terms of Service", "html")
    app.add_config_value(
        "ai_assistant_panel_terms_link_text", "Terms of Service", "html"
    )
    # Trusted author HTML (from conf.py, NOT end-user input) — injected
    # verbatim into the sheet body.  Empty → built-in default copy.
    app.add_config_value("ai_assistant_panel_terms_html", "", "html")

    # ── Phase B: Share sheet ───────────────────────────────────────────────
    # ``ai_assistant_panel_share`` (bool, default True)
    #     Show a "Share" button in the sub-bar that opens a small modal
    #     with copy-link and intent-share targets (email, X, LinkedIn,
    #     Reddit, Hacker News by default).  Customisable via
    #     ``ai_assistant_panel_share_targets`` (list[dict]).
    app.add_config_value("ai_assistant_panel_share", True, "html")
    app.add_config_value("ai_assistant_panel_share_label", "Share", "html")
    # User list overrides defaults entirely; empty list ⇒ defaults.
    app.add_config_value("ai_assistant_panel_share_targets", [], "html")

    # ── Project Links sheet ────────────────────────────────────────────────
    # ``ai_assistant_panel_links`` (bool, default True)
    #     Master switch for the "Project Links" slide-over sheet.  When True
    #     (default), clicking either the Source button (left subbar) or the
    #     Site button (right subbar) opens this sheet showing rich cards for
    #     the GitHub repository and the project website.
    #     Set False to bypass the sheet; each button then opens its URL
    #     directly in a new tab instead (graceful fallback).
    app.add_config_value("ai_assistant_panel_links", True, "html")

    # ``ai_assistant_panel_links_title`` (str, default "Project Links")
    #     Heading displayed in the slide-over sheet header.
    app.add_config_value("ai_assistant_panel_links_title", "Project Links", "html")

    # ``ai_assistant_panel_links_html`` (str, default "")
    #     Trusted author HTML injected verbatim into the Links sheet body.
    #     When non-empty this completely replaces the built-in two-card
    #     layout (source + site cards).  Same security model as
    #     ``ai_assistant_panel_privacy_html`` — MUST come from conf.py,
    #     never from end-user input.  Empty → built-in cards.
    app.add_config_value("ai_assistant_panel_links_html", "", "html")

    # ── Source Repository button (left subbar cluster) ─────────────────────
    # ``ai_assistant_panel_source`` (bool, default True)
    #     Show a GitHub icon + "Source" label button in the LEFT sub-bar
    #     cluster.  Clicking opens the Links sheet (or the source URL
    #     directly when panelLinks is False).
    app.add_config_value("ai_assistant_panel_source", True, "html")

    # ``ai_assistant_panel_source_url`` (str, default "")
    #     URL of the source repository (e.g.
    #     "https://github.com/scikit-plots/scikit-plots").
    #     Validated client-side by _isSafeHref; unsafe URLs are ignored.
    app.add_config_value("ai_assistant_panel_source_url", "", "html")

    # ``ai_assistant_panel_source_label`` (str)
    #     Heading text inside the Links sheet source card.
    #     Default: "Source Repository".
    app.add_config_value("ai_assistant_panel_source_label", "Source Repository", "html")

    # ``ai_assistant_panel_source_desc`` (str, default "")
    #     Optional one-line description shown below the heading in the
    #     source card (e.g. "Contribute, report issues, and browse the code").
    app.add_config_value("ai_assistant_panel_source_desc", "", "html")

    # ``ai_assistant_panel_source_btn_label`` (str, default "Source")
    #     Text label on the subbar button itself (collapsed to icon-only
    #     on narrow panels via CSS).
    app.add_config_value("ai_assistant_panel_source_btn_label", "Source", "html")

    # ── Site (website) button (right subbar cluster, after Share) ──────────
    # ``ai_assistant_panel_site`` (bool, default True)
    #     Show a globe icon + "Website" label button in the RIGHT sub-bar
    #     cluster, positioned after the Share button.  Clicking opens the
    #     Links sheet (or the site URL directly when panelLinks is False).
    app.add_config_value("ai_assistant_panel_site", True, "html")

    # ``ai_assistant_panel_site_url`` (str, default "")
    #     URL of the project website (e.g.
    #     "https://scikit-plots.github.io/").
    #     Validated client-side by _isSafeHref; unsafe URLs are ignored.
    app.add_config_value("ai_assistant_panel_site_url", "", "html")

    # ``ai_assistant_panel_site_label`` (str)
    #     Heading text inside the Links sheet site card.
    #     Default: "Project Website".
    app.add_config_value("ai_assistant_panel_site_label", "Project Website", "html")

    # ``ai_assistant_panel_site_desc`` (str, default "")
    #     Optional one-line description shown below the heading in the
    #     website card (e.g. "Full documentation, tutorials, and examples").
    app.add_config_value("ai_assistant_panel_site_desc", "", "html")

    # ``ai_assistant_panel_site_btn_label`` (str, default "Website")
    #     Text label on the subbar button itself (collapsed to icon-only
    #     on narrow panels via CSS).
    app.add_config_value("ai_assistant_panel_site_btn_label", "Website", "html")

    # ── Phase B: Hamburger overflow menu ──────────────────────────────────
    # ``ai_assistant_panel_hamburger`` (bool, default True)
    #     Show a hamburger / overflow button in the sub-bar that duplicates
    #     the most-used controls (Privacy, Terms, Share, Keyboard help,
    #     New chat, Export) in a single popover.  Mobile-friendly; on small
    #     screens the individual icons collapse into this menu.
    app.add_config_value("ai_assistant_panel_hamburger", True, "html")

    # ``ai_assistant_panel_feedback`` (bool, default True)
    #     Show the "Was this helpful?" block after assistant replies.
    app.add_config_value("ai_assistant_panel_feedback", True, "html")

    # ``ai_assistant_panel_feedback_question`` (str)
    #     The prompt text.  Default: "Was this helpful?".
    app.add_config_value(
        "ai_assistant_panel_feedback_question", "Was this helpful?", "html"
    )

    # ``ai_assistant_panel_feedback_options`` (list[dict], default [])
    #     2+ options (any count ≥ 2).  Each: {"emoji": str, "title": str (hover/aria),
    #     "value": str (sent in the feedback event)}.  Empty list → built-in
    #     11-emoji default (gradient from 😡 "Terrible" to 🤩 "Excellent!",
    #     with 😐 "Neutral" at the midpoint value 0).
    #     The JS widget auto-scales emoji size (tiers 1-8) so all buttons fit,
    #     wrapping to multiple rows for counts above 10.
    app.add_config_value("ai_assistant_panel_feedback_options", [], "html")

    # ``ai_assistant_panel_feedback_placeholder`` / ``_submit`` / ``_thanks``
    #     Free-text box placeholder, submit-button label, post-submit message.
    app.add_config_value("ai_assistant_panel_feedback_placeholder", "", "html")
    app.add_config_value("ai_assistant_panel_feedback_submit", "Send feedback", "html")
    app.add_config_value(
        "ai_assistant_panel_feedback_thanks",
        "Thanks for your feedback!",
        "html",
    )

    # ``ai_assistant_panel_feedback_log`` (bool, default False)
    #     When True the JS also console.log()s each submission (dev aid).
    #     The submission is ALWAYS dispatched as a DOM CustomEvent
    #     ('ai-assistant-feedback') for doc-author analytics hooks; the
    #     extension itself never stores or transmits it.
    app.add_config_value("ai_assistant_panel_feedback_log", False, "html")

    # ``ai_assistant_panel_feedback_scale`` (str, default "auto")
    #     Controls the numeric values assigned to each emoji in the per-answer
    #     feedback row.  Why numeric and signed: the previous string values
    #     ("positive" / "neutral" / "negative") cannot be averaged or fed
    #     directly to a model-training pipeline.  Signed integers centred on
    #     zero are the canonical form for a Likert-style rating.
    #
    #     Accepted values (see :data:`_FEEDBACK_SCALES`):
    #
    #     * ``"auto"``   — pick from the length of
    #                      ``ai_assistant_panel_feedback_options``:
    #                      2→even-2, 3→odd-3, 4→even-4, 5→odd-5,
    #                      6→even-6, 7→odd-7, 8→even-8, 9→odd-9,
    #                      10→even-10, 11→odd-11.  Any other length falls back
    #                      to a generated symmetric range (2+ supported).
    #     * ``"even-2"`` — [-1, +1]                      (no neutral)
    #     * ``"odd-3"``  — [-1, 0, +1]
    #     * ``"even-4"`` — [-2, -1, +1, +2]              (no neutral)
    #     * ``"odd-5"``  — [-2, -1, 0, +1, +2]
    #     * ``"even-6"`` — [-3, -2, -1, +1, +2, +3]      (no neutral)
    #     * ``"odd-7"``  — [-3, -2, -1, 0, +1, +2, +3]
    #     * ``"even-8"`` — [-4..-1, +1..+4]              (no neutral)
    #     * ``"odd-9"``  — [-4..-1, 0, +1..+4]
    #     * ``"even-10"``— [-5..-1, +1..+5]              (no neutral)
    #     * ``"odd-11"`` — [-5..-1, 0, +1..+5]           (neutral at 0)
    #
    #     The values are computed server-side, serialised into
    #     ``window.AI_ASSISTANT_CONFIG.panelFeedbackScale`` as a list of ints,
    #     and rendered by the JS widget as a ``title``/``data-value`` tooltip
    #     so the final user can see the numeric weight on hover.
    app.add_config_value("ai_assistant_panel_feedback_scale", "auto", "html")

    # ── Feedback POST endpoint ───────────────────────────────────────────────
    # ``ai_assistant_panel_feedback_endpoint`` (str, default "")
    #     URL of the remote endpoint that receives POST /v1/feedback requests.
    #     When empty (default), feedback is dispatched as a CustomEvent only.
    #     Read from os.environ — NEVER hardcode a URL in conf.py.
    #     Example: os.environ.get("FEEDBACK_ENDPOINT", "")
    app.add_config_value("ai_assistant_panel_feedback_endpoint", "", "html")

    # ``ai_assistant_panel_feedback_token`` (str, default "")
    #     Bearer token for write-authenticated feedback endpoints.
    #     Sent as Authorization: Bearer <token> when non-empty.
    #     Read from os.environ — NEVER hardcode a token in conf.py.
    #     Example: os.environ.get("FEEDBACK_WRITE_TOKEN", "")
    app.add_config_value("ai_assistant_panel_feedback_token", "", "html")

    # ── Global share endpoint ────────────────────────────────────────────────
    # ``ai_assistant_global_share_endpoint`` (str, default "")
    #     URL of the CF Worker endpoint for global share storage.
    #     When empty (default), the global share tier is not rendered.
    #     Example: os.environ.get("SHARE_ENDPOINT", "")
    app.add_config_value("ai_assistant_global_share_endpoint", "", "html")

    # ``ai_assistant_global_share_token`` (str, default "")
    #     Bearer token for POST /v1/share.
    #     Example: os.environ.get("SHARE_WRITE_TOKEN", "")
    app.add_config_value("ai_assistant_global_share_token", "", "html")

    # ``ai_assistant_global_share_ttl_days`` (int, default 30)
    #     Number of days before a global share link expires (KV entry TTL).
    #     Minimum: 1.  Maximum: 365.  Enforced server-side.
    app.add_config_value("ai_assistant_global_share_ttl_days", 30, "html")

    # ── Training contribution endpoint ───────────────────────────────────────
    # ``ai_assistant_training_endpoint`` (str, default "")
    #     URL of the HF Spaces proxy endpoint for training contributions.
    #     When empty (default), the contribution UI is not rendered.
    #     No write token — the proxy validates using its own HF_TOKEN.
    #     Example: os.environ.get("TRAINING_ENDPOINT", "")
    app.add_config_value("ai_assistant_training_endpoint", "", "html")

    # ── Endpoint Profile Registry ────────────────────────────────────────────
    # ``ai_assistant_endpoint_profiles`` (dict[str, dict], default {})
    #     Named endpoint profiles enabling runtime-switchable proxy backends.
    #     Each key is a profile identifier; each value is a dict with fields:
    #       label         (str)  — Human-readable name for the profile switcher UI.
    #       chat          (str)  — BASE URL; /v1/chat/completions appended by JS.
    #       share         (str)  — BASE URL; /v1/share appended by JS (P1 global share).
    #       feedback      (str)  — BASE URL; /v1/feedback appended by JS (P3 feedback).
    #       training      (str)  — BASE URL; /v1/contribute appended by JS (P2 training).
    #       shareToken    (str)  — Authorization: Bearer token for share writes.
    #       feedbackToken (str)  — Authorization: Bearer token for feedback writes.
    #       ttlDays       (int)  — Share TTL override (0 = use global setting).
    #     All fields are optional; empty string disables the feature.
    #
    #     ALL profiles are baked into the rendered HTML at build time.
    #     The browser _EP registry reads them and the user/developer can switch
    #     profiles at runtime via localStorage — NO rebuild needed.
    #
    #     Example (conf.py):
    #       import os
    #       ai_assistant_endpoint_profiles = {
    #           "cf": {
    #               "label": "Cloudflare Worker",
    #               "chat":         os.environ.get("CF_WORKER_URL", ""),
    #               "share":        os.environ.get("CF_WORKER_URL", ""),
    #               "feedback":     os.environ.get("CF_WORKER_URL", ""),
    #               "training":     "",
    #               "shareToken":   os.environ.get("SHARE_WRITE_TOKEN", ""),
    #               "feedbackToken": os.environ.get("FEEDBACK_WRITE_TOKEN", ""),
    #           },
    #           "hf": {
    #               "label": "HF Spaces",
    #               "chat":         os.environ.get("HF_SPACE_URL", ""),
    #               "share":        os.environ.get("CF_WORKER_URL", ""),
    #               "feedback":     os.environ.get("HF_SPACE_URL", ""),
    #               "training":     os.environ.get("HF_SPACE_URL", ""),
    #               "shareToken":   os.environ.get("SHARE_WRITE_TOKEN", ""),
    #               "feedbackToken": os.environ.get("FEEDBACK_WRITE_TOKEN", ""),
    #           },
    #       }
    app.add_config_value("ai_assistant_endpoint_profiles", {}, "html")

    # ``ai_assistant_endpoint_default_profile`` (str, default "")
    #     Profile key to activate on first page load (before any localStorage
    #     override).  Must match a key in ``ai_assistant_endpoint_profiles``.
    #     When empty, the first defined profile is used automatically.
    #     Example: ai_assistant_endpoint_default_profile = "cf"
    app.add_config_value("ai_assistant_endpoint_default_profile", "", "html")

    # ``ai_assistant_panel_privacy_title`` / ``_link_text`` (str)
    #     Heading shown in the slide-over and the small header link label.
    app.add_config_value(
        "ai_assistant_panel_privacy_title",
        "Privacy & Responsibility",
        "html",
    )
    app.add_config_value(
        "ai_assistant_panel_privacy_link_text",
        "Privacy & Responsibility",
        "html",
    )

    # ``ai_assistant_panel_privacy_html`` (str, default "")
    #     Full custom body for the privacy sheet.  TRUSTED author content
    #     (from conf.py, not end-user input) — injected verbatim.  Empty →
    #     the built-in structured default that explicitly separates the
    #     extension's responsibility from the integrated model's.
    app.add_config_value("ai_assistant_panel_privacy_html", "", "html")

    # ``ai_assistant_search_bar`` (bool, default False) — OPT-IN.
    #     Mounts a standalone AI search input that forwards its text into the
    #     panel.  Default OFF so the theme's own search is never affected.
    app.add_config_value("ai_assistant_search_bar", False, "html")

    # ``ai_assistant_search_bar_selector`` (str, default "")
    #     CSS selector of the host element to append the search-bar into.
    #     If empty or not found nothing happens (safe no-op).
    app.add_config_value("ai_assistant_search_bar_selector", "", "html")

    # ``ai_assistant_search_bar_position`` (str, default "bottom")
    #     Where inside the host element the search-bar is inserted.
    #     "top"    → prepend before the first child — appears at the very top
    #                of the sidebar, giving users immediate 1-click AI access
    #                without scrolling past navigation links.
    #     "bottom" → append after the last child (default; pre-existing
    #                behaviour; safe fallback for any unrecognised value).
    app.add_config_value("ai_assistant_search_bar_position", "top", "html")

    # ``ai_assistant_search_bar_mini`` (bool, default False)
    #     Compact inline variant when True; full-width block when False.
    app.add_config_value("ai_assistant_search_bar_mini", False, "html")

    # ``ai_assistant_panel_search_placeholder`` (str)
    #     Placeholder for the standalone search-bar input.
    app.add_config_value(
        "ai_assistant_panel_search_placeholder",
        "Ask AI about these docs\u2026",
        "html",
    )

    # ---- Static files ------------------------------------------------------
    static_path = Path(__file__).parent / "_static"
    if str(static_path) not in app.config.html_static_path:
        app.config.html_static_path.append(str(static_path))

    app.add_css_file("ai-assistant.css")
    app.add_js_file("ai-assistant.js")

    # ---- Event hooks -------------------------------------------------------
    app.connect("html-page-context", add_ai_assistant_context)
    app.connect("build-finished", generate_markdown_files)
    app.connect("build-finished", generate_llms_txt)

    return {
        "version": _VERSION,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
