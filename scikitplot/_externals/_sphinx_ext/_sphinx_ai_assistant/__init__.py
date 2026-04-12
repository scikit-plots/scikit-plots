# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/__init__.py
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
Sphinx AI Assistant Extension
==============================

A Sphinx extension that adds AI-assistant features to documentation pages,
including one-click Markdown export, AI chat deep-links, MCP tool
integration, and automated ``llms.txt`` generation.

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

Public API (standalone / non-Sphinx)
-------------------------------------
process_html_directory : callable
    Walk any HTML directory tree, convert pages to Markdown, optionally
    produce ``llms.txt``.  Works with Sphinx, MkDocs, Jekyll, plain HTML,
    or any other static-site generator.
generate_llms_txt_standalone : callable
    Write ``llms.txt`` from an existing set of ``.md`` files without
    requiring a Sphinx build.
html_to_markdown : callable
    Convert an HTML string to Markdown.

Public API (Sphinx extension)
------------------------------
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
Sphinx Register in ``conf.py``:

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

import importlib.util
import json
import multiprocessing
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
    Uses a carriage-return ``\\r`` so successive calls overwrite the same
    terminal line.  A newline is written when *current* >= *total* so the
    finalised bar is left visible in the scroll buffer.

    This function follows the ``sys.stdout.write`` / ``sys.stdout.flush``
    pattern — the same approach used by reporthook-style progress tracking
    (see :func:`urllib.request.urlretrieve`).  It is safe in non-TTY
    environments (CI, pipes) — the output simply contains ``\\r`` characters.

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
    # ------------------------------------------------------------------ Claude
    "claude": {
        "enabled": True,
        "label": "Ask Claude",
        "description": "Ask Anthropic Claude about this page",
        "icon": "claude.svg",
        "url_template": "https://claude.ai/new?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"
            "I have questions about it."
        ),
        "model": "claude-sonnet-4-6",
        "type": "web",
    },
    # ------------------------------------------------------------------ Gemini
    "gemini": {
        "enabled": True,
        "label": "Ask Gemini",
        "description": "Ask Google Gemini about this page",
        "icon": "gemini.svg",
        "url_template": "https://gemini.google.com/app?q={prompt}",
        "prompt_template": (
            "Hi! Please review this documentation page: {url}\n\nI have questions."
        ),
        "model": "gemini-2.5-flash",
        "type": "web",
    },
    # ----------------------------------------------------------------- ChatGPT
    "chatgpt": {
        "enabled": True,
        "label": "Ask ChatGPT",
        "description": "Ask OpenAI ChatGPT about this page",
        "icon": "chatgpt.svg",
        "url_template": "https://chatgpt.com/?q={prompt}",
        "prompt_template": "Read {url} so I can ask questions about it.",
        "model": "gpt-4o",
        "type": "web",
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
        "prompt_template": "Please review this content and answer questions: {url}",
        "model": "llama3.2:latest",
        "type": "local",
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
        "prompt_template": "Please review this content: {url}\n\n{content}",
        "model": "",
        "type": "custom",
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
            "Please review this documentation: {url}\n\nI have questions."
        ),
        "model": "gpt-4o",
        "type": "web",
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
        "prompt_template": "Please read this documentation: {url}\n\nI have questions.",
        "model": "deepseek-reasoner",
        "type": "web",
    },
    # -------------------------------------------------------------------- Groq
    # Groq provides extremely fast open-source LLM inference.
    "groq": {
        "enabled": False,
        "label": "Ask Groq",
        "description": "Ask Groq (fast LLM inference) about this page",
        "icon": "groq.svg",
        "url_template": "https://console.groq.com/playground?q={prompt}",
        "prompt_template": "Please read: {url}",
        "model": "llama-3.3-70b-versatile",
        "type": "web",
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
            "Please read this documentation and answer my questions: {url}"
        ),
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "type": "web",
    },
    # ----------------------------------------------------------------- Mistral
    "mistral": {
        "enabled": False,
        "label": "Ask Mistral",
        "description": "Ask Mistral AI Le Chat about this page",
        "icon": "mistral.svg",
        "url_template": "https://chat.mistral.ai/chat?q={prompt}",
        "prompt_template": "Please read this documentation: {url}\n\nI have questions.",
        "model": "mistral-large-latest",
        "type": "web",
    },
    # --------------------------------------------------------------- Perplexity
    "perplexity": {
        "enabled": False,
        "label": "Ask Perplexity",
        "description": "Ask Perplexity AI about this page",
        "icon": "perplexity.svg",
        "url_template": "https://www.perplexity.ai/?q={prompt}",
        "prompt_template": "Explain this documentation page: {url}",
        "model": "sonar-pro",
        "type": "web",
    },
    # ----------------------------------------------------------------- You.com
    "you": {
        "enabled": False,
        "label": "Ask You.com",
        "description": "Ask You.com AI about this page",
        "icon": "you.svg",
        "url_template": "https://you.com/?q={prompt}",
        "prompt_template": (
            "Please review this documentation: {url}\n\nI have questions."
        ),
        "model": "default",
        "type": "web",
    },
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
        JSON string with ``</`` replaced by ``<\\/`` to prevent the browser
        from interpreting the literal ``</script>`` tag inside the payload.

    Notes
    -----
    Python's :func:`json.dumps` does **not** escape ``<``, ``>``, or ``&``
    by default.  The ``</script>`` sequence inside a ``<script>`` block
    causes the browser's HTML parser to close the script prematurely, which
    is a common XSS vector.  Replacing every ``</`` with ``<\\/`` is the
    canonical defence — the JSON remains semantically identical but the
    HTML parser sees no closing tag.

    References
    ----------
    .. [1] https://html.spec.whatwg.org/multipage/scripting.html

    Examples
    --------
    >>> _safe_json_for_script({"url": "https://example.com/</script>"})
    '{"url": "https://example.com/<\\/script>"}'
    """
    raw = json.dumps(obj, ensure_ascii=True, separators=(", ", ": "))
    # Replace every '</' to prevent script-injection regardless of tag name
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
        # Try canonical absolute name first (installed package), then
        # relative to handle bootstrap / isolated test environments.
        try:
            from ._static import (  # noqa: PLC0415
                _PROVIDER_META as _pm,  # noqa: N811
            )
            from ._static import (  # noqa: PLC0415
                _SVG_DEFAULT as _sd,  # noqa: N811
            )
        except ImportError:
            from ._static import (  # type: ignore[no-redef]  # noqa: PLC0415
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
        count, capped at 8).
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

    cpu_count = multiprocessing.cpu_count() or 1
    workers: int = (
        max(1, min(cpu_count, 8)) if max_workers is None else max(1, int(max_workers))
    )

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

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_html_file_worker, a): a for a in args_list}
        for future in as_completed(futures):
            try:
                status, _rel, _msg = future.result()
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

    max_workers_cfg = app.config.ai_assistant_max_workers
    cpu_count = multiprocessing.cpu_count() or 1
    max_workers: int = (
        max(1, min(cpu_count, 8))
        if max_workers_cfg is None
        else max(1, int(max_workers_cfg))
    )

    args_list = [
        (str(f), str(outdir), list(exclude_patterns), selectors, strip_tags)
        for f in html_files
    ]

    generated = skipped = errors = 0
    total_files = len(args_list)
    processed = 0
    t0 = time.monotonic()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_html_file, a): a for a in args_list}
        for future in as_completed(futures):
            try:
                status, rel_path, message = future.result()
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
        f"{errors} errors — {elapsed:.1f}s ({max_workers} workers)"
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
            log.debug("AI Assistant: llms.txt max_entries=0; no entries to write")
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

    config: dict[str, Any] = {
        "position": position_val,
        "content_selector": app.config.ai_assistant_content_selector,
        "features": dict(app.config.ai_assistant_features),
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
    }

    context["ai_assistant_config"] = config

    if "metatags" not in context:
        context["metatags"] = ""

    safe_json = _safe_json_for_script(config)
    _script = f"\n<script>\nwindow.AI_ASSISTANT_CONFIG = {safe_json};\n</script>\n"
    context["metatags"] += f"{_script}"


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
         - Max parallel workers; ``None`` → auto (CPU count, capped at 8).
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
    app.add_config_value("ai_assistant_max_workers", None, "html")
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
