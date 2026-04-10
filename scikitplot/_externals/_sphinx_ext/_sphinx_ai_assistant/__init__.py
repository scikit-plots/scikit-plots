# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/__init__.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# This module was copied and adapted from the sphinx-ai-assistant project.
# https://github.com/mlazag/sphinx-ai-assistant
#
# Authors: Mladen Zagorac, The scikit-plots developers
# SPDX-License-Identifier: MIT
"""
Sphinx AI Assistant Extension
==============================

A Sphinx extension — and standalone HTML post-processor — that adds
AI-assistant features to any documentation site, including one-click
Markdown export, AI chat deep-links, MCP tool integration, automated
``llms.txt`` generation, Jupyter notebook cell integration, and support
for any static-HTML site generator (not only Sphinx).

The module has **two distinct layers**:

Core layer (Sphinx-free)
    Importable without Sphinx, BeautifulSoup, or markdownify.  All
    security helpers, the HTML→Markdown converter, the multi-process
    HTML walker, the standalone directory processor, and the Jupyter
    widget live here.

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
display_jupyter_ai_button : callable
    Inject an AI-assistant button strip into the current Jupyter notebook
    output cell so users can ask Claude, ChatGPT, Gemini, Ollama, or any
    configured provider about any visualisation or cell output.

Public API (Sphinx extension)
------------------------------
setup : callable
    Sphinx extension entry point.

Notes
-----
**Developer note** — import discipline:

Every import of ``sphinx.*``, ``bs4``, ``markdownify``, and ``IPython``
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
Sphinx ``conf.py``:

.. code-block:: python

    extensions = [
        "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
    ]
    html_theme = "pydata_sphinx_theme"
    ai_assistant_enabled = True
    ai_assistant_theme_preset = "pydata_sphinx_theme"
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

Jupyter notebook:

.. code-block:: python

    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
        display_jupyter_ai_button,
    )
    import matplotlib.pyplot as plt
    plt.plot([1, 2, 3])
    plt.show()
    display_jupyter_ai_button(
        content="A line chart showing values 1, 2, 3",
        providers=["claude", "chatgpt", "gemini"],
    )
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path
from typing import (
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
    from sphinx.builders.html import StandaloneHTMLBuilder

__all__ = [
    "add_ai_assistant_context",
    "generate_llms_txt",
    "generate_markdown_files",
    "AIAssistantDirective",
    "display_jupyter_notebook_ai_button",
    "display_jupyter_ai_button",
    "generate_llms_txt_standalone",
    "process_html_directory",
    "html_to_markdown_converter",
    "html_to_markdown",
]

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_VERSION: str = "0.3.0"

# ---------------------------------------------------------------------------
# Module-level cached singletons (lazy, private)
# ---------------------------------------------------------------------------

_logger = None                   # sphinx.util.logging.getLogger — initialised lazily
_SphinxMarkdownConverter = None  # markdownify subclass — built lazily

# NOTE: ``bs4``, ``markdownify``, and ``IPython`` are intentionally NOT
# imported at module scope.  All callers import them locally when needed.


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
# Internal helpers — type coercion
# ---------------------------------------------------------------------------

def _coerce_to_list(
    val: Any,
    *,
    default: Optional[List[str]] = None,
) -> List[str]:
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
    global _logger
    if _logger is None:
        from sphinx.util import logging as _sphinx_logging
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
    global _SphinxMarkdownConverter
    if _SphinxMarkdownConverter is not None:
        return _SphinxMarkdownConverter

    from markdownify import MarkdownConverter  # type: ignore[import]

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
            content = text if text else (el.get_text() or "")
            classes: List[str] = list(el.get("class") or [])
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
            classes: List[str] = list(el.get("class") or [])
            if "admonition" in classes:
                title_el = el.find("p", class_="admonition-title")
                if title_el:
                    title_text = title_el.get_text(strip=True)
                    content = (text or "").strip()
                    if content.startswith(title_text):
                        content = content[len(title_text):].strip()
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
            content = text if text else (el.get_text() or "")
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
_OLLAMA_RECOMMENDED_MODELS: Tuple[str, ...] = (
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
_DEFAULT_PROVIDERS: Dict[str, Dict[str, Any]] = {

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
            "Hi! Please review this documentation page: {url}\n\n"
            "I have questions."
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
        "prompt_template": (
            "Read {url} so I can ask questions about it."
        ),
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
        "prompt_template": (
            "Please review this content and answer questions: {url}"
        ),
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
        "prompt_template": (
            "Please review this content: {url}\n\n{content}"
        ),
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
        "prompt_template": (
            "Please read this documentation: {url}\n\nI have questions."
        ),
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
        "prompt_template": (
            "Please read: {url}"
        ),
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
        "prompt_template": (
            "Please read this documentation: {url}\n\nI have questions."
        ),
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
        "prompt_template": (
            "Explain this documentation page: {url}"
        ),
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
_PROVIDER_REQUIRED_KEYS: Tuple[str, ...] = (
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
_DEFAULT_MCP_TOOLS: Dict[str, Dict[str, Any]] = {
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
    """Serialise *obj* to a JSON string safe for inline ``<script>`` injection.

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
    '{"url": "https://example.com/<\\\\/script>"}'
    """
    raw = json.dumps(obj, ensure_ascii=True, separators=(", ", ": "))
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
_DANGEROUS_CSS_CHARS_RE = re.compile(r"[<>]")
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
            f"ai_assistant_base_url must start with http:// or https://; "
            f"got {url!r}"
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
    >>> _validate_css_selector('<script>bad</script>')
    False
    """
    return not bool(_DANGEROUS_CSS_CHARS_RE.search(selector))


def _validate_mcp_tool(tool: Dict[str, Any], name: str = "") -> List[str]:
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
    >>> _validate_mcp_tool({"enabled": False, "type": "vscode",
    ...                     "label": "VS Code", "description": "x"})
    []
    """
    errors: List[str] = []
    prefix = f"MCP tool {name!r}: " if name else "MCP tool: "
    for key in ("enabled", "type", "label", "description"):
        if key not in tool:
            errors.append(f"{prefix}missing required key {key!r}")
    server_url = str(tool.get("server_url", "")).strip()
    if server_url and not _URL_SCHEME_RE.match(server_url):
        errors.append(
            f"{prefix}server_url {server_url!r} must use http:// or https://"
        )
    return errors


def _sanitize_selectors(selectors: List[str]) -> List[str]:
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


def _validate_provider(provider: Dict[str, Any], name: str = "") -> List[str]:
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
    errors: List[str] = []
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
        errors.append(
            f"{prefix}url_template {url_tpl!r} must use http:// or https://"
        )

    if ptype == "local":
        api_url = str(provider.get("api_base_url", ""))
        if api_url and not _validate_ollama_url(api_url):
            errors.append(
                f"{prefix}api_base_url {api_url!r} must target localhost / "
                f"127.0.0.1 for local providers"
            )

    return errors


def _filter_providers(
    providers: Dict[str, Any],
    *,
    require_enabled: bool = False,
) -> Dict[str, Any]:
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
    result: Dict[str, Any] = {}
    for name, prov in providers.items():
        url_tpl = str(prov.get("url_template", ""))
        if not _validate_provider_url_template(url_tpl):
            continue
        if require_enabled and not prov.get("enabled", False):
            continue
        result[name] = prov
    return result


# ---------------------------------------------------------------------------
# Theme selector presets
# ---------------------------------------------------------------------------

#: Mapping of ``html_theme`` names to ordered CSS selector tuples.
_THEME_SELECTOR_PRESETS: Dict[str, Tuple[str, ...]] = {
    "pydata_sphinx_theme": (
        "article.bd-article",
        "div.bd-article-container article",
        'div[role="main"]',
        "main",
    ),
    "furo": (
        'article[role="main"]',
        "div.page",
        "main",
    ),
    "sphinx_book_theme": (
        "article.bd-article",
        'div[role="main"]',
        "main",
    ),
    "sphinx_rtd_theme": (
        "div.rst-content",
        'div[role="main"]',
        "main",
    ),
    "alabaster": (
        "div.document",
        'div[role="main"]',
        "div.body",
        "main",
    ),
    "classic": (
        "div.body",
        "div.document",
        "main",
    ),
    "nature": (
        "div.body",
        'div[role="main"]',
        "main",
    ),
    "agogo": (
        "div.document",
        'div[role="main"]',
        "main",
    ),
    "haiku": (
        "div.content",
        'div[role="main"]',
        "main",
    ),
    "scrolls": (
        "div.content",
        "main",
    ),
    "press": (
        "div.body",
        "main",
    ),
    "bootstrap": (
        "div.section",
        "div.body",
        "main",
    ),
    "piccolo_theme": (
        "div.document",
        'div[role="main"]',
        "main",
    ),
    "insegel": (
        "div.rst-content",
        'div[role="main"]',
        "main",
    ),
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
    preset: Optional[str],
    custom_selectors: List[str],
) -> Tuple[str, ...]:
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
    preset_sels: Tuple[str, ...] = ()
    if preset:
        preset_sels = _THEME_SELECTOR_PRESETS.get(str(preset).strip(), ())

    seen: set = set()
    merged: List[str] = []
    for sel in (
        list(custom_selectors)
        + list(preset_sels)
        + list(_DEFAULT_CONTENT_SELECTORS)
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
    strip_tags: Union[str, List[str], None] = None,
) -> str:
    """Convert an HTML string to Markdown using the Sphinx-tuned converter.

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
    '# Hello\\n\\nWorld\\n\\n'
    """
    tags: List[str] = _coerce_to_list(strip_tags, default=["script", "style"])

    try:
        from bs4 import BeautifulSoup as _BS  # noqa: PLC0415
        soup = _BS(html_content, "html.parser")
        for tag in soup(tags):
            tag.decompose()
        html_content = str(soup)
    except ImportError:
        pass

    ConverterClass = _build_converter_class()
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
_DEFAULT_CONTENT_SELECTORS: Tuple[str, ...] = (
    "article.bd-article",               # pydata-sphinx-theme ≥ 0.13
    'div[role="main"]',                 # pydata (older), RTD, generic
    'article[role="main"]',             # Furo theme
    "div.rst-content",                  # Read the Docs theme
    "div.document",                     # Sphinx Classic / Alabaster
    "div.body",                         # Older Sphinx themes
    "div.bd-article-container article", # pydata nested wrapper
    "div.content",                      # Haiku / Scrolls
    "div.section",                      # Bootstrap / older Sphinx
    "div.md-content",                   # MkDocs / Material
    "article.md-content__inner",        # MkDocs Material inner
    "div.vp-doc",                       # VitePress
    "section.page-inner",               # GitBook
    "article.post-content",             # Jekyll
    "article",                          # Generic / Hugo / Docusaurus
    "main",                             # Generic HTML5
)


def _process_html_file_worker(
    args: Tuple[str, str, str, List[str], List[str], List[str]],
) -> Tuple[str, str, str]:
    """Extended worker: convert one HTML file to Markdown with separate I/O dirs.

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
        from bs4 import BeautifulSoup  # type: ignore[import]

        html_content = html_file.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html_content, "html.parser")

        main_content = None
        for selector in selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content is None:
            return ("skipped", rel_str, "No main content element found")

        markdown_content = html_to_markdown(
            str(main_content),
            strip_tags=list(strip_tags),
        )

        # Mirror the relative path under output_dir
        md_file = output_dir / rel_path.with_suffix(".md")
        md_file.parent.mkdir(parents=True, exist_ok=True)
        md_file.write_text(markdown_content, encoding="utf-8")

        return ("success", rel_str, "")

    except Exception as exc:  # noqa: BLE001
        return ("error", rel_str, str(exc))


def _process_single_html_file(
    args: Tuple[str, str, List[str], List[str], List[str]],
) -> Tuple[str, str, str]:
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
    return _process_html_file_worker((
        html_file_str,
        outdir_str,
        outdir_str,        # output_dir == input_dir → inline mode
        exclude_patterns,
        selectors,
        strip_tags,
    ))


# ---------------------------------------------------------------------------
# Standalone (non-Sphinx) HTML directory processor  — PUBLIC
# ---------------------------------------------------------------------------

def process_html_directory(
    input_dir: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    selectors: Optional[List[str]] = None,
    theme_preset: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
    strip_tags: Optional[List[str]] = None,
    max_workers: Optional[int] = None,
    recursive: bool = True,
    generate_llms: bool = False,
    base_url: str = "",
    llms_txt_max_entries: Optional[int] = None,
    llms_txt_full_content: bool = False,
) -> Dict[str, int]:
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
    import multiprocessing
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

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
            "genindex", "search", "py-modindex", "_sources", "_static",
        ]

    if strip_tags is None:
        strip_tags = ["script", "style", "nav", "footer", "header"]

    effective_selectors: List[str] = list(
        _resolve_content_selectors(theme_preset, selectors or [])
    )

    glob_fn = input_path.rglob if recursive else input_path.glob
    html_files = list(glob_fn("*.html"))

    cpu_count = multiprocessing.cpu_count() or 1
    workers: int = (
        max(1, min(cpu_count, 8))
        if max_workers is None
        else max(1, int(max_workers))
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

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_html_file_worker, a): a
            for a in args_list
        }
        for future in as_completed(futures):
            try:
                status, _rel, _msg = future.result()
            except Exception:  # noqa: BLE001
                errors += 1
                continue
            if status == "success":
                generated += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1

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
    md_root: Union[str, Path],
    *,
    base_url: str = "",
    output_file: Optional[Union[str, Path]] = None,
    project_name: str = "Documentation",
    max_entries: Optional[int] = None,
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
        Path(output_file).resolve()
        if output_file is not None
        else root / "llms.txt"
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
# Jupyter notebook AI button — PUBLIC
# ---------------------------------------------------------------------------

#: CSS selectors tried by the Jupyter widget to locate cell output content.
#: JupyterLab and classic Notebook use different DOM structures.
_JUPYTER_CONTENT_SELECTORS: Tuple[str, ...] = (
    # JupyterLab ≥ 4
    ".jp-OutputArea-output",
    ".jp-OutputArea",
    # JupyterLab 3
    ".lm-Widget.jp-OutputArea-child",
    # Classic Notebook
    ".output_area",
    ".output_text",
    ".output_html",
    # VSCode Jupyter
    ".cell-output-ipywidget-background",
    # Fallback
    ".output",
)


def _build_jupyter_widget_html(
    content: Optional[str] = None,
    *,
    providers: Union[str, List[str], None] = None,
    provider_configs: Optional[Dict[str, Any]] = None,
    position: str = "inline",
    page_url: str = "",
    widget_id: Optional[str] = None,
    intention: Optional[str] = None,
    custom_context: Optional[str] = None,
    custom_prompt_prefix: Optional[str] = None,
    notebook_mode: bool = False,
    include_outputs: bool = True,
    include_raw_image: bool = False,
    mcp_tools: Optional[Dict[str, Any]] = None,
) -> str:
    """Build self-contained HTML+JS for the Jupyter AI button strip.

    Parameters
    ----------
    content : str or None, optional
        Explicit text content to include in the AI prompt.  When ``None``
        and *notebook_mode* is ``False``, the widget JS captures text from
        the surrounding Jupyter output area automatically.  When ``None``
        and *notebook_mode* is ``True``, all visible notebook cells are
        captured.
    providers : str or list of str or None, optional
        Ordered list of provider names to show as buttons.  A bare ``str``
        is accepted and treated as a single-element list.  Defaults to
        ``["claude", "chatgpt", "gemini", "ollama"]``.
    provider_configs : dict or None, optional
        Full provider config overrides.  Merged over
        :data:`_DEFAULT_PROVIDERS`.
    position : str, optional
        One of ``"inline"`` (default) or ``"floating"``.
    page_url : str, optional
        URL embedded in provider prompt templates as ``{url}``.
    widget_id : str or None, optional
        Unique DOM ID for the widget.  Auto-generated when ``None``.
    intention : str or None, optional
        User's stated goal for this AI interaction (e.g.
        ``"explain this chart"``, ``"find the bug"``, ``"review notebook"``).
        Prepended to the prompt as ``"Goal: <intention>"``.
    custom_context : str or None, optional
        Additional background context injected after *intention* and before
        the provider's prompt template.
    custom_prompt_prefix : str or None, optional
        Raw text prepended *before* everything else in the final prompt.
    notebook_mode : bool, optional
        When ``True`` the JS walks the entire notebook DOM, collecting all
        cell inputs and (optionally) their outputs into a single prompt.
    include_outputs : bool, optional
        When *notebook_mode* is ``True``, controls whether cell outputs are
        included alongside cell inputs.  Defaults to ``True``.
    include_raw_image : bool, optional
        When ``True``, the widget JS scans ``<img>`` and ``<canvas>``
        elements in the captured output area and appends image metadata
        (dimensions, alt text, type) to the prompt.  For ``<canvas>``
        elements the pixel content is converted to a compact base64 PNG
        thumbnail (max 300 × 300 px) and included inline.  Defaults to
        ``False``.

        .. note::
            Full-resolution base64 images exceed most URL length limits
            (~2 KB).  The thumbnail is a compressed preview for context;
            for high-fidelity vision analysis, paste the image directly
            into the AI provider's chat interface.

    mcp_tools : dict or None, optional
        MCP tool configurations to surface alongside provider buttons.
        Each value is a dict matching :data:`_DEFAULT_MCP_TOOLS` schema.
        When ``None``, no MCP buttons are rendered.

    Returns
    -------
    str
        Self-contained HTML string (no external resources required).

    Notes
    -----
    The generated widget:

    * Is fully self-contained (no external CSS/JS dependencies).
    * Escapes all user-supplied strings through :func:`_safe_json_for_script`
      before embedding them in the ``<script>`` block — prevents XSS.
    * Captures surrounding cell output text via
      :data:`_JUPYTER_CONTENT_SELECTORS` when *content* is ``None`` and
      *notebook_mode* is ``False``.
    * In *notebook_mode* traverses ``.jp-Cell`` / ``.cell`` elements in
      document order, optionally including ``.jp-OutputArea`` text.
    * Validates provider ``url_template`` before rendering buttons;
      providers with dangerous schemes are silently skipped.
    * *intention*, *custom_context*, *custom_prompt_prefix*, and all
      MCP tool fields are sanitised via :func:`_safe_json_for_script`;
      no raw string interpolation into JS occurs.
    * When *include_raw_image* is ``True``, ``<canvas>`` elements are
      captured via ``toDataURL()``; cross-origin images are skipped
      gracefully (browser security policy prevents reading foreign pixels).
    """
    import hashlib
    import time as _time

    # ── Generate stable widget ID ────────────────────────────────────────────
    if widget_id is None:
        raw_id = f"{_time.monotonic_ns()}_{content!r}_{notebook_mode}"
        widget_id = "ai-btn-" + hashlib.md5(raw_id.encode()).hexdigest()[:8]

    # ── Resolve provider list ────────────────────────────────────────────────
    provider_list: List[str] = _coerce_to_list(
        providers, default=["claude", "chatgpt", "gemini", "ollama"]
    )

    # ── Merge default configs with any user overrides ────────────────────────
    merged_configs: Dict[str, Any] = {}
    for pname in provider_list:
        base = dict(_DEFAULT_PROVIDERS.get(pname, {}))
        if provider_configs and pname in provider_configs:
            base.update(provider_configs[pname])
        merged_configs[pname] = base

    # ── Security: filter providers with unsafe url_template ──────────────────
    safe_configs = _filter_providers(merged_configs)

    # ── Build button specs list (preserves requested order) ──────────────────
    buttons = []
    for pname in provider_list:
        cfg = safe_configs.get(pname, {})
        if not cfg:
            continue
        buttons.append({
            "name": pname,
            "label": str(cfg.get("label", pname)),
            "url_template": str(cfg.get("url_template", "")),
            "prompt_template": str(cfg.get("prompt_template", "Ask about: {url}")),
            "type": str(cfg.get("type", "web")),
            "enabled": bool(cfg.get("enabled", True)),
        })

    # ── Validate page URL ────────────────────────────────────────────────────
    validated_page_url = ""
    if page_url:
        try:
            validated_page_url = _validate_base_url(page_url)
        except ValueError:
            validated_page_url = ""

    # ── Validate and serialise MCP tools ────────────────────────────────────
    safe_mcp: Dict[str, Any] = {}
    if mcp_tools:
        for tname, tcfg in mcp_tools.items():
            errs = _validate_mcp_tool(tcfg, name=tname)
            if not errs:
                safe_mcp[tname] = tcfg

    # ── Serialise everything through _safe_json_for_script (XSS guard) ───────
    buttons_json          = _safe_json_for_script(buttons)
    content_json          = _safe_json_for_script(content)
    selectors_json        = _safe_json_for_script(list(_JUPYTER_CONTENT_SELECTORS))
    page_url_json         = _safe_json_for_script(validated_page_url)
    intention_json        = _safe_json_for_script(intention)
    custom_context_json   = _safe_json_for_script(custom_context)
    prompt_prefix_json    = _safe_json_for_script(custom_prompt_prefix)
    notebook_mode_json    = "true" if notebook_mode else "false"
    include_outputs_json  = "true" if include_outputs else "false"
    include_raw_image_json = "true" if include_raw_image else "false"
    mcp_tools_json        = _safe_json_for_script(safe_mcp)

    position_style = (
        "position:fixed;bottom:16px;right:16px;z-index:9999;"
        if position == "floating"
        else "margin:8px 0;display:inline-block;"
    )

    html = f"""
<div id="{widget_id}" style="{position_style}font-family:sans-serif;">
  <style>
    #{widget_id} .ai-btn-strip {{
      display:flex;flex-wrap:wrap;gap:6px;align-items:center;
      padding:6px 8px;
      background:rgba(255,255,255,0.92);
      border:1px solid #ddd;border-radius:8px;
      box-shadow:0 1px 4px rgba(0,0,0,.08);
    }}
    #{widget_id} .ai-btn {{
      display:inline-flex;align-items:center;gap:5px;
      padding:4px 10px;font-size:12px;font-weight:500;
      border:1px solid #ccc;border-radius:5px;
      background:#fff;color:#333;cursor:pointer;
      transition:background .15s,box-shadow .15s;
      white-space:nowrap;text-decoration:none;
    }}
    #{widget_id} .ai-btn:hover {{
      background:#f5f5f5;box-shadow:0 1px 3px rgba(0,0,0,.12);
    }}
    #{widget_id} .ai-btn.local-disabled {{
      opacity:.5;cursor:not-allowed;
    }}
    #{widget_id} .ai-btn-label {{font-size:11px;color:#888;margin-right:4px;}}
  </style>
  <div class="ai-btn-strip">
    <span class="ai-btn-label">Ask AI:</span>
    <div id="{widget_id}-buttons"></div>
  </div>
  <script>
  (function() {{
    var BUTTONS          = {buttons_json};
    var EXPLICIT_CONTENT = {content_json};
    var SELECTORS        = {selectors_json};
    var PAGE_URL         = {page_url_json};
    var INTENTION        = {intention_json};
    var CUSTOM_CONTEXT   = {custom_context_json};
    var PROMPT_PREFIX    = {prompt_prefix_json};
    var NOTEBOOK_MODE    = {notebook_mode_json};
    var INCLUDE_OUTPUTS  = {include_outputs_json};
    var INCLUDE_RAW_IMAGE = {include_raw_image_json};
    var MCP_TOOLS        = {mcp_tools_json};

    function getCellContent() {{
      var root = document.getElementById("{widget_id}");
      var outputArea = root ? root.closest(
        ".jp-OutputArea, .output_area, .cell-output, .output"
      ) : null;
      if (outputArea) {{
        for (var i = 0; i < SELECTORS.length; i++) {{
          var el = outputArea.querySelector(SELECTORS[i]);
          if (el) return el.innerText || el.textContent || "";
        }}
        return outputArea.innerText || outputArea.textContent || "";
      }}
      return "";
    }}

    function getNotebookContent() {{
      var cells = document.querySelectorAll(
        ".jp-Cell, .cell, .code_cell, .text_cell, .markdown_cell"
      );
      if (!cells.length) return getCellContent();
      var parts = [];
      var idx = 0;
      cells.forEach(function(cell) {{
        idx++;
        var inputEl = cell.querySelector(
          ".jp-InputArea-editor, .input_area, .CodeMirror-code, .cm-content"
        );
        var inputText = inputEl
          ? (inputEl.innerText || inputEl.textContent || "").trim()
          : "";
        if (!inputText) return;
        parts.push("--- Cell " + idx + " ---");
        parts.push(inputText);
        if (INCLUDE_OUTPUTS) {{
          var outputEl = cell.querySelector(
            ".jp-OutputArea, .output_area, .output"
          );
          var outputText = outputEl
            ? (outputEl.innerText || outputEl.textContent || "").trim()
            : "";
          if (outputText) {{
            parts.push("Output:");
            parts.push(outputText);
          }}
        }}
      }});
      return parts.join("\n");
    }}

    function getContent() {{
      if (EXPLICIT_CONTENT !== null && EXPLICIT_CONTENT !== undefined) {{
        return String(EXPLICIT_CONTENT);
      }}
      return NOTEBOOK_MODE ? getNotebookContent() : getCellContent();
    }}

    /* ── Capture images/canvas from output area (raw image mode) ── */
    function captureImages(outputArea) {{
      if (!INCLUDE_RAW_IMAGE || !outputArea) return "";
      var parts = [];
      // Canvas elements (matplotlib, plotly SVG-to-canvas, etc.)
      var canvases = outputArea.querySelectorAll("canvas");
      canvases.forEach(function(cv) {{
        try {{
          var w = cv.width, h = cv.height;
          if (!w || !h) return;
          // Create thumbnail at max 300px
          var scale = Math.min(1, 300 / Math.max(w, h));
          var tw = Math.round(w * scale), th = Math.round(h * scale);
          var tc = document.createElement("canvas");
          tc.width = tw; tc.height = th;
          var ctx = tc.getContext("2d");
          if (ctx) {{
            ctx.drawImage(cv, 0, 0, tw, th);
            var dataUrl = tc.toDataURL("image/png", 0.6);
            parts.push("[Canvas " + w + "x" + h + "px | thumbnail:" + dataUrl + "]");
          }}
        }} catch(e) {{ parts.push("[Canvas: cross-origin, cannot read]"); }}
      }});
      // <img> elements
      var imgs = outputArea.querySelectorAll("img");
      imgs.forEach(function(img) {{
        var src = img.src || "";
        var alt = img.alt || "image";
        var w = img.naturalWidth || img.width || 0;
        var h = img.naturalHeight || img.height || 0;
        if (src.startsWith("data:")) {{
          parts.push("[Image (base64, " + w + "x" + h + "): " + alt + " | " + src + "]");
        }} else if (src) {{
          parts.push("[Image: " + alt + " (" + w + "x" + h + "px) src=" + src + "]");
        }}
      }});
      return parts.length ? "\n\n[Visual outputs]\n" + parts.join("\n") : "";
    }}

    function buildPrompt(btn, content) {{
      var parts = [];
      if (PROMPT_PREFIX)    parts.push(String(PROMPT_PREFIX));
      if (INTENTION)        parts.push("Goal: " + String(INTENTION));
      if (CUSTOM_CONTEXT)   parts.push("Context: " + String(CUSTOM_CONTEXT));
      // Append image metadata when include_raw_image=True
      var root = document.getElementById("{widget_id}");
      var outputArea = root ? root.closest(
        ".jp-OutputArea, .output_area, .cell-output, .output"
      ) : (document.querySelector(".jp-OutputArea") || null);
      var imageText = captureImages(outputArea);
      if (imageText) content = content + imageText;
      var main = btn.prompt_template
        .replace(/\\{{url\\}}/g, PAGE_URL || window.location.href)
        .replace(/\\{{content\\}}/g, content);
      parts.push(main);
      return parts.join("\n\n");
    }}

    function buildUrl(btn) {{
      var content = getContent();
      var prompt = buildPrompt(btn, content);
      return btn.url_template.replace(/\\{{prompt\\}}/g, encodeURIComponent(prompt));
    }}

    var container = document.getElementById("{widget_id}-buttons");
    BUTTONS.forEach(function(btn) {{
      var isLocal = btn.type === "local";
      var a = document.createElement("a");
      a.className = "ai-btn" + (isLocal && !btn.enabled ? " local-disabled" : "");
      a.textContent = btn.label;
      a.title = isLocal ? "Local provider \u2014 requires local AI server" : btn.label;
      if (!isLocal || btn.enabled) {{
        a.href = "#";
        a.onclick = function(e) {{
          e.preventDefault();
          var url = buildUrl(btn);
          if (url) window.open(url, "_blank", "noopener,noreferrer");
        }};
      }}
      container.appendChild(a);
    }});

    /* ── MCP tool buttons ── */
    if (MCP_TOOLS && typeof MCP_TOOLS === "object") {{
      var mcpKeys = Object.keys(MCP_TOOLS);
      if (mcpKeys.length > 0) {{
        var sep = document.createElement("span");
        sep.style.cssText = "width:1px;background:#ddd;align-self:stretch;margin:0 4px;";
        container.appendChild(sep);
        var mcpLabel = document.createElement("span");
        mcpLabel.className = "ai-btn-label";
        mcpLabel.textContent = "MCP:";
        mcpLabel.style.marginLeft = "4px";
        container.appendChild(mcpLabel);
        mcpKeys.forEach(function(tname) {{
          var tcfg = MCP_TOOLS[tname];
          if (!tcfg || !tcfg.enabled) return;
          var btn2 = document.createElement("a");
          btn2.className = "ai-btn";
          btn2.textContent = String(tcfg.label || tname);
          btn2.title = String(tcfg.description || tname);
          var sUrl = String(tcfg.server_url || tcfg.mcpb_url || "");
          if (sUrl) {{
            btn2.href = sUrl;
            btn2.target = "_blank";
            btn2.rel = "noopener noreferrer";
          }}
          container.appendChild(btn2);
        }});
      }}
    }}
  }})();
  </script>
</div>"""
    return html


def display_jupyter_ai_button(
    content: Optional[str] = None,
    *,
    providers: Union[str, List[str], None] = None,
    provider_configs: Optional[Dict[str, Any]] = None,
    position: str = "inline",
    page_url: str = "",
    widget_id: Optional[str] = None,
    intention: Optional[str] = None,
    custom_context: Optional[str] = None,
    custom_prompt_prefix: Optional[str] = None,
    notebook_mode: bool = False,
    include_outputs: bool = True,
    include_raw_image: bool = False,
    mcp_tools: Optional[Dict[str, Any]] = None,
) -> None:
    """Inject an AI-assistant button strip into the current Jupyter output cell.

    Call this function directly after a visualisation or cell output to add
    one-click buttons that send the cell content to Claude, ChatGPT, Gemini,
    Ollama (local), or any other configured AI provider.

    Parameters
    ----------
    content : str or None, optional
        Explicit text/description to include in the AI prompt.  When
        ``None`` and *notebook_mode* is ``False``, the widget JS captures
        text from the surrounding Jupyter output area automatically
        (supports JupyterLab >= 3, classic Notebook, and VS Code Jupyter).
        When ``None`` and *notebook_mode* is ``True``, all visible
        notebook cells are captured.
    providers : str or list of str or None, optional
        Ordered list of provider names to show.  A bare ``str`` is treated
        as a single-element list.  Valid names: ``"claude"``,
        ``"chatgpt"``, ``"gemini"``, ``"ollama"``, ``"mistral"``,
        ``"perplexity"``, ``"copilot"``, ``"groq"``, ``"you"``.
        Defaults to ``["claude", "chatgpt", "gemini", "ollama"]``.
    provider_configs : dict or None, optional
        Per-provider config overrides merged over :data:`_DEFAULT_PROVIDERS`.
    position : str, optional
        ``"inline"`` (default) or ``"floating"``.
    page_url : str, optional
        URL embedded in provider prompt templates as ``{url}``.
    widget_id : str or None, optional
        Unique DOM ID.  Auto-generated when ``None``.
    intention : str or None, optional
        User's stated goal (e.g. ``"explain this chart"``, ``"fix the error"``).
        Prepended to the prompt as ``"Goal: <intention>"``.
    custom_context : str or None, optional
        Additional background context injected into the prompt.
    custom_prompt_prefix : str or None, optional
        Raw text prepended before everything else in the final prompt.
    notebook_mode : bool, optional
        When ``True``, captures all notebook cells instead of just the
        current output area.
    include_outputs : bool, optional
        When *notebook_mode* is ``True``, whether to include cell outputs.
        Defaults to ``True``.
    include_raw_image : bool, optional
        When ``True``, scans ``<img>`` and ``<canvas>`` elements in the
        output area and appends image metadata (and thumbnails for canvas
        elements) to the AI prompt.  Defaults to ``False``.
    mcp_tools : dict or None, optional
        MCP tool configs to render as "Connect" buttons beside the AI
        provider buttons.  Each value must match :data:`_DEFAULT_MCP_TOOLS`
        schema.  When ``None``, no MCP buttons are shown.

    Returns
    -------
    None
        The widget is displayed via :func:`IPython.display.display`.

    Raises
    ------
    ImportError
        If IPython is not installed.
    ValueError
        If *position* is not ``"inline"`` or ``"floating"``.

    Notes
    -----
    **Security**: All user-supplied strings are serialised through
    :func:`_safe_json_for_script` before embedding in the ``<script>``
    block, preventing XSS.

    Examples
    --------
    After a matplotlib plot:

    .. code-block:: python

        from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
            display_jupyter_ai_button,
        )
        import matplotlib.pyplot as plt
        plt.plot([1, 2, 3])
        plt.show()
        display_jupyter_ai_button(
            content="A line chart showing values 1, 2, 3.",
            providers=["claude", "chatgpt", "gemini"],
            intention="Explain the trend",
            include_raw_image=True,
        )
    """
    if position not in {"inline", "floating"}:
        raise ValueError(
            f"position must be 'inline' or 'floating'; got {position!r}"
        )

    try:
        from IPython.display import display, HTML  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "display_jupyter_ai_button requires IPython. "
            "Install with: pip install ipython"
        ) from exc

    html = _build_jupyter_widget_html(
        content=content,
        providers=providers,
        provider_configs=provider_configs,
        position=position,
        page_url=page_url,
        widget_id=widget_id,
        intention=intention,
        custom_context=custom_context,
        custom_prompt_prefix=custom_prompt_prefix,
        notebook_mode=notebook_mode,
        include_outputs=include_outputs,
        include_raw_image=include_raw_image,
        mcp_tools=mcp_tools,
    )
    display(HTML(html))


def display_jupyter_notebook_ai_button(
    intention: Optional[str] = None,
    *,
    providers: Union[str, List[str], None] = None,
    provider_configs: Optional[Dict[str, Any]] = None,
    position: str = "inline",
    page_url: str = "",
    widget_id: Optional[str] = None,
    include_outputs: bool = True,
    include_raw_image: bool = False,
    custom_context: Optional[str] = None,
    custom_prompt_prefix: Optional[str] = None,
    mcp_tools: Optional[Dict[str, Any]] = None,
) -> None:
    """Send the entire Jupyter notebook to an AI provider for review.

    A convenience wrapper around :func:`display_jupyter_ai_button` with
    ``notebook_mode=True``.  Call this anywhere in your notebook to let an
    AI review, interpret, or debug the full notebook content.

    The JS widget traverses every ``.jp-Cell`` (or ``.cell`` in classic
    Notebook) in DOM order, collecting input code and optionally cell
    outputs into a single structured prompt.

    Parameters
    ----------
    intention : str or None, optional
        Your stated goal for the AI review.  Common values:

        * ``"Review this notebook for errors"``
        * ``"Explain what this analysis does"``
        * ``"Fix the failing cell"``
        * ``"Suggest improvements to this code"``

        When ``None`` no goal annotation is added.
    providers : str or list of str or None, optional
        Provider names to show as buttons.  A bare ``str`` is treated as a
        single-element list.  Defaults to
        ``["claude", "chatgpt", "gemini", "ollama"]``.
    provider_configs : dict or None, optional
        Per-provider config overrides merged over :data:`_DEFAULT_PROVIDERS`.
    position : str, optional
        ``"inline"`` (default) or ``"floating"``.
    page_url : str, optional
        URL embedded in provider prompt templates as ``{url}``.
    widget_id : str or None, optional
        Unique DOM ID.  Auto-generated when ``None``.
    include_outputs : bool, optional
        When ``True`` (default), cell outputs are included in the captured
        text.  Set to ``False`` to send only source code.
    include_raw_image : bool, optional
        When ``True``, captures canvas/image thumbnails from all cell output
        areas and appends them to the prompt.  Defaults to ``False``.
    custom_context : str or None, optional
        Additional background context (domain, data description, etc.).
    custom_prompt_prefix : str or None, optional
        Raw text prepended before everything else in the final prompt.
    mcp_tools : dict or None, optional
        MCP tool configs rendered as "Connect" buttons.  When ``None``,
        no MCP buttons are shown.

    Returns
    -------
    None
        The widget is displayed via :func:`IPython.display.display`.

    Raises
    ------
    ImportError
        If IPython is not installed.
    ValueError
        If *position* is not ``"inline"`` or ``"floating"``.

    Notes
    -----
    **DOM traversal**: The JS uses ``document.querySelectorAll`` with
    ``.jp-Cell, .cell, .code_cell, .text_cell, .markdown_cell``.  This
    covers JupyterLab >= 3 and classic Notebook.  VS Code Jupyter and
    other environments fall back gracefully to cell-level capture.

    Examples
    --------
    At the end of a notebook to request a full review:

    .. code-block:: python

        from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
            display_jupyter_notebook_ai_button,
        )
        display_jupyter_notebook_ai_button(
            intention="Review this notebook for bugs and suggest improvements",
            providers=["claude", "chatgpt"],
            include_outputs=True,
        )
    """
    display_jupyter_ai_button(
        content=None,
        providers=providers,
        provider_configs=provider_configs,
        position=position,
        page_url=page_url,
        widget_id=widget_id,
        intention=intention,
        custom_context=custom_context,
        custom_prompt_prefix=custom_prompt_prefix,
        notebook_mode=True,
        include_outputs=include_outputs,
        include_raw_image=include_raw_image,
        mcp_tools=mcp_tools,
    )

# ---------------------------------------------------------------------------
# Build-time hooks (Sphinx layer)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sphinx RST directive and role
# ---------------------------------------------------------------------------

def _build_ai_assistant_directive_node(
    providers: List[str],
    position: str,
    intention: Optional[str],
    custom_context: Optional[str],
    page_url: str,
    mcp_tools: Optional[Dict[str, Any]],
    include_raw_image: bool,
) -> str:
    """Return a self-contained HTML widget snippet for embedding in RST.

    Parameters
    ----------
    providers : list of str
        Provider names to show.
    position : str
        Widget position (``"inline"`` or ``"floating"``).
    intention : str or None
        User goal annotation prepended to prompts.
    custom_context : str or None
        Additional background context.
    page_url : str
        URL for the ``{url}`` prompt placeholder.
    mcp_tools : dict or None
        MCP tool configs for "Connect" buttons.
    include_raw_image : bool
        Whether to capture image/canvas elements.

    Returns
    -------
    str
        Raw HTML string safe for embedding in a Sphinx ``raw`` directive.
    """
    return _build_jupyter_widget_html(
        content=None,
        providers=providers or ["claude", "chatgpt", "gemini"],
        position=position,
        page_url=page_url,
        intention=intention,
        custom_context=custom_context,
        include_raw_image=include_raw_image,
        mcp_tools=mcp_tools,
    )


class AIAssistantDirective:
    """Sphinx ``.. ai-assistant::`` directive.

    Embeds the AI-assistant button widget inline in any RST page.
    Works in both Sphinx documentation pages and standalone HTML files.

    Usage in RST
    ------------
    .. code-block:: rst

        .. ai-assistant::
           :providers: claude, chatgpt, gemini
           :position: inline
           :intention: Help me understand this page
           :include_raw_image: false

    Options
    -------
    providers : str, optional
        Comma-separated list of provider names.  Defaults to
        ``claude, chatgpt, gemini``.
    position : str, optional
        ``"inline"`` (default) or ``"floating"``.
    intention : str, optional
        Goal annotation prepended to all prompts.
    custom_context : str, optional
        Extra context injected into all prompts.
    page_url : str, optional
        Explicit URL for ``{url}`` in prompt templates.
    include_raw_image : str, optional
        ``"true"`` to capture canvas/image elements.  Defaults to
        ``"false"``.

    Notes
    -----
    This directive is registered by :func:`setup` as ``ai-assistant``.
    It generates a ``raw:: html`` node so it works with any HTML builder.
    """

    # Docutils directive interface — populated by Sphinx when registered
    required_arguments = 0
    optional_arguments = 0
    has_content = False
    option_spec: Dict[str, Any] = {}  # populated in setup() after docutils import

    def run(self) -> List[Any]:  # type: ignore[type-arg]
        """Generate the raw HTML node for this directive."""
        try:
            from docutils import nodes  # type: ignore[import]
        except ImportError:
            return []

        providers_raw = str(self.options.get("providers", "claude,chatgpt,gemini"))
        providers = [p.strip() for p in providers_raw.split(",") if p.strip()]
        position = str(self.options.get("position", "inline")).strip()
        intention_raw = str(self.options.get("intention", "")).strip() or None
        custom_context_raw = str(self.options.get("custom_context", "")).strip() or None
        page_url = str(self.options.get("page_url", "")).strip()
        include_raw_image = (
            str(self.options.get("include_raw_image", "false")).strip().lower()
            in ("true", "1", "yes")
        )

        html = _build_ai_assistant_directive_node(
            providers=providers,
            position=position,
            intention=intention_raw,
            custom_context=custom_context_raw,
            page_url=page_url,
            mcp_tools=None,
            include_raw_image=include_raw_image,
        )
        raw_node = nodes.raw("", html, format="html")
        return [raw_node]


def _ai_ask_role(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Any,
    options: Optional[Dict[str, Any]] = None,
    content: Optional[List[str]] = None,
) -> Any:
    """Sphinx ``:ai_ask:`` inline role.

    Creates an inline AI-query link.  The role text becomes the
    ``intention`` annotation in the prompt, and the current page URL is
    used as context.

    Usage in RST
    ------------
    .. code-block:: rst

        Read the API reference :ai_ask:`How does this function work?` and
        ask Claude for clarification.

    Parameters
    ----------
    name : str
        Role name (``"ai_ask"``).
    rawtext : str
        Raw RST source text.
    text : str
        The interpreted text (the question / intention).
    lineno : int
        Line number in the source document.
    inliner : docutils.parsers.rst.states.Inliner
        The RST inliner.
    options : dict or None, optional
        Role options.
    content : list or None, optional
        Role content (unused).

    Returns
    -------
    tuple
        ``([node], [])`` — single raw HTML node and empty messages list.
    """
    try:
        from docutils import nodes  # type: ignore[import]
    except ImportError:
        return [], []

    safe_text = str(text).strip()
    html = _build_jupyter_widget_html(
        content=None,
        providers=["claude", "chatgpt", "gemini"],
        intention=safe_text if safe_text else None,
        position="inline",
    )
    raw_node = nodes.raw("", html, format="html")
    return [raw_node], []


def generate_markdown_files(app: "Sphinx", exception: Optional[Exception]) -> None:
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
    if exception is not None:
        return

    log = _get_logger()

    from sphinx.builders.html import StandaloneHTMLBuilder

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

    import multiprocessing
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    outdir = Path(builder.outdir)
    exclude_patterns: List[str] = list(
        app.config.ai_assistant_markdown_exclude_patterns
    )

    preset: Optional[str] = (
        getattr(app.config, "ai_assistant_theme_preset", None) or None
    )
    selectors: List[str] = list(
        _resolve_content_selectors(
            preset,
            list(app.config.ai_assistant_content_selectors),
        )
    )
    strip_tags: List[str] = list(
        getattr(app.config, "ai_assistant_strip_tags", ["script", "style"])
    )

    html_files = list(outdir.rglob("*.html"))
    log.info(
        f"AI Assistant: Generating Markdown for {len(html_files)} HTML files…"
    )

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
    t0 = time.monotonic()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_html_file, a): a
            for a in args_list
        }
        for future in as_completed(futures):
            try:
                status, rel_path, message = future.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                log.warning(f"AI Assistant: Worker error: {exc}")
                continue
            if status == "success":
                generated += 1
            elif status == "skipped":
                skipped += 1
                if message:
                    log.debug(f"AI Assistant: Skipped {rel_path}: {message}")
            else:
                errors += 1
                log.warning(
                    f"AI Assistant: Failed to convert {rel_path}: {message}"
                )

    elapsed = time.monotonic() - t0
    log.info(
        f"AI Assistant: {generated} generated, {skipped} skipped, "
        f"{errors} errors — {elapsed:.1f}s ({max_workers} workers)"
    )


def generate_llms_txt(app: "Sphinx", exception: Optional[Exception]) -> None:
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
    if exception is not None:
        return
    if not app.config.ai_assistant_generate_markdown:
        return
    if not app.config.ai_assistant_generate_llms_txt:
        return

    from sphinx.builders.html import StandaloneHTMLBuilder

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

    max_entries: Optional[int] = getattr(
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

    log.info(
        f"AI Assistant: llms.txt written with {len(md_files)} entries"
    )



def _cfg_str(config: Any, key: str) -> Optional[str]:
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
    app: "Sphinx",
    pagename: str,
    templatename: str,
    context: Dict[str, Any],
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

    providers_raw: Dict[str, Any] = dict(app.config.ai_assistant_providers)
    providers_safe = _filter_providers(providers_raw)

    config: Dict[str, Any] = {
        "position": position_val,
        "content_selector": app.config.ai_assistant_content_selector,
        "features": dict(app.config.ai_assistant_features),
        "providers": providers_safe,
        "mcp_tools": dict(app.config.ai_assistant_mcp_tools),
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
        "includeRawImage": _cfg_bool(app.config, "ai_assistant_include_raw_image", False),
        "notebookMode": _cfg_bool(app.config, "ai_assistant_notebook_mode", False),
        "includeOutputs": _cfg_bool(app.config, "ai_assistant_include_outputs", True),
    }

    context["ai_assistant_config"] = config

    if "metatags" not in context:
        context["metatags"] = ""

    safe_json = _safe_json_for_script(config)
    context["metatags"] += (
        f"\n<script>\nwindow.AI_ASSISTANT_CONFIG = {safe_json};\n</script>\n"
    )


# ---------------------------------------------------------------------------
# Sphinx extension entry point
# ---------------------------------------------------------------------------

def setup(app: "Sphinx") -> Dict[str, Any]:
    """Register the AI-assistant extension with a Sphinx application.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application being configured.

    Returns
    -------
    dict
        Sphinx extension metadata::

            {
                "version": "0.3.0",
                "parallel_read_safe": True,
                "parallel_write_safe": True,
            }
    """
    # ---- Core toggles ------------------------------------------------------
    app.add_config_value("ai_assistant_enabled", True, "html")
    app.add_config_value("ai_assistant_position", "sidebar", "html")
    app.add_config_value("ai_assistant_content_selector", "article", "html")
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

    # ---- MCP tool configuration -------------------------------------------
    app.add_config_value(
        "ai_assistant_mcp_tools",
        {
            "vscode": {
                "enabled": False,
                "type": "vscode",
                "label": "Connect to VS Code",
                "description": "Install MCP server in VS Code",
                "icon": "vscode.svg",
                "server_name": "",
                "server_url": "",
                "transport": "sse",
            },
            "claude_desktop": {
                "enabled": False,
                "type": "claude_desktop",
                "label": "Connect to Claude",
                "description": "Download and install MCP extension",
                "icon": "claude.svg",
                "mcpb_url": "",
            },
        },
        "html",
    )

    # ---- Prompt customisation config values --------------------------------
    app.add_config_value("ai_assistant_intention", None, "html")
    app.add_config_value("ai_assistant_custom_context", None, "html")
    app.add_config_value("ai_assistant_custom_prompt_prefix", None, "html")
    app.add_config_value("ai_assistant_include_raw_image", False, "html")
    app.add_config_value("ai_assistant_notebook_mode", False, "html")
    app.add_config_value("ai_assistant_include_outputs", True, "html")

    # ---- Ollama / local model config helper --------------------------------
    # Expose the recommended model list so conf.py can reference it:
    #   from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
    #       _OLLAMA_RECOMMENDED_MODELS,
    #   )
    app.add_config_value("ai_assistant_ollama_model", "llama3.2:latest", "html")

    # ---- Static files ------------------------------------------------------
    static_path = Path(__file__).parent / "_static"
    if str(static_path) not in app.config.html_static_path:
        app.config.html_static_path.append(str(static_path))

    app.add_css_file("ai-assistant.css")
    app.add_js_file("ai-assistant.js")

    # ---- Sphinx RST directive and role -------------------------------------
    try:
        from docutils.parsers.rst import Directive, directives  # type: ignore[import]

        # Bind option_spec so directives.unchanged is available
        AIAssistantDirective.option_spec = {
            "providers":         directives.unchanged,
            "position":          directives.unchanged,
            "intention":         directives.unchanged,
            "custom_context":    directives.unchanged,
            "page_url":          directives.unchanged,
            "include_raw_image": directives.unchanged,
        }
        # Make AIAssistantDirective inherit from Directive properly
        AIAssistantDirectiveRegistered = type(
            "AIAssistantDirectiveRegistered",
            (Directive,),
            dict(AIAssistantDirective.__dict__),
        )
        app.add_directive("ai-assistant", AIAssistantDirectiveRegistered)
        app.add_role("ai_ask", _ai_ask_role)
    except Exception:  # noqa: BLE001
        pass  # Docutils unavailable; directive/role silently skipped

    # ---- Event hooks -------------------------------------------------------
    app.connect("html-page-context", add_ai_assistant_context)
    app.connect("build-finished", generate_markdown_files)
    app.connect("build-finished", generate_llms_txt)

    return {
        "version": _VERSION,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
