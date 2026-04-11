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

A Sphinx extension that adds AI-assistant features to documentation pages,
including one-click Markdown export, AI chat deep-links, MCP tool
integration, and automated ``llms.txt`` generation.

All heavy optional dependencies (``sphinx``, ``bs4``, ``markdownify``) are
imported **lazily** — only when a feature is actually invoked.  Importing
this module at the top level is always safe and has no side effects.

Public API
----------
setup : callable
    Sphinx extension entry point.  Called automatically by Sphinx when this
    module is listed in ``conf.py extensions``.

Notes
-----
**Developer note** — import discipline:

Every import of ``sphinx.*``, ``bs4``, and ``markdownify`` lives *inside*
the function that needs it, guarded by a try/except where appropriate.
Nothing is imported at module scope except stdlib modules.  This keeps
``import time`` cost near zero and avoids ``ImportError`` at load time when
optional packages are absent.

**Security note**:

* :func:`_safe_json_for_script` escapes ``</script>`` sequences to prevent
  script-injection attacks when config is serialised into an HTML page.
* :func:`_is_path_within` prevents path-traversal attacks in the
  multi-process HTML walker.
* :func:`_validate_base_url` rejects non-HTTP(S) schemes in the base URL
  configuration value.
* :func:`_validate_position` rejects unknown widget-position strings.
* :func:`_validate_provider_url_template` rejects non-HTTP(S) schemes in
  AI-provider URL templates, blocking ``javascript:`` and ``data:`` vectors.
* :func:`_validate_css_selector` rejects selectors containing HTML-injection
  characters (``<`` or ``>``).
* ``ai_assistant_providers`` URL templates are validated before use, and
  unsafe entries are silently dropped from the serialised page config.

References
----------
.. [1] https://github.com/mlazag/sphinx-ai-assistant
.. [2] https://llmstxt.org/

Examples
--------
Register in ``conf.py``:

.. code-block:: python

    extensions = [
        "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
    ]
    html_theme = "pydata_sphinx_theme"   # scikit-learn / NumPy style
    ai_assistant_enabled = True
    ai_assistant_theme_preset = "pydata_sphinx_theme"  # auto-selects CSS selectors
    ai_assistant_generate_markdown = True
    ai_assistant_generate_llms_txt = True
    html_baseurl = "https://docs.example.com"
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
)

if TYPE_CHECKING:  # pragma: no cover — only for type checkers, never at runtime
    from sphinx.application import Sphinx
    from sphinx.builders.html import StandaloneHTMLBuilder

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

_VERSION: str = "0.2.0"

# ---------------------------------------------------------------------------
# Module-level cached singletons (lazy, private)
# ---------------------------------------------------------------------------

_logger = None                   # sphinx.util.logging.getLogger — initialised lazily
_SphinxMarkdownConverter = None  # markdownify subclass — built lazily

# NOTE: ``bs4.BeautifulSoup`` is intentionally NOT imported at module scope.
# Doing so would violate the lazy-import contract documented in this module's
# docstring and would raise ``ImportError`` in environments where the package
# is absent.  All callers import it locally when they need it.


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
                Fenced code block (``\\`\\`\\`lang\\n...\\`\\`\\```) or inline
                backtick span.
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
            This method reads (but never mutates) the original element so
            that parallel processing remains safe.
            """
            classes: List[str] = list(el.get("class") or [])
            if "admonition" in classes:
                title_el = el.find("p", class_="admonition-title")
                if title_el:
                    title_text = title_el.get_text(strip=True)
                    content = (text or "").strip()
                    # Strip leading occurrence of the title from converted text
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
    .. [2] https://cheatsheetseries.owasp.org/cheatsheets/XSS_Filter_Evasion_Cheat_Sheet.html

    Examples
    --------
    >>> _safe_json_for_script({"url": "https://example.com/</script>"})
    '{"url": "https://example.com/<\\\\/script>"}'
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

    Notes
    -----
    Both paths are resolved to absolute forms before comparison so that
    relative components (``..``) cannot escape the boundary.

    Examples
    --------
    >>> from pathlib import Path
    >>> _is_path_within(Path("/a/b/c"), Path("/a/b"))
    True
    >>> _is_path_within(Path("/a/../etc/passwd"), Path("/a"))
    False
    """
    try:
        resolved_path = path.resolve()
        resolved_parent = parent.resolve()
        resolved_path.relative_to(resolved_parent)
        return True
    except ValueError:
        return False


_URL_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)

# Characters that have no valid place in a CSS selector and signal an
# HTML-injection attempt.  Note: brackets ``[]`` and quotes ``"'`` are
# intentionally allowed because attribute selectors legitimately use them
# (e.g. ``div[role="main"]``).
_DANGEROUS_CSS_CHARS_RE = re.compile(r"[<>]")

#: Widget positions accepted by the JavaScript widget.
_ALLOWED_POSITIONS: frozenset = frozenset({"sidebar", "title", "floating", "none"})


def _validate_base_url(url: str) -> str:
    """Validate and normalise a documentation base URL.

    Parameters
    ----------
    url : str
        The candidate URL (e.g. ``"https://docs.example.com"``).

    Returns
    -------
    str
        The input URL, stripped of trailing slashes.

    Raises
    ------
    ValueError
        If *url* is non-empty and does not begin with ``http://`` or
        ``https://``, preventing ``javascript:``, ``data:``, or other
        dangerous scheme injections.

    Examples
    --------
    >>> _validate_base_url("https://docs.example.com/")
    'https://docs.example.com'
    >>> _validate_base_url("")
    ''
    >>> _validate_base_url("javascript:alert(1)")
    Traceback (most recent call last):
        ...
    ValueError: ai_assistant_base_url must start with http:// or https://; ...
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

    Notes
    -----
    Validation prevents user-supplied strings from leaking unexpected
    values into the serialised page configuration.

    Examples
    --------
    >>> _validate_position("sidebar")
    'sidebar'
    >>> _validate_position("TITLE")
    'title'
    >>> _validate_position("evil")
    Traceback (most recent call last):
        ...
    ValueError: ai_assistant_position must be one of ...
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
        The URL template string for an AI provider
        (e.g. ``"https://claude.ai/new?q={prompt}"``).

    Returns
    -------
    bool
        ``True`` when the template is empty (no URL) or starts with
        ``http://`` or ``https://``; ``False`` for any other scheme
        (``javascript:``, ``data:``, ``ftp:``, etc.).

    Notes
    -----
    Only the scheme prefix is checked, not the full URL structure.  This
    matches the behaviour of :func:`_validate_base_url` and is sufficient
    to block the most dangerous injection vectors while remaining simple
    and deterministic.

    Examples
    --------
    >>> _validate_provider_url_template("https://claude.ai/new?q={prompt}")
    True
    >>> _validate_provider_url_template("javascript:alert(1)")
    False
    >>> _validate_provider_url_template("")
    True
    """
    stripped = str(url_template).strip()
    if not stripped:
        return True
    return bool(_URL_SCHEME_RE.match(stripped))


def _validate_css_selector(selector: str) -> bool:
    """Return *True* if *selector* contains no HTML-injection characters.

    Parameters
    ----------
    selector : str
        A CSS selector string (e.g. ``"article.bd-article"``).

    Returns
    -------
    bool
        ``False`` when the selector contains ``<`` or ``>``, which have no
        valid place in a CSS selector and indicate an injection attempt;
        ``True`` otherwise.

    Notes
    -----
    Attribute selectors that legitimately contain ``"`` or ``'`` (such as
    ``div[role="main"]``) are explicitly allowed.

    Examples
    --------
    >>> _validate_css_selector('div[role="main"]')
    True
    >>> _validate_css_selector('<script>bad</script>')
    False
    """
    return not bool(_DANGEROUS_CSS_CHARS_RE.search(selector))


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


# ---------------------------------------------------------------------------
# Theme selector presets
# ---------------------------------------------------------------------------

#: Mapping of ``html_theme`` names to ordered CSS selector tuples.
#: Each tuple lists selectors from most-specific to least-specific.
#: Used by :func:`_resolve_content_selectors` when
#: ``ai_assistant_theme_preset`` is set.
_THEME_SELECTOR_PRESETS: Dict[str, Tuple[str, ...]] = {
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
}


def _resolve_content_selectors(
    preset: Optional[str],
    custom_selectors: List[str],
) -> Tuple[str, ...]:
    """Merge theme-preset and user-defined CSS selectors into one ordered tuple.

    Parameters
    ----------
    preset : str or None
        A key from :data:`_THEME_SELECTOR_PRESETS` (e.g.
        ``"pydata_sphinx_theme"``), or ``None`` to skip preset lookup.
    custom_selectors : list of str
        User-supplied selectors from ``ai_assistant_content_selectors``.
        These take priority over preset selectors.

    Returns
    -------
    tuple of str
        Deduplicated selectors: custom first, preset second,
        :data:`_DEFAULT_CONTENT_SELECTORS` as final fallback.
        All selectors are validated by :func:`_validate_css_selector`;
        unsafe entries are silently removed.
        Never returns an empty tuple (falls back to the module default).

    Notes
    -----
    Deduplication preserves first-occurrence order.  This guarantees that
    theme-specific selectors are tried before the generic HTML5 fallbacks.

    Examples
    --------
    >>> _resolve_content_selectors("furo", ["div.custom"])
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
    strip_tags: Optional[List[str]] = None,
) -> str:
    """Convert an HTML string to Markdown using the Sphinx-tuned converter.

    Parameters
    ----------
    html_content : str
        Raw HTML string to convert.
    strip_tags : list of str or None, optional
        HTML tag names whose elements (including all their content) are
        removed before conversion.  Defaults to ``["script", "style"]``
        when ``None``.

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
    available the stripping step is skipped silently; ``markdownify``'s own
    ``strip=`` option still removes the listed tags from the output.

    The converter itself is cached between calls (module-level singleton).

    See Also
    --------
    _build_converter_class : Factory for the converter class.

    Examples
    --------
    >>> html_to_markdown("<h1>Hello</h1><p>World</p>")  # doctest: +SKIP
    '# Hello\\n\\nWorld\\n\\n'
    """
    if strip_tags is None:
        strip_tags = ["script", "style"]

    # Lazy bs4 import — avoids module-scope ImportError when bs4 is absent.
    # TypeError (BeautifulSoup is None) and ImportError are both handled here.
    try:
        from bs4 import BeautifulSoup as _BS  # noqa: PLC0415
        soup = _BS(html_content, "html.parser")
        for tag in soup(strip_tags):
            tag.decompose()
        html_content = str(soup)
    except ImportError:
        # bs4 not installed; markdownify's strip= option handles tag removal.
        pass

    ConverterClass = _build_converter_class()
    return ConverterClass(
        heading_style="ATX",
        bullets="*",
        strong_em_symbol="**",
        strip=list(strip_tags),
    ).convert(html_content)


# Legacy alias kept for backwards compatibility
html_to_markdown_converter = html_to_markdown


# ---------------------------------------------------------------------------
# Multi-process worker — must be module-level for pickling
# ---------------------------------------------------------------------------

#: CSS selectors tried in order to locate the main page content.
#: Each theme uses a different element; we probe until one matches.
_DEFAULT_CONTENT_SELECTORS: Tuple[str, ...] = (
    "article.bd-article",           # pydata-sphinx-theme ≥ 0.13
    'div[role="main"]',             # pydata-sphinx-theme (older), RTD, generic
    'article[role="main"]',         # Furo theme
    "div.rst-content",              # Read the Docs theme
    "div.document",                 # Sphinx Classic / Alabaster
    "div.body",                     # Older Sphinx themes
    "div.bd-article-container article",  # pydata nested wrapper
    "div.content",                  # Haiku / Scrolls
    "div.section",                  # Bootstrap / older Sphinx
    "main",                         # Generic HTML5
    "article",                      # Final fallback
)


def _process_single_html_file(
    args: Tuple[str, str, List[str], List[str], List[str]],
) -> Tuple[str, str, str]:
    """Process one HTML file and write the companion ``.md`` file.

    This function is intentionally **at module scope** so that it can be
    serialised by :mod:`multiprocessing` (``pickle`` requires module-level
    callables).

    Parameters
    ----------
    args : tuple
        A 5-tuple ``(html_file_path, outdir_path, exclude_patterns,
        selectors, strip_tags)``::

            html_file_path : str
                Absolute path of the ``.html`` file to convert.
            outdir_path : str
                Absolute path of the Sphinx HTML output directory.
            exclude_patterns : list of str
                Substrings; files whose relative path contains any of these
                are skipped without error.
            selectors : list of str
                CSS selectors tried in order to locate the page's main
                content.
            strip_tags : list of str
                HTML tag names removed (with content) before Markdown
                conversion (e.g. ``["script", "style", "nav"]``).

    Returns
    -------
    tuple of (str, str, str)
        ``(status, relative_path, message)`` where *status* is one of
        ``"success"``, ``"skipped"``, or ``"error"``.

    Notes
    -----
    **Security** — path-traversal guard:
    The function verifies that *html_file_path* resolves to a path within
    *outdir_path* before reading it.  Any path that escapes the output
    directory is rejected with status ``"error"``.

    **Encoding** — files are read and written as UTF-8.  Encoding errors
    are replaced with the Unicode replacement character (U+FFFD) to avoid
    worker crashes on malformed HTML.
    """
    html_file_str, outdir_str, exclude_patterns, selectors, strip_tags = args

    html_file = Path(html_file_str)
    outdir = Path(outdir_str)

    # ---- Security: path-traversal guard ------------------------------------
    if not _is_path_within(html_file, outdir):
        return ("error", str(html_file), "Path-traversal attempt blocked")

    try:
        rel_path = html_file.relative_to(outdir)
    except ValueError:
        return ("error", str(html_file), "File is outside output directory")

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

        md_file = html_file.with_suffix(".md")
        md_file.write_text(markdown_content, encoding="utf-8")

        return ("success", rel_str, "")

    except Exception as exc:  # noqa: BLE001
        return ("error", rel_str, str(exc))


# ---------------------------------------------------------------------------
# Build-time hooks
# ---------------------------------------------------------------------------

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

    Notes
    -----
    Config values read from *app.config*:

    * ``ai_assistant_generate_markdown`` — master switch.
    * ``ai_assistant_theme_preset`` — optional theme name to auto-select
      CSS selectors (e.g. ``"pydata_sphinx_theme"``).
    * ``ai_assistant_content_selectors`` — ordered CSS selectors for main
      content detection.  Merged with preset selectors when both are set.
    * ``ai_assistant_markdown_exclude_patterns`` — list of path substrings
      to skip.
    * ``ai_assistant_strip_tags`` — HTML tags removed before conversion.
    * ``ai_assistant_max_workers`` — maximum parallel worker processes
      (``None`` → auto-detect, capped at 8).
    """
    if exception is not None:
        return

    log = _get_logger()

    from sphinx.builders.html import StandaloneHTMLBuilder  # lazy sphinx import

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

    # Resolve CSS selectors: merge preset + custom, sanitize, deduplicate.
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
        (str(f), str(outdir), exclude_patterns, selectors, strip_tags)
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
            else:  # "error"
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

    Notes
    -----
    The base URL is resolved from, in priority order:

    1. ``html_baseurl`` (standard Sphinx config).
    2. ``ai_assistant_base_url`` (extension-specific override).
    3. Empty string — relative paths are used.

    Config values read from *app.config*:

    * ``ai_assistant_generate_llms_txt`` — master switch.
    * ``ai_assistant_llms_txt_max_entries`` — cap the number of entries
      written (``None`` → unlimited).
    * ``ai_assistant_llms_txt_full_content`` — when ``True``, embed the
      full Markdown content of each page inline in ``llms.txt``
      (``llms-full.txt`` convention).

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

    from sphinx.builders.html import StandaloneHTMLBuilder  # lazy

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

    # Apply entry cap if configured.
    max_entries: Optional[int] = getattr(
        app.config, "ai_assistant_llms_txt_max_entries", None
    )
    if max_entries is not None:
        cap = max(0, int(max_entries))
        md_files = md_files[:cap]
        if not md_files:
            log.debug(
                "AI Assistant: llms.txt max_entries=0; no entries to write"
            )
            return

    full_content: bool = bool(
        getattr(app.config, "ai_assistant_llms_txt_full_content", False)
    )

    llms_txt = outdir / "llms.txt"
    project_name: str = getattr(app.config, "project", "Documentation")
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
    **Security**: The configuration dict is serialised with
    :func:`_safe_json_for_script` which escapes ``</script>`` sequences,
    preventing script-injection attacks via adversarially crafted config
    values.

    **Position validation**: ``ai_assistant_position`` is validated against
    :data:`_ALLOWED_POSITIONS`.  Invalid values are replaced with
    ``"sidebar"`` and a warning is logged.

    **Provider filtering**: Providers whose ``url_template`` does not use
    the ``http://`` or ``https://`` scheme are silently removed from the
    serialised config to prevent ``javascript:`` or ``data:`` injection
    via the browser-side widget.
    """
    if not app.config.ai_assistant_enabled:
        return

    # ---- Validate widget position -----------------------------------------
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

    # ---- Filter providers with unsafe URL templates -----------------------
    providers_raw: Dict[str, Any] = dict(app.config.ai_assistant_providers)
    providers_safe: Dict[str, Any] = {
        name: prov
        for name, prov in providers_raw.items()
        if _validate_provider_url_template(str(prov.get("url_template", "")))
    }

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

    # ---- AI provider configuration ----------------------------------------
    app.add_config_value(
        "ai_assistant_providers",
        {
            "claude": {
                "enabled": True,
                "label": "Ask Claude",
                "description": "Open AI chat with this page context",
                "icon": "claude.svg",
                "url_template": "https://claude.ai/new?q={prompt}",
                "prompt_template": (
                    "Hi! Please read this documentation page: {url}\n\n"
                    "I have questions about it."
                ),
            },
            "chatgpt": {
                "enabled": True,
                "label": "Ask ChatGPT",
                "description": "Open AI chat with this page context",
                "icon": "chatgpt.svg",
                "url_template": "https://chatgpt.com/?q={prompt}",
                "prompt_template": (
                    "Read {url} so I can ask questions about it."
                ),
            },
        },
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
