# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_example_conf.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# This module was adapted from the sphinx-ai-assistant project.
# https://github.com/mlazag/sphinx-ai-assistant/blob/main/example_conf.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Example Sphinx ``conf.py`` for the AI-assistant extension.

This file documents every available configuration value with its type,
default, and purpose.  It is tuned for the **pydata-sphinx-theme**
(scikit-learn / NumPy style) but all settings are theme-agnostic unless
noted otherwise.

Usage:

Copy the relevant sections into your project's ``conf.py`` and adjust.

Theme note:

pydata-sphinx-theme renders the main article content inside::

    <article class="bd-article" role="main">

so the CSS selectors are configured accordingly.  For other themes see the
comments next to each selector value.

Quick-start minimal config:

The absolute minimum to enable the extension in ``conf.py``::

    extensions = ["scikitplot._externals._sphinx_ext._sphinx_ai_assistant"]
    ai_assistant_enabled = True
    html_baseurl = "https://docs.example.com"

All other values shown below are optional — the extension ships with sensible
defaults for every setting.

.. seealso:
  * https://huggingface.co/spaces/scikit-plots/ai/tree/main
"""

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
# Add the AI-assistant extension to the Sphinx build.
#
# Developer note: The extension entry point is ``setup(app)`` inside
# ``__init__.py``.  Sphinx calls it automatically when the dotted name is
# listed here.  The extension registers config values, static files, and
# three build-event hooks:
#   * ``html-page-context`` -> ``add_ai_assistant_context``
#         Injects the ``window.AI_ASSISTANT_CONFIG`` script block and the
#         widget CSS/JS into every HTML page.
#   * ``build-finished``    -> ``generate_markdown_files``
#         Writes a ``.md`` companion next to every ``.html`` file using a
#         parallel ProcessPoolExecutor worker pool.
#   * ``build-finished``    -> ``generate_llms_txt``
#         Writes ``llms.txt`` listing all generated ``.md`` URLs.
extensions = [
    # ... your other extensions ...
    "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
]

# ---------------------------------------------------------------------------
# HTML theme — pydata-sphinx-theme (scikit-learn / NumPy / SciPy style)
# ---------------------------------------------------------------------------
# The pydata-sphinx-theme is the standard theme for major scientific-Python
# projects (scikit-learn, NumPy, SciPy, pandas, Matplotlib).
#
# The extension supports every Sphinx theme through its theme-preset system.
# All preset names accepted by ``ai_assistant_theme_preset`` (below):
#
#   Sphinx themes:
#     "pydata_sphinx_theme"  — scikit-learn / NumPy / SciPy / pandas style
#     "furo"                 — Furo theme (https://pradyunsg.me/furo/)
#     "sphinx_rtd_theme"     — Read the Docs theme
#     "sphinx_book_theme"    — Jupyter Book / sphinx-book-theme
#     "alabaster"            — Sphinx default theme
#     "classic"              — Sphinx Classic
#     "nature"               — Sphinx Nature theme
#     "agogo"                — Sphinx Agogo theme
#     "haiku"                — Sphinx Haiku theme
#     "scrolls"              — Sphinx Scrolls theme
#     "press"                — Sphinx Press theme
#     "bootstrap"            — sphinx-bootstrap-theme
#     "piccolo_theme"        — sphinx-piccolo-theme
#     "insegel"              — Insegel theme
#     "groundwork"           — Groundwork theme
#
#   Non-Sphinx static site generators:
#     "mkdocs"               — MkDocs (non-Sphinx, standalone use)
#     "mkdocs_material"      — MkDocs Material
#     "jekyll"               — Jekyll static sites
#     "hugo"                 — Hugo static sites
#     "hexo"                 — Hexo static sites
#     "docusaurus"           — Docusaurus (Meta)
#     "vitepress"            — VitePress (Vue-powered)
#     "gitbook"              — GitBook
#     "plain_html"           — Generic HTML5 without a framework
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # ---- Navigation --------------------------------------------------------
    # Alignment of the top-level navigation items: "left" | "right" | "content".
    "navbar_align": "left",

    # Number of sidebar navigation levels expanded by default.
    # Increase for projects with deep module hierarchies (e.g. 4 for APIs).
    "show_nav_level": 2,

    # Maximum depth the sidebar TOC renders (collapsed levels show as links).
    "navigation_depth": 4,

    # Number of in-page TOC levels visible in the right-hand column.
    "show_toc_level": 2,

    # ---- Search ------------------------------------------------------------
    # Placeholder text rendered inside the top-bar search input.
    "search_bar_text": "Search the docs …",

    # ---- Social / header icons --------------------------------------------
    # Icon links rendered in the top-right header area.
    # Each dict requires: "name" (tooltip), "url", "icon" (Font Awesome class).
    # Full icon list: https://fontawesome.com/icons
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/your-org/your-project",
            "icon": "fa-brands fa-github",
        },
        # Add PyPI, Discord, Twitter/X, or Binder links the same way.
    ],

    # ---- Footer -----------------------------------------------------------
    # Components shown on the left / right of the footer bar.
    # Built-in names: "copyright", "sphinx-version", "last-updated",
    # "theme-version".
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],

    # ---- pydata extras ----------------------------------------------------
    # Renders an "Edit this page" button linking to the source on GitHub/GitLab.
    # Requires the ``html_context`` dict below with repo details.
    "use_edit_page_button": False,

    # Shows a banner warning users they are on a non-stable docs version.
    # Reads a version-switcher JSON file to decide whether to display the banner.
    "show_version_warning_banner": True,
}

# Context variables consumed by the edit-page button and by some AI-assistant
# prompt templates when constructing absolute page URLs.
html_context = {
    "github_user": "your-org",
    "github_repo": "your-project",
    "github_version": "main",   # branch the docs are built from
    "doc_path": "docs",         # path to the docs directory inside the repo
}

# Base URL of the DEPLOYED documentation site.
#
# User note: This is the MOST IMPORTANT setting for the AI-assistant.
# It serves two purposes:
#   1. Constructs the absolute page URL passed to AI providers in prompts
#      (e.g. "Please read https://docs.example.com/api/module.html").
#   2. Prefixes ``.md`` file paths in ``llms.txt`` so LLM tools can fetch
#      the Markdown version of each page over the network.
#
# Omitting both ``html_baseurl`` and ``ai_assistant_base_url`` causes
# ``llms.txt`` to contain only relative paths, which AI tools cannot
# resolve without additional context.
#
# Developer note: ``add_ai_assistant_context`` reads ``app.config.html_baseurl``
# first, then falls back to ``ai_assistant_base_url`` when that is empty.
html_baseurl = "https://docs.example.com"

# ---------------------------------------------------------------------------
# AI Assistant — master switch
# ---------------------------------------------------------------------------

# Type:    bool
# Default: True
#
# Master on/off switch for the entire extension.  When False:
#   * No widget JavaScript or CSS is injected into HTML pages.
#   * No ``.md`` companion files are generated.
#   * No ``llms.txt`` is written.
#
# User note: Set False for local or draft builds to speed up the build and
# avoid generating files that won't be deployed.  The extension remains in
# ``extensions`` so the Sphinx config stays valid across environments; just
# override this value in a CI or local ``conf.py`` override file.
ai_assistant_enabled = True

# ---------------------------------------------------------------------------
# AI Assistant — button placement
# ---------------------------------------------------------------------------

# Type:    str — one of "sidebar" | "title" | "floating" | "none"
# Default: "sidebar"
#
# Controls where the AI-assistant button/panel is inserted in each HTML page.
#
#   "sidebar"  — Right sidebar, above the in-page TOC.
#                Best for pydata-sphinx-theme and furo.  Always visible on
#                desktop without cluttering the main content area.
#
#   "title"    — Inline, adjacent to the page's <h1> heading.
#                Works for all themes.  Good when there is no right sidebar
#                (e.g. landing pages, Classic theme).
#
#   "floating" — Fixed button anchored to the bottom-right viewport corner.
#                Always visible regardless of scroll position.  May overlap
#                other fixed UI elements (cookie banners, live-chat widgets).
#
#   "none"     — No visible button is rendered.  The extension still injects
#                ``window.AI_ASSISTANT_CONFIG`` so custom JavaScript can
#                activate the widget programmatically.
#
# Developer note: ``add_ai_assistant_context`` validates this value via
# ``_validate_position``, which lowercases the input and raises ``ValueError``
# for any string not in ``_ALLOWED_POSITIONS`` before it reaches the widget.
ai_assistant_position = "sidebar"

# ---------------------------------------------------------------------------
# AI Assistant — content selector for client-side Markdown conversion
# ---------------------------------------------------------------------------

# Type:    str  (single CSS selector)
# Default: "article"
#
# The CSS selector the browser-side JavaScript widget uses to locate the
# main content area of the current page.  Used by the "Copy as Markdown"
# button and by AI-provider prompt templates that embed ``{content}``.
#
# Only ONE selector is evaluated client-side (the widget does not try
# multiple fallbacks).  Choose the most specific selector that matches the
# main prose element of your theme on every page.
#
# Theme-specific values:
#   pydata-sphinx-theme:   "article.bd-article"
#   Furo theme:            'article[role="main"]'
#   Alabaster / Classic:   "div.document"
#   Read the Docs:         'div[role="main"]'
#   sphinx-book-theme:     "article.bd-article"
#
# Security note: ``_validate_css_selector`` rejects selectors containing
# ``<`` or ``>`` to prevent HTML-injection into the ``window.AI_ASSISTANT_CONFIG``
# script block that is embedded verbatim in every page.
ai_assistant_content_selector = "article.bd-article"

# ---------------------------------------------------------------------------
# AI Assistant — CSS selectors for server-side Markdown extraction
# ---------------------------------------------------------------------------

# Type:    list[str]  (ordered CSS selectors)
# Default: _DEFAULT_CONTENT_SELECTORS  (16 built-in selectors covering every
#          major Sphinx theme, MkDocs, Jekyll, Hugo, and plain HTML)
#
# The BUILD-TIME Markdown generator probes each selector in order and uses
# the FIRST match found in each HTML file.  This list applies globally to
# every page in the documentation tree.
#
# User note: Put the most specific selector for your theme FIRST.  Generic
# fallbacks ("main", "article") should be last so they only apply to pages
# that do not match the primary selector (custom landing pages, 404 pages).
#
# Developer note: Selectors are validated by ``_sanitize_selectors`` which
# calls ``_validate_css_selector`` on each entry and silently drops any that
# contain ``<`` or ``>``.  Workers call ``BeautifulSoup.select_one(selector)``
# in order until a match is found; pages with no match are counted as
# "skipped" in the stats dict returned by ``process_html_directory``.
ai_assistant_content_selectors = [
    "article.bd-article",      # pydata-sphinx-theme >= 0.13
    'div[role="main"]',        # pydata-sphinx-theme (older), RTD
    'article[role="main"]',    # Furo
    "div.document",            # Classic / Alabaster
    "main",                    # Generic HTML5
    "div.body",                # Very old Sphinx themes
    "article",                 # Last-resort fallback
]

# Type:    str or None
# Default: None
#
# Auto-select CSS selectors for a known theme by name.  The preset's selectors
# are MERGED with ``ai_assistant_content_selectors``:
#   1. custom selectors  (highest priority)
#   2. preset selectors
#   3. module defaults   (lowest priority — _DEFAULT_CONTENT_SELECTORS)
#
# Deduplication is applied so each selector appears at most once.
#
# Supported preset names (see ``_THEME_SELECTOR_PRESETS`` in ``__init__.py``):
#   Sphinx themes:  "pydata_sphinx_theme", "furo", "sphinx_rtd_theme",
#                   "sphinx_book_theme", "alabaster", "classic", "nature",
#                   "agogo", "haiku", "scrolls", "press", "bootstrap",
#                   "piccolo_theme", "insegel", "groundwork"
#   Non-Sphinx:     "mkdocs", "mkdocs_material", "jekyll", "hugo",
#                   "hexo", "docusaurus", "vitepress", "gitbook", "plain_html"
#
# Developer note: ``_resolve_content_selectors(preset, custom_selectors)``
# looks up the preset in ``_THEME_SELECTOR_PRESETS``, merges the three
# sources, sanitises the result, and always guarantees a non-empty tuple.
# An unknown preset name is silently ignored (no error raised).
# ai_assistant_theme_preset = "pydata_sphinx_theme"

# ---------------------------------------------------------------------------
# AI Assistant — Markdown file generation
# ---------------------------------------------------------------------------

# Type:    bool
# Default: True
#
# When True, a ``.md`` companion file is written next to every ``.html``
# file immediately after the Sphinx build completes.  These Markdown files
# are what AI providers read when given a documentation page URL — they are
# far smaller and cleaner than raw HTML, and they contain only the prose
# content without navigation chrome.
#
# Requires:  pip install beautifulsoup4 markdownify
#   * beautifulsoup4 — HTML parsing and CSS selector matching.
#   * markdownify    — HTML-to-Markdown conversion.
#
# Developer note: ``generate_markdown_files`` is wired to Sphinx's
# ``build-finished`` event.  It discovers all ``.html`` files in the output
# directory, builds an args list, and submits each file to a
# ``ProcessPoolExecutor`` worker pool.  Each worker:
#   1. Checks the file is within the output directory (path-traversal guard).
#   2. Skips files whose relative path matches ``ai_assistant_markdown_exclude_patterns``.
#   3. Parses the HTML with BeautifulSoup and tries each CSS selector in
#      ``ai_assistant_content_selectors`` until a match is found.
#   4. Strips unwanted tags via ``ai_assistant_strip_tags``.
#   5. Converts the remaining HTML to Markdown via ``SphinxMarkdownConverter``
#      (a markdownify subclass tuned for Sphinx code blocks and admonitions).
#   6. Writes the ``.md`` file alongside the ``.html`` file (inline mode)
#      or to a separate output directory (standalone mode).
ai_assistant_generate_markdown = True

# Type:    list[str]  (path substrings)
# Default: ["genindex", "search", "py-modindex"]
#
# HTML files whose relative path contains any of these substrings are skipped
# during Markdown generation.  The check is a case-sensitive substring match
# on the POSIX-style relative path from the output directory root.
#
# User note: Add patterns for pages that are generated by Sphinx but contain
# no useful prose — index pages, download pages, large auto-generated tables,
# or changelog pages that would produce enormous Markdown files.  Skipped
# files are counted in the "skipped" key of the stats dict but generate no
# warnings.
#
# Developer note: The check is performed in the worker before any file I/O
# so skipped files have negligible CPU cost.  The worker returns
# ``("skipped", rel_path, "")`` and the orchestrator increments the skip
# counter in the progress stats.
ai_assistant_markdown_exclude_patterns = [
    "genindex",
    "search",
    "py-modindex",
    "_sources",     # Sphinx rst source download files — no prose
    "_static",      # CSS, JS, images — not documentation text
]

# Type:    list[str]  (HTML tag names, lower-case)
# Default: ["script", "style", "nav", "footer"]
#
# HTML elements whose entire subtree (tag + all nested children) is removed
# BEFORE Markdown conversion.  This eliminates site chrome that would
# otherwise pollute the Markdown output with navigation links, cookie notices,
# and boilerplate footer text.
#
# User note: Add tag names for any site-specific elements embedded inside the
# main content selector that should not appear in the Markdown.  Common
# additions: "aside" (sidebars), "header" (page headers inside articles),
# "form" (embedded search forms), "table.field-list" (not a tag but shows the
# principle — use the CSS selector approach for non-tag elements).
#
# Developer note: Removal is performed by ``BeautifulSoup.Tag.decompose()``
# inside the worker, operating on the parsed DOM before markdownify processes
# it.  The same list is also passed to markdownify's ``strip=`` option as a
# secondary safety net for cases where BeautifulSoup is unavailable.
ai_assistant_strip_tags = ["script", "style", "nav", "footer"]

# Type:    int or None
# Default: 1  (single worker; set None for auto cpu_count)
#
# Maximum number of parallel worker processes for Markdown generation.
#
#   1     — single process (extension default).  Use for CI and debugging —
#           stack traces are clean and sequential, and pdb works normally
#           inside workers.
#   None  — auto: os.cpu_count() or 1.  Scales to the local machine
#           without overwhelming CI containers (which often have 2 CPUs).
#           Pass ``None`` or ``"auto"`` to enable auto-detection.
#   N > 1 — explicit cap.  Useful to limit memory usage on memory-constrained
#           machines when converting very large HTML files.
#
# Developer note: Workers are spawned by ``ProcessPoolExecutor``.  The
# worker functions (``_process_single_html_file``, ``_process_html_file_worker``)
# MUST be defined at module scope — not as lambdas or closures — so the
# multiprocessing pickle protocol can serialise them across process boundaries.
# ``_resolve_max_workers`` translates ``None`` / ``"auto"`` / positive integers
# to a safe value before the executor is created; it raises ``ValueError`` for
# zero or negative inputs so errors surface before any process is spawned.
ai_assistant_max_workers = 1  # None / "auto" or a positive integer

# ---------------------------------------------------------------------------
# AI Assistant — Ollama default model
# ---------------------------------------------------------------------------

# Type:    str
# Default: "llama3.2:latest"
#
# Default Ollama model used when the ``ollama`` provider is enabled and no
# ``model`` override is given inside ``ai_assistant_providers["ollama"]``.
# The widget JS reads this value to pre-select the model in the Ollama panel.
#
# User note: Pull any model from the recommended list with
# ``ollama pull <model>`` before setting it here.
#
# Recommended models (all freely available via ``ollama pull``):
#   Anthropic-API-compatible:
#     "qwen3:latest"         — Alibaba Qwen 3 (excellent code + reasoning)
#     "qwen3:8b"             — Qwen 3 8 B (fast, lower RAM)
#     "llama3.3:latest"      — Meta Llama 3.3 70 B (strong general model)
#     "llama3.2:latest"      — Meta Llama 3.2 (lightweight, fast)
#     "glm4:latest"          — THUDM GLM-4 (strong multilingual)
#   Google Gemma (Apache-2.0):
#     "gemma3:latest"        — Google Gemma 3 (predecessor; Gemma 4 in progress)
#     "gemma3:12b"           — Gemma 3 12 B
#     "gemma3:4b"            — Gemma 3 4 B (very fast)
#   Code-specialised:
#     "codellama:latest"     — Meta Code Llama
#     "deepseek-r1:latest"   — DeepSeek R1 reasoning model (locally)
#     "deepseek-coder-v2:latest" — DeepSeek Coder V2
#     "phi4:latest"          — Microsoft Phi-4
#     "phi4-mini:latest"     — Phi-4 Mini (very fast)
#     "mistral:latest"       — Mistral 7 B
#
# The full list is available as ``_OLLAMA_RECOMMENDED_MODELS`` in __init__.py.
#
# Developer note: This value is passed by ``add_ai_assistant_context`` into
# the per-page ``window.AI_ASSISTANT_CONFIG`` JSON object under the
# ``ollamaModel`` key.  It can be overridden at runtime by the user via the
# widget UI.  Registered in ``setup()`` as a Sphinx config value with
# ``rebuild = "html"``.
ai_assistant_ollama_model = "llama3.2:latest"

# ---------------------------------------------------------------------------
# AI Assistant — llms.txt generation
# ---------------------------------------------------------------------------

# Type:    bool
# Default: True
#
# When True, an ``llms.txt`` index file is written to the documentation
# output root after the Sphinx build.  ``llms.txt`` is the emerging standard
# (https://llmstxt.org/) for giving AI tools a machine-readable index of all
# documentation pages in Markdown format.
#
# Each entry in ``llms.txt`` is either:
#   * a relative path  — when no base URL is configured: "api/module.md"
#   * an absolute URL  — when base URL is set:
#                        "https://docs.example.com/api/module.md"
#
# Developer note: ``generate_llms_txt`` is wired to ``build-finished``.
# It calls ``generate_llms_txt_standalone`` internally, which is also
# available as a public standalone function for non-Sphinx pipeline use.
# The function finds all ``.md`` files in the output directory recursively,
# sorts them alphabetically, caps the list if ``ai_assistant_llms_txt_max_entries``
# is set, and writes one entry per line (or a full-content block when
# ``ai_assistant_llms_txt_full_content`` is True).
ai_assistant_generate_llms_txt = True

# Type:    str
# Default: ""  (falls back to html_baseurl)
#
# Explicit base URL prepended to ``.md`` file paths in ``llms.txt``.
# When empty (the default), both ``generate_llms_txt`` and
# ``add_ai_assistant_context`` automatically use ``html_baseurl``.
#
# User note: Only set this when your ``.md`` files are served from a
# DIFFERENT domain or path than the HTML documentation (e.g. a CDN for raw
# Markdown, separate from the HTML docs domain).  In most deployments
# ``html_baseurl`` is sufficient — leave this empty.
#
# Security note: ``_validate_base_url`` rejects any non-empty value that
# does not start with ``http://`` or ``https://`` and raises ``ValueError``
# so misconfigured builds fail loudly at startup rather than silently writing
# broken URLs into ``llms.txt``.
ai_assistant_base_url = ""  # use html_baseurl above

# Type:    int or None
# Default: None  (unlimited)
#
# Cap on the number of entries written to ``llms.txt``.
# None means every discovered ``.md`` file is included.
#
# User note: Use a positive integer limit for very large documentation sites
# where the full ``llms.txt`` would exceed practical LLM context-window
# budgets.  The cap is applied AFTER alphabetical sorting, so the included
# entries are always the lexicographically first N files (typically root-level
# pages and the beginning of the API reference).
ai_assistant_llms_txt_max_entries = None

# Type:    bool
# Default: False
#
# When True, the full Markdown content of every page is embedded INLINE in
# ``llms.txt``, separated by ``---`` markers, producing a single self-contained
# file that AI tools can load into their context without additional HTTP requests.
#
# User note: This produces a VERY large file for sites with many pages.
# Rough estimate: 500 pages × 10 KB/page ≈ 5 MB ``llms.txt``.
# Enable only for small-to-medium projects, or combine with
# ``ai_assistant_llms_txt_max_entries`` to cap the total size.
#
# Developer note: Full content is read lazily — one ``.md`` file at a time —
# inside ``generate_llms_txt_standalone``, so peak memory usage is bounded by
# the size of the largest single page, not the entire site.
ai_assistant_llms_txt_full_content = True

# ---------------------------------------------------------------------------
# AI Assistant — feature flags
# ---------------------------------------------------------------------------

# Type:    dict[str, bool]
# Default: {"markdown_export": True, "view_markdown": True,
#           "ai_chat": True, "mcp_integration": True}
#
# Granular feature toggles that control which UI sections the widget renders.
# Set individual keys to False to hide that section without disabling the
# entire extension.
#
# Key descriptions:
#
#   "markdown_export"
#       Show the "Copy as Markdown" button.  This is the primary widget
#       action: the widget extracts the page content identified by
#       ``ai_assistant_content_selector``, converts it to Markdown via a
#       browser-side Turndown/Markdownify call, and writes it to the
#       clipboard.  Disable if clipboard access is blocked by your CSP.
#
#   "view_markdown"
#       Show a "View Markdown" link that opens the pre-generated ``.md``
#       companion file in a new browser tab.  The URL is computed as:
#       ``html_baseurl + current_page_path + ".md"``.
#       Requires ``ai_assistant_generate_markdown = True`` at build time so
#       the ``.md`` files actually exist on the server.
#
#   "ai_chat"
#       Show the row of AI-provider buttons (ChatGPT, Claude, Gemini, …).
#       Each button opens the provider's web UI with a pre-filled prompt
#       containing the page URL and/or Markdown content.
#       Set False to remove all provider buttons while keeping the widget.
#
#   "mcp_integration"
#       Show MCP server "Connect" buttons (VS Code, Claude Desktop, Cursor,
#       Windsurf).  These let users install the documentation's MCP server
#       into their AI coding tools directly from the documentation page.
#       Requires at least one enabled entry in ``ai_assistant_mcp_tools``.
#       Set False to hide all MCP buttons without modifying that dict.
#
# Developer note: Feature flags are serialised into the per-page
# ``window.AI_ASSISTANT_CONFIG.features`` JSON object by
# ``_safe_json_for_script`` and consumed by the widget JavaScript at runtime.
# No server round-trip is needed — the widget reads the flags from the
# inline script block injected by ``add_ai_assistant_context``.
ai_assistant_features = {
    # ── Core export features ──────────────────────────────────────────────
    # Copy the current page as Markdown (main button + dropdown item).
    "markdown_export": True,
    # Open the companion .md file in a new tab.
    "view_markdown": True,

    # ── AI chat deep-links ────────────────────────────────────────────────
    # Show enabled AI provider buttons (Claude, ChatGPT, Gemini, …).
    "ai_chat": True,

    # ── MCP integration ───────────────────────────────────────────────────
    # Show MCP "Connect" buttons for VS Code, Claude Desktop, etc.
    # Requires at least one enabled entry in ``ai_assistant_mcp_tools``.
    "mcp_integration": True,   # False ← disable MCP section on this site

    # ── Theme toggle ──────────────────────────────────────────────────────
    # Dark / light / system color-scheme toggle button.
    "theme_toggle": True,   # False ← hide if the theme handles it already

    # ── PDF export ────────────────────────────────────────────────────────
    # "Export as PDF" centered button + URL/Print mode toggle.
    # IMPORTANT: must be set explicitly — the JS default is True, but relying
    # on JS defaults means conf.py cannot disable it cleanly.
    "pdf_export": True,

    # ── AI panel ─────────────────────────────────────────────────────────
    # Floating in-page AI assistant chat panel (slide-in drawer).
    # IMPORTANT: this key MUST be present and True for the panel button to
    # appear in the dropdown.  The JS safe default is False (opt-in) — if
    # this key is absent from conf.py the panel is invisible.
    # Root cause of BUG-1: partial ai_assistant_features dict that omits
    # this key → JS defaults ai_panel = False → button never created.
    "ai_panel": True,
}

# ---------------------------------------------------------------------------
# AI Assistant — prompt customisation
# ---------------------------------------------------------------------------

# Type:    str or None
# Default: None
#
# Short goal annotation prepended to every AI prompt as "Goal: <text>\n\n".
# Use this to tell the AI WHY the user is reading the documentation so its
# answers are more targeted from the start.
#
# User note: Keep this to one sentence or phrase.  It is prepended to every
# prompt sent to every enabled provider, so verbose values inflate all prompts
# and consume context-window tokens on every interaction.
#
# Example:
#   ai_assistant_intention = "Help me understand the API and give code examples."
ai_assistant_intention = None

# Type:    str or None
# Default: None
#
# Background context string injected into AI prompts after the goal annotation.
# Use this to describe the project, its audience, or domain so AI providers
# calibrate their responses without requiring users to provide this context
# themselves on every session.
#
# User note: Describe the project type, primary language, and target audience.
# This context is sent with every prompt, so information that is already
# obvious from the page content (e.g. "this is a Python library") is
# redundant — include only what is NOT deducible from the documentation text.
#
# Example:
#   ai_assistant_custom_context = (
#       "scikitplot is a Python library for visualising ML metrics. "
#       "Users are data scientists familiar with scikit-learn and Matplotlib."
#   )
ai_assistant_custom_context = None

# Type:    str or None
# Default: None
#
# Raw string prepended verbatim at the very BEGINNING of every prompt, before
# the goal annotation and custom context.  Use for system-level instructions
# that must precede all other content (e.g. persona instructions, output
# format constraints).
#
# User note: This is an advanced escape hatch for careful prompt engineering.
# Most projects do not need it — prefer ``ai_assistant_intention`` and
# ``ai_assistant_custom_context`` which have structured, predictable positions
# in the final prompt.
#
# Example:
#   ai_assistant_custom_prompt_prefix = "You are a helpful Python expert.\n\n"
ai_assistant_custom_prompt_prefix = None

# Type:    bool
# Default: False
#
# When True, the widget captures a screenshot of the current page (via the
# browser's Canvas API) and encodes it as a base64 data-URI that is appended
# to the prompt sent to AI providers that accept image input (e.g. ChatGPT-4o,
# Claude, Gemini).
#
# User note: Enable only when visual context (diagrams, rendered plots,
# layout screenshots) is genuinely useful for the documentation page type.
# Each screenshot inflates the prompt significantly — typically 100–400 KB
# base64 per page — and consumes the provider's vision-token budget.  Most
# text-heavy documentation pages gain nothing from this option.
#
# Security note: Raw image data is embedded directly in the URL-encoded
# prompt or POST body.  Ensure your CSP permits the ``data:`` URI scheme for
# the image source if you restrict it.  The extension never uploads images to
# Anthropic-controlled servers — all data goes directly from the user's
# browser to the chosen AI provider.
#
# Developer note: ``add_ai_assistant_context`` reads this value via
# ``_cfg_bool`` and serialises it into
# ``window.AI_ASSISTANT_CONFIG.includeRawImage``.  The widget JS then
# conditionally captures a screenshot when the user clicks an AI-provider
# button.  Registered in ``setup()`` with ``rebuild = "html"``.
ai_assistant_include_raw_image = False

# ---------------------------------------------------------------------------
# AI Assistant — PDF export
# ---------------------------------------------------------------------------

# Type:    str or None
# Default: None  (→ window.print() / browser Save as PDF)
#
# URL opened when the user clicks "Export as PDF" in URL mode.
#
# User note:
#   Empty string / None → the button triggers ``window.print()`` and the user
#   can choose "Save as PDF" in the browser print dialog.  This works on every
#   Sphinx deployment with zero configuration.
#
#   Non-empty string → the button opens that URL in a new tab.  Use this when
#   your deployment generates server-side PDFs (e.g. a CI artifact, a
#   GitBook-style endpoint, or a custom PDF route):
#
#     ai_assistant_pdf_export_url = "https://docs.example.com/pdf/{pagename}.pdf"
#     ai_assistant_pdf_export_url = "~gitbook/pdf?version=stable"
#
# Developer note: serialised into ``window.AI_ASSISTANT_CONFIG.pdfExportUrl``
# by ``_cfg_str``.  The JS ``handlePdfExport`` reads the mode from
# ``sessionStorage`` (persisted by the toggle) and either opens this URL or
# calls ``window.print()``.
ai_assistant_pdf_export_url = None  # None / "" → window.print()

# Type:    bool
# Default: True
#
# Show the URL / Print mode toggle row below the "Export as PDF" button.
#
# User note:
#   True  (default) — the toggle is visible; the user can switch between
#                     URL mode (opens pdfExportUrl) and Print mode
#                     (window.print()) interactively.  Their choice is
#                     persisted in sessionStorage for the session.
#   False           — the toggle is hidden; the button always uses the mode
#                     inferred from ``ai_assistant_pdf_export_url``
#                     (URL mode when non-empty, Print mode otherwise).
#
# Developer note: serialised into ``window.AI_ASSISTANT_CONFIG.pdfUrlModeToggle``.
# The JS ``createPdfSection`` reads this to decide whether to render the
# toggle row.  The JS default is ``cfg.pdfUrlModeToggle !== false`` → always
# true when the key is absent.  Registering it here makes conf.py control
# explicit and avoids the implicit JS default.
ai_assistant_pdf_url_mode_toggle = True

# ---------------------------------------------------------------------------
# AI Assistant — AI panel (floating chat drawer)
# ---------------------------------------------------------------------------

# Type:    str
# Default: "AI Assistant"
#
# Title text displayed in the floating AI panel's header bar.
#
# User note: Keep short — typically your project name + "Assistant", e.g.
# "scikit-plots AI Assistant" or just "AI Assistant".
ai_assistant_panel_title = "AI Assistant"

# Type:    str
# Default: "Ask a question about this page…"
#
# Placeholder text shown inside the panel's textarea input field.
ai_assistant_panel_placeholder = "Ask a question about this page\u2026"

# Type:    bool
# Default: False
#
# When True, the AI panel posts the user's question together with the current
# page's Markdown content to the Anthropic /v1/messages API and streams the
# response into the chat window.
#
# When False (default), the panel renders as a UI stub: the send button
# appears functional, a simulated 400 ms "thinking" delay occurs, and then
# a stub message is shown explaining that API mode is disabled.  This is
# useful for demonstrating the UI without incurring API costs.
#
# IMPORTANT — production deployment:
#   The browser-side fetch goes directly to ``https://api.anthropic.com/v1/messages``.
#   The Anthropic API requires an API key via the ``x-api-key`` header.
#   To avoid exposing your API key in the browser bundle, route requests
#   through a reverse proxy that injects the header server-side, e.g.:
#
#     Nginx:
#       location /api/anthropic/ {
#           proxy_pass https://api.anthropic.com/;
#           proxy_set_header x-api-key "$ANTHROPIC_API_KEY";
#       }
#
#   Then point the JS fetch URL at your proxy endpoint.  The simplest safe
#   deployment is ``False`` (stub mode) for public docs.
ai_assistant_panel_api_enabled = False

# Type:    list[str]
# Default: []   (no chips shown)
#
# Quick-suggestion chips displayed in the panel welcome screen.
# When the user clicks a chip its text is pre-filled into the input textarea.
# Provide 0–5 short, actionable questions relevant to your documentation.
#
# Example::
#
#     ai_assistant_panel_quick_questions = [
#         "What does this module do?",
#         "Show me a usage example.",
#         "What are the main parameters?",
#     ]
#
# User note:
#   • Keep each string short (< 60 chars) so it fits on one chip line.
#   • More than 5 items are silently truncated to 5.
#   • Set to [] (empty list) to hide the chips entirely.
#
# Developer note:
#   The list is forwarded verbatim as ``panelQuickQuestions`` in the JS
#   config object.  The JS reads it with Array.isArray guard and slices
#   to a maximum of 5 items before rendering chip buttons.
ai_assistant_panel_quick_questions = [
    "What does this page cover?",
    "Show me a quick usage example.",
    "What are the key parameters?",
]

# Type:    bool
# Default: True
#
# When True (default), the panel shows a "Speak with your assistant" pill
# banner above the input area (only in browsers that support Web Speech API).
# Clicking the banner or the microphone icon inside the input starts voice
# recognition; the transcribed text is inserted into the input field.
#
# When False, all speech-related UI is hidden regardless of browser support.
#
# Browser support: Chrome, Edge, Safari (desktop + iOS). Firefox does NOT
# support the Web Speech API and will never show the banner even when True.
#
# User note:
#   No audio is sent to any server — speech recognition runs entirely in
#   the browser via the platform's built-in speech engine.
#   The mic button inside the input group is also hidden when False.
ai_assistant_panel_speak_banner = True

# Type:    str
# Default: "Ask Us"
#
# Label displayed on the floating trigger pill that appears at the bottom-right
# corner of the viewport when the user minimizes the panel (rather than closing
# it).  Clicking the pill restores the panel with the conversation intact.
#
# User note: Keep very short (1–3 words).  Combined with a chat icon SVG.
ai_assistant_panel_trigger_label = "Ask Us"

# Type:    bool
# Default: True
#
# Controls whether the floating trigger pill is created eagerly on every page
# load (True, default) or only after the user has opened then minimized the
# panel (False).
#
# User note: When True the "Ask Us" pill is immediately visible at the
# bottom-right of every page so readers can open the AI assistant with a
# single click.  When False users must first click the toolbar expand button
# then "AI Assistant" before the pill appears — two extra clicks.
#
# Developer note: The eager path runs inside createAIAssistantUI() at init
# time and is gated by ``features.ai_panel`` so the pill is never rendered
# on builds where the panel feature is disabled.  The idempotency guard in
# createAIPanel() (C-4) ensures a second pill is never created.
ai_assistant_panel_start_minimized = True

# ===========================================================================
# v0.3 — resize, persistence, shortcut, proxy, feedback, privacy, search-bar
# ===========================================================================
#
# Developer note: every key in this section is read by ai-assistant.js via
# ``window.AI_ASSISTANT_CONFIG``.  setup() registers it, add_ai_assistant_
# context() serialises it, the JS consumes it.  All three layers must agree
# (the "config flows one way" invariant); a parity test enforces it.
#
# Every default below reproduces the pre-v0.3 behaviour, so simply upgrading
# changes nothing until you opt in.

# ── R1: panel resize ──────────────────────────────────────────────────────
# The panel gains a drag grip in its TOP-LEFT corner (it is anchored bottom-
# right, so it grows up/left).  Size is clamped to the viewport and persisted
# in sessionStorage.  No config needed — it is always available except when
# the panel is maximized or on very small screens.

# ── Conversation persistence ──────────────────────────────────────────────
# Type:    bool
# Default: True
# True  → the conversation is stored in sessionStorage so it survives
#         navigation between pages of the same docs site, and is cleared on
#         tab close or when the user presses "Start a new chat".
# False → in-memory only (any navigation loses the conversation).
# User note: sessionStorage is per-tab and never leaves the browser.
ai_assistant_panel_persist = True

# ── R7: keyboard shortcut ─────────────────────────────────────────────────
# Type:    str
# Default: "Alt+Shift+A"
# Chord that toggles the panel open/closed.  MUST include at least one
# modifier (Ctrl / Alt / Cmd|Meta); a bare key is rejected so site-wide
# typing is never hijacked.  Set to "" to disable the global listener.
# Examples: "Ctrl+Shift+Space", "Cmd+J", "Alt+K".
# Developer note: parsed by _parseShortcut() in ai-assistant.js; an invalid
# or modifier-less value is treated exactly like "" (feature off, no error).
ai_assistant_panel_shortcut = "Alt+Shift+A"

# ── API mode proxy (C-2) ──────────────────────────────────────────────────
# Type:    str
# Default: "" (empty)
# A browser CANNOT call https://api.anthropic.com directly: the endpoint
# sends no CORS headers for web origins and requires an API key that would
# leak if embedded in static JS.  So API mode MUST point at YOUR OWN proxy
# (a serverless function / gateway) that injects the key server-side and
# forwards the Anthropic /v1/messages-shaped body.  While empty, enabling
# ai_assistant_panel_api_enabled raises a clear, actionable error in the
# panel instead of silently failing.
#   Example endpoint: ai_assistant_panel_api_url = "https://api.example.com/ai"
ai_assistant_panel_api_url = "https://scikit-plots-ai.hf.space"  # "/_proxy/hf"  # "/_proxy/anthropic"

# Type:    str
# Default: "" → JS default model ("claude-sonnet-4-20250514")
# Optional model name forwarded in the proxy request body.
ai_assistant_panel_api_model = "scikit-plots/Qwen2.5-Coder-7B-Instruct"  # "scikit-plots/gpt-oss-20b"  # "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Runtime environment helpers
# ---------------------------------------------------------------------------
# import os MUST appear at module scope — never use __import__("os") inline.
# __import__() is opaque to linters, bypasses static analysis, and is
# slightly slower (re-imports the module object on every evaluation rather
# than hitting sys.modules from a module-level binding).
import os  # noqa: E402 — conf.py files commonly place imports after preamble

# Single proxy base URL resolved once at build time from the environment.
# ⚠️  PRODUCTION WARNING
# ─────────────────────────────────────────────────────────────────────────────
# The default fallback "http://localhost:8787" is for LOCAL DEVELOPMENT ONLY.
# If AI_PROXY_BASE is not set in your CI/CD environment, every model endpoint
# will silently point at localhost and all panel API calls will fail for
# readers of your published documentation.
#
# Set the environment variable (or CI/CD secret) to your deployed proxy:
#   Local dev  : export AI_PROXY_BASE=http://localhost:8787
#   Staging/CI : export AI_PROXY_BASE=https://<org>-ai-proxy.hf.space  # https://scikit-plots-ai.hf.space
#   Production : export AI_PROXY_BASE=https://hf-proxy.<subdomain>.workers.dev
#
# SECURITY: API tokens (HF_TOKEN, ANTHROPIC_API_KEY, …) MUST NEVER appear
# here. They live only in the proxy's server-side environment / secret store.
# ─────────────────────────────────────────────────────────────────────────────
# _AI_PROXY_BASE: str = os.environ.get("AI_PROXY_BASE", "https://scikit-plots-ai.hf.space")
_AI_PROXY_BASE: str = os.environ.get("AI_PROXY_BASE", "http://localhost:8787")

# ════════════════════════════════════════════════════════════════════════════
#  Phase B — Multi-model panel, Terms of Service, Share sheet, Hamburger menu
# ════════════════════════════════════════════════════════════════════════════
#
# ── ai_assistant_panel_api_models  (multi-model registry) ────────────────────
#
# Type:    str | list[str] | list[dict]
# Default: []  (empty → legacy single-model path via panel_api_url / _model)
#
# When non-empty the floating panel renders:
#   1. A dedicated "Choose a model" sheet — same slide-over UX as Privacy.
#      Opened by the [⚙ model ▾] button in the sub-bar, BEFORE Privacy.
#   2. An inline picker beside the mic + send buttons (Claude-bar style),
#      controllable via ``ai_assistant_panel_inline_model_picker``.
#   3. A model attribution field in every feedback event payload
#      (``detail.model = {id, provider, model}``).
#
# Accepted shapes (normalised server-side by ``_normalize_panel_models``):
#
#   A. ``"gpt-4o"``                       — single string; uses panel_api_url
#                                            as the shared proxy endpoint.
#
#   B. ``["gpt-4o", "claude-sonnet-4-6"]``— list of strings; same shared-proxy
#                                            shorthand for several models.
#
#   C. list[dict]                         — full per-entry configuration; the
#                                            production form.  Each dict MUST
#                                            contain:
#                                              id       (unique, sessionStorage
#                                                        key)
#                                              provider (closed-set whitelist;
#                                                        see ``_PANEL_MODEL_PROVIDERS``)
#                                              model    (wire model name the
#                                                        proxy will forward)
#                                              endpoint (the proxy URL; http/
#                                                        https or site-relative)
#                                            Optional: label, description,
#                                              info_url, default, icon.
#
# ┌─────────────────────────────────────────────────────────────────────────┐
# │ SECURITY — API KEYS NEVER APPEAR IN conf.py                             │
# │                                                                         │
# │ The browser never sees an API key.  Keys live ONLY on the proxy         │
# │ server-side.  ``endpoint`` here is a URL pointing at YOUR proxy; the    │
# │ proxy reads the upstream key from an environment variable (which on     │
# │ GitHub Actions / Vercel / Netlify comes from a secret store) and        │
# │ injects it into the upstream request.                                   │
# │                                                                         │
# │ Example GitHub Actions snippet for a Vercel serverless proxy:           │
# │                                                                         │
# │   # .github/workflows/deploy.yml                                        │
# │   - name: Deploy proxy                                                  │
# │     env:                                                                │
# │       ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}               │
# │       OPENAI_API_KEY:    ${{ secrets.OPENAI_API_KEY }}                  │
# │       GEMINI_API_KEY:    ${{ secrets.GEMINI_API_KEY }}                  │
# │       MISTRAL_API_KEY:   ${{ secrets.MISTRAL_API_KEY }}                 │
# │       HF_TOKEN:          ${{ secrets.HF_TOKEN }}                        │
# │     run: vercel deploy --prod                                           │
# │                                                                         │
# │ HuggingFace Inference API proxy (FastAPI/Vercel example):               │
# │                                                                         │
# │   # api/hf.py  (deployed at /_proxy/hf)                                 │
# │   import os, httpx                                                      │
# │   HF_TOKEN = os.environ["HF_TOKEN"]   # from secrets, never conf.py     │
# │   HF_BASE  = "https://api-inference.huggingface.co/models"              │
# │                                                                         │
# │   def handler(req, model):                                              │
# │       url = f"{HF_BASE}/{model}/v1/chat/completions"                    │
# │       r = httpx.post(url, timeout=120,                                  │
# │           headers={"Authorization": f"Bearer {HF_TOKEN}",               │
# │                    "Content-Type": "application/json"},                 │
# │           content=req.body)                                             │
# │       # Forward SSE for streaming (content-type: text/event-stream)     │
# │       return (r.status_code, r.content,                                 │
# │               {"content-type": r.headers.get("content-type",            │
# │                                              "application/json")})      │
# │                                                                         │
# │ In your proxy code (e.g. api/anthropic.py on Vercel):                   │
# │                                                                         │
# │   import os, httpx                                                      │
# │   def handler(req):                                                     │
# │       key = os.environ["ANTHROPIC_API_KEY"]   # never logged            │
# │       r = httpx.post(                                                   │
# │           "https://api.anthropic.com/v1/messages",                      │
# │           headers={"x-api-key": key,                                    │
# │                    "anthropic-version": "2023-06-01",                   │
# │                    "content-type": "application/json"},                 │
# │           content=req.body, timeout=60,                                 │
# │       )                                                                 │
# │       return (r.status_code, r.content,                                 │
# │               {"content-type": "application/json"})                     │
# │                                                                         │
# │ NEVER write a key in conf.py.  NEVER expose a key in any value that     │
# │ reaches the browser (this extension validates ``endpoint`` against an   │
# │ http/https/site-relative allow-list; the JS only POSTs to ``endpoint``).│
# └─────────────────────────────────────────────────────────────────────────┘
#
# Full production example (uncomment + adapt):
#
# ai_assistant_panel_api_models = [
#     # ── HuggingFace GPT-OSS-20B  ─────────────────────────────────────────
#     # The HuggingFace Inference API uses the OpenAI-compat
#     # /v1/chat/completions format.  The JS auto-enables SSE streaming for
#     # the huggingface provider.
#     #
#     # Proxy example (Python / FastAPI):
#     #
#     #   import os, httpx
#     #   HF_TOKEN = os.environ["HF_TOKEN"]   # from secrets store
#     #   INFERENCE_BASE = "https://api-inference.huggingface.co/models"
#     #
#     #   async def hf_proxy(model: str, body: bytes):
#     #       url = f"{INFERENCE_BASE}/{model}/v1/chat/completions"
#     #       r = await httpx.AsyncClient().post(
#     #           url,
#     #           headers={"Authorization": f"Bearer {HF_TOKEN}",
#     #                    "Content-Type": "application/json"},
#     #           content=body, timeout=120,
#     #       )
#     #       return Response(r.content, media_type=r.headers["content-type"])
#     #
#     {
#         "id":          "gpt-oss-20b-skplt",
#         "label":       "GPT-OSS 20B (scikit-plots)",
#         "provider":    "huggingface",
#         "model":       "scikit-plots/gpt-oss-20b",
#         "endpoint":    "/_proxy/hf",        # same proxy, model field selects checkpoint
#         "info_url":    "https://huggingface.co/scikit-plots/gpt-oss-20b",
#         "description": (
#             "scikit-plots fine-tune of GPT-OSS-20B — trained on the full "
#             "scikit-plots documentation corpus for higher answer accuracy."
#         ),
#     },
#     {
#         "id":          "hf-gpt-oss-20b",
#         "label":       "GPT-OSS 20B (HuggingFace)",
#         "provider":    "huggingface",
#         "model":       "openai/gpt-oss-20b",
#         "endpoint":    "/_proxy/hf",        # YOUR HuggingFace proxy
#         "info_url":    "https://huggingface.co/openai/gpt-oss-20b",
#         "description": (
#             "OpenAI open-source 20B model via HuggingFace Inference API "
#             "(OpenAI-compat, SSE streaming enabled)."
#         ),
#     },
#     {
#         "id":          "claude-sonnet-4-6",
#         "label":       "Claude Sonnet 4.6",
#         "provider":    "anthropic",
#         "model":       "claude-sonnet-4-20250514",
#         "endpoint":    "/_proxy/anthropic",       # YOUR proxy
#         "default":     True,
#         "info_url":    "https://www.anthropic.com/claude",
#         "description": "Anthropic flagship — strong reasoning, long context.",
#     },
#     {
#         "id":          "gpt-4o",
#         "label":       "GPT-4o",
#         "provider":    "openai",
#         "model":       "gpt-4o",
#         "endpoint":    "/_proxy/openai",
#         "info_url":    "https://openai.com/index/hello-gpt-4o/",
#     },
#     {
#         "id":          "gemini-2.5-flash",
#         "label":       "Gemini 2.5 Flash",
#         "provider":    "google",
#         "model":       "gemini-2.5-flash",
#         "endpoint":    "/_proxy/gemini",
#         "info_url":    "https://ai.google.dev/gemini-api/docs/models",
#     },
#     {
#         "id":          "mistral-large",
#         "label":       "Mistral Large",
#         "provider":    "mistral",
#         "model":       "mistral-large-latest",
#         "endpoint":    "/_proxy/mistral",
#         "info_url":    "https://mistral.ai/news/mistral-large/",
#     },
#     {
#         "id":          "deepseek-reasoner",
#         "label":       "DeepSeek R1",
#         "provider":    "deepseek",
#         "model":       "deepseek-reasoner",
#         "endpoint":    "/_proxy/deepseek",
#         "info_url":    "https://api-docs.deepseek.com/",
#     },
#     {
#         "id":          "llama3.2-local",
#         "label":       "Llama 3.2 (local Ollama)",
#         "provider":    "ollama",
#         "model":       "llama3.2:latest",
#         "endpoint":    "http://localhost:11434/v1/chat/completions",
#         "info_url":    "https://ollama.com/library/llama3.2",
#         "description": "Fully offline, runs against your local Ollama server.",
#     },
# ]
ai_assistant_panel_api_models = [
    # ── Paid-tier entries (commented out — require deployed proxies) ──────
    # Uncomment once you have a running proxy with the relevant API key.
    # {
    #     "id":          "claude-sonnet-4-6",
    #     "label":       "Claude Sonnet 4.6",
    #     "provider":    "anthropic",
    #     "model":       "claude-sonnet-4-20250514",
    #     "endpoint":    "/_proxy/anthropic",   # proxy injects ANTHROPIC_API_KEY
    #     "info_url":    "https://www.anthropic.com/claude",
    #     "description": "Anthropic sonnet flagship — strong reasoning, long context.",
    # },
    # {
    #     "id":          "gpt-4o",
    #     "label":       "GPT-4o",
    #     "provider":    "openai",
    #     "model":       "gpt-4o",
    #     "endpoint":    "/_proxy/openai",      # proxy injects OPENAI_API_KEY
    #     "info_url":    "https://openai.com/index/hello-gpt-4o/",
    #     "description": "Openai chatgpt.",
    # },
    # {
    #     "id":          "gemini-2.5-flash",
    #     "label":       "Gemini 2.5 Flash",
    #     "provider":    "google",
    #     "model":       "gemini-2.5-flash",
    #     "endpoint":    "/_proxy/gemini",      # proxy injects GEMINI_API_KEY
    #     "info_url":    "https://ai.google.dev/gemini-api/docs/models",
    #     "description": "Google gemini.",
    # },
    #
    # ── HuggingFace GPT-OSS-20B (upstream OpenAI release) ────────────────
    # Same proxy, model field selects the upstream checkpoint.
    # Proxy MUST inject a valid HuggingFace API token server-side:
    #     Authorization: Bearer ${HF_TOKEN}
    # NEVER embed the token in this conf.py file.
    # Model card: https://huggingface.co/openai/gpt-oss-20b
    # https://router.huggingface.co/v1/chat/completions
    {
        "id":          "gpt-oss-20b-hf",
        "label":       "GPT-OSS 20B (OpenAI/HuggingFace)",
        "provider":    "huggingface",
        "model":       "openai/gpt-oss-20b",
        "endpoint":    "https://router.huggingface.co/v1/chat/completions",
        "info_url":    "https://huggingface.co/openai/gpt-oss-20b",
        "description": (
            "OpenAI open-source 20B via HuggingFace Inference API — "
            "OpenAI-compat /v1/chat/completions, SSE streaming enabled."
        ),
    },
    {
        "default":     True,
        "id":          "Qwen2.5-Coder-7B-Instruct-hf",
        "label":       "Qwen2.5-Coder-7B-Instruct (Qwen/HuggingFace)",
        "provider":    "huggingface",
        "model":       "Qwen/Qwen2.5-Coder-7B-Instruct",
        "endpoint":    "https://router.huggingface.co/v1/chat/completions",
        "info_url":    "https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct",
        "description": (
            "Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). "
            "For more:https://github.com/QwenLM"
        ),
    },
    {
        "id":          "Qwen2.5-Coder-32B-Instruct-hf",
        "label":       "Qwen2.5-Coder-32B-Instruct (Qwen/HuggingFace)",
        "provider":    "huggingface",
        "model":       "Qwen/Qwen2.5-Coder-32B-Instruct",
        "endpoint":    "https://router.huggingface.co/v1/chat/completions",
        "info_url":    "https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct",
        "description": (
            "Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as CodeQwen). "
            "For more:https://github.com/QwenLM"
        ),
    },
    # ── scikit-plots fine-tuned GPT-OSS-20B ──────────────────────────────
    # Fine-tuned for scikit-plots documentation Q&A.
    # Model card: https://huggingface.co/scikit-plots/gpt-oss-20b
    #
    # IMPORTANT — endpoint resolution (environment-aware):
    #   AI_PROXY_BASE is the single knob that selects which free proxy to use.
    #   Set it as an environment variable or CI/CD secret:
    #
    #   Local development (dev_proxy.py on port 8787):
    #       export AI_PROXY_BASE=http://localhost:8787
    #
    #   Staging / CI (HuggingFace Space — Option A, always free):
    #       export AI_PROXY_BASE=https://scikit-plots-ai.hf.space
    #
    #   Production (Cloudflare Worker — Option B, 100 000 req/day free):
    #       export AI_PROXY_BASE=https://hf-proxy.<your-subdomain>.workers.dev
    #
    # The _PROXY_BASE import at the top of this block reads the env var with
    # a sensible fallback so local builds work without any shell setup, and
    # CI/CD secrets transparently select the production proxy.
    #
    # SECURITY: API tokens (HF_TOKEN, ANTHROPIC_API_KEY, …) MUST NEVER appear
    # here.  They live only in the proxy's environment secret store.
    {
        "id":          "gpt-oss-20b-skplt",
        "label":       "GPT-OSS 20B (scikit-plots/HuggingFace)",
        "provider":    "huggingface",
        "model":       "scikit-plots/gpt-oss-20b",
        # Resolved at build time from AI_PROXY_BASE; see comment above.
        # Replace with a literal URL once your proxy is deployed, e.g.:
        #   "endpoint": "https://scikit-plots-ai.hf.space/v1/chat/completions",
        "endpoint":    _AI_PROXY_BASE + "/v1/chat/completions",
        "info_url":    "https://huggingface.co/scikit-plots/gpt-oss-20b",
        "description": (
            "(future) scikit-plots fine-tune of GPT-OSS-20B — trained on the full "
            "scikit-plots documentation corpus for higher answer accuracy."
        ),
    },
    {
        "id":          "Qwen2.5-Coder-7B-Instruct-skplt",
        "label":       "Qwen2.5-Coder-7B-Instruct (scikit-plots/HuggingFace)",
        "provider":    "huggingface",
        "model":       "scikit-plots/Qwen2.5-Coder-7B-Instruct",
        # Resolved at build time from AI_PROXY_BASE; see comment above.
        # Replace with a literal URL once your proxy is deployed, e.g.:
        #   "endpoint": "https://scikit-plots-ai.hf.space/v1/chat/completions",
        "endpoint":    _AI_PROXY_BASE + "/v1/chat/completions",
        "info_url":    "https://huggingface.co/scikit-plots/Qwen2.5-Coder-7B-Instruct",
        "description": (
            "(future) scikit-plots fine-tune of Qwen2.5-Coder-7B-Instruct — trained on the full "
            "scikit-plots documentation corpus for higher answer accuracy."
        ),
    },
    {
        "id":          "Qwen2.5-Coder-32B-Instruct-skplt",
        "label":       "Qwen2.5-Coder-32B-Instruct (scikit-plots/HuggingFace)",
        "provider":    "huggingface",
        "model":       "scikit-plots/Qwen2.5-Coder-32B-Instruct",
        # Resolved at build time from AI_PROXY_BASE; see comment above.
        # Replace with a literal URL once your proxy is deployed, e.g.:
        #   "endpoint": "https://scikit-plots-ai.hf.space/v1/chat/completions",
        "endpoint":    _AI_PROXY_BASE + "/v1/chat/completions",
        "info_url":    "https://huggingface.co/scikit-plots/Qwen2.5-Coder-32B-Instruct",
        "description": (
            "(future) scikit-plots fine-tune of Qwen2.5-Coder-32B-Instruct — trained on the full "
            "scikit-plots documentation corpus for higher answer accuracy."
        ),
    },
]


# ── ai_assistant_panel_inline_model_picker ───────────────────────────────────
# Type:    bool
# Default: True
# Render the inline model <select> beside mic + send (Claude-bar layout).
# The dedicated sheet button in the sub-bar remains available regardless.
ai_assistant_panel_inline_model_picker = True

# ── ai_assistant_panel_api_streaming ─────────────────────────────────────────
# Type:    bool
# Default: True
#
# Master switch for SSE (Server-Sent Events) streaming in API mode.
#
#   True  (default)
#       The JS requests ``stream: true`` for every OpenAI-compat provider
#       (HuggingFace, Groq, Cloudflare, Cerebras, Together, Fireworks,
#       SambaNova, Ollama, OpenAI, Mistral, DeepSeek, …) and renders tokens
#       as they arrive, producing a live typewriter effect.
#
#       Your proxy MUST forward the ``content-type: text/event-stream``
#       header and the raw SSE frames without buffering.  All four free
#       proxy options below are written to do this correctly.
#
#   False
#       The JS always requests ``stream: false`` and waits for the complete
#       JSON response.  Use False when your hosting platform buffers SSE
#       frames before delivering the response (some PaaS providers, certain
#       reverse-proxy configurations).  The extension detects this
#       automatically at runtime too: if ``stream: true`` is requested but
#       the response arrives as ``application/json`` rather than
#       ``text/event-stream``, the non-streaming parser handles it silently.
#
# Developer note: This flag is serialised into
# ``window.AI_ASSISTANT_CONFIG.panelApiStreaming`` by
# ``add_ai_assistant_context`` and read by ``_panelApiCall`` in
# ``ai-assistant.js``.  Setting it False overrides the per-provider default
# without removing the provider from ``_STREAMING_PROVIDERS``; it is a
# deployment-time tuning knob, not a provider capability flag.
ai_assistant_panel_api_streaming = True


# ===========================================================================
# FREE PROXY DEPLOYMENT — FOUR ZERO-COST OPTIONS
# ===========================================================================
#
# The browser cannot call any AI provider API (HuggingFace, Anthropic, …)
# directly: providers send no CORS headers for arbitrary web origins, and
# every provider requires a secret API token that must never appear in
# static JS served to readers.
#
# A thin server-side proxy is therefore REQUIRED.  It injects the token
# (from an environment secret, never from this file) and forwards the
# request.
#
# All four options below are truly free — no credit card, no paid tier.
#
# ── Option A: HuggingFace Space (CPU tier, always on) ────────────────────────
#
# Deploy a ~60-line FastAPI app as a Docker Space under your HF org.
# CPU Spaces are completely free and always reachable.  Your HF_TOKEN
# lives in the Space's secret store — never in conf.py or in the browser.
#
# Files needed in your Space repo:
#
#   app.py  ────────────────────────────────────────────────────────────────
#   import os, httpx
#   from fastapi import FastAPI, Request, Response
#   from fastapi.middleware.cors import CORSMiddleware
#
#   app = FastAPI()
#   app.add_middleware(CORSMiddleware, allow_origins=["*"],
#                      allow_methods=["POST","OPTIONS"],
#                      allow_headers=["Content-Type"])
#   HF_TOKEN   = os.environ["HF_TOKEN"]   # Space secret — never hardcode
#   HF_BASE    = "https://api-inference.huggingface.co/models"
#
#   @app.post("/v1/chat/completions")
#   async def proxy(req: Request) -> Response:
#       body = await req.body()
#       import json
#       model = json.loads(body).get("model", "openai/gpt-oss-20b")
#       r = await httpx.AsyncClient(timeout=120).post(
#           f"{HF_BASE}/{model}/v1/chat/completions",
#           content=body,
#           headers={"Authorization": f"Bearer {HF_TOKEN}",
#                    "Content-Type": "application/json"})
#       return Response(r.content, status_code=r.status_code,
#                       media_type=r.headers.get("content-type","application/json"))
#
#   @app.get("/health")
#   def health(): return {"status": "ok"}
#   ────────────────────────────────────────────────────────────────────────
#
#   Dockerfile  ────────────────────────────────────────────────────────────
#   FROM python:3.11-slim
#   WORKDIR /app
#   RUN pip install fastapi uvicorn httpx
#   COPY app.py .
#   EXPOSE 7860
#   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
#   ────────────────────────────────────────────────────────────────────────
#
# Secrets: Space → Settings → Repository secrets → HF_TOKEN = hf_xxxx...
#
# Once live at https://scikit-plots-ai.hf.space, set:
#
# ai_assistant_panel_api_models = [
#     {
#         "default":     True,
#         "id":          "gpt-oss-20b-skplt",
#         "label":       "GPT-OSS 20B (scikit-plots)",
#         "provider":    "huggingface",
#         "model":       "scikit-plots/gpt-oss-20b",
#         "endpoint":    "https://scikit-plots-ai.hf.space/v1/chat/completions",
#         "info_url":    "https://huggingface.co/scikit-plots/gpt-oss-20b",
#         "description": "scikit-plots fine-tune — trained on full docs corpus.",
#     },
#     {
#         "id":          "hf-gpt-oss-20b",
#         "label":       "GPT-OSS 20B (OpenAI)",
#         "provider":    "huggingface",
#         "model":       "openai/gpt-oss-20b",
#         "endpoint":    "https://scikit-plots-ai.hf.space/v1/chat/completions",
#         "info_url":    "https://huggingface.co/openai/gpt-oss-20b",
#         "description": "OpenAI open-source 20B via HuggingFace Inference API.",
#     },
# ]
#
#
# ── Option B: Cloudflare Worker (100 000 req/day free, zero server) ──────────
#
# worker.js  (deploy with: wrangler deploy)  ─────────────────────────────
# export default {
#   async fetch(request, env) {
#     if (request.method === "OPTIONS")
#       return new Response(null, { headers: cors() });
#     if (request.method !== "POST")
#       return new Response("Method Not Allowed", { status: 405 });
#     const body = await request.text();
#     let model = "openai/gpt-oss-20b";
#     try { const p = JSON.parse(body); if (p.model) model = p.model; } catch(_){}
#     const url =
#       `https://api-inference.huggingface.co/models/${model}/v1/chat/completions`;
#     const r = await fetch(url, {
#       method: "POST",
#       headers: { "Authorization": `Bearer ${env.HF_TOKEN}`,
#                  "Content-Type": "application/json" },
#       body });
#     return new Response(r.body, { status: r.status,
#       headers: { "Content-Type": r.headers.get("content-type") ||
#                  "application/json", ...cors() }});
#   }
# };
# function cors() {
#   return { "Access-Control-Allow-Origin": "*",
#            "Access-Control-Allow-Methods": "POST, OPTIONS",
#            "Access-Control-Allow-Headers": "Content-Type" }; }
# ────────────────────────────────────────────────────────────────────────
#
# Deploy:
#   npm install -g wrangler
#   wrangler login
#   wrangler init hf-proxy        # copy worker.js above
#   wrangler secret put HF_TOKEN  # paste hf_xxxx...
#   wrangler deploy
#   # → https://hf-proxy.<your-subdomain>.workers.dev
#
# ai_assistant_panel_api_models = [
#     {
#         "default":  True,
#         "id":       "hf-gpt-oss-20b",
#         "label":    "GPT-OSS 20B (HuggingFace)",
#         "provider": "huggingface",
#         "model":    "openai/gpt-oss-20b",
#         "endpoint": "https://hf-proxy.<your-subdomain>.workers.dev",
#     },
# ]
#
#
# ── Option C: Local dev_proxy.py (development only — never deploy) ────────────
#
# Run alongside your local http.server / Live Server during development.
#
#   dev_proxy.py  ──────────────────────────────────────────────────────────
#   import os, json, httpx
#   from http.server import BaseHTTPRequestHandler, HTTPServer
#   HF_TOKEN = os.environ["HF_TOKEN"]    # export HF_TOKEN=hf_xxxx in shell
#   HF_BASE  = "https://api-inference.huggingface.co/models"
#   class H(BaseHTTPRequestHandler):
#     def do_OPTIONS(self): self._cors()
#     def do_POST(self):
#       b = self.rfile.read(int(self.headers.get("Content-Length",0)))
#       model = json.loads(b).get("model","openai/gpt-oss-20b")
#       r = httpx.post(f"{HF_BASE}/{model}/v1/chat/completions", content=b,
#             headers={"Authorization":f"Bearer {HF_TOKEN}",
#                      "Content-Type":"application/json"}, timeout=120)
#       self.send_response(r.status_code)
#       self.send_header("Content-Type",
#                        r.headers.get("content-type","application/json"))
#       self._cors_headers(); self.end_headers(); self.wfile.write(r.content)
#     def _cors(self): self.send_response(204); self._cors_headers(); self.end_headers()
#     def _cors_headers(self):
#       self.send_header("Access-Control-Allow-Origin","*")
#       self.send_header("Access-Control-Allow-Methods","POST, OPTIONS")
#       self.send_header("Access-Control-Allow-Headers","Content-Type")
#   HTTPServer(("",8787), H).serve_forever()
#   ────────────────────────────────────────────────────────────────────────
#
# Usage:
#   export HF_TOKEN=hf_xxxx...
#   python dev_proxy.py &
#
# Then for local builds only:
#   ai_assistant_panel_api_models = [
#       { "id": "hf-local", "provider": "huggingface",
#         "model": "openai/gpt-oss-20b",
#         "endpoint": "http://localhost:8787/v1/chat/completions" },
#   ]
#
# Use an environment variable to switch endpoints per environment:
#
# import os
# _PROXY = os.environ.get("AI_PROXY_BASE",
#                         "https://scikit-plots-ai.hf.space")
# ai_assistant_panel_api_models = [
#     { "id": "hf-gpt-oss-20b", "provider": "huggingface",
#       "model": "openai/gpt-oss-20b",
#       "endpoint": f"{_PROXY}/v1/chat/completions", "default": True },
# ]
# Set AI_PROXY_BASE as a GitHub Actions / CI repo secret pointing at
# the deployed HF Space or Cloudflare Worker URL for production builds.
#
#
# ── Option D: HuggingFace ZeroGPU Space (free shared GPU, self-host model) ────
#
# Host the model itself — zero inference cost forever.
# ZeroGPU gives free shared-GPU time to public Spaces.
#
# Minimal Gradio Space exposing an OpenAI-compat REST endpoint:
#
#   app.py  ────────────────────────────────────────────────────────────────
#   import gradio as gr, spaces
#   from transformers import pipeline
#   MODEL_ID = "scikit-plots/gpt-oss-20b"    # or "openai/gpt-oss-20b"
#   pipe = pipeline("text-generation", model=MODEL_ID, device_map="auto")
#
#   @spaces.GPU
#   def respond(message, system=""):
#       msgs = ([{"role":"system","content":system}] if system else []) + \
#              [{"role":"user","content":message}]
#       return pipe(msgs, max_new_tokens=512)[0]["generated_text"][-1]["content"]
#
#   gr.Interface(fn=respond, inputs=["text","text"], outputs="text").launch()
#   ────────────────────────────────────────────────────────────────────────
#
# Point the extension at the Gradio Space URL and use provider="huggingface".
# The @spaces.GPU decorator gates GPU allocation per-request at zero cost.

# ── Terms of Service sheet (sibling of Privacy & Responsibility) ─────────────
# Type:    bool / str / str (HTML, trusted author input)
ai_assistant_panel_terms             = True
ai_assistant_panel_terms_title       = "Terms of Service"
ai_assistant_panel_terms_link_text   = "Terms of Service"
# Empty string → built-in default copy that covers documentation context,
# acceptable use, model-provider responsibility, and the feedback collection
# notice (kept in sync with how the feedback event payload is built).
# Author override is INJECTED VERBATIM — keep this trusted (from conf.py only,
# never user input).
ai_assistant_panel_terms_html        = ""

# ── Share sheet (copy-link + intent shares) ──────────────────────────────────
ai_assistant_panel_share         = True
ai_assistant_panel_share_label   = "Share"
# Empty list → built-in defaults: copy_link, email, x, linkedin, reddit,
# hacker_news.  Each entry is filtered through the same URL allow-list as
# AI providers (only http/https/mailto: schemes survive).  Each url_template
# supports {url} and {title} placeholders (URL-encoded client-side).
# Custom example:
#   ai_assistant_panel_share_targets = [
#       {"id": "internal_slack",
#        "label": "Slack #docs",
#        "url_template": "https://example.slack.com/share?u={url}"},
#   ]
ai_assistant_panel_share_targets = []

# ── Project Links sheet (GitHub source + documentation site) ─────────────────
#
# The Links sheet is a slide-over panel, identical in UX to the Share / Model /
# Privacy sheets, that presents clickable cards for the project's source
# repository and its documentation website.
#
# Two sub-bar buttons are also added:
#   • GitHub icon  (left cluster, after the hamburger hint)  → opens the sheet
#   • Globe  icon  (right cluster, after the Share button)   → opens the sheet
#
# Both buttons and the sheet can be suppressed independently via the boolean
# flags below.  When the sheet itself is disabled (``panelLinks = False``) the
# buttons instead navigate directly to the configured URL (or are hidden when
# no URL is configured).
#
# Quick-start (scikit-plots) ── add these four lines to conf.py:
#
#   ai_assistant_panel_source_url = "https://github.com/scikit-plots/scikit-plots"
#   ai_assistant_panel_site_url   = "https://scikit-plots.github.io/"
#   ai_assistant_panel_source_label = "GitHub Source"
#   ai_assistant_panel_site_label   = "Documentation"
#

# ─ master sheet switch ───────────────────────────────────────────────────────
# Type:    bool
# Default: True
# Disabling hides the sheet entirely and also removes the Source / Site sub-bar
# buttons.  The per-button flags (panel_source / panel_site) are subordinate to
# this: they only take effect when this master switch is True.
ai_assistant_panel_links = True

# ─ sheet header ──────────────────────────────────────────────────────────────
# Type:    str
# Default: "Project Links"
# Title shown at the top of the Links slide-over panel.
ai_assistant_panel_links_title = "Project Links"

# ─ custom HTML footer ────────────────────────────────────────────────────────
# Type:    str  (raw HTML — trusted author input from conf.py only)
# Default: ""  (nothing injected)
# Optional block appended below the card list inside the Links sheet.
# Useful for a paragraph of extra context, a changelog link, a citation …
# Author override is INJECTED VERBATIM — keep this trusted (conf.py only,
# never user input).
#
# Example:
#   ai_assistant_panel_links_html = """
#     <p style="font-size:0.75rem;color:var(--pst-color-text-muted)">
#       scikit-plots is maintained by volunteers. Star us on GitHub! ⭐
#     </p>
#   """
ai_assistant_panel_links_html = ""

# ─ source (GitHub) card & subbar button ──────────────────────────────────────
# Type:    bool
# Default: True
# Master switch for the "Source" entry.  When False the card inside the sheet
# AND the GitHub icon button in the left sub-bar cluster are both hidden.
ai_assistant_panel_source = True

# Type:    str  (URL — http / https only; other schemes are rejected client-side)
# Default: ""  (card/button hidden when empty)
# URL of the project's source repository.  Typically a GitHub / GitLab URL.
#
# scikit-plots default:
ai_assistant_panel_source_url = "https://github.com/scikit-plots/scikit-plots"

# Type:    str
# Default: "Source"  (shown as the card title and the sub-bar button label)
# Human-readable label for the source card and the sub-bar icon button.
#
# scikit-plots default:
ai_assistant_panel_source_label = "GitHub Source"

# Type:    str
# Default: ""  (no description line rendered when empty)
# One-line description shown beneath the card title in the sheet.
# Keep it brief — it is truncated with ellipsis at narrow widths.
#
# scikit-plots default:
ai_assistant_panel_source_desc = "Browse and contribute to the scikit-plots codebase"

# Type:    str
# Default: "Source"
# Label used for the subbar button next to the GitHub icon.
# Use a very short word (≤ 8 chars) — labels are hidden automatically at
# narrow panel widths via the [data-narrow] CSS rule.
ai_assistant_panel_source_btn_label = "Source"

# ─ site (documentation website) card & subbar button ─────────────────────────
# Type:    bool
# Default: True
# Master switch for the "Site" entry.  When False the card inside the sheet
# AND the Globe icon button in the right sub-bar cluster are both hidden.
ai_assistant_panel_site = True

# Type:    str  (URL — http / https only)
# Default: ""  (card/button hidden when empty)
# URL of the project's documentation website or home page.
#
# scikit-plots default:
ai_assistant_panel_site_url = "https://scikit-plots.github.io/"

# Type:    str
# Default: "Website"  (shown as the card title and the sub-bar button label)
# Human-readable label for the site card and the Globe sub-bar button.
#
# scikit-plots default:
ai_assistant_panel_site_label = "Documentation Site"

# Type:    str
# Default: ""  (no description line rendered when empty)
# One-line description shown beneath the card title in the sheet.
#
# scikit-plots default:
ai_assistant_panel_site_desc = "Explore the full API reference and user guide"

# Type:    str
# Default: "Website"
# Label used for the subbar button next to the Globe icon.
# Recommend ≤ 8 chars (longer labels are hidden at narrow widths).
ai_assistant_panel_site_btn_label = "Website"

# ── Hamburger overflow menu ──────────────────────────────────────────────────
# Type:    bool
# Default: True
# When True a ☰ menu button appears at the LEFT of the sub-bar and duplicates
# every sheet entry-point (model, privacy, terms, share, links) in a single
# popover.  Designed for narrow viewports: the right-cluster labels collapse
# via CSS at ≤ 460 px, and the hamburger is the canonical entry-point.
ai_assistant_panel_hamburger = True

# ── How to access feedback for model training ───────────────────────────────
#
# The extension itself NEVER stores or transmits a feedback submission.
# Instead, every submit dispatches an ``ai-assistant-feedback`` CustomEvent
# on ``document`` with the shape (schemaVersion: 1):
#
#   {
#     schemaVersion: 1,
#     ratingValue:   -1 | 0 | +1 | ...,   // SIGNED INT (training signal)
#     ratingLabel:   "negative" | ...,    // string
#     rating:        "negative" | ...,    // legacy alias of ratingLabel
#     message:       "free-text…",
#     query:         "the user's question",
#     answer:        "the model's full reply",
#     model:         {id, provider, model} | null,
#     answerIndex:   <int>,
#     page:          "https://docs.example.com/x.html",
#     ts:            <ms epoch>,
#     sessionId:     "<uuid>"             // idempotency key
#   }
#
# Doc authors attach their own listener (e.g. in a custom JS file added via
# html_js_files) and forward to the storage of their choice:
#
#   document.addEventListener("ai-assistant-feedback", function (ev) {
#       fetch("/_collect/feedback", {
#           method:  "POST",
#           headers: {"content-type": "application/json"},
#           body:    JSON.stringify(ev.detail),
#           keepalive: true,    // survives page-unload races
#       });
#   });
#
# The "/_collect/feedback" endpoint should:
#   • require an origin / referer check;
#   • dedup on (sessionId, answerIndex);
#   • write to your store of choice (S3 JSONL, BigQuery, Postgres, …);
#   • NEVER echo back submissions to the browser.
#
# When ai_assistant_panel_feedback_log = True the JS also console.log()s
# each submission — useful while developing the listener.

# ── R5: feedback block ("Was this helpful?") ──────────────────────────────
# Type:    bool
# Default: True
# Show a feedback prompt after each assistant reply.
ai_assistant_panel_feedback = True

# Type:    str  — the question text.
ai_assistant_panel_feedback_question = "Was this helpful?"

# Type:    list[dict]
# Default: [] → built-in 10-emoji gradient
#          (😡 Terrible → 😞 Poor → 😟 Unsatisfied → 🙁 No → 😑 Not really
#           → 🙂 Somewhat → 😊 Mostly yes → 😄 Good → 😁 Very good → 🤩 Excellent!)
# Range:   2–10 options.  Each dict:
#   {"emoji": "<char>", "title": "<hover/aria text>", "value": "<sent value>"}
# The JS widget auto-scales emoji size (via CSS data-count) so all buttons
# always stay on one line regardless of count.
# The chosen value + free text are dispatched as a DOM CustomEvent
# 'ai-assistant-feedback' (detail = {rating, message, page, ts}) so doc
# authors can wire their own analytics.  The extension stores/sends nothing.
#
# ── Example A: minimal 3-emoji (classic thumbs) ───────────────────────────
# ai_assistant_panel_feedback_options = [
#     {"emoji": "\U0001F641", "title": "No",           "value": "negative"},  # 🙁
#     {"emoji": "\U0001F610", "title": "Not sure",     "value": "neutral" },  # 😐
#     {"emoji": "\U0001F600", "title": "Yes, it was!", "value": "positive"},  # 😀
# ]
#
# ── Example B: 5-emoji (5-point Likert) ────────────────────────────────────
# ai_assistant_panel_feedback_options = [
#     {"emoji": "\U0001F621", "title": "Terrible",      "value": "terrible" },  # 😡
#     {"emoji": "\U0001F641", "title": "No",            "value": "negative" },  # 🙁
#     {"emoji": "\U0001F610", "title": "Not sure",      "value": "neutral"  },  # 😐
#     {"emoji": "\U0001F642", "title": "Mostly yes",    "value": "positive" },  # 🙂
#     {"emoji": "\U0001F600", "title": "Excellent!",    "value": "excellent"},  # 😀
# ]
#
# ── Example C: 10-emoji full gradient (DEFAULT — leave list empty) ─────────
# ai_assistant_panel_feedback_options = [
#     {"emoji": "\U0001F621", "title": "Terrible",      "value": "terrible"         },  # 😡 -5
#     {"emoji": "\U0001F61E", "title": "Poor",          "value": "poor"             },  # 😞 -4
#     {"emoji": "\U0001F61F", "title": "Unsatisfied",   "value": "unsatisfied"      },  # 😟 -3
#     {"emoji": "\U0001F641", "title": "No",            "value": "negative"         },  # 🙁 -2
#     {"emoji": "\U0001F611", "title": "Not really",    "value": "slightly_negative"},  # 😑 -1
#     {"emoji": "\U0001F642", "title": "Somewhat",      "value": "slightly_positive"},  # 🙂 +1
#     {"emoji": "\U0001F60A", "title": "Mostly yes",    "value": "mostly_positive"  },  # 😊 +2
#     {"emoji": "\U0001F604", "title": "Good",          "value": "good"             },  # 😄 +3
#     {"emoji": "\U0001F601", "title": "Very good",     "value": "very_good"        },  # 😁 +4
#     {"emoji": "\U0001F929", "title": "Excellent!",    "value": "excellent"        },  # 🤩 +5
# ]
#
# Leave empty to use the 11-emoji default shown in Example C with 0 option:
ai_assistant_panel_feedback_options = []

# Type:    str — placeholder for the optional free-text box.
ai_assistant_panel_feedback_placeholder = ""

# Type:    str — submit-button label.
ai_assistant_panel_feedback_submit = "Send feedback"

# Type:    str — message shown after submission.
ai_assistant_panel_feedback_thanks = "Thanks for your feedback!"

# Type:    bool
# Default: False
# When True the JS also console.log()s each feedback submission (dev aid).
ai_assistant_panel_feedback_log = False

# ── R2: privacy / responsibility sheet ────────────────────────────────────
# A slide-over inside the panel, opened from a small header link.  The
# built-in default copy is structured for a beginner→expert reader and
# explicitly separates THIS extension's responsibility (formats the page,
# stores nothing server-side, zero network calls in stub mode) from the
# integrated MODEL's responsibility (in API mode the answer, retention and
# logging belong to that provider; we cannot see or control their logs).
#
# Type:    str — heading + the small header link label.
ai_assistant_panel_privacy_title = "Privacy & Responsibility"
ai_assistant_panel_privacy_link_text = "Privacy & Responsibility"

# Type:    str
# Default: "" → built-in structured default
# Full custom body HTML for the sheet.  TRUSTED author content (it comes
# from this conf.py, never from end-user input) and is injected verbatim.
# Provide your own organisation's policy here when needed, e.g.:
#   ai_assistant_panel_privacy_html = """
#       <h4>Our policy</h4><p>…</p>
#       <h4>Model provider</h4><p>…</p>
#   """
ai_assistant_panel_privacy_html = ""

# ── R8: standalone AI search-bar (OPT-IN) ─────────────────────────────────
# Type:    bool
# Default: False
# Mounts an extra search input that forwards its text into the AI panel as
# the first question.  It renders the extension's OWN element and never
# touches the theme's search DOM, so PyData / Furo / RTD search keep working
# untouched.  Off by default — zero risk to existing search.
ai_assistant_search_bar = False

# Type:    str
# Default: "" (no-op)
# CSS selector of the host element to append the search-bar into.  If empty
# or not found, nothing happens (safe no-op).  Example for pydata theme:
#   ai_assistant_search_bar_selector = ".bd-sidebar-primary"
ai_assistant_search_bar_selector = ".bd-sidebar-primary"  # or ""

# Type:    str  ("top" | "bottom")
# Default: "bottom"
#
# Where inside the host element the search-bar is inserted.
#
#   "top"    → Prepended before the first child so the bar appears at the
#              very top of the sidebar — immediately visible without scrolling
#              past navigation links.  Recommended for sidebar placement.
#   "bottom" → Appended after the last child (default; pre-existing
#              behaviour).  Use when appending to a non-sidebar host element
#              where top position would displace existing first-child content.
#
# User note: "top" gives users the best discoverability — the AI search input
# is the first thing they see when they glance at the sidebar.
# Any value other than "top" is treated as "bottom" (safe fallback).
#
# Example (pydata theme, sidebar top):
#   ai_assistant_search_bar_selector = ".bd-sidebar-primary"
#   ai_assistant_search_bar_position = "top"
ai_assistant_search_bar_position = "top"

# Type:    bool
# Default: False
# Compact inline variant when True; full-width block when False.
ai_assistant_search_bar_mini = False  # accept full width

# Type:    str — placeholder for the standalone search-bar input.
ai_assistant_panel_search_placeholder = "Ask AI about these docs\u2026"

# Type:    dict[str, dict]
# Default: _DEFAULT_PROVIDERS  (12 providers; 3 enabled by default —
#          chatgpt, claude, gemini — 9 disabled)
#
# Full AI provider registry.  Keys are short identifiers (e.g. "claude",
# "ollama"); values are provider config dicts.
#
# Required keys per provider (all validated by ``_validate_provider``):
#
#   "enabled"          bool  — shown and clickable when True
#   "label"            str   — button text visible to users
#   "description"      str   — tooltip / aria-label
#   "icon"             str   — SVG filename in _static/ (or data-URI fallback
#                              from ``_PROVIDER_META`` when file is absent)
#   "url_template"     str   — URL opened in a new tab; {prompt} is replaced
#                              with the URL-encoded prompt string.
#                              Use "" for API-only or local providers.
#   "prompt_template"  str   — prompt body; {url} → absolute .md page URL;
#                              {content} → raw Markdown text of the page
#   "model"            str   — model identifier forwarded to the widget JS
#   "type"             str   — "web" | "local" | "api" | "custom"
#
# Optional keys:
#   "api_base_url"     str   — base endpoint for "local" / "api" type
#                              providers (e.g. "http://localhost:11434" for
#                              Ollama).
#   "fetch_mode"       str   — controls what the widget injects into the prompt:
#                              "url"     → pass the page URL (default; Claude /
#                                          ChatGPT read URLs autonomously).
#                              "content" → inject pre-extracted Markdown body
#                                          instead of the URL.  Required for
#                                          providers that do NOT reliably fetch
#                                          arbitrary external URLs.
#                              "both"    → include both URL "url" and Markdown content "content".
#                              "paste"   → instruct the user to paste content
#                                          manually.
#                              Omitting ``fetch_mode`` defaults to ``"url"``.
#
# Provider type semantics:
#   "web"    — Opens a browser tab with the prompt pre-filled via a URL
#              query parameter.  No API key needed from the user.
#   "local"  — Localhost-only AI server.  ``api_base_url`` MUST target a
#              loopback address; ``_validate_ollama_url`` rejects remote
#              hosts to prevent data exfiltration via the widget.
#   "api"    — Direct browser-to-API calls (requires CORS on the server).
#   "custom" — User-defined endpoint; fill ``api_base_url`` and ``model``
#              for any OpenAI-compatible local or internal server.
#
# Security guarantees:
#   * ``url_template`` is validated by ``_validate_provider_url_template``:
#     only http:// and https:// are accepted; javascript: and data: URLs
#     are always rejected regardless of other settings.
#   * For type "local", ``api_base_url`` is additionally validated by
#     ``_validate_ollama_url`` to enforce loopback-only origins.
#   * All providers are sanitised by ``_filter_providers`` before being
#     embedded into each page's ``window.AI_ASSISTANT_CONFIG`` script block
#     via ``_safe_json_for_script`` (which escapes ``</`` to prevent XSS).
#
# Tip — extend rather than replace the default registry to inherit all
# upstream defaults and only override the entries you need:
#
#   from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
#       _DEFAULT_PROVIDERS,
#   )
#   ai_assistant_providers = {
#       **_DEFAULT_PROVIDERS,
#       "claude": {**_DEFAULT_PROVIDERS["claude"], "enabled": True},
#       "ollama": {
#           **_DEFAULT_PROVIDERS["ollama"],
#           "enabled": True,
#           "model": "qwen3:latest",
#       },
#   }
ai_assistant_providers = {
    # --- Tier 1: enabled by default ----------------------------------------

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
    },

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
            "Hi! Please read this documentation page: {url}\n\n"  # + "{content}"
            "I have questions about it."
        ),
        # Model identifier.  Forwarded to the widget for future API-mode use.
        "model": "claude-sonnet-4-6",
        # "web" opens a browser tab; no API key is required from the user.
        "type": "web",
    },

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
    },

    # --- Tier 2: local / fully offline ------------------------------------

    # Ollama — fully offline, privacy-preserving local inference.
    # Prerequisites:
    #   1. Install Ollama: https://ollama.com
    #   2. Pull a model:   ollama pull qwen3:latest
    #   3. Start server:   ollama serve   (runs at http://localhost:11434)
    #   4. Set "enabled": True below.
    #
    # Recommended models (pull with ``ollama pull <model>``):
    #   Anthropic-API-compatible:
    #     qwen3:latest, qwen3:8b, qwen3:14b, qwen3:32b
    #     glm4:latest, llama3.3:latest, llama3.2:latest
    #     llama3.2:3b, llama3.2:1b
    #   Google Gemma (Apache-2.0):
    #     gemma3:latest, gemma3:12b, gemma3:4b
    #     (once available: gemma4:latest via HuggingFace)
    #   Code-specialised:
    #     codellama:latest, deepseek-r1:latest, deepseek-coder-v2:latest
    #     phi4:latest, phi4-mini:latest, mistral:latest
    #
    # Security: api_base_url MUST remain a loopback address.
    # ``_validate_ollama_url`` rejects any remote URL to prevent the widget
    # from exfiltrating documentation content to external servers.
    # "ollama": {
    #     "enabled": True,
    #     "label": "Ask Ollama (Local)",
    #     "description": (
    #         "Ask a locally running Ollama model — fully offline, "
    #         "privacy-preserving. Supports Gemma 4, Qwen 3, Llama 3.3, "
    #         "DeepSeek R1, Phi-4, Mistral and more."
    #     ),
    #     "icon": "ollama.svg",
    #     "url_template": "http://localhost:3000/?q={prompt}",
    #     "api_base_url": "http://localhost:11434",   # loopback only
    #     "prompt_template": "Please review this content and answer questions: {url}",
    #     "model": "qwen3:latest",  # or any _OLLAMA_RECOMMENDED_MODELS entry
    #     "type": "local",
    # },

    # --- Tier 3: custom / user-defined endpoint ---------------------------

    # Custom — stub for any user-defined LLM endpoint (private server,
    # internal corporate LLM, OpenAI-compatible proxy, etc.).
    # Override ``api_base_url``, ``model``, and optionally ``url_template``.
    #
    #   from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
    #       _DEFAULT_PROVIDERS,
    #   )
    #   ai_assistant_providers = {
    #       **_DEFAULT_PROVIDERS,
    #       "custom": {
    #           **_DEFAULT_PROVIDERS["custom"],
    #           "enabled": True,
    #           "label": "Ask Internal AI",
    #           "api_base_url": "http://internal-llm.corp/v1",
    #           "model": "company-llm-v2",
    #       },
    #   }
    # "custom": {
    #     "enabled": False,
    #     "label": "Ask Custom AI",
    #     "description": "Ask your custom or self-hosted AI endpoint",
    #     "icon": "custom.svg",
    #     "url_template": "",
    #     "api_base_url": "",
    #     "prompt_template": "Please review this content: {url}\n\n",
    #     "model": "",
    #     "type": "custom",
    # },

    # --- Tier 4: others — disabled by default (alphabetical) --------------

    # Copilot — Microsoft Copilot (GPT-4o powered).
    # "copilot": {
    #     "enabled": False,
    #     "label": "Ask Copilot",
    #     "description": "Ask Microsoft Copilot about this page",
    #     "icon": "copilot.svg",
    #     "url_template": "https://copilot.microsoft.com/?q={prompt}",
    #     "prompt_template": (
    #         "Please review this documentation: {url}\n\nI have questions."
    #     ),
    #     "model": "gpt-4o",
    #     "type": "web",
    # },

    # DeepSeek — strong open-weight reasoning models.
    # Also available via Ollama: ollama pull deepseek-r1:latest
    # "deepseek": {
    #     "enabled": False,
    #     "label": "Ask DeepSeek",
    #     "description": "Ask DeepSeek AI about this page",
    #     "icon": "deepseek.svg",
    #     "url_template": "https://chat.deepseek.com/?q={prompt}",
    #     "prompt_template": "Please read this documentation: {url}\n\nI have questions.",
    #     "model": "deepseek-reasoner",
    #     "type": "web",
    # },

    # Groq — extremely fast open-source LLM inference.
    # "groq": {
    #     "enabled": False,
    #     "label": "Ask Groq",
    #     "description": "Ask Groq (fast LLM inference) about this page",
    #     "icon": "groq.svg",
    #     "url_template": "https://console.groq.com/playground?q={prompt}",
    #     "prompt_template": "Please read: {url}",
    #     "model": "llama-3.3-70b-versatile",
    #     "type": "web",
    # },

    # HuggingFace Chat — supports many open-source models (Llama, Mistral,
    # Qwen, Gemma 4 (31 B, Apache-2.0), etc.) for free.
    # "huggingface": {
    #     "enabled": False,
    #     "label": "Ask HuggingFace",
    #     "description": (
    #         "Ask HuggingFace Chat about this page — supports Llama, Qwen, "
    #         "Gemma 4, Mistral and other open models for free"
    #     ),
    #     "icon": "huggingface.svg",
    #     "url_template": "https://huggingface.co/chat/?q={prompt}",
    #     "prompt_template": (
    #         "Please read this documentation and answer my questions: {url}"
    #     ),
    #     "model": "meta-llama/Llama-3.3-70B-Instruct",
    #     "type": "web",
    # },

    # Mistral — Mistral AI Le Chat.
    # "mistral": {
    #     "enabled": False,
    #     "label": "Ask Mistral",
    #     "description": "Ask Mistral AI Le Chat about this page",
    #     "icon": "mistral.svg",
    #     "url_template": "https://chat.mistral.ai/chat?q={prompt}",
    #     "prompt_template": "Please read this documentation: {url}\n\nI have questions.",
    #     "model": "mistral-large-latest",
    #     "type": "web",
    # },

    # Uncomment to add Perplexity:
    # "perplexity": {
    #     "enabled": False,
    #     "label": "Ask Perplexity",
    #     "description": "Ask Perplexity AI about this page",
    #     "icon": "perplexity.svg",
    #     "url_template": "https://www.perplexity.ai/?q={prompt}",
    #     "prompt_template": "Explain this documentation page: {url}",
    #     "model": "sonar-pro",
    #     "type": "web",
    # },

    # Uncomment to add You.com:
    # "you": {
    #     "enabled": False,
    #     "label": "Ask You.com",
    #     "description": "Ask You.com AI about this page",
    #     "icon": "you.svg",
    #     "url_template": "https://you.com/?q={prompt}",
    #     "prompt_template": (
    #         "Please review this documentation: {url}\n\nI have questions."
    #     ),
    #     "model": "default",
    #     "type": "web",
    # },
}

# ---------------------------------------------------------------------------
# AI Assistant — MCP tool configuration
# ---------------------------------------------------------------------------

# Type:    dict[str, dict]
# Default: _DEFAULT_MCP_TOOLS  (5 tools: vscode, claude_desktop, cursor,
#          windsurf, generic — all disabled by default)
#
# MCP (Model Context Protocol) server integration configuration.  When a tool
# is enabled, the widget renders a "Connect" button that lets users install
# the documentation's MCP server into their AI coding tool.  This lets tools
# like Claude Code, VS Code Copilot, and Cursor query the documentation
# directly via the MCP protocol during development.
#
# Required keys per tool (validated by ``_validate_mcp_tool``):
#   "enabled"      bool  — button shown when True
#   "type"         str   — "vscode" | "claude_desktop" | "cursor" |
#                          "windsurf" | "generic"
#   "label"        str   — button text
#   "description"  str   — tooltip text
#   "icon"         str   — SVG filename in _static/
#
# Type-specific optional keys:
#   "server_name"  str   — MCP server identifier (vscode/cursor/windsurf/generic)
#   "server_url"   str   — SSE endpoint URL; must use http:// or https://
#   "transport"    str   — "sse" (remote server) | "stdio" (local process)
#   "mcpb_url"     str   — mcpb:// or https:// deep-link for Claude Desktop
#
# Security note: ``server_url`` is validated by ``_validate_mcp_tool`` using
# ``_URL_SCHEME_RE``; only http:// and https:// are accepted.  javascript:
# and data: URLs are always rejected.
#
# Tip — extend rather than replace to inherit all upstream defaults:
#
#   from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
#       _DEFAULT_MCP_TOOLS,
#   )
#   ai_assistant_mcp_tools = {
#       **_DEFAULT_MCP_TOOLS,
#       "vscode": {
#           **_DEFAULT_MCP_TOOLS["vscode"],
#           "enabled": True,
#           "server_name": "my-docs-mcp",
#           "server_url": "https://mcp.example.com/sse",
#       },
#   }
ai_assistant_mcp_tools = {
    "vscode": {
        # Enable once your MCP server is deployed at a stable HTTPS URL.
        "enabled": False,  # set True to show the VS Code button
        # Widget renders a VS Code "Install Extension" style deep-link that
        # opens VS Code and installs / activates the MCP server.
        "type": "vscode",
        "label": "Connect to VS Code",
        "description": "Install your MCP server directly into VS Code.",
        "icon": "vscode.svg",
        # Unique identifier shown in the VS Code MCP server sidebar entry.
        "server_name": "your-docs-mcp-server",
        # SSE endpoint URL of your deployed MCP server.
        # Use https:// for production; http://localhost is accepted locally.
        "server_url": "https://your-docs-mcp-server/sse",
        # Transport protocol:
        #   "sse"   — HTTP Server-Sent Events (standard for remote servers)
        #   "stdio" — standard I/O pipes (local command-line MCP servers)
        "transport": "sse",
    },

    "claude_desktop": {
        # Enable once you have a deployable mcpb:// bundle or config ZIP.
        "enabled": False,  # set True to show the Claude button
        # Widget renders a Claude Desktop "mcpb://" deep-link button that
        # opens Claude Desktop and installs the MCP server configuration.
        "type": "claude_desktop",
        "label": "Connect to Claude",
        "description": "Download and run the Claude MCP bundle.",
        "icon": "claude.svg",
        # mcpb:// URL (Claude Desktop's custom protocol for MCP bundles), or
        # an https:// URL pointing to a downloadable MCP config ZIP archive.
        "mcpb_url": "https://docs.example.com/_static/your-mcpb-config.zip",
    },

    "cursor": {
        # Enable once your MCP server is deployed at a stable HTTPS URL.
        "enabled": False,
        # Widget renders a Cursor IDE "Install MCP Server" style deep-link
        # that opens Cursor and activates the MCP server integration.
        "type": "cursor",
        "label": "Connect to Cursor",
        "description": "Connect the MCP server to Cursor IDE.",
        "icon": "cursor.svg",
        # Unique identifier shown in the Cursor MCP server sidebar entry.
        "server_name": "your-docs-mcp-server",
        # SSE endpoint URL of your deployed MCP server.
        # Use https:// for production; http://localhost is accepted locally.
        "server_url": "https://your-docs-mcp-server/sse",
        # Transport protocol:
        #   "sse"   — HTTP Server-Sent Events (standard for remote servers)
        #   "stdio" — standard I/O pipes (local command-line MCP servers)
        "transport": "sse",
    },

    "windsurf": {
        # Enable once your MCP server is deployed at a stable HTTPS URL.
        "enabled": False,
        # Widget renders a Windsurf IDE "Install MCP Server" style deep-link
        # that opens Windsurf and activates the MCP server integration.
        "type": "windsurf",
        "label": "Connect to Windsurf",
        "description": "Connect the MCP server to Windsurf IDE.",
        "icon": "windsurf.svg",
        # Unique identifier shown in the Windsurf MCP server sidebar entry.
        "server_name": "your-docs-mcp-server",
        # SSE endpoint URL of your deployed MCP server.
        "server_url": "https://your-docs-mcp-server/sse",
        "transport": "sse",
    },

    "generic": {
        # Generic fallback for any MCP-compatible client not covered above.
        # Enable to surface a plain "Connect MCP Server" button that exposes
        # the server_url to users who configure their client manually.
        "enabled": False,
        "type": "generic",
        "label": "Connect MCP Server",
        "description": "Connect any MCP-compatible server.",
        "icon": "vscode.svg",
        # Unique identifier used in generic MCP server configs.
        "server_name": "your-docs-mcp-server",
        # SSE endpoint URL of your deployed MCP server.
        "server_url": "https://your-docs-mcp-server/sse",
        "transport": "sse",
    },
}
