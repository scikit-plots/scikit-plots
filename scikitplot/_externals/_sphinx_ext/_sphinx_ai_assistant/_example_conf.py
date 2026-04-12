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

Usage
-----
Copy the relevant sections into your project's ``conf.py`` and adjust.

Theme note
----------
pydata-sphinx-theme renders the main article content inside::

    <article class="bd-article" role="main">

so the CSS selectors are configured accordingly.  For other themes see the
comments next to each selector value.

Quick-start minimal config
--------------------------
The absolute minimum to enable the extension in ``conf.py``::

    extensions = ["scikitplot._externals._sphinx_ext._sphinx_ai_assistant"]
    ai_assistant_enabled = True
    html_baseurl = "https://docs.example.com"

All other values shown below are optional — the extension ships with sensible
defaults for every setting.
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
# Common preset names accepted by ``ai_assistant_theme_preset`` (below):
#   "pydata_sphinx_theme"  — scikit-learn / NumPy / SciPy / pandas style
#   "furo"                 — Furo theme (https://pradyunsg.me/furo/)
#   "sphinx_rtd_theme"     — Read the Docs theme
#   "sphinx_book_theme"    — Jupyter Book / sphinx-book-theme
#   "alabaster"            — Sphinx default theme
#   "classic"              — Sphinx Classic
#   "mkdocs"               — MkDocs (non-Sphinx, standalone use)
#   "mkdocs_material"      — MkDocs Material
#   "jekyll"               — Jekyll static sites
#   "hugo"                 — Hugo static sites
#   "plain_html"           — Generic HTML5 without a framework
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
# Default: None  (auto-detect)
#
# Maximum number of parallel worker processes for Markdown generation.
#
#   None  — auto: max(1, min(cpu_count, 8)).  Scales to the local machine
#           without overwhelming CI containers (which often have 2 CPUs).
#   1     — single process.  Use for debugging — stack traces are clean and
#           sequential, and pdb works normally inside workers.
#   N > 1 — explicit cap.  Useful to limit memory usage on memory-constrained
#           machines when converting very large HTML files.
#
# Developer note: Workers are spawned by ``ProcessPoolExecutor``.  The
# worker functions (``_process_single_html_file``, ``_process_html_file_worker``)
# MUST be defined at module scope — not as lambdas or closures — so the
# multiprocessing pickle protocol can serialise them across process boundaries.
ai_assistant_max_workers = None

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
#       Show the row of AI-provider buttons (Claude, ChatGPT, Gemini, …).
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
    "markdown_export": True,
    "view_markdown": True,
    "ai_chat": True,
    "mcp_integration": True,
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

# ---------------------------------------------------------------------------
# AI Assistant — AI provider configuration
# ---------------------------------------------------------------------------

# Type:    dict[str, dict]
# Default: _DEFAULT_PROVIDERS  (12 providers; 3 enabled by default —
#          claude, gemini, chatgpt — 9 disabled)
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
# Optional key:
#   "api_base_url"     str   — base endpoint for "local" / "api" type
#                              providers (e.g. "http://localhost:11434" for
#                              Ollama).
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

    "claude": {
        # Enabled: users can click "Ask Claude" without any setup.
        "enabled": True,
        # Button label shown inside the AI-assistant panel.
        "label": "Ask Claude",
        # Tooltip / screen-reader accessible description.
        "description": "Ask Claude about this documentation page.",
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
    },

    "chatgpt": {
        "enabled": True,
        "label": "Ask ChatGPT",
        "description": "Ask ChatGPT about this documentation page.",
        "icon": "chatgpt.svg",
        "url_template": "https://chatgpt.com/?q={prompt}",
        # ChatGPT's browsing mode fetches and reads the URL when told to.
        "prompt_template": "Read {url} so I can ask questions about it.",
        "model": "gpt-4o",
        "type": "web",
    },

    "gemini": {
        "enabled": True,
        "label": "Ask Gemini",
        "description": "Ask Google Gemini about this documentation page.",
        "icon": "gemini.svg",
        "url_template": "https://gemini.google.com/app?q={prompt}",
        "prompt_template": "Please review this documentation: {url}\n\nI have questions.",
        "model": "gemini-2.5-flash",
        "type": "web",
    },

    # --- Tier 2: disabled by default — uncomment and configure to enable ---

    # Ollama — fully offline, privacy-preserving local inference.
    # Prerequisites:
    #   1. Install Ollama: https://ollama.com
    #   2. Pull a model:  ollama pull qwen3:latest
    #   3. Start server:  ollama serve   (runs at http://localhost:11434)
    #   4. Set "enabled": True below.
    #
    # Supported models (pull with ``ollama pull <model>``):
    #   qwen3:latest, llama3.3:latest, llama3.2:latest, gemma3:latest,
    #   deepseek-r1:latest, phi4-mini:latest, mistral:latest, codellama:latest
    #
    # Security: api_base_url MUST remain a loopback address.
    # ``_validate_ollama_url`` rejects any remote URL to prevent the widget
    # from exfiltrating documentation content to external servers.
    # "ollama": {
    #     "enabled": True,
    #     "label": "Ask Ollama (Local)",
    #     "description": "Fully offline local inference — no data leaves your machine.",
    #     "icon": "ollama.svg",
    #     "url_template": "http://localhost:3000/?q={prompt}",
    #     "api_base_url": "http://localhost:11434",   # loopback only
    #     "prompt_template": "Please review this content: {url}",
    #     "model": "qwen3:latest",
    #     "type": "local",
    # },

    # DeepSeek — strong open-weight reasoning models.
    # Also available via Ollama: ollama pull deepseek-r1:latest
    # "deepseek": {
    #     "enabled": False,
    #     "label": "Ask DeepSeek",
    #     "description": "Ask DeepSeek AI about this page.",
    #     "icon": "deepseek.svg",
    #     "url_template": "https://chat.deepseek.com/?q={prompt}",
    #     "prompt_template": "Please read this documentation: {url}\n\nI have questions.",
    #     "model": "deepseek-reasoner",
    #     "type": "web",
    # },

    # Uncomment to add Perplexity or any other AI chat service:
    # "perplexity": {
    #     "enabled": True,
    #     "label": "Ask Perplexity",
    #     "url_template": "https://www.perplexity.ai/?q={prompt}",
    #     "prompt_template": "Explain this documentation page: {url}",
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
        "enabled": False,
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
        "enabled": False,
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
}
