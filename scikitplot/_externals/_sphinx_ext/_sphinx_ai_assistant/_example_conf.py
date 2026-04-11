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
# Authors: Mladen Zagorac, The scikit-plots developers
# SPDX-License-Identifier: MIT
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

Widget structure (v0.4.0)
--------------------------
Both the Sphinx extension and the Jupyter widget now share an identical
split-button UX:

    ┌────────────────────────┬──┐
    │  📄  Copy page         │▾ │
    └────────────────────────┴──┘

Clicking **Copy page** (primary) copies the page/cell content as Markdown
to the clipboard and briefly shows "Copied!".

Clicking **▾** opens a dropdown:

    ┌─────────────────────────────────────┐
    │  📄  Copy page                      │  → copy Markdown to clipboard
    │  [M]  View as Markdown              │  → open .md URL in new tab
    │─────────────────────────────────────│
    │  🔶  Ask Claude                     │  → open Claude with page context
    │      Ask Claude about this page     │
    │  🟢  Ask ChatGPT                    │
    │      Ask ChatGPT about this page    │
    │  🔵  Ask Gemini                     │
    │  ...                                │
    │─────────────────────────────────────│  (only when MCP tools enabled)
    │  🔷  Connect to VS Code             │
    └─────────────────────────────────────┘

Jupyter defaults (v0.4.0 changes)
-----------------------------------
* ``include_outputs`` now defaults to ``False`` in all Jupyter functions.
  Pass ``include_outputs=True`` explicitly to include cell output text.
* ``include_raw_image`` remains ``False`` by default.  Pass
  ``include_raw_image=True`` to capture canvas/img thumbnails.
"""

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    # ... your other extensions ...
    "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
]

# ---------------------------------------------------------------------------
# HTML theme — pydata-sphinx-theme (scikit-learn / NumPy / SciPy style)
# ---------------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    # ---- Navigation --------------------------------------------------------
    "navbar_align": "left",
    "show_nav_level": 2,
    "navigation_depth": 4,
    "show_toc_level": 2,

    # ---- Search ------------------------------------------------------------
    "search_bar_text": "Search the docs …",

    # ---- Social / header icons --------------------------------------------
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/your-org/your-project",
            "icon": "fa-brands fa-github",
        },
    ],

    # ---- Footer -----------------------------------------------------------
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],

    # ---- pydata extras ----------------------------------------------------
    "use_edit_page_button": False,
    "show_version_warning_banner": True,
}

# Required for edit-page button (if enabled above)
html_context = {
    "github_user": "your-org",
    "github_repo": "your-project",
    "github_version": "main",
    "doc_path": "docs",
}

# Base URL — used by llms.txt and AI provider prompt templates
html_baseurl = "https://docs.example.com"

# ---------------------------------------------------------------------------
# AI Assistant — master switch
# ---------------------------------------------------------------------------

# Enable or disable the entire extension (default: True)
ai_assistant_enabled = True

# ---------------------------------------------------------------------------
# AI Assistant — button placement
# ---------------------------------------------------------------------------

# Where to render the AI-assistant button.
# "sidebar"  → right sidebar, above the page TOC (works well with pydata)
# "title"    → next to the page heading
# "floating" → fixed floating button at bottom-right corner
# "none"     → disabled (do not render the button)
ai_assistant_position = "sidebar"

# ---------------------------------------------------------------------------
# AI Assistant — content selector for client-side Markdown conversion
# ---------------------------------------------------------------------------

# CSS selector used by the JavaScript widget to extract page content for
# copy-as-Markdown and AI-chat features.
#
# pydata-sphinx-theme:  "article.bd-article"  or  'div[role="main"]'
# Furo theme:           'article[role="main"]'
# Alabaster / Classic:  "div.document"
# Read the Docs theme:  'div[role="main"]'
ai_assistant_content_selector = "article.bd-article"

# ---------------------------------------------------------------------------
# AI Assistant — CSS selectors for server-side Markdown extraction
# ---------------------------------------------------------------------------

# Ordered list of CSS selectors the *build-time* Markdown generator probes
# to locate the main content element in each HTML file.  The first selector
# that matches is used.
ai_assistant_content_selectors = [
    "article.bd-article",      # pydata-sphinx-theme >= 0.13
    'div[role="main"]',        # pydata-sphinx-theme (older), RTD
    'article[role="main"]',    # Furo
    "div.document",            # Classic / Alabaster
    "main",                    # Generic HTML5
    "div.body",                # Very old themes
    "article",                 # Last-resort fallback
]

# Optionally specify a theme preset to add theme-specific selectors
# automatically.  Supported: "pydata_sphinx_theme", "furo", "mkdocs",
# "mkdocs_material", "jekyll", "hugo", "docusaurus", "vitepress", etc.
# ai_assistant_theme_preset = "pydata_sphinx_theme"

# ---------------------------------------------------------------------------
# AI Assistant — Markdown file generation
# ---------------------------------------------------------------------------

# Generate a ``.md`` companion for every ``.html`` file after the build.
# Requires: pip install beautifulsoup4 markdownify
ai_assistant_generate_markdown = True

# Path substrings excluded from Markdown generation
ai_assistant_markdown_exclude_patterns = [
    "genindex",
    "search",
    "py-modindex",
    "_sources",          # Sphinx source download files
    "_static",           # Static assets have no readable prose
]

# HTML tags stripped (with their content) before Markdown conversion.
# "nav", "footer", "header" are stripped to remove site chrome.
ai_assistant_strip_tags = ["script", "style", "nav", "footer"]

# Maximum number of parallel worker processes for Markdown generation.
# None → auto-detect (CPU count, capped at 8).
ai_assistant_max_workers = None

# ---------------------------------------------------------------------------
# AI Assistant — llms.txt generation
# ---------------------------------------------------------------------------

# Write an llms.txt index file listing all generated .md page URLs.
# See: https://llmstxt.org/
ai_assistant_generate_llms_txt = True

# Base URL prepended to .md paths in llms.txt.
# Falls back to html_baseurl when empty.
ai_assistant_base_url = ""  # use html_baseurl above

# Limit the number of entries in llms.txt (None = unlimited)
ai_assistant_llms_txt_max_entries = None

# When True, embed the full Markdown content of each page inside llms.txt.
# Warning: produces a very large file for large documentation sites.
ai_assistant_llms_txt_full_content = False

# ---------------------------------------------------------------------------
# AI Assistant — feature flags
# ---------------------------------------------------------------------------

ai_assistant_features = {
    # Copy page content as Markdown to clipboard (primary button action)
    "markdown_export": True,
    # Open raw Markdown of the current page in a new browser tab
    "view_markdown": True,
    # Render deep-links to Claude / ChatGPT / Gemini with page context
    "ai_chat": True,
    # Show MCP server installation buttons
    "mcp_integration": False,
}

# ---------------------------------------------------------------------------
# AI Assistant — prompt customisation
# ---------------------------------------------------------------------------

# Optional goal annotation prepended to all AI prompts as "Goal: <text>".
# Set to None to omit.
ai_assistant_intention = None
# Example: ai_assistant_intention = "Help me understand the API"

# Optional background context injected into all AI prompts.
ai_assistant_custom_context = None
# Example: ai_assistant_custom_context = "This is a Python ML library."

# Optional raw prefix prepended before everything else in the final prompt.
ai_assistant_custom_prompt_prefix = None

# ---------------------------------------------------------------------------
# AI Assistant — Jupyter / notebook settings
# ---------------------------------------------------------------------------

# When True, the widget JS captures all notebook cells (notebook review mode).
# Typically set per-call via display_jupyter_notebook_ai_button().
ai_assistant_notebook_mode = False

# When True (Sphinx builds), include cell outputs in captured content.
# Note: In Jupyter Python functions, include_outputs defaults to False since
# v0.4.0 — pass include_outputs=True explicitly when needed.
ai_assistant_include_outputs = True

# When True, capture canvas/img thumbnails and append to the AI prompt.
# Defaults to False — set True only when visual output is important.
ai_assistant_include_raw_image = False

# ---------------------------------------------------------------------------
# AI Assistant — AI provider configuration
# ---------------------------------------------------------------------------

# Full provider registry.  Each provider must include all required keys:
# enabled, label, description, icon, url_template, prompt_template, model, type.
#
# Tip: import _DEFAULT_PROVIDERS to extend rather than replace:
#   from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
#       _DEFAULT_PROVIDERS,
#   )
#   ai_assistant_providers = {
#       **_DEFAULT_PROVIDERS,
#       "claude": {**_DEFAULT_PROVIDERS["claude"], "enabled": True},
#   }
#
# Providers support:
#   type = "web"    — opens a browser tab with pre-filled prompt
#   type = "local"  — local AI server (e.g. Ollama); disabled by default
#   type = "api"    — API-only provider
#   type = "custom" — user-defined endpoint
#
# url_template: {prompt} placeholder is replaced with the encoded prompt
# prompt_template: {url} is the page Markdown URL; {content} is page text
ai_assistant_providers = {
    "claude": {
        "enabled": True,
        "label": "Ask Claude",
        "description": "Ask Claude about this documentation page.",
        "icon": "claude.svg",
        "url_template": "https://claude.ai/new?q={prompt}",
        "prompt_template": (
            "Hi! Please read this documentation page: {url}\n\n"
            "I have questions about it."
        ),
        "model": "claude-sonnet-4-6",
        "type": "web",
    },
    "chatgpt": {
        "enabled": True,
        "label": "Ask ChatGPT",
        "description": "Ask ChatGPT about this documentation page.",
        "icon": "chatgpt.svg",
        "url_template": "https://chatgpt.com/?q={prompt}",
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
    # Uncomment and configure to enable additional providers:
    # "ollama": {
    #     "enabled": True,  # requires local Ollama server
    #     "label": "Ask Ollama (Local)",
    #     "description": "Ask a locally running Ollama model — fully offline.",
    #     "icon": "ollama.svg",
    #     "url_template": "http://localhost:3000/?q={prompt}",
    #     "api_base_url": "http://localhost:11434",
    #     "prompt_template": "Please review this content: {url}",
    #     "model": "qwen3:latest",   # or llama3.2:latest, gemma3:latest, etc.
    #     "type": "local",
    # },
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
}

# ---------------------------------------------------------------------------
# AI Assistant — MCP tool configuration
# ---------------------------------------------------------------------------

ai_assistant_mcp_tools = {
    "vscode": {
        "enabled": False,                 # set True to show the VS Code button
        "type": "vscode",
        "label": "Connect to VS Code",
        "description": "Install your MCP server directly into VS Code.",
        "icon": "vscode.svg",
        "server_name": "your-docs-mcp-server",
        "server_url": "https://your-docs-mcp-server/sse",
        "transport": "sse",               # "sse" or "stdio"
    },
    "claude_desktop": {
        "enabled": False,                 # set True to show the Claude button
        "type": "claude_desktop",
        "label": "Connect to Claude",
        "description": "Download and run the Claude MCP bundle.",
        "icon": "claude.svg",
        "mcpb_url": "https://docs.example.com/_static/your-mcpb-config.zip",
    },
}

# ---------------------------------------------------------------------------
# AI Assistant — Jupyter notebook usage examples
# ---------------------------------------------------------------------------
# (These are not conf.py settings — they are usage examples for notebooks.)

# Basic usage after a matplotlib plot:
#
#   from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
#       display_jupyter_ai_button,
#   )
#   import matplotlib.pyplot as plt
#
#   plt.plot([1, 2, 3])
#   plt.show()
#   display_jupyter_ai_button(
#       content="A line chart showing values 1, 2, 3.",
#       providers=["claude", "chatgpt", "gemini"],
#       intention="Explain the trend",
#       # include_raw_image=True,   # opt-in: capture canvas/img thumbnails
#   )

# Full notebook review (at the end of a notebook):
#
#   from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
#       display_jupyter_notebook_ai_button,
#   )
#   display_jupyter_notebook_ai_button(
#       intention="Review this notebook for bugs and suggest improvements",
#       providers=["claude", "chatgpt"],
#       include_outputs=True,    # opt-in: include cell outputs (tracebacks, etc.)
#   )

# With Ollama (fully local, offline):
#
#   display_jupyter_ai_button(
#       providers=["ollama"],
#       provider_configs={
#           "ollama": {"enabled": True, "model": "qwen3:latest"},
#       },
#       intention="Explain this analysis",
#   )

# With Markdown URL context (recommended when page_url is known):
#
#   display_jupyter_ai_button(
#       page_url="https://docs.example.com/api/module.html",
#       # Widget will derive https://docs.example.com/api/module.md
#       # and use it in all AI provider prompts — same as Sphinx widget.
#       providers=["claude", "chatgpt"],
#   )
