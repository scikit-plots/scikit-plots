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
    "article.bd-article",      # pydata-sphinx-theme ≥ 0.13
    'div[role="main"]',        # pydata-sphinx-theme (older), RTD
    'article[role="main"]',    # Furo
    "div.document",            # Classic / Alabaster
    "main",                    # Generic HTML5
    "div.body",                # Very old themes
    "article",                 # Last-resort fallback
]

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

# ---------------------------------------------------------------------------
# AI Assistant — feature flags
# ---------------------------------------------------------------------------

ai_assistant_features = {
    # Copy page content as Markdown to clipboard
    "markdown_export": True,
    # Open raw Markdown of the current page in a new browser tab
    "view_markdown": True,
    # Render deep-links to Claude / ChatGPT with page context
    "ai_chat": True,
    # Show MCP server installation buttons
    "mcp_integration": False,
}

# ---------------------------------------------------------------------------
# AI Assistant — AI provider configuration
# ---------------------------------------------------------------------------

ai_assistant_providers = {
    "claude": {
        "enabled": True,
        "label": "Ask Claude",
        "description": "Ask Claude about this documentation page.",
        "icon": "claude.svg",
        "url_template": "https://claude.ai/new?q={prompt}",
        "prompt_template": (
            "Get familiar with the documentation at {url} "
            "so I can ask questions about it."
        ),
    },
    "chatgpt": {
        "enabled": True,
        "label": "Ask ChatGPT",
        "description": "Ask ChatGPT about this documentation page.",
        "icon": "chatgpt.svg",
        "url_template": "https://chatgpt.com/?q={prompt}",
        "prompt_template": (
            "Get familiar with the documentation at {url} "
            "so I can ask questions about it."
        ),
    },
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
