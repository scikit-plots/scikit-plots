# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_static/__init__.py
#
# flake8: noqa: D213
#
# Authors: Mladen Zagorac, The scikit-plots developers
# SPDX-License-Identifier: MIT / BSD-3-Clause

"""
Static-asset subpackage for the Sphinx AI Assistant extension.

This module provides **inline SVG icons** as base64 data URIs so the
extension widget remains fully self-contained with zero network requests.
Icons are used as fallbacks when the corresponding ``.svg`` file is absent
from the ``_static/`` directory on disk.

Public API:

_PROVIDER_META : dict
    Maps lower-cased provider name → ``{"icon": <data-uri>, "desc": str}``.
_SVG_COPY, _SVG_MARKDOWN, _SVG_CLAUDE, … : str
    Individual base64 data-URI constants.

Notes
-----
**Developer note** — Adding a new icon:

1. Create a minimal monochrome SVG (24x24 px recommended, ``currentColor``
   stroke so dark-mode filter works automatically).
2. Base64-encode it::

       python -c "import base64; print(base64.b64encode(open('icon.svg','rb').read()).decode())"

3. Assign the result to a new ``_SVG_<NAME>`` constant below.
4. Add an entry to ``_PROVIDER_META`` (for providers) or ``_ICON_META``
   (for panel action icons).
5. Add a corresponding test in ``tests/test___init__.py``.

**Security note** — Icons are injected only as ``src`` attributes on
``<img>`` elements (via the Python extension) or as ``data:`` URIs
returned by ``_resolve_icon()``.  They are never written to the
filesystem and contain no executable content.

Examples
--------
>>> from scikitplot._externals._sphinx_ext._sphinx_ai_assistant._static import (
...     _PROVIDER_META,
...     _SVG_DEFAULT,
... )
>>> assert "claude" in _PROVIDER_META
>>> assert _PROVIDER_META["claude"]["icon"].startswith("data:image/svg+xml;base64,")
"""

from __future__ import annotations

import base64
import pathlib

__all__ = [
    "_ICON_META",
    "_PROVIDER_META",
    "_SVG_CHATGPT",
    "_SVG_CLAUDE",
    "_SVG_COPY",
    "_SVG_COPY_ANSWER",
    "_SVG_DEFAULT",
    "_SVG_EXPORT_TXT",
    "_SVG_GEMINI",
    "_SVG_KEYBOARD",
    "_SVG_MARKDOWN",
    "_SVG_NEW_CHAT",
    "_SVG_OLLAMA",
    "_SVG_PRIVACY",
    "_SVG_SEARCH_AI",
    "export_svg2base64",
]


def export_svg2base64(directory: str | None = None) -> dict[str, str]:
    """Encode every ``*.svg`` file in *directory* as a base64 data URI.

    Parameters
    ----------
    directory : str or None, optional
        Directory to scan.  Defaults to the current working directory.

    Returns
    -------
    dict of str → str
        Mapping of ``filename.svg`` → ``data:image/svg+xml;base64,...``.
    """
    root = pathlib.Path(directory) if directory else pathlib.Path.cwd()
    return {
        f.name: f"data:image/svg+xml;base64,{base64.b64encode(f.read_bytes()).decode()}"
        for f in sorted(root.glob("*.svg"))
    }


# ---------------------------------------------------------------------------
# Inline SVG icons — base64 data URIs
# ---------------------------------------------------------------------------
# Convention: stroke="currentColor" so the dark-mode CSS filter
# (brightness(1.6)) works without duplicating icon variants.

# ── Existing icons ──────────────────────────────────────────────────────────

_SVG_COPY: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAx"
    "NiAxNiIgZmlsbD0ibm9uZSI+PHJlY3QgeD0iNS41IiB5PSIyLjUiIHdpZHRoPSI4IiBoZWln"
    "aHQ9IjkiIHJ4PSIxLjUiIHN0cm9rZT0iIzI0MjkyZiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48"
    "cmVjdCB4PSIyLjUiIHk9IjUuNSIgd2lkdGg9IjgiIGhlaWdodD0iOSIgcng9IjEuNSIgZmls"
    "bD0id2hpdGUiIHN0cm9rZT0iIzI0MjkyZiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48L3N2Zz4="
)

_SVG_MARKDOWN: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAx"
    "NiAxNiI+PHJlY3QgeD0iMC41IiB5PSIyLjUiIHdpZHRoPSIxNSIgaGVpZ2h0PSIxMSIgcng9"
    "IjIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzI0MjkyZiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48"
    "cGF0aCBmaWxsPSIjMjQyOTJmIiBkPSJNMyAxMVY1LjVoMS41TDYgNy41IDcuNSA1LjVIOVYx"
    "MUg3LjVWNy44TDYgOS44IDQuNSA3LjhWMTFIM3ptNy41IDBMOC41IDloMS41VjUuNWgxLjVW"
    "OUgxM0wxMC41IDExeiIvPjwvc3ZnPg=="
)

_SVG_CLAUDE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzc0NUI0RiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEy"
    "czQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEg"
    "MC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPjwvc3Zn"
    "Pg=="
)

_SVG_CHATGPT: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzEwYTM3ZiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEy"
    "czQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTEgMTVoLTJ2LTZo"
    "MnY2em0wLThoLTJWN2gydjJ6Ii8+PC9zdmc+"
)

_SVG_GEMINI: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzQyODVGNCIgZD0iTTEyIDJsLTEgOUgzbDcuNSA1LjUtMi41"
    "IDguNUwxMiAxOWw0IDYtMi41LTguNUwyMSAxMWgtOHoiLz48L3N2Zz4="
)

_SVG_OLLAMA: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9r"
    "ZT0iIzMzMyIgc3Ryb2tlLXdpZHRoPSIyIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0i"
    "NCIgZmlsbD0iIzMzMyIvPjwvc3ZnPg=="
)

# ── Fallback for unknown providers/tools ────────────────────────────────────
# NOTE: Only one _SVG_DEFAULT constant. Previous version had a duplicate
# assignment — removed in this revision (BUG-1 fix).
_SVG_DEFAULT: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzg4OCIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDgg"
    "MTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTEgMTVoLTJ2LTZoMnY2em0w"
    "LThoLTJWN2gydjJ6Ii8+PC9zdmc+"
)

# ── New panel-action icons (added) ───────────────────────────────────────────

# Refresh / rotate-ccw: "Start a new chat"
_SVG_NEW_CHAT: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PHBvbHlsaW5lIHBvaW50cz0iMSA0IDEgMTAgNyAxMCIvPjxwYXRo"
    "IGQ9Ik0zLjUxIDE1YTkgOSAwIDEgMCAuNDktNC41Ii8+PC9zdmc+"
)

# Download arrow: "Export AI conversation as txt"
_SVG_EXPORT_TXT: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PHBhdGggZD0iTTIxIDE1djRhMiAyIDAgMCAxLTIgMkg1YTIgMiAw"
    "IDAgMS0yLTJ2LTQiLz48cG9seWxpbmUgcG9pbnRzPSI3IDEwIDEyIDE1IDE3IDEw"
    "Ii8+PGxpbmUgeDE9IjEyIiB5MT0iMTUiIHgyPSIxMiIgeTI9IjMiLz48L3N2Zz4="
)

# Two overlapping rectangles: "Copy this answer"
_SVG_COPY_ANSWER: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PHJlY3QgeD0iOSIgeT0iOSIgd2lkdGg9IjEzIiBoZWlnaHQ9IjEz"
    "IiByeD0iMiIgcnk9IjIiLz48cGF0aCBkPSJNNSAxNUg0YTIgMiAwIDAgMS0yLTJW"
    "NGEyIDIgMCAwIDEgMi0yaDlhMiAyIDAgMCAxIDIgMnYxIi8+PC9zdmc+"
)

# Shield: Privacy Policy
_SVG_PRIVACY: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PHBhdGggZD0iTTEyIDIyczgtNCA4LTEwVjVsLTgtMy04IDN2N2Mw"
    "IDYgOCAxMCA4IDEweiIvPjwvc3ZnPg=="
)

# Magnifier with plus: AI Search
_SVG_SEARCH_AI: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PGNpcmNsZSBjeD0iMTEiIGN5PSIxMSIgcj0iOCIvPjxsaW5lIHgx"
    "PSIyMSIgeTE9IjIxIiB4Mj0iMTYuNjUiIHkyPSIxNi42NSIvPjxwYXRoIGQ9Ik04"
    "IDExaDZNMTEgOHY2IiBzdHJva2Utd2lkdGg9IjEuNSIvPjwvc3ZnPg=="
)

# Keyboard: shortcut hint
_SVG_KEYBOARD: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PHJlY3QgeD0iMiIgeT0iNiIgd2lkdGg9IjIwIiBoZWlnaHQ9IjEy"
    "IiByeD0iMiIvPjxwYXRoIGQ9Ik02IDEwaC4wMU0xMCAxMGguMDFNMTQgMTBoLjAx"
    "TTE4IDEwaC4wMU04IDE0aDgiLz48L3N2Zz4="
)


# ---------------------------------------------------------------------------
# Provider icon + description registry
# ---------------------------------------------------------------------------

#: Map lower-cased provider / MCP-tool name →
#: ``{"icon": <base64-data-URI>, "desc": <str>}``.
#:
#: Used as a fallback when the corresponding ``.svg`` file is absent from
#: the ``_static/`` directory on disk.  Keys match those in
#: ``_DEFAULT_PROVIDERS`` and ``_DEFAULT_MCP_TOOLS`` in the parent module.
_PROVIDER_META: dict[str, dict[str, str]] = {
    # AI providers
    "claude": {"icon": _SVG_CLAUDE, "desc": "Anthropic Claude AI"},
    "chatgpt": {"icon": _SVG_CHATGPT, "desc": "OpenAI ChatGPT"},
    "gemini": {"icon": _SVG_GEMINI, "desc": "Google Gemini AI"},
    "ollama": {"icon": _SVG_OLLAMA, "desc": "Local Ollama model"},
    "mistral": {"icon": _SVG_DEFAULT, "desc": "Mistral AI"},
    "perplexity": {"icon": _SVG_DEFAULT, "desc": "Perplexity AI"},
    "copilot": {"icon": _SVG_DEFAULT, "desc": "GitHub Copilot"},
    "groq": {"icon": _SVG_DEFAULT, "desc": "Groq fast inference"},
    "you": {"icon": _SVG_DEFAULT, "desc": "You.com AI search"},
    "deepseek": {"icon": _SVG_DEFAULT, "desc": "DeepSeek AI"},
    "huggingface": {"icon": _SVG_DEFAULT, "desc": "Hugging Face Hub"},
    "custom": {"icon": _SVG_DEFAULT, "desc": "Custom AI endpoint"},
    # MCP tool keys
    "vscode": {"icon": _SVG_DEFAULT, "desc": "VS Code MCP server"},
    "claude_desktop": {"icon": _SVG_CLAUDE, "desc": "Claude Desktop MCP"},
    "cursor": {"icon": _SVG_DEFAULT, "desc": "Cursor IDE MCP"},
    "windsurf": {"icon": _SVG_DEFAULT, "desc": "Windsurf IDE MCP"},
    "generic": {"icon": _SVG_DEFAULT, "desc": "Generic MCP server"},
}

#: Map action key → ``{"icon": <data-URI>, "desc": <str>}``
#: for panel header and footer action buttons.
#: Used as fallbacks when the SVG file is missing from ``_static/``.
_ICON_META: dict[str, dict[str, str]] = {
    "copy": {"icon": _SVG_COPY, "desc": "Copy page as Markdown"},
    "markdown": {"icon": _SVG_MARKDOWN, "desc": "View as Markdown"},
    "new-chat": {"icon": _SVG_NEW_CHAT, "desc": "Start a new chat"},
    "export-txt": {"icon": _SVG_EXPORT_TXT, "desc": "Export conversation as txt"},
    "copy-answer": {"icon": _SVG_COPY_ANSWER, "desc": "Copy this answer"},
    "privacy": {"icon": _SVG_PRIVACY, "desc": "Privacy Policy"},
    "search-ai": {"icon": _SVG_SEARCH_AI, "desc": "AI search"},
    "keyboard": {"icon": _SVG_KEYBOARD, "desc": "Keyboard shortcuts"},
}
