# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_static/__init__.py
#
# This module was copied and adapted from the sphinx-ai-assistant project.
# https://github.com/mlazag/sphinx-ai-assistant
#
# Authors: Mladen Zagorac, The scikit-plots developers
# SPDX-License-Identifier: MIT

"""
Static-asset subpackage for the Sphinx AI Assistant extension.

This module provides **inline SVG icons** as base64 data URIs so the
extension widget remains fully self-contained with zero network requests.
Icons are used as fallbacks when the corresponding ``.svg`` file is absent
from the ``_static/`` directory on disk.

Public API
----------
_PROVIDER_META : dict
    Maps lower-cased provider name → ``{"icon": <data-uri>, "desc": str}``.
_SVG_COPY, _SVG_MARKDOWN, _SVG_CLAUDE, … : str
    Individual base64 data-URI constants.

Notes
-----
**Developer note** — Adding a new provider icon:

1. Create a minimal monochrome SVG (16x16 or 24x24 px).
2. Base64-encode it::

       python -c "import base64; print(base64.b64encode(open('icon.svg','rb').read()).decode())"

3. Assign the result to a new ``_SVG_<NAME>`` constant below.
4. Add an entry to ``_PROVIDER_META``.
5. Add a corresponding test in ``tests/test___init__.py``.

**Security note** — Icons are injected only as CSS ``background-image``
values in the browser; they are never written to the filesystem by Python
code and contain no executable content.

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

__all__ = [
    "_PROVIDER_META",
    "_SVG_CHATGPT",
    "_SVG_CLAUDE",
    "_SVG_COPY",
    "_SVG_DEFAULT",
    "_SVG_GEMINI",
    "_SVG_MARKDOWN",
    "_SVG_OLLAMA",
]

# ---------------------------------------------------------------------------
# Inline SVG icons — base64 data URIs
# ---------------------------------------------------------------------------
# Each constant holds a minimal monochrome SVG encoded as a base64 data URI
# for use as a CSS ``background-image`` value.  Keeping them inline means
# the widget is fully self-contained with zero network requests.

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

_SVG_DEFAULT: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzg4OCIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDgg"
    "MTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTEgMTVoLTJ2LTZoMnY2em0w"
    "LThoLTJWN2gydjJ6Ii8+PC9zdmc+"
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
    "claude": {"icon": _SVG_CLAUDE, "desc": "Anthropic's Claude AI"},
    "chatgpt": {"icon": _SVG_CHATGPT, "desc": "OpenAI's ChatGPT"},
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
