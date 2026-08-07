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

# https://www.svgviewer.dev/svg-to-data-uri
# https://www.svgrepo.com/
# https://github.com/FirefoxUX/acorn-icons/tree/main/icons
# https://github.com/mozilla-firefox/firefox/tree/main/toolkit/themes/shared/icons

from __future__ import annotations

import base64
import pathlib

__all__ = [
    "_ICON_META",
    "_PROVIDER_META",
    "_SVG_BOT_ASSISTANT",
    "_SVG_BRUSH_SPARKLE",
    "_SVG_CELEBRATION",
    "_SVG_CHATGPT",
    "_SVG_CHECK_ANSWER",
    "_SVG_CLAUDE",
    "_SVG_COMMENT_DISCUSSION",
    "_SVG_COPY",
    "_SVG_COPY_ANSWER",
    "_SVG_DEFAULT",
    "_SVG_ERROR_ALERT",
    "_SVG_EXPORT_TXT",
    "_SVG_GEMINI",
    "_SVG_KEYBOARD",
    "_SVG_MARKDOWN",
    "_SVG_MENU_SPARKLE",
    "_SVG_MINIMIZE_COLLAPSE",
    "_SVG_NEW_CHAT",
    "_SVG_NEW_CHAT_COMPOSE",
    "_SVG_OLLAMA",
    "_SVG_PRINTER",
    "_SVG_PRIVACY",
    "_SVG_RETRY",
    "_SVG_SEARCH_AI",
    "_SVG_SEARCH_SPARKLE",
    "_SVG_SHIELD_ALERT",
    "_SVG_SPARKLE",
    "_SVG_SPARKLE_LEFT_2_NOVA_UP_DOWN",
    "_SVG_SPARKLE_RIGHT_1_NOVA_TOP",
    "_SVG_SPARKLE_RIGHT_2_NOVA_UP_DOWN",
    "_SVG_SYNC_RETRY",
    "_SVG_SYNC_RETRY_REVERSE",
    "_SVG_THUMB_DOWN",
    "_SVG_THUMB_UP",
    "_SVG_UPLOAD",
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

# Compose / spark: alternate "Start a new chat" glyph (filled, 16x16 viewBox).
# Additive and inert — NOT yet wired to the active "new-chat" _ICON_META entry;
# provided for future usage.  To adopt it, point the "new-chat" entry below
# (and ICONS.newChat in ai-assistant.js) at this constant instead.
_SVG_NEW_CHAT_COMPOSE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdC"
    "b3g9IjAgMCAxNiAxNiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2IiBmaWxsPSJu"
    "b25lIj48cGF0aCBmaWxsPSIjMjQyOTJmIiBkPSJNMTMuNzUgMTBjMC0yLjM4"
    "OS0uOTgzLTQuMTMxLTEuNjk2LTUuMDgtLjE5LjQzNy0uNzQgMS4zMy0yLjA1"
    "NCAxLjMzLS43NjUgMC0xLjMtLjMzNC0xLjY0My0uNzEyLS4zMDYtLjMzOC0u"
    "NDU1LS43MDYtLjUxMy0uOTAybC0uMDEtLjAzN2MtLjA5Ny0uMzYtLjE3Ni0u"
    "NzM4LS4yNTMtMS4wNzktLjA3OS0uMzUtLjE1OS0uNjc2LS4yNjItLjk4LS4x"
    "MDgtLjMxOC0uMjQtLjYtLjQxMi0uODQ0LS44NjUgMS44Mi0yLjAwMiAzLjA1"
    "LTIuODk5IDQuMTUxQzIuOTggNy4xMTEgMi4yNSA4LjIyIDIuMjUgMTBjMCAx"
    "LjU0NS45MjMgMi45NTUgMi4zNzQgMy44MzEtLjA3NC0uMjc3LS4xMTUtLjU2"
    "NS0uMTIzLS44NTVsLS4wMDEtLjEwNGMwLS45NTcuNTIyLTEuNzg0IDEuMTA3"
    "LTIuNDcyLjU4LS42ODUgMS4zNTItMS4zNzEgMS45NTItMS45NjhsLjAyNC0u"
    "MDIyYy4yNDUtLjIyLjYyMi0uMjEzLjg1OC4wMjIuNjEzLjYxIDEuMzcyIDEu"
    "MzEgMS45NTYgMi4wMTIuNTc1LjY5MSAxLjEwMyAxLjUyIDEuMTAzIDIuNDI4"
    "bC0uMDAxLjEwNGMtLjAwOC4yOS0uMDQ5LjU3OC0uMTIzLjg1NSAxLjQ1LS44"
    "NzYgMi4zNzQtMi4yODYgMi4zNzQtMy44MzFNMTUgMTBjMCAyLjgxNy0yLjI0"
    "MSA1LjA0Ni01LjAzNiA1Ljc1NmwtLjEzMy4wMzJjLS4yOTcuMDctLjYwMS0u"
    "MDg1LS43Mi0uMzY2LS4xMTgtLjI4LS4wMTYtLjYwNy4yNDItLjc3bC4wNTMt"
    "LjAzNWMuNTI4LS4zNjIuODI0LS45NjcuODQzLTEuNjc0bC4wMDEtLjA3YzAt"
    "LjQ0My0uMjcxLS45NzctLjgxNC0xLjYzLS40MTYtLjUtLjkyLS45OS0xLjQz"
    "OC0xLjQ5NC0uNTIuNS0xLjAxOS45NjUtMS40MzggMS40Ni0uNTMzLjYyNy0u"
    "ODEgMS4xNjQtLjgxIDEuNjYzbC4wMDEuMDcxYy4wMi43My4zMzUgMS4zNTMu"
    "ODk2IDEuNzEuMjU4LjE2My4zNi40ODguMjQxLjc3LS4xMTQuMjcyLS40MDMu"
    "NDI1LS42OTEuMzcxbC0uMDI4LS4wMDZDMy4zMTMgMTUuMTE2IDEgMTIuODYy"
    "IDEgMTBjMC0yLjIyLjk1Ny0zLjYxMSAyLjAzOS00Ljk0MUM0LjExOSAzLjcz"
    "IDUuMzA1IDIuNDczIDYuMTA0LjRsLjAxNC0uMDMzYy4wNzMtLjE2LjIxLS4y"
    "ODQuMzgtLjMzOC4xODEtLjA1Ny4zNzgtLjAzLjUzNi4wNzZsLjA3NC4wNWMu"
    "NzU2LjUzMyAxLjE0OCAxLjI2IDEuMzk0IDEuOTgzLjEyNi4zNy4yMTguNzUu"
    "Mjk5IDEuMTA3LjA4My4zNjguMTUyLjcwMy4yNCAxLjAzbC4wMDguMDI2Yy4w"
    "MjUuMDc0LjA5Ni4yNDUuMjM1LjM5OC4xNDIuMTU3LjM1Ni4zMDEuNzE2LjMw"
    "MS4zNCAwIC41MzctLjExMS42Ni0uMjIyLjEzNy0uMTIzLjIxNi0uMjc3LjI2"
    "LS4zODVsLjAxMi0uMDM2Yy4wMy0uMDk0LjA2My0uMjYzLjEwMS0uNDc4LjAx"
    "OC0uMS4wNC0uMjIxLjA2My0uMzEyLjAwOS0uMDM1LjAzLS4xMjMuMDc1LS4y"
    "MDguMDEzLS4wMjYuMDgyLS4xNjUuMjQyLS4yNjMuMDk3LS4wNi4yNC0uMTA5"
    "LjQwOC0uMDg4LjE0Mi4wMTcuMjQ4LjA3OC4zMTkuMTM1bC4wMjguMDIzLjA0"
    "OS4wNDZDMTIuNiAzLjU3NSAxNSA1Ljk5NiAxNSAxMCIvPjwvc3ZnPg=="
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


# Choose a model
_SVG_MODEL: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PHJlY3QgeD0iNCIgeT0iNCIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2"
    "IiByeD0iMiIvPjxyZWN0IHg9IjkiIHk9IjkiIHdpZHRoPSI2IiBoZWlnaHQ9IjYi"
    "Lz48bGluZSB4MT0iOSIgeTE9IjIiIHgyPSI5IiB5Mj0iNCIvPjxsaW5lIHgxPSIx"
    "NSIgeTE9IjIiIHgyPSIxNSIgeTI9IjQiLz48bGluZSB4MT0iOSIgeTE9IjIwIiB4"
    "Mj0iOSIgeTI9IjIyIi8+PGxpbmUgeDE9IjE1IiB5MT0iMjAiIHgyPSIxNSIgeTI9"
    "IjIyIi8+PGxpbmUgeDE9IjIiIHkxPSI5IiB4Mj0iNCIgeTI9IjkiLz48bGluZSB4"
    "MT0iMiIgeTE9IjE1IiB4Mj0iNCIgeTI9IjE1Ii8+PGxpbmUgeDE9IjIwIiB5MT0i"
    "OSIgeDI9IjIyIiB5Mj0iOSIvPjxsaW5lIHgxPSIyMCIgeTE9IjE1IiB4Mj0iMjIi"
    "IHkyPSIxNSIvPjwvc3ZnPg=="
)

# Terms of Service
_SVG_TERMS: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PHBhdGggZD0iTTE0IDJINmEyIDIgMCAwIDAtMiAydjE2YTIgMiAw"
    "IDAgMCAyIDJoMTJhMiAyIDAgMCAwIDItMlY4eiIvPjxwb2x5bGluZSBwb2ludHM9"
    "IjE0IDIgMTQgOCAyMCA4Ii8+PGxpbmUgeDE9IjkiIHkxPSIxMyIgeDI9IjE1IiB5"
    "Mj0iMTMiLz48bGluZSB4MT0iOSIgeTE9IjE3IiB4Mj0iMTUiIHkyPSIxNyIvPjwv"
    "c3ZnPg=="
)

# GitHub Octicon "comment-discussion" (filled, 16x16 viewBox).
# Additive fallback for a future discussion/conversation action.
_SVG_COMMENT_DISCUSSION: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGRhdGEtY29tcG9uZW50"
    "PSJPY3RpY29uIiBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgY2xhc3M9"
    "Im9jdGljb24gb2N0aWNvbi1jb21tZW50LWRpc2N1c3Npb24iIHZpZXdCb3g9IjAgMCAxNiAx"
    "NiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2IiBmaWxsPSJjdXJyZW50Q29sb3IiIGRpc3BsYXk9"
    "ImlubGluZS1ibG9jayIgb3ZlcmZsb3c9InZpc2libGUiIHN0eWxlPSJ2ZXJ0aWNhbC1hbGln"
    "bjp0ZXh0LWJvdHRvbSI+PHBhdGggZD0iTTEuNzUgMWg4LjVjLjk2NiAwIDEuNzUuNzg0IDEu"
    "NzUgMS43NXY1LjVBMS43NSAxLjc1IDAgMCAxIDEwLjI1IDEwSDcuMDYxbC0yLjU3NCAyLjU3"
    "M0ExLjQ1OCAxLjQ1OCAwIDAgMSAyIDExLjU0M1YxMGgtLjI1QTEuNzUgMS43NSAwIDAgMSAw"
    "IDguMjV2LTUuNUMwIDEuNzg0Ljc4NCAxIDEuNzUgMVpNMS41IDIuNzV2NS41YzAgLjEzOC4x"
    "MTIuMjUuMjUuMjVoMWEuNzUuNzUgMCAwIDEgLjc1Ljc1djIuMTlsMi43Mi0yLjcyYS43NDku"
    "NzQ5IDAgMCAxIC41My0uMjJoMy41YS4yNS4yNSAwIDAgMCAuMjUtLjI1di01LjVhLjI1LjI1"
    "IDAgMCAwLS4yNS0uMjVoLTguNWEuMjUuMjUgMCAwIDAtLjI1LjI1Wm0xMyAyYS4yNS4yNSAw"
    "IDAgMC0uMjUtLjI1aC0uNWEuNzUuNzUgMCAwIDEgMC0xLjVoLjVjLjk2NiAwIDEuNzUuNzg0"
    "IDEuNzUgMS43NXY1LjVBMS43NSAxLjc1IDAgMCAxIDE0LjI1IDEySDE0djEuNTQzYTEuNDU4"
    "IDEuNDU4IDAgMCAxLTIuNDg3IDEuMDNMOS4yMiAxMi4yOGEuNzQ5Ljc0OSAwIDAgMSAuMzI2"
    "LTEuMjc1Ljc0OS43NDkgMCAwIDEgLjczNC4yMTVsMi4yMiAyLjIydi0yLjE5YS43NS43NSAw"
    "IDAgMSAuNzUtLjc1aDFhLjI1LjI1IDAgMCAwIC4yNS0uMjVaIi8+PC9zdmc+Cg=="
)

# GitHub Octicon "upload" (filled, 16x16 viewBox).
# Additive fallback for a future file-upload action; no behaviour is wired here.
_SVG_UPLOAD: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGRhdGEtY29tcG9uZW50PSJP"
    "Y3RpY29uIiBhcmlhLWhpZGRlbj0idHJ1ZSIgZm9jdXNhYmxlPSJmYWxzZSIgY2xhc3M9Im9jdGlj"
    "b24gb2N0aWNvbi11cGxvYWQiIHZpZXdCb3g9IjAgMCAxNiAxNiIgd2lkdGg9IjE2IiBoZWlnaHQ9"
    "IjE2IiBmaWxsPSJjdXJyZW50Q29sb3IiIGRpc3BsYXk9ImlubGluZS1ibG9jayIgb3ZlcmZsb3c9"
    "InZpc2libGUiIHN0eWxlPSJ2ZXJ0aWNhbC1hbGlnbjp0ZXh0LWJvdHRvbSI+PHBhdGggZD0iTTIu"
    "NzUgMTRBMS43NSAxLjc1IDAgMCAxIDEgMTIuMjV2LTIuNWEuNzUuNzUgMCAwIDEgMS41IDB2Mi41"
    "YzAgLjEzOC4xMTIuMjUuMjUuMjVoMTAuNWEuMjUuMjUgMCAwIDAgLjI1LS4yNXYtMi41YS43NS43"
    "NSAwIDAgMSAxLjUgMHYyLjVBMS43NSAxLjc1IDAgMCAxIDEzLjI1IDE0WiIvPjxwYXRoIGQ9Ik0x"
    "MS43OCA0LjcyYS43NDkuNzQ5IDAgMSAxLTEuMDYgMS4wNkw4Ljc1IDMuODExVjkuNWEuNzUuNzUg"
    "MCAwIDEtMS41IDBWMy44MTFMNS4yOCA1Ljc4YS43NDkuNzQ5IDAgMSAxLTEuMDYtMS4wNmwzLjI1"
    "LTMuMjVhLjc0OS43NDkgMCAwIDEgMS4wNiAwbDMuMjUgMy4yNVoiLz48L3N2Zz4K"
)


# Printer: PDF browser-print mode
_SVG_PRINTER: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAx"
    "NiAxNiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2IiBmaWxsPSJjdXJyZW50Q29sb3IiIGFyaWEt"
    "aGlkZGVuPSJ0cnVlIiBmb2N1c2FibGU9ImZhbHNlIj4KICA8cGF0aCBkPSJNNCAyLjc1QzQg"
    "MS43ODQgNC43ODQgMSA1Ljc1IDFoNC41QzExLjIxNiAxIDEyIDEuNzg0IDEyIDIuNzVWNWgu"
    "MjVBMi43NSAyLjc1IDAgMCAxIDE1IDcuNzV2My41QTEuNzUgMS43NSAwIDAgMSAxMy4yNSAx"
    "M0gxMnYuMjVBMS43NSAxLjc1IDAgMCAxIDEwLjI1IDE1aC00LjVBMS43NSAxLjc1IDAgMCAx"
    "IDQgMTMuMjVWMTNIMi43NUExLjc1IDEuNzUgMCAwIDEgMSAxMS4yNXYtMy41QTIuNzUgMi43"
    "NSAwIDAgMSAzLjc1IDVINFYyLjc1Wm0xLjUgMFY1aDVWMi43NWEuMjUuMjUgMCAwIDAtLjI1"
    "LS4yNWgtNC41YS4yNS4yNSAwIDAgMC0uMjUuMjVaTTMuNzUgNi41QTEuMjUgMS4yNSAwIDAg"
    "MCAyLjUgNy43NXYzLjVjMCAuMTM4LjExMi4yNS4yNS4yNUg0di0uNzVDNCA5Ljc4NCA0Ljc4"
    "NCA5IDUuNzUgOWg0LjVjLjk2NiAwIDEuNzUuNzg0IDEuNzUgMS43NXYuNzVoMS4yNWEuMjUu"
    "MjUgMCAwIDAgLjI1LS4yNXYtMy41YTEuMjUgMS4yNSAwIDAgMC0xLjI1LTEuMjVoLTguNVpt"
    "MS43NSA0LjI1djIuNWMwIC4xMzguMTEyLjI1LjI1LjI1aDQuNWEuMjUuMjUgMCAwIDAgLjI1"
    "LS4yNXYtMi41YS4yNS4yNSAwIDAgMC0uMjUtLjI1aC00LjVhLjI1LjI1IDAgMCAwLS4yNS4y"
    "NVpNMTIgOGEuNzUuNzUgMCAxIDEgMS41IDBBLjc1Ljc1IDAgMCAxIDEyIDhaIi8+Cjwvc3Zn"
    "Pgo="
)

# Share this page
_SVG_SHARE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PGNpcmNsZSBjeD0iMTgiIGN5PSI1IiByPSIzIi8+PGNpcmNsZSBj"
    "eD0iNiIgY3k9IjEyIiByPSIzIi8+PGNpcmNsZSBjeD0iMTgiIGN5PSIxOSIgcj0i"
    "MyIvPjxsaW5lIHgxPSI4LjU5IiB5MT0iMTMuNTEiIHgyPSIxNS40MiIgeTI9IjE3"
    "LjQ5Ii8+PGxpbmUgeDE9IjE1LjQxIiB5MT0iNi41MSIgeDI9IjguNTkiIHkyPSIx"
    "MC40OSIvPjwvc3ZnPg=="
)

# Open menu
_SVG_MENU: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PGxpbmUgeDE9IjMiIHkxPSI2IiB4Mj0iMjEiIHkyPSI2Ii8+PGxp"
    "bmUgeDE9IjMiIHkxPSIxMiIgeDI9IjIxIiB5Mj0iMTIiLz48bGluZSB4MT0iMyIg"
    "eTE9IjE4IiB4Mj0iMjEiIHkyPSIxOCIvPjwvc3ZnPg=="
)

# Model information
_SVG_INFO: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9"
    "IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Ut"
    "d2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2lu"
    "PSJyb3VuZCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiLz48bGluZSB4"
    "MT0iMTIiIHkxPSIxNiIgeDI9IjEyIiB5Mj0iMTIiLz48bGluZSB4MT0iMTIiIHkx"
    "PSI4IiB4Mj0iMTIuMDEiIHkyPSI4Ii8+PC9zdmc+"
)

# ── Phase C additions (icon-sync task: JS ICONS ↔ __init__.py ↔ *.svg) ──────
# Each constant here is base64(open("<slug>.svg","rb").read()) using the
# `export_svg2base64()` helper's exact convention: currentColor swapped for
# the concrete #24292f used by every other on-disk icon (data-URIs render
# via <img src="...">, which can't inherit CSS `color`/currentColor).
# Mirrors ICONS.botAssistant / thumbUp / thumbDown / checkAns / syncRetry /
# syncRetryReverse / sparkle / sparkleAlt in ai-assistant.js — see the
# module docstring's "Adding a new icon" steps for the full workflow.

# Copilot-style assistant/robot glyph (Octicon copilot, MIT-licensed path data)
_SVG_BOT_ASSISTANT: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTcuOTk4IDE1LjAzNWMtNC41"
    "NjIgMC03Ljg3My0yLjkxNC03Ljk5OC0zLjc0OVY5LjMzOGMuMDg1LS42MjguNjc3LTEu"
    "Njg2IDEuNTg4LTIuMDY1LjAxMy0uMDcuMDI0LS4xNDMuMDM2LS4yMTguMDI5LS4xODMu"
    "MDYtLjM4NC4xMjYtLjYxMi0uMjAxLS41MDgtLjI1NC0xLjA4NC0uMjU0LTEuNjU2IDAt"
    "Ljg3LjEyOC0xLjc2OS42OTMtMi40ODQuNTc5LS43MzMgMS40OTQtMS4xMjQgMi43MjQt"
    "MS4yNjEgMS4yMDYtLjEzNCAyLjI2Mi4wMzQgMi45NDQuNzY1LjA1LjA1My4wOTYuMTA4"
    "LjEzOS4xNjUuMDQ0LS4wNTcuMDk0LS4xMTIuMTQzLS4xNjUuNjgyLS43MzEgMS43Mzgt"
    "Ljg5OSAyLjk0NC0uNzY1IDEuMjMuMTM3IDIuMTQ1LjUyOCAyLjcyNCAxLjI2MS41NjYu"
    "NzE1LjY5MyAxLjYxNC42OTMgMi40ODQgMCAuNTcyLS4wNTMgMS4xNDgtLjI1NCAxLjY1"
    "Ni4wNjYuMjI4LjA5OC40MjkuMTI2LjYxMi4wMTIuMDc2LjAyNC4xNDguMDM3LjIxOC45"
    "MjQuMzg1IDEuNTIyIDEuNDcxIDEuNTkxIDIuMDk1djEuODcyYzAgLjc2Ni0zLjM1MSAz"
    "Ljc5NS04LjAwMiAzLjc5NVptMC0xLjQ4NWMyLjI4IDAgNC41ODQtMS4xMSA1LjAwMi0x"
    "LjQzM1Y3Ljg2MmwtLjAyMy0uMTE2Yy0uNDkuMjEtMS4wNzUuMjkxLTEuNzI3LjI5MS0x"
    "LjE0NiAwLTIuMDU5LS4zMjctMi43MS0uOTkxQTMuMjIyIDMuMjIyIDAgMCAxIDggNi4z"
    "MDNhMy4yNCAzLjI0IDAgMCAxLS41NDQuNzQzYy0uNjUuNjY0LTEuNTYzLjk5MS0yLjcx"
    "Ljk5MS0uNjUyIDAtMS4yMzYtLjA4MS0xLjcyNy0uMjkxbC0uMDIzLjExNnY0LjI1NWMu"
    "NDE5LjMyMyAyLjcyMiAxLjQzMyA1LjAwMiAxLjQzM1pNNi43NjIgMi44M2MtLjE5My0u"
    "MjA2LS42MzctLjQxMy0xLjY4Mi0uMjk3LTEuMDE5LjExMy0xLjQ3OS40MDQtMS43MTMu"
    "Ny0uMjQ3LjMxMi0uMzY5Ljc4OS0uMzY5IDEuNTU0IDAgLjc5My4xMjkgMS4xNzEuMzA4"
    "IDEuMzcxLjE2Mi4xODEuNTE5LjM3OSAxLjQ0Mi4zNzkuODUzIDAgMS4zMzktLjIzNSAx"
    "LjYzOC0uNTQuMzE1LS4zMjIuNTI3LS44MjcuNjE3LTEuNTUzLjExNy0uOTM1LS4wMzct"
    "MS4zOTUtLjI0MS0xLjYxNFptNC4xNTUtLjI5N2MtMS4wNDQtLjExNi0xLjQ4OC4wOTEt"
    "MS42ODEuMjk3LS4yMDQuMjE5LS4zNTkuNjc5LS4yNDIgMS42MTQuMDkxLjcyNi4zMDMg"
    "MS4yMzEuNjE4IDEuNTUzLjI5OS4zMDUuNzg0LjU0IDEuNjM4LjU0LjkyMiAwIDEuMjgt"
    "LjE5OCAxLjQ0Mi0uMzc5LjE3OS0uMi4zMDgtLjU3OC4zMDgtMS4zNzEgMC0uNzY1LS4x"
    "MjMtMS4yNDItLjM3LTEuNTU0LS4yMzMtLjI5Ni0uNjkzLS41ODctMS43MTMtLjdaIi8+"
    "PHBhdGggZD0iTTYuMjUgOS4wMzdhLjc1Ljc1IDAgMCAxIC43NS43NXYxLjUwMWEuNzUu"
    "NzUgMCAwIDEtMS41IDBWOS43ODdhLjc1Ljc1IDAgMCAxIC43NS0uNzVabTQuMjUuNzV2"
    "MS41MDFhLjc1Ljc1IDAgMCAxLTEuNSAwVjkuNzg3YS43NS43NSAwIDAgMSAxLjUgMFoi"
    "Lz48L3N2Zz4="
)

# GitHub Octicon thumbsup / thumbsdown (MIT-licensed path data)
_SVG_THUMB_UP: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTguMzQ3LjYzMUEuNzUuNzUg"
    "MCAwIDEgOS4xMjMuMjZsLjIzOC4wNGEzLjI1IDMuMjUgMCAwIDEgMi41OTEgNC4wOThM"
    "MTEuNDk0IDZoLjY2NWEzLjI1IDMuMjUgMCAwIDEgMy4xMTggNC4xNjdsLTEuMTM1IDMu"
    "ODU5QTIuNzUxIDIuNzUxIDAgMCAxIDExLjUwMyAxNkg2LjU4NmEzLjc1IDMuNzUgMCAw"
    "IDEtMi4xODQtLjcwMkExLjc1IDEuNzUgMCAwIDEgMyAxNkgxLjc1QTEuNzUgMS43NSAw"
    "IDAgMSAwIDE0LjI1di02LjVDMCA2Ljc4NC43ODQgNiAxLjc1IDZoMy40MTdhLjI1LjI1"
    "IDAgMCAwIC4yMTctLjEyN1pNNC43NSAxMy42NDlsLjM5Ni4zM2MuNDA0LjMzNy45MTQu"
    "NTIxIDEuNDQuNTIxaDQuOTE3YTEuMjUgMS4yNSAwIDAgMCAxLjItLjg5N2wxLjEzNS0z"
    "Ljg1OUExLjc1IDEuNzUgMCAwIDAgMTIuMTU5IDcuNUgxMC41YS43NS43NSAwIDAgMS0u"
    "NzIxLS45NTZsLjczMS0yLjU1OGExLjc1IDEuNzUgMCAwIDAtMS4xMjctMi4xNEw2LjY5"
    "IDYuNjExYTEuNzUgMS43NSAwIDAgMS0xLjUyMy44ODlINC43NVpNMy4yNSA3LjVoLTEu"
    "NWEuMjUuMjUgMCAwIDAtLjI1LjI1djYuNWMwIC4xMzguMTEyLjI1LjI1LjI1SDNhLjI1"
    "LjI1IDAgMCAwIC4yNS0uMjVaIi8+PC9zdmc+"
)
_SVG_THUMB_DOWN: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTcuNjUzIDE1LjM2OWEuNzUu"
    "NzUgMCAwIDEtLjc3Ni4zNzFsLS4yMzgtLjA0YTMuMjUgMy4yNSAwIDAgMS0yLjU5MS00"
    "LjA5OUw0LjUwNiAxMGgtLjY2NUEzLjI1IDMuMjUgMCAwIDEgLjcyMyA1LjgzM2wxLjEz"
    "NS0zLjg1OUEyLjc1IDIuNzUgMCAwIDEgNC40ODIgMEg5LjQzYy43OC4wMDMgMS41Mzgu"
    "MjUgMi4xNjguNzAyQTEuNzUyIDEuNzUyIDAgMCAxIDEyLjk4OSAwaDEuMjcyQTEuNzUg"
    "MS43NSAwIDAgMSAxNiAxLjc1djYuNUExLjc1IDEuNzUgMCAwIDEgMTQuMjUgMTBoLTMu"
    "NDE3YS4yNS4yNSAwIDAgMC0uMjE3LjEyN1pNMTEuMjUgMi4zNTFsLS4zOTYtLjMzYTIu"
    "MjQ4IDIuMjQ4IDAgMCAwLTEuNDQtLjUyMUg0LjQ5NmExLjI1IDEuMjUgMCAwIDAtMS4x"
    "OTkuODk3TDIuMTYyIDYuMjU2QTEuNzUgMS43NSAwIDAgMCAzLjg0MSA4LjVINS41YS43"
    "NS43NSAwIDAgMSAuNzIxLjk1NmwtLjczMSAyLjU1OGExLjc1IDEuNzUgMCAwIDAgMS4x"
    "MjcgMi4xNEw5LjMxIDkuMzg5YTEuNzUgMS43NSAwIDAgMSAxLjUyMy0uODg5aC40MTda"
    "bTEuNSA2LjE0OWgxLjVhLjI1LjI1IDAgMCAwIC4yNS0uMjV2LTYuNWEuMjUuMjUgMCAw"
    "IDAtLjI1LS4yNUgxM2EuMjUuMjUgMCAwIDAtLjI1LjI1WiIvPjwvc3ZnPg=="
)

# Checkmark — swapped in for copy-answer for ~1.6s after a confirmed copy
_SVG_CHECK_ANSWER: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Utd2lkdGg9"
    "IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+"
    "PHBvbHlsaW5lIHBvaW50cz0iMjAgNiA5IDE3IDQgMTIiLz48L3N2Zz4="
)

# Two-arrow sync glyph (Octicon sync) — retry action
_SVG_SYNC_RETRY: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTEuNzA1IDguMDA1YS43NS43"
    "NSAwIDAgMSAuODM0LjY1NiA1LjUgNS41IDAgMCAwIDkuNTkyIDIuOTdsLTEuMjA0LTEu"
    "MjA0YS4yNS4yNSAwIDAgMSAuMTc3LS40MjdoMy42NDZhLjI1LjI1IDAgMCAxIC4yNS4y"
    "NXYzLjY0NmEuMjUuMjUgMCAwIDEtLjQyNy4xNzdsLTEuMzgtMS4zOEE3LjAwMiA3LjAw"
    "MiAwIDAgMSAxLjA1IDguODRhLjc1Ljc1IDAgMCAxIC42NTYtLjgzNFpNOCAyLjVhNS40"
    "ODcgNS40ODcgMCAwIDAtNC4xMzEgMS44NjlsMS4yMDQgMS4yMDRBLjI1LjI1IDAgMCAx"
    "IDQuODk2IDZIMS4yNUEuMjUuMjUgMCAwIDEgMSA1Ljc1VjIuMTA0YS4yNS4yNSAwIDAg"
    "MSAuNDI3LS4xNzdsMS4zOCAxLjM4QTcuMDAyIDcuMDAyIDAgMCAxIDE0Ljk1IDcuMTZh"
    "Ljc1Ljc1IDAgMCAxLTEuNDkuMTc4QTUuNSA1LjUgMCAwIDAgOCAyLjVaIi8+PC9zdmc+"
)
# Mirrored variant (redo / alternate-direction) — not wired to any control yet
_SVG_SYNC_RETRY_REVERSE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTYs"
    "MCkgc2NhbGUoLTEsMSkiPjxwYXRoIGQ9Ik0xLjcwNSA4LjAwNWEuNzUuNzUgMCAwIDEg"
    "LjgzNC42NTYgNS41IDUuNSAwIDAgMCA5LjU5MiAyLjk3bC0xLjIwNC0xLjIwNGEuMjUu"
    "MjUgMCAwIDEgLjE3Ny0uNDI3aDMuNjQ2YS4yNS4yNSAwIDAgMSAuMjUuMjV2My42NDZh"
    "LjI1LjI1IDAgMCAxLS40MjcuMTc3bC0xLjM4LTEuMzhBNy4wMDIgNy4wMDIgMCAwIDEg"
    "MS4wNSA4Ljg0YS43NS43NSAwIDAgMSAuNjU2LS44MzRaTTggMi41YTUuNDg3IDUuNDg3"
    "IDAgMCAwLTQuMTMxIDEuODY5bDEuMjA0IDEuMjA0QS4yNS4yNSAwIDAgMSA0Ljg5NiA2"
    "SDEuMjVBLjI1LjI1IDAgMCAxIDEgNS43NVYyLjEwNGEuMjUuMjUgMCAwIDEgLjQyNy0u"
    "MTc3bDEuMzggMS4zOEE3LjAwMiA3LjAwMiAwIDAgMSAxNC45NSA3LjE2YS43NS43NSAw"
    "IDAgMS0xLjQ5LjE3OEE1LjUgNS41IDAgMCAwIDggMi41WiIvPjwvZz48L3N2Zz4="
)

# Single counter-clockwise arrow (Lucide rotate-ccw) — the "retry" action.
# Byte-for-byte mirror of ai-assistant.js ICONS.retry so the same-named
# icon renders identically in JS-driven and server-rendered contexts.
# (_SVG_SYNC_RETRY above is the older Octicon two-arrow "sync" glyph,
# kept as a reserved variant; the retry meta now points here.)
_SVG_RETRY: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3Vycm"
    "VudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIg"
    "c3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBvbHlsaW5lIHBvaW50cz0iMSA0IDEgMT"
    "AgNyAxMCIvPjxwYXRoIGQ9Ik0zLjUxIDE1YTkgOSAwIDEgMCAuNDktNC41Ii8+PC9z"
    "dmc+"
)

# Octicon sparkle — four-pointed star with concave sides
_SVG_SPARKLE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTcuMTk4LjU3Yy4yNzUtLjc1"
    "MiAxLjM0LS43NTIgMS42MTUgMGwuODQ5IDIuMzE3YTUuODE5IDUuODE5IDAgMCAwIDMu"
    "NDYyIDMuNDYzbDIuMzE3Ljg0OGMuNzUzLjI3NS43NTMgMS4zNCAwIDEuNjE1bC0yLjMx"
    "Ny44NDlhNS44MTUgNS44MTUgMCAwIDAtMy40NjIgMy40NjJsLS44NDkgMi4zMTdjLS4y"
    "NzUuNzUzLTEuMzQuNzUzLTEuNjE1IDBsLS44NDgtMi4zMTdhNS44MTkgNS44MTkgMCAw"
    "IDAtMy40NjMtMy40NjJMLjU3IDguODEzYy0uNzUyLS4yNzUtLjc1Mi0xLjM0IDAtMS42"
    "MTVsMi4zMTctLjg0OEE1LjgyMyA1LjgyMyAwIDAgMCA2LjM1IDIuODg3TDcuMTk4LjU3"
    "Wm0uNTYyIDIuODMzQTcuMzIzIDcuMzIzIDAgMCAxIDMuNDAzIDcuNzZsLS42NzMuMjQ2"
    "LjY3My4yNDZhNy4zMjQgNy4zMjQgMCAwIDEgNC4zNTcgNC4zNTZsLjI0Ni42NzMuMjQ2"
    "LS42NzNhNy4zMjIgNy4zMjIgMCAwIDEgNC4zNTYtNC4zNTZsLjY3My0uMjQ2LS42NzMt"
    "LjI0NmE3LjMyNCA3LjMyNCAwIDAgMS00LjM1Ni00LjM1N2wtLjI0Ni0uNjczLS4yNDYu"
    "NjczWiIvPjwvc3ZnPg=="
)
# Two-tier "AI sparkle" — one large 4-point star, one small offset star
_SVG_SPARKLE_RIGHT_1_NOVA_TOP: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAyMCAyMCIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTkgMS41Yy4xOCAwIC4zNC4x"
    "Mi4zOS4yOWwuODIgMi43MmE0LjcgNC43IDAgMCAwIDMuMTUgMy4xNWwyLjcyLjgyYS40"
    "LjQgMCAwIDEgMCAuNzdsLTIuNzIuODJhNC43IDQuNyAwIDAgMC0zLjE1IDMuMTVsLS44"
    "MiAyLjcyYS40LjQgMCAwIDEtLjc3IDBsLS44Mi0yLjcyYTQuNyA0LjcgMCAwIDAtMy4x"
    "NS0zLjE1bC0yLjcyLS44MmEuNC40IDAgMCAxIDAtLjc3bDIuNzItLjgyQTQuNyA0Ljcg"
    "MCAwIDAgNy42IDQuNTFsLjgyLTIuNzJBLjQuNCAwIDAgMSA5IDEuNVoiLz48cGF0aCBk"
    "PSJNMTUuNSAxYy4xNiAwIC4zLjEuMzQuMjZsLjMyLjk5Yy4xNS40Ny41Mi44NC45OS45"
    "OWwuOTkuMzJhLjM2LjM2IDAgMCAxIDAgLjY4bC0uOTkuMzJhMS41NiAxLjU2IDAgMCAw"
    "LS45OS45OWwtLjMyLjk5YS4zNi4zNiAwIDAgMS0uNjggMGwtLjMyLS45OWExLjU2IDEu"
    "NTYgMCAwIDAtLjk5LS45OWwtLjk5LS4zMmEuMzYuMzYgMCAwIDEgMC0uNjhsLjk5LS4z"
    "MmMuNDctLjE1Ljg0LS41Mi45OS0uOTlsLjMyLS45OUEuMzYuMzYgMCAwIDEgMTUuNSAx"
    "WiIvPjwvc3ZnPg=="
)

# Paintbrush + AI sparkle — future "customize/style" affordance, not
# wired to any control yet. Mirrors ai-assistant.js ICONS.brushSparkle.
_SVG_BRUSH_SPARKLE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAyMCAyMCIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTE0Ljg3NzcgNy4yODMxMUwx"
    "NS4yMjYgOC4zNTM5QzE1LjMzNDMgOC42Nzk0NSAxNS41MTcxIDguOTc1MjcgMTUuNzU5"
    "OSA5LjIxNzg0QzE2LjAwMjYgOS40NjA0MiAxNi4yOTg3IDkuNjQzMDggMTYuNjI0NSA5"
    "Ljc1MTNMMTcuNjk2MSAxMC4wOTkzTDE3LjcxNzUgMTAuMTA0N0MxNy44MDAxIDEwLjEz"
    "MzggMTcuODcxNiAxMC4xODc4IDE3LjkyMjIgMTAuMjU5MkMxNy45NzI4IDEwLjMzMDcg"
    "MTggMTAuNDE2IDE4IDEwLjUwMzVDMTggMTAuNTkxMSAxNy45NzI4IDEwLjY3NjQgMTcu"
    "OTIyMiAxMC43NDc5QzE3Ljg3MTYgMTAuODE5MyAxNy44MDAxIDEwLjg3MzMgMTcuNzE3"
    "NSAxMC45MDI0TDE2LjY0NTkgMTEuMjUwNEMxNi4zMjAxIDExLjM1ODYgMTYuMDI0IDEx"
    "LjU0MTMgMTUuNzgxMyAxMS43ODM5QzE1LjUzODUgMTIuMDI2NSAxNS4zNTU3IDEyLjMy"
    "MjMgMTUuMjQ3NCAxMi42NDc4TDE0Ljg5OTEgMTMuNzE4NkMxNC44NyAxMy44MDExIDE0"
    "LjgxNiAxMy44NzI2IDE0Ljc0NDUgMTMuOTIzMkMxNC42NzMgMTMuOTczNyAxNC41ODc2"
    "IDE0LjAwMDkgMTQuNSAxNC4wMDA5QzE0LjQxMjQgMTQuMDAwOSAxNC4zMjcgMTMuOTcz"
    "NyAxNC4yNTU1IDEzLjkyMzJDMTQuMjQ4NCAxMy45MTgyIDE0LjI0MTUgMTMuOTEzIDE0"
    "LjIzNDggMTMuOTA3NkMxNC4xNzM2IDEzLjg1ODQgMTQuMTI3MSAxMy43OTMgMTQuMTAw"
    "OCAxMy43MTg2TDEzLjc1MjUgMTIuNjQ3OEMxMy43MzM4IDEyLjU5MSAxMy43MTI4IDEy"
    "LjUzNTEgMTMuNjg5NiAxMi40ODAyQzEzLjU3OTYgMTIuMjE5NiAxMy40MjAyIDExLjk4"
    "MiAxMy4yMTk2IDExLjc4MDhDMTMuMTgxNSAxMS43NDI2IDEzLjE0MiAxMS43MDU4IDEz"
    "LjEwMTQgMTEuNjcwNkMxMi44ODMgMTEuNDgxNSAxMi42MjkyIDExLjMzNjcgMTIuMzU0"
    "MSAxMS4yNDUxTDExLjI4MjQgMTAuODk3MUMxMS4xOTk4IDEwLjg2NzkgMTEuMTI4MyAx"
    "MC44MTM5IDExLjA3NzcgMTAuNzQyNUMxMS4wMjcxIDEwLjY3MTEgMTEgMTAuNTg1NyAx"
    "MSAxMC40OTgyQzExIDEwLjQxMDcgMTEuMDI3MSAxMC4zMjUzIDExLjA3NzcgMTAuMjUz"
    "OUMxMS4xMjgzIDEwLjE4MjQgMTEuMTk5OCAxMC4xMjg0IDExLjI4MjQgMTAuMDk5M0wx"
    "Mi4zNTQxIDkuNzUxM0MxMi42NzU5IDkuNjQwMjYgMTIuOTY3NiA5LjQ1NjM0IDEzLjIw"
    "NjUgOS4yMTM5MkMxMy40NDU0IDguOTcxNTEgMTMuNjI0OSA4LjY3NzE2IDEzLjczMTEg"
    "OC4zNTM5TDE0LjA3OTQgNy4yODMxMUMxNC4xMDg1IDcuMjAwNTcgMTQuMTYyNSA3LjEy"
    "OTExIDE0LjIzNCA3LjA3ODU1QzE0LjMwNTUgNy4wMjggMTQuMzkxIDcuMDAwODUgMTQu"
    "NDc4NSA3LjAwMDg1QzE0LjU2NjEgNy4wMDA4NSAxNC42NTE1IDcuMDI4IDE0LjcyMyA3"
    "LjA3ODU1QzE0Ljc5NDUgNy4xMjkxMSAxNC44NDg2IDcuMjAwNTcgMTQuODc3NyA3LjI4"
    "MzExWk0xOS43ODI5IDE1LjIxNEwxOS4wMTc1IDE0Ljk2NTVDMTguNzg0NyAxNC44ODgy"
    "IDE4LjU3MzMgMTQuNzU3NyAxOC4zOTk5IDE0LjU4NDRDMTguMjI2NSAxNC40MTEyIDE4"
    "LjA5NTkgMTQuMTk5OSAxOC4wMTg2IDEzLjk2NzNMMTcuNzY5OCAxMy4yMDI1QzE3Ljc0"
    "OSAxMy4xNDM1IDE3LjcxMDQgMTMuMDkyNSAxNy42NTkzIDEzLjA1NjRDMTcuNjA4MiAx"
    "My4wMjAzIDE3LjU0NzIgMTMuMDAwOSAxNy40ODQ3IDEzLjAwMDlDMTcuNDIyMSAxMy4w"
    "MDA5IDE3LjM2MTEgMTMuMDIwMyAxNy4zMSAxMy4wNTY0QzE3LjI1ODkgMTMuMDkyNSAx"
    "Ny4yMjAzIDEzLjE0MzUgMTcuMTk5NSAxMy4yMDI1TDE2Ljk1MDggMTMuOTY3M0MxNi44"
    "NzUgMTQuMTk4MiAxNi43NDY3IDE0LjQwODUgMTYuNTc2MSAxNC41ODE2QzE2LjQwNTUg"
    "MTQuNzU0OCAxNi4xOTcxIDE0Ljg4NjIgMTUuOTY3MiAxNC45NjU1TDE1LjIwMTcgMTUu"
    "MjE0QzE1LjE0MjcgMTUuMjM0OCAxNS4wOTE2IDE1LjI3MzQgMTUuMDU1NSAxNS4zMjQ0"
    "QzE1LjAxOTQgMTUuMzc1NSAxNSAxNS40MzY0IDE1IDE1LjQ5OUMxNSAxNS41NjE1IDE1"
    "LjAxOTQgMTUuNjIyNCAxNS4wNTU1IDE1LjY3MzVDMTUuMDkxNiAxNS43MjQ1IDE1LjE0"
    "MjcgMTUuNzYzMSAxNS4yMDE3IDE1Ljc4MzlMMTUuOTY3MiAxNi4wMzI0QzE2LjIwMDMg"
    "MTYuMTEwMSAxNi40MTIgMTYuMjQxMiAxNi41ODU1IDE2LjQxNTFDMTYuNzU4OSAxNi41"
    "ODkxIDE2Ljg4OTIgMTYuODAxMiAxNi45NjYxIDE3LjAzNDRMMTcuMjE0OCAxNy43OTkz"
    "QzE3LjIzNTcgMTcuODU4MiAxNy4yNzQzIDE3LjkwOTMgMTcuMzI1MyAxNy45NDU0QzE3"
    "LjM3NjQgMTcuOTgxNSAxNy40Mzc0IDE4LjAwMDkgMTcuNSAxOC4wMDA5QzE3LjU2MjUg"
    "MTguMDAwOSAxNy42MjM1IDE3Ljk4MTUgMTcuNjc0NiAxNy45NDU0QzE3LjcyNTcgMTcu"
    "OTA5MyAxNy43NjQzIDE3Ljg1ODIgMTcuNzg1MSAxNy43OTkzTDE4LjAzMzkgMTcuMDM0"
    "NEMxOC4xMTEyIDE2LjgwMTkgMTguMjQxOCAxNi41OTA2IDE4LjQxNTIgMTYuNDE3M0Mx"
    "OC41ODg2IDE2LjI0NCAxOC44MDAxIDE2LjExMzYgMTkuMDMyOCAxNi4wMzYzTDE5Ljc5"
    "ODIgMTUuNzg3N0MxOS44NTcyIDE1Ljc2NjkgMTkuOTA4MyAxNS43MjgzIDE5Ljk0NDQg"
    "MTUuNjc3M0MxOS45ODA2IDE1LjYyNjMgMjAgMTUuNTY1MyAyMCAxNS41MDI4QzIwIDE1"
    "LjQ0MDMgMTkuOTgwNiAxNS4zNzkzIDE5Ljk0NDQgMTUuMzI4M0MxOS45MDgzIDE1LjI3"
    "NzIgMTkuODU3MiAxNS4yMzg3IDE5Ljc5ODIgMTUuMjE3OUwxOS43ODI5IDE1LjIxNFpN"
    "NC45OTk5NyAyLjVDNC45OTk5NyAyLjIyMzg2IDUuMjIzODMgMiA1LjQ5OTk3IDJIMTQu"
    "NUMxNC43NzYxIDIgMTUgMi4yMjM4NiAxNSAyLjVWNi4wOTk3OUMxNC44MzQ4IDYuMDM0"
    "NzcgMTQuNjU4IDYuMDAwODUgMTQuNDc4NSA2LjAwMDg1QzE0LjMxNDUgNi4wMDA4NSAx"
    "NC4xNTI3IDYuMDI5MTkgMTQgNi4wODM3MVYzSDEzVjUuNTAyMTlDMTMgNS43NzgzNCAx"
    "Mi43NzYxIDYuMDAyMTkgMTIuNSA2LjAwMjE5QzEyLjIyMzggNi4wMDIxOSAxMiA1Ljc3"
    "ODM0IDEyIDUuNTAyMTlWM0gxMVY0LjVDMTEgNC43NzYxNCAxMC43NzYxIDUgMTAuNSA1"
    "QzEwLjIyMzggNSA5Ljk5OTk3IDQuNzc2MTQgOS45OTk5NyA0LjVWM0g1Ljk5OTk3Vjku"
    "MDA0MzZIMTEuNDE2NUwxMC45NjE3IDkuMTUyMDVMMTAuOTUgOS4xNTYxOUMxMC42NzI0"
    "IDkuMjU0MDUgMTAuNDMxOCA5LjQzNTYgMTAuMjYxNiA5LjY3NTk0QzEwLjE4OTQgOS43"
    "Nzc5OCAxMC4xMzEzIDkuODg4NDcgMTAuMDg4NCAxMC4wMDQ0SDUuOTk5OTlWMTEuMDA0"
    "M0M1Ljk5OTk5IDExLjU1NjYgNi40NDc3IDEyLjAwNDMgNi45OTk5OSAxMi4wMDQzSDgu"
    "NTA0MDhDOC43ODAyMyAxMi4wMDQzIDkuMDA0MDggMTIuMjI4MiA5LjAwNDA4IDEyLjUw"
    "NDNWMTYuMDAxN0M5LjAwNDA4IDE2LjU1NCA5LjQ1MTggMTcuMDAxNyAxMC4wMDQxIDE3"
    "LjAwMTdDMTAuNTU2NCAxNy4wMDE3IDExLjAwNDEgMTYuNTUzOSAxMS4wMDQxIDE2LjAw"
    "MTdWMTIuNTA0M0MxMS4wMDQxIDEyLjI0MjkgMTEuMjA0NyAxMi4wMjg0IDExLjQ2MDMg"
    "MTIuMDA2MkwxMi4wNDEgMTIuMTk0OEMxMi4xOTA0IDEyLjI0NSAxMi4zMjgyIDEyLjMy"
    "MzkgMTIuNDQ2OCAxMi40MjY2QzEyLjQ2OSAxMi40NDU4IDEyLjQ5MDYgMTIuNDY2IDEy"
    "LjUxMTUgMTIuNDg2OUMxMi42MjExIDEyLjU5NjkgMTIuNzA4MiAxMi43MjY3IDEyLjc2"
    "ODQgMTIuODY5MUMxMi43ODA4IDEyLjg5ODcgMTIuNzkyMSAxMi45Mjg3IDEyLjgwMjIg"
    "MTIuOTU5MkwxMi44MDI3IDEyLjk2MDhMMTIuODE2OSAxMy4wMDQzSDEyLjAwNDFWMTYu"
    "MDAxN0MxMi4wMDQxIDE3LjEwNjIgMTEuMTA4NyAxOC4wMDE3IDEwLjAwNDEgMTguMDAx"
    "N0M4Ljg5OTUxIDE4LjAwMTcgOC4wMDQwOCAxNy4xMDYyIDguMDA0MDggMTYuMDAxN1Yx"
    "My4wMDQzSDYuOTk5OTlDNS44OTU0MiAxMy4wMDQzIDQuOTk5OTkgMTIuMTA4OSA0Ljk5"
    "OTk5IDExLjAwNDNMNC45OTk5NyAyLjVaIi8+PC9zdmc+"
)

# Hamburger + AI sparkle — future "AI menu" trigger, not wired to any
# control yet. Mirrors ai-assistant.js ICONS.menuSparkle.
_SVG_MENU_SPARKLE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTExLjIxOC4zNzJjLS4xMjIt"
    "LjQ4Ny0uODE0LS40ODctLjkzNiAwbC0uMjg1IDEuMTRhNC4xMDIgNC4xMDIgMCAwIDEt"
    "Mi45ODUgMi45ODVsLTEuMTQuMjg1Yy0uNDg3LjEyMi0uNDg3LjgxNCAwIC45MzZsMS4x"
    "NC4yODVhNC4xMDIgNC4xMDIgMCAwIDEgMi45ODUgMi45ODVsLjI4NSAxLjE0Yy4xMjIu"
    "NDg3LjgxNC40ODcuOTM2IDBsLjI4NS0xLjE0YTQuMTAyIDQuMTAyIDAgMCAxIDIuOTg1"
    "LTIuOTg1bDEuMTQtLjI4NWMuNDg3LS4xMjIuNDg3LS44MTQgMC0uOTM2bC0xLjE0LS4y"
    "ODVhNC4xMDIgNC4xMDIgMCAwIDEtMi45ODUtMi45ODVsLS4yODUtMS4xNFpNMy41NjEg"
    "NC41Yy4yODIgMCAuNDc4LjI4OC40NTQuNTY5LS4wMS4xMi0uMDEuMjQyIDAgLjM2Mi4w"
    "MjQuMjgtLjE3Mi41NjktLjQ1NC41NjlILjc1YS43NS43NSAwIDAgMSAwLTEuNWgyLjgx"
    "MVptNC43NzggNC4zMTdjLS4xLS4yMDctLjMyLS4zMTctLjU1LS4zMTdILjc1YS43NS43"
    "NSAwIDAgMCAwIDEuNWg3LjQ0MmMuMjYgMCAuNDU1LS4yNDYuMzg4LS40OTdhMy44ODYg"
    "My44ODYgMCAwIDAtLjI0MS0uNjg2Wk0xMSAxMy4yNWEuNzUuNzUgMCAwIDEtLjc1Ljc1"
    "SC43NWEuNzUuNzUgMCAwIDEgMC0xLjVoOS41YS43NS43NSAwIDAgMSAuNzUuNzVaIi8+"
    "PC9zdmc+"
)

# AI "sparkles" cluster (Lucide sparkles) — one large 4-point star + two small
# twinkles. A DISTINCT glyph from _SVG_SPARKLE_RIGHT_1_NOVA_TOP (the filled two-tier star),
# in the same stroke convention as _SVG_CELEBRATION so the "I'm Feeling Lucky"
# trigger + shuffle read as a matching pair. Mirrors ai-assistant.js
# ICONS.menuSparkleRight byte-for-byte (base64 of the exact inline SVG).
_SVG_SPARKLE_RIGHT_2_NOVA_UP_DOWN: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3Vycm"
    "VudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIg"
    "c3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIG"
    "Q9Ik05LjkzNyAxNS41QTIgMiAwIDAgMCA4LjUgMTQuMDYzbC02LjEzNS0xLjU4MmEu"
    "NS41IDAgMCAxIDAtLjk2Mkw4LjUgOS45MzZBMiAyIDAgMCAwIDkuOTM3IDguNWwxLj"
    "U4Mi02LjEzNWEuNS41IDAgMCAxIC45NjMgMEwxNC4wNjMgOC41QTIgMiAwIDAgMCAx"
    "NS41IDkuOTM3bDYuMTM1IDEuNTgxYS41LjUgMCAwIDEgMCAuOTY0TDE1LjUgMTQuMD"
    "YzYTIgMiAwIDAgMC0xLjQzNyAxLjQzN2wtMS41ODIgNi4xMzVhLjUuNSAwIDAgMS0u"
    "OTYzIDB6Ii8+PHBhdGggZD0iTTIwIDN2NCIvPjxwYXRoIGQ9Ik0yMiA1aC00Ii8+PH"
    "BhdGggZD0iTTQgMTd2MiIvPjxwYXRoIGQ9Ik01IDE4SDMiLz48L3N2Zz4="
)

# Security alert shield (Octicon shield) — future "security notice" or
# "verified/protected" indicator, not wired to any control yet.
# Mirrors ai-assistant.js ICONS.shieldAlert.
_SVG_SHIELD_ALERT: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTcuNDY3LjEzM2ExLjc0OCAx"
    "Ljc0OCAwIDAgMSAxLjA2NiAwbDUuMjUgMS42OEExLjc1IDEuNzUgMCAwIDEgMTUgMy40"
    "OFY3YzAgMS41NjYtLjMyIDMuMTgyLTEuMzAzIDQuNjgyLS45ODMgMS40OTgtMi41ODUg"
    "Mi44MTMtNS4wMzIgMy44NTVhMS42OTcgMS42OTcgMCAwIDEtMS4zMyAwYy0yLjQ0Ny0x"
    "LjA0Mi00LjA0OS0yLjM1Ny01LjAzMi0zLjg1NUMxLjMyIDEwLjE4MiAxIDguNTY2IDEg"
    "N1YzLjQ4YTEuNzUgMS43NSAwIDAgMSAxLjIxNy0xLjY2N1ptLjYxIDEuNDI5YS4yNS4y"
    "NSAwIDAgMC0uMTUzIDBsLTUuMjUgMS42OGEuMjUuMjUgMCAwIDAtLjE3NC4yMzhWN2Mw"
    "IDEuMzU4LjI3NSAyLjY2NiAxLjA1NyAzLjg2Ljc4NCAxLjE5NCAyLjEyMSAyLjM0IDQu"
    "MzY2IDMuMjk3YS4xOTYuMTk2IDAgMCAwIC4xNTQgMGMyLjI0NS0uOTU2IDMuNTgyLTIu"
    "MTA0IDQuMzY2LTMuMjk4QzEzLjIyNSA5LjY2NiAxMy41IDguMzYgMTMuNSA3VjMuNDhh"
    "LjI1MS4yNTEgMCAwIDAtLjE3NC0uMjM3bC01LjI1LTEuNjhaTTguNzUgNC43NXYzYS43"
    "NS43NSAwIDAgMS0xLjUgMHYtM2EuNzUuNzUgMCAwIDEgMS41IDBaTTkgMTAuNWExIDEg"
    "MCAxIDEtMiAwIDEgMSAwIDAgMSAyIDBaIi8+PC9zdmc+"
)

# Search + AI sparkle — primary icon for .ai-assistant-searchbar
# (see ai-assistant.js _buildSearchBar). Mirrors ICONS.searchSparkle;
# _SVG_SEARCH_AI above remains a defined fallback, not removed.
_SVG_SEARCH_SPARKLE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMjQyOTJmIiBzdHJva2Utd2lkdGg9"
    "IjEuMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5k"
    "Ij48cGF0aCBkPSJNMTYuMjQgMTEuMzggQTYuMyA2LjMgMCAxIDEgMTMuMTUgNS4wNCIv"
    "PjxwYXRoIGQ9Ik0xMy42NSAxNi42MCBMMTkuNDUgMjIuNDAiIHN0cm9rZS13aWR0aD0i"
    "Mi4zNSIvPjxwYXRoIGQ9Ik0xNi41IDUuMDIgUTE2Ljc4IDYuMjIgMTcuMTggNi41MiBR"
    "MTcuNTIgNi43OCAxOC43NSA3LjA1IFExNy41MiA3LjMyIDE3LjE4IDcuNTggUTE2Ljc4"
    "IDcuODggMTYuNSA5LjA4IFExNi4yMiA3Ljg4IDE1LjgyIDcuNTggUTE1LjQ4IDcuMzIg"
    "MTQuMjUgNy4wNSBRMTUuNDggNi43OCAxNS44MiA2LjUyIFExNi4yMiA2LjIyIDE2LjUg"
    "NS4wMiBaIiBzdHJva2Utd2lkdGg9IjEiLz48L3N2Zz4="
)

# Firefox "Nova" sparkle, restructured (dropped the browser-chrome-only
# @media -moz-pref() toggle and fill="context-fill" — see the long
# comment on ICONS.sparkleNova in ai-assistant.js for why). Primary
# icon for .ai-assistant-panel-logo; sparkle/sparkle-alt remain as
# defined alternates, not removed.
_SVG_SPARKLE_LEFT_2_NOVA_UP_DOWN: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAg"
    "MCAxNiAxNiIgZmlsbD0iIzI0MjkyZiI+PHBhdGggZD0iTTIuNjIyIDExLjI1MmEuNDA5"
    "LjQwOSAwIDAgMSAuNzU2IDBxLjA0LjA5Ni4wNzguMTk3LjEyMi4zMjYuMTk1LjYxNGMu"
    "MDM4LjE0My4xNTQuMjUuMjk3LjI4NmE2IDYgMCAwIDEgLjc5OC4yNy40MS40MSAwIDAg"
    "MSAuMDAyLjc2IDYgNiAwIDAgMS0uOC4yNy40LjQgMCAwIDAtLjI5Ni4yODQgNiA2IDAg"
    "MCAxLS4yMTMuNjZsLS4wNjIuMTU2YS40MDkuNDA5IDAgMCAxLS43NTQgMGwtLjA2My0u"
    "MTU2YTYgNiAwIDAgMS0uMjEzLS42Ni40LjQgMCAwIDAtLjI5NC0uMjg0IDYgNiAwIDAg"
    "MS0uODAxLS4yNy40MS40MSAwIDAgMSAuMDAyLS43NiA2IDYgMCAwIDEgLjc5OC0uMjcu"
    "NC40IDAgMCAwIC4yOTYtLjI4NiA2IDYgMCAwIDEgLjI3NC0uODExIi8+PHBhdGggZmls"
    "bC1ydWxlPSJldmVub2RkIiBkPSJNOC4zMTQgMy4xM2MuNjI3LTEuNDk4IDIuNzM5LTEu"
    "NDk4IDMuMzY2IDBxLjExLjI2NC4yMTMuNTRoLjAwMmMuMjI0LjU5Ni40IDEuMTYyLjUz"
    "OCAxLjY4OC4wMjIuMDg1LjA5OC4xNzYuMjM4LjIxM2wuMzQ2LjA5NGExNiAxNiAwIDAg"
    "MSAxLjg0Ni42NDVjMS40OTkuNjI3IDEuNTI0IDIuNzM2LjAxIDMuMzcxcS0uMzMuMTQt"
    "LjY4My4yNjlhMTcgMTcgMCAwIDEtMS41Mi40NzljLS4xMzcuMDM1LS4yMTMuMTI0LS4y"
    "MzUuMjA4YTE3IDE3IDAgMCAxLS41ODQgMS44MDhsLS4wMDEuMDAxYTE2IDE2IDAgMCAx"
    "LS4xNzIuNDI1Yy0uNjEgMS40NjEtMi42NDUgMS40OTItMy4zMTYuMDk0YTEgMSAwIDAg"
    "MS0uMDQ2LS4wOTEgMTcgMTcgMCAwIDEtLjc1Ni0yLjIzNnYtLjAwMmMtLjAyMi0uMDgt"
    "LjA5NS0uMTcxLS4yMzUtLjIwOGExNi40IDE2LjQgMCAwIDEtMi4yLS43NDRoLS4wMDFj"
    "LTEuNTE2LS42MzUtMS40OTMtMi43NDYuMDA2LTMuMzc0cS4zNjMtLjE1Ljc1LS4yOTJh"
    "MTYgMTYgMCAwIDEgMS40NDMtLjQ0N2guMDAyYy4xMzgtLjAzNy4yMTQtLjEyNi4yMzct"
    "LjIxM0ExNyAxNyAwIDAgMSA4LjEgMy42NzFWMy42N3EuMTA0LS4yNzYuMjE0LS41NG0x"
    "Ljk4Mi41NzlhLjMyMi4zMjIgMCAwIDAtLjU5OCAwbC0uMTk1LjQ5MmExNiAxNiAwIDAg"
    "MC0uNDkgMS41MzV2LjAwMUExLjgxIDEuODEgMCAwIDEgNy43IDcuMDIxbC4wMDEuMDAx"
    "YTE1IDE1IDAgMCAwLTEuOTkzLjY3M2gtLjAwMWEuMzI0LjMyNCAwIDAgMC0uMDA0LjYw"
    "NXEuMzAzLjEyNS42MTcuMjQzbC4zNi4xMjdxLjUzNS4xOCAxLjAyMy4zMDZsLjAwMi4w"
    "MDFjLjYwNi4xNiAxLjEzNC42MiAxLjMwNyAxLjI4Mi4xMzMuNTA5LjMwNiAxLjA2Mi41"
    "MzEgMS42NDRsLjE0NC4zNi4wMTYuMDNhLjMyLjMyIDAgMCAwIC41OTIgMHYtLjAwMXEu"
    "MDc3LS4xODYuMTU2LS4zODljLjIyNi0uNTg0LjM5OS0xLjEzNy41MzItMS42NDZhMS44"
    "MSAxLjgxIDAgMCAxIDEuMzA2LTEuMjhsLjMzMi0uMDlhMTUgMTUgMCAwIDAgMS42Njct"
    "LjU4NmwuMDAzLS4wMDFhLjMyMy4zMjMgMCAwIDAtLjAwNS0uNjA2cS0uMzMtLjEzOC0u"
    "NjgxLS4yNjRsLS4wMDMtLjAwMWExNSAxNSAwIDAgMC0xLjMwOS0uNDA3IDEuODIgMS44"
    "MiAwIDAgMS0xLjMxMi0xLjI4NCAxNSAxNSAwIDAgMC0uMzQ0LTEuMTM2bC0uMTQ2LS40"
    "MDFhMTUgMTUgMCAwIDAtLjE5NS0uNDkyIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiLz48cGF0"
    "aCBkPSJNMi42MjIgMS4yNTJhLjQwOS40MDkgMCAwIDEgLjc1NiAwcS4wNC4wOTYuMDc4"
    "LjE5Ny4xMjIuMzI3LjE5NS42MTRjLjAzOC4xNDMuMTU0LjI1LjI5Ny4yODZhNiA2IDAg"
    "MCAxIC43OTguMjcuNDEuNDEgMCAwIDEgLjAwMi43NiA2IDYgMCAwIDEtLjguMjcuNC40"
    "IDAgMCAwLS4yOTYuMjg0IDYgNiAwIDAgMS0uMjEzLjY2bC0uMDYyLjE1NmEuNDA5LjQw"
    "OSAwIDAgMS0uNzU0IDBsLS4wNjMtLjE1NmE2IDYgMCAwIDEtLjIxMy0uNjYuNC40IDAg"
    "MCAwLS4yOTQtLjI4NCA2IDYgMCAwIDEtLjgwMS0uMjcuNDEuNDEgMCAwIDEgLjAwMi0u"
    "NzZxLjEzLS4wNTUuMjcyLS4xMDYuMjc2LS4wOTkuNTI2LS4xNjRhLjQuNCAwIDAgMCAu"
    "Mjk2LS4yODYgNiA2IDAgMCAxIC4yNzQtLjgxMSIvPjwvc3ZnPg=="
)
# Lucide "minimize-2" (inward-pointing diagonal arrows) — the visual inverse
# of ICONS.maximize's outward-pointing arrows in ai-assistant.js. Used as the
# icon for the dedicated "collapse full screen" panel-header button, distinct
# from _SVG_MINIMIZE above (which collapses the panel to the floating trigger
# pill) and from the pre-existing corner-bracket "restore" glyph (kept
# unchanged for backward compatibility — see ICONS.restore in ai-assistant.js).
_SVG_MINIMIZE_COLLAPSE: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxOCIg"
    "aGVpZ2h0PSIxOCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9"
    "ImN1cnJlbnRDb2xvciIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91"
    "bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLW1p"
    "bmltaXplLTIiPjxwb2x5bGluZSBwb2ludHM9IjQgMTQgMTAgMTQgMTAgMjAiPjwvcG9s"
    "eWxpbmU+PHBvbHlsaW5lIHBvaW50cz0iMjAgMTAgMTQgMTAgMTQgNCI+PC9wb2x5bGlu"
    "ZT48bGluZSB4MT0iMTQiIHgyPSIyMSIgeTE9IjEwIiB5Mj0iMyI+PC9saW5lPjxsaW5l"
    "IHgxPSIzIiB4Mj0iMTAiIHkxPSIyMSIgeTI9IjE0Ij48L2xpbmU+PC9zdmc+Cg=="
)

# Party popper + confetti — mirrors ICONS.celebration in ai-assistant.js
# byte-for-byte (same hand-authored path/circle data, re-encoded here rather
# than redrawn, per the JS/Python icon-parity convention this module
# otherwise follows for every other _ICON_META entry). Used by the "Surprise
# me" Lucky-picker source — see _LUCKY_SOURCES in ai-assistant.js.
_SVG_CELEBRATION: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3VycmVudENv"
    "bG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxp"
    "bmVqb2luPSJyb3VuZCIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGQ9Ik0zLjUgMjAuNSA5"
    "IDhsNyA0LjUtMTIuNSA4WiIvPjxwYXRoIGQ9Ik05IDhsMS4zLTMuNSIvPjxwYXRoIGQ9Ik0x"
    "NCAzLjUgMTUgNS41Ii8+PHBhdGggZD0iTTE5IDYuNWwyIDEiLz48Y2lyY2xlIGN4PSIyMCIg"
    "Y3k9IjExLjUiIHI9IjEiIGZpbGw9ImN1cnJlbnRDb2xvciIgc3Ryb2tlPSJub25lIi8+PGNp"
    "cmNsZSBjeD0iMTYiIGN5PSI0IiByPSIxIiBmaWxsPSJjdXJyZW50Q29sb3IiIHN0cm9rZT0i"
    "bm9uZSIvPjxjaXJjbGUgY3g9IjIxLjUiIGN5PSIxNiIgcj0iMSIgZmlsbD0iY3VycmVudENv"
    "bG9yIiBzdHJva2U9Im5vbmUiLz48L3N2Zz4="
)

# MUI "ErrorIcon" outline, re-authored with stroke="currentColor" (dropped the
# original fill/MuiSvgIcon-color* class attributes so it themes the same way
# every other icon in this module does). Not wired to any control yet —
# reserved for a future error/alert affordance (e.g. a failed-request state
# in the panel body). Mirrors the "reserved for future use" pattern already
# established by _SVG_SPARKLE / _SVG_SPARKLE_RIGHT_1_NOVA_TOP above.
_SVG_ERROR_ALERT: str = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iY3Vycm"
    "VudENvbG9yIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIg"
    "c3Ryb2tlLWxpbmVqb2luPSJyb3VuZCI+PHBhdGggZD0iTTEyIDh2NE0xMiAxNmguMD"
    "FNMyA3Ljk0djguMTJjMCAuMzQgMCAuNTEuMDUuNjYuMDUuMTQuMTIuMjYuMjIuMzcu"
    "MS4xMS4yNS4yLjU1LjM2bDcuNCA0LjExYy4yOC4xNi40My4yNC41OC4yNy4xMy4wMy"
    "4yNy4wMy40IDAgLjE1LS4wMy4zLS4xMS41OC0uMjdsNy40LTQuMTFjLjMtLjE2LjQ1"
    "LS4yNS41NS0uMzYuMS0uMTEuMTctLjIzLjIyLS4zNy4wNS0uMTUuMDUtLjMyLjA1LS"
    "42NlY3Ljk0YzAtLjM0IDAtLjUxLS4wNS0uNjYtLjA1LS4xNC0uMTItLjI3LS4yMi0u"
    "MzctLjEtLjEyLS4yNS0uMi0uNTUtLjM3bC03LjQtNC4xMWMtLjI4LS4xNi0uNDMtLj"
    "I0LS41OC0uMjdhMSAxIDAgMCAwLS40IDBjLS4xNS4wMy0uMy4xMS0uNTguMjdsLTcu"
    "NCA0LjExYy0uMy4xNy0uNDUuMjUtLjU1LjM3LS4xLjEtLjE3LjIzLS4yMi4zNy0uMD"
    "UuMTUtLjA1LjMyLS4wNS42NloiLz48L3N2Zz4="
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
    "new-chat-compose": {"icon": _SVG_NEW_CHAT_COMPOSE, "desc": "Start a new chat"},
    "export-txt": {"icon": _SVG_EXPORT_TXT, "desc": "Export conversation as txt"},
    "copy-answer": {"icon": _SVG_COPY_ANSWER, "desc": "Copy this answer"},
    "privacy": {"icon": _SVG_PRIVACY, "desc": "Privacy Policy"},
    "search-ai": {"icon": _SVG_SEARCH_AI, "desc": "AI search"},
    "search-sparkle": {
        "icon": _SVG_SEARCH_SPARKLE,
        "desc": "AI-powered search (primary searchbar icon)",
    },
    "keyboard": {"icon": _SVG_KEYBOARD, "desc": "Keyboard shortcuts"},
    # ── Phase B additions ─────────────────────────────────────────────
    "model": {"icon": _SVG_MODEL, "desc": "Choose a model"},
    "terms": {"icon": _SVG_TERMS, "desc": "Terms of Service"},
    "share": {"icon": _SVG_SHARE, "desc": "Share this page"},
    "comment-discussion": {"icon": _SVG_COMMENT_DISCUSSION, "desc": "Discussion"},
    "upload": {"icon": _SVG_UPLOAD, "desc": "Upload a file"},
    "printer": {"icon": _SVG_PRINTER, "desc": "Print or save as PDF"},
    "menu": {"icon": _SVG_MENU, "desc": "Open menu"},
    "info": {"icon": _SVG_INFO, "desc": "Model information"},
    # ── Phase C additions — mirrors ai-assistant.js ICONS.{key} ────────
    "bot-assistant": {"icon": _SVG_BOT_ASSISTANT, "desc": "AI assistant"},
    "thumb-up": {"icon": _SVG_THUMB_UP, "desc": "Helpful"},
    "thumb-down": {"icon": _SVG_THUMB_DOWN, "desc": "Not helpful"},
    "check-answer": {"icon": _SVG_CHECK_ANSWER, "desc": "Copied!"},
    "retry": {"icon": _SVG_RETRY, "desc": "Retry — re-send the same question"},
    # Not wired to any control yet — see ICONS.syncRetryReverse in JS.
    "retry-reverse": {"icon": _SVG_SYNC_RETRY_REVERSE, "desc": "Redo (reverse retry)"},
    "sparkle": {"icon": _SVG_SPARKLE, "desc": "AI sparkle"},
    "sparkle-alt": {
        "icon": _SVG_SPARKLE_RIGHT_1_NOVA_TOP,
        "desc": "AI sparkle (alternate)",
    },
    "sparkle-nova": {
        "icon": _SVG_SPARKLE_LEFT_2_NOVA_UP_DOWN,
        "desc": "AI sparkle, Nova style (primary panel logo)",
    },
    "brush-sparkle": {
        "icon": _SVG_BRUSH_SPARKLE,
        "desc": "AI-assisted style/customize",
    },
    "menu-sparkle": {"icon": _SVG_MENU_SPARKLE, "desc": "AI menu"},
    "sparkle-right-2-nova-up-down": {
        "icon": _SVG_SPARKLE_RIGHT_2_NOVA_UP_DOWN,
        "desc": "AI suggestion sparkle",
    },
    "shield-alert": {"icon": _SVG_SHIELD_ALERT, "desc": "Security notice"},
    # Mirrors ICONS.celebration in ai-assistant.js — the "Surprise me"
    # source icon in the "I'm Feeling Lucky" picker.
    "celebration": {
        "icon": _SVG_CELEBRATION,
        "desc": "Surprise me (I\u2019m Feeling Lucky)",
    },
    # Deliberately reuse the existing "terms" / "privacy" constants above
    # instead of introducing new path data — see ai-assistant.js
    # ICONS.termsOfService / ICONS.privacyResponsibility for the rationale
    # (consolidated to one design per concept across JS/Python/disk-file).
    "terms-of-service": {"icon": _SVG_TERMS, "desc": "Terms of Service"},
    "privacy-responsibility": {
        "icon": _SVG_PRIVACY,
        "desc": "Privacy & Responsibility",
    },
    # ── Phase D additions ─────────────────────────────────────────────
    # Mirrors ICONS.minimizeCollapse in ai-assistant.js — the dedicated
    # "collapse full screen" button that is the visual inverse of the
    # existing "maximize" action.
    "minimize-collapse": {
        "icon": _SVG_MINIMIZE_COLLAPSE,
        "desc": "Collapse full screen (exit maximized view)",
    },
    # Reserved for a future error/alert affordance — not wired to any
    # control yet. Mirrors ICONS.errorAlert in ai-assistant.js.
    "error-alert": {"icon": _SVG_ERROR_ALERT, "desc": "Error / alert notice"},
}
