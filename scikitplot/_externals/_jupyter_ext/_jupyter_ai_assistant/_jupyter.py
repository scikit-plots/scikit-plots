# scikitplot/_externals/_jupyter_ext/_jupyter_ai_assistant/_jupyter.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Jupyter AI Assistant Widget
============================

Self-contained Jupyter notebook integration for the AI-assistant
extension.  This submodule is **Sphinx-free** and **IPython-optional** —
it can be imported anywhere without requiring Sphinx, BeautifulSoup, or
markdownify.  IPython is imported lazily inside :func:`display_jupyter_ai_button`
only when the display function is actually called.

This module is an internal submodule of
:mod:`scikitplot._externals._sphinx_ext._jupyter_ai_assistant`.
Its public symbols are re-exported from the parent ``__init__.py`` so
callers import from the parent package as usual.

Public API
----------
display_jupyter_ai_button : callable
    Inject an AI-assistant expandable dropdown widget into the current
    Jupyter notebook output cell.  The widget is visually **identical** to
    the Sphinx extension split-button: a primary ``Copy page`` button that
    copies the cell content to the clipboard, an arrow toggle, and a dropdown
    with ``Copy page``, ``View as Markdown``, AI provider rows, and optionally
    MCP tool rows.
display_jupyter_notebook_ai_button : callable
    Convenience wrapper around :func:`display_jupyter_ai_button` with
    ``notebook_mode=True`` — captures the entire notebook for the AI.

Internal API
------------
_JUPYTER_CONTENT_SELECTORS : tuple of str
    CSS selectors used by the widget JS to locate cell output text in
    JupyterLab >= 4, JupyterLab 3, classic Notebook, and VS Code Jupyter.
_build_jupyter_widget_html : callable
    Build the self-contained HTML+CSS+JS dropdown widget string.

Design notes
------------
* **Zero external dependencies at import time.**  All optional imports
  (IPython, hashlib) are deferred to call time.
* **XSS prevention.**  Every user-supplied string is serialised via
  :func:`._safe_json_for_script` before embedding in ``<script>`` blocks.
* **Sphinx-identical UX.**  The widget renders a split-button with:

  - A primary ``Copy page`` button (copies cell or notebook content as text).
  - An arrow toggle (``v``) that opens/closes a dropdown.
  - ``Copy page`` and ``View as Markdown`` as first two dropdown items.
  - A separator row.
  - One menu row per AI provider: SVG icon, bold title, muted description.
  - A separator row before MCP tool rows (when present).
  - ``aria-expanded`` / ``role="menu"`` / ``role="menuitem"`` for
    accessibility.
  - Fixed-position dropdown via ``getBoundingClientRect()`` — escapes VS
    Code Jupyter cell overflow clipping.
  - Click-outside-to-close behaviour.

* **Markdown-first AI workflow.**  When ``page_url`` is provided the JS
  derives the ``.md`` URL (replacing ``.html`` suffix) and embeds it in
  the prompt template — identical to the Sphinx widget's ``getMarkdownUrl``
  logic.

* **Default changes (v0.4.0):**

  - ``include_outputs`` defaults to ``False`` (was ``True``).  Cell outputs
    are opt-in — pass ``include_outputs=True`` explicitly when needed.
  - ``include_raw_image`` defaults to ``False`` (unchanged).

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

Notes
-----
**Developer note**: This module is imported by the parent ``__init__.py``
via ``from ._jupyter import (...)``.  The parent re-exports all public
symbols so existing call sites continue to work without modification.

Examples
--------
After a matplotlib plot in a Jupyter cell:

.. code-block:: python

    from scikitplot._externals._sphinx_ext._jupyter_ai_assistant import (
        display_jupyter_ai_button,
    )
    import matplotlib.pyplot as plt

    plt.plot([1, 2, 3])
    plt.show()
    display_jupyter_ai_button(
        content="A line chart showing values 1, 2, 3.",
        providers=["claude", "chatgpt", "gemini"],
        intention="Explain the trend",
    )

At the end of a notebook for a full review:

.. code-block:: python

    from scikitplot._externals._sphinx_ext._jupyter_ai_assistant import (
        display_jupyter_notebook_ai_button,
    )

    display_jupyter_notebook_ai_button(
        intention="Review this notebook for bugs and suggest improvements",
        providers=["claude", "chatgpt"],
        include_outputs=True,
    )
"""  # noqa: D205, D400

from __future__ import annotations

import hashlib
import time as _time
from typing import (  # noqa: F401
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Shared utilities imported from parent package (no circular dependency:
# parent defines these before importing this submodule).
# ---------------------------------------------------------------------------
from . import (
    _DEFAULT_PROVIDERS,
    _coerce_to_list,
    _filter_providers,
    _safe_json_for_script,
    _validate_base_url,
    _validate_mcp_tool,
)

__all__ = [
    "_JUPYTER_CONTENT_SELECTORS",
    "_build_jupyter_widget_html",
    "display_jupyter_ai_button",
    "display_jupyter_notebook_ai_button",
]

# ---------------------------------------------------------------------------
# CSS selectors — Jupyter DOM targets
# ---------------------------------------------------------------------------

#: CSS selectors tried by the widget JS to locate cell output content.
#: Ordered from most-specific to most-general.  JupyterLab >= 4,
#: JupyterLab 3, classic Notebook, VS Code Jupyter, and a generic
#: fallback are all covered.
_JUPYTER_CONTENT_SELECTORS: tuple[str, ...] = (
    # JupyterLab >= 4
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

# ---------------------------------------------------------------------------
# Inline SVG icons — base64 data URIs
# ---------------------------------------------------------------------------
# Each constant holds a minimal monochrome SVG encoded as a base64 data URI
# for use as CSS ``background-image``.  Keeping them inline means the widget
# is fully self-contained with zero network requests.

_SVG_COPY = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAx"
    "NiAxNiIgZmlsbD0ibm9uZSI+PHJlY3QgeD0iNS41IiB5PSIyLjUiIHdpZHRoPSI4IiBoZWln"
    "aHQ9IjkiIHJ4PSIxLjUiIHN0cm9rZT0iIzI0MjkyZiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48"
    "cmVjdCB4PSIyLjUiIHk9IjUuNSIgd2lkdGg9IjgiIGhlaWdodD0iOSIgcng9IjEuNSIgZmls"
    "bD0id2hpdGUiIHN0cm9rZT0iIzI0MjkyZiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48L3N2Zz4="
)
_SVG_MARKDOWN = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAx"
    "NiAxNiI+PHJlY3QgeD0iMC41IiB5PSIyLjUiIHdpZHRoPSIxNSIgaGVpZ2h0PSIxMSIgcng9"
    "IjIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzI0MjkyZiIgc3Ryb2tlLXdpZHRoPSIxLjUiLz48"
    "cGF0aCBmaWxsPSIjMjQyOTJmIiBkPSJNMyAxMVY1LjVoMS41TDYgNy41IDcuNSA1LjVIOVYx"
    "MUg3LjVWNy44TDYgOS44IDQuNSA3LjhWMTFIM3ptNy41IDBMOC41IDloMS41VjUuNWgxLjVW"
    "OUgxM0wxMC41IDExeiIvPjwvc3ZnPg=="
)
_SVG_CLAUDE = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzc0NUI0RiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEy"
    "czQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDEg"
    "MC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPjwvc3Zn"
    "Pg=="
)
_SVG_CHATGPT = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzEwYTM3ZiIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEy"
    "czQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTEgMTVoLTJ2LTZo"
    "MnY2em0wLThoLTJWN2gydjJ6Ii8+PC9zdmc+"
)
_SVG_GEMINI = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzQyODVGNCIgZD0iTTEyIDJsLTEgOUgzbDcuNSA1LjUtMi41"
    "IDguNUwxMiAxOWw0IDYtMi41LTguNUwyMSAxMWgtOHoiLz48L3N2Zz4="
)
_SVG_OLLAMA = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9r"
    "ZT0iIzMzMyIgc3Ryb2tlLXdpZHRoPSIyIi8+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0i"
    "NCIgZmlsbD0iIzMzMyIvPjwvc3ZnPg=="
)
_SVG_DEFAULT = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAy"
    "NCAyNCI+PHBhdGggZmlsbD0iIzg4OCIgZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDgg"
    "MTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTEgMTVoLTJ2LTZoMnY2em0w"
    "LThoLTJWN2gydjJ6Ii8+PC9zdmc+"
)

#: Map provider name to icon data URI and description text.
_PROVIDER_META: dict[str, dict[str, str]] = {
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
}


# ---------------------------------------------------------------------------
# Core widget builder
# ---------------------------------------------------------------------------


def _build_jupyter_widget_html(
    content: str | None = None,
    *,
    providers: str | list[str] | None = None,
    provider_configs: dict[str, Any] | None = None,
    position: str = "inline",
    page_url: str = "",
    widget_id: str | None = None,
    intention: str | None = None,
    custom_context: str | None = None,
    custom_prompt_prefix: str | None = None,
    notebook_mode: bool = False,
    include_outputs: bool = False,
    include_raw_image: bool = False,
    mcp_tools: dict[str, Any] | None = None,
) -> str:
    """Build a self-contained HTML+CSS+JS expandable AI dropdown widget.

    The widget is visually **identical** to the Sphinx extension split-button.
    Primary button: ``Copy page`` (copies cell/notebook content to clipboard).
    Arrow toggle opens a dropdown with ``Copy page``, ``View as Markdown``,
    AI provider rows, and optional MCP tool rows.

    Parameters
    ----------
    content : str or None, optional
        Explicit text content to include in the AI prompt.  When ``None``
        and *notebook_mode* is ``False``, the widget JS captures text from
        the surrounding Jupyter output area automatically.  When ``None``
        and *notebook_mode* is ``True``, all visible notebook cells are
        captured.
    providers : str or list of str or None, optional
        Ordered list of provider names to show.  A bare ``str`` is treated
        as a single-element list.  Defaults to
        ``["claude", "chatgpt", "gemini", "ollama"]``.
    provider_configs : dict or None, optional
        Full provider config overrides merged over
        :data:`_DEFAULT_PROVIDERS`.
    position : str, optional
        ``"inline"`` (default) or ``"floating"``.
    page_url : str, optional
        URL embedded in provider prompt templates as ``{url}``.  When set,
        the widget derives a ``.md`` URL (replacing ``.html`` suffix) for
        the ``View as Markdown`` item and for AI provider prompts — the
        same logic the Sphinx widget uses via ``getMarkdownUrl()``.
    widget_id : str or None, optional
        Unique DOM ID.  Auto-generated when ``None``.
    intention : str or None, optional
        User's stated goal.  Prepended to the prompt as
        ``"Goal: <intention>"``.
    custom_context : str or None, optional
        Additional background context injected after *intention* and
        before the provider's prompt template.
    custom_prompt_prefix : str or None, optional
        Raw text prepended before everything else in the final prompt.
    notebook_mode : bool, optional
        When ``True``, the JS walks the entire notebook DOM collecting all
        cell inputs and (optionally) their outputs into a single prompt.
    include_outputs : bool, optional
        When *notebook_mode* is ``True``, controls whether cell outputs
        are included alongside cell inputs.  Defaults to ``False``.
    include_raw_image : bool, optional
        When ``True``, the widget JS scans ``<img>`` and ``<canvas>``
        elements in the captured output area and appends image metadata
        (and canvas thumbnails) to the prompt.  Defaults to ``False``.

        .. note::
            Full-resolution base64 images exceed most URL length limits.
            Canvas thumbnails are compressed to max 300 x 300 px.

    mcp_tools : dict or None, optional
        MCP tool configurations to surface as ``Connect`` buttons in the
        dropdown.  Only tools with ``enabled: True`` are rendered.  Each
        value must match :data:`._DEFAULT_MCP_TOOLS` schema.  When
        ``None``, no MCP entries are rendered.

    Returns
    -------
    str
        Self-contained HTML string (no external resources required).

    Raises
    ------
    None
        All validation errors are handled gracefully; invalid providers or
        MCP tools are silently skipped.

    Notes
    -----
    **Security**: All user-supplied strings are serialised via
    :func:`._safe_json_for_script` before embedding in ``<script>``
    blocks — prevents XSS.

    **Dropdown UX (Sphinx-identical)**:

    * Split-button: ``Copy page`` primary label + ``v`` arrow toggle.
    * Dropdown: ``Copy page`` row, ``View as Markdown`` row, separator,
      one AI provider row per provider (icon, bold title, muted description),
      separator (if MCP tools present and enabled), MCP tool rows.
    * ``aria-expanded`` / ``role="menu"`` / ``role="menuitem"`` for
      screen-reader accessibility.
    * Fixed-position dropdown via ``getBoundingClientRect()`` — prevents
      VS Code Jupyter cell overflow clipping.
    * Click-outside-to-close.

    **Markdown-first AI workflow**: When ``page_url`` is provided the JS
    derives the ``.md`` URL (``page.html`` -> ``page.md``, or ``/path/``
    -> ``/path/index.md``) and uses it in the AI prompt template — the
    same strategy the Sphinx extension uses.  When no ``page_url`` is
    provided, the captured cell text is used directly.

    Examples
    --------
    >>> html = _build_jupyter_widget_html(
    ...     widget_id="demo",
    ...     providers=["claude", "chatgpt"],
    ...     intention="Explain this chart",
    ... )
    >>> assert "demo" in html
    >>> assert "Copy page" in html
    >>> assert "Ask Claude" in html
    >>> assert "View as Markdown" in html
    """
    # ── Generate stable widget ID ────────────────────────────────────────────
    if widget_id is None:
        raw_id = f"{_time.monotonic_ns()}_{content!r}_{notebook_mode}"
        widget_id = (
            "ai-btn-" + hashlib.md5(raw_id.encode()).hexdigest()[:8]  # noqa: S324
        )

    # ── Resolve provider list ────────────────────────────────────────────────
    provider_list: list[str] = _coerce_to_list(
        providers, default=["claude", "chatgpt", "gemini", "ollama"]
    )

    # ── Merge default configs with any user overrides ────────────────────────
    merged_configs: dict[str, Any] = {}
    for pname in provider_list:
        base = dict(_DEFAULT_PROVIDERS.get(pname, {}))
        if provider_configs and pname in provider_configs:
            base.update(provider_configs[pname])
        merged_configs[pname] = base

    # ── Security: filter providers with unsafe url_template ──────────────────
    safe_configs = _filter_providers(merged_configs)

    # ── Build button specs (preserves requested order) ───────────────────────
    buttons: list[dict[str, Any]] = []
    for pname in provider_list:
        cfg = safe_configs.get(pname, {})
        if not cfg:
            continue
        meta = _PROVIDER_META.get(pname, {"icon": _SVG_DEFAULT, "desc": ""})
        buttons.append(
            {
                "name": pname,
                "label": str(cfg.get("label", pname)),
                "url_template": str(cfg.get("url_template", "")),
                "prompt_template": str(cfg.get("prompt_template", "Ask about: {url}")),
                "type": str(cfg.get("type", "web")),
                "enabled": bool(cfg.get("enabled", True)),
                "icon": meta["icon"],
                "desc": meta.get("desc", ""),
            }
        )

    # ── Validate page URL ────────────────────────────────────────────────────
    validated_page_url = ""
    if page_url:
        try:
            validated_page_url = _validate_base_url(page_url)
        except ValueError:
            validated_page_url = ""

    # ── Validate and serialise MCP tools (only enabled entries) ──────────────
    safe_mcp: dict[str, Any] = {}
    if mcp_tools:
        for tname, tcfg in mcp_tools.items():
            errs = _validate_mcp_tool(tcfg, name=tname)
            if not errs and tcfg.get("enabled", False):
                safe_mcp[tname] = tcfg

    # ── Serialise everything through _safe_json_for_script (XSS guard) ───────
    buttons_json = _safe_json_for_script(buttons)
    content_json = _safe_json_for_script(content)
    selectors_json = _safe_json_for_script(list(_JUPYTER_CONTENT_SELECTORS))
    page_url_json = _safe_json_for_script(validated_page_url)
    intention_json = _safe_json_for_script(intention)
    custom_context_json = _safe_json_for_script(custom_context)
    prompt_prefix_json = _safe_json_for_script(custom_prompt_prefix)
    notebook_mode_json = "true" if notebook_mode else "false"
    include_outputs_json = "true" if include_outputs else "false"
    include_raw_image_json = "true" if include_raw_image else "false"
    mcp_tools_json = _safe_json_for_script(safe_mcp)

    # ── Widget container positioning ─────────────────────────────────────────
    position_style = (
        "position:fixed;bottom:16px;right:16px;z-index:9999;"
        if position == "floating"
        else "margin:8px 0 24px;display:inline-block;"
    )

    # ── Build self-contained HTML ─────────────────────────────────────────────
    # Notes on f-string brace escaping:
    #   {{  -> literal { in the output  (f-string escape)
    #   }}  -> literal } in the output  (f-string escape)
    #   The JS template placeholders {url}, {content}, {prompt} inside
    #   provider configs are escaped as {{url}}, {{content}}, {{prompt}}
    #   here — the f-string expansion converts them back to {url} etc.
    #   in the output JS, where they are replaced at runtime by the widget.
    html = f"""
<div id="{widget_id}" style="{position_style}font-family:sans-serif;">
<style>
/* AI Dropdown Widget - Sphinx-identical split-button UX */
#{widget_id} {{
  display:inline-block;
  position:relative;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}}
#{widget_id} .ai-split {{
  display:inline-flex;
  align-items:stretch;
  height:35px;
  border:1px solid rgba(0,0,0,.2);
  border-radius:4px;
  background:#fff;
  box-shadow:0 1px 3px rgba(0,0,0,.08);
  overflow:hidden;
}}
#{widget_id} .ai-main {{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:0 12px;
  font-size:13.5px;
  font-weight:500;
  color:#1a1a1a;
  background:none;
  border:none;
  cursor:pointer;
  white-space:nowrap;
  user-select:none;
  transition:background .15s;
}}
#{widget_id} .ai-main:hover {{ background:rgba(0,0,0,.05); }}
#{widget_id} .ai-main.ai-copied {{ background:rgba(76,175,80,.1); }}
#{widget_id} .ai-div {{
  width:1px;
  background:rgba(0,0,0,.2);
  align-self:stretch;
}}
#{widget_id} .ai-tog {{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:28px;
  background:none;
  border:none;
  cursor:pointer;
  color:#57606a;
  font-size:11px;
  transition:background .12s;
  padding:0;
}}
#{widget_id} .ai-tog:hover {{ background:rgba(0,0,0,.05); }}
#{widget_id} .ai-icon {{
  width:16px;
  height:16px;
  flex-shrink:0;
  background-size:contain;
  background-repeat:no-repeat;
  background-position:center;
  display:inline-block;
  border-radius:2px;
}}
</style>

<div class="ai-split" role="group" aria-label="Copy page">
  <button class="ai-main" id="{widget_id}-mainbtn"
          aria-label="Copy page as Markdown">Copy page</button>
  <span class="ai-div"></span>
  <button class="ai-tog" id="{widget_id}-togbtn"
          aria-haspopup="menu" aria-expanded="false"
          aria-controls="{widget_id}-menu"
          title="Show more options">&#9662;</button>
</div>

<script>
(function() {{
  var WID              = "{widget_id}";
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
  var SVG_COPY         = "{_SVG_COPY}";
  var SVG_MARKDOWN     = "{_SVG_MARKDOWN}";

  /* ── Derive .md URL from PAGE_URL (Sphinx-identical getMarkdownUrl) ── */
  function getMdUrl() {{
    var url = PAGE_URL || window.location.href;
    var u   = url.split('#')[0];
    if (u.endsWith('.html')) return u.slice(0, -5) + '.md';
    if (u.endsWith('/'))    return u + 'index.md';
    return u + '.md';
  }}

  /* ── Content capture ── */
  function getCellContent() {{
    var root = document.getElementById(WID);
    var area = root ? root.closest(
      ".jp-OutputArea,.output_area,.cell-output,.output"
    ) : null;
    if (area) {{
      for (var i = 0; i < SELECTORS.length; i++) {{
        var el = area.querySelector(SELECTORS[i]);
        if (el) return el.innerText || el.textContent || "";
      }}
      return area.innerText || area.textContent || "";
    }}
    return "";
  }}

  function getNotebookContent() {{
    var cells = document.querySelectorAll(
      ".jp-Cell,.cell,.code_cell,.text_cell,.markdown_cell"
    );
    if (!cells.length) return getCellContent();
    var parts = [], idx = 0;
    cells.forEach(function(cell) {{
      idx++;
      var inputEl = cell.querySelector(
        ".jp-InputArea-editor,.input_area,.CodeMirror-code,.cm-content"
      );
      var inputText = inputEl
        ? (inputEl.innerText || inputEl.textContent || "").trim() : "";
      if (!inputText) return;
      parts.push("--- Cell " + idx + " ---");
      parts.push(inputText);
      if (INCLUDE_OUTPUTS) {{
        var outEl = cell.querySelector(".jp-OutputArea,.output_area,.output");
        var outText = outEl
          ? (outEl.innerText || outEl.textContent || "").trim() : "";
        if (outText) {{ parts.push("Output:"); parts.push(outText); }}
      }}
    }});
    return parts.join("\\n");
  }}

  function getContent() {{
    if (EXPLICIT_CONTENT !== null && EXPLICIT_CONTENT !== undefined)
      return String(EXPLICIT_CONTENT);
    return NOTEBOOK_MODE ? getNotebookContent() : getCellContent();
  }}

  /* ── Canvas / image capture (include_raw_image mode) ── */
  function captureImages(area) {{
    if (!INCLUDE_RAW_IMAGE || !area) return "";
    var parts = [];
    area.querySelectorAll("canvas").forEach(function(cv) {{
      try {{
        var w = cv.width, h = cv.height;
        if (!w || !h) return;
        var s = Math.min(1, 300 / Math.max(w, h));
        var tw = Math.round(w*s), th = Math.round(h*s);
        var tc = document.createElement("canvas");
        tc.width = tw; tc.height = th;
        var ctx = tc.getContext("2d");
        if (ctx) {{
          ctx.drawImage(cv, 0, 0, tw, th);
          parts.push("[Canvas "+w+"x"+h+"px | thumbnail:"+tc.toDataURL("image/png",0.6)+"]");
        }}
      }} catch(e) {{ parts.push("[Canvas: cross-origin, cannot read]"); }}
    }});
    area.querySelectorAll("img").forEach(function(img) {{
      var src = img.src || "", alt = img.alt || "image";
      var w = img.naturalWidth||img.width||0, h = img.naturalHeight||img.height||0;
      if (src.startsWith("data:"))
        parts.push("[Image (base64, "+w+"x"+h+"): "+alt+" | "+src+"]");
      else if (src)
        parts.push("[Image: "+alt+" ("+w+"x"+h+"px) src="+src+"]");
    }});
    return parts.length ? "\\n\\n[Visual outputs]\\n"+parts.join("\\n") : "";
  }}

  /* ── Prompt builder ── */
  function buildPrompt(btn, content) {{
    var parts = [];
    if (PROMPT_PREFIX)  parts.push(String(PROMPT_PREFIX));
    if (INTENTION)      parts.push("Goal: " + String(INTENTION));
    if (CUSTOM_CONTEXT) parts.push("Context: " + String(CUSTOM_CONTEXT));
    var root2 = document.getElementById(WID);
    var area2 = root2 ? root2.closest(
      ".jp-OutputArea,.output_area,.cell-output,.output"
    ) : document.querySelector(".jp-OutputArea");
    var imgText = captureImages(area2);
    if (imgText) content = content + imgText;
    /* Use .md URL when PAGE_URL is set (Sphinx-identical workflow) */
    var contextUrl = PAGE_URL ? getMdUrl() : window.location.href;
    var main = btn.prompt_template
      .split("{{url}}").join(contextUrl)
      .split("{{content}}").join(content);
    parts.push(main);
    return parts.join("\\n\\n");
  }}

  function buildAiUrl(btn) {{
    var content = getContent();
    var prompt  = buildPrompt(btn, content);
    return btn.url_template.split("{{prompt}}").join(encodeURIComponent(prompt));
  }}

  /* ── Copy to clipboard ── */
  function handleCopy() {{
    var text = getContent();
    var mainBtn = document.getElementById(WID + "-mainbtn");
    function onSuccess() {{
      if (!mainBtn) return;
      var orig = mainBtn.textContent;
      mainBtn.textContent = "Copied!";
      mainBtn.classList.add("ai-copied");
      setTimeout(function() {{
        mainBtn.textContent = orig;
        mainBtn.classList.remove("ai-copied");
      }}, 2000);
    }}
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(text).then(onSuccess).catch(function() {{
        fallbackCopy(text, onSuccess);
      }});
    }} else {{
      fallbackCopy(text, onSuccess);
    }}
    closeMenu();
  }}

  function fallbackCopy(text, cb) {{
    var ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select();
    try {{ document.execCommand("copy"); cb(); }} catch(e) {{}}
    document.body.removeChild(ta);
  }}

  /* ═══════════════════════════════════════════════════════════════════════
     BODY-PORTAL MENU — definitive VS Code Jupyter z-index fix
     ───────────────────────────────────────────────────────────────────────
     Root cause: VS Code Jupyter renders subsequent notebook cells INTO the
     same document AFTER our widget output.  Even z-index:2147483647 is
     defeated when a later DOM sibling establishes its own stacking context
     (via transform, isolation, etc.).

     Solution: two-part —
       1. Create <ul> as direct document.body child (portal pattern) so it
          lives completely outside every cell's stacking context.
       2. Call document.body.appendChild(menu) on EVERY open.  appendChild
          on an already-attached node MOVES it to the last DOM position,
          guaranteeing it is the last element painted — on top of all
          cells rendered after our widget.
     ═══════════════════════════════════════════════════════════════════════ */

  /* ── Inject menu <style> into <head> (once per widget) ── */
  (function() {{
    if (document.getElementById("ai-ms-" + WID)) return;
    var s = document.createElement("style");
    s.id = "ai-ms-" + WID;
    s.textContent = [
      "#ai-m-" + WID + "{{",
        "position:fixed;",
        "width:220px;",
        "background:#fff;",
        "border:1px solid rgba(0,0,0,.2);",
        "border-radius:4px;",
        "box-shadow:0 10px 20px rgba(0,0,0,.15);",
        "z-index:2147483647;",        /* max CSS z-index */
        "padding:4px;",
        "list-style:none;",
        "margin:0;",
        "display:none;",
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;",
        "pointer-events:auto;",
      "}}",
      "#ai-m-" + WID + " .ai-mi{{",
        "display:block;width:100%;min-height:55px;",
        "text-align:left;background:none;border:none;",
        "border-radius:4px;color:#1a1a1a;cursor:pointer;",
        "transition:background .15s;padding:0;box-sizing:border-box;",
      "}}",
      "#ai-m-" + WID + " .ai-mi:hover{{background:rgba(0,0,0,.05);}}",
      "#ai-m-" + WID + " .ai-dis{{opacity:.45;cursor:not-allowed;}}",
      "#ai-m-" + WID + " .ai-cnt{{",
        "display:flex;flex-direction:column;gap:2px;padding:8px 12px;",
      "}}",
      "#ai-m-" + WID + " .ai-ti{{",
        "display:flex;align-items:center;gap:7px;",
        "font-weight:500;font-size:13.5px;color:#1a1a1a;",
      "}}",
      "#ai-m-" + WID + " .ai-de{{",
        "font-size:11px;color:#57606a;line-height:1.3;margin-top:-2px;",
      "}}",
      "#ai-m-" + WID + " .ai-ico{{",
        "width:16px;height:16px;flex-shrink:0;border-radius:2px;",
        "background-size:contain;background-repeat:no-repeat;",
        "background-position:center;display:inline-block;",
      "}}",
      "#ai-m-" + WID + " .ai-sp{{",
        "height:1px;background:rgba(0,0,0,.12);margin:4px 0;",
      "}}",
    ].join("");
    (document.head || document.documentElement).appendChild(s);
  }})();

  /* ── Create menu <ul> as body child (portal) ── */
  var menu = (function() {{
    var existing = document.getElementById("ai-m-" + WID);
    if (existing) return existing;
    var ul = document.createElement("ul");
    ul.id = "ai-m-" + WID;
    ul.setAttribute("role", "menu");
    ul.setAttribute("aria-labelledby", WID + "-mainbtn");
    document.body.appendChild(ul);
    return ul;
  }})();

  /* ── Menu open/close ── */
  function toggleMenu() {{
    var togBtn = document.getElementById(WID + "-togbtn");
    if (!togBtn) return;

    if (menu.style.display === "block") {{
      closeMenu();
      return;
    }}

    /* CRITICAL: re-append to body on every open.
       appendChild on an attached node MOVES it to last DOM position.
       VS Code Jupyter adds subsequent cell DOM nodes after our menu;
       re-appending here ensures our menu is always painted last = on top. */
    document.body.appendChild(menu);

    /* Compute fixed coordinates from viewport rect */
    var rect   = togBtn.getBoundingClientRect();
    var menuW  = 220;
    var left   = rect.right - menuW;
    if (left < 4) left = 4;
    if (left + menuW > window.innerWidth - 4)
      left = window.innerWidth - menuW - 4;

    /* Flip upward when insufficient space below */
    var spaceBelow = window.innerHeight - rect.bottom - 8;
    if (spaceBelow < 160 && rect.top > 160) {{
      /* open above toggle */
      menu.style.bottom = (window.innerHeight - rect.top + 4) + "px";
      menu.style.top    = "auto";
    }} else {{
      menu.style.top    = (rect.bottom + 4) + "px";
      menu.style.bottom = "auto";
    }}
    menu.style.left    = left + "px";
    menu.style.display = "block";
    togBtn.setAttribute("aria-expanded", "true");
    togBtn.textContent = "\\u25b4";
  }}

  function closeMenu() {{
    menu.style.display = "none";
    var togBtn = document.getElementById(WID + "-togbtn");
    if (togBtn) {{
      togBtn.setAttribute("aria-expanded", "false");
      togBtn.textContent = "\\u25be";
    }}
  }}

  /* ── Menu item builder (uses #id-scoped CSS classes) ── */
  function makeItem(iconUri, title, desc, onClickFn, disabled) {{
    var li = document.createElement("li");
    li.setAttribute("role", "menuitem");
    li.className = "ai-mi" + (disabled ? " ai-dis" : "");

    var cnt = document.createElement("div");
    cnt.className = "ai-cnt";

    var titleRow = document.createElement("div");
    titleRow.className = "ai-ti";

    if (iconUri) {{
      var ico = document.createElement("span");
      ico.className = "ai-ico";
      ico.style.backgroundImage = "url('" + iconUri + "')";
      titleRow.appendChild(ico);
    }}

    var ts = document.createElement("span");
    ts.textContent = title;
    titleRow.appendChild(ts);

    var de = document.createElement("div");
    de.className = "ai-de";
    de.textContent = desc;

    cnt.appendChild(titleRow);
    cnt.appendChild(de);
    li.appendChild(cnt);

    if (!disabled && onClickFn) {{
      li.addEventListener("click", function(e) {{
        e.preventDefault(); e.stopPropagation();
        onClickFn();
      }});
    }}
    return li;
  }}

  function makeSep() {{
    var li = document.createElement("li");
    li.className = "ai-sp";
    li.setAttribute("role", "separator");
    return li;
  }}

  /* ── Populate menu (Sphinx-identical order) ── */

  /* 1. Copy page */
  menu.appendChild(makeItem(
    SVG_COPY, "Copy page", "Copy this page as Markdown for LLMs.",
    function() {{ handleCopy(); }}
  ));

  /* 2. View as Markdown */
  menu.appendChild(makeItem(
    SVG_MARKDOWN, "View as Markdown", "View this page as Markdown.",
    function() {{
      window.open(getMdUrl(), "_blank", "noopener,noreferrer");
      closeMenu();
    }}
  ));

  /* 3. Separator before AI providers */
  if (BUTTONS.length > 0) menu.appendChild(makeSep());

  /* 4. AI provider rows */
  BUTTONS.forEach(function(btn) {{
    var isLocal    = btn.type === "local";
    var isDisabled = isLocal && !btn.enabled;
    var desc = btn.desc || (isLocal ? "Local provider — requires local AI server" : "");
    var item = makeItem(
      btn.icon, btn.label, desc,
      isDisabled ? null : (function(b) {{
        return function() {{
          var url = buildAiUrl(b);
          if (url) window.open(url, "_blank", "noopener,noreferrer");
          closeMenu();
        }};
      }})(btn),
      isDisabled
    );
    if (isLocal && isDisabled)
      item.title = "Local provider — requires local AI server";
    menu.appendChild(item);
  }});

  /* 5. MCP tool rows (only enabled entries) */
  var mcpKeys = Object.keys(MCP_TOOLS || {{}}).filter(function(k) {{
    return MCP_TOOLS[k] && MCP_TOOLS[k].enabled;
  }});
  if (mcpKeys.length > 0) {{
    menu.appendChild(makeSep());
    mcpKeys.forEach(function(tname) {{
      var tcfg = MCP_TOOLS[tname];
      var sUrl = String(tcfg.server_url || tcfg.mcpb_url || "");
      menu.appendChild(makeItem(
        null,
        String(tcfg.label || tname),
        String(tcfg.description || ""),
        sUrl ? (function(u) {{
          return function() {{
            window.open(u, "_blank", "noopener,noreferrer");
            closeMenu();
          }};
        }})(sUrl) : null
      ));
    }});
  }}

  /* ── Wire primary button and toggle ── */
  var mainBtn = document.getElementById(WID + "-mainbtn");
  var togBtn  = document.getElementById(WID + "-togbtn");
  if (mainBtn) mainBtn.addEventListener("click", function(e) {{
    e.stopPropagation(); handleCopy();
  }});
  if (togBtn) togBtn.addEventListener("click", function(e) {{
    e.stopPropagation(); toggleMenu();
  }});

  /* ── Click-outside-to-close ──
     Use capture phase (true) so we intercept before stopPropagation
     on inner elements.  Check both the widget div and the body-portal menu. */
  document.addEventListener("click", function(e) {{
    var widget = document.getElementById(WID);
    var inWidget = widget && widget.contains(e.target);
    var inMenu   = menu.contains(e.target);
    if (!inWidget && !inMenu) closeMenu();
  }}, true);

  /* Close and reposition on scroll or resize — menu coords are stale */
  window.addEventListener("scroll", closeMenu, true);
  window.addEventListener("resize", closeMenu);

  /* ── Clean up body-portal elements when widget is removed ── */
  (function() {{
    if (typeof MutationObserver === "undefined") return;
    var obs = new MutationObserver(function() {{
      if (!document.getElementById(WID)) {{
        var m = document.getElementById("ai-m-" + WID);
        var s = document.getElementById("ai-ms-" + WID);
        if (m && m.parentNode) m.parentNode.removeChild(m);
        if (s && s.parentNode) s.parentNode.removeChild(s);
        obs.disconnect();
      }}
    }});
    obs.observe(document.body, {{childList: true, subtree: true}});
  }})();

}})();
</script>
</div>"""
    return html  # noqa: RET504


# ---------------------------------------------------------------------------
# Public display functions
# ---------------------------------------------------------------------------


def display_jupyter_ai_button(
    content: str | None = None,
    *,
    providers: str | list[str] | None = None,
    provider_configs: dict[str, Any] | None = None,
    position: str = "inline",
    page_url: str = "",
    widget_id: str | None = None,
    intention: str | None = None,
    custom_context: str | None = None,
    custom_prompt_prefix: str | None = None,
    notebook_mode: bool = False,
    include_outputs: bool = False,
    include_raw_image: bool = False,
    mcp_tools: dict[str, Any] | None = None,
) -> None:
    """
    Inject an AI-assistant expandable dropdown widget into the current
    Jupyter output cell.

    Call this function directly after a visualisation or cell output to add
    a ``Copy page`` split-button with an expandable dropdown that mirrors
    the Sphinx extension UI exactly.  The primary button copies the cell
    content to the clipboard; the dropdown offers ``Copy page``, ``View as
    Markdown``, AI provider rows, and optionally MCP tool rows.

    Parameters
    ----------
    content : str or None, optional
        Explicit text/description to include in the AI prompt.  When
        ``None`` and *notebook_mode* is ``False``, the widget JS captures
        text from the surrounding Jupyter output area automatically.
        When ``None`` and *notebook_mode* is ``True``, all visible
        notebook cells are captured.
    providers : str or list of str or None, optional
        Ordered list of provider names to show.  A bare ``str`` is treated
        as a single-element list.  Valid names: ``"claude"``,
        ``"chatgpt"``, ``"gemini"``, ``"ollama"``, ``"mistral"``,
        ``"perplexity"``, ``"copilot"``, ``"groq"``, ``"you"``,
        ``"deepseek"``, ``"huggingface"``, ``"custom"``.
        Defaults to ``["claude", "chatgpt", "gemini", "ollama"]``.
    provider_configs : dict or None, optional
        Per-provider config overrides merged over :data:`_DEFAULT_PROVIDERS`.
    position : str, optional
        ``"inline"`` (default) or ``"floating"``.
    page_url : str, optional
        URL embedded in provider prompt templates as ``{url}``.  When set,
        the ``.md`` URL is derived (Sphinx-identical logic) for both
        ``View as Markdown`` and AI provider prompts.
    widget_id : str or None, optional
        Unique DOM ID.  Auto-generated when ``None``.
    intention : str or None, optional
        User's stated goal.  Prepended to the prompt as
        ``"Goal: <intention>"``.
    custom_context : str or None, optional
        Additional background context injected into the prompt.
    custom_prompt_prefix : str or None, optional
        Raw text prepended before everything else in the final prompt.
    notebook_mode : bool, optional
        When ``True``, captures all notebook cells instead of just the
        current output area.
    include_outputs : bool, optional
        When *notebook_mode* is ``True``, whether to include cell outputs.
        Defaults to ``False``.
    include_raw_image : bool, optional
        When ``True``, scans ``<img>`` and ``<canvas>`` elements in the
        output area and appends image metadata (and canvas thumbnails) to
        the AI prompt.  Defaults to ``False``.
    mcp_tools : dict or None, optional
        MCP tool configs to render as rows in the dropdown.  Only entries
        with ``enabled: True`` are shown.  When ``None``, no MCP rows
        are shown.

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
    :func:`._safe_json_for_script` before embedding in the ``<script>``
    block, preventing XSS.

    **UX**: Sphinx-identical split-button.  Primary button copies page
    content; arrow toggle opens the dropdown with ``Copy page``,
    ``View as Markdown``, AI providers, and optional MCP tools.  Fixed
    dropdown positioning avoids VS Code cell clipping.

    Examples
    --------
    After a matplotlib plot:

    .. code-block:: python

        from scikitplot._externals._sphinx_ext._jupyter_ai_assistant import (
            display_jupyter_ai_button,
        )
        import matplotlib.pyplot as plt

        plt.plot([1, 2, 3])
        plt.show()
        display_jupyter_ai_button(
            content="A line chart showing values 1, 2, 3.",
            providers=["claude", "chatgpt", "gemini"],
            intention="Explain the trend",
        )
    """  # noqa: D205
    if position not in {"inline", "floating"}:
        raise ValueError(f"position must be 'inline' or 'floating'; got {position!r}")

    try:
        from IPython.display import (  # type: ignore[import]  # noqa: PLC0415
            HTML,
            display,
        )
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
    intention: str | None = None,
    *,
    providers: str | list[str] | None = None,
    provider_configs: dict[str, Any] | None = None,
    position: str = "inline",
    page_url: str = "",
    widget_id: str | None = None,
    include_outputs: bool = False,
    include_raw_image: bool = False,
    custom_context: str | None = None,
    custom_prompt_prefix: str | None = None,
    mcp_tools: dict[str, Any] | None = None,
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

        When ``None``, no goal annotation is added.
    providers : str or list of str or None, optional
        Provider names to show as rows in the dropdown.  A bare ``str``
        is treated as a single-element list.  Defaults to
        ``["claude", "chatgpt", "gemini", "ollama"]``.
    provider_configs : dict or None, optional
        Per-provider config overrides merged over :data:`_DEFAULT_PROVIDERS`.
    position : str, optional
        ``"inline"`` (default) or ``"floating"``.
    page_url : str, optional
        URL embedded in provider prompt templates as ``{url}``.  The widget
        derives the ``.md`` equivalent for ``View as Markdown`` and AI
        prompts.
    widget_id : str or None, optional
        Unique DOM ID.  Auto-generated when ``None``.
    include_outputs : bool, optional
        When ``True``, cell outputs are included in the captured text.
        Defaults to ``False`` — pass ``True`` to include error tracebacks,
        print output, or display results.
    include_raw_image : bool, optional
        When ``True``, captures canvas/image thumbnails from all cell
        output areas.  Defaults to ``False``.
    custom_context : str or None, optional
        Additional background context (domain, data description, etc.).
    custom_prompt_prefix : str or None, optional
        Raw text prepended before everything else in the final prompt.
    mcp_tools : dict or None, optional
        MCP tool configs rendered as rows in the dropdown.  When ``None``,
        no MCP rows are shown.

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
    other environments fall back gracefully to single-cell capture.

    Examples
    --------
    At the end of a notebook to request a full review:

    .. code-block:: python

        from scikitplot._externals._sphinx_ext._jupyter_ai_assistant import (
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
