# scikitplot/_externals/_sphinx_ext/_sphinx_ai_assistant/_jupyter.py
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
:mod:`scikitplot._externals._sphinx_ext._sphinx_ai_assistant`.
Its public symbols are re-exported from the parent ``__init__.py`` so
callers import from the parent package as usual.

Public API
----------
display_jupyter_ai_button : callable
    Inject an AI-assistant expandable dropdown widget into the current
    Jupyter notebook output cell.  The widget is visually identical to
    the Sphinx extension split-button: a primary "Ask AI" button, an
    arrow toggle, and a dropdown list with icon, title, and description
    for every enabled provider.
display_jupyter_notebook_ai_button : callable
    Convenience wrapper around :func:`display_jupyter_ai_button` with
    ``notebook_mode=True`` — sends the entire notebook to the AI.

Internal API
------------
_JUPYTER_CONTENT_SELECTORS : tuple of str
    CSS selectors used by the widget JS to locate cell output text in
    JupyterLab ≥ 4, JupyterLab 3, classic Notebook, and VS Code Jupyter.
_build_jupyter_widget_html : callable
    Build the self-contained HTML+CSS+JS dropdown widget string.

Design notes
------------
* **Zero external dependencies at import time.**  All optional imports
  (IPython, hashlib) are deferred to call time.
* **XSS prevention.**  Every user-supplied string is serialised via
  :func:`._safe_json_for_script` before embedding in ``<script>`` blocks.
* **Expandable dropdown UX** (Sphinx-identical).  The widget renders a
  split-button with:

  - A primary "Ask AI" label button (no action — opens the dropdown on
    click of the arrow toggle).
  - An arrow toggle button (``▾``) that opens/closes a dropdown list.
  - One menu row per provider with: SVG icon, bold title, muted
    description.
  - Separator rows between provider groups.
  - A "Copied!" flash state on the primary button (future: copy action).
  - ``aria-expanded`` / ``role="menu"`` / ``role="menuitem"`` for
    accessibility.
  - Click-outside-to-close behaviour.

* **JS template placeholders** use ``{url}``, ``{content}``, ``{prompt}``
  in provider configs.  Inside the f-string these are produced via
  ``{{url}}``, ``{{content}}``, ``{{prompt}}`` to escape f-string brace
  processing while still yielding the literal ``{url}`` token in the
  output JS.

Notes
-----
**Developer note**: This module is imported by the parent ``__init__.py``
via ``from ._jupyter import (...)``.  The parent re-exports all public
symbols so that existing call sites (``from ...  import
display_jupyter_ai_button``) continue to work without modification.

Examples
--------
After a matplotlib plot in a Jupyter cell:

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
    )

At the end of a notebook for a full review:

.. code-block:: python

    from scikitplot._externals._sphinx_ext._sphinx_ai_assistant import (
        display_jupyter_notebook_ai_button,
    )

    display_jupyter_notebook_ai_button(
        intention="Review this notebook for bugs and suggest improvements",
        providers=["claude", "chatgpt"],
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
#: Ordered from most-specific to most-general.  JupyterLab ≥ 4,
#: JupyterLab 3, classic Notebook, VS Code Jupyter, and a generic
#: fallback are all covered.
_JUPYTER_CONTENT_SELECTORS: tuple[str, ...] = (
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

# ---------------------------------------------------------------------------
# Inline SVG icons — base64 data URIs
# ---------------------------------------------------------------------------
# Each constant holds a minimal monochrome SVG encoded as a base64 data URI
# for use as CSS ``background-image``.  Keeping them inline means the widget
# is fully self-contained with zero network requests.

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

# Map provider name → (SVG data URI, description text)
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
    include_outputs: bool = True,
    include_raw_image: bool = False,
    mcp_tools: dict[str, Any] | None = None,
) -> str:
    """Build a self-contained HTML+CSS+JS expandable AI dropdown widget.

    The widget is visually identical to the Sphinx extension split-button:
    a primary "Ask AI" label, an arrow toggle, and a dropdown list with
    icon, title, and description for every enabled provider.  No external
    resources are required — all CSS, JS, and SVG icons are inlined.

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
        URL embedded in provider prompt templates as ``{url}``.
    widget_id : str or None, optional
        Unique DOM ID.  Auto-generated when ``None``.
    intention : str or None, optional
        User's stated goal (e.g. ``"explain this chart"``,
        ``"find the bug"``).  Prepended to the prompt as
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
        are included alongside cell inputs.  Defaults to ``True``.
    include_raw_image : bool, optional
        When ``True``, the widget JS scans ``<img>`` and ``<canvas>``
        elements in the captured output area and appends image metadata
        (and canvas thumbnails) to the prompt.  Defaults to ``False``.

        .. note::
            Full-resolution base64 images exceed most URL length limits.
            Canvas thumbnails are compressed to max 300 x 300 px.

    mcp_tools : dict or None, optional
        MCP tool configurations to surface as "Connect" buttons in the
        dropdown alongside the AI provider rows.  Each value must match
        :data:`._DEFAULT_MCP_TOOLS` schema.  When ``None``, no MCP
        entries are rendered.

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

    **Dropdown UX**:

    * Split-button: primary label (no direct action) + ``▾`` arrow toggle.
    * Dropdown: one row per provider with ``<img>`` icon, bold title,
      muted description.
    * Separator row between the provider list and any MCP buttons.
    * ``aria-expanded`` / ``role="menu"`` / ``role="menuitem"`` for
      screen-reader accessibility.
    * Click-outside-to-close (``document.addEventListener("click", ...)``,
      stops propagation inside the widget).

    **JS template placeholders**: ``{url}``, ``{content}``, ``{prompt}``
    in provider configs become ``{{url}}``, ``{{content}}``,
    ``{{prompt}}`` in this f-string (escaping f-string brace processing)
    and ``{url}``, ``{content}``, ``{prompt}`` in the emitted JS.

    Examples
    --------
    >>> html = _build_jupyter_widget_html(
    ...     widget_id="demo",
    ...     providers=["claude", "chatgpt"],
    ...     intention="Explain this chart",
    ... )
    >>> assert "demo" in html
    >>> assert "Ask AI" in html
    >>> assert "Ask Claude" in html
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
                "desc": meta["desc"],
            }
        )

    # ── Validate page URL ────────────────────────────────────────────────────
    validated_page_url = ""
    if page_url:
        try:
            validated_page_url = _validate_base_url(page_url)
        except ValueError:
            validated_page_url = ""

    # ── Validate and serialise MCP tools ─────────────────────────────────────
    safe_mcp: dict[str, Any] = {}
    if mcp_tools:
        for tname, tcfg in mcp_tools.items():
            errs = _validate_mcp_tool(tcfg, name=tname)
            if not errs:
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
        else "margin:8px 0;display:inline-block;"
    )

    # ── Build self-contained HTML ─────────────────────────────────────────────
    # Notes on f-string brace escaping below:
    #   {{  → literal {   in the output (f-string escape)
    #   }}  → literal }   in the output (f-string escape)
    #   \\  → literal \   in the output (Python string escape)
    #   The JS placeholders {url}, {content}, {prompt} inside provider
    #   template strings are NOT expanded here — they are replaced at
    #   runtime by the widget JS using String.prototype.split/join.
    html = f"""
<div id="{widget_id}" style="{position_style}font-family:sans-serif;">
<style>
/* ── AI Dropdown Widget — Sphinx-identical split-button UX ── */
#{widget_id} {{
  display:inline-block;
  position:relative;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
}}
#{widget_id} .aijup-split {{
  display:inline-flex;
  align-items:stretch;
  border:1px solid #c8cdd3;
  border-radius:6px;
  background:#fff;
  box-shadow:0 1px 3px rgba(0,0,0,.08);
  overflow:hidden;
}}
#{widget_id} .aijup-main {{
  display:inline-flex;
  align-items:center;
  gap:6px;
  padding:5px 11px;
  font-size:12.5px;
  font-weight:600;
  color:#24292f;
  background:none;
  border:none;
  cursor:default;
  white-space:nowrap;
  user-select:none;
}}
#{widget_id} .aijup-tog {{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  width:26px;
  background:none;
  border:none;
  border-left:1px solid #c8cdd3;
  cursor:pointer;
  color:#57606a;
  font-size:11px;
  transition:background .12s;
  padding:0;
}}
#{widget_id} .aijup-tog:hover {{
  background:#f3f4f6;
}}
#{widget_id} .aijup-menu {{
  display:none;
  position:absolute;
  top:calc(100% + 4px);
  left:0;
  min-width:220px;
  background:#fff;
  border:1px solid #c8cdd3;
  border-radius:8px;
  box-shadow:0 8px 24px rgba(0,0,0,.12);
  z-index:10000;
  padding:4px 0;
  list-style:none;
  margin:0;
}}
#{widget_id} .aijup-menu.open {{
  display:block;
}}
#{widget_id} .aijup-item {{
  display:flex;
  align-items:center;
  gap:10px;
  padding:7px 14px;
  cursor:pointer;
  transition:background .1s;
  text-decoration:none;
  color:inherit;
  border:none;
  background:none;
  width:100%;
  text-align:left;
  box-sizing:border-box;
}}
#{widget_id} .aijup-item:hover {{
  background:#f3f4f6;
}}
#{widget_id} .aijup-item.disabled {{
  opacity:.45;
  cursor:not-allowed;
}}
#{widget_id} .aijup-icon {{
  width:18px;
  height:18px;
  flex-shrink:0;
  border-radius:3px;
  background-size:contain;
  background-repeat:no-repeat;
  background-position:center;
}}
#{widget_id} .aijup-text {{
  display:flex;
  flex-direction:column;
  min-width:0;
}}
#{widget_id} .aijup-title {{
  font-size:12.5px;
  font-weight:600;
  color:#24292f;
  line-height:1.3;
}}
#{widget_id} .aijup-desc {{
  font-size:11px;
  color:#57606a;
  line-height:1.3;
  white-space:nowrap;
  overflow:hidden;
  text-overflow:ellipsis;
}}
#{widget_id} .aijup-sep {{
  height:1px;
  background:#eaecef;
  margin:4px 0;
}}
</style>

<div class="aijup-split" role="group" aria-label="Ask AI">
  <span class="aijup-main" id="{widget_id}-mainbtn" aria-label="Ask AI">
    &#129302; Ask AI
  </span>
  <button class="aijup-tog" id="{widget_id}-togbtn"
          aria-haspopup="menu" aria-expanded="false"
          aria-controls="{widget_id}-menu"
          onclick="(function(){{
            var m=document.getElementById('{widget_id}-menu');
            var t=document.getElementById('{widget_id}-togbtn');
            var open=m.classList.toggle('open');
            t.setAttribute('aria-expanded',open?'true':'false');
            t.textContent=open?'\\u25b4':'\\u25be';
          }})();event.stopPropagation();"
          title="Show AI providers">&#9662;</button>
</div>

<ul class="aijup-menu" id="{widget_id}-menu"
    role="menu" aria-labelledby="{widget_id}-mainbtn">
</ul>

<script>
(function() {{
  var W_ID             = "{widget_id}";
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

  /* ── Content capture ── */
  function getCellContent() {{
    var root = document.getElementById(W_ID);
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
    return parts.join("\\n");
  }}

  function getContent() {{
    if (EXPLICIT_CONTENT !== null && EXPLICIT_CONTENT !== undefined) {{
      return String(EXPLICIT_CONTENT);
    }}
    return NOTEBOOK_MODE ? getNotebookContent() : getCellContent();
  }}

  /* ── Canvas / image capture (include_raw_image mode) ── */
  function captureImages(outputArea) {{
    if (!INCLUDE_RAW_IMAGE || !outputArea) return "";
    var parts = [];
    var canvases = outputArea.querySelectorAll("canvas");
    canvases.forEach(function(cv) {{
      try {{
        var w = cv.width, h = cv.height;
        if (!w || !h) return;
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
    return parts.length ? "\\n\\n[Visual outputs]\\n" + parts.join("\\n") : "";
  }}

  /* ── Prompt builder ── */
  function buildPrompt(btn, content) {{
    var parts = [];
    if (PROMPT_PREFIX)  parts.push(String(PROMPT_PREFIX));
    if (INTENTION)      parts.push("Goal: " + String(INTENTION));
    if (CUSTOM_CONTEXT) parts.push("Context: " + String(CUSTOM_CONTEXT));
    var root2 = document.getElementById(W_ID);
    var outputArea = root2 ? root2.closest(
      ".jp-OutputArea, .output_area, .cell-output, .output"
    ) : (document.querySelector(".jp-OutputArea") || null);
    var imageText = captureImages(outputArea);
    if (imageText) content = content + imageText;
    /* Replace {{url}} and {{content}} placeholders — using split/join to
       avoid regex backslash escaping inside an f-string.             */
    var main = btn.prompt_template
      .split("{{url}}").join(PAGE_URL || window.location.href)
      .split("{{content}}").join(content);
    parts.push(main);
    return parts.join("\\n\\n");
  }}

  function buildUrl(btn) {{
    var content = getContent();
    var prompt  = buildPrompt(btn, content);
    /* Replace {{prompt}} placeholder — using split/join (see above). */
    return btn.url_template
      .split("{{prompt}}").join(encodeURIComponent(prompt));
  }}

  /* ── Render dropdown menu items ── */
  var menu = document.getElementById(W_ID + "-menu");

  BUTTONS.forEach(function(btn) {{
    var isLocal    = btn.type === "local";
    var isDisabled = isLocal && !btn.enabled;
    var li = document.createElement("li");
    li.setAttribute("role", "menuitem");
    li.className = "aijup-item" + (isDisabled ? " disabled" : "");
    li.title = isLocal ? "Local provider \u2014 requires local AI server" : btn.label;

    /* Icon */
    var icon = document.createElement("span");
    icon.className = "aijup-icon";
    if (btn.icon) icon.style.backgroundImage = "url('" + btn.icon + "')";

    /* Text block */
    var textDiv = document.createElement("div");
    textDiv.className = "aijup-text";
    var titleSpan = document.createElement("span");
    titleSpan.className = "aijup-title";
    titleSpan.textContent = btn.label;
    var descSpan = document.createElement("span");
    descSpan.className = "aijup-desc";
    descSpan.textContent = btn.desc || "";
    textDiv.appendChild(titleSpan);
    textDiv.appendChild(descSpan);

    li.appendChild(icon);
    li.appendChild(textDiv);

    if (!isDisabled) {{
      li.style.cursor = "pointer";
      li.addEventListener("click", (function(b) {{
        return function(e) {{
          e.preventDefault();
          e.stopPropagation();
          var url = buildUrl(b);
          if (url) window.open(url, "_blank", "noopener,noreferrer");
          /* Close menu */
          menu.classList.remove("open");
          var tog = document.getElementById(W_ID + "-togbtn");
          if (tog) {{ tog.setAttribute("aria-expanded","false"); tog.textContent="\\u25be"; }}
        }};
      }})(btn));
    }}

    menu.appendChild(li);
  }});

  /* ── MCP tool rows ── */
  if (MCP_TOOLS && typeof MCP_TOOLS === "object") {{
    var mcpKeys = Object.keys(MCP_TOOLS).filter(function(k) {{
      return MCP_TOOLS[k] && MCP_TOOLS[k].enabled;
    }});
    if (mcpKeys.length > 0) {{
      var sep = document.createElement("li");
      sep.className = "aijup-sep";
      sep.setAttribute("role", "separator");
      menu.appendChild(sep);

      mcpKeys.forEach(function(tname) {{
        var tcfg = MCP_TOOLS[tname];
        var li2 = document.createElement("li");
        li2.setAttribute("role", "menuitem");
        li2.className = "aijup-item";
        li2.title = String(tcfg.description || tname);

        var icon2 = document.createElement("span");
        icon2.className = "aijup-icon";

        var textDiv2 = document.createElement("div");
        textDiv2.className = "aijup-text";
        var titleSpan2 = document.createElement("span");
        titleSpan2.className = "aijup-title";
        titleSpan2.textContent = String(tcfg.label || tname);
        var descSpan2 = document.createElement("span");
        descSpan2.className = "aijup-desc";
        descSpan2.textContent = String(tcfg.description || "");
        textDiv2.appendChild(titleSpan2);
        textDiv2.appendChild(descSpan2);

        li2.appendChild(icon2);
        li2.appendChild(textDiv2);

        var sUrl = String(tcfg.server_url || tcfg.mcpb_url || "");
        if (sUrl) {{
          li2.addEventListener("click", (function(u) {{
            return function(e) {{
              e.stopPropagation();
              window.open(u, "_blank", "noopener,noreferrer");
              menu.classList.remove("open");
              var tog2 = document.getElementById(W_ID + "-togbtn");
              if (tog2) {{ tog2.setAttribute("aria-expanded","false"); tog2.textContent="\\u25be"; }}
            }};
          }})(sUrl));
        }}

        menu.appendChild(li2);
      }});
    }}
  }}

  /* ── Click-outside-to-close ── */
  document.addEventListener("click", function() {{
    menu.classList.remove("open");
    var tog3 = document.getElementById(W_ID + "-togbtn");
    if (tog3) {{ tog3.setAttribute("aria-expanded","false"); tog3.textContent="\\u25be"; }}
  }});
  /* Prevent widget clicks from closing the menu */
  var widget = document.getElementById(W_ID);
  if (widget) {{
    widget.addEventListener("click", function(e) {{ e.stopPropagation(); }});
  }}

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
    include_outputs: bool = True,
    include_raw_image: bool = False,
    mcp_tools: dict[str, Any] | None = None,
) -> None:
    """
    Inject an AI-assistant expandable dropdown widget into the current
    Jupyter output cell.

    Call this function directly after a visualisation or cell output to add
    an expandable split-button that opens a dropdown list of AI providers.
    Each provider row shows an icon, bold title, and description.  Clicking
    any row opens the AI chat in a new tab with the cell content pre-loaded
    into the prompt.

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
        User's stated goal (e.g. ``"explain this chart"``,
        ``"fix the error"``).  Prepended to the prompt as
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
        Defaults to ``True``.
    include_raw_image : bool, optional
        When ``True``, scans ``<img>`` and ``<canvas>`` elements in the
        output area and appends image metadata (and canvas thumbnails) to
        the AI prompt.  Defaults to ``False``.
    mcp_tools : dict or None, optional
        MCP tool configs to render as rows in the dropdown alongside the
        AI provider buttons.  Each value must match
        :data:`_DEFAULT_MCP_TOOLS` schema.  When ``None``, no MCP rows
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

    **UX**: The widget renders a Sphinx-identical split-button dropdown.
    The arrow toggle (``▾``) opens the menu; clicking outside closes it.
    Provider rows are fully clickable.

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
    include_outputs: bool = True,
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
        URL embedded in provider prompt templates as ``{url}``.
    widget_id : str or None, optional
        Unique DOM ID.  Auto-generated when ``None``.
    include_outputs : bool, optional
        When ``True`` (default), cell outputs are included in the captured
        text.  Set to ``False`` to send only source code.
    include_raw_image : bool, optional
        When ``True``, captures canvas/image thumbnails from all cell
        output areas and appends them to the prompt.  Defaults to
        ``False``.
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
