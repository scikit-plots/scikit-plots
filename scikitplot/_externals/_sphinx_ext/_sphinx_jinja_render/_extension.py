# scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/_extension.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Sphinx extension entry point for the ``_sphinx_jinja_render`` submodule.

Wires the ``_sphinx_jinja_render`` helpers into a Sphinx build by connecting
``build-inited`` and ``html-page-context`` events.  This module
contains **only** Sphinx plumbing — all business logic lives in
sibling modules.

Register Sphinx event handlers that:

1. Render RST templates before Sphinx reads source files
   (``builder-inited`` event).
2. Inject the JupyterLite REPL URL into every HTML page's Jinja2
   context (``html-page-context`` event).

Notes
-----
Developer
    ``setup()`` is the only public symbol required by Sphinx.  Keep it
    thin — all logic lives in the private helpers and in the other
    submodules.

    The ``parallel_read_safe`` and ``parallel_write_safe`` flags are set
    to ``True`` because neither handler writes shared mutable state.

User
    Add ``"scikitplot._externals._sphinx_ext._sphinx_jinja_render"`` (or the value of
    :data:`~._constants.EXTENSION_NAME`) to ``extensions`` in
    ``conf.py``.  No further configuration is needed for default
    behaviour.

    To use a custom bootstrap code snippet, drop a replacement
    ``_bootstrap_code.py.txt`` file into the ``_sphinx_jinja_render`` package
    directory.
"""

from __future__ import annotations

import importlib  # noqa: F401
import os  # noqa: F401
import textwrap  # noqa: F401
from pathlib import Path
from typing import Any

from sphinx.application import Sphinx

from ._bootstrap import load_bootstrap_code
from ._rst_renderer import render_rst_templates
from ._url_builder import build_repl_url

# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


def _on_builder_inited(app: Sphinx) -> None:
    """
    Render RST templates when Sphinx initialises the builder.

    Triggered by the ``builder-inited`` Sphinx event.  Searches
    ``app.srcdir`` for ``*.rst.template`` files and renders each one.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The running Sphinx application instance.

    Notes
    -----
    Developer
        ``app.srcdir`` is guaranteed to exist at this event stage.  No
        existence check is needed before calling
        :func:`~._rst_renderer.render_rst_templates`.
    """
    # app.env.url_helper_bootstrap_code = load_bootstrap_code()  # type: ignore[attr-defined]
    src_dir = Path(app.srcdir)
    context = dict(getattr(app.config, "index_template_kwargs", {})) or {}
    render_rst_templates(src_dir, context=context)


def _on_html_page_context(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    """
    Inject the REPL URL into the HTML page context.

    Triggered by the ``html-page-context`` Sphinx event.  Adds the key
    ``"repl_url"`` to *context* so that Jinja2 HTML templates can render
    it as ``{{ repl_url }}``.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The running Sphinx application instance.
    pagename : str
        Dotted page name of the page being rendered.
        Sphinx page name (e.g. ``"index"`` or ``"api/module"``).
    templatename : str
        Name of the Jinja2 HTML template being rendered.
    context : dict[str, Any]
        Mutable Jinja2 context dict — add keys here to expose them in
        templates.
    doctree : docutils.nodes.document or None
        Parsed doctree for the page (may be ``None`` for non-document
        non-RST pages).

    Notes
    -----
    Developer
        The function mutates *context* in place — this is the pattern
        Sphinx documents for ``html-page-context`` handlers.
    """
    # bootstrap_code: str = getattr(
    #     app.env, "url_helper_bootstrap_code", ""  # type: ignore[attr-defined]
    # )
    code: str = load_bootstrap_code()
    context["repl_url"] = build_repl_url(code)


# ---------------------------------------------------------------------------
# Sphinx setup
# ---------------------------------------------------------------------------


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension with a Sphinx application.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application being configured.

    Returns
    -------
    dict[str, Any]
        Extension metadata consumed by Sphinx.

    Notes
    -----
    Developer
        ``version`` is intentionally left as ``"0.0.1"`` — it is the
        extension schema version, not the scikitplot package version.
        Update it only when the extension's public interface changes in
        a backward-incompatible way.

    Examples
    --------
    In ``conf.py``::

        extensions = ["scikitplot._externals._sphinx_ext._sphinx_jinja_render"]
    """
    # ---- Feature flags -----------------------------------------------------
    app.add_config_value(
        "index_template_kwargs",
        {
            "development_link": "devel/index",
        },
        "html",
    )

    # ---- Event hooks -------------------------------------------------------
    app.connect("builder-inited", _on_builder_inited)
    app.connect("html-page-context", _on_html_page_context)

    return {
        "version": "0.0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
