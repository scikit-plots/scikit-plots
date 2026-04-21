# scikitplot/_externals/_sphinx_ext/_sphinx_jinja_render/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot._externals._sphinx_ext._sphinx_jinja_render
======================================================

This package provides:

* **RST template rendering** — Jinja2-based pre-processing of
  ``*.rst.template`` files into plain ``*.rst`` files.
* **REPL URL injection** — builds a JupyterLite REPL URL containing a
  bootstrap snippet and injects it into every HTML page context.

Only the symbols listed in :data:`__all__` are considered stable.
Everything else (modules prefixed with ``_``) is an implementation
detail and may change without notice.

Notes
-----
Developer
    Import from this ``__init__`` in consumer code, not from the
    internal modules.  This keeps the public surface area minimal and
    lets us rename or split internals without breaking callers.

User
    Register the extension by adding the value of :data:`EXTENSION_NAME`
    to ``extensions`` in your Sphinx ``conf.py``::

        extensions = ["scikitplot._externals._sphinx_ext._sphinx_jinja_render"]
"""  # noqa: D205, D400

from __future__ import annotations

from ._bootstrap import load_bootstrap_code
from ._constants import (  # noqa: F401
    BOOTSTRAP_CODE_FILENAME,
    DEFAULT_KERNEL_NAME,
    EXTENSION_NAME,
    EXTENSION_VERSION,
    JUPYTERLITE_BASE_URL,
    SKPLT_JUPYTERLITE_BASE_URL,
    WASM_BOOTSTRAP_CODE,
    WASM_FALLBACK_CODE,
)
from ._extension import setup
from ._rst_renderer import render_rst_templates
from ._url_builder import build_repl_url

__all__: list[str] = [  # noqa: RUF022
    # Extension entry point
    "setup",
    # Core builders
    "build_repl_url",
    "load_bootstrap_code",
    "render_rst_templates",
    # Constants
    "BOOTSTRAP_CODE_FILENAME",
    "DEFAULT_KERNEL_NAME",
    "EXTENSION_NAME",
    "JUPYTERLITE_BASE_URL",
    "SKPLT_JUPYTERLITE_BASE_URL",
    "WASM_BOOTSTRAP_CODE",
]
