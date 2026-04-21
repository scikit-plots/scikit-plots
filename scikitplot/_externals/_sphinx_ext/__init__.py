# scikitplot/_externals/_sphinx_ext/__init__.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot._externals._sphinx_ext
=================================

Private namespace for vendored Sphinx extensions.

All child submodules are loaded **lazily**: importing this package does
not pull in Sphinx, BeautifulSoup, markdownify, or any other heavy
dependency.  Only the specific submodule that is accessed at runtime
triggers its own import chain.

Submodules
----------
_sphinx_ai_assistant
    AI-assistant Sphinx extension (markdown export, llms.txt, AI chat
    links).  Copied and adapted from ``mlazag/sphinx-ai-assistant``
    (MIT licence).  Requires Sphinx ≥ 5 at *call time*, not at import
    time.
_sphinx_jinja_render
    URL helper Sphinx extension.  Provides RST template preprocessing
    (Jinja2 ``.rst.template`` → ``.rst``) and JupyterLite REPL URL
    injection into HTML page contexts.

Notes
-----
*Users:* To register an extension in a Sphinx project, add the full
dotted path to ``extensions`` in ``conf.py``::

    extensions = [
        "scikitplot._externals._sphinx_ext._sphinx_ai_assistant",
        "scikitplot._externals._sphinx_ext._sphinx_jinja_render",
    ]

*Developers:* To add a new private Sphinx extension submodule, append
its name to ``_PRIVATE_SUBMODULES``.  No other change is required for
lazy loading.

Examples
--------
>>> # Safe: no Sphinx needed yet
>>> from scikitplot._externals import _sphinx_ext
>>> # Sphinx imported here, on demand:
>>> ai = _sphinx_ext._sphinx_ai_assistant
>>> jr = _sphinx_ext._sphinx_jinja_render
"""

from __future__ import annotations

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Lazy-load registry
# ---------------------------------------------------------------------------

#: Names of all private submodules supported by the lazy loader.
#: Add new submodule names here to make them accessible via attribute access.
_PRIVATE_SUBMODULES: frozenset[str] = frozenset(
    {
        "_sphinx_ai_assistant",
        "_sphinx_jinja_render",
    }
)


def __getattr__(name: str) -> object:
    """
    Lazy submodule loader.

    Parameters
    ----------
    name : str
        Attribute name requested on this package.

    Returns
    -------
    object
        The requested lazily-loaded submodule, cached in ``globals()``
        for subsequent accesses.

    Raises
    ------
    AttributeError
        If *name* is not a recognised private submodule listed in
        :data:`_PRIVATE_SUBMODULES`.

    Examples
    --------
    >>> from scikitplot._externals import _sphinx_ext
    >>> jinja_render = _sphinx_ext._sphinx_jinja_render  # triggers import on first access
    """
    if name in _PRIVATE_SUBMODULES:
        import importlib

        module = importlib.import_module(f".{name}", package=__name__)
        globals()[name] = module  # cache for subsequent attribute access
        return module
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available lazy submodules: {sorted(_PRIVATE_SUBMODULES)}"
    )
