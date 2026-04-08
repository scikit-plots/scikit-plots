# scikitplot/externals/_sphinx_ext/__init__.py
#
# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore
#
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
"""
scikitplot.externals._sphinx_ext
=================================

Private namespace for vendored Sphinx extensions.

All child submodules are loaded **lazily**: importing this package does not
pull in Sphinx, BeautifulSoup, markdownify, or any other heavy dependency.

Submodules
----------
_sphinx_ai_assistant
    AI-assistant Sphinx extension (markdown export, llms.txt, AI chat links).
    Copied and adapted from ``mlazag/sphinx-ai-assistant`` (MIT licence).
    Requires Sphinx ≥ 5 at *call time*, not at import time.

Notes
-----
To register the AI-assistant extension in a Sphinx project, add the full
dotted path to ``extensions`` in ``conf.py``::

    extensions = [
        "scikitplot.externals._sphinx_ext._sphinx_ai_assistant",
    ]

Examples
--------
>>> # Safe: no Sphinx needed yet
>>> from scikitplot.externals import _sphinx_ext
>>> # Sphinx imported here, on demand:
>>> ai = _sphinx_ext._sphinx_ai_assistant
"""
from __future__ import annotations

__all__: list[str] = []

# LazyLoad
_PRIVATE_SUBMODULES: frozenset[str] = frozenset({"_sphinx_ai_assistant"})


def __getattr__(name: str) -> object:
    """Lazy submodule loader.

    Parameters
    ----------
    name : str
        Attribute name requested on this package.

    Returns
    -------
    object
        The requested lazy submodule.

    Raises
    ------
    AttributeError
        If *name* is not a recognised private submodule.
    """
    if name in _PRIVATE_SUBMODULES:
        import importlib
        module = importlib.import_module(f".{name}", package=__name__)
        globals()[name] = module
        return module
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available lazy submodules: {sorted(_PRIVATE_SUBMODULES)}"
    )
