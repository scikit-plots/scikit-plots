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

Attributes
----------
_sphinx_gallery_grid
    Theme-independent owner of the public ``gallery-grid`` directive.
_sphinx_collection
    Shared filtering, grouping, browser metadata and live-control engine used
    by collection-style directives.
_sphinx_youtube_core
    Dependency-free YouTube URL grammar and player option primitives.
_sphinx_youtube_gallery
    Typed YouTube video/channel adapter that renders through ``gallery-grid``.
_pydata_component_list
    PyData Sphinx Theme-specific owner of the public ``component-list``
    inventory directive.
_sphinxcontrib_youtube
    Vendored standalone YouTube/Vimeo/PeerTube player directives used by the
    gallery adapter.

Notes
-----
*Users:* To register an extension in a Sphinx project, add the full
dotted path to ``extensions`` in ``conf.py``::

    extensions = [
        "scikitplot._externals._sphinx_ext._sphinx_youtube_gallery",
    ]

*Developers:* Add bundled extensions to ``_CORE_PRIVATE_SUBMODULES``.
Known sibling extensions that are intentionally outside this replacement may
be listed in ``_OPTIONAL_PRIVATE_SUBMODULES`` and are exposed only when they
actually exist. ``__all__`` stays empty on purpose: star-importing a private
extension namespace would eagerly load Sphinx-dependent children. ``dir()``
still exposes the lazy registry for introspection.

Examples
--------
>>> # Safe: no Sphinx needed yet.
>>> import _sphinx_ext
>>> sorted(name for name in _sphinx_ext._PRIVATE_SUBMODULES if name.startswith("_sphinx_"))[:2]
['_sphinx_collection', '_sphinx_gallery_grid']
"""

from __future__ import annotations

from importlib.util import find_spec

__all__: list[str] = []

# ---------------------------------------------------------------------------
# Lazy-load registry
# ---------------------------------------------------------------------------

#: Submodules shipped by this replacement.
_CORE_PRIVATE_SUBMODULES: frozenset[str] = frozenset(
    {
        "_sphinx_youtube_gallery",
        "_sphinx_youtube_core",
        "_sphinx_collection",
        "_sphinxcontrib_youtube",
        "_sphinx_gallery_grid",
        "_pydata_component_list",
    }
)

#: Unrelated siblings known to exist in some complete scikit-plots trees.
#: They are not bundled in this replacement, but remain lazily reachable when
#: the archive is merged into a tree that actually contains them. This avoids
#: either deleting unrelated functionality or advertising modules that are
#: absent from a standalone extraction.
_OPTIONAL_PRIVATE_SUBMODULES: frozenset[str] = frozenset(
    {"_sphinx_ai_assistant", "_sphinx_jinja_render"}
)

_PRIVATE_SUBMODULES: frozenset[str] = _CORE_PRIVATE_SUBMODULES | frozenset(
    name
    for name in _OPTIONAL_PRIVATE_SUBMODULES
    if find_spec(f"{__name__}.{name}") is not None
)


def __dir__() -> list[str]:
    """Return normal module attributes plus lazily available child modules."""
    return sorted(set(globals()) | set(_PRIVATE_SUBMODULES))


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
    >>> import _sphinx_ext
    >>> "_sphinx_gallery_grid" in _sphinx_ext._PRIVATE_SUBMODULES
    True
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
