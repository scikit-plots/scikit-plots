# scikitplot/_externals/_sphinx_ext/_sphinx_youtube_gallery/__init__.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Render YouTube catalogs as Sphinx galleries with optional local reader controls.

Enable ``_sphinx_ext._sphinx_youtube_gallery`` for a standalone extension tree,
or ``scikitplot._externals._sphinx_ext._sphinx_youtube_gallery`` inside the main
package. Use one namespace consistently within a Sphinx application.
Dependencies are loaded automatically when Sphinx calls :func:`setup`.
Importing the package alone does not register any directives.

The ``youtube-gallery`` directive accepts an inline video list, a homogeneous
``videos:`` or ``channels:`` wrapper, or a UTF-8 catalog file. A video catalog
can also be projected into a deduplicated offline channel index with
``:view: channels``. Its build-time query selects the records;
optional ``:interactive:`` controls search, filter, sort, and bounded display
of the emitted cards locally (12 by default, shared presets, and Load 12 more). The adapter delegates the entire V4 controls -> status -> cards structure
to ``gallery-grid`` / ``_sphinx_collection``; it never embeds match counts in
the search controls and validates the delegated structure before returning it. Namespaced ``grid-*``, ``card-*`` and ``video-*`` options customize
the generated layout and players. See ``README.md`` beside this module for
configuration, complete examples, option ownership and migration notes.
"""

from __future__ import annotations

__all__ = ["setup"]


def setup(app):
    """
    Register the ``youtube-gallery`` directive with a Sphinx application.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        The Sphinx application to extend.

    Returns
    -------
    dict
        Extension metadata declaring parallel read/write safety.
    """
    from .directive import setup as _setup  # ruff: ignore[import-outside-top-level]

    return _setup(app)
