"""
Generic ``gallery-grid`` Sphinx extension.

This package owns the public ``gallery-grid`` directive and delegates domain-
agnostic selection/browser behavior to ``_sphinx_collection``.  Enhanced grids
emit the shared V4 collection status placeholder as a root-level sibling; the
controls shell never owns result-count text. It is not tied to a Sphinx theme;
``youtube-gallery`` is one typed adapter that renders through this same grid
engine.
"""

from __future__ import annotations

__all__ = [  # ruff: ignore[undefined-export]
    "GalleryGridDirective",
    "setup",
]


def __getattr__(name: str):
    if name in __all__:
        from .directive import (  # ruff: ignore[import-outside-top-level]
            GalleryGridDirective,
            setup,
        )

        return {"GalleryGridDirective": GalleryGridDirective, "setup": setup}[name]
    raise AttributeError(name)
