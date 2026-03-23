"""
scikitplot.corpus._registry
============================
Runtime component registry for the corpus pipeline.

Provides :class:`ComponentRegistry` — a central look-up table mapping
string keys to:

* :class:`~scikitplot.corpus._base.ChunkerBase` subclasses
* :class:`~scikitplot.corpus._base.FilterBase` subclasses
* :class:`~scikitplot.corpus._base.DocumentReader` subclasses
* :class:`~scikitplot.corpus._normalizers.NormalizerBase` subclasses

The registry is populated automatically when built-in components are
imported, and accepts third-party registrations at runtime.

Usage
-----
>>> from scikitplot.corpus._registry import ComponentRegistry, registry
>>> registry.list_chunkers()
['fixed_window', 'paragraph', 'sentence']
>>> chunker_cls = registry.get_chunker("sentence")
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _registry,
)
from ._registry import *  # noqa: F403

__all__ = []
__all__ += _registry.__all__
