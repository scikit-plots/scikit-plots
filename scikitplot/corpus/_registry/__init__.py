"""
scikitplot.corpus._registry
============================
Runtime component registry for the corpus pipeline.

Provides :class:`ComponentRegistry` — a central look-up table mapping
string keys to:

* chunkers satisfying ``ChunkerProtocol``, each with its config dataclass
* :class:`~scikitplot.corpus._base.FilterBase` subclasses
* :class:`~scikitplot.corpus._base.DocumentReader` subclasses
* :class:`~scikitplot.corpus._normalizers.NormalizerBase` subclasses

:meth:`ComponentRegistry.register_builtins` adds the built-in components
(keeping any name already registered), and third-party components can be
registered at runtime.

Examples
--------
>>> from scikitplot.corpus._registry import ComponentRegistry, registry
>>> registry.register_builtins()
>>> registry.list_chunkers()
['fixed_window', 'paragraph', 'semantic', 'sentence', 'word']
>>> chunker_cls = registry.get_chunker("sentence")
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _registry,
)
from ._registry import *  # noqa: F403

__all__ = []
__all__ += _registry.__all__
