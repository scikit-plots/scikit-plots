# scikitplot/annoy/_mixins/__init__.py
"""
Composable mixins for :mod:`~scikitplot.annoy`.

This package provides *Python-side* mixins used by the high-level Annoy wrapper
(API layer). It intentionally does **not** modify or wrap the low-level C-API
surface directly; instead it composes small, single-responsibility behaviors
that high-level classes can inherit.

- Stable surface: exported names here are treated as public within
  :mod:`~scikitplot.annoy`.
- Dependency-light imports: optional heavy dependencies (for example pandas,
  pyarrow, scipy, matplotlib, mlflow) must be imported lazily *inside* the
  methods that require them (leaf modules enforce this).
- **No hidden behavior**: these mixins remain deterministic and explicit
  (no implicit sampling, truncation, or size-based heuristics).

Notes
-----
- Mixins must not define ``__init__``. The high-level
  :class:`~scikitplot.annoy.Index` inherits the low-level backend constructor
  unchanged.
- Imports are intentionally lazy to avoid importing optional heavy dependencies
  at package import time. Accessing an exported name triggers import of the leaf
  module that defines it.

See Also
--------
scikitplot.cexternals._annoy.annoylib.Annoy
    Low-level Annoy implementation (C-extension).
scikitplot.annoy._base
    High-level user-facing wrapper that composes these mixins.
"""

from __future__ import annotations

# from importlib import import_module
# from typing import TYPE_CHECKING, Any
from ._io import IndexIOMixin
from ._meta import MetaMixin
from ._ndarray import NDArrayMixin
from ._pickle import CompressMode, PickleMixin, PickleMode
from ._plotting import PlottingMixin
from ._vectors import VectorOpsMixin

# Keep the public surface explicit and stable.
__all__: tuple[str] = (
    "CompressMode",
    "IndexIOMixin",
    "MetaMixin",
    "NDArrayMixin",
    "PickleMixin",
    "PickleMode",
    "PlottingMixin",
    "VectorOpsMixin",
)
