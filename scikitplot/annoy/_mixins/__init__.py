# scikitplot/annoy/_mixins/__init__.py
"""
Composable mixins for :mod:`~scikitplot.annoy`.

This package provides *Python-side* mixins used by the high-level Annoy wrapper
(API layer). It intentionally does **not** modify or wrap the low-level C-API
directly; instead it composes small, single-responsibility behaviors that
high-level classes can inherit.

Design goals
------------
- Stable surface: exported names here are treated as public within
  :mod:`~scikitplot.annoy`.
- Dependency-light imports: optional heavy dependencies (for example pandas,
  pyarrow, scipy, matplotlib, mlflow) must be imported lazily inside the
  methods that require them (leaf modules enforce this).
- **No hidden behavior**: these mixins should remain deterministic and explicit
  (no implicit sampling, truncation, or size-based heuristics).

See Also
--------
scikitplot.cexternals._annoy.annoylib.Annoy
    Low-level Annoy implementation (C-extension).
scikitplot.annoy._base
    High-level user-facing wrapper that composes these mixins.
"""

from __future__ import annotations

from ._io import IndexIOMixin, PickleIOMixin
from ._manifest import ManifestMixin
from ._ndarray import NDArrayExportMixin
from ._pickle import CompressMode, PickleMixin, PickleMode
from ._plotting import PlottingMixin
from ._vectors import VectorOpsMixin

__all__: tuple[str, ...] = (
    "CompressMode",
    "IndexIOMixin",
    "ManifestMixin",
    "NDArrayExportMixin",
    "PickleIOMixin",
    "PickleMixin",
    "PickleMode",
    "PlottingMixin",
    "VectorOpsMixin",
)
