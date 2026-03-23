"""
scikitplot.corpus._export
==========================
Multi-format corpus export for
:class:`~scikitplot.corpus._schema.CorpusDocument` lists.

All formats are written **atomically**: data is first written to a
``*.tmp`` sibling file, then renamed over the target path. This prevents
partially-written files from being read by downstream processes if the
export is interrupted.

Python compatibility
--------------------
Python 3.8-3.15. Only ``csv``, ``json``, and ``pickle`` (all stdlib)
and ``numpy`` are hard requirements. All other backends are optional.
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _export,
)
from ._export import *  # noqa: F403

__all__ = []
__all__ += _export.__all__
