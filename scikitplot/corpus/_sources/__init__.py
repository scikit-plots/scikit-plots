"""
scikitplot.corpus._sources
==========================
Higher-level source abstraction over files, directories, URL lists,
and glob patterns.

Provides :class:`CorpusSource` — a declarative descriptor that the
pipeline resolves into a stream of ``(input, provenance)`` pairs, feeding
:meth:`~scikitplot.corpus._base.DocumentReader.create` or
:meth:`~scikitplot.corpus._base.DocumentReader.from_url`.

Why this exists
---------------
:class:`~scikitplot.corpus._pipeline.CorpusPipeline` accepts individual
files or explicit lists. Real corpora often need to express:

* "all ``*.txt`` in this directory tree"
* "these 400 URLs from a manifest file"
* "this S3 prefix (future)"

:class:`CorpusSource` encapsulates that logic in one place so the pipeline
never needs to know where documents come from.

Usage
-----
>>> from pathlib import Path
>>> from scikitplot.corpus._sources import CorpusSource, SourceKind
>>> src = CorpusSource.from_directory(Path("corpus/"), pattern="*.txt")
>>> for entry in src.iter_entries():
...     print(entry.path_or_url)
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _source,
)
from ._source import *  # noqa: F403

__all__ = []
__all__ += _source.__all__
