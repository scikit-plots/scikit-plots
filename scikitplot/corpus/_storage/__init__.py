"""
scikitplot.corpus._storage
===========================
Persistence layer for :class:`~scikitplot.corpus._schema.CorpusDocument`
collections.

Provides a pluggable storage backend contract (:class:`StorageBase`) and
three built-in implementations:

:class:`InMemoryStorage`
    Thread-safe dict store. No dependencies. Testing and prototyping only.

:class:`JSONLStorage`
    Append-friendly JSONL flat-file store. Atomic writes, zero
    dependencies beyond stdlib.

:class:`SQLiteStorage`
    SQLite-backed store via stdlib ``sqlite3``. Full-text search via
    FTS5. No external dependencies.

All backends implement the same four-method contract:
``save`` / ``save_batch`` / ``get`` / ``query``.
"""  # noqa: D205, D400

from __future__ import annotations

from . import (
    _storage,
)
from ._storage import *  # noqa: F403

__all__ = []
__all__ += _storage.__all__
