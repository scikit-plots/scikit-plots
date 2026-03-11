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

from scikitplot.corpus._storage._storage import (
    InMemoryStorage,
    JSONLStorage,
    QueryResult,
    SQLiteStorage,
    StorageBase,
    StorageQuery,
)

__all__ = [
    "InMemoryStorage",
    "JSONLStorage",
    "QueryResult",
    "SQLiteStorage",
    "StorageBase",
    "StorageQuery",
]
