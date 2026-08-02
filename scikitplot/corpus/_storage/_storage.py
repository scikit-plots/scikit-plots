# scikitplot/corpus/_storage/_storage.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._storage._storage
=====================================
Storage backend implementations for persisting and retrieving
:class:`~scikitplot.corpus._schema.CorpusDocument` collections.

Design invariants:

* ``StorageBase`` defines the minimum contract: ``save``, ``save_batch``,
  ``get``, ``query``. All implementations must satisfy it exactly.
* Documents are stored and returned as validated
  :class:`~scikitplot.corpus._schema.CorpusDocument` instances — never
  as raw dicts. Serialisation/deserialisation is an implementation detail
  of each backend.
* Writes are always atomic (temp-file-then-rename or transaction commit).
  Partial writes are not possible from the caller's perspective.
* ``query`` returns a ``QueryResult`` so callers can inspect total count
  separately from the page slice without a second round-trip.
* All backends are thread-safe for concurrent reads; write serialisation
  is the caller's responsibility for ``InMemoryStorage`` and
  ``JSONLStorage`` (SQLiteStorage uses WAL and is safer for concurrent
  writes).

Python compatibility:

Python 3.8-3.15. Only stdlib used: ``json``, ``sqlite3``, ``threading``,
``pathlib``, ``dataclasses``.
"""  # noqa: D205, D400

from __future__ import annotations

import abc
import contextlib
import json
import logging
import os
import pathlib
import sqlite3
import threading
from dataclasses import dataclass, field  # noqa: F401
from typing import Any, Dict, Iterator, List, Optional, Sequence  # noqa: F401

from .._atomic import atomic_write_path
from .._schema import (  # noqa: F401
    ChunkingStrategy,
    CorpusDocument,
    SectionType,
    SourceType,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InMemoryStorage",
    "JSONLStorage",
    "QueryResult",
    "SQLiteStorage",
    "StorageBase",
    "StorageQuery",
]


# ===========================================================================
# Query / result containers
# ===========================================================================


@dataclass(frozen=True)
class StorageQuery:
    """
    Query parameters for :meth:`StorageBase.query`.

    Parameters
    ----------
    input_path : str or None, optional
        Filter to documents from this source file. Default: ``None`` (all).
    source_type : str or None, optional
        Filter to documents with this ``source_type`` value. Default: ``None``.
    language : str or None, optional
        Filter by ISO 639-1 language code. Default: ``None``.
    section_type : str or None, optional
        Filter by ``SectionType`` value string. Default: ``None``.
    collection_id : str or None, optional
        Filter by corpus collection identifier. Default: ``None``.
    full_text : str or None, optional
        Full-text search string. Supported by ``SQLiteStorage`` (FTS5)
        only; ignored by ``InMemoryStorage`` and ``JSONLStorage``.
        Default: ``None``.
    limit : int, optional
        Maximum number of results to return. Default: ``100``.
    offset : int, optional
        Zero-based result offset for pagination. Default: ``0``.
    """

    input_path: str | None = None
    source_type: str | None = None
    language: str | None = None
    section_type: str | None = None
    collection_id: str | None = None
    full_text: str | None = None
    limit: int = 100
    offset: int = 0


@dataclass(frozen=True)
class QueryResult:
    """
    Result container returned by :meth:`StorageBase.query`.

    Parameters
    ----------
    documents : list[CorpusDocument]
        Page of results matching the query.
    total : int
        Total number of matching documents (before ``limit``/``offset``).
    query : StorageQuery
        The query that produced this result (for traceability).
    """

    documents: list[CorpusDocument]
    total: int
    query: StorageQuery

    @property
    def has_more(self) -> bool:
        """Return ``True`` if there are more pages beyond this one."""
        return self.query.offset + len(self.documents) < self.total


# ===========================================================================
# Abstract base
# ===========================================================================


class StorageBase(abc.ABC):
    """
    Abstract base class for all corpus storage backends.

    All implementations must be safe to construct and use without holding
    any external resource until the first ``save``/``get`` call.

    See Also
    --------
    InMemoryStorage : Dict-backed, testing only.
    JSONLStorage    : Flat JSONL file, zero dependencies.
    SQLiteStorage   : SQLite with FTS5, no external dependencies.
    """

    @abc.abstractmethod
    def save(self, doc: CorpusDocument) -> None:
        """
        Persist a single document.

        Parameters
        ----------
        doc : CorpusDocument
            Document to store. Must be validated before calling.
        """

    @abc.abstractmethod
    def save_batch(self, docs: Sequence[CorpusDocument]) -> None:
        """
        Persist a batch of documents atomically.

        Parameters
        ----------
        docs : sequence of CorpusDocument
            Documents to store. May be empty (no-op).
        """

    @abc.abstractmethod
    def get(self, doc_id: str) -> CorpusDocument | None:
        """
        Retrieve a document by its identifier.

        Parameters
        ----------
        doc_id : str
            The ``CorpusDocument.doc_id`` to look up.

        Returns
        -------
        CorpusDocument or None
            The stored document, or ``None`` if not found.
        """

    @abc.abstractmethod
    def query(self, q: StorageQuery) -> QueryResult:
        """
        Retrieve documents matching the query parameters.

        Parameters
        ----------
        q : StorageQuery
            Query specification.

        Returns
        -------
        QueryResult
        """

    def count(self) -> int:
        """
        Return the total number of stored documents.

        Default implementation uses a ``StorageQuery`` with no filters.
        Override for backends that can compute this more efficiently.

        Returns
        -------
        int
        """
        return self.query(StorageQuery(limit=0, offset=0)).total

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.__class__.__name__}()"


# ===========================================================================
# Helper: doc → dict / dict → doc
# ===========================================================================


def _doc_to_dict(doc: CorpusDocument) -> dict[str, Any]:
    """Serialise ``doc`` to a JSON-safe dict (no embedding)."""
    return doc.to_dict(include_embedding=False)


def _dict_to_doc(data: dict[str, Any]) -> CorpusDocument:
    """Deserialise a dict produced by ``_doc_to_dict``."""
    return CorpusDocument.from_dict(data)


def _matches_query(doc: CorpusDocument, q: StorageQuery) -> bool:
    """Return ``True`` if ``doc`` satisfies the non-full-text filters."""
    if q.input_path and doc.input_path != q.input_path:
        return False
    if q.source_type and doc.source_type.value != q.source_type:
        return False
    if q.language and doc.language != q.language:
        return False
    if q.section_type and doc.section_type.value != q.section_type:
        return False
    if q.collection_id and doc.collection_id != q.collection_id:  # noqa: SIM103
        return False
    return True


# ===========================================================================
# InMemoryStorage
# ===========================================================================


class InMemoryStorage(StorageBase):
    """
    Thread-safe in-memory dict store.

    Stores documents as :class:`~scikitplot.corpus._schema.CorpusDocument`
    objects directly in a ``dict`` keyed by ``doc_id``. Insertion order is
    preserved (Python 3.7+ dict guarantee).

    .. warning::
       This backend is intended for testing and prototyping only. Data is
       lost when the process exits.

    Parameters
    ----------
    None

    Examples
    --------
    >>> store = InMemoryStorage()
    >>> store.save(doc)
    >>> store.count()
    1
    >>> store.get(doc.doc_id).text == doc.text
    True
    """

    def __init__(self) -> None:
        self._store: dict[str, CorpusDocument] = {}
        self._lock: threading.Lock = threading.Lock()

    def save(self, doc: CorpusDocument) -> None:  # noqa: D417
        """
        Store ``doc`` by ``doc_id``. Overwrites if already present.

        Parameters
        ----------
        doc : CorpusDocument
        """
        if not hasattr(doc, "doc_id"):
            raise TypeError(
                "InMemoryStorage.save: object must have attribute 'doc_id', "
                f"got {type(doc).__name__}."
            )
        with self._lock:
            self._store[doc.doc_id] = doc

    def save_batch(self, docs: Sequence[CorpusDocument]) -> None:  # noqa: D417
        """
        Store a batch of documents atomically (single lock acquisition).

        Parameters
        ----------
        docs : sequence of CorpusDocument
        """
        if not docs:
            return
        with self._lock:
            for doc in docs:
                self._store[doc.doc_id] = doc

    def get(self, doc_id: str) -> CorpusDocument | None:  # noqa: D417
        """
        Return the document with the given ``doc_id``, or ``None``.

        Parameters
        ----------
        doc_id : str
        """
        with self._lock:
            return self._store.get(doc_id)

    def query(self, q: StorageQuery) -> QueryResult:  # noqa: D417
        """
        Filter documents by the query parameters.

        Full-text search (``q.full_text``) is not supported — the
        ``full_text`` field is ignored silently.

        Parameters
        ----------
        q : StorageQuery

        Returns
        -------
        QueryResult
        """
        with self._lock:
            docs = list(self._store.values())

        matching = [d for d in docs if _matches_query(d, q)]
        total = len(matching)
        page = matching[q.offset : q.offset + q.limit] if q.limit > 0 else []
        return QueryResult(documents=page, total=total, query=q)

    def count(self) -> int:
        """Return total stored document count in O(1)."""
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        """Remove all documents from the store."""
        with self._lock:
            self._store.clear()

    def __repr__(self) -> str:  # noqa: D105
        return f"InMemoryStorage(n_docs={len(self._store)})"


# ===========================================================================
# JSONLStorage
# ===========================================================================


class JSONLStorage(StorageBase):
    """
    Append-friendly JSONL (newline-delimited JSON) flat-file store.

    Documents are written one JSON object per line. On construction the
    file is read into an in-memory index keyed by ``doc_id`` for O(1)
    ``get`` performance. Writes append to the file and update the index.

    .. warning::
       ``save_batch`` writes all documents atomically to a temporary file
       then renames it over the original. This reorders existing documents
       on disk (puts new documents at the end). Concurrent writers would
       corrupt the file; use one writer at a time.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the ``.jsonl`` file. Created if absent.

    Examples
    --------
    >>> store = JSONLStorage(Path("corpus.jsonl"))
    >>> store.save(doc)
    >>> store.count()
    1
    """

    def __init__(self, path: pathlib.Path | str) -> None:
        self._path = pathlib.Path(path)
        self._lock: threading.Lock = threading.Lock()
        self._index: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read the JSONL file into the in-memory index."""
        if not self._path.exists():
            return
        n = 0
        errors = 0
        with self._path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()  # noqa: PLW2901
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._index[data["doc_id"]] = data
                    n += 1
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(
                        "JSONLStorage: skipping malformed line %d in %s: %s.",
                        lineno,
                        self._path,
                        exc,
                    )
                    errors += 1
        logger.info(
            "JSONLStorage: loaded %d documents from %s (%d errors).",
            n,
            self._path,
            errors,
        )

    def _rewrite(self, index: dict[str, Any] | None = None) -> None:
        """Atomically rewrite the JSONL file from ``index`` (default: current).

        Passing an explicit ``index`` enables copy-on-write saves: the caller
        writes the *next* generation to disk and only commits it to
        ``self._index`` after the atomic replace succeeds (CORPUS-STO-002).
        """
        source = self._index if index is None else index

        def _write(tmp: pathlib.Path) -> None:
            with tmp.open("w", encoding="utf-8") as fh:
                for data in source.values():
                    fh.write(json.dumps(data, ensure_ascii=False))
                    fh.write("\n")

        atomic_write_path(self._path, _write, suffix=".jsonl")

    # ------------------------------------------------------------------
    # StorageBase contract
    # ------------------------------------------------------------------

    def save(self, doc: CorpusDocument) -> None:  # noqa: D417
        """
        Append or update a document.

        If the ``doc_id`` already exists, the file is rewritten (update
        semantics). If new, the doc is appended (O(1) write).

        Parameters
        ----------
        doc : CorpusDocument
        """
        data = _doc_to_dict(doc)
        with self._lock:
            if doc.doc_id in self._index:
                # Update rewrites the whole file. Copy-on-write: build the next
                # generation, publish it atomically, and only then swap it in.
                # A failed write leaves both memory and disk at the previous
                # generation (CORPUS-STO-002) — no memory-ahead-of-disk.
                new_index = dict(self._index)
                new_index[doc.doc_id] = data
                self._rewrite(new_index)
                self._index = new_index
            else:
                # Append fast-path (O(1)): write + fsync to disk FIRST, update
                # memory only after. On failure the doc never enters memory, so
                # a query can never return a document that is not durable. A
                # partial trailing line is tolerated by ``_load`` on reload.
                self._path.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(data, ensure_ascii=False) + "\n"
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                    os.fsync(fh.fileno())
                self._index[doc.doc_id] = data

    def save_batch(self, docs: Sequence[CorpusDocument]) -> None:  # noqa: D417
        """
        Save a batch, rewriting the file atomically once.

        Parameters
        ----------
        docs : sequence of CorpusDocument
        """
        if not docs:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # Copy-on-write: serialize into the next generation, publish it
            # atomically, and commit to memory only on success. A failed write
            # (or an unserializable doc) leaves memory and disk unchanged
            # (CORPUS-STO-002).
            new_index = dict(self._index)
            for doc in docs:
                new_index[doc.doc_id] = _doc_to_dict(doc)
            self._rewrite(new_index)
            self._index = new_index

    def get(self, doc_id: str) -> CorpusDocument | None:  # noqa: D417
        """
        Retrieve a document by ``doc_id``.

        Parameters
        ----------
        doc_id : str
        """
        with self._lock:
            data = self._index.get(doc_id)
        if data is None:
            return None
        return _dict_to_doc(data)

    def query(self, q: StorageQuery) -> QueryResult:  # noqa: D417
        """
        Filter documents by query parameters.

        Full-text search is not supported and is ignored.

        Parameters
        ----------
        q : StorageQuery
        """
        with self._lock:
            all_dicts = list(self._index.values())

        matching: list[CorpusDocument] = []
        for data in all_dicts:
            try:
                doc = _dict_to_doc(data)
                if _matches_query(doc, q):
                    matching.append(doc)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "JSONLStorage.query: skipping malformed record %r: %s.",
                    data.get("doc_id", "?"),
                    exc,
                )

        total = len(matching)
        page = matching[q.offset : q.offset + q.limit] if q.limit > 0 else []
        return QueryResult(documents=page, total=total, query=q)

    def count(self) -> int:
        """Return total stored document count in O(1)."""
        with self._lock:
            return len(self._index)

    def __repr__(self) -> str:  # noqa: D105
        return f"JSONLStorage(path={self._path!r}, n_docs={len(self._index)})"


# ===========================================================================
# SQLiteStorage
# ===========================================================================

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS corpus_documents (
    doc_id          TEXT PRIMARY KEY,
    input_path     TEXT NOT NULL,
    source_type     TEXT,
    section_type    TEXT,
    language        TEXT,
    collection_id   TEXT,
    chunk_index     INTEGER,
    char_start      INTEGER,
    char_end        INTEGER,
    json_data       TEXT NOT NULL
);
"""

_CREATE_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS corpus_fts
USING fts5(doc_id UNINDEXED, text);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_input_path ON corpus_documents(input_path);",
    "CREATE INDEX IF NOT EXISTS idx_source_type ON corpus_documents(source_type);",
    "CREATE INDEX IF NOT EXISTS idx_language    ON corpus_documents(language);",
    "CREATE INDEX IF NOT EXISTS idx_collection  ON corpus_documents(collection_id);",
]


_UPSERT_DOC_SQL = """
INSERT OR REPLACE INTO corpus_documents
    (doc_id, input_path, source_type, section_type,
     language, collection_id, chunk_index, char_start,
     char_end, json_data)
VALUES
    (:doc_id, :input_path, :source_type, :section_type,
     :language, :collection_id, :chunk_index, :char_start,
     :char_end, :json_data)
"""

_UPSERT_FTS_SQL = "INSERT OR REPLACE INTO corpus_fts(doc_id, text) VALUES (?, ?);"


class SQLiteStorage(StorageBase):
    r"""
    SQLite-backed corpus store with FTS5 full-text search.

    Uses stdlib ``sqlite3`` — no external dependencies. Full-text search
    is available via FTS5 (``StorageQuery.full_text``).

    The database uses WAL (Write-Ahead Logging) mode for better concurrent
    read throughput. A single connection is held per ``SQLiteStorage``
    instance; use separate instances for multiple threads if needed.

    Parameters
    ----------
    db_path : pathlib.Path or str
        Path to the SQLite database file. Created if absent.
        Pass ``\":memory:\"`` for a purely in-memory database.

    Examples
    --------
    >>> store = SQLiteStorage(Path("corpus.db"))
    >>> store.save_batch(docs)
    >>> result = store.query(StorageQuery(source_type="book", limit=10))
    >>> result.total
    487
    """

    def __init__(self, db_path: pathlib.Path | str = ":memory:") -> None:
        self._db_path: str = str(db_path) if str(db_path) != ":memory:" else ":memory:"
        if self._db_path != ":memory:":
            pathlib.Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = self._connect()
        self._lock: threading.Lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open and configure the SQLite connection."""
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage transactions explicitly
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._lock, self._conn:
            self._conn.execute(_CREATE_TABLE_SQL)
            self._conn.execute(_CREATE_FTS_SQL)
            for idx_sql in _CREATE_INDEXES_SQL:
                self._conn.execute(idx_sql)

    def _doc_to_row(self, doc: CorpusDocument) -> dict[str, Any]:
        """Convert doc to a sqlite row dict."""
        return {
            "doc_id": doc.doc_id,
            "input_path": doc.input_path,
            "source_type": doc.source_type.value,
            "section_type": doc.section_type.value,
            "language": doc.language,
            "collection_id": doc.collection_id,
            "chunk_index": doc.chunk_index,
            "char_start": doc.char_start,
            "char_end": doc.char_end,
            "json_data": json.dumps(_doc_to_dict(doc), ensure_ascii=False),
        }

    def _row_to_doc(self, row: sqlite3.Row) -> CorpusDocument:
        """Deserialise a sqlite row back to a CorpusDocument."""
        return _dict_to_doc(json.loads(row["json_data"]))

    def _upsert(self, doc: CorpusDocument, cursor: sqlite3.Cursor) -> None:
        """INSERT OR REPLACE a single document row (documents + FTS)."""
        cursor.execute(_UPSERT_DOC_SQL, self._doc_to_row(doc))
        cursor.execute(_UPSERT_FTS_SQL, (doc.doc_id, doc.text))

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit all-or-nothing transaction over the autocommit connection.

        The connection uses ``isolation_level=None`` (autocommit), so a plain
        ``with self._conn:`` block does NOT open a transaction — each statement
        commits immediately. This issues ``BEGIN IMMEDIATE`` up front (reserving
        the write lock so the batch cannot be interleaved), commits on success,
        and rolls back on any exception. A batch is therefore truly atomic and
        the ``corpus_documents`` / ``corpus_fts`` tables never diverge
        (CORPUS-STO-001).
        """
        conn = self._conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    # ------------------------------------------------------------------
    # StorageBase contract
    # ------------------------------------------------------------------

    def save(self, doc: CorpusDocument) -> None:  # noqa: D417
        """
        Persist a single document (upsert by ``doc_id``).

        Parameters
        ----------
        doc : CorpusDocument
        """
        # Serialize BEFORE mutating so a bad doc raises before any write.
        row = self._doc_to_row(doc)
        with self._lock, self._transaction() as conn:
            conn.execute(_UPSERT_DOC_SQL, row)
            conn.execute(_UPSERT_FTS_SQL, (doc.doc_id, doc.text))

    def save_batch(self, docs: Sequence[CorpusDocument]) -> None:  # noqa: D417
        """
        Persist a batch of documents atomically (all-or-nothing).

        All rows are serialized and validated before any write; the documents
        and FTS rows are then committed inside one explicit transaction, so a
        failure at any point leaves the database exactly as before the batch
        (CORPUS-STO-001).

        Parameters
        ----------
        docs : sequence of CorpusDocument
        """
        if not docs:
            return
        # 1. Serialize/validate every row up front. An unserializable doc raises
        #    here, before the transaction opens, so nothing is persisted.
        doc_rows = [self._doc_to_row(doc) for doc in docs]
        fts_rows = [(doc.doc_id, doc.text) for doc in docs]
        # 2. One real transaction covering documents + FTS together.
        with self._lock, self._transaction() as conn:
            conn.executemany(_UPSERT_DOC_SQL, doc_rows)
            conn.executemany(_UPSERT_FTS_SQL, fts_rows)
        logger.info("SQLiteStorage: saved %d documents.", len(docs))

    def get(self, doc_id: str) -> CorpusDocument | None:  # noqa: D417
        """
        Retrieve a document by ``doc_id``.

        Parameters
        ----------
        doc_id : str
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT json_data FROM corpus_documents WHERE doc_id = ?;",
                (doc_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return _dict_to_doc(json.loads(row["json_data"]))

    def query(self, q: StorageQuery) -> QueryResult:  # noqa: D417
        """
        Query documents with optional full-text search (FTS5).

        Parameters
        ----------
        q : StorageQuery
        """
        conditions: list[str] = []
        params: list[Any] = []

        if q.full_text:
            # FTS5 subquery
            conditions.append(
                "doc_id IN (SELECT doc_id FROM corpus_fts WHERE text MATCH ?)"
            )
            params.append(q.full_text)
        if q.input_path:
            conditions.append("input_path = ?")
            params.append(q.input_path)
        if q.source_type:
            conditions.append("source_type = ?")
            params.append(q.source_type)
        if q.language:
            conditions.append("language = ?")
            params.append(q.language)
        if q.section_type:
            conditions.append("section_type = ?")
            params.append(q.section_type)
        if q.collection_id:
            conditions.append("collection_id = ?")
            params.append(q.collection_id)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM corpus_documents {where};"  # noqa: S608
        select_sql = (
            f"SELECT json_data FROM corpus_documents {where} "  # noqa: S608
            f"ORDER BY rowid LIMIT ? OFFSET ?;"
        )

        with self._lock:
            total = self._conn.execute(count_sql, params).fetchone()[0]
            if q.limit > 0:
                rows = self._conn.execute(
                    select_sql, [*params, q.limit, q.offset]
                ).fetchall()
            else:
                rows = []

        docs: list[CorpusDocument] = []
        for row in rows:
            try:
                docs.append(_dict_to_doc(json.loads(row["json_data"])))
            except Exception as exc:  # noqa: BLE001
                logger.warning("SQLiteStorage.query: skipping malformed row: %s.", exc)

        return QueryResult(documents=docs, total=total, query=q)

    def count(self) -> int:
        """Return total stored document count via fast SQL COUNT."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM corpus_documents;"
            ).fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def __repr__(self) -> str:  # noqa: D105
        return f"SQLiteStorage(db_path={self._db_path!r})"
