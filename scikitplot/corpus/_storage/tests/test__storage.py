"""
tests/test__storage.py
========================
Tests for scikitplot.corpus._storage.
All three backends are covered: InMemoryStorage, JSONLStorage, SQLiteStorage.
"""
from __future__ import annotations

import pathlib
import tempfile

import pytest

from .._storage import (
    InMemoryStorage,
    JSONLStorage,
    QueryResult,
    SQLiteStorage,
    StorageQuery,
)
from ..._schema import CorpusDocument


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(
    source_file: str = "f.txt",
    chunk_index: int = 0,
    text: str = "Hello world test.",
    language: str | None = "en",
    source_type: str = "book",
    collection_id: str | None = "col1",
) -> CorpusDocument:
    return CorpusDocument.create(
        source_file=source_file,
        chunk_index=chunk_index,
        text=text,
        language=language,
        collection_id=collection_id,
    )


# ===========================================================================
# StorageQuery
# ===========================================================================


class TestStorageQuery:
    def test_defaults(self) -> None:
        q = StorageQuery()
        assert q.limit == 100
        assert q.offset == 0
        assert q.source_file is None

    def test_frozen(self) -> None:
        q = StorageQuery()
        with pytest.raises((AttributeError, TypeError)):
            q.limit = 50  # type: ignore[misc]


# ===========================================================================
# QueryResult
# ===========================================================================


class TestQueryResult:
    def test_has_more_true(self) -> None:
        doc = _make_doc()
        q = StorageQuery(limit=1, offset=0)
        r = QueryResult(documents=[doc], total=5, query=q)
        assert r.has_more is True

    def test_has_more_false(self) -> None:
        doc = _make_doc()
        q = StorageQuery(limit=10, offset=0)
        r = QueryResult(documents=[doc], total=1, query=q)
        assert r.has_more is False


# ===========================================================================
# Shared backend test matrix
# ===========================================================================


def _all_backends(tmp_path: pathlib.Path) -> list:
    return [
        InMemoryStorage(),
        JSONLStorage(tmp_path / "corpus.jsonl"),
        SQLiteStorage(":memory:"),
    ]


class BackendContract:
    """Mixin: run the same tests against every backend."""

    store: object  # subclasses set this

    def _fresh_doc(self, idx: int = 0) -> CorpusDocument:
        return _make_doc(chunk_index=idx, text=f"Document number {idx} here.")

    def test_save_and_get(self, tmp_path: pathlib.Path) -> None:
        doc = self._fresh_doc()
        self.store.save(doc)
        retrieved = self.store.get(doc.doc_id)
        assert retrieved is not None
        assert retrieved.doc_id == doc.doc_id
        assert retrieved.text == doc.text

    def test_get_missing_returns_none(self) -> None:
        assert self.store.get("nonexistent_id_xyz") is None

    def test_save_batch(self) -> None:
        docs = [self._fresh_doc(i) for i in range(5)]
        self.store.save_batch(docs)
        assert self.store.count() == 5

    def test_save_batch_empty_is_noop(self) -> None:
        self.store.save_batch([])
        assert self.store.count() == 0

    def test_count_after_saves(self) -> None:
        for i in range(3):
            self.store.save(self._fresh_doc(i))
        assert self.store.count() == 3

    def test_query_all(self) -> None:
        docs = [self._fresh_doc(i) for i in range(4)]
        self.store.save_batch(docs)
        result = self.store.query(StorageQuery(limit=10))
        assert result.total == 4
        assert len(result.documents) == 4

    def test_query_limit_and_offset(self) -> None:
        docs = [self._fresh_doc(i) for i in range(6)]
        self.store.save_batch(docs)
        result = self.store.query(StorageQuery(limit=2, offset=2))
        assert len(result.documents) == 2
        assert result.total == 6

    def test_query_by_source_file(self) -> None:
        doc_a = CorpusDocument.create("a.txt", 0, "text from a file here.")
        doc_b = CorpusDocument.create("b.txt", 0, "text from b file here.")
        self.store.save_batch([doc_a, doc_b])
        result = self.store.query(StorageQuery(source_file="a.txt", limit=10))
        assert result.total == 1
        assert result.documents[0].source_file == "a.txt"

    def test_query_by_language(self) -> None:
        doc_en = CorpusDocument.create("f.txt", 0, "English text here now.", language="en")
        doc_de = CorpusDocument.create("f.txt", 1, "Deutsches Beispiel hier.", language="de")
        self.store.save_batch([doc_en, doc_de])
        result = self.store.query(StorageQuery(language="en", limit=10))
        assert result.total == 1
        assert result.documents[0].language == "en"

    def test_upsert_overwrites(self) -> None:
        doc = self._fresh_doc()
        self.store.save(doc)
        updated = doc.replace(normalized_text="updated content here.")
        self.store.save(updated)
        retrieved = self.store.get(doc.doc_id)
        assert retrieved is not None
        assert retrieved.normalized_text == "updated content here."
        # Count should still be 1
        assert self.store.count() == 1

    def test_empty_query_returns_zero_total(self) -> None:
        result = self.store.query(StorageQuery(limit=0))
        assert result.total == 0


class TestInMemoryStorage(BackendContract):
    def setup_method(self) -> None:
        self.store = InMemoryStorage()

    def test_clear(self) -> None:
        self.store.save_batch([self._fresh_doc(i) for i in range(3)])
        self.store.clear()
        assert self.store.count() == 0

    def test_repr(self) -> None:
        assert "InMemoryStorage" in repr(self.store)

    def test_save_non_corpus_doc_raises(self) -> None:
        with pytest.raises(TypeError):
            self.store.save("not a document")  # type: ignore[arg-type]


class TestJSONLStorage(BackendContract):
    # @classmethod
    # def setup_class(cls):
    #     tmp_path = pytest.ensuretemp("jsonl")  # returns a pathlib.Path
    #     cls.path = tmp_path / "corpus.jsonl"
    #     cls.store = JSONLStorage(cls.path)

    def setup_method(self, method) -> None:
        tmp_dir = pathlib.Path(tempfile.mkdtemp())
        self.path = tmp_dir / "corpus.jsonl"
        self.store = JSONLStorage(self.path)

    def test_persists_across_reload(self) -> None:
        doc = self._fresh_doc()
        self.store.save(doc)
        # Re-open from same file
        store2 = JSONLStorage(self.path)
        retrieved = store2.get(doc.doc_id)
        assert retrieved is not None
        assert retrieved.doc_id == doc.doc_id

    def test_repr(self) -> None:
        assert "JSONLStorage" in repr(self.store)


class TestSQLiteStorage(BackendContract):
    def setup_method(self) -> None:
        self.store = SQLiteStorage(":memory:")

    def test_full_text_search(self) -> None:
        doc_a = CorpusDocument.create("f.txt", 0, "Python programming language guide.")
        doc_b = CorpusDocument.create("f.txt", 1, "Completely unrelated sports news.")
        self.store.save_batch([doc_a, doc_b])
        result = self.store.query(StorageQuery(full_text="Python", limit=10))
        assert result.total == 1
        assert "Python" in result.documents[0].text

    def test_file_persistence(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "test.db"
        store1 = SQLiteStorage(path)
        doc = self._fresh_doc()
        store1.save(doc)
        store1.close()
        store2 = SQLiteStorage(path)
        retrieved = store2.get(doc.doc_id)
        assert retrieved is not None
        assert retrieved.doc_id == doc.doc_id
        store2.close()

    def test_repr(self) -> None:
        assert "SQLiteStorage" in repr(self.store)
