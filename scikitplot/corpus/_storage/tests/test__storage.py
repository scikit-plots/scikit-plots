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

from .._storage import (  # noqa: F401
    InMemoryStorage,
    JSONLStorage,
    QueryResult,
    SQLiteStorage,
    StorageQuery,
)
from ..._schema import CorpusDocument, SourceType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_doc(
    input_path: str = "f.txt",
    chunk_index: int = 0,
    text: str = "Hello world test.",
    language: str | None = "en",
    source_type: str = "book",
    collection_id: str | None = "col1",
) -> CorpusDocument:
    return CorpusDocument.create(
        input_path=input_path,
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
        assert q.input_path is None
        assert q.full_text is None

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

    def test_query_by_input_path(self) -> None:
        doc_a = CorpusDocument.create("a.txt", 0, "text from a file here.")
        doc_b = CorpusDocument.create("b.txt", 0, "text from b file here.")
        self.store.save_batch([doc_a, doc_b])
        result = self.store.query(StorageQuery(input_path="a.txt", limit=10))
        assert result.total == 1
        assert result.documents[0].input_path == "a.txt"

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

    def test_full_text_search_orders_by_bm25(self) -> None:
        weak = CorpusDocument.create(
            "weak.txt", 0, "python appears once beside unrelated filler words."
        )
        strong = CorpusDocument.create(
            "strong.txt", 0, "python python python python focused python guide."
        )
        self.store.save_batch([weak, strong])

        result = self.store.query(StorageQuery(full_text="python", limit=10))

        assert result.total == 2
        assert [doc.doc_id for doc in result.documents] == [strong.doc_id, weak.doc_id]

    def test_full_text_update_replaces_old_terms(self) -> None:
        doc_id = "fixed-sqlite-document"
        original = CorpusDocument.create(
            "f.txt", 0, "obsoletealpha token", doc_id=doc_id
        )
        replacement = CorpusDocument.create(
            "f.txt", 0, "replacementbeta token", doc_id=doc_id
        )

        self.store.save(original)
        self.store.save(replacement)

        assert self.store.count() == 1
        assert self.store.query(StorageQuery(full_text="obsoletealpha")).total == 0
        result = self.store.query(StorageQuery(full_text="replacementbeta"))
        assert result.total == 1
        assert result.documents[0].text == replacement.text
        assert self.store._conn.execute(
            "SELECT COUNT(*) FROM corpus_fts WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0] == 1

    def test_batch_duplicate_doc_id_uses_last_version(self) -> None:
        doc_id = "duplicate-batch-document"
        first = CorpusDocument.create(
            "f.txt", 0, "firstversiontoken", doc_id=doc_id
        )
        final = CorpusDocument.create(
            "f.txt", 0, "finalversiontoken", doc_id=doc_id
        )

        self.store.save_batch([first, final])

        assert self.store.count() == 1
        stored = self.store.get(doc_id)
        assert stored is not None
        assert stored.text == final.text
        assert self.store.query(StorageQuery(full_text="firstversiontoken")).total == 0
        assert self.store.query(StorageQuery(full_text="finalversiontoken")).total == 1
        assert self.store._conn.execute(
            "SELECT COUNT(*) FROM corpus_fts WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0] == 1

    def test_legacy_duplicate_fts_rows_deduplicated_on_reopen(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "legacy-duplicates.db"
        doc_id = "legacy-duplicate-document"
        store = SQLiteStorage(path)
        store.save(
            CorpusDocument.create(
                "legacy.txt", 0, "canonical document text", doc_id=doc_id
            )
        )

        # Reproduce the legacy INSERT OR REPLACE behavior. FTS5 does not enforce
        # uniqueness for an UNINDEXED doc_id, so this creates a second row.
        store._conn.execute(
            "INSERT OR REPLACE INTO corpus_fts(doc_id, text) VALUES (?, ?)",
            (doc_id, "newestlegacytoken"),
        )
        assert store._conn.execute(
            "SELECT COUNT(*) FROM corpus_fts WHERE doc_id = ?", (doc_id,)
        ).fetchone()[0] == 2
        store.close()

        reopened = SQLiteStorage(path)
        try:
            assert reopened._conn.execute(
                "SELECT COUNT(*) FROM corpus_fts WHERE doc_id = ?", (doc_id,)
            ).fetchone()[0] == 1
            assert reopened.query(StorageQuery(full_text="canonical")).total == 0
            result = reopened.query(StorageQuery(full_text="newestlegacytoken"))
            assert result.total == 1
            assert result.documents[0].doc_id == doc_id
        finally:
            reopened.close()

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


class TestSQLiteAtomicity:
    """CORPUS-STO-001: save_batch / save are all-or-nothing (docs + FTS)."""

    def _store(self) -> SQLiteStorage:
        return SQLiteStorage(":memory:")

    @staticmethod
    def _doc(idx: int, text: str, **md) -> CorpusDocument:
        return CorpusDocument.create(
            "f.txt", idx, text, source_type=SourceType.BOOK,
            metadata=(md or None),
        )

    @staticmethod
    def _count(store: SQLiteStorage) -> int:
        return store._conn.execute(
            "SELECT COUNT(*) FROM corpus_documents"
        ).fetchone()[0]

    @staticmethod
    def _fts_count(store: SQLiteStorage) -> int:
        return store._conn.execute("SELECT COUNT(*) FROM corpus_fts").fetchone()[0]

    def test_batch_with_unserializable_doc_persists_nothing(self) -> None:
        """The reproduction: one serializable + one unserializable doc."""
        store = self._store()
        good = self._doc(0, "hello world")
        bad = self._doc(1, "bye world", bad=object())  # unserializable metadata
        with pytest.raises(TypeError):
            store.save_batch([good, bad])
        assert self._count(store) == 0          # no persisted prefix
        assert store.get(good.doc_id) is None
        assert self._fts_count(store) == 0      # docs/FTS never diverge

    def test_successful_batch_is_consistent(self) -> None:
        store = self._store()
        docs = [self._doc(i, f"doc {i} lorem ipsum") for i in range(5)]
        store.save_batch(docs)
        assert self._count(store) == self._fts_count(store) == 5
        assert all(store.get(d.doc_id) is not None for d in docs)

    def test_db_failure_mid_batch_rolls_back_docs_and_fts(self, monkeypatch) -> None:
        store = self._store()
        from .. import _storage as _stg  # the module defining save_batch

        # Break the FTS insert so document upserts and FTS deletions execute
        # first, then fail inside the same transaction. Everything must roll back.
        monkeypatch.setattr(
            _stg,
            "_INSERT_FTS_SQL",
            "INSERT INTO __no_such_table__(doc_id, text) VALUES (?, ?);",
        )
        with pytest.raises(Exception):  # noqa: B017 - injected sqlite error
            store.save_batch([self._doc(0, "a"), self._doc(1, "b")])
        assert self._count(store) == 0          # docs rolled back
        assert self._fts_count(store) == 0      # FTS rolled back

    def test_single_save_with_bad_doc_persists_nothing(self) -> None:
        store = self._store()
        with pytest.raises(TypeError):
            store.save(self._doc(0, "x", bad=object()))
        assert self._count(store) == 0 and self._fts_count(store) == 0

    def test_single_save_keeps_docs_and_fts_consistent(self) -> None:
        store = self._store()
        store.save(self._doc(0, "consistent text"))
        assert self._count(store) == 1 and self._fts_count(store) == 1


class TestJSONLDivergence:
    """CORPUS-STO-002: JSONL memory and disk never diverge on a failed write."""

    @staticmethod
    def _doc(idx, text, doc_id=None, **md):
        return CorpusDocument.create(
            "f.txt", idx, text, source_type=SourceType.BOOK,
            doc_id=doc_id, metadata=(md or None),
        )

    @staticmethod
    def _boom(*_a, **_k):
        raise OSError("injected disk failure")

    def test_update_write_failure_keeps_previous_generation(
        self, tmp_path, monkeypatch
    ) -> None:
        from .. import _storage as _stg

        path = tmp_path / "c.jsonl"
        store = JSONLStorage(path)
        did = "fixedid00000001"
        store.save(self._doc(0, "v1 text", doc_id=did))
        monkeypatch.setattr(_stg, "atomic_write_path", self._boom)
        with pytest.raises(OSError):
            store.save(self._doc(0, "v2 UPDATED", doc_id=did))  # update -> rewrite
        assert store.get(did).text == "v1 text"                  # memory unchanged
        assert JSONLStorage(path).get(did).text == "v1 text"     # disk unchanged

    def test_batch_write_failure_commits_nothing(self, tmp_path, monkeypatch) -> None:
        from .. import _storage as _stg

        path = tmp_path / "c.jsonl"
        store = JSONLStorage(path)
        store.save(self._doc(0, "orig", doc_id="keep000000000001"))
        new = [self._doc(1, "new one"), self._doc(2, "new two")]
        monkeypatch.setattr(_stg, "atomic_write_path", self._boom)
        with pytest.raises(OSError):
            store.save_batch(new)
        assert all(store.get(d.doc_id) is None for d in new)     # not in memory
        assert JSONLStorage(path).get("keep000000000001") is not None

    def test_append_fsync_failure_not_committed(self, tmp_path, monkeypatch) -> None:
        from .. import _storage as _stg

        path = tmp_path / "c.jsonl"
        store = JSONLStorage(path)
        d = self._doc(5, "appended text")
        monkeypatch.setattr(_stg.os, "fsync", self._boom)
        with pytest.raises(OSError):
            store.save(d)
        assert store.get(d.doc_id) is None  # never entered memory (no memory-ahead)

    def test_batch_serialization_failure_leaves_prev_generation(self, tmp_path) -> None:
        path = tmp_path / "c.jsonl"
        store = JSONLStorage(path)
        store.save(self._doc(0, "keep", doc_id="keep000000000001"))
        with pytest.raises(TypeError):
            store.save_batch([self._doc(1, "ok"), self._doc(2, "bad", x=object())])
        assert store.get(self._doc(1, "ok").doc_id) is None
        assert store.get("keep000000000001") is not None


# ===========================================================================
# Free text is never FTS5 syntax (finding CX-01)
# ===========================================================================

_TECH_DOCS = [
    "Use roc_auc_score() from sklearn.metrics to compute the area under the ROC curve.",
    "The C++ extension annoylib is built with meson.",
    "ValueError: Input contains NaN, infinity or a value too large for dtype('float64').",
    "Set random-state=42 for reproducible splits.",
    "roc and auc and score are mentioned separately, sklearn and metrics too.",
]

#: Queries whose best answer is decided by the query itself, not by corpus
#: statistics: each names words only one document holds together.
_TECH_QUERIES = [
    ("roc_auc_score()", 0),
    ("C++ extension", 1),
    ('dtype("float64")', 2),
    ("ValueError: Input contains NaN", 2),
    ("random-state", 3),
    ("NEAR(roc curve)", 0),
    ("area under the curve?", 0),
]

#: Queries that used to be FTS5 syntax errors or empty results. They must run
#: and find something; *which* document ranks first depends on IDF over the
#: corpus (in five documents, "is" is rarer than "roc").
_FORMERLY_BROKEN = [
    "sklearn.metrics:roc_auc_score",
    "what is roc auc",
    "AND",
    "NOT sklearn",
    "roc*",
]


class TestFreeTextSearch:
    def setup_method(self) -> None:
        self.store = SQLiteStorage(":memory:")
        self.docs = [CorpusDocument.create("t.md", i, t) for i, t in enumerate(_TECH_DOCS)]
        self.store.save_batch(self.docs)

    @pytest.mark.parametrize(("query", "best"), _TECH_QUERIES)
    def test_search_text_ranks_the_answer_first(self, query, best) -> None:
        ranked = self.store.search_text(query, 3)
        assert ranked and ranked[0][0].doc_id == self.docs[best].doc_id
        scores = [score for _, score in ranked]
        assert scores == sorted(scores, reverse=True) and scores[0] > 0

    def test_an_exact_identifier_outranks_its_scattered_parts(self) -> None:
        ranked = [doc.doc_id for doc, _ in self.store.search_text("roc_auc_score()", 5)]
        assert ranked.index(self.docs[0].doc_id) < ranked.index(self.docs[4].doc_id)

    @pytest.mark.parametrize("query", _FORMERLY_BROKEN)
    def test_formerly_broken_queries_run_and_find(self, query) -> None:
        assert self.store.search_text(query, 5)

    @pytest.mark.parametrize(
        "query", [q for q, _ in _TECH_QUERIES] + _FORMERLY_BROKEN + ['"', "()", "*", "-x"]
    )
    def test_full_text_never_raises(self, query) -> None:
        self.store.query(StorageQuery(full_text=query, limit=10))

    @pytest.mark.parametrize("query", [q for q, _ in _TECH_QUERIES])
    def test_full_text_finds_a_literal_occurrence(self, query) -> None:
        """A document holding the query verbatim matches it as a phrase."""
        doc = CorpusDocument.create("lit.md", 0, f"prefix {query} suffix")
        self.store.save(doc)
        found = self.store.query(StorageQuery(full_text=query, limit=50)).documents
        assert doc.doc_id in {d.doc_id for d in found}

    @pytest.mark.parametrize("query", ["", "   ", "()", "?!"])
    def test_no_word_means_no_result(self, query) -> None:
        assert self.store.search_text(query, 3) == []

    @pytest.mark.parametrize(
        ("kwargs", "error"),
        [({"text": 1}, TypeError), ({"text": "x", "limit": 0}, ValueError), ({"text": "x", "limit": True}, TypeError)],
    )
    def test_arguments_are_validated(self, kwargs, error) -> None:
        with pytest.raises(error):
            self.store.search_text(**kwargs)

    def test_collection_filter(self) -> None:
        other = CorpusDocument.create("o.md", 0, "roc_auc_score() elsewhere", collection_id="other")
        self.store.save(other)
        ranked = self.store.search_text("roc_auc_score()", 10, collection_id="other")
        assert [doc.doc_id for doc, _ in ranked] == [other.doc_id]

    def test_without_fts5_it_says_so(self) -> None:
        self.store._has_fts5 = False
        with pytest.raises(RuntimeError, match="FTS5"):
            self.store.search_text("x", 1)
