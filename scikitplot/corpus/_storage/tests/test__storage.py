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
    _FTS5_UNREPRESENTABLE,
    _fts5_any,
    _fts5_phrase,
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

    def teardown_method(self) -> None:
        # R3-8: every store is closed; an unclosed SQLite connection is a
        # ResourceWarning on Python 3.13.
        self.store.close()

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

    def setup_method(self) -> None:
        self._opened: list[SQLiteStorage] = []

    def teardown_method(self) -> None:
        for store in self._opened:
            store.close()

    def _store(self) -> SQLiteStorage:
        store = SQLiteStorage(":memory:")
        self._opened.append(store)
        return store

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

#: Characters no FTS5 string can hold (finding CX-05): NUL, and lone
#: surrogates at both ends of the high and the low range.
_UNREPRESENTABLE = ["\x00", "\ud800", "\udbff", "\udc00", "\udfff"]
_UNREPRESENTABLE_IDS = ["U+0000", "U+D800", "U+DBFF", "U+DC00", "U+DFFF"]

#: Where the character sits: alone, repeated, before a word, inside a
#: compound, after an identifier, and between quoted words.
_UNREPRESENTABLE_QUERIES = [
    "{c}",
    "{c}{c}",
    "{c}null",
    "roc{c}auc",
    "roc_auc_score(){c}",
    'dtype("float64"){c}"x"',
]

#: Characters an FTS5 string can hold, which must reach SQLite unchanged: the
#: neighbours of the surrogate range, controls, separators, a BOM,
#: noncharacters and the astral planes.
_KEPT = [
    chr(c)
    for c in (
        *(0x01, 0x1F, 0x7F, 0x80, 0x9F, 0x2028, 0xD7FF, 0xE000),
        *(0xFDD0, 0xFEFF, 0xFFFF, 0x1F600, 0x20000, 0x10FFFF),
    )
]

#: Parts of queries: words, FTS5 syntax, quotes, the spaced characters (a
#: surrogate pair written as two code points among them) and kept ones. Every
#: part alone and every ordered pair is checked, so each character meets
#: every neighbour.
_PAIR_PARTS = (
    *("roc", "auc", "score", "C++", "AND", "OR", "NOT", "NEAR", "*", "^", ":"),
    *('"', "(", ")", "_", ".", "-", " ", "  "),
    *(chr(c) for c in (0x00, 0xD800, 0xDBFF, 0xDC00, 0xDFFF, 0xD83D, 0xDE00)),
    *(chr(c) for c in (0x01, 0x1F, 0x2028, 0xFEFF, 0x1F600)),
)


class TestFreeTextSearch:
    def teardown_method(self) -> None:
        self.store.close()

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

    # -- CX-05: characters no FTS5 string can hold ---------------------------
    # SQLite raised "unterminated string" for a NUL or a lone surrogate on both
    # paths. Each must now behave exactly as a space does.

    @pytest.mark.parametrize("char", _UNREPRESENTABLE, ids=_UNREPRESENTABLE_IDS)
    @pytest.mark.parametrize("template", _UNREPRESENTABLE_QUERIES)
    def test_fts5_terms_treat_the_character_as_a_space(self, char, template) -> None:
        query = template.format(c=char)
        spaced = template.format(c=" ")
        assert _fts5_phrase(query) == _fts5_phrase(spaced)
        assert _fts5_any(query) == _fts5_any(spaced)

    @pytest.mark.parametrize("char", _UNREPRESENTABLE, ids=_UNREPRESENTABLE_IDS)
    @pytest.mark.parametrize("template", _UNREPRESENTABLE_QUERIES)
    def test_search_matches_the_spaced_query(self, char, template) -> None:
        query = template.format(c=char)
        spaced = template.format(c=" ")
        ranked = [(doc.doc_id, score) for doc, score in self.store.search_text(query, 5)]
        assert ranked == [(doc.doc_id, score) for doc, score in self.store.search_text(spaced, 5)]
        found = self.store.query(StorageQuery(full_text=query, limit=10)).documents
        expected = self.store.query(StorageQuery(full_text=spaced, limit=10)).documents
        assert [d.doc_id for d in found] == [d.doc_id for d in expected]

    def test_a_separated_identifier_finds_the_documents_of_its_words(self) -> None:
        found = {doc.doc_id for doc, _ in self.store.search_text("roc\x00auc\udfffscore", 5)}
        assert {self.docs[0].doc_id, self.docs[4].doc_id} <= found

    def test_the_class_is_exactly_nul_and_the_surrogates(self) -> None:
        every = "".join(map(chr, range(0x110000)))
        spaced = [ord(c) for c in _FTS5_UNREPRESENTABLE.findall(every)]
        assert spaced == [0x0000, *range(0xD800, 0xE000)]

    @pytest.mark.parametrize("char", _KEPT, ids=[f"U+{ord(c):04X}" for c in _KEPT])
    def test_characters_fts5_can_hold_are_kept(self, char) -> None:
        assert _fts5_phrase(f"a{char}b") == f'"a{char}b"'
        self.store.search_text(f"roc{char}auc", 5)
        self.store.query(StorageQuery(full_text=f"roc{char}auc", limit=10))

    def test_a_surrogate_pair_written_as_two_code_points_is_two_spaces(self) -> None:
        pair = chr(0xD83D) + chr(0xDE00)
        assert _fts5_any(f"roc{pair}auc") == _fts5_any("roc  auc")
        assert _fts5_phrase(f"roc{chr(0x1F600)}auc") == f'"roc{chr(0x1F600)}auc"'

    def test_one_connection_survives_every_failure_mode_in_turn(self) -> None:
        """CPython 3.11+ reported a surrogate after a failed NUL query as the NUL error."""
        for query in ["\x00", "\ud800", "roc\x00auc", "roc\udfffauc"]:
            self.store.search_text(query, 3)
            self.store.query(StorageQuery(full_text=query, limit=3))
        ranked = self.store.search_text("roc_auc_score()", 3)
        assert ranked[0][0].doc_id == self.docs[0].doc_id

    @pytest.mark.parametrize("char", ["\x00", "\ud800"], ids=["U+0000", "U+D800"])
    def test_long_runs_are_one_separator(self, char) -> None:
        run = char * 100_000
        assert _fts5_any(run) is None
        assert self.store.search_text(run, 3) == []
        ranked = self.store.search_text(f"{run}roc_auc_score(){run}", 3)
        assert ranked == self.store.search_text("roc_auc_score()", 3)

    def test_every_part_and_pair_of_parts_matches_the_spaced_query(self) -> None:
        queries = list(_PAIR_PARTS) + [a + b for a in _PAIR_PARTS for b in _PAIR_PARTS]
        for query in queries:
            spaced = _FTS5_UNREPRESENTABLE.sub(" ", query)
            assert _fts5_any(query) == _fts5_any(spaced), ascii(query)
            assert _fts5_phrase(query) == _fts5_phrase(spaced), ascii(query)
            got = [(d.doc_id, s) for d, s in self.store.search_text(query, 5)]
            assert got == [(d.doc_id, s) for d, s in self.store.search_text(spaced, 5)]
            found = self.store.query(StorageQuery(full_text=query, limit=5)).documents
            want = self.store.query(StorageQuery(full_text=spaced, limit=5)).documents
            assert [d.doc_id for d in found] == [d.doc_id for d in want], ascii(query)


# ===========================================================================
# CX-06: a lone surrogate is refused by the UTF-8 backends, before any write
# ===========================================================================

#: Where a lone surrogate can sit in a document, as a replace() and the path
#: the refusal must name.
_SURROGATE_PLACES = [
    ({"text": "roc" + chr(0xDFFF) + "auc"}, "$.text"),
    ({"raw_text": "raw" + chr(0xD800)}, "$.raw_text"),
    ({"metadata": {"k": "a" + chr(0xDBFF) + "b"}}, "$.metadata.k"),
    ({"metadata": {"k" + chr(0xDC00): "v"}}, "$.metadata.'k\\udc00' (a key)"),
    ({"tokens": ["ok", "t" + chr(0xD800)]}, "$.tokens[1]"),
]


def _utf8_store(kind, tmp_path):
    if kind == "sqlite":
        return SQLiteStorage(tmp_path / "s.db")
    return JSONLStorage(tmp_path / "s.jsonl")


def _snapshot(store, kind, tmp_path):
    path = tmp_path / ("s.db" if kind == "sqlite" else "s.jsonl")
    return store.count(), path.read_bytes() if kind == "jsonl" and path.exists() else None


class TestLoneSurrogateDocuments:
    @pytest.mark.parametrize("kind", ["sqlite", "jsonl"])
    @pytest.mark.parametrize(("change", "where"), _SURROGATE_PLACES)
    def test_save_refuses_and_names_the_place(self, kind, change, where, tmp_path) -> None:
        store = _utf8_store(kind, tmp_path)
        try:
            store.save(_make_doc(chunk_index=0))
            before = _snapshot(store, kind, tmp_path)
            bad = _make_doc(chunk_index=1).replace(**change)
            backend = "SQLiteStorage" if kind == "sqlite" else "JSONLStorage"
            message = (
                f"{backend}: document {bad.doc_id!r}: {where} holds a lone surrogate,"
                f" which {backend} cannot encode as UTF-8"
            )
            with pytest.raises(ValueError, match="holds a lone surrogate") as caught:
                store.save(bad)
            assert str(caught.value) == message
            assert _snapshot(store, kind, tmp_path) == before
            store.save(_make_doc(chunk_index=2))
            assert store.count() == before[0] + 1
        finally:
            store.close()

    @pytest.mark.parametrize("kind", ["sqlite", "jsonl"])
    def test_a_batch_with_one_such_document_writes_nothing(self, kind, tmp_path) -> None:
        store = _utf8_store(kind, tmp_path)
        try:
            store.save(_make_doc(chunk_index=0))
            before = _snapshot(store, kind, tmp_path)
            bad = _make_doc(chunk_index=2).replace(text="x" + chr(0xD800))
            batch = [_make_doc(chunk_index=1), bad, _make_doc(chunk_index=3)]
            with pytest.raises(ValueError, match=f"document {bad.doc_id!r}: \\$\\.text"):
                store.save_batch(batch)
            assert _snapshot(store, kind, tmp_path) == before
        finally:
            store.close()

    def test_in_memory_storage_keeps_such_a_document(self) -> None:
        store = InMemoryStorage()
        doc = _make_doc().replace(metadata={"k": "a" + chr(0xD800) + "b"})
        store.save(doc)
        assert store.get(doc.doc_id).metadata == {"k": "a" + chr(0xD800) + "b"}

    @pytest.mark.parametrize("kind", ["sqlite", "jsonl"])
    def test_nul_is_still_stored(self, kind, tmp_path) -> None:
        store = _utf8_store(kind, tmp_path)
        try:
            doc = _make_doc(text="roc\x00auc here")
            store.save(doc)
            assert store.get(doc.doc_id).text == "roc\x00auc here"
        finally:
            store.close()
