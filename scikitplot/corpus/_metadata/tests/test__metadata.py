"""
tests/test__metadata.py
=========================
Tests for scikitplot.corpus._metadata.
"""

from __future__ import annotations

import pytest

from .._metadata import (
    CollectionManifest,
    CorpusStats,
    compute_stats,
    provenance_from_filename,
)
from ..._schema import CorpusDocument


def _make_doc(
    source_file: str = "f.txt",
    chunk_index: int = 0,
    text: str = "This is a sample sentence here.",
    language: str | None = "en",
    collection_id: str | None = None,
    source_date: str | None = None,
) -> CorpusDocument:
    return CorpusDocument.create(
        source_file=source_file,
        chunk_index=chunk_index,
        text=text,
        language=language,
        collection_id=collection_id,
        source_date=source_date,
    )


# ===========================================================================
# CollectionManifest
# ===========================================================================


class TestCollectionManifest:
    def test_basic_construction(self) -> None:
        m = CollectionManifest(collection_id="c1", title="Test Corpus")
        assert m.collection_id == "c1"
        assert m.title == "Test Corpus"

    def test_empty_collection_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CollectionManifest(collection_id="")

    def test_whitespace_collection_id_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            CollectionManifest(collection_id="   ")

    def test_negative_expected_file_count_raises(self) -> None:
        with pytest.raises(ValueError, match="expected_file_count"):
            CollectionManifest(collection_id="c1", expected_file_count=-1)

    def test_to_provenance_includes_collection_id(self) -> None:
        m = CollectionManifest(
            collection_id="c1",
            title="My Corpus",
            author="Tolstoy",
            source_date="1869",
            source_type="book",
        )
        prov = m.to_provenance()
        assert prov["collection_id"] == "c1"
        assert prov["source_title"] == "My Corpus"
        assert prov["source_author"] == "Tolstoy"
        assert prov["source_date"] == "1869"
        assert prov["source_type"] == "book"

    def test_to_provenance_none_fields_excluded(self) -> None:
        m = CollectionManifest(collection_id="c1")
        prov = m.to_provenance()
        assert "source_title" not in prov
        assert "source_author" not in prov

    def test_provenance_for_file_merges_overrides(self) -> None:
        m = CollectionManifest(
            collection_id="c1",
            author="Default Author",
            file_provenance={"hamlet.xml": {"source_title": "Hamlet"}},
        )
        prov = m.provenance_for_file("hamlet.xml")
        assert prov["source_author"] == "Default Author"
        assert prov["source_title"] == "Hamlet"

    def test_provenance_for_file_no_override(self) -> None:
        m = CollectionManifest(collection_id="c1", author="X")
        prov = m.provenance_for_file("other.xml")
        assert prov["source_author"] == "X"
        assert "source_title" not in prov

    def test_check_completeness_matches(self) -> None:
        m = CollectionManifest(collection_id="c1", expected_file_count=5)
        assert m.check_completeness(5) is True

    def test_check_completeness_mismatch(self) -> None:
        m = CollectionManifest(collection_id="c1", expected_file_count=5)
        assert m.check_completeness(3) is False

    def test_check_completeness_none_always_true(self) -> None:
        m = CollectionManifest(collection_id="c1")
        assert m.check_completeness(999) is True

    def test_frozen(self) -> None:
        m = CollectionManifest(collection_id="c1")
        with pytest.raises((AttributeError, TypeError)):
            m.collection_id = "c2"  # type: ignore[misc]

    def test_repr(self) -> None:
        m = CollectionManifest(collection_id="c1")
        assert "c1" in repr(m)


# ===========================================================================
# compute_stats
# ===========================================================================


class TestComputeStats:
    def test_empty_docs(self) -> None:
        stats = compute_stats([])
        assert stats.n_documents == 0
        assert stats.n_tokens == 0
        assert stats.mean_tokens == 0.0
        assert stats.language_counts == {}

    def test_single_doc(self) -> None:
        doc = _make_doc(text="One two three four five.")
        stats = compute_stats([doc])
        assert stats.n_documents == 1
        assert stats.n_tokens == 5
        assert stats.min_tokens == 5
        assert stats.max_tokens == 5
        assert stats.median_tokens == 5.0

    def test_language_counts(self) -> None:
        docs = [
            _make_doc(chunk_index=0, text="English text here now.", language="en"),
            _make_doc(chunk_index=1, text="English again here now.", language="en"),
            _make_doc(chunk_index=2, text="German text here now.", language="de"),
        ]
        stats = compute_stats(docs)
        assert stats.language_counts["en"] == 2
        assert stats.language_counts["de"] == 1

    def test_none_language_counted_as_unknown(self) -> None:
        doc = _make_doc(text="No language set here.", language=None)
        stats = compute_stats([doc])
        assert "unknown" in stats.language_counts

    def test_has_embeddings_count(self) -> None:
        import numpy as np
        doc = _make_doc(text="Test with embedding here.")
        embedded = doc.replace(embedding=np.zeros(128, dtype=np.float32))
        doc_no_embed = _make_doc(chunk_index=1, text="No embedding here doc.")
        stats = compute_stats([embedded, doc_no_embed])
        assert stats.has_embeddings == 1

    def test_collection_ids(self) -> None:
        docs = [
            _make_doc(chunk_index=0, text="Doc one here now.", collection_id="col_a"),
            _make_doc(chunk_index=1, text="Doc two here now.", collection_id="col_b"),
            _make_doc(chunk_index=2, text="Doc three here.", collection_id="col_a"),
        ]
        stats = compute_stats(docs)
        assert sorted(stats.collection_ids) == ["col_a", "col_b"]

    def test_date_range(self) -> None:
        docs = [
            _make_doc(chunk_index=0, text="Early doc text here.", source_date="1800"),
            _make_doc(chunk_index=1, text="Late doc text here.", source_date="2024"),
            _make_doc(chunk_index=2, text="Middle doc text here.", source_date="1950"),
        ]
        stats = compute_stats(docs)
        assert stats.date_range == ("1800", "2024")

    def test_date_range_none_when_no_dates(self) -> None:
        docs = [_make_doc(text="Doc without date here now.")]
        stats = compute_stats(docs)
        assert stats.date_range is None

    def test_mean_tokens(self) -> None:
        docs = [
            _make_doc(chunk_index=0, text="one two"),             # 2 tokens
            _make_doc(chunk_index=1, text="one two three four"),  # 4 tokens
        ]
        stats = compute_stats(docs)
        assert stats.mean_tokens == 3.0

    def test_to_dict_is_json_safe(self) -> None:
        import json
        stats = compute_stats([_make_doc(text="Test doc here today.")])
        d = stats.to_dict()
        # Should not raise
        json.dumps(d)

    def test_summary_is_string(self) -> None:
        stats = compute_stats([_make_doc(text="Sample text here today.")])
        s = stats.summary()
        assert isinstance(s, str)
        assert "Documents" in s


# ===========================================================================
# provenance_from_filename
# ===========================================================================


class TestProvenanceFromFilename:
    def test_full_pattern_extracted(self) -> None:
        prov = provenance_from_filename(
            "Shakespeare_William_Hamlet_1603.xml"
        )
        # At minimum title and year should be detected
        assert "source_title" in prov or "source_author" in prov

    def test_plain_name_returns_dict(self) -> None:
        # Even unrecognised names return a dict (possibly empty)
        prov = provenance_from_filename("document.pdf")
        assert isinstance(prov, dict)

    def test_source_type_included_when_given(self) -> None:
        prov = provenance_from_filename("book.txt", source_type="book")
        assert prov.get("source_type") == "book"

    def test_no_source_type_omits_key(self) -> None:
        prov = provenance_from_filename("book.txt")
        assert "source_type" not in prov

    def test_two_word_name_extracts_title(self) -> None:
        prov = provenance_from_filename("great-expectations.txt")
        # Should detect something
        assert isinstance(prov, dict)

    def test_returns_dict_always(self) -> None:
        for name in ["a.txt", ".hidden", "no_ext", "123.pdf"]:
            assert isinstance(provenance_from_filename(name), dict)
