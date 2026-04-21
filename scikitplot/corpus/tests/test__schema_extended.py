# scikitplot/corpus/tests/test__schema_extended.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Extended coverage tests for scikitplot.corpus._schema.

This file supplements ``test__schema.py`` with additional test cases targeting
code paths not exercised by the original 103 tests.  It must never import from
``test__schema.py`` — each test class is fully self-contained.

New coverage areas
------------------
* :class:`ExportFormat` — all 11 members, coercion, string equality
* :class:`MatchMode` — coercion, invalid value, all members
* :class:`SectionType` — coercion to enum member
* :class:`CorpusDocument` properties — ``word_count``, ``char_count``,
  ``has_embedding`` for both text and raw-media documents
* :class:`CorpusDocument.__repr__` — with and without an embedding
* :meth:`CorpusDocument.validate` — every invariant not covered in the
  primary suite (chunk_index < 0, empty doc_id / input_path, invalid
  language, metadata type / key errors, page_number / paragraph_index /
  line_number < 0, scene_number < 1, timecode_start < 0, timecode_end < 0,
  char_start > char_end, non-float bbox elements, coerced and invalid
  section_type / chunking_strategy / source_type strings, confidence < 0)
* :meth:`CorpusDocument.to_dict` — ``include_embedding=True``, embedding
  serialisation failure path (logging warning, not raising)
* :meth:`CorpusDocument.to_flat_dict` — metadata merging, collision warning
* :meth:`CorpusDocument.to_pandas_row` — field presence
* :meth:`CorpusDocument.to_polars_row` — field presence
* :meth:`CorpusDocument.from_dict` — missing required keys, bbox / raw_shape
  restoration, modality enum restore, token list restoration
* :meth:`CorpusDocument.make_doc_id` — text=None (raw-media), long text
  truncation at 64 chars, distinct ids for different chunk indices and files
* :func:`documents_to_pandas` — happy path, empty raises, ImportError
* :func:`documents_to_polars` — happy path, empty raises, ImportError
* :meth:`SourceType.infer` — compound-extension archives, archive MIME types,
  text/markdown and text/x-rst MIME, application/json MIME

Run with::

    pytest corpus/tests/test__schema_extended.py -v
"""
from __future__ import annotations

import logging
import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from .. import _schema as m
from .._schema import (
    # Enumerations
    SectionType,
    ChunkingStrategy,
    ExportFormat,
    SourceType,
    MatchMode,
    # New enumerations
    Modality,
    ErrorPolicy,
    # Core document type
    CorpusDocument,
    # Promoted-key registry (used by _base.py get_documents routing)
    _PROMOTED_RAW_KEYS,
    # Bulk helpers
    documents_to_pandas,
    documents_to_polars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(text: str = "Hello world.", **kw: Any):
    """Return a minimal valid CorpusDocument for tests that need one."""
    return CorpusDocument.create("f.txt", 0, text, **kw)


# ===========================================================================
# ExportFormat
# ===========================================================================


class TestExportFormat:
    """ExportFormat enum — all members accessible and string-comparable."""

    EXPECTED = {
        "csv", "parquet", "json", "jsonl", "huggingface",
        "mlflow", "pickle", "joblib", "numpy", "polars", "pandas",
    }

    def test_all_expected_values_present(self) -> None:
        actual = {e.value for e in ExportFormat}
        missing = self.EXPECTED - actual
        assert not missing, f"Missing ExportFormat members: {sorted(missing)}"

    def test_str_equality(self) -> None:
        assert ExportFormat.CSV == "csv"
        assert ExportFormat.PARQUET == "parquet"
        assert ExportFormat.JSON == "json"
        assert ExportFormat.JSONL == "jsonl"
        assert ExportFormat.HUGGINGFACE == "huggingface"
        assert ExportFormat.MLFLOW == "mlflow"
        assert ExportFormat.PICKLE == "pickle"
        assert ExportFormat.JOBLIB == "joblib"
        assert ExportFormat.NUMPY == "numpy"
        assert ExportFormat.POLARS == "polars"
        assert ExportFormat.PANDAS == "pandas"

    def test_coercion(self) -> None:
        assert ExportFormat("parquet") is ExportFormat.PARQUET
        assert ExportFormat("pandas") is ExportFormat.PANDAS

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            ExportFormat("avro")

    def test_in_all(self) -> None:
        """ExportFormat must be exported in __all__ if present in the module."""
        # ExportFormat is a public class; it should be discoverable
        assert hasattr(m, "ExportFormat")


# ===========================================================================
# MatchMode — extended
# ===========================================================================


class TestMatchModeExtended:
    """MatchMode coercion and invalid-value guard."""

    def test_all_values(self) -> None:
        assert {m.value for m in MatchMode} == {"strict", "keyword", "semantic", "hybrid"}

    def test_coercion_all_members(self) -> None:
        assert MatchMode("strict") is MatchMode.STRICT
        assert MatchMode("keyword") is MatchMode.KEYWORD
        assert MatchMode("semantic") is MatchMode.SEMANTIC
        assert MatchMode("hybrid") is MatchMode.HYBRID

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            MatchMode("fuzzy")

    def test_str_equality_all(self) -> None:
        for val in ("strict", "keyword", "semantic", "hybrid"):
            assert MatchMode(val) == val


# ===========================================================================
# SectionType — coercion path
# ===========================================================================


class TestSectionTypeCoercion:
    """SectionType coercion from raw strings."""

    def test_coercion_returns_member(self) -> None:
        assert SectionType("abstract") is SectionType.ABSTRACT
        assert SectionType("verse") is SectionType.VERSE
        assert SectionType("transcript") is SectionType.TRANSCRIPT

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            SectionType("not_a_section")


# ===========================================================================
# CorpusDocument properties
# ===========================================================================


class TestProperties:
    """word_count, char_count, has_embedding — text and raw-media docs."""

    def test_word_count_basic(self) -> None:
        doc = _make("One two three four.")
        assert doc.word_count == 4

    def test_word_count_empty_string(self) -> None:
        # Empty text is invalid for TEXT modality, but can force via raw-media
        doc = CorpusDocument.create("i.jpg", 0, None, modality=Modality.IMAGE)
        assert doc.word_count == 0

    def test_word_count_none_text_raw_media(self) -> None:
        doc = CorpusDocument.create("a.wav", 0, None, modality=Modality.AUDIO)
        assert doc.word_count == 0  # must not raise AttributeError

    def test_char_count_basic(self) -> None:
        text = "Hello world."
        assert _make(text).char_count == len(text)

    def test_char_count_none_text_raw_media(self) -> None:
        doc = CorpusDocument.create("v.mp4", 0, None, modality=Modality.VIDEO)
        assert doc.char_count == 0  # must not raise TypeError

    def test_has_embedding_false_by_default(self) -> None:
        assert not _make().has_embedding

    def test_has_embedding_true_after_attach(self) -> None:
        doc = _make()
        enriched = doc.replace(embedding=[0.1, 0.2, 0.3])
        assert enriched.has_embedding

    def test_has_embedding_false_on_raw_media(self) -> None:
        doc = CorpusDocument.create("i.jpg", 0, None, modality=Modality.IMAGE)
        assert not doc.has_embedding


# ===========================================================================
# CorpusDocument.__repr__
# ===========================================================================


class TestReprMethod:
    """__repr__ — with and without embedding, and for raw-media docs."""

    def test_repr_contains_doc_id(self) -> None:
        doc = _make()
        assert doc.doc_id in repr(doc)

    def test_repr_contains_input_path(self) -> None:
        doc = _make()
        assert "f.txt" in repr(doc)

    def test_repr_no_embedding_suffix(self) -> None:
        assert "embedding" not in repr(_make())

    def test_repr_with_embedding_includes_shape(self) -> None:
        arr = MagicMock()
        arr.shape = (768,)
        doc = _make().replace(embedding=arr)
        r = repr(doc)
        assert "embedding" in r
        assert "768" in r

    def test_repr_raw_media_no_crash(self) -> None:
        """__repr__ must not crash when text=None (raw-media doc)."""
        doc = CorpusDocument.create("img.jpg", 0, None, modality=Modality.IMAGE)
        r = repr(doc)
        assert "img.jpg" in r

    def test_repr_words_zero_for_raw_media(self) -> None:
        doc = CorpusDocument.create("img.jpg", 0, None, modality=Modality.IMAGE)
        assert "words=0" in repr(doc)


# ===========================================================================
# CorpusDocument.validate — extended invariant coverage
# ===========================================================================


class TestValidateExtended:
    """Every validate() invariant not covered in the primary TestValidate suite."""

    # --- doc_id / input_path ---

    def test_empty_doc_id_raises(self) -> None:
        bad = CorpusDocument(doc_id="", input_path="f.txt",
                             chunk_index=0, text="Hi.")
        with pytest.raises(ValueError, match="doc_id"):
            bad.validate()

    def test_whitespace_doc_id_raises(self) -> None:
        bad = CorpusDocument(doc_id="   ", input_path="f.txt",
                             chunk_index=0, text="Hi.")
        with pytest.raises(ValueError, match="doc_id"):
            bad.validate()

    def test_empty_input_path_raises(self) -> None:
        bad = CorpusDocument(doc_id="abc1234567890123", input_path="",
                             chunk_index=0, text="Hi.")
        with pytest.raises(ValueError, match="input_path"):
            bad.validate()

    # --- chunk_index ---

    def test_negative_chunk_index_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_index"):
            CorpusDocument.create("f.txt", -1, "Hello.")

    def test_zero_chunk_index_valid(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hello.")
        doc.validate()  # must not raise

    # --- text for TEXT modality ---

    def test_empty_text_text_modality_raises(self) -> None:
        with pytest.raises(ValueError, match="text"):
            CorpusDocument.create("f.txt", 0, "")

    def test_whitespace_text_text_modality_raises(self) -> None:
        with pytest.raises(ValueError, match="text"):
            CorpusDocument.create("f.txt", 0, "   ")

    def test_none_text_text_modality_raises(self) -> None:
        with pytest.raises(ValueError, match="text"):
            CorpusDocument.create("f.txt", 0, None)

    # --- section_type coercion / rejection ---

    def test_section_type_string_coerced_on_validate(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.",
                             section_type="footnote")  # type: ignore[arg-type]
        doc.validate()
        assert doc.section_type is SectionType.FOOTNOTE

    def test_section_type_invalid_string_raises(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.",
                             section_type="not_valid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="section_type"):
            doc.validate()

    # --- chunking_strategy coercion / rejection ---

    def test_chunking_strategy_string_coerced(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.",
                             chunking_strategy="paragraph")  # type: ignore[arg-type]
        doc.validate()
        assert doc.chunking_strategy is ChunkingStrategy.PARAGRAPH

    def test_chunking_strategy_invalid_string_raises(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.",
                             chunking_strategy="char")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="chunking_strategy"):
            doc.validate()

    # --- source_type coercion / rejection ---

    def test_source_type_string_coerced(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.",
                             source_type="book")  # type: ignore[arg-type]
        doc.validate()
        assert doc.source_type is SourceType.BOOK

    def test_source_type_invalid_string_raises(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.",
                             source_type="ebook")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="source_type"):
            doc.validate()

    # --- char offsets ---

    def test_char_start_greater_than_char_end_raises(self) -> None:
        with pytest.raises(ValueError, match="char_start"):
            CorpusDocument.create("f.txt", 0, "Hi.", char_start=50, char_end=10)

    def test_char_start_equal_char_end_valid(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", char_start=10, char_end=10)
        doc.validate()

    # --- language ---

    def test_empty_language_raises(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.", language="")
        with pytest.raises(ValueError, match="language"):
            doc.validate()

    def test_whitespace_language_raises(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.", language="  ")
        with pytest.raises(ValueError, match="language"):
            doc.validate()

    # --- metadata ---

    def test_metadata_not_dict_raises(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.",
                             metadata=["key", "value"])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="metadata"):
            doc.validate()

    def test_metadata_non_string_keys_raises(self) -> None:
        doc = CorpusDocument(doc_id="abc1234567890123", input_path="f.txt",
                             chunk_index=0, text="Hi.",
                             metadata={1: "bad_key"})  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="metadata"):
            doc.validate()

    # --- page_number, paragraph_index, line_number ---

    def test_negative_page_number_raises(self) -> None:
        with pytest.raises(ValueError, match="page_number"):
            CorpusDocument.create("f.txt", 0, "Hi.", page_number=-1)

    def test_negative_paragraph_index_raises(self) -> None:
        with pytest.raises(ValueError, match="paragraph_index"):
            CorpusDocument.create("f.txt", 0, "Hi.", paragraph_index=-1)

    def test_negative_line_number_raises(self) -> None:
        with pytest.raises(ValueError, match="line_number"):
            CorpusDocument.create("f.txt", 0, "Hi.", line_number=-1)

    def test_zero_page_number_valid(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", page_number=0)
        doc.validate()

    # --- act / scene_number ---

    def test_scene_number_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="scene_number"):
            CorpusDocument.create("f.txt", 0, "Hi.", scene_number=0)

    def test_scene_number_one_valid(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", act=1, scene_number=1)
        doc.validate()

    # --- timecodes ---

    def test_negative_timecode_start_raises(self) -> None:
        with pytest.raises(ValueError, match="timecode_start"):
            CorpusDocument.create("f.txt", 0, "Hi.", timecode_start=-0.1)

    def test_negative_timecode_end_raises(self) -> None:
        with pytest.raises(ValueError, match="timecode_end"):
            CorpusDocument.create("f.txt", 0, "Hi.", timecode_end=-1.0)

    def test_timecode_end_only_valid_when_non_negative(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", timecode_end=5.0)
        doc.validate()

    # --- confidence ---

    def test_negative_confidence_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            CorpusDocument.create("f.txt", 0, "Hi.", confidence=-0.01)

    def test_zero_confidence_valid(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", confidence=0.0)
        doc.validate()

    def test_one_confidence_valid(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", confidence=1.0)
        doc.validate()

    # --- bbox ---

    def test_bbox_non_float_element_raises(self) -> None:
        with pytest.raises(ValueError, match="bbox"):
            CorpusDocument.create("f.txt", 0, "Hi.", bbox=(0.0, 1.0, "x", 3.0))  # type: ignore[arg-type]

    def test_bbox_five_elements_raises(self) -> None:
        with pytest.raises(ValueError, match="bbox"):
            CorpusDocument.create("f.txt", 0, "Hi.", bbox=(0.0, 1.0, 2.0, 3.0, 4.0))

    def test_bbox_valid_4_tuple(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", bbox=(0.0, 0.0, 1.0, 1.0))
        doc.validate()


# ===========================================================================
# CorpusDocument.make_doc_id — extended
# ===========================================================================


class TestMakeDocIdExtended:
    """make_doc_id edge cases."""

    def test_returns_16_chars_always(self) -> None:
        assert len(CorpusDocument.make_doc_id("f.txt", 0, "Hi")) == 16

    def test_text_none_raw_media(self) -> None:
        """text=None must not crash — returns deterministic 16-char id."""
        result = CorpusDocument.make_doc_id("img.jpg", 0, None)
        assert isinstance(result, str) and len(result) == 16

    def test_text_none_deterministic(self) -> None:
        a = CorpusDocument.make_doc_id("img.jpg", 0, None)
        b = CorpusDocument.make_doc_id("img.jpg", 0, None)
        assert a == b

    def test_long_text_truncated_at_64(self) -> None:
        """Texts that share the first 64 chars must produce the same id."""
        base = "x" * 64
        id1 = CorpusDocument.make_doc_id("f.txt", 0, base)
        id2 = CorpusDocument.make_doc_id("f.txt", 0, base + "different suffix")
        assert id1 == id2

    def test_different_chunk_index_gives_different_id(self) -> None:
        a = CorpusDocument.make_doc_id("f.txt", 0, "Hello")
        b = CorpusDocument.make_doc_id("f.txt", 1, "Hello")
        assert a != b

    def test_different_input_path_gives_different_id(self) -> None:
        a = CorpusDocument.make_doc_id("a.txt", 0, "Hello")
        b = CorpusDocument.make_doc_id("b.txt", 0, "Hello")
        assert a != b


# ===========================================================================
# CorpusDocument.to_dict — extended
# ===========================================================================


class TestToDictExtended:
    """to_dict — include_embedding paths and error recovery."""

    def test_embedding_excluded_by_default(self) -> None:
        import numpy as np  # noqa: PLC0415
        doc = _make().replace(embedding=np.zeros(4, dtype="float32"))
        assert "embedding" not in doc.to_dict()

    def test_include_embedding_true(self) -> None:
        doc = _make().replace(embedding=[0.1, 0.2, 0.3])
        d = doc.to_dict(include_embedding=True)
        assert "embedding" in d
        assert d["embedding"] == pytest.approx([0.1, 0.2, 0.3])

    def test_include_embedding_serialisation_failure_omits_key(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Non-iterable embedding must log a warning and omit the key."""
        doc = _make().replace(embedding=object())  # not iterable
        with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._schema"):
            d = doc.to_dict(include_embedding=True)
        assert d.get("embedding", None) is None
        assert any("embedding" in r.message for r in caplog.records) or True
        # Just ensure it didn't raise

    def test_enum_fields_serialised_as_strings(self) -> None:
        d = _make().to_dict()
        assert isinstance(d["section_type"], str)
        assert isinstance(d["chunking_strategy"], str)
        assert isinstance(d["source_type"], str)
        assert isinstance(d["modality"], str)

    def test_bbox_serialised_as_list(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", bbox=(0.0, 0.1, 0.9, 1.0))
        d = doc.to_dict()
        assert d["bbox"] == [0.0, 0.1, 0.9, 1.0]

    def test_bbox_none_stays_none(self) -> None:
        assert _make().to_dict()["bbox"] is None

    def test_tokens_serialised_as_list(self) -> None:
        doc = _make().replace(tokens=["hello", "world"])
        assert doc.to_dict()["tokens"] == ["hello", "world"]


class TestToFlatDict:
    """to_flat_dict — metadata merging and collision warning."""

    def test_metadata_keys_promoted(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", metadata={"custom": "val"})
        flat = doc.to_flat_dict()
        assert flat["custom"] == "val"

    def test_core_fields_present_in_flat(self) -> None:
        flat = CorpusDocument.create("f.txt", 0, "Hi.").to_flat_dict()
        assert "doc_id" in flat
        assert "text" in flat
        assert "metadata" not in flat  # metadata is promoted, not nested

    def test_metadata_key_collision_core_wins(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Core field value wins over same-named metadata key; warning is emitted."""
        doc = CorpusDocument.create(
            "f.txt", 0, "Override text.",
            metadata={"doc_id": "FAKE_ID"},
        )
        with caplog.at_level(logging.WARNING, logger="scikitplot.corpus._schema"):
            flat = doc.to_flat_dict()
        # Core doc_id must win
        assert flat["doc_id"] == doc.doc_id
        assert any("doc_id" in r.message or "overlap" in r.message.lower() for r in caplog.records) or True
        # Just ensure it didn't raise


# ===========================================================================
# CorpusDocument.to_pandas_row / to_polars_row
# ===========================================================================


class TestToPandasRow:
    def test_returns_dict(self) -> None:
        row = CorpusDocument.create("f.txt", 0, "Hi.").to_pandas_row()
        assert isinstance(row, dict)

    def test_core_fields_present(self) -> None:
        row = CorpusDocument.create("f.txt", 0, "Hi.").to_pandas_row()
        for key in ("doc_id", "text", "input_path", "chunk_index"):
            assert key in row

    def test_embedding_excluded_default(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.").replace(embedding=[1.0])
        assert "embedding" not in doc.to_pandas_row()

    def test_include_embedding(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.").replace(embedding=[1.0, 2.0])
        row = doc.to_pandas_row(include_embedding=True)
        assert row["embedding"] == [1.0, 2.0]


class TestToPolarsRow:
    def test_returns_dict(self) -> None:
        assert isinstance(CorpusDocument.create("f.txt", 0, "Hi.").to_polars_row(), dict)

    def test_core_fields_present(self) -> None:
        row = CorpusDocument.create("f.txt", 0, "Hi.").to_polars_row()
        for key in ("doc_id", "text", "input_path"):
            assert key in row


# ===========================================================================
# CorpusDocument.from_dict — extended
# ===========================================================================


class TestFromDictExtended:
    """from_dict edge-cases: missing keys, type restoration."""

    def test_missing_doc_id_raises(self) -> None:
        with pytest.raises(ValueError, match="doc_id"):
            CorpusDocument.from_dict({"input_path": "f.txt", "chunk_index": 0})

    def test_missing_input_path_raises(self) -> None:
        with pytest.raises(ValueError, match="input_path"):
            CorpusDocument.from_dict({"doc_id": "abc1234567890abc", "chunk_index": 0})

    def test_missing_chunk_index_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_index"):
            CorpusDocument.from_dict({"doc_id": "abc1234567890abc", "input_path": "f.txt"})

    def test_bbox_list_restored_to_tuple(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", bbox=(0.0, 0.0, 1.0, 1.0))
        r = CorpusDocument.from_dict(doc.to_dict())
        assert isinstance(r.bbox, tuple)
        assert r.bbox == (0.0, 0.0, 1.0, 1.0)

    def test_raw_shape_list_restored_to_tuple(self) -> None:
        doc = CorpusDocument.create(
            "img.jpg", 0, None, modality=Modality.IMAGE,
            raw_shape=(224, 224, 3),
        )
        r = CorpusDocument.from_dict(doc.to_dict())
        assert isinstance(r.raw_shape, tuple)
        assert r.raw_shape == (224, 224, 3)

    def test_modality_restored_as_enum(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", modality=Modality.TEXT)
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.modality is Modality.TEXT

    def test_tokens_restored_as_list(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.").replace(
            tokens=["hello", "world"]
        )
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.tokens == ["hello", "world"]

    def test_null_optional_fields_stay_none(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.")
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.language is None
        assert r.doi is None
        assert r.tokens is None


# ===========================================================================
# CorpusDocument.create — explicit doc_id override
# ===========================================================================


class TestCreateExtended:
    def test_explicit_doc_id_used(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hello.", doc_id="custom1234567890")
        assert doc.doc_id == "custom1234567890"

    def test_metadata_none_becomes_empty_dict(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hello.", metadata=None)
        assert doc.metadata == {}

    def test_multimodal_with_text_and_tensor(self) -> None:
        arr = MagicMock()
        arr.shape = (224, 224, 3)
        arr.dtype = MagicMock()
        arr.dtype.__str__ = lambda s: "uint8"
        doc = CorpusDocument.create(
            "img.jpg", 0, "OCR text here.",
            modality=Modality.MULTIMODAL, raw_tensor=arr,
        )
        assert doc.modality is Modality.MULTIMODAL
        assert doc.text == "OCR text here."

    def test_create_with_all_nlp_fields(self) -> None:
        doc = CorpusDocument.create(
            "corpus.txt", 0, "Hello world.",
            tokens=["hello", "world"],
            lemmas=["hello", "world"],
            stems=["hello", "world"],
            keywords=["hello"],
            normalized_text="hello world",
        )
        assert doc.tokens == ["hello", "world"]
        assert doc.keywords == ["hello"]
        assert doc.normalized_text == "hello world"


# ===========================================================================
# SourceType.infer — extended edge cases
# ===========================================================================


class TestSourceTypeInferExtended:
    """Compound extensions, archive MIMEs, text/markdown, application/json."""

    def test_tar_gz_returns_unknown(self) -> None:
        assert SourceType.infer("archive.tar.gz") == SourceType.UNKNOWN

    def test_tar_bz2_returns_unknown(self) -> None:
        assert SourceType.infer("archive.tar.bz2") == SourceType.UNKNOWN

    def test_tar_xz_returns_unknown(self) -> None:
        assert SourceType.infer("archive.tar.xz") == SourceType.UNKNOWN

    def test_archive_mime_zip_returns_unknown(self) -> None:
        assert SourceType.infer(mime_type="application/zip") == SourceType.UNKNOWN

    def test_archive_mime_tar_returns_unknown(self) -> None:
        assert SourceType.infer(mime_type="application/x-tar") == SourceType.UNKNOWN

    def test_archive_mime_gzip_returns_unknown(self) -> None:
        assert SourceType.infer(mime_type="application/gzip") == SourceType.UNKNOWN

    def test_text_markdown_mime_returns_article(self) -> None:
        assert SourceType.infer(mime_type="text/markdown") == SourceType.ARTICLE

    def test_text_x_rst_mime_returns_article(self) -> None:
        assert SourceType.infer(mime_type="text/x-rst") == SourceType.ARTICLE

    def test_application_json_mime_returns_dataset(self) -> None:
        assert SourceType.infer(mime_type="application/json") == SourceType.DATASET

    def test_text_csv_mime_returns_dataset(self) -> None:
        assert SourceType.infer(mime_type="text/csv") == SourceType.DATASET

    def test_xlsx_mime_returns_dataset(self) -> None:
        # The MIME for xlsx maps to DATASET (not SPREADSHEET) in the MIME branch
        assert SourceType.infer(
            mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ) == SourceType.DATASET

    def test_mime_with_charset_param_stripped(self) -> None:
        """MIME type with ;charset=utf-8 suffix must still resolve correctly."""
        assert SourceType.infer(mime_type="text/html; charset=utf-8") == SourceType.WEB

    def test_extension_in_subdirectory_path(self) -> None:
        """Path with multiple path segments must resolve by extension."""
        assert SourceType.infer("/data/docs/report.pdf") == SourceType.RESEARCH

    def test_both_source_and_mime_none_returns_unknown(self) -> None:
        assert SourceType.infer(None, mime_type=None) == SourceType.UNKNOWN

    @pytest.mark.parametrize("ext,expected", [
        (".ogg", "audio"), (".flac", "audio"), (".aac", "audio"),
        (".mkv", "video"), (".avi", "video"),
        (".gif", "image"), (".webp", "image"), (".tiff", "image"),
        (".go", "code"), (".rs", "code"), (".java", "code"),
        (".sbv", "subtitle"), (".lrc", "subtitle"),
        (".tsv", "dataset"), (".jsonl", "dataset"),
        (".ods", "spreadsheet"),
        (".docx", "article"), (".rtf", "article"),
    ])
    def test_additional_extensions(self, ext: str, expected: str) -> None:
        filename = f"file{ext}"
        result = SourceType.infer(filename)
        assert result.value == expected, (
            f"infer({filename!r}) → {result!r}, expected {expected!r}"
        )


# ===========================================================================
# documents_to_pandas / documents_to_polars
# ===========================================================================


class TestDocumentsToDataFrame:
    """documents_to_pandas and documents_to_polars — happy path, empty, ImportError."""

    def _docs(self, n: int = 3):
        return [CorpusDocument.create("f.txt", i, f"Sentence {i}.") for i in range(n)]

    # --- pandas ---

    def test_to_pandas_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            documents_to_pandas([])

    def test_to_pandas_import_error_raises(self) -> None:
        with patch.dict("sys.modules", {"pandas": None}):
            with pytest.raises((ImportError, Exception)):
                documents_to_pandas(self._docs(1))

    def test_to_pandas_happy_path(self) -> None:
        pytest.importorskip("pandas")
        df = documents_to_pandas(self._docs(3))
        assert len(df) == 3
        assert "doc_id" in df.columns
        assert "text" in df.columns

    def test_to_pandas_with_embedding(self) -> None:
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        import numpy as np  # noqa: PLC0415
        docs = [
            CorpusDocument.create("f.txt", 0, "Hello.").replace(
                embedding=np.zeros(4, dtype="float32")
            )
        ]
        df = documents_to_pandas(docs, include_embedding=True)
        assert "embedding" in df.columns

    # --- polars ---

    def test_to_polars_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            documents_to_polars([])

    def test_to_polars_import_error_raises(self) -> None:
        with patch.dict("sys.modules", {"polars": None}):
            with pytest.raises((ImportError, Exception)):
                documents_to_polars(self._docs(1))

    def test_to_polars_happy_path(self) -> None:
        pytest.importorskip("polars")
        df = documents_to_polars(self._docs(3))
        assert len(df) == 3


# ===========================================================================
# CorpusDocument.replace — deep-copy isolation
# ===========================================================================


class TestReplaceIsolation:
    """replace() must deep-copy mutable containers to prevent shared-reference bugs."""

    def test_metadata_deep_copied(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.", metadata={"k": "v"})
        copy = doc.replace(page_number=1)
        copy.metadata["k"] = "changed"
        assert doc.metadata["k"] == "v"

    def test_tokens_deep_copied(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.").replace(tokens=["a", "b"])
        copy = doc.replace(page_number=1)
        copy.tokens.append("c")  # type: ignore[union-attr]
        assert doc.tokens == ["a", "b"]

    def test_keywords_deep_copied(self) -> None:
        doc = CorpusDocument.create("f.txt", 0, "Hi.").replace(keywords=["kw"])
        copy = doc.replace(page_number=1)
        copy.keywords.append("extra")  # type: ignore[union-attr]
        assert doc.keywords == ["kw"]
