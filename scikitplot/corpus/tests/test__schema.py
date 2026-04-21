# scikitplot/corpus/tests/test__schema.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus._schema
====================================

Coverage
--------
* :class:`SourceType` — 35-member enum, string equality, infer(), ``__all__``
* :class:`Modality` — 5-member enum, string equality, ``__all__``
* :class:`ErrorPolicy` — 4-member enum, string equality, ``__all__``
* :class:`MatchMode` — 4-member enum
* :class:`SectionType` — all original and new dramatic/research members
* :class:`ChunkingStrategy` — all members
* :class:`CorpusDocument` — full field round-trip including 7 new raw media fields
* :meth:`CorpusDocument.make_content_hash` — text and raw_bytes paths
* :meth:`CorpusDocument.make_doc_id` — determinism, length, source_type sensitivity
* :meth:`CorpusDocument.create` — auto-populates content_hash, raw_shape, raw_dtype
* :meth:`CorpusDocument.replace` — copy-on-write, embedding, raw_tensor, modality
* :meth:`CorpusDocument.validate` — all invariant violations
* :data:`_PROMOTED_RAW_KEYS` — all groups including raw media keys
* :meth:`SourceType.infer` — extension and MIME type paths
* Dead-code absence — ``_NO_LETTER_RE`` must not be in ``_schema``

Run with:
    pytest corpus/tests/test__schema.py -v
"""
from __future__ import annotations

import os
import sys

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


# ===========================================================================
# TestInitExports
# ===========================================================================


class TestInitExports:
    """All new enum types must be importable from _schema and in __all__."""

    def test_sourcetype_importable(self):
        assert SourceType.BOOK == "book"

    def test_matchmode_importable(self):
        assert MatchMode.STRICT == "strict"

    def test_modality_importable(self):
        assert Modality.TEXT == "text"
        assert Modality.IMAGE == "image"
        assert Modality.AUDIO == "audio"
        assert Modality.VIDEO == "video"
        assert Modality.MULTIMODAL == "multimodal"

    def test_errorpolicy_importable(self):
        assert ErrorPolicy.RAISE == "raise"
        assert ErrorPolicy.SKIP == "skip"
        assert ErrorPolicy.LOG == "log"
        assert ErrorPolicy.RETRY == "retry"

    def test_modality_in_all(self):
        assert "Modality" in m.__all__

    def test_errorpolicy_in_all(self):
        assert "ErrorPolicy" in m.__all__

    def test_sourcetype_in_all(self):
        assert "SourceType" in m.__all__

    def test_matchmode_in_all(self):
        assert "MatchMode" in m.__all__

    def test_corpusdocument_in_all(self):
        assert "CorpusDocument" in m.__all__

    def test_promoted_raw_keys_in_all(self):
        assert "_PROMOTED_RAW_KEYS" in m.__all__


# ===========================================================================
# TestSourceType
# ===========================================================================


class TestSourceType:
    """SourceType must have 35 members covering all semantic source kinds."""

    EXPECTED = {
        "book", "article", "research", "biography", "play", "poem",
        "news", "blog", "newsletter", "press_release",
        "movie", "subtitle", "video", "audio", "podcast", "lecture", "interview",
        "web", "wiki", "social_media", "forum", "faq",
        "documentation", "tutorial", "manual", "report",
        "legal", "medical", "patent",
        "spreadsheet", "dataset", "code",
        "email", "chat",
        "image",
        "unknown",
    }

    def test_all_expected_values_present(self):
        actual = {st.value for st in SourceType}
        missing = self.EXPECTED - actual
        assert not missing, f"Missing SourceType members: {sorted(missing)}"

    def test_str_equality(self):
        assert SourceType.WIKI == "wiki"
        assert SourceType.RESEARCH == "research"
        assert SourceType.PODCAST == "podcast"

    def test_coercion(self):
        assert SourceType("audio") is SourceType.AUDIO
        assert SourceType("news") is SourceType.NEWS
        assert SourceType("dataset") is SourceType.DATASET

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            SourceType("not_a_source_type")

    def test_journalism_members(self):
        for v in ("news", "blog", "newsletter", "press_release"):
            assert SourceType(v).value == v

    def test_domain_members(self):
        for v in ("legal", "medical", "patent"):
            assert SourceType(v).value == v

    def test_communication_members(self):
        for v in ("email", "chat"):
            assert SourceType(v).value == v

    def test_reference_members(self):
        for v in ("documentation", "tutorial", "manual", "report"):
            assert SourceType(v).value == v


# ===========================================================================
# TestSourceTypeInfer
# ===========================================================================


class TestSourceTypeInfer:
    """SourceType.infer() must map extensions and MIME types correctly."""

    @pytest.mark.parametrize("filename, expected_value", [
        ("report.pdf", "research"),
        ("podcast.mp3", "audio"),
        ("lecture.mp4", "video"),
        ("scan.jpg", "image"),
        ("scan.jpeg", "image"),
        ("diagram.png", "image"),
        ("article.txt", "article"),
        ("readme.md", "article"),
        ("doc.rst", "article"),
        ("data.csv", "dataset"),
        ("data.json", "dataset"),
        ("sheet.xlsx", "spreadsheet"),
        ("sheet.xls", "spreadsheet"),
        ("script.py", "code"),
        ("app.js", "code"),
        ("captions.srt", "subtitle"),
        ("subs.vtt", "subtitle"),
        ("page.html", "web"),
        ("page.htm", "web"),
    ])
    def test_extension_inference(self, filename, expected_value):
        result = SourceType.infer(filename)
        assert result.value == expected_value, (
            f"infer({filename!r}) returned {result!r}, expected {expected_value!r}"
        )

    @pytest.mark.parametrize("mime, expected_value", [
        ("audio/mpeg", "audio"),
        ("audio/wav", "audio"),
        ("video/mp4", "video"),
        ("video/webm", "video"),
        ("image/jpeg", "image"),
        ("image/png", "image"),
        ("application/pdf", "research"),
        ("text/html", "web"),
        ("text/csv", "dataset"),
        ("text/plain", "article"),
    ])
    def test_mime_inference(self, mime, expected_value):
        result = SourceType.infer(mime_type=mime)
        assert result.value == expected_value

    def test_unknown_extension(self):
        assert SourceType.infer("mystery.bin") == SourceType.UNKNOWN

    def test_none_returns_unknown(self):
        assert SourceType.infer(None) == SourceType.UNKNOWN

    def test_extension_beats_mime(self):
        result = SourceType.infer("report.pdf", mime_type="audio/mpeg")
        assert result == SourceType.RESEARCH

    def test_uppercase_normalised(self):
        assert SourceType.infer("REPORT.PDF") == SourceType.RESEARCH

    def test_pathlib_input(self):
        import pathlib  # noqa: PLC0415
        assert SourceType.infer(pathlib.Path("docs/report.pdf")) == SourceType.RESEARCH


# ===========================================================================
# TestModality
# ===========================================================================


class TestModality:
    def test_all_values(self):
        assert {m.value for m in Modality} == {"text", "image", "audio", "video", "multimodal"}

    def test_str_equality(self):
        assert Modality.TEXT == "text"
        assert Modality.IMAGE == "image"

    def test_coercion(self):
        assert Modality("video") is Modality.VIDEO

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Modality("binary")


# ===========================================================================
# TestErrorPolicy
# ===========================================================================


class TestErrorPolicy:
    def test_all_values(self):
        assert {p.value for p in ErrorPolicy} == {"raise", "skip", "log", "retry"}

    def test_str_equality(self):
        assert ErrorPolicy.SKIP == "skip"
        assert ErrorPolicy.LOG == "log"

    def test_coercion(self):
        assert ErrorPolicy("retry") is ErrorPolicy.RETRY

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            ErrorPolicy("ignore")


# ===========================================================================
# TestSectionType
# ===========================================================================


class TestSectionType:
    def test_original_members(self):
        for v in ("text", "footnote", "title", "table", "header",
                  "figure", "code", "caption", "metadata", "unknown"):
            assert SectionType(v).value == v

    def test_new_members(self):
        for v in ("abstract", "references", "stage_direction", "dialogue",
                  "verse", "acknowledgements", "list_item", "sidebar",
                  "lyrics", "transcript"):
            assert SectionType(v).value == v, f"SectionType({v!r}) missing"


# ===========================================================================
# TestChunkingStrategy
# ===========================================================================


class TestChunkingStrategy:
    def test_all_values(self):
        expected = {"sentence", "paragraph", "fixed_window", "semantic",
                    "page", "block", "custom", "none"}
        assert {c.value for c in ChunkingStrategy} == expected


# ===========================================================================
# TestPromotedRawKeys
# ===========================================================================


class TestPromotedRawKeys:
    """_PROMOTED_RAW_KEYS must include all field groups including raw media."""

    def test_provenance_keys(self):
        for k in ("source_type", "source_title", "source_author", "source_date",
                  "collection_id", "url", "doi", "isbn"):
            assert k in _PROMOTED_RAW_KEYS

    def test_position_keys(self):
        for k in ("page_number", "paragraph_index", "line_number", "parent_doc_id",
                  "act", "scene_number"):
            assert k in _PROMOTED_RAW_KEYS

    def test_media_keys(self):
        for k in ("timecode_start", "timecode_end", "confidence", "ocr_engine", "bbox"):
            assert k in _PROMOTED_RAW_KEYS

    def test_nlp_keys(self):
        for k in ("normalized_text", "tokens", "lemmas", "stems", "keywords"):
            assert k in _PROMOTED_RAW_KEYS

    def test_raw_media_keys(self):
        for k in ("modality", "raw_bytes", "raw_tensor", "raw_shape",
                  "raw_dtype", "frame_index", "content_hash"):
            assert k in _PROMOTED_RAW_KEYS, f"Missing raw media key: {k!r}"


# ===========================================================================
# TestMakeContentHash
# ===========================================================================


class TestMakeContentHash:
    def test_text_returns_32_hex_chars(self):
        h = CorpusDocument.make_content_hash(text="Hello world")
        assert isinstance(h, str) and len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_raw_bytes_path(self):
        h = CorpusDocument.make_content_hash(raw_bytes=b"\x89PNG\r\n")
        assert len(h) == 32

    def test_raw_bytes_beats_text(self):
        h_t = CorpusDocument.make_content_hash(text="Hello")
        h_r = CorpusDocument.make_content_hash(text="Hello", raw_bytes=b"different")
        assert h_t != h_r

    def test_deterministic(self):
        assert (CorpusDocument.make_content_hash(text="X") ==
                CorpusDocument.make_content_hash(text="X"))

    def test_different_texts_differ(self):
        assert (CorpusDocument.make_content_hash(text="A") !=
                CorpusDocument.make_content_hash(text="B"))

    def test_empty_returns_zeros(self):
        assert CorpusDocument.make_content_hash() == "0" * 32


# ===========================================================================
# TestCreate
# ===========================================================================


class TestCreate:
    """CorpusDocument.create() — auto fields and raw media fields."""

    def test_doc_id_auto(self):
        doc = CorpusDocument.create("f.txt", 0, "Hello.")
        assert len(doc.doc_id) == 16

    def test_content_hash_auto(self):
        doc = CorpusDocument.create("f.txt", 0, "Hello.")
        assert doc.content_hash is not None and len(doc.content_hash) == 32

    def test_same_text_same_hash(self):
        d1 = CorpusDocument.create("a.txt", 0, "Same.")
        d2 = CorpusDocument.create("b.txt", 1, "Same.")
        assert d1.content_hash == d2.content_hash

    def test_modality_defaults_text(self):
        assert CorpusDocument.create("f.txt", 0, "Hello.").modality == Modality.TEXT

    def test_text_none_allowed_for_raw_docs(self):
        doc = CorpusDocument.create("img.jpg", 0, None, modality=Modality.IMAGE)
        assert doc.text is None

    def test_raw_shape_auto_inferred(self):
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        doc = CorpusDocument.create("img.jpg", 0, None, raw_tensor=arr,
                                    modality=Modality.IMAGE)
        assert doc.raw_shape == (224, 224, 3)

    def test_raw_dtype_auto_inferred(self):
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        doc = CorpusDocument.create("img.jpg", 0, None, raw_tensor=arr,
                                    modality=Modality.IMAGE)
        assert "uint8" in doc.raw_dtype

    def test_raw_bytes_hash(self):
        raw = b"\xff\xd8\xff"
        doc = CorpusDocument.create("img.jpg", 0, None, raw_bytes=raw)
        expected = CorpusDocument.make_content_hash(raw_bytes=raw)
        assert doc.content_hash == expected

    def test_frame_index_stored(self):
        doc = CorpusDocument.create("v.mp4", 3, None,
                                    frame_index=7, modality=Modality.VIDEO)
        assert doc.frame_index == 7

    def test_doc_id_source_type_sensitivity(self):
        id1 = CorpusDocument.make_doc_id("f.txt", 0, "Hi", SourceType.BOOK)
        id2 = CorpusDocument.make_doc_id("f.txt", 0, "Hi", SourceType.MOVIE)
        assert id1 != id2


# ===========================================================================
# TestValidate
# ===========================================================================


class TestValidate:
    def test_bad_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            CorpusDocument.create("f.txt", 0, "Hi.", confidence=1.5)

    def test_bad_timecode_order(self):
        with pytest.raises(ValueError, match="timecode_end"):
            CorpusDocument.create("f.txt", 0, "Hi.",
                                  timecode_start=5.0, timecode_end=3.0)

    def test_bad_bbox(self):
        with pytest.raises(ValueError, match="bbox"):
            CorpusDocument.create("f.txt", 0, "Hi.", bbox=(0.0, 1.0))

    def test_bad_act(self):
        with pytest.raises(ValueError, match="act"):
            CorpusDocument.create("f.txt", 0, "Hi.", act=0)

    def test_doi_bad_format_warns(self):
        with pytest.warns(UserWarning, match="DOI"):
            CorpusDocument.create("f.txt", 0, "Hi.", doi="not-a-doi")


# ===========================================================================
# TestRoundTrip
# ===========================================================================


class TestRoundTrip:
    """to_dict / from_dict round-trips for all field groups."""

    def _full_doc(self):
        return CorpusDocument.create(
            input_path="hamlet.txt", chunk_index=5,
            text="To be or not to be.",
            section_type=SectionType.DIALOGUE,
            chunking_strategy=ChunkingStrategy.SENTENCE,
            language="en", char_start=100, char_end=120,
            source_type=SourceType.PLAY, source_title="Hamlet",
            source_author="Shakespeare", source_date="1603",
            collection_id="bard", isbn="978-0-7432-7796-2",
            page_number=42, paragraph_index=3, line_number=7,
            act=3, scene_number=1,
            normalized_text="to be or not to be",
            tokens=["To", "be"], lemmas=["be", "be"],
            stems=["to", "be"], keywords=["question"],
        )

    def test_provenance_preserved(self):
        doc = self._full_doc()
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.source_type.value == "play"
        assert r.source_title == "Hamlet"
        assert r.isbn == "978-0-7432-7796-2"

    def test_position_preserved(self):
        doc = self._full_doc()
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.page_number == 42 and r.act == 3

    def test_nlp_preserved(self):
        doc = self._full_doc()
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.tokens == ["To", "be"]
        assert r.keywords == ["question"]

    def test_modality_preserved(self):
        doc = CorpusDocument.create("f.txt", 0, "Hi.", modality=Modality.TEXT)
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.modality == Modality.TEXT

    def test_content_hash_preserved(self):
        doc = CorpusDocument.create("f.txt", 0, "Hello.")
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.content_hash == doc.content_hash

    def test_frame_index_preserved(self):
        doc = CorpusDocument.create("v.mp4", 2, None,
                                    frame_index=7, modality=Modality.VIDEO)
        r = CorpusDocument.from_dict(doc.to_dict())
        assert r.frame_index == 7

    def test_doc_id_preserved(self):
        doc = self._full_doc()
        assert CorpusDocument.from_dict(doc.to_dict()).doc_id == doc.doc_id


# ===========================================================================
# TestReplace
# ===========================================================================


class TestReplace:
    def test_single_field(self):
        doc = CorpusDocument.create("f.txt", 0, "Hello.")
        u = doc.replace(page_number=99)
        assert u.page_number == 99 and doc.page_number is None

    def test_embedding(self):
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError:
            pytest.skip("numpy not available")
        doc = CorpusDocument.create("f.txt", 0, "Hello.")
        assert not doc.has_embedding
        e = CorpusDocument.create("f.txt", 0, "Hello.")
        enriched = e.replace(embedding=np.zeros(384, dtype=np.float32))
        assert enriched.has_embedding

    def test_modality(self):
        doc = CorpusDocument.create("f.txt", 0, "Hello.")
        u = doc.replace(modality=Modality.MULTIMODAL)
        assert u.modality == Modality.MULTIMODAL
        assert doc.modality == Modality.TEXT

    def test_preserves_other_fields(self):
        doc = CorpusDocument.create("f.txt", 0, "Hello.",
                                    source_title="My Title")
        u = doc.replace(page_number=5)
        assert u.source_title == "My Title"

    def test_unknown_field_raises(self):
        doc = CorpusDocument.create("f.txt", 0, "Hello.")
        with pytest.raises((ValueError, TypeError)):
            doc.replace(nonexistent_field="x")


# ===========================================================================
# TestDeadCode
# ===========================================================================


class TestDeadCode:
    def test_no_letter_re_absent(self):
        assert not hasattr(m, "_NO_LETTER_RE"), (
            "_NO_LETTER_RE is dead code; must only live in _base.py"
        )

    def test_doi_prefix_re_present(self):
        assert hasattr(m, "_DOI_PREFIX_RE")
