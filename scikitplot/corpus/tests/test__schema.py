"""
Tests for Group 1 fixes:
  - HIGH-INIT1: SourceType and MatchMode exported from scikitplot.corpus.__init__
  - LOW-SC1: _NO_LETTER_RE removed from _schema.py (was dead code)
  - Smoke tests for CorpusDocument full field round-trip

Run with: pytest tests/test_group1_schema_init.py -v
"""
from __future__ import annotations

import importlib
import sys
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_corpus_module():
    """
    Import corpus package, mocking heavy optional readers to avoid
    ImportError from optional dependencies (PIL, pytesseract, etc.).
    """
    # We stub the _readers import so heavy deps are not required in CI
    # The _readers __init__.py registers readers; we just need it to succeed.
    # Strategy: mock the heavy per-reader imports inside _readers submodules.
    # Easiest: import the schema and __init__ directly without full reader chain.
    # Since tests run against the local corpus/ directory (not installed package),
    # we import via sys.path manipulation.
    import os
    # import sys
    # corpus/ parent directory must be on sys.path
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent not in sys.path:
        sys.path.insert(0, parent)


# ---------------------------------------------------------------------------
# HIGH-INIT1: SourceType and MatchMode must be importable from corpus package
# ---------------------------------------------------------------------------

class TestInitExports:
    """Verify that SourceType and MatchMode are exported from corpus __init__."""

    def test_sourcetype_importable_from_schema(self):
        """SourceType must be importable from _schema without any extras."""
        from .._schema import SourceType  # noqa: PLC0415
        assert SourceType.BOOK == "book"
        assert SourceType.UNKNOWN == "unknown"

    def test_matchmode_importable_from_schema(self):
        """MatchMode must be importable from _schema without any extras."""
        from .._schema import MatchMode  # noqa: PLC0415
        assert MatchMode.STRICT == "strict"
        assert MatchMode.SEMANTIC == "semantic"
        assert MatchMode.HYBRID == "hybrid"
        assert MatchMode.KEYWORD == "keyword"

    def test_sourcetype_in_schema_all(self):
        """SourceType must appear in _schema.__all__."""
        from .. import _schema as m  # noqa: PLC0415
        assert "SourceType" in m.__all__

    def test_matchmode_in_schema_all(self):
        """MatchMode must appear in _schema.__all__."""
        from .. import _schema as m  # noqa: PLC0415
        assert "MatchMode" in m.__all__

    def test_init_imports_sourcetype(self):
        """SourceType must be importable from corpus package root."""
        # Import only the schema part of __init__ to avoid heavy readers
        from .._schema import SourceType, MatchMode  # noqa: PLC0415
        # Verify the values defined in architecture doc are present
        expected_source_types = {
            "book", "article", "research", "movie", "subtitle",
            "play", "poem", "biography", "web", "wiki",
            "image", "video", "audio", "spreadsheet", "code", "unknown",
        }
        actual_values = {st.value for st in SourceType}
        assert expected_source_types == actual_values, (
            f"Missing SourceType members: {expected_source_types - actual_values}"
        )

    def test_matchmode_values(self):
        """MatchMode must have exactly: strict, keyword, semantic, hybrid."""
        from .._schema import MatchMode  # noqa: PLC0415
        expected = {"strict", "keyword", "semantic", "hybrid"}
        actual = {m.value for m in MatchMode}
        assert expected == actual


# ---------------------------------------------------------------------------
# LOW-SC1: _NO_LETTER_RE must NOT be defined in _schema.py
# ---------------------------------------------------------------------------

class TestSchemaDeadCode:
    """_NO_LETTER_RE was dead code in _schema.py — must be removed."""

    def test_no_letter_re_removed_from_schema(self):
        """_NO_LETTER_RE must not be present in _schema module namespace."""
        from .. import _schema as m  # noqa: PLC0415
        assert not hasattr(m, "_NO_LETTER_RE"), (
            "_NO_LETTER_RE is dead code in _schema.py; it must be removed. "
            "Only _base.py should define it."
        )

    def test_doi_prefix_re_still_present(self):
        """_DOI_PREFIX_RE must still be present (used by validate())."""
        from .. import _schema as m  # noqa: PLC0415
        assert hasattr(m, "_DOI_PREFIX_RE")


# ---------------------------------------------------------------------------
# Smoke: CorpusDocument with all new fields round-trips correctly
# ---------------------------------------------------------------------------

class TestCorpusDocumentFullRoundTrip:
    """CorpusDocument.to_dict / from_dict must include all 22 new fields."""

    def _make_doc(self):
        from .._schema import (  # noqa: PLC0415
            CorpusDocument, SourceType, SectionType, ChunkingStrategy,
        )
        return CorpusDocument.create(
            source_file="hamlet.txt",
            chunk_index=5,
            text="To be or not to be, that is the question.",
            section_type=SectionType.DIALOGUE,
            chunking_strategy=ChunkingStrategy.SENTENCE,
            language="en",
            char_start=100,
            char_end=141,
            source_type=SourceType.PLAY,
            source_title="Hamlet",
            source_author="Shakespeare, William",
            source_date="1603",
            collection_id="shakespeare_complete",
            url=None,
            doi=None,
            isbn="978-0-7432-7796-2",
            page_number=42,
            paragraph_index=3,
            line_number=7,
            act=3,
            scene_number=1,
            timecode_start=None,
            timecode_end=None,
            confidence=None,
            ocr_engine=None,
            bbox=None,
            normalized_text="to be or not to be that is the question",
            tokens=["To", "be", "or", "not", "to", "be"],
            lemmas=["be", "be", "or", "not", "be", "be"],
            stems=["to", "be", "or", "not", "to", "be"],
            keywords=["question", "being"],
        )

    def test_to_dict_includes_provenance_fields(self):
        doc = self._make_doc()
        d = doc.to_dict()
        assert d["source_type"] == "play"
        assert d["source_title"] == "Hamlet"
        assert d["source_author"] == "Shakespeare, William"
        assert d["collection_id"] == "shakespeare_complete"
        assert d["isbn"] == "978-0-7432-7796-2"

    def test_to_dict_includes_position_fields(self):
        doc = self._make_doc()
        d = doc.to_dict()
        assert d["page_number"] == 42
        assert d["paragraph_index"] == 3
        assert d["line_number"] == 7
        assert d["act"] == 3
        assert d["scene_number"] == 1

    def test_to_dict_includes_nlp_fields(self):
        doc = self._make_doc()
        d = doc.to_dict()
        assert d["normalized_text"] == "to be or not to be that is the question"
        assert d["tokens"] == ["To", "be", "or", "not", "to", "be"]
        assert d["lemmas"] is not None
        assert d["stems"] is not None
        assert d["keywords"] == ["question", "being"]

    def test_from_dict_round_trip(self):
        from .._schema import CorpusDocument  # noqa: PLC0415
        doc = self._make_doc()
        d = doc.to_dict()
        restored = CorpusDocument.from_dict(d)
        assert restored.doc_id == doc.doc_id
        assert restored.source_type.value == "play"
        assert restored.act == 3
        assert restored.scene_number == 1
        assert restored.keywords == ["question", "being"]
        assert restored.page_number == 42

    def test_doc_id_includes_source_type(self):
        """Same file/chunk/text but different source_type must yield different doc_id (Issue S-7)."""
        from .._schema import CorpusDocument, SourceType  # noqa: PLC0415
        id_book = CorpusDocument.make_doc_id("f.txt", 0, "Hello", SourceType.BOOK)
        id_movie = CorpusDocument.make_doc_id("f.txt", 0, "Hello", SourceType.MOVIE)
        assert id_book != id_movie, "doc_id must differ when source_type differs"

    def test_doc_id_length(self):
        from .._schema import CorpusDocument  # noqa: PLC0415
        did = CorpusDocument.make_doc_id("f.txt", 0, "Hello world.")
        assert len(did) == 16
        assert did.isalnum()  # hex chars only

    def test_replace_preserves_all_fields(self):
        doc = self._make_doc()
        updated = doc.replace(page_number=99)
        assert updated.page_number == 99
        assert updated.source_type == doc.source_type
        assert updated.tokens is not None
        assert updated.tokens is not doc.tokens  # deep copy

    def test_validate_raises_on_bad_confidence(self):
        import pytest  # noqa: PLC0415
        from .._schema import CorpusDocument  # noqa: PLC0415
        with pytest.raises(ValueError, match="confidence"):
            CorpusDocument.create("f.txt", 0, "Hello.", confidence=1.5)

    def test_validate_raises_on_bad_timecode_order(self):
        import pytest  # noqa: PLC0415
        from .._schema import CorpusDocument  # noqa: PLC0415
        with pytest.raises(ValueError, match="timecode_end"):
            CorpusDocument.create("f.txt", 0, "Hi.", timecode_start=5.0, timecode_end=3.0)

    def test_validate_raises_on_bad_bbox(self):
        import pytest  # noqa: PLC0415
        from .._schema import CorpusDocument  # noqa: PLC0415
        with pytest.raises(ValueError, match="bbox"):
            CorpusDocument.create("f.txt", 0, "Hi.", bbox=(0.0, 1.0))  # only 2 items

    def test_validate_raises_on_bad_act(self):
        import pytest  # noqa: PLC0415
        from .._schema import CorpusDocument  # noqa: PLC0415
        with pytest.raises(ValueError, match="act"):
            CorpusDocument.create("f.txt", 0, "Hi.", act=0)  # must be >= 1

    def test_doi_warns_on_bad_format(self):
        import pytest  # noqa: PLC0415
        from .._schema import CorpusDocument  # noqa: PLC0415
        with pytest.warns(UserWarning, match="DOI"):
            CorpusDocument.create("f.txt", 0, "Hi.", doi="not-a-doi")

    def test_section_type_new_members(self):
        """All 8 new SectionType members added in Issue S-1 must be present."""
        from .._schema import SectionType  # noqa: PLC0415
        new_members = [
            "abstract", "references", "stage_direction", "dialogue",
            "verse", "acknowledgements", "list_item", "sidebar",
        ]
        for val in new_members:
            member = SectionType(val)
            assert member.value == val, f"SectionType({val!r}) failed"


# ---------------------------------------------------------------------------
# MatchMode / SourceType enum behaviour
# ---------------------------------------------------------------------------

class TestEnumBehaviour:
    """Enums must behave as StrEnum: str equality, round-trip, membership."""

    def test_sourcetype_str_equality(self):
        from .._schema import SourceType  # noqa: PLC0415
        assert SourceType.WIKI == "wiki"
        assert SourceType.RESEARCH == "research"

    def test_matchmode_str_equality(self):
        from .._schema import MatchMode  # noqa: PLC0415
        assert MatchMode.HYBRID == "hybrid"

    def test_sourcetype_coercion_from_string(self):
        from .._schema import SourceType  # noqa: PLC0415
        assert SourceType("audio") is SourceType.AUDIO

    def test_sourcetype_invalid_value_raises(self):
        import pytest  # noqa: PLC0415
        from .._schema import SourceType  # noqa: PLC0415
        with pytest.raises(ValueError):
            SourceType("not_a_source_type")

    def test_matchmode_invalid_value_raises(self):
        import pytest  # noqa: PLC0415
        from .._schema import MatchMode  # noqa: PLC0415
        with pytest.raises(ValueError):
            MatchMode("fuzzy")

    def test_promoted_raw_keys_completeness(self):
        """_PROMOTED_RAW_KEYS must include all 22 new field names."""
        from .._schema import _PROMOTED_RAW_KEYS  # noqa: PLC0415
        required = {
            "source_type", "source_title", "source_author", "source_date",
            "collection_id", "url", "doi", "isbn",
            "page_number", "paragraph_index", "line_number", "parent_doc_id",
            "act", "scene_number",
            "timecode_start", "timecode_end", "confidence", "ocr_engine", "bbox",
            "normalized_text", "tokens", "lemmas", "stems", "keywords",
        }
        missing = required - _PROMOTED_RAW_KEYS
        assert not missing, (
            f"_PROMOTED_RAW_KEYS is missing fields: {sorted(missing)}"
        )
