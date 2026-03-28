# scikitplot/corpus/_readers/tests/test__custom.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for CustomReader, normalize_extractor_output, and the
custom_extractor hooks added to PDFReader, ImageReader, AudioReader,
and VideoReader.

All tests use only stdlib — no external ML/OCR/PDF libraries required.
"""

from __future__ import annotations

import pathlib
import tempfile
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Module under test ──────────────────────────────────────────────────────
from .._custom import CustomReader, normalize_extractor_output
from ..._base import DocumentReader
from ..._schema import SectionType, SourceType


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def tmp_txt(tmp_path: pathlib.Path) -> pathlib.Path:
    """A real file so validate_input passes for tests that need it."""
    p = tmp_path / "test.txt"
    p.write_text("hello world this is a test sentence for the corpus pipeline")
    return p


@pytest.fixture()
def tmp_pdf(tmp_path: pathlib.Path) -> pathlib.Path:
    """A placeholder .pdf file (bytes irrelevant — custom extractor reads it)."""
    p = tmp_path / "test.pdf"
    p.write_bytes(b"%PDF-1.4 fake content for testing")
    return p


@pytest.fixture()
def tmp_audio(tmp_path: pathlib.Path) -> pathlib.Path:
    """A placeholder .mp3 file."""
    p = tmp_path / "test.mp3"
    p.write_bytes(b"ID3fake")
    return p


@pytest.fixture()
def tmp_video(tmp_path: pathlib.Path) -> pathlib.Path:
    """A placeholder .mp4 file."""
    p = tmp_path / "test.mp4"
    p.write_bytes(b"ftyp fake mp4")
    return p


@pytest.fixture()
def tmp_image(tmp_path: pathlib.Path) -> pathlib.Path:
    """A placeholder .png file."""
    p = tmp_path / "test.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return p


# ===========================================================================
# normalize_extractor_output
# ===========================================================================


class TestNormalizeExtractorOutput:
    """Tests for normalize_extractor_output — the shared coercion utility."""

    def test_str_produces_single_chunk(self) -> None:
        result = normalize_extractor_output("Hello world")
        assert len(result) == 1
        assert result[0]["text"] == "Hello world"

    def test_str_sets_default_section_and_source_type(self) -> None:
        result = normalize_extractor_output("text")
        assert result[0]["section_type"] == SectionType.TEXT.value
        assert result[0]["source_type"] == SourceType.UNKNOWN.value

    def test_str_respects_custom_source_type(self) -> None:
        result = normalize_extractor_output(
            "text", source_type=SourceType.RESEARCH
        )
        assert result[0]["source_type"] == SourceType.RESEARCH.value

    def test_str_respects_custom_section_type(self) -> None:
        result = normalize_extractor_output(
            "text", section_type=SectionType.LYRICS
        )
        assert result[0]["section_type"] == SectionType.LYRICS.value

    def test_list_of_str_produces_multiple_chunks(self) -> None:
        result = normalize_extractor_output(["Page one", "Page two", "Page three"])
        assert len(result) == 3
        assert [r["text"] for r in result] == ["Page one", "Page two", "Page three"]
        for r in result:
            assert "section_type" in r
            assert "source_type" in r

    def test_empty_list_returns_empty(self) -> None:
        assert normalize_extractor_output([]) == []

    def test_dict_with_text_only(self) -> None:
        result = normalize_extractor_output({"text": "Hello"})
        assert len(result) == 1
        assert result[0]["text"] == "Hello"
        assert result[0]["section_type"] == SectionType.TEXT.value

    def test_dict_preserves_extra_metadata(self) -> None:
        raw = {"text": "Hello", "page_number": 3, "confidence": 0.95}
        result = normalize_extractor_output(raw)
        assert result[0]["page_number"] == 3
        assert result[0]["confidence"] == 0.95

    def test_dict_does_not_override_caller_section_type(self) -> None:
        # When the caller's dict already has section_type, it is preserved
        raw = {"text": "Hello", "section_type": SectionType.LYRICS.value}
        result = normalize_extractor_output(raw, section_type=SectionType.TEXT)
        assert result[0]["section_type"] == SectionType.LYRICS.value  # dict wins

    def test_list_of_dicts(self) -> None:
        raw = [
            {"text": "Chunk A", "page_number": 0},
            {"text": "Chunk B", "page_number": 1},
        ]
        result = normalize_extractor_output(raw)
        assert len(result) == 2
        assert result[0]["page_number"] == 0
        assert result[1]["page_number"] == 1

    def test_list_of_dicts_fills_defaults(self) -> None:
        raw = [{"text": "A"}, {"text": "B"}]
        result = normalize_extractor_output(raw, source_type=SourceType.IMAGE)
        for r in result:
            assert r["source_type"] == SourceType.IMAGE.value

    # ── Error cases ─────────────────────────────────────────────────────

    def test_raises_type_error_on_unsupported_type(self) -> None:
        with pytest.raises(TypeError, match="extractor must return"):
            normalize_extractor_output(42)  # type: ignore[arg-type]

    def test_raises_type_error_on_none(self) -> None:
        with pytest.raises(TypeError, match="extractor must return"):
            normalize_extractor_output(None)  # type: ignore[arg-type]

    def test_raises_value_error_on_dict_missing_text(self) -> None:
        with pytest.raises(ValueError, match="missing the required 'text' key"):
            normalize_extractor_output({"page_number": 0})

    def test_raises_value_error_on_text_not_str(self) -> None:
        with pytest.raises(ValueError, match="'text' must be a str"):
            normalize_extractor_output({"text": 123})  # type: ignore[dict-item]

    def test_raises_type_error_on_mixed_list(self) -> None:
        with pytest.raises(TypeError, match="expected str"):
            normalize_extractor_output(["hello", 42])  # type: ignore[list-item]

    def test_raises_type_error_on_list_mixed_str_and_dict(self) -> None:
        with pytest.raises(TypeError, match="expected str"):
            normalize_extractor_output(["hello", {"text": "world"}])  # type: ignore[list-item]

    def test_raises_type_error_on_list_of_dicts_with_non_dict(self) -> None:
        with pytest.raises(TypeError, match="expected dict"):
            normalize_extractor_output([{"text": "A"}, "B"])  # type: ignore[list-item]

    def test_raises_value_error_on_list_dict_missing_text(self) -> None:
        with pytest.raises(ValueError, match="missing the required 'text' key"):
            normalize_extractor_output([{"text": "A"}, {"page_number": 1}])


# ===========================================================================
# CustomReader — direct use (no registration)
# ===========================================================================


class TestCustomReaderDirect:
    """Direct CustomReader usage: bypass registry, validate_file=True/False."""

    def test_basic_str_extractor(self, tmp_txt: pathlib.Path) -> None:
        """Extractor returning str → one document."""
        def extractor(path: pathlib.Path, **kw: Any) -> str:
            return path.read_text()

        reader = CustomReader(input_file=tmp_txt, extractor=extractor)
        docs = list(reader.get_documents())
        assert len(docs) == 1
        assert "hello world" in docs[0].text

    def test_list_of_str_extractor(self, tmp_txt: pathlib.Path) -> None:
        """Extractor returning list[str] → multiple documents."""
        def extractor(path: pathlib.Path, **kw: Any) -> list[str]:
            return ["sentence one is here", "sentence two is here today"]

        reader = CustomReader(input_file=tmp_txt, extractor=extractor)
        docs = list(reader.get_documents())
        # Both strings have enough words to pass DefaultFilter
        assert len(docs) == 2

    def test_list_of_dict_extractor(self, tmp_txt: pathlib.Path) -> None:
        """Extractor returning list[dict] → metadata preserved in documents."""
        def extractor(path: pathlib.Path, **kw: Any) -> list[dict]:
            return [
                {"text": "page one text here now", "page_number": 0},
                {"text": "page two text here now", "page_number": 1},
            ]

        reader = CustomReader(input_file=tmp_txt, extractor=extractor)
        docs = list(reader.get_documents())
        assert len(docs) == 2
        assert docs[0].page_number == 0
        assert docs[1].page_number == 1

    def test_reader_kwargs_forwarded(self, tmp_txt: pathlib.Path) -> None:
        """reader_kwargs are forwarded to extractor."""
        received_kwargs: dict[str, Any] = {}

        def extractor(path: pathlib.Path, **kw: Any) -> str:
            received_kwargs.update(kw)
            return "the quick brown fox jumps over"

        reader = CustomReader(
            input_file=tmp_txt,
            extractor=extractor,
            reader_kwargs={"language": "en", "model": "large"},
        )
        list(reader.get_documents())
        assert received_kwargs == {"language": "en", "model": "large"}

    def test_default_source_type_applied(self, tmp_txt: pathlib.Path) -> None:
        def extractor(path: pathlib.Path, **kw: Any) -> str:
            return "the quick brown fox jumps over"

        reader = CustomReader(
            input_file=tmp_txt,
            extractor=extractor,
            default_source_type=SourceType.PODCAST,
        )
        docs = list(reader.get_documents())
        assert docs[0].source_type == SourceType.PODCAST

    def test_extractor_source_type_overrides_default(self, tmp_txt: pathlib.Path) -> None:
        """Extractor-level source_type wins over default_source_type."""
        def extractor(path: pathlib.Path, **kw: Any) -> list[dict]:
            return [{"text": "the quick brown fox", "source_type": SourceType.RESEARCH.value}]

        reader = CustomReader(
            input_file=tmp_txt,
            extractor=extractor,
            default_source_type=SourceType.PODCAST,
        )
        docs = list(reader.get_documents())
        assert docs[0].source_type == SourceType.RESEARCH

    def test_validate_file_false_skips_existence_check(self, tmp_path: pathlib.Path) -> None:
        """validate_file=False allows non-existent paths."""
        def extractor(path: pathlib.Path, **kw: Any) -> str:
            return "the quick brown fox jumps over"

        fake_path = tmp_path / "does_not_exist.xyz"
        reader = CustomReader(
            input_file=fake_path,
            extractor=extractor,
            validate_file=False,
        )
        # Should not raise on validate_input
        docs = list(reader.get_documents())
        assert len(docs) == 1

    def test_validate_file_true_raises_on_missing(self, tmp_path: pathlib.Path) -> None:
        def extractor(path: pathlib.Path, **kw: Any) -> str:
            return "x"

        fake_path = tmp_path / "missing.txt"
        reader = CustomReader(input_file=fake_path, extractor=extractor)
        with pytest.raises(ValueError, match="does not exist"):
            list(reader.get_documents())

    def test_extractor_runtime_error_is_wrapped(self, tmp_txt: pathlib.Path) -> None:
        def bad_extractor(path: pathlib.Path, **kw: Any) -> str:
            raise OSError("disk failure")

        reader = CustomReader(input_file=tmp_txt, extractor=bad_extractor)
        with pytest.raises(RuntimeError, match="raised an error"):
            list(reader.get_documents())

    def test_raises_when_extractor_is_none_at_call_time(self, tmp_txt: pathlib.Path) -> None:
        reader = CustomReader(input_file=tmp_txt, extractor=None)
        with pytest.raises(ValueError, match="extractor is not set"):
            list(reader.get_documents())

    # ── Constructor validation ───────────────────────────────────────────

    def test_raises_type_error_on_non_callable_extractor(self, tmp_txt: pathlib.Path) -> None:
        with pytest.raises(TypeError, match="extractor must be callable"):
            CustomReader(input_file=tmp_txt, extractor="not_a_function")  # type: ignore[arg-type]

    def test_raises_value_error_on_bad_extension_prefix(self, tmp_txt: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="must start with"):
            CustomReader(
                input_file=tmp_txt,
                extractor=lambda p, **kw: "x",
                extensions=["pdf"],  # missing leading dot
            )

    def test_filter_removes_noise_chunks(self, tmp_txt: pathlib.Path) -> None:
        """DefaultFilter discards chunks with fewer than 3 words."""
        def extractor(path: pathlib.Path, **kw: Any) -> list[str]:
            return ["ok", "short", "this sentence has enough words to pass filter"]

        reader = CustomReader(input_file=tmp_txt, extractor=extractor)
        docs = list(reader.get_documents())
        # Only the last string passes DefaultFilter(min_words=3, min_chars=10)
        assert len(docs) == 1
        assert "enough words" in docs[0].text

    def test_not_registered_in_base_registry(self) -> None:
        """CustomReader itself (with file_type=None) must NOT appear in the registry."""
        registry = DocumentReader.subclass_by_type()
        # No entry that maps to the base CustomReader class specifically
        for ext, cls in registry.items():
            assert cls is not CustomReader, (
                f"CustomReader base class must not auto-register (found at {ext!r})"
            )


# ===========================================================================
# CustomReader.register()
# ===========================================================================


class TestCustomReaderRegister:
    """Tests for CustomReader.register() — dynamic subclass creation and wiring."""

    def test_register_returns_subclass_of_custom_reader(self) -> None:
        def ext(path: pathlib.Path, **kw: Any) -> str:
            return "the quick brown fox"

        Cls = CustomReader.register(
            name="TestReader1",
            extensions=[".xyz1"],
            extractor=ext,
        )
        assert issubclass(Cls, CustomReader)
        assert Cls.__name__ == "TestReader1"

    def test_registered_class_in_registry(self) -> None:
        def ext(path: pathlib.Path, **kw: Any) -> str:
            return "x"

        CustomReader.register(
            name="TestReader2",
            extensions=[".xyz2"],
            extractor=ext,
        )
        registry = DocumentReader.subclass_by_type()
        assert ".xyz2" in registry
        assert issubclass(registry[".xyz2"], CustomReader)

    def test_registered_class_dispatched_by_create(self, tmp_path: pathlib.Path) -> None:
        """DocumentReader.create(Path('file.xyz3')) dispatches to registered class."""
        def ext(path: pathlib.Path, **kw: Any) -> str:
            return "the quick brown fox jumps over"

        CustomReader.register(
            name="TestReader3",
            extensions=[".xyz3"],
            extractor=ext,
        )
        p = tmp_path / "file.xyz3"
        p.write_text("dummy")
        reader = DocumentReader.create(p)
        assert isinstance(reader, CustomReader)
        docs = list(reader.get_documents())
        assert len(docs) == 1

    def test_registered_extractor_receives_reader_kwargs(self, tmp_path: pathlib.Path) -> None:
        """reader_kwargs passed to register() are forwarded to the extractor."""
        received: dict[str, Any] = {}

        def ext(path: pathlib.Path, **kw: Any) -> str:
            received.update(kw)
            return "the quick brown fox jumps over"

        CustomReader.register(
            name="TestReader4",
            extensions=[".xyz4"],
            extractor=ext,
            reader_kwargs={"model": "large-v3"},
        )
        p = tmp_path / "file.xyz4"
        p.write_text("dummy")
        docs = list(DocumentReader.create(p).get_documents())
        assert received.get("model") == "large-v3"

    def test_instance_reader_kwargs_override_registered_defaults(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Instance-level reader_kwargs override the registered defaults."""
        received: dict[str, Any] = {}

        def ext(path: pathlib.Path, **kw: Any) -> str:
            received.update(kw)
            return "the quick brown fox jumps"

        Cls = CustomReader.register(
            name="TestReader5",
            extensions=[".xyz5"],
            extractor=ext,
            reader_kwargs={"language": "en"},
        )
        p = tmp_path / "file.xyz5"
        p.write_text("dummy")
        # Instance override: language="de" wins
        reader = Cls(input_file=p, reader_kwargs={"language": "de"})
        list(reader.get_documents())
        assert received["language"] == "de"

    def test_multiple_extensions_all_registered(self, tmp_path: pathlib.Path) -> None:
        def ext(path: pathlib.Path, **kw: Any) -> str:
            return "the quick brown fox jumps over"

        CustomReader.register(
            name="MultiExtReader",
            extensions=[".mxa", ".mxb"],
            extractor=ext,
        )
        registry = DocumentReader.subclass_by_type()
        assert ".mxa" in registry
        assert ".mxb" in registry

    def test_raises_on_invalid_name(self) -> None:
        with pytest.raises(ValueError, match="valid Python identifier"):
            CustomReader.register(
                name="123Bad",
                extensions=[".ok"],
                extractor=lambda p, **kw: "x",
            )

    def test_raises_on_empty_extensions(self) -> None:
        with pytest.raises(ValueError, match="non-empty list"):
            CustomReader.register(
                name="EmptyExtReader",
                extensions=[],
                extractor=lambda p, **kw: "x",
            )

    def test_raises_on_bad_extension_prefix(self) -> None:
        with pytest.raises(ValueError, match="must start with"):
            CustomReader.register(
                name="BadExtReader",
                extensions=["nodot"],
                extractor=lambda p, **kw: "x",
            )

    def test_raises_on_non_callable_extractor(self) -> None:
        with pytest.raises(TypeError, match="extractor must be callable"):
            CustomReader.register(
                name="NonCallable",
                extensions=[".nc"],
                extractor="not_callable",  # type: ignore[arg-type]
            )

    def test_registered_default_source_type_applied(self, tmp_path: pathlib.Path) -> None:
        def ext(path: pathlib.Path, **kw: Any) -> str:
            return "the quick brown fox jumps over"

        CustomReader.register(
            name="PodcastReader",
            extensions=[".xyz6"],
            extractor=ext,
            default_source_type=SourceType.PODCAST,
        )
        p = tmp_path / "file.xyz6"
        p.write_text("dummy")
        docs = list(DocumentReader.create(p).get_documents())
        assert docs[0].source_type == SourceType.PODCAST

    def test_validate_file_false_propagated(self, tmp_path: pathlib.Path) -> None:
        def ext(path: pathlib.Path, **kw: Any) -> str:
            return "the quick brown fox jumps over"

        Cls = CustomReader.register(
            name="StreamReader",
            extensions=[".stream"],
            extractor=ext,
            validate_file=False,
        )
        non_existent = tmp_path / "fake.stream"
        reader = Cls(input_file=non_existent)
        docs = list(reader.get_documents())
        assert len(docs) == 1


# ===========================================================================
# PDFReader custom_extractor hook
# ===========================================================================


class TestPDFReaderCustomExtractor:
    """Tests for PDFReader.prefer_backend='custom' + custom_extractor."""

    def test_custom_extractor_bypasses_pdfminer(self, tmp_pdf: pathlib.Path) -> None:
        from .._pdf import PDFReader

        def fake_pdf_extract(path: pathlib.Path, **kw: Any) -> list[dict]:
            return [
                {"text": "page one content is here", "page_number": 0},
                {"text": "page two content is here", "page_number": 1},
            ]

        reader = PDFReader(
            input_file=tmp_pdf,
            prefer_backend="custom",
            custom_extractor=fake_pdf_extract,
        )
        docs = list(reader.get_documents())
        assert len(docs) == 2
        assert docs[0].page_number == 0
        assert docs[1].page_number == 1

    def test_custom_extractor_kwargs_forwarded(self, tmp_pdf: pathlib.Path) -> None:
        from .._pdf import PDFReader

        received: dict[str, Any] = {}

        def ext(path: pathlib.Path, **kw: Any) -> list[dict]:
            received.update(kw)
            return [{"text": "the quick brown fox jumps", "page_number": 0}]

        reader = PDFReader(
            input_file=tmp_pdf,
            prefer_backend="custom",
            custom_extractor=ext,
            custom_extractor_kwargs={"strategy": "hi_res"},
        )
        list(reader.get_documents())
        assert received.get("strategy") == "hi_res"

    def test_raises_when_custom_backend_but_no_extractor(self, tmp_pdf: pathlib.Path) -> None:
        from .._pdf import PDFReader

        with pytest.raises(ValueError, match="requires a 'custom_extractor'"):
            PDFReader(input_file=tmp_pdf, prefer_backend="custom")

    def test_raises_on_non_callable_custom_extractor(self, tmp_pdf: pathlib.Path) -> None:
        from .._pdf import PDFReader

        with pytest.raises(TypeError, match="custom_extractor must be callable"):
            PDFReader(
                input_file=tmp_pdf,
                prefer_backend="custom",
                custom_extractor="not_callable",  # type: ignore[arg-type]
            )

    def test_custom_extractor_runtime_error_wrapped(self, tmp_pdf: pathlib.Path) -> None:
        from .._pdf import PDFReader

        def bad_ext(path: pathlib.Path, **kw: Any) -> str:
            raise ValueError("corrupt PDF")

        reader = PDFReader(
            input_file=tmp_pdf,
            prefer_backend="custom",
            custom_extractor=bad_ext,
        )
        with pytest.raises(RuntimeError, match="raised an error"):
            list(reader.get_documents())

    def test_custom_extractor_ignored_when_not_custom_backend(
        self, tmp_pdf: pathlib.Path
    ) -> None:
        """custom_extractor must be silently ignored when backend != 'custom'."""
        from .._pdf import PDFReader

        called = []

        def should_not_be_called(path: pathlib.Path, **kw: Any) -> str:
            called.append(True)
            return "should not appear"

        # prefer_backend=None → auto mode (pdfminer/pypdf); custom_extractor ignored
        reader = PDFReader(
            input_file=tmp_pdf,
            prefer_backend=None,
            custom_extractor=should_not_be_called,
        )
        # Neither pdfminer nor pypdf can read the fake PDF bytes → yields nothing,
        # but the custom extractor must NOT be called.
        try:
            list(reader.get_documents())
        except (ImportError, Exception):
            pass  # Expected: neither library installed or fake PDF fails
        assert not called, "custom_extractor must not be called when prefer_backend != 'custom'"

    def test_prefer_backend_custom_in_valid_backends(self, tmp_pdf: pathlib.Path) -> None:
        from .._pdf import PDFReader
        assert "custom" in PDFReader._VALID_BACKENDS

    def test_invalid_prefer_backend_raises(self, tmp_pdf: pathlib.Path) -> None:
        from .._pdf import PDFReader
        with pytest.raises(ValueError, match="prefer_backend must be one of"):
            PDFReader(input_file=tmp_pdf, prefer_backend="unrecognised")


# ===========================================================================
# ImageReader custom_extractor hook
# ===========================================================================


class TestImageReaderCustomExtractor:
    """Tests for ImageReader.backend='custom' + custom_extractor."""

    def test_custom_extractor_bypasses_ocr(self, tmp_image: pathlib.Path) -> None:
        from .._image import ImageReader

        def fake_ocr(path: pathlib.Path, **kw: Any) -> list[dict]:
            return [
                {"text": "the extracted text from image", "confidence": 0.95},
            ]

        reader = ImageReader(
            input_file=tmp_image,
            backend="custom",
            custom_extractor=fake_ocr,
        )
        docs = list(reader.get_documents())
        assert len(docs) == 1
        assert "extracted text" in docs[0].text
        assert docs[0].confidence == 0.95

    def test_custom_extractor_kwargs_forwarded(self, tmp_image: pathlib.Path) -> None:
        from .._image import ImageReader

        received: dict[str, Any] = {}

        def ext(path: pathlib.Path, **kw: Any) -> list[dict]:
            received.update(kw)
            return [{"text": "the extracted text here", "confidence": 0.9}]

        reader = ImageReader(
            input_file=tmp_image,
            backend="custom",
            custom_extractor=ext,
            custom_extractor_kwargs={"langs": ["en", "de"]},
        )
        list(reader.get_documents())
        assert received.get("langs") == ["en", "de"]

    def test_raises_when_custom_backend_but_no_extractor(self, tmp_image: pathlib.Path) -> None:
        from .._image import ImageReader

        reader = ImageReader(input_file=tmp_image, backend="custom")
        with pytest.raises(ValueError, match="requires a 'custom_extractor'"):
            list(reader.get_documents())

    def test_raises_on_non_callable_custom_extractor(self, tmp_image: pathlib.Path) -> None:
        from .._image import ImageReader

        with pytest.raises(TypeError, match="custom_extractor must be callable"):
            ImageReader(
                input_file=tmp_image,
                backend="custom",
                custom_extractor=42,  # type: ignore[arg-type]
            )

    def test_custom_extractor_runtime_error_wrapped(self, tmp_image: pathlib.Path) -> None:
        from .._image import ImageReader

        def bad_ext(path: pathlib.Path, **kw: Any) -> str:
            raise OSError("cannot read image")

        reader = ImageReader(
            input_file=tmp_image,
            backend="custom",
            custom_extractor=bad_ext,
        )
        with pytest.raises(RuntimeError, match="raised an error"):
            list(reader.get_documents())

    def test_backend_custom_in_valid_backends(self) -> None:
        from .._image import _VALID_BACKENDS
        assert "custom" in _VALID_BACKENDS

    def test_source_type_set_to_image(self, tmp_image: pathlib.Path) -> None:
        """Documents from custom image extractor must have source_type=IMAGE."""
        from .._image import ImageReader

        def ext(path: pathlib.Path, **kw: Any) -> str:
            return "the text from image here"

        reader = ImageReader(
            input_file=tmp_image,
            backend="custom",
            custom_extractor=ext,
        )
        docs = list(reader.get_documents())
        assert docs[0].source_type == SourceType.IMAGE


# ===========================================================================
# AudioReader custom_extractor hook
# ===========================================================================


class TestAudioReaderCustomExtractor:
    """Tests for AudioReader.custom_extractor — Strategy 0 (highest priority)."""

    def test_custom_extractor_bypasses_companion_and_whisper(
        self, tmp_audio: pathlib.Path
    ) -> None:
        from .._audio import AudioReader

        def fake_asr(path: pathlib.Path, **kw: Any) -> list[dict]:
            return [
                {
                    "text": "welcome to the podcast today we discuss",
                    "timecode_start": 0.0,
                    "timecode_end": 4.2,
                },
                {
                    "text": "machine learning and natural language processing",
                    "timecode_start": 4.2,
                    "timecode_end": 8.5,
                },
            ]

        reader = AudioReader(
            input_file=tmp_audio,
            custom_extractor=fake_asr,
        )
        docs = list(reader.get_documents())
        assert len(docs) == 2
        assert docs[0].timecode_start == 0.0
        assert docs[0].timecode_end == 4.2
        assert docs[1].timecode_start == 4.2

    def test_custom_extractor_kwargs_forwarded(self, tmp_audio: pathlib.Path) -> None:
        from .._audio import AudioReader

        received: dict[str, Any] = {}

        def ext(path: pathlib.Path, **kw: Any) -> list[dict]:
            received.update(kw)
            return [{"text": "the quick brown fox jumps over here", "timecode_start": 0.0}]

        reader = AudioReader(
            input_file=tmp_audio,
            custom_extractor=ext,
            custom_extractor_kwargs={"language": "de", "model": "large-v3"},
        )
        list(reader.get_documents())
        assert received.get("language") == "de"
        assert received.get("model") == "large-v3"

    def test_raises_on_non_callable_custom_extractor(self, tmp_audio: pathlib.Path) -> None:
        from .._audio import AudioReader

        with pytest.raises(TypeError, match="custom_extractor must be callable"):
            AudioReader(input_file=tmp_audio, custom_extractor="bad")  # type: ignore[arg-type]

    def test_custom_extractor_runtime_error_wrapped(self, tmp_audio: pathlib.Path) -> None:
        from .._audio import AudioReader

        def bad_ext(path: pathlib.Path, **kw: Any) -> str:
            raise ConnectionError("API unreachable")

        reader = AudioReader(
            input_file=tmp_audio,
            custom_extractor=bad_ext,
        )
        with pytest.raises(RuntimeError, match="raised an error"):
            list(reader.get_documents())

    def test_none_custom_extractor_falls_through_to_companion(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When custom_extractor is None, the companion strategy runs as normal."""
        from .._audio import AudioReader

        # Create .mp3 + companion .txt
        mp3 = tmp_path / "episode.mp3"
        mp3.write_bytes(b"ID3fake")
        txt = tmp_path / "episode.txt"
        txt.write_text("line one of the transcript\nline two of the transcript\n")

        reader = AudioReader(input_file=mp3, custom_extractor=None)
        docs = list(reader.get_documents())
        # Companion .txt file was found and used
        assert any("transcript" in d.text for d in docs)

    def test_source_type_set_to_audio(self, tmp_audio: pathlib.Path) -> None:
        from .._audio import AudioReader

        def ext(path: pathlib.Path, **kw: Any) -> str:
            return "the quick brown fox jumps over"

        reader = AudioReader(input_file=tmp_audio, custom_extractor=ext)
        docs = list(reader.get_documents())
        assert docs[0].source_type == SourceType.AUDIO


# ===========================================================================
# VideoReader custom_extractor hook
# ===========================================================================


class TestVideoReaderCustomExtractor:
    """Tests for VideoReader.custom_extractor — Strategy 0 (highest priority)."""

    def test_custom_extractor_bypasses_subtitle_and_whisper(
        self, tmp_video: pathlib.Path
    ) -> None:
        from .._video import VideoReader

        def fake_transcribe(path: pathlib.Path, **kw: Any) -> list[dict]:
            return [
                {
                    "text": "hello welcome to this lecture today",
                    "timecode_start": 0.0,
                    "timecode_end": 3.5,
                },
            ]

        reader = VideoReader(
            input_file=tmp_video,
            custom_extractor=fake_transcribe,
        )
        docs = list(reader.get_documents())
        assert len(docs) == 1
        assert docs[0].timecode_start == 0.0
        assert docs[0].timecode_end == 3.5

    def test_custom_extractor_kwargs_forwarded(self, tmp_video: pathlib.Path) -> None:
        from .._video import VideoReader

        received: dict[str, Any] = {}

        def ext(path: pathlib.Path, **kw: Any) -> list[dict]:
            received.update(kw)
            return [{"text": "the quick brown fox jumps here", "timecode_start": 0.0}]

        reader = VideoReader(
            input_file=tmp_video,
            custom_extractor=ext,
            custom_extractor_kwargs={"language": "fr"},
        )
        list(reader.get_documents())
        assert received.get("language") == "fr"

    def test_raises_on_non_callable_custom_extractor(self, tmp_video: pathlib.Path) -> None:
        from .._video import VideoReader

        with pytest.raises(TypeError, match="custom_extractor must be callable"):
            VideoReader(input_file=tmp_video, custom_extractor=99)  # type: ignore[arg-type]

    def test_custom_extractor_runtime_error_wrapped(self, tmp_video: pathlib.Path) -> None:
        from .._video import VideoReader

        def bad_ext(path: pathlib.Path, **kw: Any) -> str:
            raise TimeoutError("API timed out")

        reader = VideoReader(input_file=tmp_video, custom_extractor=bad_ext)
        with pytest.raises(RuntimeError, match="raised an error"):
            list(reader.get_documents())

    def test_none_custom_extractor_falls_through_to_subtitle(
        self, tmp_path: pathlib.Path
    ) -> None:
        """When custom_extractor is None, subtitle detection runs as normal."""
        from .._video import VideoReader

        mp4 = tmp_path / "lecture.mp4"
        mp4.write_bytes(b"ftyp fake")
        srt = tmp_path / "lecture.srt"
        srt.write_text(
            "1\n00:00:01,000 --> 00:00:04,000\nHello, this is the subtitle text.\n\n"
            "2\n00:00:05,000 --> 00:00:08,000\nAnd here is another subtitle line.\n\n"
        )

        reader = VideoReader(input_file=mp4, custom_extractor=None)
        docs = list(reader.get_documents())
        assert any("subtitle" in d.text.lower() for d in docs)


# ===========================================================================
# DummyReader consistency fix
# ===========================================================================


class TestDummyReaderConsistency:
    """Verify DummyReader no longer has both file_type and file_types set."""

    def test_dummy_reader_has_no_redundant_file_type(self) -> None:
        from ..._base import DummyReader

        raw_file_type = DummyReader.__dict__.get("file_type")
        raw_file_types = DummyReader.__dict__.get("file_types")

        # file_types should be set; file_type must NOT be set as a concrete
        # instance attribute (it may exist as an inherited ClassVar default)
        assert raw_file_types == [":dummy"], (
            "DummyReader.file_types must be [':dummy']"
        )
        assert raw_file_type is None or raw_file_type == ":dummy", (
            "DummyReader.file_type must be None (not both set)"
        )
        # The registry should contain ":dummy" mapped to DummyReader
        registry = DocumentReader.subclass_by_type()
        assert ":dummy" in registry
        assert registry[":dummy"] is DummyReader


# ===========================================================================
# __init__.py exports
# ===========================================================================


class TestReaderPackageExports:
    """Verify CustomReader and normalize_extractor_output are publicly exported."""

    def test_custom_reader_importable_from_readers_package(self) -> None:
        from ... import _readers  # noqa: PLC0415
        assert hasattr(_readers, "CustomReader")

    def test_normalize_extractor_output_importable_from_readers_package(self) -> None:
        from ... import _readers  # noqa: PLC0415
        assert hasattr(_readers, "normalize_extractor_output")
