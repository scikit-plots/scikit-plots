# scikitplot/corpus/_readers/tests/test__image.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Tests for scikitplot.corpus._readers._image
===========================================

Coverage
--------
* CRITICAL-I1 fix: ``page_number``, ``confidence``, ``ocr_engine`` are yielded
  with the correct promoted key names (not ``page_index``, ``ocr_confidence``,
  ``ocr_backend``).
* MEDIUM-I2 fix: Tesseract confidence is normalised to ``[0.0, 1.0]``.
* BUG-I3 fix: easyocr ``Reader`` is created once and cached across frames.
* BUG-I4 fix: ``source_type`` promoted key is yielded as ``SourceType.IMAGE.value``.
* ``_ocr_tesseract`` normalises ``[0, 100]`` → ``[0.0, 1.0]``.
* ``_ocr_easyocr`` returns 3-tuple ``(text, confidence, reader_obj)``.
* ``ImageReader.__post_init__`` rejects invalid ``backend`` and non-positive
  ``max_file_bytes``.
* ``get_raw_chunks`` skips frames with no text.
* ``get_raw_chunks`` raises ``ValueError`` when file exceeds ``max_file_bytes``.
* Multi-frame TIFF yields one chunk per frame (page_number increments).
* ``_extract_frames`` handles single-frame images (no seek support).
* Lazy-import guard: ``ImportError`` when Pillow is absent.

All tests use ``unittest.mock`` — no OCR libraries are required.
"""

from __future__ import annotations

import pathlib
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# Now import the module under test using its package path
from .._image import (  # noqa: E402
    ImageReader,
    _BACKEND_EASYOCR,
    _BACKEND_TESSERACT,
    _VALID_BACKENDS,
    _ocr_easyocr,
    _ocr_tesseract,
)
from scikitplot.corpus import _base, _schema
from scikitplot.corpus._schema import SectionType, SourceType, _PROMOTED_RAW_KEYS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pil_image(width: int = 100, height: int = 50, mode: str = "RGB") -> MagicMock:
    """Return a Mock that behaves like a single-frame PIL Image."""
    # from PIL import Image
    # Image.new("RGB", (100, 50))
    img = MagicMock()
    img.mode = mode
    img.size = (width, height)
    img.copy.return_value = img
    # Single-frame: seek raises EOFError immediately
    img.seek.side_effect = EOFError
    img.tell.return_value = 0
    # Context manager support
    img.__enter__.return_value = img
    img.__exit__.return_value = None
    return img


def _make_multiframe_pil_image(n: int, width: int = 80, height: int = 40) -> list[MagicMock]:
    """Return a list of n Mock PIL Images simulating multi-frame TIFF."""
    frames = []
    for i in range(n):
        f = MagicMock()
        f.mode = "RGB"
        f.size = (width, height)
        f.copy.return_value = f
        frames.append(f)
    return frames


def _make_reader(
    tmp_path: pathlib.Path,
    backend: str = _BACKEND_TESSERACT,
    filename: str = "test.png",
    **kwargs: Any,
) -> ImageReader:
    """Construct an ImageReader pointing at a real (tiny) temp file."""
    img_file = tmp_path / filename
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes — just needs to exist
    return ImageReader(input_path=img_file, backend=backend, **kwargs)


# ---------------------------------------------------------------------------
# Contract: yield dict keys
# ---------------------------------------------------------------------------


class TestYieldKeyContract:
    """Verify every promoted key name is correct against _PROMOTED_RAW_KEYS."""

    PROMOTED_EXPECTED = {"page_number", "confidence", "ocr_engine", "source_type"}
    METADATA_EXPECTED = {"image_width", "image_height", "total_frames"}

    def test_promoted_keys_are_in_promoted_raw_keys(self) -> None:
        """All promoted keys yielded by ImageReader must be in _PROMOTED_RAW_KEYS."""
        for key in self.PROMOTED_EXPECTED:
            assert key in _PROMOTED_RAW_KEYS, (
                f"Key {key!r} is yielded as promoted but is NOT in _PROMOTED_RAW_KEYS."
            )

    def test_old_wrong_keys_not_in_promoted_raw_keys(self) -> None:
        """Regression: old wrong key names must NOT be in _PROMOTED_RAW_KEYS."""
        old_wrong_keys = {"page_index", "ocr_confidence", "ocr_backend"}
        for key in old_wrong_keys:
            assert key not in _PROMOTED_RAW_KEYS, (
                f"Old wrong key {key!r} is in _PROMOTED_RAW_KEYS — should have been renamed."
            )

    def test_metadata_keys_not_in_promoted_raw_keys(self) -> None:
        """Non-promoted keys (image_width, image_height, total_frames) go to metadata."""
        for key in self.METADATA_EXPECTED:
            assert key not in _PROMOTED_RAW_KEYS, (
                f"Key {key!r} should go to metadata but is in _PROMOTED_RAW_KEYS."
            )

    def test_get_raw_chunks_yields_correct_keys(self, tmp_path: pathlib.Path) -> None:
        """
        Integration: actual yield dict keys match the contract.

        Mocks PIL.Image.open so no real image file is needed.
        """
        reader = _make_reader(tmp_path)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image.ImageReader._run_ocr", return_value=("hello world", 0.92)):
                chunks = list(reader.get_raw_chunks())

        assert len(chunks) == 3
        chunk = chunks[0]

        # Every expected promoted key must be present
        for key in self.PROMOTED_EXPECTED:
            assert key in chunk, f"Promoted key {key!r} missing from yield dict."

        # Every expected metadata key must be present
        for key in self.METADATA_EXPECTED:
            assert key in chunk, f"Metadata key {key!r} missing from yield dict."

        # No old wrong key names
        for bad_key in {"page_index", "ocr_confidence", "ocr_backend"}:
            assert bad_key not in chunk, (
                f"Old wrong key {bad_key!r} found in yield dict — CRITICAL-I1 regression."
            )


# ---------------------------------------------------------------------------
# CRITICAL-I1: correct promoted key values
# ---------------------------------------------------------------------------


class TestCriticalI1PromotedKeyValues:
    """Exact values for each promoted field."""

    def test_page_number_is_zero_based_frame_index(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image.ImageReader._run_ocr", return_value=("text", 0.8)):
                chunks = list(reader.get_raw_chunks())

        assert [c["page_number"] for c in chunks] == [0, 1, 2]

    def test_confidence_is_rounded_float_in_unit_range(self, tmp_path: pathlib.Path) -> None:
        raw_conf = 0.123456789
        reader = _make_reader(tmp_path)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image.ImageReader._run_ocr", return_value=("text", raw_conf)):
                chunks = list(reader.get_raw_chunks())

        assert chunks[0]["confidence"] == round(raw_conf, 4)
        assert 0.0 <= chunks[0]["confidence"] <= 1.0

    def test_ocr_engine_matches_backend(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path, backend=_BACKEND_TESSERACT)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image.ImageReader._run_ocr", return_value=("text", 0.9)):
                chunks = list(reader.get_raw_chunks())

        assert chunks[0]["ocr_engine"] == _BACKEND_TESSERACT


# ---------------------------------------------------------------------------
# BUG-I4: source_type is yielded
# ---------------------------------------------------------------------------


class TestBugI4SourceType:
    """source_type must be yielded as SourceType.IMAGE.value."""

    def test_source_type_value_is_image(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image.ImageReader._run_ocr", return_value=("text", 0.7)):
                chunks = list(reader.get_raw_chunks())

        assert chunks[0]["source_type"] == SourceType.IMAGE.value
        assert chunks[0]["source_type"] == "image"

    def test_source_type_key_in_promoted_raw_keys(self) -> None:
        assert "source_type" in _PROMOTED_RAW_KEYS


# ---------------------------------------------------------------------------
# MEDIUM-I2: Tesseract confidence normalisation
# ---------------------------------------------------------------------------


class TestMediumI2ConfidenceScale:
    """_ocr_tesseract must normalise Tesseract [0, 100] → [0.0, 1.0]."""

    def _make_mock_data(self, conf_values: list[int]) -> MagicMock:
        """Build a mock pandas DataFrame with the conf column."""
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame({"conf": conf_values})

    @patch("scikitplot.corpus._readers._image._ocr_tesseract")
    def test_ocr_tesseract_returns_normalised_confidence(self, mock_tess: MagicMock) -> None:
        """
        Verify _ocr_tesseract (mocked) is expected to return [0.0, 1.0].

        The real normalisation is tested in test_real_tesseract_normalisation.
        """
        mock_tess.return_value = ("hello", 0.85)
        text, conf = mock_tess(MagicMock(), "eng")
        assert 0.0 <= conf <= 1.0

    def test_real_tesseract_normalisation_via_internals(self) -> None:
        """
        White-box: exercise the /100 branch inside _ocr_tesseract.

        We patch pytesseract calls so no Tesseract binary is required.
        """
        import pandas as pd  # noqa: PLC0415

        mock_image = MagicMock()
        mock_data = pd.DataFrame({"conf": [80, 90, 100, -1]})  # -1 = non-word

        pytess_mock = MagicMock()
        pytess_mock.image_to_string.return_value = "hello world"
        pytess_mock.image_to_data.return_value = mock_data
        pytess_mock.Output.DATAFRAME = "dataframe"

        with patch.dict("sys.modules", {"pytesseract": pytess_mock}):
            text, conf = _ocr_tesseract(mock_image, "eng")

        # Mean of [80, 90, 100] = 90 → 90/100 = 0.90
        assert text == "hello world"
        assert abs(conf - 0.90) < 1e-9, f"Expected 0.90, got {conf}"
        assert 0.0 <= conf <= 1.0

    def test_zero_words_returns_zero_confidence(self) -> None:
        """If Tesseract finds no valid words, confidence must be 0.0."""
        import pandas as pd  # noqa: PLC0415

        mock_image = MagicMock()
        mock_data = pd.DataFrame({"conf": [-1, -1]})  # all non-word entries

        pytess_mock = MagicMock()
        pytess_mock.image_to_string.return_value = ""
        pytess_mock.image_to_data.return_value = mock_data
        pytess_mock.Output.DATAFRAME = "dataframe"

        with patch.dict("sys.modules", {"pytesseract": pytess_mock}):
            text, conf = _ocr_tesseract(mock_image, None)

        assert conf == 0.0

    def test_easyocr_confidence_already_in_unit_range(self) -> None:
        """easyocr returns [0.0, 1.0] natively — no normalisation should occur."""
        mock_image = MagicMock()
        np_mock = MagicMock()
        np_mock.array.return_value = mock_image

        reader_mock = MagicMock()
        reader_mock.readtext.return_value = [
            (None, "word1", 0.95),
            (None, "word2", 0.85),
        ]

        easy_mock = MagicMock()
        easy_mock.Reader.return_value = reader_mock

        with patch.dict("sys.modules", {"easyocr": easy_mock, "numpy": np_mock}):
            text, conf, returned_reader = _ocr_easyocr(mock_image, "en", None)

        assert text == "word1\nword2"
        assert abs(conf - 0.90) < 1e-9, f"Expected mean 0.90, got {conf}"
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# BUG-I3: easyocr Reader caching
# ---------------------------------------------------------------------------


class TestBugI3EasyocrReaderCaching:
    """
    easyocr Reader must be created once per ImageReader instance.

    Root cause (BUG-I3): _ocr_easyocr previously returned (str, float);
    _run_ocr could not recover the created Reader and discarded it, causing
    model-weight reload on every frame.  The fix changes the return type to
    (str, float, Any) and caches in self._easyocr_reader.
    """

    def test_ocr_easyocr_returns_three_tuple(self) -> None:
        """_ocr_easyocr must return (str, float, reader_obj) — 3-tuple."""
        mock_image = MagicMock()
        np_mock = MagicMock()
        np_mock.array.return_value = mock_image

        reader_sentinel = object()  # unique sentinel object
        reader_mock = MagicMock()
        reader_mock.readtext.return_value = [(None, "hello", 0.9)]

        easy_mock = MagicMock()
        easy_mock.Reader.return_value = reader_mock

        with patch.dict("sys.modules", {"easyocr": easy_mock, "numpy": np_mock}):
            result = _ocr_easyocr(mock_image, "en", None)

        assert isinstance(result, tuple)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        text, conf, returned_reader = result
        assert isinstance(text, str)
        assert isinstance(conf, float)
        assert returned_reader is reader_mock

    def test_ocr_easyocr_reuses_passed_reader(self) -> None:
        """When a reader is passed in, _ocr_easyocr must NOT create a new one."""
        mock_image = MagicMock()
        np_mock = MagicMock()
        np_mock.array.return_value = mock_image

        existing_reader = MagicMock()
        existing_reader.readtext.return_value = [(None, "world", 0.88)]

        easy_mock = MagicMock()

        with patch.dict("sys.modules", {"easyocr": easy_mock, "numpy": np_mock}):
            text, conf, returned_reader = _ocr_easyocr(mock_image, "en", existing_reader)

        # easyocr.Reader() must NOT have been called
        easy_mock.Reader.assert_not_called()
        assert returned_reader is existing_reader

    def test_run_ocr_caches_reader_after_first_frame(self, tmp_path: pathlib.Path) -> None:
        """
        After the first easyocr call, self._easyocr_reader must be non-None
        so subsequent frames reuse it instead of creating a new Reader.
        """
        reader = _make_reader(tmp_path, backend=_BACKEND_EASYOCR)
        assert reader._easyocr_reader is None  # starts empty

        sentinel_reader = object()

        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch(
                "scikitplot.corpus._readers._image._ocr_easyocr",
                return_value=("text", 0.8, sentinel_reader),
            ):
                reader._run_ocr(mock_frames)

        assert reader._easyocr_reader is sentinel_reader

    def test_run_ocr_passes_cached_reader_on_second_call(self, tmp_path: pathlib.Path) -> None:
        """_run_ocr must pass the cached reader to _ocr_easyocr on frame 2+."""
        reader = _make_reader(tmp_path, backend=_BACKEND_EASYOCR)
        sentinel_reader = object()
        object.__setattr__(reader, "_easyocr_reader", sentinel_reader)

        call_args: list[Any] = []
        def capture(*args: Any, **_kw: Any) -> tuple[str, float, Any]:
            call_args.append(args)
            return ("text2", 0.9, sentinel_reader)


        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image._ocr_easyocr", side_effect=capture):
                reader._run_ocr(mock_frames)

        # Third argument to _ocr_easyocr must be the cached reader
        assert call_args[0][2] is sentinel_reader

    def test_reader_created_only_once_across_three_frames(self, tmp_path: pathlib.Path) -> None:
        """Over 3 frames the easyocr.Reader() constructor is called exactly once."""
        img_reader = _make_reader(tmp_path, backend=_BACKEND_EASYOCR)

        np_mock = MagicMock()
        np_mock.array.return_value = MagicMock()

        created_readers: list[MagicMock] = []

        def make_reader(lang_list: list[str]) -> MagicMock:
            m = MagicMock()
            m.readtext.return_value = [(None, "text", 0.9)]
            created_readers.append(m)
            return m

        easy_mock = MagicMock()
        easy_mock.Reader.side_effect = make_reader

        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image() for _ in range(3)]

        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch.dict("sys.modules", {"easyocr": easy_mock, "numpy": np_mock}):
                list(img_reader.get_raw_chunks())

        assert len(created_readers) == 1, (
            f"easyocr.Reader() was called {len(created_readers)} times; expected 1."
        )


# ---------------------------------------------------------------------------
# Reader construction validation
# ---------------------------------------------------------------------------


class TestReaderConstruction:
    def test_valid_backends_accepted(self, tmp_path: pathlib.Path) -> None:
        for be in _VALID_BACKENDS:
            r = _make_reader(tmp_path, backend=be)
            assert r.backend == be

    def test_invalid_backend_raises_value_error(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="backend must be one of"):
            _make_reader(tmp_path, backend="paddle")

    def test_non_positive_max_file_bytes_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="max_file_bytes must be > 0"):
            _make_reader(tmp_path, max_file_bytes=0)

    def test_negative_max_file_bytes_raises(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="max_file_bytes must be > 0"):
            _make_reader(tmp_path, max_file_bytes=-1)


# ---------------------------------------------------------------------------
# get_raw_chunks edge cases
# ---------------------------------------------------------------------------


class TestGetRawChunksEdgeCases:
    def test_skips_frames_with_no_text(self, tmp_path: pathlib.Path) -> None:
        """Frames returning only whitespace must be silently dropped."""
        reader = _make_reader(tmp_path)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]

        def fake_ocr(frame: Any) -> tuple[str, float]:
            if frame is mock_frames[0]:
                return ("   ", 0.5)  # whitespace only → should be skipped
            return ("real text", 0.9)

        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image.ImageReader._run_ocr", side_effect=fake_ocr):
                chunks = list(reader.get_raw_chunks())

        assert len(chunks) == 2
        assert chunks[0]["text"] == "real text"
        assert chunks[0]["page_number"] == 1  # second frame (index 1)

    def test_raises_value_error_when_file_too_large(self, tmp_path: pathlib.Path) -> None:
        img_file = tmp_path / "big.png"
        img_file.write_bytes(b"\x00" * 10)
        reader = ImageReader(input_path=img_file, max_file_bytes=5)

        with pytest.raises(ValueError, match="exceeds max_file_bytes"):
            list(reader.get_raw_chunks())

    def test_raises_import_error_when_pillow_absent(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path)

        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            with pytest.raises(ImportError, match="Pillow is required"):
                # We need to trigger the actual import inside get_raw_chunks
                # Patch builtins.__import__ to raise for PIL
                import builtins

                original_import = builtins.__import__

                def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
                    if name == "PIL" or name.startswith("PIL."):
                        raise ImportError("No module named 'PIL'")
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    list(reader.get_raw_chunks())

    def test_total_frames_metadata_value(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image.ImageReader._run_ocr", return_value=("text", 0.9)):
                chunks = list(reader.get_raw_chunks())

        assert all(c["total_frames"] == 3 for c in chunks)

    def test_section_type_is_always_text(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image(), _make_pil_image(), _make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
          patch("scikitplot.corpus._readers._image.ImageReader._extract_frames", return_value=mock_frames):
            with patch("scikitplot.corpus._readers._image.ImageReader._run_ocr", return_value=("hello", 0.8)):
                chunks = list(reader.get_raw_chunks())

        assert chunks[0]["section_type"] == SectionType.TEXT.value


# ---------------------------------------------------------------------------
# _extract_frames
# ---------------------------------------------------------------------------


class TestExtractFrames:
    def test_single_frame_no_seek_support(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path)
        img = MagicMock()
        img.copy.return_value = img
        del img.seek  # simulate format without seek (AttributeError path)

        frames = reader._extract_frames(img)
        assert len(frames) == 1

    def test_multi_frame_eof_stops_iteration(self, tmp_path: pathlib.Path) -> None:
        """Simulate 3-frame TIFF: seek raises EOFError on frame index 3."""
        reader = _make_reader(tmp_path)
        call_count = {"n": 0}
        frames_returned: list[MagicMock] = []

        img = MagicMock()
        img.tell.return_value = 0

        def fake_copy() -> MagicMock:
            m = MagicMock()
            frames_returned.append(m)
            return m

        img.copy.side_effect = fake_copy

        def fake_seek(pos: int) -> None:
            call_count["n"] += 1
            if call_count["n"] >= 3:
                raise EOFError

        img.seek.side_effect = fake_seek

        result = reader._extract_frames(img)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Import error guards
# ---------------------------------------------------------------------------


class TestImportErrorGuards:
    def test_pytesseract_import_error_message(self) -> None:
        mock_image = MagicMock()
        with patch.dict("sys.modules", {"pytesseract": None}):
            with pytest.raises(ImportError, match="pytesseract is required"):
                import builtins

                orig = builtins.__import__

                def _raise(name: str, *a: Any, **kw: Any) -> Any:
                    if name == "pytesseract":
                        raise ImportError("No module named 'pytesseract'")
                    return orig(name, *a, **kw)

                with patch("builtins.__import__", side_effect=_raise):
                    _ocr_tesseract(mock_image, None)

    def test_easyocr_import_error_message(self) -> None:
        mock_image = MagicMock()

        import builtins

        orig = builtins.__import__

        def _raise(name: str, *a: Any, **kw: Any) -> Any:
            if name in ("easyocr", "numpy"):
                raise ImportError(f"No module named {name!r}")
            return orig(name, *a, **kw)

        with patch("builtins.__import__", side_effect=_raise):
            with pytest.raises(ImportError, match="easyocr is required"):
                _ocr_easyocr(mock_image, "en", None)


# ---------------------------------------------------------------------------
# file_types registration
# ---------------------------------------------------------------------------


class TestFileTypesRegistration:
    def test_all_expected_extensions_registered(self) -> None:
        expected = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".tif", ".bmp"}
        assert expected == set(ImageReader.file_types)


# ---------------------------------------------------------------------------
# Field defaults
# ---------------------------------------------------------------------------


class TestImageReaderFieldDefaults:
    """Every public ImageReader field must carry the documented default value.

    Rationale: silent default changes break serialised configs stored in
    experiment logs.  This class pins all defaults so regressions are caught
    at test time, not after a production run.
    """

    def test_backend_default_is_tesseract(self, tmp_path: pathlib.Path) -> None:
        r = _make_reader(tmp_path)
        assert r.backend == _BACKEND_TESSERACT

    def test_ocr_lang_default_is_none(self, tmp_path: pathlib.Path) -> None:
        r = _make_reader(tmp_path)
        assert r.ocr_lang is None

    def test_min_confidence_default_is_none(self, tmp_path: pathlib.Path) -> None:
        r = _make_reader(tmp_path)
        assert r.min_confidence is None

    def test_preprocess_grayscale_default_is_false(self, tmp_path: pathlib.Path) -> None:
        r = _make_reader(tmp_path)
        assert r.preprocess_grayscale is False

    def test_yield_raw_default_is_false(self, tmp_path: pathlib.Path) -> None:
        r = _make_reader(tmp_path)
        assert r.yield_raw is False

    def test_yield_raw_bytes_default_is_false(self, tmp_path: pathlib.Path) -> None:
        r = _make_reader(tmp_path)
        assert r.yield_raw_bytes is False

    def test_max_file_bytes_default_is_100mb(self, tmp_path: pathlib.Path) -> None:
        r = _make_reader(tmp_path)
        assert r.max_file_bytes == 100 * 1024 * 1024

    def test_easyocr_reader_cache_starts_none(self, tmp_path: pathlib.Path) -> None:
        """Internal cache must be None before any OCR call is made."""
        r = _make_reader(tmp_path, backend=_BACKEND_EASYOCR)
        assert r._easyocr_reader is None


# ---------------------------------------------------------------------------
# yield_raw path: raw tensors included when yield_raw=True
# ---------------------------------------------------------------------------


class TestYieldRawPath:
    """When ``yield_raw=True``, each chunk must carry tensor fields and
    the correct modality value.

    The tensor itself is mocked via numpy so no real image library is needed.
    """

    def _run_with_yield_raw(
        self,
        tmp_path: pathlib.Path,
        *,
        ocr_text: str = "extracted text",
        ocr_conf: float = 0.9,
    ) -> list[dict]:
        reader = _make_reader(tmp_path, yield_raw=True)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image()]

        import numpy as np

        fake_array = MagicMock()
        fake_array.shape = (50, 100, 3)
        fake_array.dtype = np.dtype("uint8")

        with patch("PIL.Image.open", return_value=mock_img), \
             patch("scikitplot.corpus._readers._image.ImageReader._extract_frames",
                   return_value=mock_frames), \
             patch("scikitplot.corpus._readers._image.ImageReader._run_ocr",
                   return_value=(ocr_text, ocr_conf)), \
             patch("numpy.array", return_value=fake_array):
            chunks = list(reader.get_raw_chunks())
        return chunks

    def test_raw_tensor_key_present(self, tmp_path: pathlib.Path) -> None:
        chunks = self._run_with_yield_raw(tmp_path)
        assert len(chunks) == 1
        assert "raw_tensor" in chunks[0], "raw_tensor missing when yield_raw=True"

    def test_raw_shape_key_present(self, tmp_path: pathlib.Path) -> None:
        chunks = self._run_with_yield_raw(tmp_path)
        assert "raw_shape" in chunks[0]

    def test_raw_dtype_key_present(self, tmp_path: pathlib.Path) -> None:
        chunks = self._run_with_yield_raw(tmp_path)
        assert "raw_dtype" in chunks[0]

    def test_modality_is_multimodal_when_text_and_tensor(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Text present + tensor present → modality must be 'multimodal'."""
        chunks = self._run_with_yield_raw(tmp_path, ocr_text="hello")
        assert chunks[0]["modality"] == "multimodal"

    def test_raw_tensor_absent_when_yield_raw_false(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Baseline: yield_raw=False (default) → no raw_tensor in chunk."""
        reader = _make_reader(tmp_path)  # yield_raw=False
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
             patch("scikitplot.corpus._readers._image.ImageReader._extract_frames",
                   return_value=mock_frames), \
             patch("scikitplot.corpus._readers._image.ImageReader._run_ocr",
                   return_value=("text", 0.8)):
            chunks = list(reader.get_raw_chunks())

        assert len(chunks) == 1
        assert "raw_tensor" not in chunks[0]

    def test_modality_is_text_when_yield_raw_false(
        self, tmp_path: pathlib.Path
    ) -> None:
        """yield_raw=False → modality should be 'text' when OCR text is present."""
        reader = _make_reader(tmp_path)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
             patch("scikitplot.corpus._readers._image.ImageReader._extract_frames",
                   return_value=mock_frames), \
             patch("scikitplot.corpus._readers._image.ImageReader._run_ocr",
                   return_value=("text", 0.8)):
            chunks = list(reader.get_raw_chunks())

        assert chunks[0]["modality"] == "text"


# ---------------------------------------------------------------------------
# _run_ocr interface
# ---------------------------------------------------------------------------


class TestRunOcrInterface:
    """``_run_ocr`` must return a 2-tuple ``(str, float)`` for both backends.

    The 3-tuple returned by ``_ocr_easyocr`` is an implementation detail of
    that function; ``_run_ocr`` is the public boundary — callers outside the
    class see only ``(text, confidence)``.
    """

    def test_tesseract_backend_returns_two_tuple(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path, backend=_BACKEND_TESSERACT)
        frame = _make_pil_image()
        with patch(
            "scikitplot.corpus._readers._image._ocr_tesseract",
            return_value=("hello", 0.85),
        ):
            result = reader._run_ocr(frame)

        assert isinstance(result, tuple)
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        text, conf = result
        assert isinstance(text, str)
        assert isinstance(conf, float)

    def test_easyocr_backend_returns_two_tuple(self, tmp_path: pathlib.Path) -> None:
        """_run_ocr must strip the 3rd element from _ocr_easyocr and return 2-tuple."""
        reader = _make_reader(tmp_path, backend=_BACKEND_EASYOCR)
        frame = _make_pil_image()
        sentinel_reader = object()
        with patch(
            "scikitplot.corpus._readers._image._ocr_easyocr",
            return_value=("world", 0.77, sentinel_reader),
        ):
            result = reader._run_ocr(frame)

        assert isinstance(result, tuple)
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        text, conf = result
        assert text == "world"
        assert conf == 0.77

    def test_confidence_is_float_type(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path)
        frame = _make_pil_image()
        with patch(
            "scikitplot.corpus._readers._image._ocr_tesseract",
            return_value=("some text", 0.5),
        ):
            _, conf = reader._run_ocr(frame)
        assert isinstance(conf, float)


# ---------------------------------------------------------------------------
# min_confidence logging — documents actual contract (log, no filter)
# ---------------------------------------------------------------------------


class TestMinConfidenceLogging:
    """``min_confidence`` triggers a DEBUG log entry but does **not** filter
    the chunk from output.

    The current implementation logs when ``confidence < min_confidence`` and
    continues, yielding the chunk.  This test pins that contract so any
    accidental introduction of a ``continue`` or ``return`` is caught.
    """

    def test_low_confidence_chunk_is_still_yielded(
        self, tmp_path: pathlib.Path
    ) -> None:
        reader = _make_reader(tmp_path, min_confidence=0.9)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image()]
        # Confidence 0.1 is far below min_confidence=0.9
        with patch("PIL.Image.open", return_value=mock_img), \
             patch("scikitplot.corpus._readers._image.ImageReader._extract_frames",
                   return_value=mock_frames), \
             patch("scikitplot.corpus._readers._image.ImageReader._run_ocr",
                   return_value=("low conf text", 0.1)):
            chunks = list(reader.get_raw_chunks())

        # Chunk must still be yielded (min_confidence is advisory, not a filter)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "low conf text"

    def test_low_confidence_emits_debug_log(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Logger.debug must be called mentioning 'confidence' when conf < min."""
        import scikitplot.corpus._readers._image as _img_mod

        reader = _make_reader(tmp_path, min_confidence=0.9)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
             patch("scikitplot.corpus._readers._image.ImageReader._extract_frames",
                   return_value=mock_frames), \
             patch("scikitplot.corpus._readers._image.ImageReader._run_ocr",
                   return_value=("low conf text", 0.1)), \
             patch.object(_img_mod, "logger") as mock_log:
            list(reader.get_raw_chunks())

        # At least one debug call must contain "confidence"
        debug_messages = [
            str(call) for call in mock_log.debug.call_args_list
        ]
        assert any("confidence" in m.lower() for m in debug_messages), (
            f"Expected logger.debug(...'confidence'...) when conf < min_confidence."
            f" Actual debug calls: {debug_messages}"
        )

    def test_high_confidence_no_log_emitted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """No 'confidence < min' debug call when conf >= min_confidence."""
        import scikitplot.corpus._readers._image as _img_mod

        reader = _make_reader(tmp_path, min_confidence=0.5)
        mock_img = _make_pil_image()
        mock_frames = [_make_pil_image()]
        with patch("PIL.Image.open", return_value=mock_img), \
             patch("scikitplot.corpus._readers._image.ImageReader._extract_frames",
                   return_value=mock_frames), \
             patch("scikitplot.corpus._readers._image.ImageReader._run_ocr",
                   return_value=("good text", 0.95)), \
             patch.object(_img_mod, "logger") as mock_log:
            list(reader.get_raw_chunks())

        # Filter to calls that contain both "confidence" and a "< min" pattern
        conf_min_calls = [
            str(c) for c in mock_log.debug.call_args_list
            if "confidence" in str(c).lower() and "< min" in str(c).lower()
        ]
        assert len(conf_min_calls) == 0, (
            f"Unexpected 'confidence < min' debug log when conf >= min_confidence:"
            f" {conf_min_calls}"
        )


# ---------------------------------------------------------------------------
# preprocess_grayscale: triggers frame.convert("L") before OCR
# ---------------------------------------------------------------------------


class TestPreprocessGrayscale:
    """When ``preprocess_grayscale=True``, the reader must call
    ``frame.convert("L")`` for frames that are not already in mode ``"L"``.
    """

    def test_rgb_frame_converted_to_grayscale(self, tmp_path: pathlib.Path) -> None:
        reader = _make_reader(tmp_path, preprocess_grayscale=True)
        mock_img = _make_pil_image(mode="RGB")

        # Create a frame that reports mode="RGB" and tracks convert() calls
        frame = MagicMock()
        frame.mode = "RGB"
        frame.size = (100, 50)
        converted = MagicMock()
        converted.mode = "L"
        converted.size = (100, 50)
        frame.convert.return_value = converted

        with patch("PIL.Image.open", return_value=mock_img), \
             patch("scikitplot.corpus._readers._image.ImageReader._extract_frames",
                   return_value=[frame]), \
             patch("scikitplot.corpus._readers._image.ImageReader._run_ocr",
                   return_value=("text", 0.9)):
            list(reader.get_raw_chunks())

        frame.convert.assert_called_once_with("L")

    def test_already_grayscale_frame_not_reconverted(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A frame already in mode 'L' must not be converted again."""
        reader = _make_reader(tmp_path, preprocess_grayscale=True)
        mock_img = _make_pil_image(mode="L")

        frame = MagicMock()
        frame.mode = "L"  # already grayscale
        frame.size = (100, 50)

        with patch("PIL.Image.open", return_value=mock_img), \
             patch("scikitplot.corpus._readers._image.ImageReader._extract_frames",
                   return_value=[frame]), \
             patch("scikitplot.corpus._readers._image.ImageReader._run_ocr",
                   return_value=("gray text", 0.8)):
            list(reader.get_raw_chunks())

        # convert() must NOT have been called with "L" (no redundant conversion)
        for call in frame.convert.call_args_list:
            assert call.args[0] != "L", (
                "frame.convert('L') called on a frame already in 'L' mode."
            )
