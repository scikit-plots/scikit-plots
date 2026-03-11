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

# ---------------------------------------------------------------------------
# Relative import helpers
# ---------------------------------------------------------------------------
# Tests are run from the repo root with:
#   pytest corpus/_readers/tests/test_image.py
# We add the package root to sys.path so relative imports resolve.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]  # …/corpus_src
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Stub out scikitplot.corpus so the reader module can be imported without the
# full package present.  We only need _schema constants and _base.DocumentReader.
def _build_stubs() -> None:
    """Install minimal stubs for scikitplot.corpus._schema and _base."""
    # ---- stub scikitplot package hierarchy ----
    for name in ("scikitplot", "scikitplot.corpus"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

    # ---- _schema stub ----
    import importlib.util as ilu

    schema_spec = ilu.spec_from_file_location(
        "scikitplot.corpus._schema",
        str(_REPO_ROOT / "corpus" / "_schema.py"),
    )
    schema_mod = ilu.module_from_spec(schema_spec)
    sys.modules["scikitplot.corpus._schema"] = schema_mod
    schema_spec.loader.exec_module(schema_mod)

    # ---- _base stub (minimal DocumentReader) ----
    base_spec = ilu.spec_from_file_location(
        "scikitplot.corpus._base",
        str(_REPO_ROOT / "corpus" / "_base.py"),
    )
    base_mod = ilu.module_from_spec(base_spec)
    sys.modules["scikitplot.corpus._base"] = base_mod
    base_spec.loader.exec_module(base_mod)


_build_stubs()

# Now import the module under test using its package path
from .._image import (  # noqa: E402
    ImageReader,
    _BACKEND_EASYOCR,
    _BACKEND_TESSERACT,
    _VALID_BACKENDS,
    _ocr_easyocr,
    _ocr_tesseract,
)
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
    return ImageReader(input_file=img_file, backend=backend, **kwargs)


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
        reader = ImageReader(input_file=img_file, max_file_bytes=5)

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
