"""
scikitplot.corpus._readers._image
=================================
OCR-based text extraction from raster image files.

Supported formats: PNG, JPEG/JPG, GIF, WEBP, TIFF/TIF, BMP.

Backend chain
-------------
1. **pytesseract** (primary) — wraps Google's Tesseract OCR engine.
   Installed via ``pip install pytesseract``; requires ``tesseract``
   binary on ``PATH``.  Returns per-word confidence scores that are
   aggregated to a chunk-level mean confidence.
2. **easyocr** (secondary) — deep-learning OCR; slower but often more
   accurate on low-quality scans and non-Latin scripts.
   Installed via ``pip install easyocr``.
3. **PIL-only stub** — if neither OCR library is available, the reader
   raises ``ImportError`` with actionable install instructions rather
   than silently yielding empty text.

Multi-frame support
-------------------
GIF and TIFF files may contain multiple frames or pages. Each frame is
extracted and yielded as a separate chunk with a ``page_number`` metadata
key. Single-frame images yield exactly one chunk.

Design
------
* **Lazy imports** — neither ``PIL``, ``pytesseract``, nor ``easyocr``
  is imported at module level. ``ImportError`` only fires at first
  ``get_raw_chunks()`` call, never at import time.
* **CI/Docker safety** — no network calls, no model downloads at import
  time. ``easyocr`` lazily downloads its model weights on first use;
  callers must explicitly opt in via ``backend="easyocr"``.
* **Section type** — all chunks are labelled
  :attr:`~scikitplot.corpus._schema.SectionType.TEXT`. Callers that need
  finer-grained section detection (title vs body) can post-process
  ``metadata["image_width"]`` and ``metadata["image_height"]``.

Python compatibility
--------------------
Python 3.8-3.15. ``Pillow``, ``pytesseract``, and ``easyocr`` are
all optional runtime dependencies.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Generator, List, Optional, Tuple  # noqa: F401

from scikitplot.corpus._base import DocumentReader
from scikitplot.corpus._schema import SectionType, SourceType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Recognised image extensions registered by this reader
# ---------------------------------------------------------------------------
_IMAGE_EXTENSIONS: list[str] = [
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".tiff",
    ".tif",
    ".bmp",
]

# ---------------------------------------------------------------------------
# OCR backend identifiers
# ---------------------------------------------------------------------------
_BACKEND_TESSERACT = "tesseract"
_BACKEND_EASYOCR = "easyocr"
_VALID_BACKENDS: Tuple[str, ...] = (_BACKEND_TESSERACT, _BACKEND_EASYOCR)  # noqa: UP006


def _ocr_tesseract(
    image: Any,
    lang: str | None,
) -> tuple[str, float]:
    """
    Run Tesseract OCR on a PIL Image and return ``(text, mean_confidence)``.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to process. Must already be converted to RGB or L mode.
    lang : str or None
        Tesseract language string (e.g. ``"eng"``, ``"deu"``, ``"eng+deu"``).
        ``None`` uses Tesseract's default (usually English).

    Returns
    -------
    text : str
        Extracted text with newlines preserved.
    mean_confidence : float
        Mean confidence score normalised to ``[0.0, 1.0]``. Returns ``0.0``
        if no words were detected. Tesseract reports ``[0, 100]`` internally;
        this function divides by 100 so both backends use the same scale.

    Raises
    ------
    ImportError
        If ``pytesseract`` is not installed.
    RuntimeError
        If the Tesseract binary is not found on ``PATH``.
    """
    try:
        import pytesseract  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pytesseract is required for ImageReader with backend='tesseract'."
            " Install it with:\n"
            "  pip install pytesseract\n"
            "And install the Tesseract binary:\n"
            "  apt-get install tesseract-ocr   # Debian/Ubuntu\n"
            "  brew install tesseract           # macOS"
        ) from exc

    kwargs: dict[str, Any] = {}
    if lang is not None:
        kwargs["lang"] = lang

    try:
        # image_to_data returns per-word confidence; image_to_string is simpler
        # but loses confidence info. We call image_to_data for confidence then
        # image_to_string for clean text with layout preservation.
        text = pytesseract.image_to_string(image, **kwargs)

        # Compute mean confidence from data output
        import pandas as pd  # noqa: F401, PLC0415

        data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DATAFRAME, **kwargs
        )
        conf_col = data["conf"]
        # Tesseract uses -1 for non-word entries; filter those out.
        valid_conf = conf_col[conf_col >= 0]
        # Tesseract reports confidence in [0, 100]; normalise to [0.0, 1.0]
        # so both backends use the same scale.  CorpusDocument.confidence
        # is validated to be in [0.0, 1.0].
        raw_mean = float(valid_conf.mean()) if len(valid_conf) > 0 else 0.0
        mean_conf = raw_mean / 100.0

    except ImportError:
        # pandas not available — skip confidence computation
        text = pytesseract.image_to_string(image, **kwargs)
        mean_conf = 0.0  # unknown confidence; 0.0 is safe default in [0.0, 1.0]
        logger.debug("ImageReader: pandas not installed; OCR confidence not computed.")

    return text, mean_conf


def _ocr_easyocr(
    image: Any,
    lang: str | None,
    easyocr_reader: Any | None,
) -> tuple[str, float, Any]:
    """
    Run easyocr OCR on a PIL Image and return ``(text, mean_confidence, reader)``.

    Parameters
    ----------
    image : PIL.Image.Image
        Image to process.
    lang : str or None
        Language code list string for easyocr, e.g. ``"en"`` or ``"en,de"``.
        Defaults to ``"en"`` if ``None``.
    easyocr_reader : easyocr.Reader or None
        Pre-constructed easyocr ``Reader`` instance (for caching). If
        ``None``, a new ``Reader`` is created for the given language.
        The created reader is returned as the third element so the caller
        can cache it and avoid reloading model weights for subsequent frames.

    Returns
    -------
    text : str
        Extracted text joined with newlines.
    mean_confidence : float
        Mean confidence score in ``[0.0, 1.0]``. Returns ``0.0`` when no
        text regions are detected. easyocr natively returns ``[0.0, 1.0]``;
        no normalisation required.
    reader : easyocr.Reader
        The ``Reader`` instance used (created or the one passed in). Callers
        MUST cache this and pass it back on subsequent calls to avoid
        reloading model weights (~100 MB per language) on every frame.

    Raises
    ------
    ImportError
        If ``easyocr`` or ``numpy`` is not installed.
    """
    try:
        import easyocr  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "easyocr is required for ImageReader with backend='easyocr'."
            " Install it with:\n"
            "  pip install easyocr"
        ) from exc

    # easyocr accepts a language list; default to English
    lang_list = [lang] if lang else ["en"]

    reader_obj = easyocr_reader
    if reader_obj is None:
        logger.debug(
            "ImageReader: creating easyocr.Reader for languages %s.", lang_list
        )
        reader_obj = easyocr.Reader(lang_list)

    # Convert PIL image to numpy array for easyocr
    img_array = np.array(image)
    results = reader_obj.readtext(img_array)

    if not results:
        return "", 0.0, reader_obj

    # results is list of (bounding_box, text, confidence)
    texts = [r[1] for r in results]
    confidences = [r[2] for r in results]
    text = "\n".join(texts)
    mean_conf = float(sum(confidences) / len(confidences))

    return text, mean_conf, reader_obj


@dataclass
class ImageReader(DocumentReader):
    """
    OCR-based text extraction from raster image files.

    Iterates over all frames in the image (for multi-frame GIF and TIFF),
    runs OCR on each, and yields one raw chunk dict per frame. Single-frame
    images yield exactly one chunk.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the image file.
    backend : str, optional
        OCR backend to use. One of ``"tesseract"`` (default) or
        ``"easyocr"``. ``"tesseract"`` requires the Tesseract binary on
        ``PATH`` plus ``pip install pytesseract``.  ``"easyocr"``
        requires ``pip install easyocr`` and downloads model weights on
        first use.
    ocr_lang : str or None, optional
        Language hint for the OCR engine.

        - For Tesseract: standard Tesseract language string, e.g.
          ``"eng"``, ``"deu"``, ``"eng+deu"``.
        - For easyocr: ISO 639-1 language code, e.g. ``"en"``, ``"de"``.

        ``None`` uses each backend's default (usually English).
    min_confidence : float, optional
        Minimum mean OCR confidence in ``[0.0, 1.0]`` (both backends).
        Chunks with lower confidence are still yielded but logged at
        ``DEBUG`` with a warning flag. Set to ``None`` to disable
        threshold-based logging. Default: ``None``.
    max_file_bytes : int, optional
        Maximum file size in bytes. Files larger than this limit raise
        ``ValueError`` before any bytes are read. Default: 100 MB.
    preprocess_grayscale : bool, optional
        When ``True``, convert each frame to grayscale (``"L"`` mode)
        before OCR. This often improves Tesseract accuracy on coloured
        backgrounds. Default: ``False``.
    chunker : ChunkerBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filter_ : FilterBase or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    filename_override : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.
    default_language : str or None, optional
        Inherited from :class:`~scikitplot.corpus._base.DocumentReader`.

    Attributes
    ----------
    file_types : list of str
        Class variable. Registered extensions:
        ``[".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff", ".tif", ".bmp"]``.

    Raises
    ------
    ValueError
        If ``backend`` is not one of the supported values.
    ValueError
        If the file exceeds ``max_file_bytes``.
    ImportError
        If the required OCR library (or Pillow) is not installed.

    See Also
    --------
    scikitplot.corpus._readers.VideoReader : Video transcription reader.
    scikitplot.corpus._readers.PDFReader : PDF text extraction reader.

    Notes
    -----
    **Tesseract accuracy tips:**

    - Use ``preprocess_grayscale=True`` for images with coloured text
      backgrounds.
    - Install additional Tesseract language packs for non-English corpora:
      ``apt-get install tesseract-ocr-deu`` (German), etc.
    - Very low-resolution images (< 150 DPI) tend to produce poor results.
      Consider upscaling with Pillow before passing to the reader.

    **easyocr note:** Model weights (~100 MB per language) are downloaded
    automatically on first use. This is a side effect; in CI/Docker
    pipelines, pre-cache the weights or use Tesseract instead.

    Examples
    --------
    Default Tesseract backend:

    >>> from pathlib import Path
    >>> reader = ImageReader(input_file=Path("scan.png"), ocr_lang="eng")
    >>> docs = list(reader.get_documents())
    >>> print(docs[0].text[:100])

    Multi-page TIFF:

    >>> reader = ImageReader(input_file=Path("document.tiff"))
    >>> docs = list(reader.get_documents())
    >>> print(f"Extracted {len(docs)} pages")
    """

    file_types: ClassVar[list[str]] = _IMAGE_EXTENSIONS

    backend: str = field(default=_BACKEND_TESSERACT)
    """OCR backend. One of ``"tesseract"`` (default) or ``"easyocr"``."""

    ocr_lang: str | None = field(default=None)
    """Language hint for the OCR engine. ``None`` uses the backend default."""

    min_confidence: float | None = field(default=None)
    """Minimum OCR confidence for debug logging. ``None`` disables logging."""

    max_file_bytes: int = field(default=100 * 1024 * 1024)
    """Maximum file size in bytes. Default: 100 MB."""

    preprocess_grayscale: bool = field(default=False)
    """Convert frames to grayscale before OCR when ``True``."""

    # Internal: easyocr Reader instance (cached across frames)
    _easyocr_reader: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:  # noqa: D105
        super().__post_init__()
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"ImageReader: backend must be one of {_VALID_BACKENDS};"
                f" got {self.backend!r}."
            )
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"ImageReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Run OCR on each frame of the image and yield one chunk per frame.

        Yields
        ------
        dict
            Keys:

            ``"source_type"``
                Always :attr:`~scikitplot.corpus._schema.SourceType.IMAGE`.
                Promoted to :attr:`CorpusDocument.source_type`.
            ``"text"``
                OCR-extracted text for this frame.
            ``"section_type"``
                Always :attr:`~scikitplot.corpus._schema.SectionType.TEXT`.
            ``"page_number"``
                Zero-based frame/page index within the file.
                Promoted to :attr:`CorpusDocument.page_number`.
            ``"image_width"``
                Width of the frame in pixels (goes to ``metadata``).
            ``"image_height"``
                Height of the frame in pixels (goes to ``metadata``).
            ``"confidence"``
                Mean confidence score in ``[0.0, 1.0]`` (both backends).
                Promoted to :attr:`CorpusDocument.confidence`.
            ``"ocr_engine"``
                Name of the backend that produced this chunk.
                Promoted to :attr:`CorpusDocument.ocr_engine`.
            ``"total_frames"``
                Total number of frames/pages in the image file
                (goes to ``metadata``).

        Raises
        ------
        ValueError
            If the file exceeds ``max_file_bytes``.
        ImportError
            If Pillow or the OCR library is not installed.
        """
        try:
            from PIL import Image  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for ImageReader."
                " Install it with:\n"
                "  pip install Pillow"
            ) from exc

        file_size = self.input_file.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"ImageReader: {self.file_name} is {file_size:,} bytes,"
                f" which exceeds max_file_bytes={self.max_file_bytes:,}."
            )

        logger.info(
            "ImageReader: opening %s (backend=%s).", self.file_name, self.backend
        )

        with Image.open(self.input_file) as img:
            frames = self._extract_frames(img)
            total = len(frames)
            logger.debug("ImageReader: %d frame(s) in %s.", total, self.file_name)

            for page_idx, frame in enumerate(frames):
                if self.preprocess_grayscale and frame.mode != "L":
                    frame = frame.convert("L")  # noqa: PLW2901
                elif frame.mode not in ("RGB", "RGBA", "L"):
                    frame = frame.convert("RGB")  # noqa: PLW2901

                width, height = frame.size

                text, confidence = self._run_ocr(frame)

                if not text.strip():
                    logger.debug(
                        "ImageReader: frame %d of %s yielded no text.",
                        page_idx,
                        self.file_name,
                    )
                    continue

                if self.min_confidence is not None and confidence < self.min_confidence:
                    logger.debug(
                        "ImageReader: frame %d confidence %.1f < min %.1f.",
                        page_idx,
                        confidence,
                        self.min_confidence,
                    )

                yield {
                    "text": text,
                    "section_type": SectionType.TEXT.value,
                    # promoted → CorpusDocument.source_type
                    "source_type": SourceType.IMAGE.value,
                    # promoted → CorpusDocument.page_number
                    "page_number": page_idx,
                    "image_width": width,  # non-promoted → metadata
                    "image_height": height,  # non-promoted → metadata
                    "confidence": round(
                        confidence, 4
                    ),  # promoted → CorpusDocument.confidence
                    "ocr_engine": self.backend,  # promoted → CorpusDocument.ocr_engine
                    "total_frames": total,  # non-promoted → metadata
                }

        logger.info(
            "ImageReader: finished %s — %d frame(s) processed.",
            self.file_name,
            total,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_frames(self, img: Any) -> list[Any]:
        """
        Extract all frames from a potentially multi-frame image.

        Parameters
        ----------
        img : PIL.Image.Image
            Open image object.

        Returns
        -------
        list of PIL.Image.Image
            One entry per frame/page. For single-frame formats,
            returns a list with one element.
        """
        frames = []
        try:
            while True:
                # Copy frame so it remains valid after the context exits
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass  # Reached end of frames — expected
        except AttributeError:
            # Format doesn't support seek (single-frame)
            frames = [img.copy()]
        return frames

    def _run_ocr(self, frame: Any) -> tuple[str, float]:
        """
        Dispatch to the configured OCR backend.

        Parameters
        ----------
        frame : PIL.Image.Image
            Single image frame to OCR.

        Returns
        -------
        text : str
            Extracted text.
        confidence : float
            Mean OCR confidence in ``[0.0, 1.0]``.

        Notes
        -----
        For the ``easyocr`` backend, the ``easyocr.Reader`` instance is
        created on the first call and cached in ``self._easyocr_reader``
        via ``object.__setattr__`` (required because the parent dataclass
        is frozen-compatible).  Subsequent calls reuse the cached reader,
        avoiding the ~100 MB per-language model-weight reload on every frame.
        """
        if self.backend == _BACKEND_TESSERACT:
            return _ocr_tesseract(frame, self.ocr_lang)
        # easyocr path — unpack 3-tuple (text, confidence, reader)
        text, conf, reader_obj = _ocr_easyocr(
            frame, self.ocr_lang, self._easyocr_reader
        )
        # Cache the reader after first creation so subsequent frames reuse it.
        # object.__setattr__ bypasses dataclass field-assignment restrictions.
        if self._easyocr_reader is None:
            object.__setattr__(self, "_easyocr_reader", reader_obj)
        return text, conf


__all__ = ["ImageReader"]
