# scikitplot/corpus/_readers/_pdf.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._readers.pdf
==============================
PDF document reader for the scikitplot corpus pipeline.

Extracts text from PDF files page-by-page using a two-backend cascade:

1. **pdfminer.six** (primary) — layout-aware extraction with accurate
   reading-order reconstruction via ``extract_pages`` and ``LTTextContainer``.
   Returns per-page text blocks.

2. **pypdf** (secondary) — simpler heuristic extraction via
   ``page.extract_text()``. Used when pdfminer.six is not installed or when
   it fails to extract any text from a page (e.g. some encrypted PDFs).

Both backends are optional; ``ImportError`` is raised at first
``get_raw_chunks()`` call (not at import time) if neither is available.

Design notes
------------
* One raw chunk per PDF page — each page becomes one (or more) CorpusDocument
  instances depending on the injected chunker.
* ``page_number`` (zero-based) is a promoted first-class field.
* ``source_type`` defaults to ``SourceType.UNKNOWN``; callers may override via
  ``DocumentReader.create(path, source_type=SourceType.RESEARCH)``.
* Encrypted PDFs are opened with ``password`` when supplied.
* Pages that yield no text after stripping are logged and skipped.

Python compatibility
--------------------
Python 3.8-3.15. Zero runtime imports at module level. ``pdfminer.six`` and
``pypdf`` are optional lazy imports.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import (  # noqa: F401
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
)

from .._base import DocumentReader
from .._schema import SectionType
from ._custom import normalize_extractor_output

logger = logging.getLogger(__name__)

__all__ = ["PDFReader"]

# Maximum file size accepted before raising. 2 GB covers most real PDFs; very
# large document sets should be split into individual files first.
_DEFAULT_MAX_FILE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _extract_page_text_pdfminer(pdf_path: Any, page_number: int) -> str | None:
    """
    Extract text from a single PDF page using pdfminer.six.

    Parameters
    ----------
    pdf_path : pathlib.Path
        Path to the PDF file.
    page_number : int
        Zero-based page index to extract.

    Returns
    -------
    str or None
        Extracted text, or ``None`` if pdfminer could not open the file.

    Raises
    ------
    ImportError
        If ``pdfminer`` is not installed.
    """
    try:
        from pdfminer.high_level import extract_pages  # type: ignore[] # noqa: PLC0415
        from pdfminer.layout import (  # type: ignore[] # noqa: PLC0415, F401
            LTAnno,
            LTChar,
            LTTextContainer,
        )
    except ImportError as exc:
        raise ImportError(
            "pdfminer.six is required for PDFReader (primary backend)."
            " Install it with:\n  pip install pdfminer.six"
        ) from exc

    try:
        # extract_pages yields one LTPage per page; page_numbers is 0-based list
        pages = list(extract_pages(str(pdf_path), page_numbers=[page_number]))
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "PDFReader: pdfminer failed on page %d of %s: %s",
            page_number,
            pdf_path.name,
            exc,
        )
        return None

    if not pages:
        return None

    lines: list[str] = []
    for element in pages[0]:
        if isinstance(element, LTTextContainer):
            lines.append(element.get_text())

    return "".join(lines)


def _extract_page_text_pypdf(
    pdf_path: Any, page_number: int, password: str
) -> str | None:
    """
    Extract text from a single PDF page using pypdf.

    Parameters
    ----------
    pdf_path : pathlib.Path
        Path to the PDF file.
    page_number : int
        Zero-based page index.
    password : str
        Decryption password. Empty string for unencrypted PDFs.

    Returns
    -------
    str or None
        Extracted text, or ``None`` on error.

    Raises
    ------
    ImportError
        If ``pypdf`` is not installed.
    """
    try:
        from pypdf import PdfReader as _PdfReader  # type: ignore[] # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pypdf is required for PDFReader (fallback backend)."
            " Install it with:\n  pip install pypdf"
        ) from exc

    try:
        reader = _PdfReader(str(pdf_path))
        if reader.is_encrypted:
            result = reader.decrypt(password)
            if result == 0:
                logger.warning(
                    "PDFReader: failed to decrypt %s (wrong password?).",
                    pdf_path.name,
                )
                return None
        if page_number >= len(reader.pages):
            return None
        return reader.pages[page_number].extract_text() or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "PDFReader: pypdf failed on page %d of %s: %s",
            page_number,
            pdf_path.name,
            exc,
        )
        return None


def _count_pdf_pages_pdfminer(pdf_path: Any) -> int | None:
    """Return page count using pdfminer, or ``None`` on failure."""
    try:
        from pdfminer.pdfdocument import PDFDocument  # type: ignore[] # noqa: PLC0415
        from pdfminer.pdfpage import PDFPage  # type: ignore[] # noqa: PLC0415
        from pdfminer.pdfparser import PDFParser  # type: ignore[] # noqa: PLC0415
    except ImportError:
        return None
    try:
        with open(pdf_path, "rb") as fh:  # noqa: PTH123
            parser = PDFParser(fh)
            doc = PDFDocument(parser)
            return sum(1 for _ in PDFPage.create_pages(doc))
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "PDFReader: pdfminer page count failed for %s: %s",
            pdf_path,
            exc,
        )
        return None


def _count_pdf_pages_pypdf(pdf_path: Any, password: str) -> int | None:
    """Return page count using pypdf, or ``None`` on failure."""
    try:
        from pypdf import PdfReader as _PdfReader  # type: ignore[] # noqa: PLC0415
    except ImportError:
        return None
    try:
        reader = _PdfReader(str(pdf_path))
        if reader.is_encrypted:
            reader.decrypt(password)
        return len(reader.pages)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "PDFReader: pypdf page count failed for %s: %s",
            pdf_path,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# PDFReader dataclass
# ---------------------------------------------------------------------------


@dataclass
class PDFReader(DocumentReader):
    """
    PDF document reader with pdfminer.six → pypdf cascade.

    Yields one raw chunk per PDF page. Pages that produce no extractable
    text after stripping are silently skipped (logged at DEBUG level).

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the ``.pdf`` file.
    password : str, optional
        Decryption password for encrypted PDFs. Default: ``""`` (none).
    max_file_bytes : int, optional
        Maximum file size in bytes before raising ``ValueError``. Default:
        2 GB.
    prefer_backend : str or None, optional
        Force a specific extraction backend. One of ``"pdfminer"``,
        ``"pypdf"``, or ``None`` (auto: try pdfminer first). Default:
        ``None``.
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
    file_type : str
        Class variable. Always ``".pdf"``.
    file_types : list of str
        Class variable. Registered extensions:
        ``[".pdf"]``.

    Raises
    ------
    ValueError
        If ``prefer_backend`` is not a recognised value.
    ValueError
        If the file exceeds ``max_file_bytes``.
    ImportError
        If neither pdfminer.six nor pypdf is installed.

    See Also
    --------
    scikitplot.corpus._readers.ALTOReader : ALTO XML in ZIP reader.
    scikitplot.corpus._readers.ImageReader : OCR reader for raster images.

    Notes
    -----
    **Backend selection:**

    * ``pdfminer.six`` preserves reading order better than pypdf for
      multi-column layouts and PDFs with complex glyph mappings.
    * ``pypdf`` is lighter and faster for simple single-column PDFs.
    * When ``prefer_backend=None``, pdfminer is tried first; if it yields
      no text for a page, pypdf is tried as fallback on that page.

    **Promoted fields in each raw chunk dict:**

    * ``"page_number"`` — zero-based page index (int).
    * ``"section_type"`` — always ``SectionType.TEXT``.

    **Encrypted PDFs:** Supply ``password`` when constructing the reader.
    If the password is wrong, affected pages are skipped with a warning.

    Examples
    --------
    Default usage (auto backend):

    >>> from pathlib import Path
    >>> reader = PDFReader(input_file=Path("paper.pdf"))
    >>> docs = list(reader.get_documents())
    >>> print(f"Extracted {len(docs)} chunks from {reader.file_name}")

    Encrypted PDF with forced pypdf backend:

    >>> reader = PDFReader(
    ...     input_file=Path("secure.pdf"),
    ...     password="hunter2",
    ...     prefer_backend="pypdf",
    ... )
    >>> docs = list(reader.get_documents())

    Research PDF with provenance:

    >>> from scikitplot.corpus._base import DocumentReader
    >>> from scikitplot.corpus._schema import SourceType
    >>> reader = DocumentReader.create(
    ...     Path("arxiv_paper.pdf"),
    ...     source_type=SourceType.RESEARCH,
    ...     doi="10.1038/s41586-021-00099-z",
    ... )
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".pdf"
    file_types: ClassVar[list[str] | None] = [".pdf"]

    _VALID_BACKENDS: ClassVar[tuple[str, ...]] = ("pdfminer", "pypdf", "custom")

    password: str = field(default="")
    """Decryption password for encrypted PDFs. Empty string means unencrypted."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Maximum file size in bytes. Default: 2 GB."""

    prefer_backend: str | None = field(default=None)
    """
    Force a specific extraction backend. One of ``"pdfminer"``, ``"pypdf"``,
    ``"custom"``, or ``None`` (auto: try pdfminer first, pypdf fallback).
    Default: ``None``.

    When ``"custom"``, :attr:`custom_extractor` **must** be provided.
    """

    custom_extractor: Callable[..., Any] | None = field(default=None, repr=False)
    """
    User-supplied PDF extraction callable, active only when
    ``prefer_backend="custom"``.

    Signature::

        def extractor(path: pathlib.Path, **kwargs) -> ExtractorOutput

    where ``ExtractorOutput`` is ``str``, ``list[str]``, ``dict``, or
    ``list[dict]``.  Every dict must contain a ``"text"`` key.

    Common use-cases: ``pdfplumber``, ``pymupdf`` (fitz), ``docling``,
    ``surya``, or any proprietary PDF engine.  Ignored when
    ``prefer_backend`` is not ``"custom"``.  Default: ``None``.

    Examples
    --------
    >>> import pdfplumber
    >>> def pdfplumber_extract(path, **kw):
    ...     with pdfplumber.open(path) as pdf:
    ...         return [{"text": p.extract_text() or "", "page_number": i}
    ...                 for i, p in enumerate(pdf.pages)]
    >>> reader = PDFReader(
    ...     input_file=Path("paper.pdf"),
    ...     prefer_backend="custom",
    ...     custom_extractor=pdfplumber_extract,
    ... )
    """

    custom_extractor_kwargs: dict[str, Any] = field(default_factory=dict)
    """
    Extra keyword arguments forwarded to :attr:`custom_extractor` on every
    call.  Only used when ``prefer_backend="custom"``.  Default: ``{}``.
    """

    def __post_init__(self) -> None:
        """Validate PDF reader constructor fields.

        Raises
        ------
        ValueError
            If ``prefer_backend`` is not in
            ``{"pdfminer", "pypdf", "custom", None}``.
        ValueError
            If ``prefer_backend="custom"`` but ``custom_extractor`` is
            ``None``.
        TypeError
            If ``custom_extractor`` is not callable (and not ``None``).
        ValueError
            If ``max_file_bytes <= 0``.
        """
        super().__post_init__()
        if (
            self.prefer_backend is not None
            and self.prefer_backend not in self._VALID_BACKENDS
        ):
            raise ValueError(
                f"PDFReader: prefer_backend must be one of"
                f" {self._VALID_BACKENDS} or None;"
                f" got {self.prefer_backend!r}."
            )
        if self.prefer_backend == "custom" and self.custom_extractor is None:
            raise ValueError(
                "PDFReader: prefer_backend='custom' requires a "
                "'custom_extractor' callable.  Pass one via "
                "custom_extractor=my_fn, or choose a built-in backend."
            )
        if self.custom_extractor is not None and not callable(self.custom_extractor):
            raise TypeError(
                f"PDFReader: custom_extractor must be callable or None; "
                f"got {type(self.custom_extractor).__name__!r}."
            )
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"PDFReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield one raw chunk dict per PDF page.

        When ``prefer_backend="custom"`` and :attr:`custom_extractor` is
        set, delegates entirely to the extractor callable and normalises
        its return value via
        :func:`~scikitplot.corpus._readers._custom.normalize_extractor_output`.
        Otherwise, attempts pdfminer.six first; falls back to pypdf on
        pages where pdfminer returns no text.  Pages with no extractable
        text are skipped.

        Yields
        ------
        dict
            Keys:

            ``"text"``
                Page text with whitespace preserved from the PDF layout.
            ``"section_type"``
                Always :attr:`~scikitplot.corpus._schema.SectionType.TEXT`.
            ``"page_number"``
                Zero-based page index (promoted to first-class field).

        Raises
        ------
        ValueError
            If the file exceeds ``max_file_bytes``.
        ImportError
            If neither pdfminer.six nor pypdf is installed (built-in path).
        RuntimeError
            If ``prefer_backend="custom"`` and the extractor raises.
        """
        # ── Custom extractor path ──────────────────────────────────────
        if self.prefer_backend == "custom":
            # custom_extractor is guaranteed non-None by __post_init__
            assert self.custom_extractor is not None  # noqa: S101
            extractor_name = getattr(
                self.custom_extractor, "__name__", repr(self.custom_extractor)
            )
            logger.info(
                "PDFReader: using custom extractor %r on %s.",
                extractor_name,
                self.file_name,
            )
            try:
                raw = self.custom_extractor(
                    self.input_file, **self.custom_extractor_kwargs
                )
            except Exception as exc:
                raise RuntimeError(
                    f"PDFReader: custom extractor {extractor_name!r} raised "
                    f"an error processing {self.file_name!r}: {exc}"
                ) from exc
            chunks = normalize_extractor_output(
                raw,
                source_type=self.source_provenance.get("source_type", "unknown"),
                section_type=SectionType.TEXT,
            )
            logger.info(
                "PDFReader: custom extractor returned %d chunk(s) from %s.",
                len(chunks),
                self.file_name,
            )
            yield from chunks
            return

        file_size = self.input_file.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"PDFReader: {self.file_name} is {file_size:,} bytes, which"
                f" exceeds max_file_bytes={self.max_file_bytes:,}."
                f" Increase max_file_bytes or split the PDF before ingestion."
            )

        # Determine total page count — needed for the iteration loop.
        n_pages = self._count_pages()
        if n_pages is None or n_pages == 0:
            logger.warning(
                "PDFReader: could not determine page count for %s."
                " Attempting page-by-page extraction until first failure.",
                self.file_name,
            )
            n_pages = 10_000  # large sentinel; loop breaks on None text

        logger.info(
            "PDFReader: opening %s (%d page(s), backend=%s).",
            self.file_name,
            n_pages,
            self.prefer_backend or "auto",
        )

        pages_yielded = 0
        for page_idx in range(n_pages):
            text = self._extract_page(page_idx)
            if text is None:
                # Sentinel exceeded or both backends failed — stop
                break
            stripped = text.strip()
            if not stripped:
                logger.debug(
                    "PDFReader: page %d of %s has no extractable text; skipping.",
                    page_idx,
                    self.file_name,
                )
                continue
            yield {
                "text": stripped,
                "section_type": SectionType.TEXT.value,
                "page_number": page_idx,
            }
            pages_yielded += 1

        logger.info(
            "PDFReader: finished %s — %d page(s) yielded text.",
            self.file_name,
            pages_yielded,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_pages(self) -> int | None:
        """Return total page count using the best available backend."""
        if self.prefer_backend == "pypdf":
            return _count_pdf_pages_pypdf(self.input_file, self.password)
        count = _count_pdf_pages_pdfminer(self.input_file)
        if count is None:
            count = _count_pdf_pages_pypdf(self.input_file, self.password)
        return count

    def _extract_page(self, page_idx: int) -> str | None:
        """
        Extract text from page ``page_idx`` using the configured backend.

        Parameters
        ----------
        page_idx : int
            Zero-based page index.

        Returns
        -------
        str or None
            Extracted text, or ``None`` if both backends failed or the
            page index is out of range.
        """
        if self.prefer_backend == "pypdf":
            return _extract_page_text_pypdf(self.input_file, page_idx, self.password)

        if self.prefer_backend == "pdfminer":
            return _extract_page_text_pdfminer(self.input_file, page_idx)

        # Auto mode: pdfminer primary, pypdf fallback per-page
        try:
            text = _extract_page_text_pdfminer(self.input_file, page_idx)
        except ImportError:
            text = None

        if text and text.strip():
            return text

        # pdfminer returned empty or failed — try pypdf
        try:
            fallback = _extract_page_text_pypdf(
                self.input_file, page_idx, self.password
            )
        except ImportError as e:
            if text is None:
                raise ImportError(
                    "Neither pdfminer.six nor pypdf is installed."
                    " Install at least one:\n"
                    "  pip install pdfminer.six\n"
                    "  pip install pypdf"
                ) from e
            fallback = None

        return fallback if fallback is not None else text
