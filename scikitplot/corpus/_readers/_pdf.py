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

Design notes:

* One raw chunk per PDF page — each page becomes one (or more) CorpusDocument
  instances depending on the injected chunker.
* ``page_number`` (zero-based) is a promoted first-class field.
* ``source_type`` defaults to ``SourceType.UNKNOWN``; callers may override via
  ``DocumentReader.create(path, source_type=SourceType.RESEARCH)``.
* Encrypted PDFs are opened with ``password`` when supplied.
* Pages that yield no text after stripping are logged and skipped.

Python compatibility:

Python 3.8-3.15. Zero runtime imports at module level. ``pdfminer.six``,
``pypdf`` and ``pypdfium2`` (OCR only) are optional lazy imports.

Caution for maintainers:

The distribution to require is ``pdfminer.six``. Do not add ``pdfminer``, even
though that is the import name: it is a different, older distribution that
installs the same package, and the two overwrite each other (see
``_pdfminer_import_error``). ``tools/maint_tools/check_refused_requirements.py``
enforces this in CI.
"""  # noqa: D205, D400

from __future__ import annotations

import itertools
import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import (  # noqa: F401
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
)

from .._base import DocumentReader
from .._schema import SectionType

logger = logging.getLogger(__name__)

__all__ = ["PDFReader"]

# Maximum file size accepted before raising. 2 GB covers most real PDFs; very
# large document sets should be split into individual files first.
# BUG-05 fix: renamed from _PDF_DEFAULT_MAX_FILE_BYTES to clarify this is
# PDF-specific. _text.py has 500 MB, _xml.py has 200 MB (ClassVar) —
# identical name across files implied a shared constant when each is independent.
_PDF_DEFAULT_MAX_FILE_BYTES: int = 2 * 1024 * 1024 * 1024  # 2 GB


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


class _IncompletePdfminerError(ImportError):
    """The package ``pdfminer`` is installed but cannot be imported whole."""


def _pdfminer_import_error(exc: ImportError) -> ImportError:
    """
    Return the error to raise when pdfminer.six cannot be imported.

    Parameters
    ----------
    exc : ImportError
        What importing ``pdfminer.high_level`` or ``pdfminer.layout`` raised.

    Returns
    -------
    ImportError
        An install hint when the package ``pdfminer`` is absent; an
        ``_IncompletePdfminerError`` when it is present but a module or a
        name in it is missing.

    Notes
    -----
    **Developer note:** Two distributions install the import package
    ``pdfminer``: ``pdfminer.six``, which this reader needs, and
    ``pdfminer``, an older project without ``pdfminer.high_level``.
    Installed together they overwrite each other's files, and uninstalling
    one removes files of the other. The usual symptoms are ``cannot import
    name 'HOCRConverter' from 'pdfminer.converter'`` and ``No module named
    'pdfminer.converter'``. Never add ``pdfminer`` to a requirement list:
    ``tools/maint_tools/check_refused_requirements.py`` fails if it is.
    """
    if isinstance(exc, ModuleNotFoundError) and exc.name == "pdfminer":
        return ImportError(
            "pdfminer.six is required for PDFReader (primary backend)."
            " Install it with:\n  pip install pdfminer.six"
        )
    return _IncompletePdfminerError(
        f"PDFReader: the installed package 'pdfminer' is incomplete ({exc})."
        " The distributions 'pdfminer' and 'pdfminer.six' both install it and"
        " must not be installed together. Reinstall pdfminer.six alone:\n"
        "  pip uninstall pdfminer pdfminer.six\n"
        "  pip install pdfminer.six"
    )


def _reason(exc: BaseException) -> str:
    """Return an exception as ``Type: message``, or ``Type`` without one."""
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class _PdfminerPages:
    """
    The text of a PDF's pages from one pass of pdfminer.six over the file.

    Parameters
    ----------
    pdf_path : pathlib.Path
        Path to the PDF file.
    password : str, optional
        Decryption password. Empty string for unencrypted PDFs.
    kwargs : Mapping or None, optional
        Keyword arguments for ``pdfminer.high_level.extract_pages``. A
        mapping under ``"laparams"`` is turned into a
        ``pdfminer.layout.LAParams``.

    Notes
    -----
    **Developer note:** ``pdfminer.high_level.extract_pages`` opens the file
    and walks the page tree from the first page on every call, so one call
    per page costs pages times pages. One iterator is kept open instead and
    advanced a page at a time; only the page being read is held. Asking for
    any page but the next one, or for a page after one that failed, opens a
    new iterator at that page.
    """

    def __init__(
        self,
        pdf_path: Any,
        password: str = "",
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._path = pdf_path
        self._password = password
        self._kwargs = kwargs
        self._pages: Any = None
        #: Why the first page that failed could not be read, if any did.
        self.error: str | None = None
        # Index of the page the open iterator yields next.
        self._at = 0

    def text(self, page_number: int) -> str | None:
        """
        Return the text of one page.

        Parameters
        ----------
        page_number : int
            Zero-based page index.

        Returns
        -------
        str or None
            Extracted text, or ``None`` if pdfminer could not read the page
            or the file has no such page.

        Raises
        ------
        ImportError
            If ``pdfminer`` is not installed.
        ValueError
            If pdfminer does not accept the keyword arguments.
        """
        try:
            from pdfminer.high_level import extract_pages  # type: ignore[] # noqa: PLC0415
            from pdfminer.layout import (  # type: ignore[] # noqa: PLC0415
                LAParams,
                LTTextContainer,
            )
        except ImportError as exc:
            raise _pdfminer_import_error(exc) from exc

        if self._pages is None or page_number != self._at:
            self.close()
            options = dict(self._kwargs or {})
            if isinstance(options.get("laparams"), Mapping):
                try:
                    options["laparams"] = LAParams(**options["laparams"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "PDFReader: pdfminer_kwargs['laparams'] is not accepted"
                        f" by pdfminer.layout.LAParams: {exc}"
                    ) from exc
            try:
                # page_numbers is a container of 0-based indices; None is all.
                self._pages = extract_pages(
                    str(self._path),
                    password=self._password,
                    page_numbers=(
                        range(page_number, sys.maxsize) if page_number else None
                    ),
                    **options,
                )
            except TypeError as exc:
                # A generator function binds its arguments when it is called.
                raise ValueError(
                    "PDFReader: pdfminer_kwargs is not accepted by"
                    f" pdfminer.high_level.extract_pages: {exc}"
                ) from exc
            self._at = page_number
        try:
            page = next(self._pages, None)
        except Exception as exc:  # noqa: BLE001
            # Broad catch: pdfminer can raise PDFSyntaxError, PDFEncryptionError,
            # PDFPasswordIncorrect, UnicodeDecodeError, OSError, and more
            # depending on the PDF's internal structure and pdfminer version.
            # All are equally unrecoverable for this page -- log and return
            # None so the caller falls through to the pypdf backup path. The
            # iterator is finished after raising; the next call opens one.
            logger.debug(
                "PDFReader: pdfminer failed on page %d of %s: %s",
                page_number,
                self._path.name,
                exc,
            )
            if self.error is None:
                self.error = _reason(exc)
            self.close()
            return None
        self._at = page_number + 1
        if page is None:
            return None
        return "".join(
            element.get_text()
            for element in page
            if isinstance(element, LTTextContainer)
        )

    def close(self) -> None:
        """Close the open iterator, and with it the file."""
        if self._pages is not None:
            self._pages.close()
            self._pages = None


class _PypdfPages:
    """
    The text of a PDF's pages from one pypdf reader.

    Parameters
    ----------
    pdf_path : pathlib.Path
        Path to the PDF file.
    password : str, optional
        Decryption password. Empty string for unencrypted PDFs.

    Notes
    -----
    **Developer note:** The file is opened, and decrypted, on the first page
    asked for and once only; a file that cannot be opened or decrypted is
    not tried again.
    """

    def __init__(self, pdf_path: Any, password: str = "") -> None:
        self._path = pdf_path
        self._password = password
        self._reader: Any = None
        self._n_pages = 0
        self._unreadable = False
        #: Why the first page that failed could not be read, if any did.
        self.error: str | None = None

    def text(self, page_number: int) -> str | None:
        """
        Return the text of one page.

        Parameters
        ----------
        page_number : int
            Zero-based page index.

        Returns
        -------
        str or None
            Extracted text, or ``None`` on error or if the file has no such
            page.

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

        if self._unreadable:
            return None
        try:
            if self._reader is None:
                reader = _PdfReader(str(self._path))
                if reader.is_encrypted and reader.decrypt(self._password) == 0:
                    logger.warning(
                        "PDFReader: failed to decrypt %s (wrong password?).",
                        self._path.name,
                    )
                    self._unreadable = True
                    self.error = "the password does not decrypt the file"
                    return None
                self._n_pages = len(reader.pages)
                self._reader = reader
            if page_number >= self._n_pages:
                return None
            return self._reader.pages[page_number].extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            # Broad catch: pypdf raises PdfReadError, PdfStreamError, AssertionError,
            # DependencyError, and others across versions for malformed or encrypted pages.
            # Return None so the caller falls through to the next extraction strategy.
            logger.debug(
                "PDFReader: pypdf failed on page %d of %s: %s",
                page_number,
                self._path.name,
                exc,
            )
            if self.error is None:
                self.error = _reason(exc)
            if self._reader is None:
                self._unreadable = True
            return None

    def close(self) -> None:
        """Drop the reader, and with it the file's bytes."""
        self._reader = None


class _PageTexts:
    """
    The text of a PDF's pages from the backend a reader is configured for.

    Parameters
    ----------
    pdf_path : pathlib.Path
        Path to the PDF file.
    password : str
        Decryption password. Empty string for unencrypted PDFs.
    prefer_backend : str or None
        ``"pdfminer"``, ``"pypdf"``, or ``None`` for pdfminer with a
        per-page fallback to pypdf.
    pdfminer_kwargs : Mapping or None, optional
        Keyword arguments for ``pdfminer.high_level.extract_pages``.
    """

    def __init__(
        self,
        pdf_path: Any,
        password: str,
        prefer_backend: str | None,
        pdfminer_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._prefer_backend = prefer_backend
        self._pdfminer = _PdfminerPages(pdf_path, password, pdfminer_kwargs)
        self._warned = False
        #: Number of pages a backend returned text (possibly empty) for.
        self.n_read = 0
        self._pypdf = _PypdfPages(pdf_path, password)

    def text(self, page_idx: int) -> str | None:
        """
        Return the text of page ``page_idx``.

        Parameters
        ----------
        page_idx : int
            Zero-based page index.

        Returns
        -------
        str or None
            Extracted text, or ``None`` if both backends failed or the
            page index is out of range.

        Raises
        ------
        ImportError
            If the configured backend, or in auto mode every backend, is not
            installed.
        """
        text = self._backend_text(page_idx)
        if text is not None:
            self.n_read += 1
        return text

    def errors(self) -> list[str]:
        """Return the first failure of each backend, as ``backend: reason``."""
        found = (("pdfminer", self._pdfminer.error), ("pypdf", self._pypdf.error))
        return [f"{name}: {reason}" for name, reason in found if reason]

    def _backend_text(self, page_idx: int) -> str | None:
        """Return the text of page ``page_idx`` from the configured backend."""
        if self._prefer_backend == "pypdf":
            return self._pypdf.text(page_idx)

        if self._prefer_backend == "pdfminer":
            return self._pdfminer.text(page_idx)

        # Auto mode: pdfminer primary, pypdf fallback per-page
        try:
            text = self._pdfminer.text(page_idx)
        except ImportError as exc:
            # Absent pdfminer.six is an ordinary reason to use pypdf; a
            # damaged install is not, so it is said once per read.
            if isinstance(exc, _IncompletePdfminerError) and not self._warned:
                self._warned = True
                logger.warning(
                    "PDFReader: reading with pypdf, because pdfminer could not"
                    " be imported. %s",
                    exc,
                )
            text = None

        if text and text.strip():
            return text

        # pdfminer returned empty or failed -- try pypdf
        try:
            fallback = self._pypdf.text(page_idx)
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

    def close(self) -> None:
        """Close both backends."""
        self._pdfminer.close()
        self._pypdf.close()


def _count_pdf_pages_pdfminer(pdf_path: Any, password: str = "") -> int | None:
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
            doc = PDFDocument(parser, password=password)
            return sum(1 for _ in PDFPage.create_pages(doc))
    except Exception as exc:  # noqa: BLE001
        # Broad catch: pdfminer page-count raises same variety of errors as
        # per-page extraction (see _extract_page_text_pdfminer).  Page count
        # is best-effort; return None so the caller uses pypdf as fallback.
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
        # Broad catch: same reasoning as pypdf per-page extraction above.
        logger.debug(
            "PDFReader: pypdf page count failed for %s: %s",
            pdf_path,
            exc,
        )
        return None


# ---------------------------------------------------------------------------
# OCR of rendered pages
# ---------------------------------------------------------------------------

#: Values of ``PDFReader.ocr`` other than a callable.
_OCR_POLICIES: tuple[str, ...] = ("off", "auto", "always")

#: Values of ``PDFReader.ocr_engine`` other than a callable.
_OCR_ENGINES: tuple[str, ...] = ("tesseract", "easyocr")

#: Highest ``ocr_dpi``. An A4 page at 1200 dpi is about 140 million pixels,
#: some 400 MiB as RGB; above that one page can exhaust a worker's memory.
_OCR_MAX_DPI: int = 1200

#: pypdfium2 ``PdfPage.render`` arguments the reader sets itself.
_PYPDFIUM2_RESERVED: frozenset[str] = frozenset({"scale"})

#: PDFium is not thread-safe: two threads inside it at once can crash the
#: process or corrupt each other's pages (reproduced with eight readers in a
#: ``ThreadPoolExecutor``, as ``CorpusBuilder(max_workers=8)`` runs them).
#: Every call into pypdfium2 holds this lock; the OCR engine does not.
_PDFIUM_LOCK = threading.Lock()


def _page_ranges(pages: Sequence[int]) -> str:
    """
    Return ascending page numbers as ranges, such as ``"1, 3-5, 9"``.

    Parameters
    ----------
    pages : sequence of int
        Ascending, distinct page numbers.

    Returns
    -------
    str
    """
    parts: list[str] = []
    start = previous = None
    for page in [*pages, None]:
        if previous is not None and page == previous + 1:
            previous = page
            continue
        if start is not None:
            parts.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    return ", ".join(parts)


def _is_positive_int(value: Any) -> bool:
    """Return whether *value* is an ``int`` (not ``bool``) of at least 1."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


class _PageOcr:
    """
    OCR of the pages of one PDF, rendered with pypdfium2.

    Parameters
    ----------
    pdf_path : pathlib.Path
        Path to the PDF file.
    password : str
        Decryption password. Empty string for unencrypted PDFs.
    policy : str or callable
        ``"auto"``, ``"always"``, or ``policy(page_number, text) -> bool``.
    min_chars : int
        Under ``"auto"``, a page whose text layer has fewer characters than
        this is read with OCR.
    max_pages : int or None
        Most pages read with OCR; ``None`` for no limit.
    engine : str or callable
        ``"tesseract"``, ``"easyocr"``, or ``engine(image)`` returning the
        text or ``(text, confidence)``.
    lang : str or None
        Language hint for a built-in engine.
    dpi : int
        Resolution pages are rendered at.
    pypdfium2_kwargs : Mapping or None
        Keyword arguments for pypdfium2's ``PdfPage.render``.

    Notes
    -----
    **Developer note:** The document is opened on the first page that needs
    OCR and closed by :meth:`close`. Every call into pypdfium2 holds
    ``_PDFIUM_LOCK``, because PDFium is not thread-safe; the engine runs
    outside it. Rendering failures are per page: the
    page keeps its text layer and is named in :attr:`failed`. A file that
    pypdfium2 cannot open fails every page the same way. Errors from the
    engine are not caught: a missing Tesseract binary or a bug in a custom
    engine is not a property of one page.
    """

    def __init__(  # noqa: PLR0913
        self,
        pdf_path: Any,
        password: str,
        *,
        policy: str | Callable[[int, str], bool],
        min_chars: int,
        max_pages: int | None,
        engine: str | Callable[[Any], Any],
        lang: str | None,
        dpi: int,
        pypdfium2_kwargs: Mapping[str, Any] | None,
    ) -> None:
        self._path = pdf_path
        self._password = password
        self._policy = policy
        self._min_chars = min_chars
        self._max_pages = max_pages
        self._engine = engine
        self._lang = lang
        self._dpi = dpi
        self._render_kwargs = dict(pypdfium2_kwargs or {})
        self._doc: Any = None
        self._open_error: str | None = None
        self._easyocr_reader: Any = None
        #: The value recorded as each OCR chunk's ``ocr_engine``.
        self.engine_name = engine if isinstance(engine, str) else "custom"
        #: Number of pages read with OCR.
        self.n_read = 0
        #: Pages that needed OCR after ``max_pages`` was reached.
        self.over_limit: list[int] = []
        #: ``(page, reason)`` for each page that could not be rendered.
        self.failed: list[tuple[int, str]] = []

    def wants(self, page_number: int, text: str) -> bool:
        """
        Return whether a page is read with OCR.

        Parameters
        ----------
        page_number : int
            Zero-based page index.
        text : str
            The page's text layer, stripped.

        Returns
        -------
        bool

        Raises
        ------
        TypeError
            If a policy callable returns anything but a ``bool``.
        """
        if self._policy == "always":
            return True
        if self._policy == "auto":
            return len(text) < self._min_chars
        decision = self._policy(page_number, text)  # type: ignore[operator]
        if not isinstance(decision, bool):
            raise TypeError(
                "PDFReader: an ocr policy must return a bool; got"
                f" {type(decision).__name__!r} for page {page_number}."
            )
        return decision

    def read(self, page_number: int) -> tuple[str, float | None] | None:
        """
        Return the OCR text of one page and its confidence.

        Parameters
        ----------
        page_number : int
            Zero-based page index.

        Returns
        -------
        tuple of (str, float or None), or None
            The text and the engine's confidence in ``[0.0, 1.0]`` (``None``
            when the engine gives none). ``None`` when ``max_pages`` is
            reached or the page cannot be rendered.

        Raises
        ------
        ImportError
            If pypdfium2 or the engine's library is not installed.
        ValueError
            If pypdfium2 does not accept ``pypdfium2_kwargs``.
        TypeError
            If a custom engine returns something other than a text or a
            ``(text, confidence)`` pair.
        """
        if self._max_pages is not None and self.n_read >= self._max_pages:
            self.over_limit.append(page_number)
            return None
        image = self._render(page_number)
        if image is None:
            return None
        self.n_read += 1
        # Outside _PDFIUM_LOCK: OCR is the slow part and runs in parallel.
        return self._recognise(image)

    def _render(self, page_number: int) -> Any:
        """Return one page as a PIL image, or ``None`` if it cannot be rendered."""
        if self._open_error is not None:
            self.failed.append((page_number, self._open_error))
            return None
        try:
            import pypdfium2  # type: ignore[] # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pypdfium2 is required for PDFReader with ocr other than 'off'"
                " (it renders the pages). Install it with:\n  pip install pypdfium2"
            ) from exc
        with _PDFIUM_LOCK:
            if self._doc is None:
                try:
                    self._doc = pypdfium2.PdfDocument(
                        str(self._path), password=self._password or None
                    )
                except Exception as exc:  # noqa: BLE001
                    # Broad catch: pypdfium2 raises PdfiumError, OSError and
                    # others for a file it cannot open; each page then keeps
                    # its text layer.
                    self._open_error = _reason(exc)
                    self.failed.append((page_number, self._open_error))
                    return None
            try:
                return self._render_locked(page_number)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                # Broad catch: a damaged page raises PdfiumError, IndexError or
                # MemoryError depending on the file; the page keeps its text
                # layer.
                self.failed.append((page_number, _reason(exc)))
                return None

    def _render_locked(self, page_number: int) -> Any:
        """
        Render one page; the caller holds ``_PDFIUM_LOCK``.

        The image returned owns its pixels. ``PdfBitmap.to_pil`` shares the
        bitmap's buffer for ``L``, ``RGBA`` and ``RGBX`` bitmaps (a
        ``grayscale`` render, for one), and closing the bitmap frees that
        buffer, so the image is copied, or converted, before it is closed.
        """
        page = self._doc[page_number]
        try:
            try:
                bitmap = page.render(scale=self._dpi / 72, **self._render_kwargs)
            except TypeError as exc:
                if not self._render_kwargs:
                    raise
                raise ValueError(
                    "PDFReader: pypdfium2_kwargs is not accepted by"
                    f" pypdfium2's PdfPage.render: {exc}"
                ) from exc
            try:
                image = bitmap.to_pil()
                if image.mode in ("RGB", "L"):
                    return image.copy()
                return image.convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()

    def _recognise(self, image: Any) -> tuple[str, float | None]:
        """Run the engine on one image."""
        if self._engine == "tesseract":
            from ._image import _ocr_tesseract  # noqa: PLC0415

            return _ocr_tesseract(image, self._lang)
        if self._engine == "easyocr":
            from ._image import _ocr_easyocr  # noqa: PLC0415

            text, confidence, self._easyocr_reader = _ocr_easyocr(
                image, self._lang, self._easyocr_reader
            )
            return text, confidence
        result = self._engine(image)  # type: ignore[operator]
        if isinstance(result, str):
            return result, None
        if (
            isinstance(result, tuple)
            and len(result) == 2  # noqa: PLR2004
            and isinstance(result[0], str)
            and isinstance(result[1], (int, float))
            and not isinstance(result[1], bool)
            and 0.0 <= result[1] <= 1.0
        ):
            return result[0], float(result[1])
        raise TypeError(
            "PDFReader: an ocr_engine callable must return a str or a"
            f" (str, confidence in [0, 1]) pair; got {result!r:.80}."
        )

    def close(self) -> None:
        """Close the document."""
        if self._doc is not None:
            with _PDFIUM_LOCK:
                self._doc.close()
            self._doc = None


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
    input_path : pathlib.Path
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
    pdfminer_kwargs : Mapping or None, optional
        Keyword arguments for ``pdfminer.high_level.extract_pages``, used
        when pdfminer.six reads a page; pypdf ignores them. A mapping under
        ``"laparams"`` is turned into a ``pdfminer.layout.LAParams``, so
        the layout analysis can be set without importing pdfminer.
        ``password``, ``page_numbers`` and ``maxpages`` are set by the
        reader and are not accepted. ``None`` uses pdfminer's defaults. See
        the Notes for a reproducible read. Default: ``None``.
    ocr : {"off", "auto", "always"} or callable, optional
        Which pages are rendered and read with OCR. ``"off"`` reads the text
        layer only. ``"auto"`` reads a page with OCR when its text layer has
        fewer than ``ocr_min_chars`` characters: scanned pages, slides
        exported as pictures. ``"always"`` reads every page with OCR. A
        callable ``policy(page_number, text) -> bool`` decides per page from
        the zero-based page number and the stripped text layer. The OCR text
        replaces the text layer when it is not empty. Default: ``"off"``.
    ocr_min_chars : int, optional
        Under ``ocr="auto"``, a text layer shorter than this is not trusted.
        Raise it for scans that carry only a page label as text. Default: 1
        (only pages with no text).
    ocr_max_pages : int or None, optional
        Most pages read with OCR per file. Pages past the limit keep their
        text layer and are named in one warning. ``None`` for no limit.
        Default: 64.
    ocr_engine : {"tesseract", "easyocr"} or callable, optional
        OCR engine, as in :class:`ImageReader`, or a callable taking a PIL
        image and returning its text or ``(text, confidence)`` with the
        confidence in ``[0, 1]``. Recorded in each OCR chunk's
        ``ocr_engine`` (``"custom"`` for a callable). Default:
        ``"tesseract"``.
    ocr_lang : str or None, optional
        Language hint for a built-in engine (``"eng"``, ``"eng+deu"`` for
        Tesseract; ``"en"`` for easyocr). Default: ``None``.
    ocr_dpi : int, optional
        Resolution pages are rendered at, from 1 to 1200. Default: 300.
    pypdfium2_kwargs : Mapping or None, optional
        Keyword arguments for pypdfium2's ``PdfPage.render``, which renders
        pages for OCR (for example ``{"grayscale": True}``). ``scale`` is set
        from ``ocr_dpi`` and is not accepted. Default: ``None``.
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

    **Scanned pages (OCR):** A PDF of scanned pages has no text layer, and
    one exported from slides may carry only page labels. With ``ocr="auto"``
    such pages are rendered with pypdfium2 (``pip install pypdfium2``) and
    read by the OCR engine. A chunk read with OCR carries ``ocr_engine`` and,
    from the built-in engines, ``confidence``. OCR takes about one to two
    seconds per page at 300 dpi with Tesseract, hence ``ocr_max_pages``. With
    ``ocr="off"`` a file whose pages hold no text gives one warning that
    names the option. The images a page lists as resources are not used: a
    page can list images it does not draw. PDFium is not thread-safe, so
    readers running in parallel threads (``CorpusBuilder(max_workers=...)``)
    render one page at a time; the OCR engine runs in parallel.

    **Reproducible text:** pdfminer.six orders text boxes that are equally
    far apart by their address in memory, so the text of a page with many
    such boxes (a figure with repeated labels, a grid) can come out in a
    different order on another read of the same file.
    ``pdfminer_kwargs={"laparams": {"boxes_flow": None}}`` orders the boxes
    by position instead: the same text on every read, without pdfminer's
    grouping of boxes into columns.

    **Encrypted PDFs:** Supply ``password`` when constructing the reader.
    If the password is wrong, affected pages are skipped with a warning.

    Examples
    --------
    Default usage (auto backend):

    >>> from pathlib import Path
    >>> reader = PDFReader(input_path=Path("paper.pdf"))
    >>> docs = list(reader.get_documents())
    >>> print(f"Extracted {len(docs)} chunks from {reader.file_name}")

    Encrypted PDF with forced pypdf backend:

    >>> reader = PDFReader(
    ...     input_path=Path("secure.pdf"),
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

    # BUG-03 fix: file_type removed — file_types is the canonical
    # registration mechanism. Defining both caused __init_subclass__
    # to silently ignore file_type (file_types wins).
    file_types: ClassVar[list[str] | None] = [".pdf"]

    _VALID_BACKENDS: ClassVar[tuple[str, ...]] = ("pdfminer", "pypdf", "custom")

    # extract_pages arguments the reader passes itself. maxpages is among
    # them because the reader reads every page and may open the file again.
    _PDFMINER_RESERVED: ClassVar[frozenset[str]] = frozenset(
        {"password", "page_numbers", "maxpages"}
    )

    password: str = field(default="")
    """Decryption password for encrypted PDFs. Empty string means unencrypted."""

    max_file_bytes: int = field(default=_PDF_DEFAULT_MAX_FILE_BYTES)
    """Maximum file size in bytes. Default: 2 GB."""

    prefer_backend: str | None = field(default=None)
    """
    Force a specific extraction backend. One of ``"pdfminer"``, ``"pypdf"``,
    ``"custom"``, or ``None`` (auto: try pdfminer first, pypdf fallback).
    Default: ``None``.

    When ``"custom"``, :attr:`custom_extractor` **must** be provided.
    """

    pdfminer_kwargs: Mapping[str, Any] | None = field(default=None)
    """
    Keyword arguments for ``pdfminer.high_level.extract_pages``; ``None``
    uses pdfminer's defaults. Default: ``None``.
    """

    ocr: str | Callable[[int, str], bool] = field(default="off")
    """
    Which pages are read with OCR: ``"off"``, ``"auto"`` (pages whose text
    layer has fewer than :attr:`ocr_min_chars` characters), ``"always"``, or
    ``policy(page_number, text) -> bool``. Default: ``"off"``.
    """

    ocr_min_chars: int = field(default=1)
    """Under ``ocr="auto"``, the text layer length below which OCR runs. Default: 1."""

    ocr_max_pages: int | None = field(default=64)
    """Most pages read with OCR per file; ``None`` for no limit. Default: 64."""

    ocr_engine: str | Callable[[Any], Any] = field(default="tesseract")
    """``"tesseract"``, ``"easyocr"``, or ``engine(image)``. Default: ``"tesseract"``."""

    ocr_lang: str | None = field(default=None)
    """Language hint for a built-in OCR engine. Default: ``None``."""

    ocr_dpi: int = field(default=300)
    """Resolution pages are rendered at for OCR, 1 to 1200. Default: 300."""

    pypdfium2_kwargs: Mapping[str, Any] | None = field(default=None)
    """
    Keyword arguments for pypdfium2's ``PdfPage.render``, used for OCR;
    ``scale`` is set from :attr:`ocr_dpi`. Default: ``None``.
    """

    # BUG-08/09 fix: custom_extractor and custom_extractor_kwargs are
    # inherited from DocumentReader. Redeclaring them here changed the
    # dataclass __init__ field order (subclass fields come after base fields
    # in MRO but re-declaration moved them to the subclass position). Removed.

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
        TypeError
            If ``pdfminer_kwargs`` is not a mapping with ``str`` keys (and
            not ``None``).
        ValueError
            If an ``ocr`` field is outside its documented values.
        TypeError
            If ``ocr_lang`` is not a ``str`` or ``None``, or
            ``pypdfium2_kwargs`` is not a mapping with ``str`` keys.
        ValueError
            If ``pypdfium2_kwargs`` holds ``scale``.
        ValueError
            If ``pdfminer_kwargs`` holds an argument the reader sets itself.
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
        # The base-class DocumentReader._iter_raw_chunks treats any non-None
        # custom_extractor as Strategy 0 (highest priority) and dispatches to
        # it before get_raw_chunks is called.  Clear the field for every
        # backend other than 'custom' so the base class never sees it.
        if self.prefer_backend != "custom" and self.custom_extractor is not None:
            self.custom_extractor = None
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"PDFReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )
        if self.pdfminer_kwargs is not None:
            if not (
                isinstance(self.pdfminer_kwargs, Mapping)
                and all(isinstance(key, str) for key in self.pdfminer_kwargs)
            ):
                raise TypeError(
                    "PDFReader: pdfminer_kwargs must be a mapping of"
                    " pdfminer.high_level.extract_pages argument names to"
                    f" values, or None; got {self.pdfminer_kwargs!r}."
                )
            reserved = sorted(
                self._PDFMINER_RESERVED.intersection(self.pdfminer_kwargs)
            )
            if reserved:
                raise ValueError(
                    f"PDFReader: pdfminer_kwargs must not hold {reserved}; the"
                    " reader sets these arguments itself."
                )

        self._validate_ocr()

    def _validate_ocr(self) -> None:
        """Validate the OCR fields; see :meth:`__post_init__`."""
        if not (callable(self.ocr) or self.ocr in _OCR_POLICIES):
            raise ValueError(
                f"PDFReader: ocr must be one of {_OCR_POLICIES} or a callable"
                f" (page_number, text) -> bool; got {self.ocr!r}."
            )
        if not (callable(self.ocr_engine) or self.ocr_engine in _OCR_ENGINES):
            raise ValueError(
                f"PDFReader: ocr_engine must be one of {_OCR_ENGINES} or a"
                f" callable taking an image; got {self.ocr_engine!r}."
            )
        if not _is_positive_int(self.ocr_min_chars):
            raise ValueError(
                "PDFReader: ocr_min_chars must be a positive int;"
                f" got {self.ocr_min_chars!r}."
            )
        if self.ocr_max_pages is not None and not _is_positive_int(self.ocr_max_pages):
            raise ValueError(
                "PDFReader: ocr_max_pages must be a positive int or None;"
                f" got {self.ocr_max_pages!r}."
            )
        if not (_is_positive_int(self.ocr_dpi) and self.ocr_dpi <= _OCR_MAX_DPI):
            raise ValueError(
                f"PDFReader: ocr_dpi must be an int from 1 to {_OCR_MAX_DPI};"
                f" got {self.ocr_dpi!r}."
            )
        if self.ocr_lang is not None and not isinstance(self.ocr_lang, str):
            raise TypeError(
                f"PDFReader: ocr_lang must be a str or None; got {self.ocr_lang!r}."
            )
        if self.pypdfium2_kwargs is not None:
            if not (
                isinstance(self.pypdfium2_kwargs, Mapping)
                and all(isinstance(key, str) for key in self.pypdfium2_kwargs)
            ):
                raise TypeError(
                    "PDFReader: pypdfium2_kwargs must be a mapping of pypdfium2"
                    " PdfPage.render argument names to values, or None;"
                    f" got {self.pypdfium2_kwargs!r}."
                )
            reserved = sorted(_PYPDFIUM2_RESERVED.intersection(self.pypdfium2_kwargs))
            if reserved:
                raise ValueError(
                    f"PDFReader: pypdfium2_kwargs must not hold {reserved}; the"
                    " reader sets the scale from ocr_dpi."
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

        A page no backend can read is skipped, and the skipped pages are
        named in one warning after the last page. A file whose pages are
        read but hold no text (scanned images) gives no chunk and one
        warning.

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
        ValueError
            If no backend can read any page: the file is not a complete
            PDF, or it is encrypted and ``password`` is wrong.
        ValueError
            If pdfminer.six does not accept ``pdfminer_kwargs``.
        ImportError
            If neither pdfminer.six nor pypdf is installed (built-in path).
        ImportError
            If ``ocr`` is not ``"off"``, a page needs OCR, and pypdfium2 or
            the engine's library is not installed.
        RuntimeError
            If ``prefer_backend="custom"`` and the extractor raises.
        """
        # ── Custom extractor path ──────────────────────────────────────
        if self.prefer_backend == "custom":
            yield from self._iter_custom_extractor_chunks()
            return

        file_size = self.input_path.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"PDFReader: {self.file_name} is {file_size:,} bytes, which"
                f" exceeds max_file_bytes={self.max_file_bytes:,}."
                f" Increase max_file_bytes or split the PDF before ingestion."
            )

        # The page count tells a page that failed from the end of the file.
        n_pages = self._count_pages() or None
        if n_pages is None:
            logger.warning(
                "PDFReader: could not determine page count for %s."
                " Reading page by page until a page cannot be read.",
                self.file_name,
            )

        logger.info(
            "PDFReader: opening %s (%s page(s), backend=%s, ocr=%s).",
            self.file_name,
            "unknown" if n_pages is None else n_pages,
            self.prefer_backend or "auto",
            self.ocr if isinstance(self.ocr, str) else "custom",
        )

        pages_yielded = 0
        unreadable: list[int] = []
        pages = _PageTexts(
            self.input_path,
            self.password,
            self.prefer_backend,
            self.pdfminer_kwargs,
        )
        ocr = self._page_ocr()
        try:
            for page_idx in itertools.count() if n_pages is None else range(n_pages):
                text = pages.text(page_idx)
                if text is None:
                    if n_pages is None:
                        # Without a count, the first such page is the end.
                        break
                    unreadable.append(page_idx)
                    continue
                chunk = self._page_chunk(page_idx, text, ocr)
                if chunk is None:
                    logger.debug(
                        "PDFReader: page %d of %s has no extractable text; skipping.",
                        page_idx,
                        self.file_name,
                    )
                    continue
                yield chunk
                pages_yielded += 1
        finally:
            # Also reached when the consumer stops early.
            pages.close()
            if ocr is not None:
                ocr.close()

        self._report(pages, ocr, unreadable, n_pages, pages_yielded)

        logger.info(
            "PDFReader: finished %s -- %d of %d page(s) read yielded text.",
            self.file_name,
            pages_yielded,
            pages.n_read,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _report(
        self,
        pages: _PageTexts,
        ocr: _PageOcr | None,
        unreadable: list[int],
        n_pages: int | None,
        pages_yielded: int,
    ) -> None:
        """
        Raise for a file no page was read from; warn about what was skipped.

        Raises
        ------
        ValueError
            If no backend read any page and a backend reported why.
        """

        errors = "; ".join(pages.errors())
        if pages.n_read == 0 and errors:
            raise ValueError(
                f"PDFReader: cannot read {self.file_name}: no backend could read"
                f" a page ({errors}). Check that the file is a complete PDF and,"
                " if it is encrypted, that 'password' is right."
            )
        if unreadable:
            logger.warning(
                "PDFReader: %d of %d page(s) of %s could not be read and are"
                " skipped (pages %s; %s).",
                len(unreadable),
                n_pages,
                self.file_name,
                _page_ranges(unreadable),
                errors or "no backend reported a reason",
            )
        if ocr is not None:
            self._warn_ocr(ocr)
        if pages.n_read and pages_yielded == 0:
            logger.warning(
                "PDFReader: none of the %d page(s) read from %s has extractable"
                " text%s",
                pages.n_read,
                self.file_name,
                (
                    ". The file probably holds scanned images; pass ocr='auto'"
                    " to read them with OCR."
                    if ocr is None
                    else ", with OCR as configured."
                ),
            )

    def _page_ocr(self) -> _PageOcr | None:
        """Return the OCR of this read, or ``None`` when ``ocr="off"``."""
        if self.ocr == "off":
            return None
        return _PageOcr(
            self.input_path,
            self.password,
            policy=self.ocr,
            min_chars=self.ocr_min_chars,
            max_pages=self.ocr_max_pages,
            engine=self.ocr_engine,
            lang=self.ocr_lang,
            dpi=self.ocr_dpi,
            pypdfium2_kwargs=self.pypdfium2_kwargs,
        )

    def _page_chunk(
        self, page_idx: int, text: str, ocr: _PageOcr | None
    ) -> dict[str, Any] | None:
        """
        Return the raw chunk of one page, or ``None`` if it has no text.

        The OCR text replaces the text layer when the policy asks for OCR
        and OCR finds text; otherwise the text layer is kept.
        """
        stripped = text.strip()
        chunk: dict[str, Any] = {
            "text": stripped,
            # raw_text: backend extraction result before leading/trailing
            # whitespace removal -- preserves original page boundary spacing
            # for before/after comparison against the normalised form.
            "raw_text": text,
            "section_type": SectionType.TEXT.value,
            "page_number": page_idx,
        }
        if ocr is not None and ocr.wants(page_idx, stripped):
            result = ocr.read(page_idx)
            if result is not None and result[0].strip():
                ocr_text, confidence = result
                chunk["text"] = ocr_text.strip()
                chunk["raw_text"] = ocr_text
                chunk["ocr_engine"] = ocr.engine_name
                if confidence is not None:
                    chunk["confidence"] = round(confidence, 4)
        return chunk if chunk["text"] else None

    def _warn_ocr(self, ocr: _PageOcr) -> None:
        """Warn about pages OCR was asked for and did not read."""
        if ocr.over_limit:
            logger.warning(
                "PDFReader: %d more page(s) of %s needed OCR after"
                " ocr_max_pages=%d was reached (pages %s); their text layer is"
                " used. Pass ocr_max_pages=None to read them all.",
                len(ocr.over_limit),
                self.file_name,
                self.ocr_max_pages,
                _page_ranges(ocr.over_limit),
            )
        if ocr.failed:
            logger.warning(
                "PDFReader: %d page(s) of %s could not be rendered for OCR"
                " (pages %s; %s); their text layer is used.",
                len(ocr.failed),
                self.file_name,
                _page_ranges([index for index, _ in ocr.failed]),
                ocr.failed[0][1],
            )

    def _count_pages(self) -> int | None:
        """Return total page count using the best available backend."""
        if self.prefer_backend == "pypdf":
            return _count_pdf_pages_pypdf(self.input_path, self.password)
        count = _count_pdf_pages_pdfminer(self.input_path, self.password)
        if count is None:
            count = _count_pdf_pages_pypdf(self.input_path, self.password)
        return count
