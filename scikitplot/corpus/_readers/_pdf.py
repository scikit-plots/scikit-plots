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

Python 3.8-3.15. Zero runtime imports at module level. ``pdfminer.six`` and
``pypdf`` are optional lazy imports.

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
            "PDFReader: opening %s (%s page(s), backend=%s).",
            self.file_name,
            "unknown" if n_pages is None else n_pages,
            self.prefer_backend or "auto",
        )

        pages_yielded = 0
        unreadable: list[int] = []
        pages = _PageTexts(
            self.input_path,
            self.password,
            self.prefer_backend,
            self.pdfminer_kwargs,
        )
        try:
            for page_idx in itertools.count() if n_pages is None else range(n_pages):
                text = pages.text(page_idx)
                if text is None:
                    if n_pages is None:
                        # Without a count, the first such page is the end.
                        break
                    unreadable.append(page_idx)
                    continue
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
                    # raw_text: backend extraction result before leading/trailing
                    # whitespace removal -- preserves original page boundary spacing
                    # for before/after comparison against the normalised form.
                    "raw_text": text,
                    "section_type": SectionType.TEXT.value,
                    "page_number": page_idx,
                }
                pages_yielded += 1
        finally:
            # Also reached when the consumer stops early.
            pages.close()

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
                ", ".join(str(index) for index in unreadable),
                errors or "no backend reported a reason",
            )
        if pages.n_read and pages_yielded == 0:
            logger.warning(
                "PDFReader: none of the %d page(s) read from %s has extractable"
                " text. The file probably holds scanned images; PDFReader does"
                " not run OCR.",
                pages.n_read,
                self.file_name,
            )

        logger.info(
            "PDFReader: finished %s -- %d of %d page(s) read yielded text.",
            self.file_name,
            pages_yielded,
            pages.n_read,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_pages(self) -> int | None:
        """Return total page count using the best available backend."""
        if self.prefer_backend == "pypdf":
            return _count_pdf_pages_pypdf(self.input_path, self.password)
        count = _count_pdf_pages_pdfminer(self.input_path, self.password)
        if count is None:
            count = _count_pdf_pages_pypdf(self.input_path, self.password)
        return count
