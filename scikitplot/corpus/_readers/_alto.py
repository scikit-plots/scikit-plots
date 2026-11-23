# scikitplot/corpus/_readers/_alto.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._readers.alto
===============================

ALTO XML reader for the scikitplot corpus pipeline.

Reads ALTO XML files (Analyzed Layout and Text Object) packed inside a
ZIP archive. Each XML file in the archive corresponds to one page of a
scanned document; each ``<TextBlock>`` or ``<TextLine>`` becomes one raw
chunk carrying physical layout metadata.

ALTO standard versions supported:

v2 (``http://www.loc.gov/standards/alto/ns-v2#``)
v3 (``http://www.loc.gov/standards/alto/ns-v3#``)
v4 (``http://www.loc.gov/standards/alto/ns-v4#``)
No namespace (older files)

Promoted first-class fields per chunk:

``page_number``
    Zero-based index of the page within the archive, ordered by the
    natural sort of the XML filenames.
``bbox``
    Physical bounding box as ``(HPOS, VPOS, WIDTH, HEIGHT)`` in the
    measurement unit declared by ``MeasurementUnit`` in the ``Layout``
    header (usually 1/10 mm). Stored as a ``tuple[float, ...]``.
``confidence``
    Mean word confidence computed from the ``WC`` attributes on each
    ``<String>`` element within the block/line. Range ``[0.0, 1.0]``.
    ``None`` when no ``WC`` attributes are present.
``ocr_engine``
    Value of ``<Processing>/<processingStepSettings>`` or the
    ``<Software>`` description from the ALTO header when available.

Security:

ZIP path entries are validated against ZipSlip attacks: any entry whose
resolved path would escape the destination directory is rejected with
``ValueError`` before extraction.

Python compatibility:

Python 3.8-3.15. Uses only stdlib: ``zipfile``, ``xml.etree.ElementTree``
(with optional ``lxml`` for speed). No runtime dependencies at import time.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Dict, Generator, List, Optional, Tuple  # noqa: F401

from .._backends import BackendCandidate, run_backend_chain
from .._base import DocumentReader
from .._schema import SectionType

logger = logging.getLogger(__name__)

__all__ = ["ALTOReader"]

# ---------------------------------------------------------------------------
# ALTO namespace URIs for v2 / v3 / v4
# ---------------------------------------------------------------------------
_ALTO_NAMESPACES: tuple[str, ...] = (
    "http://www.loc.gov/standards/alto/ns-v2#",
    "http://www.loc.gov/standards/alto/ns-v3#",
    "http://www.loc.gov/standards/alto/ns-v4#",
    "",  # no namespace (legacy files)
)

# Default granularity at which chunks are yielded
_GRAN_BLOCK = "block"  # one chunk per <TextBlock>
_GRAN_LINE = "line"  # one chunk per <TextLine>
_GRAN_PAGE = "page"  # one chunk per page (all text joined)
_VALID_GRANULARITIES = (_GRAN_BLOCK, _GRAN_LINE, _GRAN_PAGE)

# Regex for natural sort: split strings into numeric and non-numeric runs
_NAT_SORT_RE = re.compile(r"(\d+)")

# Maximum bytes in a single XML member before we warn
_MEMBER_SIZE_WARN_BYTES = 100 * 1024 * 1024  # 100 MB


# ---------------------------------------------------------------------------
# Natural sort helpers
# ---------------------------------------------------------------------------


def _nat_sort_key(name: str) -> list[int | str]:
    """
    Natural-sort key for filenames.

    Splits the name at digit/non-digit boundaries so that ``page10.xml``
    sorts after ``page9.xml`` rather than before it.

    Parameters
    ----------
    name : str
        Filename (basename only).

    Returns
    -------
    list
        Mixed int/str list suitable for comparison.

    Examples
    --------
    >>> _nat_sort_key("page10.xml")
    ['page', 10, '.xml']
    >>> _nat_sort_key("0010.xml")
    ['', 10, '.xml']
    """
    parts: list[int | str] = [
        int(token) if token.isdigit() else token.lower()
        for token in _NAT_SORT_RE.split(name)
    ]
    return parts


def _sorted_xml_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """
    Return ZipInfo entries for XML files in natural sort order.

    Parameters
    ----------
    zf : zipfile.ZipFile
        Open ZIP archive.

    Returns
    -------
    list of zipfile.ZipInfo
        Entries with ``.xml`` suffix sorted by natural filename order.
    """
    xml_entries = [
        info
        for info in zf.infolist()
        if info.filename.lower().endswith(".xml") and not info.is_dir()
    ]
    xml_entries.sort(key=lambda info: _nat_sort_key(Path(info.filename).name))
    return xml_entries


# ---------------------------------------------------------------------------
# ZipSlip guard
# ---------------------------------------------------------------------------


def _safe_zip_member(info: zipfile.ZipInfo) -> str:
    """
    Validate a ZipInfo filename against ZipSlip path traversal.

    Parameters
    ----------
    info : zipfile.ZipInfo
        Entry from the ZIP archive.

    Returns
    -------
    str
        The safe filename (basename only after posix-normalising).

    Raises
    ------
    ValueError
        If the entry's path contains ``..`` components that could escape
        the extraction directory.
    """
    # Use PurePosixPath to normalise separators and strip leading slashes
    parts = PurePosixPath(info.filename).parts
    for part in parts:
        if part == "..":
            raise ValueError(
                f"ALTOReader: ZIP entry {info.filename!r} contains '..' "
                "which could escape the target directory (ZipSlip attack). "
                "Aborting extraction."
            )
    return info.filename


# ---------------------------------------------------------------------------
# ALTO XML parsing helpers
# ---------------------------------------------------------------------------


def _detect_alto_namespace(root: Any) -> str:
    """
    Extract the ALTO namespace URI from the root element tag.

    Parameters
    ----------
    root : Element
        XML root element (lxml or stdlib ET).

    Returns
    -------
    str
        ALTO namespace URI, or ``""`` for namespace-less files.

    Examples
    --------
    >>> # Tag like "{http://www.loc.gov/standards/alto/ns-v3#}alto"
    >>> _detect_alto_namespace(root)
    'http://www.loc.gov/standards/alto/ns-v3#'
    """
    tag = getattr(root, "tag", "") or ""
    if "}" in tag:
        return tag.split("}")[0].lstrip("{")
    return ""


def _ns(local: str, ns_uri: str) -> str:
    """Build a namespaced tag string ``"{uri}local"`` or plain ``"local"``."""
    return f"{{{ns_uri}}}{local}" if ns_uri else local


def _unparsable_page(exc: BaseException) -> ValueError:
    """
    Build the one error raised for an ALTO page that cannot be parsed.

    Parameters
    ----------
    exc : BaseException
        The error of the parser that ran.

    Returns
    -------
    ValueError
        ``"ALTOReader: could not parse XML: <parser message>"``. The caller
        raises it ``from exc``.

    Notes
    -----
    **Developer notes:** the lxml and the stdlib paths both build their error
    here, so the page loop of :meth:`ALTOReader.get_raw_chunks`, which skips
    a page on ``ValueError``, treats a malformed page the same whichever
    parser ran.
    """
    return ValueError(f"ALTOReader: could not parse XML: {exc}")


def _parse_xml_bytes(content: bytes) -> Any:
    """
    Parse ALTO XML bytes with lxml first and a secure stdlib fallback.

    Only the absence of lxml triggers the fallback; a parse error from an
    installed lxml is final.

    Parameters
    ----------
    content : bytes
        Raw XML of one page.

    Returns
    -------
    xml.etree.ElementTree.Element or lxml.etree._Element
        Root element of the page.

    Raises
    ------
    ValueError
        If the page is not valid XML (empty, truncated, malformed) or is
        refused by the XML safety checks, whichever parser ran. The parser's
        own error is the ``__cause__``.

    Notes
    -----
    **User notes:** a malformed page is skipped with a warning and the rest
    of the archive is read, with and without lxml.

    **Developer notes:** ``lxml.etree.XMLSyntaxError`` is a ``SyntaxError``,
    not a ``ValueError``; it is converted here with :func:`_unparsable_page`,
    as the stdlib path's errors are (CX-42).
    """

    def _lxml() -> Any:
        from lxml import etree  # type: ignore[] # noqa: PLC0415

        from ._xml_safety import hardened_lxml_parser  # noqa: PLC0415

        try:
            return etree.fromstring(content, parser=hardened_lxml_parser())
        except etree.XMLSyntaxError as exc:
            raise _unparsable_page(exc) from exc

    def _stdlib() -> Any:
        from ._xml_safety import parse_stdlib_secure  # noqa: PLC0415

        try:
            return parse_stdlib_secure(content)
        except Exception as exc:  # noqa: BLE001
            raise _unparsable_page(exc) from exc

    outcome = run_backend_chain(
        (
            BackendCandidate(
                "lxml",
                _lxml,
                fallback_exceptions=(ImportError,),
                failure_level=logging.DEBUG,
                degrades_on_failure=False,
                capability="xml:lxml",
            ),
            BackendCandidate(
                "stdlib-xml",
                _stdlib,
                fallback_exceptions=(),
                degrades_on_failure=False,
            ),
        ),
        default=None,
        logger=logger,
        component="ALTOReader",
        operation="XML parsing",
        error_code="ALTO_XML_BACKEND_UNAVAILABLE",
        stage="parse",
    )
    return outcome.value


def _attr_float(element: Any, attr: str) -> float | None:
    """
    Read a numeric XML attribute, returning ``None`` on missing or invalid value.

    Parameters
    ----------
    element : Element
        Source element.
    attr : str
        Attribute name (no namespace prefix).

    Returns
    -------
    float or None
    """
    val = element.get(attr)
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _read_ocr_engine(root: Any, ns_uri: str) -> str | None:
    """
    Extract OCR engine description from the ALTO header.

    Checks ``<Description><Processing><processingStepSettings>`` and
    ``<Description><MeasurementUnit>`` / ``<softwareName>`` paths.

    Parameters
    ----------
    root : Element
        ALTO root element.
    ns_uri : str
        Detected ALTO namespace.

    Returns
    -------
    str or None
        OCR engine description string, or ``None`` if not found.
    """
    # Try <Description><Processing><processingStepSettings>
    for path in [
        f".//{_ns('processingStepSettings', ns_uri)}",
        f".//{_ns('softwareName', ns_uri)}",
        f".//{_ns('softwareCreator', ns_uri)}",
    ]:
        el = root.find(path)
        if el is not None:
            text = (el.text or "").strip()
            if text:
                return text
    return None


#: Characters that end a line in a word broken across lines: hyphen-minus,
#: the double oblique hyphen of Fraktur printing, and the not sign some OCR
#: engines emit for it.
_LINE_END_HYPHENS: frozenset[str] = frozenset({"-", "⸗", "¬"})
#: Soft hyphen: invisible unless the word breaks there; always removed.
_SOFT_HYPHEN = "­"


def _join_lines(lines: list[str], dehyphenate: bool) -> str:
    """
    Join the text of consecutive lines of one block.

    Parameters
    ----------
    lines : list of str
        Text of each line, in reading order.
    dehyphenate : bool
        Rejoin a word broken across two lines.

    Returns
    -------
    str
        The lines joined with single spaces, or, with ``dehyphenate``, a
        line that ends in a hyphen joined to the next without a space:

        - a soft hyphen (U+00AD) is removed;
        - ``-``, ``⸗`` or ``¬`` after a letter, followed by a lower-case
          letter, is removed (``Geschichts-`` + ``auffassung`` →
          ``Geschichtsauffassung``);
        - followed by anything else, it is kept as ``-`` (``Marx-`` +
          ``Engels`` → ``Marx-Engels``).

    Notes
    -----
    **User notes:** a suspended compound broken right after its hyphen
    (``Haupt-`` + ``und Nebensatz``) is joined as ``Hauptund``; text alone
    cannot tell it from a broken word. Pass ``dehyphenate=False`` to keep
    every line break as a space.

    **Developer notes:** ALTO's own hyphenation markup (``SUBS_CONTENT``)
    is applied before this, in :func:`_line_text`; this rule covers files
    that carry the hyphen as plain text, as line-level OCR exports do.
    """
    text = ""
    for line in lines:
        if not line:
            continue
        if not text:
            text = line
            continue
        last, first = text[-1], line[0]
        if dehyphenate and last == _SOFT_HYPHEN:
            text = text[:-1] + line
        elif (
            dehyphenate
            and last in _LINE_END_HYPHENS
            and len(text) > 1
            and text[-2].isalpha()
            and first.isalpha()
        ):
            text = text[:-1] + (line if first.islower() else "-" + line)
        else:
            text = f"{text} {line}"
    return text


def _line_text(
    line: Any, ns_uri: str, dehyphenate: bool, state: dict[str, bool]
) -> tuple[str, str, list[float]]:
    """
    Return the text, the verbatim text and the word confidences of one line.

    Parameters
    ----------
    line : Element
        ``<TextLine>`` element.
    ns_uri : str
        ALTO namespace.
    dehyphenate : bool
        Apply ALTO hyphenation markup: the first part of a broken word
        (``SUBS_TYPE="HypPart1"``) is replaced by the whole word
        (``SUBS_CONTENT``), and the second part is left out. A ``<HYP>``
        without ``SUBS_CONTENT`` adds its hyphen to the line's end, for
        :func:`_join_lines`.
    state : dict
        Carries ``"pending"`` (a whole word was written for a first part)
        from one line to the next within a block.

    Returns
    -------
    text, raw : str
        The line's words; ``raw`` is the verbatim ``CONTENT`` join.
    confidences : list of float
        ``WC`` of each word.
    """
    words: list[str] = []
    raw: list[str] = []
    confs: list[float] = []
    string_tag, hyp_tag = _ns("String", ns_uri), _ns("HYP", ns_uri)
    for child in line:
        if child.tag == hyp_tag:
            if dehyphenate and words and not state.get("subs"):
                words[-1] += child.get("CONTENT") or "-"
            continue
        if child.tag != string_tag:
            continue
        content = child.get("CONTENT", "")
        if content:
            raw.append(content)
        wc = _attr_float(child, "WC")
        if wc is not None:
            confs.append(wc)
        kind, whole = child.get("SUBS_TYPE"), child.get("SUBS_CONTENT")
        state["subs"] = False
        if dehyphenate and whole and kind == "HypPart1":
            words.append(whole)
            state["pending"] = state["subs"] = True
        elif dehyphenate and whole and kind == "HypPart2" and state.get("pending"):
            state["pending"] = False
        elif content:
            words.append(content)
    return " ".join(words).strip(), " ".join(raw).strip(), confs


def _mean_confidence(confs: list[float]) -> float | None:
    """Mean ``WC`` in [0, 1]; values on a 0-100 scale are divided by 100."""
    if not confs:
        return None
    mean = sum(confs) / len(confs)
    return mean / 100.0 if mean > 1.0 else mean


def _bbox(element: Any, fallback: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(
        _attr_float(element, key) or default
        for key, default in zip(("HPOS", "VPOS", "WIDTH", "HEIGHT"), fallback)
    )


def _chunk(
    text: str,
    raw: str,
    *,
    page_number: int | None,
    bbox: tuple[float, ...] | None,
    confs: list[float],
    ocr_engine: str | None,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "text": text,
        # raw_text: the verbatim ALTO CONTENT tokens joined with spaces,
        # before any dehyphenation.
        "raw_text": raw,
        "section_type": SectionType.TEXT.value,
        "bbox": bbox,
        "confidence": _mean_confidence(confs),
        "ocr_engine": ocr_engine,
    }
    if page_number is not None:
        chunk["page_number"] = page_number
    return chunk


def _extract_block_chunks(
    block: Any,
    ns_uri: str,
    page_number: int | None,
    ocr_engine: str | None,
    granularity: str,
    *,
    dehyphenate: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract raw chunk dicts from a single ALTO ``<TextBlock>`` element.

    Parameters
    ----------
    block : Element
        ``<TextBlock>`` element.
    ns_uri : str
        Detected ALTO namespace.
    page_number : int or None
        Zero-based page index; ``None`` leaves ``page_number`` out.
    ocr_engine : str or None
        OCR engine string from the ALTO header (may be ``None``).
    granularity : str
        ``"block"`` yields one chunk per TextBlock; ``"line"`` one chunk
        per TextLine (no joining across lines).
    dehyphenate : bool, default True
        Rejoin words broken across lines (block granularity only); see
        :func:`_join_lines` and :func:`_line_text`.

    Returns
    -------
    list of dict
        Raw chunk dicts with keys ``"text"``, ``"raw_text"``,
        ``"section_type"``, ``"page_number"`` (when known), ``"bbox"``,
        ``"confidence"``, ``"ocr_engine"``.
    """
    block_bbox = _bbox(block, (0.0, 0.0, 0.0, 0.0))
    lines = block.findall(_ns("TextLine", ns_uri))
    if granularity == _GRAN_LINE:
        chunks: list[dict[str, Any]] = []
        for line in lines:
            text, raw, confs = _line_text(line, ns_uri, False, {})
            if text:
                chunks.append(
                    _chunk(
                        text,
                        raw,
                        page_number=page_number,
                        bbox=_bbox(line, block_bbox),
                        confs=confs,
                        ocr_engine=ocr_engine,
                    )
                )
        return chunks
    texts, raws, confs_block = _block_lines(lines, ns_uri, dehyphenate)
    text = _join_lines(texts, dehyphenate)
    if not text:
        return []
    return [
        _chunk(
            text,
            " ".join(r for r in raws if r),
            page_number=page_number,
            bbox=block_bbox,
            confs=confs_block,
            ocr_engine=ocr_engine,
        )
    ]


def _block_lines(
    lines: list[Any], ns_uri: str, dehyphenate: bool
) -> tuple[list[str], list[str], list[float]]:
    """Text, verbatim text and confidences of each line of one block."""
    state: dict[str, bool] = {}
    texts: list[str] = []
    raws: list[str] = []
    confs: list[float] = []
    for line in lines:
        text, raw, line_confs = _line_text(line, ns_uri, dehyphenate, state)
        texts.append(text)
        raws.append(raw)
        confs.extend(line_confs)
    return texts, raws, confs


def _extract_page_chunks(
    root: Any,
    ns_uri: str,
    page_number: int | None,
    granularity: str,
    ocr_engine: str | None,
    *,
    dehyphenate: bool = True,
) -> list[dict[str, Any]]:
    """
    Extract all raw chunks from one ALTO XML page document.

    Parameters
    ----------
    root : Element
        Root ``<alto>`` element.
    ns_uri : str
        Detected ALTO namespace URI.
    page_number : int or None
        Zero-based page index; ``None`` leaves ``page_number`` out.
    granularity : str
        Chunking granularity: ``"block"``, ``"line"``, or ``"page"``.
    ocr_engine : str or None
        OCR engine name from the ALTO header.
    dehyphenate : bool, default True
        Rejoin words broken across lines within a block.

    Returns
    -------
    list of dict
        Raw chunk dicts for this page.
    """
    all_blocks = root.findall(f".//{_ns('TextBlock', ns_uri)}")
    if not all_blocks:
        logger.debug(
            "ALTOReader: page %s — no <TextBlock> elements found (ns=%r).",
            page_number,
            ns_uri,
        )
        return []
    if granularity == _GRAN_PAGE:
        texts: list[str] = []
        raws: list[str] = []
        confs: list[float] = []
        line_tag = _ns("TextLine", ns_uri)
        for block in all_blocks:
            lines, block_raws, block_confs = _block_lines(
                block.findall(line_tag), ns_uri, dehyphenate
            )
            texts.append(_join_lines(lines, dehyphenate))
            raws.extend(r for r in block_raws if r)
            confs.extend(block_confs)
        text = " ".join(t for t in texts if t)
        if not text:
            return []
        # A page-level chunk has no single bbox.
        return [
            _chunk(
                text,
                " ".join(raws),
                page_number=page_number,
                bbox=None,
                confs=confs,
                ocr_engine=ocr_engine,
            )
        ]
    chunks: list[dict[str, Any]] = []
    for block in all_blocks:
        chunks.extend(
            _extract_block_chunks(
                block,
                ns_uri,
                page_number,
                ocr_engine,
                granularity,
                dehyphenate=dehyphenate,
            )
        )
    return chunks


def is_alto_root(root: Any) -> bool:
    """
    Return whether a parsed XML root element is an ALTO document.

    Parameters
    ----------
    root : Element
        Root element (lxml or stdlib).

    Returns
    -------
    bool
        True when the local name is ``alto`` (any case) and the namespace is
        an ALTO namespace of the Library of Congress, or absent (legacy
        exports).
    """
    tag = getattr(root, "tag", "")
    if not isinstance(tag, str):
        return False
    ns_uri = _detect_alto_namespace(root)
    local = tag.rsplit("}", 1)[-1]
    return local.lower() == "alto" and (
        not ns_uri or ns_uri.startswith("http://www.loc.gov/standards/alto/")
    )


def alto_page_chunks(
    root: Any, *, granularity: str = _GRAN_BLOCK, dehyphenate: bool = True
) -> list[dict[str, Any]]:
    """
    Return the raw chunks of one parsed ALTO page.

    Parameters
    ----------
    root : Element
        Root ``<alto>`` element (see :func:`is_alto_root`).
    granularity : {"block", "line", "page"}, default "block"
        Chunking level.
    dehyphenate : bool, default True
        Rejoin words broken across lines.

    Returns
    -------
    list of dict
        As :meth:`ALTOReader.get_raw_chunks`. ``page_number`` is the page's
        ``PHYSICAL_IMG_NR`` minus one when that is a positive integer, and
        left out otherwise: a single page file has no archive position.

    Raises
    ------
    ValueError
        For an unknown granularity.
    """
    if granularity not in _VALID_GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {_VALID_GRANULARITIES}; got {granularity!r}."
        )
    ns_uri = _detect_alto_namespace(root)
    page = root.find(f".//{_ns('Page', ns_uri)}")
    number = (page.get("PHYSICAL_IMG_NR") or "") if page is not None else ""
    page_number = int(number) - 1 if number.isdigit() and int(number) > 0 else None
    return _extract_page_chunks(
        root,
        ns_uri,
        page_number,
        granularity,
        _read_ocr_engine(root, ns_uri),
        dehyphenate=dehyphenate,
    )


# ---------------------------------------------------------------------------
# ALTOReader dataclass
# ---------------------------------------------------------------------------


@dataclass
class ALTOReader(DocumentReader):
    """
    ALTO XML reader for scanned document archives.

    Reads a ZIP archive containing one ALTO XML file per page, extracts
    text and physical layout metadata from each page, and yields raw chunk
    dicts with first-class fields ``page_number``, ``bbox``,
    ``confidence``, and ``ocr_engine``.

    ALTO namespaces v2, v3, and v4 are auto-detected from the root element.
    Pages are processed in natural-sort order of the XML filenames within
    the archive.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.zip`` archive containing ALTO XML files.
    granularity : str, optional
        Chunking level within each ALTO page. One of:

        ``"block"`` (default)
            One chunk per ``<TextBlock>``. Suitable for paragraph-level
            corpus construction.
        ``"line"``
            One chunk per ``<TextLine>``. Suitable for line-level OCR
            error analysis or fine-grained alignment.
        ``"page"``
            One chunk per page (all text joined). Suitable for
            document-level retrieval.
    max_file_bytes : int, optional
        Maximum ZIP file size in bytes before raising ``ValueError``.
        Default: 5 GB.
    xml_encoding : str or None, optional
        Force XML member encoding. ``None`` uses the encoding declared
        in each XML header (or UTF-8). Default: ``None``.
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
        Class variable. Always ``".zip"``.
    file_types : list of str
        Class variable. Registered extensions:
        ``[".zip"]``.

    Raises
    ------
    ValueError
        If ``granularity`` is not one of the valid values.
    ValueError
        If the file exceeds ``max_file_bytes``.
    ValueError
        If a ZIP entry contains a ZipSlip ``..`` path traversal sequence.

    See Also
    --------
    scikitplot.corpus._readers.PDFReader : PDF reader.
    scikitplot.corpus._readers.XMLReader : Generic XML reader.
    scikitplot.corpus._readers.ImageReader : OCR reader for raster images.

    Notes
    -----
    **ALTO namespace detection:** The reader inspects the root element tag
    of each XML file. Documents with no namespace (older ABBYY exports)
    are handled via the empty-string namespace path.

    **Word confidence (WC):** ALTO ``WC`` attributes represent per-word
    OCR confidence. Some tools emit values in ``[0, 1]``; others use
    ``[0, 100]``. ``ALTOReader`` auto-normalises values above 1.0 by
    dividing by 100, so ``confidence`` is always in ``[0.0, 1.0]`` when
    present.

    **Bounding boxes:** The ``bbox`` field stores
    ``(HPOS, VPOS, WIDTH, HEIGHT)`` in the measurement unit declared by
    ``MeasurementUnit`` in the ALTO header (commonly 1/10 mm for 300 DPI
    scans). Conversion to pixel coordinates requires the DPI value which
    is not stored in the chunk; retrieve it from ``doc.metadata`` if
    needed (add a ``"dpi"`` key in a custom reader subclass).

    **Security:** ZipSlip validation rejects any ZIP entry whose path
    contains ``..`` components before any file is read.

    Examples
    --------
    Default block-level chunking:

    >>> from pathlib import Path
    >>> reader = ALTOReader(input_path=Path("newspaper_scan.zip"))
    >>> docs = list(reader.get_documents())
    >>> print(f"Blocks extracted: {len(docs)}")
    >>> print(f"Page 0 confidence: {docs[0].confidence:.3f}")

    Line-level granularity:

    >>> reader = ALTOReader(
    ...     input_path=Path("book_scan.zip"),
    ...     granularity="line",
    ... )

    With source provenance:

    >>> from scikitplot.corpus._base import DocumentReader
    >>> from scikitplot.corpus._schema import SourceType
    >>> reader = DocumentReader.create(
    ...     Path("periodical_1920.zip"),
    ...     source_type=SourceType.ARTICLE,
    ...     source_title="The Daily Gazette",
    ...     source_date="1920-03-15",
    ... )
    """

    # CRITICAL-05 / SEV-1.4 FIX:
    # ALTOReader is intentionally NOT registered for ".zip" in the auto-registry.
    # Registration is managed by ZipReader, which:
    #   1. Owns the ".zip" extension exclusively.
    #   2. Detects ALTO XML archives automatically via _detect_alto_archive().
    #   3. Delegates the entire archive to ALTOReader when ALTO is detected.
    #   4. Falls back to per-member dispatch for mixed-content ZIPs.
    #
    # To use ALTOReader on a known ALTO archive without auto-detection:
    #   reader = ALTOReader(input_path=Path("newspaper_scans.zip"))
    file_types: ClassVar[list[str] | None] = None

    granularity: str = field(default=_GRAN_BLOCK)
    """
    Chunking granularity within each ALTO page. One of ``"block"``,
    ``"line"``, or ``"page"``. Default: ``"block"``.
    """

    max_file_bytes: int = field(default=5 * 1024 * 1024 * 1024)  # 5 GB
    """Maximum ZIP file size in bytes. Default: 5 GB."""

    xml_encoding: str | None = field(default=None)
    dehyphenate: bool = field(default=True)
    """Rejoin words broken across lines (ALTO ``SUBS_CONTENT`` or a line-end
    hyphen before a lower-case letter); ``False`` keeps every line break as a
    space. Default: ``True``."""
    """
    Force a specific encoding for XML member decoding. ``None`` uses the
    XML declaration or UTF-8. Default: ``None``.
    """

    def __post_init__(self) -> None:  # noqa: D105
        super().__post_init__()
        if self.granularity not in _VALID_GRANULARITIES:
            raise ValueError(
                f"ALTOReader: granularity must be one of"
                f" {_VALID_GRANULARITIES}; got {self.granularity!r}."
            )
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"ALTOReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Iterate over ALTO XML pages in the ZIP and yield text chunks.

        Each XML file in the archive is treated as one page, processed in
        natural-sort order. Pages with no ``<TextBlock>`` elements or no
        extractable text are skipped.

        Yields
        ------
        dict
            Keys:

            ``"text"``
                Extracted OCR text.
            ``"section_type"``
                Always :attr:`~scikitplot.corpus._schema.SectionType.TEXT`.
            ``"page_number"``
                Zero-based page index within the archive (promoted field).
            ``"bbox"``
                Bounding box ``(HPOS, VPOS, WIDTH, HEIGHT)`` as a
                ``tuple[float, ...]``, or ``None`` for page-level chunks.
            ``"confidence"``
                Mean word confidence in ``[0.0, 1.0]``, or ``None``
                when ``WC`` attributes are absent.
            ``"ocr_engine"``
                OCR engine name from the ALTO header, or ``None``.

        Raises
        ------
        ValueError
            If the file exceeds ``max_file_bytes``.
        ValueError
            If a ZIP entry triggers the ZipSlip guard.
        zipfile.BadZipFile
            If the file is not a valid ZIP archive.
        """
        file_size = self.input_path.stat().st_size
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"ALTOReader: {self.file_name} is {file_size:,} bytes, which"
                f" exceeds max_file_bytes={self.max_file_bytes:,}."
                f" Increase max_file_bytes or split the archive."
            )

        if not zipfile.is_zipfile(self.input_path):
            raise zipfile.BadZipFile(
                f"ALTOReader: {self.file_name} is not a valid ZIP archive."
            )

        with zipfile.ZipFile(self.input_path, "r") as zf:
            xml_entries = _sorted_xml_members(zf)
            if not xml_entries:
                logger.warning(
                    "ALTOReader: %s contains no .xml members.", self.file_name
                )
                return

            logger.info(
                "ALTOReader: %s — %d XML page(s), granularity=%r.",
                self.file_name,
                len(xml_entries),
                self.granularity,
            )

            total_chunks = 0
            for page_idx, info in enumerate(xml_entries):
                # ZipSlip guard — raises ValueError if unsafe
                _safe_zip_member(info)

                if info.file_size > _MEMBER_SIZE_WARN_BYTES:
                    logger.warning(
                        "ALTOReader: member %r is %d MB — this may be slow.",
                        info.filename,
                        info.file_size // (1024 * 1024),
                    )

                try:
                    raw_bytes = zf.read(info.filename)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "ALTOReader: could not read %r from %s: %s",
                        info.filename,
                        self.file_name,
                        exc,
                    )
                    continue

                if self.xml_encoding:
                    raw_bytes = raw_bytes.decode(self.xml_encoding).encode("utf-8")

                try:
                    root = _parse_xml_bytes(raw_bytes)
                except ValueError as exc:
                    logger.warning(
                        "ALTOReader: XML parse error in %r (page %d): %s",
                        info.filename,
                        page_idx,
                        exc,
                    )
                    continue

                ns_uri = _detect_alto_namespace(root)
                ocr_engine = _read_ocr_engine(root, ns_uri)

                chunks = _extract_page_chunks(
                    root,
                    ns_uri,
                    page_idx,
                    self.granularity,
                    ocr_engine,
                    dehyphenate=self.dehyphenate,
                )

                if not chunks:
                    logger.debug(
                        "ALTOReader: page %d (%s) — no text found; skipping.",
                        page_idx,
                        info.filename,
                    )
                    continue

                logger.debug(
                    "ALTOReader: page %d (%s) — %d chunk(s) extracted.",
                    page_idx,
                    info.filename,
                    len(chunks),
                )
                total_chunks += len(chunks)
                yield from chunks

        logger.info(
            "ALTOReader: finished %s — %d total chunk(s) across %d page(s).",
            self.file_name,
            total_chunks,
            len(xml_entries),
        )
