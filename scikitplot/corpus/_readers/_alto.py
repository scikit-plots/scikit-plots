"""
scikitplot.corpus._readers.alto
================================
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
    parts: list[int | str] = []
    for token in _NAT_SORT_RE.split(name):
        parts.append(int(token) if token.isdigit() else token.lower())
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


def _parse_xml_bytes(content: bytes) -> Any:
    """
    Parse XML bytes: lxml primary, stdlib fallback.

    Parameters
    ----------
    content : bytes
        Raw XML bytes.

    Returns
    -------
    Element
        Root element (lxml or stdlib).

    Raises
    ------
    ValueError
        If neither parser can parse the bytes.
    """
    try:
        from lxml import etree  # type: ignore[] # noqa: PLC0415

        return etree.fromstring(content)
    except ImportError:
        pass
    import xml.etree.ElementTree as ET  # noqa: N814, PLC0415

    try:
        return ET.fromstring(content)  # noqa: S314
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"ALTOReader: could not parse XML: {exc}") from exc


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


def _extract_block_chunks(  # noqa: PLR0912
    block: Any,
    ns_uri: str,
    page_number: int,
    ocr_engine: str | None,
    granularity: str,
) -> list[dict[str, Any]]:
    """
    Extract raw chunk dicts from a single ALTO ``<TextBlock>`` element.

    Parameters
    ----------
    block : Element
        ``<TextBlock>`` element.
    ns_uri : str
        Detected ALTO namespace.
    page_number : int
        Zero-based page index within the archive.
    ocr_engine : str or None
        OCR engine string from the ALTO header (may be ``None``).
    granularity : str
        One of ``"block"``, ``"line"``, ``"page"`` — controls how text
        is aggregated. ``"block"`` yields one chunk per TextBlock;
        ``"line"`` yields one chunk per TextLine.

    Returns
    -------
    list of dict
        Raw chunk dicts with keys ``"text"``, ``"section_type"``,
        ``"page_number"``, ``"bbox"``, ``"confidence"``, ``"ocr_engine"``.
    """
    # Block-level bounding box
    hpos = _attr_float(block, "HPOS") or 0.0
    vpos = _attr_float(block, "VPOS") or 0.0
    width = _attr_float(block, "WIDTH") or 0.0
    height = _attr_float(block, "HEIGHT") or 0.0
    block_bbox: tuple[float, ...] = (hpos, vpos, width, height)

    line_tag = _ns("TextLine", ns_uri)
    string_tag = _ns("String", ns_uri)

    if granularity == _GRAN_LINE:
        chunks: list[dict[str, Any]] = []
        for line in block.findall(line_tag):
            line_hpos = _attr_float(line, "HPOS") or hpos
            line_vpos = _attr_float(line, "VPOS") or vpos
            line_width = _attr_float(line, "WIDTH") or width
            line_height = _attr_float(line, "HEIGHT") or height
            line_bbox: tuple[float, ...] = (
                line_hpos,
                line_vpos,
                line_width,
                line_height,
            )

            words: list[str] = []
            confs: list[float] = []
            for string_el in line.findall(string_tag):
                content = string_el.get("CONTENT", "")
                if content:
                    words.append(content)
                wc = _attr_float(string_el, "WC")
                if wc is not None:
                    confs.append(wc)

            text = " ".join(words).strip()
            if not text:
                continue

            mean_conf: float | None = sum(confs) / len(confs) if confs else None
            # Clamp to [0,1] — some ALTO files use 0-100 scale for WC
            if mean_conf is not None and mean_conf > 1.0:
                mean_conf = mean_conf / 100.0

            chunks.append(
                {
                    "text": text,
                    "section_type": SectionType.TEXT.value,
                    "page_number": page_number,
                    "bbox": line_bbox,
                    "confidence": mean_conf,
                    "ocr_engine": ocr_engine,
                }
            )
        return chunks

    # Block granularity (default): one chunk per TextBlock
    words_block: list[str] = []
    confs_block: list[float] = []
    for line in block.findall(line_tag):
        for string_el in line.findall(string_tag):
            content = string_el.get("CONTENT", "")
            if content:
                words_block.append(content)
            wc = _attr_float(string_el, "WC")
            if wc is not None:
                confs_block.append(wc)

    text_block = " ".join(words_block).strip()
    if not text_block:
        return []

    mean_conf_block: float | None = (
        sum(confs_block) / len(confs_block) if confs_block else None
    )
    if mean_conf_block is not None and mean_conf_block > 1.0:
        mean_conf_block = mean_conf_block / 100.0

    return [
        {
            "text": text_block,
            "section_type": SectionType.TEXT.value,
            "page_number": page_number,
            "bbox": block_bbox,
            "confidence": mean_conf_block,
            "ocr_engine": ocr_engine,
        }
    ]


def _extract_page_chunks(
    root: Any,
    ns_uri: str,
    page_number: int,
    granularity: str,
    ocr_engine: str | None,
) -> list[dict[str, Any]]:
    """
    Extract all raw chunks from one ALTO XML page document.

    Parameters
    ----------
    root : Element
        Root ``<alto>`` element.
    ns_uri : str
        Detected ALTO namespace URI.
    page_number : int
        Zero-based page index within the ZIP archive.
    granularity : str
        Chunking granularity: ``"block"``, ``"line"``, or ``"page"``.
    ocr_engine : str or None
        OCR engine name from the ALTO header.

    Returns
    -------
    list of dict
        Raw chunk dicts for this page.
    """
    block_tag = _ns("TextBlock", ns_uri)

    all_blocks = root.findall(f".//{block_tag}")
    if not all_blocks:
        logger.debug(
            "ALTOReader: page %d — no <TextBlock> elements found (ns=%r).",
            page_number,
            ns_uri,
        )
        return []

    if granularity == _GRAN_PAGE:
        # Collect all words from all blocks/lines into one mega-chunk
        string_tag = _ns("String", ns_uri)
        words: list[str] = []
        confs: list[float] = []
        for block in all_blocks:
            for string_el in block.findall(f".//{string_tag}"):
                content = string_el.get("CONTENT", "")
                if content:
                    words.append(content)
                wc = _attr_float(string_el, "WC")
                if wc is not None:
                    confs.append(wc)
        text = " ".join(words).strip()
        if not text:
            return []
        mean_conf: float | None = sum(confs) / len(confs) if confs else None
        if mean_conf is not None and mean_conf > 1.0:
            mean_conf = mean_conf / 100.0
        return [
            {
                "text": text,
                "section_type": SectionType.TEXT.value,
                "page_number": page_number,
                "bbox": None,  # page-level chunk has no single bbox
                "confidence": mean_conf,
                "ocr_engine": ocr_engine,
            }
        ]

    # block or line granularity
    chunks: list[dict[str, Any]] = []
    for block in all_blocks:
        chunks.extend(
            _extract_block_chunks(block, ns_uri, page_number, ocr_engine, granularity)
        )
    return chunks


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

    file_type: ClassVar[str | None] = ".zip"
    file_types: ClassVar[list[str] | None] = [".zip"]

    granularity: str = field(default=_GRAN_BLOCK)
    """
    Chunking granularity within each ALTO page. One of ``"block"``,
    ``"line"``, or ``"page"``. Default: ``"block"``.
    """

    max_file_bytes: int = field(default=5 * 1024 * 1024 * 1024)  # 5 GB
    """Maximum ZIP file size in bytes. Default: 5 GB."""

    xml_encoding: str | None = field(default=None)
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
                    root, ns_uri, page_idx, self.granularity, ocr_engine
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
