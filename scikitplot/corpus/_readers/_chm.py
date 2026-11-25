# scikitplot/corpus/_readers/_chm.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Reader for Microsoft Compiled HTML Help (``.chm``) files.

A CHM file is a compressed archive of HTML pages with a table of contents
(``.hhc``), the offline help and e-book format of Windows applications
from 1997 on. :class:`CHMReader` lists its pages, decompresses them with a
pure-Python LZX decoder (no new dependency, any platform) and parses each
through the same path :class:`~scikitplot.corpus._readers.HTMLReader`
uses, in table-of-contents order.
"""

from __future__ import annotations

import html as _html
import logging
import math
import mmap
import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generator, Optional  # noqa: F401

from .._base import (
    _TEXT_RATIO_FLOOR,
    COVERAGE_WARN_SHARE,
    DocumentReader,
    ReadCoverage,
    _valid_warn_share,
)
from .._schema import SourceType
from ._chm_format import ChmArchive, ChmEntry
from ._text import _bom_encoding, _decodes, _is_codec
from ._web import _declared_html_codec, _html_chunks, _load_beautifulsoup

logger = logging.getLogger(__name__)

__all__ = ["CHMReader"]

#: Page suffixes read from an archive.
_PAGE_SUFFIXES = (".htm", ".html", ".xhtml")

#: Bytes a page may decode to before it is skipped as too large.
_DEFAULT_MAX_PAGE_BYTES = 16 * 1024 * 1024

#: Decoded bytes (pages and the stream before them) a file may cost at most.
_DEFAULT_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024

#: Bytes of decoded HTML for each character of text the ratio allows.
_MAX_BYTES_PER_CHAR = 4

_DEFAULT_MAX_PAGES = 100_000
_DEFAULT_MAX_FILE_BYTES = 1024 * 1024 * 1024

#: Windows ANSI code page of a locale's primary language (LCID & 0x3FF),
#: the encoding legacy help compilers wrote pages in.
_ANSI_CODE_PAGE = {
    0x01: "cp1256",  # Arabic
    0x04: "cp936",  # Chinese (0x0404 Taiwan and 0x0C04 Hong Kong: cp950)
    0x05: "cp1250",  # Czech
    0x08: "cp1253",  # Greek
    0x0D: "cp1255",  # Hebrew
    0x0E: "cp1250",  # Hungarian
    0x11: "cp932",  # Japanese
    0x12: "cp949",  # Korean
    0x15: "cp1250",  # Polish
    0x18: "cp1250",  # Romanian
    0x19: "cp1251",  # Russian
    0x1A: "cp1250",  # Croatian / Serbian (Latin)
    0x1B: "cp1250",  # Slovak
    0x1E: "cp874",  # Thai
    0x1F: "cp1254",  # Turkish
    0x22: "cp1251",  # Ukrainian
    0x23: "cp1251",  # Belarusian
    0x24: "cp1250",  # Slovenian
    0x25: "cp1257",  # Estonian
    0x26: "cp1257",  # Latvian
    0x27: "cp1257",  # Lithuanian
    0x2A: "cp1258",  # Vietnamese
    0x2F: "cp1251",  # Macedonian
    0x02: "cp1251",  # Bulgarian
}
_TRADITIONAL_CHINESE = {0x0404, 0x0C04, 0x1404}

#: An opening ``<object ...>`` or a closing ``</object>`` tag. Blocks are
#: paired by one scan over these tokens (:func:`_object_blocks`), not by
#: ``<object[^>]*>(.*?)</object>``, which rescans to the end from every
#: unclosed opening tag: quadratic on a hostile contents file.
_OBJECT_TAG_RE = re.compile(r"<(/?)object\b[^<>]*>", re.IGNORECASE)
_PARAM_RE = re.compile(
    r"""<param\s+name\s*=\s*["']?([^"'>\s]+)["']?\s+value\s*=\s*"([^"]*)\"""",
    re.IGNORECASE,
)


def _ansi_code_page(lcid: int) -> str | None:
    """
    Return the Windows ANSI code page for a locale identifier, if known.

    Parameters
    ----------
    lcid : int
        Windows locale identifier from the archive header.

    Returns
    -------
    str or None
        A codec name; ``None`` for Western locales (``cp1252`` follows).

    Examples
    --------
    >>> _ansi_code_page(0x0419)
    'cp1251'
    >>> _ansi_code_page(0x0404)
    'cp950'
    >>> _ansi_code_page(0x0409) is None
    True
    """
    if lcid in _TRADITIONAL_CHINESE:
        return "cp950"
    return _ANSI_CODE_PAGE.get(lcid & 0x3FF)


def _page_codec(raw: bytes, lcid: int) -> str:
    """
    Return the codec for one page: BOM, UTF-8, ``<meta>``, locale, cp1252.

    Each candidate is used only when it decodes the bytes, so the result
    depends on the bytes and the archive's locale alone.
    """
    bom = _bom_encoding(raw)
    if bom is not None:
        return bom
    if _decodes(raw, "utf-8"):
        return "utf-8"
    declared = _declared_html_codec(raw)
    if declared is not None:
        return declared
    ansi = _ansi_code_page(lcid)
    if ansi is not None and _decodes(raw, ansi):
        return ansi
    return "cp1252" if _decodes(raw, "cp1252") else "latin-1"


def _object_blocks(text: str) -> list[str]:
    """
    Return the inside of each ``<object>...</object>`` block, in order.

    One pass over the tags: an opening tag starts a block (a later one,
    before any close, restarts it), a closing tag ends the open block.
    Linear in *text*, however many tags are left unclosed.

    Examples
    --------
    >>> _object_blocks("<OBJECT a>x</OBJECT><object>y<object>z</object></object>")
    ['x', 'z']
    """
    blocks: list[str] = []
    start: int | None = None
    for match in _OBJECT_TAG_RE.finditer(text):
        if match.group(1):
            if start is not None:
                blocks.append(text[start : match.start()])
                start = None
        else:
            start = match.end()
    return blocks


def _toc(text: str, base: str) -> list[tuple[str, str]]:
    r"""
    Return ``(page key, title)`` pairs of a table of contents, in order.

    Parameters
    ----------
    text : str
        Decoded ``.hhc`` sitemap.
    base : str
        Directory of the ``.hhc`` inside the archive, for relative links.

    Returns
    -------
    list of (str, str)
        Lower-case archive paths (anchors dropped), first occurrence only.

    Examples
    --------
    >>> _toc(
    ...     '<OBJECT type="text/sitemap"><param name="Name" value="Intro">'
    ...     '<param name="Local" value="html\\intro.htm#top"></OBJECT>',
    ...     "/",
    ... )
    [('/html/intro.htm', 'Intro')]
    """
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for block in _object_blocks(text):
        params = {k.lower(): _html.unescape(v) for k, v in _PARAM_RE.findall(block)}
        local = params.get("local")
        if not local:
            continue
        if "::" in local:  # mk:@MSITStore:book.chm::/page.htm
            local = local.split("::", 1)[1]
        local = local.split("#", 1)[0].replace("\\", "/").strip()
        if not local or "://" in local:
            continue
        path = posixpath.normpath(posixpath.join(base, local))
        key = ("/" + path.lstrip("/")).lower()
        if key not in seen:
            seen.add(key)
            entries.append((key, params.get("name", "")))
    return entries


@dataclass
class CHMReader(DocumentReader):
    """
    Read a Compiled HTML Help (``.chm``) file into chunks, page by page.

    Pages are read in table-of-contents order, then any page the contents
    do not list, in archive order. Each page is parsed as
    :class:`~scikitplot.corpus._readers.HTMLReader` parses a file: title,
    headings, paragraphs, list items, table cells, quotes and code, with
    scripts and styles dropped. Images, the index and the full-text search
    data are not read.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.chm`` file.
    encoding : str or None, optional
        Codec for every page. ``None`` (default) decides per page: byte-order
        mark, strict UTF-8, the ``<meta>`` charset, the archive locale's
        Windows code page, then ``cp1252``. The codec used is in each
        chunk's ``encoding``.
    extract_tags : list of str or None, optional
        HTML tags to extract; ``None`` uses the
        :class:`~scikitplot.corpus._readers.WebReader` set.
    max_file_bytes : int, optional
        Larger files are refused. Default: 1 GiB.
    max_pages : int, optional
        Pages read at most; later ones are skipped and counted. Default:
        100 000.
    max_page_bytes : int, optional
        A page larger than this is skipped and counted (``"too-large"``).
        Default: 16 MiB.
    max_total_bytes : int, optional
        Most bytes the decompressor may produce for one file, including
        the bytes it decodes only to reach a page. Default: 2 GiB.
    max_text_chars_per_byte : float or None, optional
        Ratio limit, as for :class:`~scikitplot.corpus._readers.ZipReader`:
        the decoding limit in force is the smaller of ``max_total_bytes``
        and ``4 x max(ratio x file size, 1 Mi)``. Default: 50.
    coverage_warn_share : float, optional
        Share of pages skipped or empty above which one coverage warning
        is logged. Default: ``0.5``.

    Attributes
    ----------
    file_types : list of str
        Class variable. ``[".chm"]``.
    coverage : ReadCoverage
        After a read: pages read, empty and skipped (by reason).

    Raises
    ------
    ImportError
        If ``beautifulsoup4`` is not installed.
    ValueError
        If the file is not a CHM archive, is corrupt, or exceeds a limit
        that refuses (size, decoding).

    See Also
    --------
    scikitplot.corpus._readers.HTMLReader : One local HTML file.
    scikitplot.corpus._readers.ZipReader : Archives of other formats.

    Notes
    -----
    **User note:** Each chunk carries ``chm_page`` (the page's path in the
    archive), ``page_index`` (its reading position), ``toc_title`` (its
    title in the contents, when listed) and ``input_path``
    (``"<file>/<page>"``). ``source_type`` is
    :attr:`SourceType.DOCUMENTATION`.

    **Developer note:** The container and the LZX decoder are in
    ``_chm_format`` (standard library only). A page is decoded from the
    reset point before it, never the whole stream, and each 32 KiB frame
    at most once per read; the decoding limit counts every decoded byte,
    so a decompression bomb stops at the limit. Pure Python decodes
    real help files at about 10 MB/s, since most of their bytes are copies.

    Examples
    --------
    >>> from pathlib import Path
    >>> from scikitplot.corpus._base import DocumentReader
    >>> import scikitplot.corpus._readers
    >>> reader = DocumentReader.create(Path("manual.chm"))
    >>> docs = list(reader.get_documents())
    >>> reader.coverage.summary()  # doctest: +SKIP
    '312 read, 2 empty of 314 page(s)'
    """

    file_type: ClassVar[str] = ".chm"
    file_types: ClassVar[list[str] | None] = [".chm"]

    encoding: str | None = field(default=None)
    """Codec for every page; ``None`` decides per page."""

    extract_tags: list[str] | None = field(default=None)
    """HTML tags to extract. ``None`` uses the :class:`WebReader` set."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Larger files are refused. Default: 1 GiB."""

    max_pages: int = field(default=_DEFAULT_MAX_PAGES)
    """Pages read at most. Default: 100 000."""

    max_page_bytes: int = field(default=_DEFAULT_MAX_PAGE_BYTES)
    """Larger pages are skipped and counted. Default: 16 MiB."""

    max_total_bytes: int = field(default=_DEFAULT_MAX_TOTAL_BYTES)
    """Most bytes decoded for one file. Default: 2 GiB."""

    max_text_chars_per_byte: float | None = field(default=50)
    """Ratio limit on decoding (and on text, as for every reader)."""

    coverage_warn_share: float = field(default=COVERAGE_WARN_SHARE)
    """Share of pages skipped or empty that logs a coverage warning."""

    def __post_init__(self) -> None:
        """Validate the limits and the encoding."""
        super().__post_init__()
        for name in (
            "max_file_bytes",
            "max_pages",
            "max_page_bytes",
            "max_total_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"CHMReader: {name} must be an int > 0; got {value!r}."
                )
        if self.encoding is not None and not _is_codec(self.encoding):
            raise ValueError(
                f"CHMReader: encoding {self.encoding!r} is not a codec Python knows."
            )
        self.coverage_warn_share = _valid_warn_share(
            self.coverage_warn_share, "CHMReader"
        )
        self.coverage = ReadCoverage(unit="page")

    def _decode_limit(self, size: int) -> tuple[int, str]:
        """Return the decoding limit for a file of *size* bytes and its name."""
        limit = self.max_total_bytes
        described = f"max_total_bytes={limit:,}"
        ratio = self.max_text_chars_per_byte
        if ratio is not None:
            by_ratio = _MAX_BYTES_PER_CHAR * max(
                math.ceil(ratio * size), _TEXT_RATIO_FLOOR
            )
            if by_ratio < limit:
                limit = by_ratio
                described = (
                    f"{by_ratio:,} bytes for a file of {size:,} bytes"
                    f" (max_text_chars_per_byte={ratio}); pass a larger"
                    " max_text_chars_per_byte (or None)"
                )
        return limit, described

    def _plan(
        self, archive: ChmArchive, limit: int
    ) -> tuple[list[ChmEntry], dict[str, str]]:
        """Return the pages to read in reading order, and their contents titles."""
        pages = [
            e
            for e in archive.entries
            if e.name.lower().endswith(_PAGE_SUFFIXES)
            and not e.name.startswith(("/#", "/$", "::"))
        ]
        titles: dict[str, str] = {}
        contents = next(
            (e for e in archive.entries if e.name.lower().endswith(".hhc")), None
        )
        order: list[str] = []
        if contents is not None and contents.length <= self.max_page_bytes:
            try:
                [(_, raw)] = archive.read_many([contents], max_decoded_bytes=limit)
            except ValueError as exc:
                logger.warning(
                    "CHMReader: %s: its table of contents is unreadable (%s);"
                    " pages follow archive order.",
                    self.file_name,
                    exc,
                )
            else:
                text = raw.decode(_page_codec(raw, archive.lcid), errors="replace")
                base = posixpath.dirname(contents.name) or "/"
                for key, title in _toc(text, base):
                    order.append(key)
                    titles[key] = title
        rank = {key: i for i, key in enumerate(order)}
        listed = len(rank)
        pages.sort(key=lambda e: rank.get(e.name.lower(), listed))
        return pages, titles

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:  # noqa: PLR0912
        """
        Yield one chunk per text element of each page, in reading order.

        Yields
        ------
        dict
            The :class:`HTMLReader` keys plus ``chm_page``, ``page_index``,
            ``toc_title`` and ``input_path``.

        Raises
        ------
        ImportError
            If ``beautifulsoup4`` is not installed.
        ValueError
            If the file is not a CHM archive, is corrupt, or exceeds a limit.
        """
        coverage = ReadCoverage(unit="page")
        self.coverage = coverage
        soup = _load_beautifulsoup("CHMReader")
        size = self.input_path.stat().st_size
        if size > self.max_file_bytes:
            raise ValueError(
                f"CHMReader: {self.file_name} is {size:,} bytes, which exceeds"
                f" max_file_bytes={self.max_file_bytes:,}."
            )
        if size == 0:
            raise ValueError(f"CHMReader: {self.file_name} is empty.")
        limit, described = self._decode_limit(size)
        with open(self.input_path, "rb") as fh, mmap.mmap(
            fh.fileno(), 0, access=mmap.ACCESS_READ
        ) as view, ChmArchive(view, self.file_name) as archive:
            lcid = archive.lcid
            pages, titles = self._plan(archive, limit)
            wanted: list[ChmEntry] = []
            for entry in pages:
                if entry.length > self.max_page_bytes:
                    coverage.skip("too-large")
                elif len(wanted) >= self.max_pages:
                    coverage.skip("limit")
                else:
                    wanted.append(entry)
            try:
                content = dict(archive.read_many(wanted, max_decoded_bytes=limit))
            except ValueError as exc:
                if "decoding needs more" in str(exc):
                    raise ValueError(f"{exc} ({described}).") from exc
                raise
        for index, entry in enumerate(wanted):
            raw = content[entry]
            codec = self.encoding or _page_codec(raw, lcid)
            text = raw.decode(codec, errors="replace").replace("\r\n", "\n")
            key = entry.name.lower()
            n_chunks = 0
            for chunk in _html_chunks(text, self.extract_tags, soup):
                chunk["source_type"] = SourceType.DOCUMENTATION.value
                chunk["encoding"] = codec
                chunk["chm_page"] = entry.name
                chunk["page_index"] = index
                chunk["input_path"] = f"{self.file_name}{entry.name}"
                if titles.get(key):
                    chunk["toc_title"] = titles[key]
                n_chunks += 1
                yield chunk
            if n_chunks:
                coverage.read += 1
            else:
                coverage.empty += 1
        if coverage.is_poor(self.coverage_warn_share):
            logger.warning(
                "CHMReader: %s gave little text: %s.",
                self.file_name,
                coverage.summary(),
            )
        logger.info("CHMReader: finished %s: %s", self.file_name, coverage.summary())
