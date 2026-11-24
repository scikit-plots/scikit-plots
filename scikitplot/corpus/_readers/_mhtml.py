# scikitplot/corpus/_readers/_mhtml.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Reader for saved web pages in MHTML (``.mhtml``, ``.mht``).

MHTML (RFC 2557) is a MIME message holding a page and its resources: what
a browser writes for "Save page as... single file". :class:`MHTMLReader`
parses it with the standard library :mod:`email` package and reads each
HTML part (the page first, then any frames) through the parsing
:class:`~scikitplot.corpus._readers.HTMLReader` uses. Images, styles and
scripts are not read; nothing is fetched.
"""

from __future__ import annotations

import email
import email.policy
import logging
from dataclasses import dataclass, field
from email.message import Message
from typing import Any, ClassVar, Generator, Optional  # noqa: F401

from .._base import (
    COVERAGE_WARN_SHARE,
    DocumentReader,
    ReadCoverage,
    _valid_warn_share,
)
from .._schema import SourceType
from ._text import _bom_encoding, _decodes, _is_codec
from ._web import _declared_html_codec, _html_chunks, _load_beautifulsoup

logger = logging.getLogger(__name__)

__all__ = ["MHTMLReader"]

_DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_PART_BYTES = 50 * 1024 * 1024
_DEFAULT_MAX_PARTS = 10_000
#: Deepest multipart nesting accepted; browsers write one or two levels.
_MAX_DEPTH = 32
_HTML_TYPES = ("text/html", "application/xhtml+xml")


def _part_codec(raw: bytes, declared: str | None) -> str:
    r"""
    Return the codec for one HTML part.

    Order: byte-order mark, the part's ``Content-Type`` charset (the saver
    wrote the bytes in it), strict UTF-8, the ``<meta>`` charset, then
    ``cp1252`` and ``latin-1``. Each is used only when it decodes the bytes.

    Examples
    --------
    >>> _part_codec(b"caf\xe9", "iso-8859-1")
    'iso-8859-1'
    >>> _part_codec("caf\u00e9".encode(), "ascii")
    'utf-8'
    """
    bom = _bom_encoding(raw)
    if bom is not None:
        return bom
    if declared and _is_codec(declared) and _decodes(raw, declared):
        return declared
    if _decodes(raw, "utf-8"):
        return "utf-8"
    meta = _declared_html_codec(raw)
    if meta is not None:
        return meta
    return "cp1252" if _decodes(raw, "cp1252") else "latin-1"


def _walk(message: Message, depth: int = 0) -> Generator[Message, None, None]:
    """Yield the leaf parts of *message* in order, refusing deep nesting."""
    if depth > _MAX_DEPTH:
        raise ValueError(f"multipart nesting deeper than {_MAX_DEPTH} levels")
    if message.is_multipart():
        for part in message.get_payload():
            yield from _walk(part, depth + 1)
    else:
        yield message


def _root_first(parts: list[Message], start: str | None) -> list[Message]:
    """Put the root part first: the ``start`` Content-ID, else the first."""
    if start:
        cid = start.strip().strip("<>")
        for i, part in enumerate(parts):
            if (part.get("Content-ID") or "").strip().strip("<>") == cid:
                return [part, *parts[:i], *parts[i + 1 :]]
    return parts


@dataclass
class MHTMLReader(DocumentReader):
    """
    Read a saved web page (``.mhtml``, ``.mht``) into one chunk per element.

    Every HTML part is read, the page first (the ``start`` part, else the
    first one) and then any frames, each parsed as
    :class:`~scikitplot.corpus._readers.HTMLReader` parses a file.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.mhtml`` or ``.mht`` file.
    encoding : str or None, optional
        Codec for every HTML part. ``None`` (default) decides per part:
        byte-order mark, the part's declared charset, UTF-8, the ``<meta>``
        charset, ``cp1252``. The codec used is in each chunk's
        ``encoding``.
    extract_tags : list of str or None, optional
        HTML tags to extract; ``None`` uses the
        :class:`~scikitplot.corpus._readers.WebReader` set.
    max_file_bytes : int, optional
        Larger files are refused. Default: 256 MiB.
    max_parts : int, optional
        MIME parts accepted; a message with more is refused. Default:
        10 000.
    max_part_bytes : int, optional
        An HTML part decoding to more than this is skipped and counted
        (``"too-large"``). Default: 50 MiB.
    coverage_warn_share : float, optional
        Share of HTML parts skipped or empty above which one coverage
        warning is logged. Default: ``0.5``.

    Attributes
    ----------
    file_types : list of str
        Class variable. ``[".mhtml", ".mht"]``.
    coverage : ReadCoverage
        After a read: HTML parts read, empty and skipped.

    Raises
    ------
    ImportError
        If ``beautifulsoup4`` is not installed.
    ValueError
        If the file is too large, has no HTML part, nests too deeply or has
        too many parts.

    See Also
    --------
    scikitplot.corpus._readers.HTMLReader : A page saved as plain HTML.
    scikitplot.corpus._readers.WebReader : A live page.

    Notes
    -----
    **User note:** Each chunk carries ``mhtml_part`` (the part's
    ``Content-Location``), ``part_index`` and ``encoding``; ``url`` is
    the page's original address (``Snapshot-Content-Location`` or the
    root part's ``Content-Location``) when it is ``http(s)``.
    ``source_type`` is :attr:`SourceType.WEB`.

    **Developer note:** Parsing uses :mod:`email` with the default policy;
    quoted-printable and base64 bodies are decoded only for HTML parts, and
    a decoded part is at most the file's size, so the limits on the file
    bound the work.

    Examples
    --------
    >>> from pathlib import Path
    >>> from scikitplot.corpus._base import DocumentReader
    >>> import scikitplot.corpus._readers
    >>> reader = DocumentReader.create(Path("saved_page.mhtml"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".mhtml"
    file_types: ClassVar[list[str] | None] = [".mhtml", ".mht"]

    encoding: str | None = field(default=None)
    """Codec for every HTML part; ``None`` decides per part."""

    extract_tags: list[str] | None = field(default=None)
    """HTML tags to extract. ``None`` uses the :class:`WebReader` set."""

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """Larger files are refused. Default: 256 MiB."""

    max_parts: int = field(default=_DEFAULT_MAX_PARTS)
    """MIME parts accepted. Default: 10 000."""

    max_part_bytes: int = field(default=_DEFAULT_MAX_PART_BYTES)
    """Larger HTML parts are skipped and counted. Default: 50 MiB."""

    coverage_warn_share: float = field(default=COVERAGE_WARN_SHARE)
    """Share of HTML parts skipped or empty that logs a coverage warning."""

    def __post_init__(self) -> None:
        """Validate the limits and the encoding."""
        super().__post_init__()
        for name in ("max_file_bytes", "max_parts", "max_part_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"MHTMLReader: {name} must be an int > 0; got {value!r}."
                )
        if self.encoding is not None and not _is_codec(self.encoding):
            raise ValueError(
                f"MHTMLReader: encoding {self.encoding!r} is not a codec Python knows."
            )
        self.coverage_warn_share = _valid_warn_share(
            self.coverage_warn_share, "MHTMLReader"
        )
        self.coverage = ReadCoverage(unit="part")

    def _parts(self, message: Message) -> list[Message]:
        """Return the HTML parts, root first, within the part limit."""
        try:
            leaves = []
            for leaf in _walk(message):
                leaves.append(leaf)
                if len(leaves) > self.max_parts:
                    raise ValueError(
                        f"more than max_parts={self.max_parts:,} MIME parts"
                    )
        except (ValueError, RecursionError) as exc:
            raise ValueError(f"MHTMLReader: {self.file_name}: {exc}.") from exc
        html = [p for p in leaves if p.get_content_type() in _HTML_TYPES]
        return _root_first(html, message.get_param("start"))

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield one chunk per text element of each HTML part.

        Yields
        ------
        dict
            The :class:`HTMLReader` keys plus ``mhtml_part``, ``part_index``
            and, when known, ``url``.

        Raises
        ------
        ImportError
            If ``beautifulsoup4`` is not installed.
        ValueError
            If a limit refuses the file or it holds no HTML part.
        """
        coverage = ReadCoverage(unit="part")
        self.coverage = coverage
        soup = _load_beautifulsoup("MHTMLReader")
        size = self.input_path.stat().st_size
        if size > self.max_file_bytes:
            raise ValueError(
                f"MHTMLReader: {self.file_name} is {size:,} bytes, which exceeds"
                f" max_file_bytes={self.max_file_bytes:,}."
            )
        with open(self.input_path, "rb") as fh:
            message = email.message_from_binary_file(fh, policy=email.policy.compat32)
        parts = self._parts(message)
        if not parts:
            raise ValueError(
                f"MHTMLReader: {self.file_name} holds no HTML part; it is not a"
                " saved web page."
            )
        page_url = message.get("Snapshot-Content-Location") or parts[0].get(
            "Content-Location"
        )
        url = page_url.strip() if page_url else None
        if url is not None and not url.lower().startswith(("http://", "https://")):
            url = None
        for index, part in enumerate(parts):
            raw = part.get_payload(decode=True) or b""
            if len(raw) > self.max_part_bytes:
                coverage.skip("too-large")
                continue
            codec = self.encoding or _part_codec(raw, part.get_content_charset())
            text = raw.decode(codec, errors="replace").replace("\r\n", "\n")
            location = (part.get("Content-Location") or "").strip()
            n_chunks = 0
            for chunk in _html_chunks(text, self.extract_tags, soup):
                chunk["source_type"] = SourceType.WEB.value
                chunk["encoding"] = codec
                chunk["mhtml_part"] = location
                chunk["part_index"] = index
                if url is not None:
                    chunk["url"] = url
                n_chunks += 1
                yield chunk
            if n_chunks:
                coverage.read += 1
            else:
                coverage.empty += 1
        if coverage.is_poor(self.coverage_warn_share):
            logger.warning(
                "MHTMLReader: %s gave little text: %s.",
                self.file_name,
                coverage.summary(),
            )
        logger.info("MHTMLReader: finished %s: %s", self.file_name, coverage.summary())
