"""
scikitplot.corpus._readers.text
================================
Plain-text document reader for the scikitplot corpus pipeline.

Handles ``.txt``, ``.md``, and ``.rst`` files (and any other extension
registered by subclassing) with a robust multi-stage encoding detection
chain so that documents written in any encoding — UTF-8, UTF-16,
Latin-1, Windows-1252, and more — are read correctly without manual
configuration.

Original issues fixed (from remarx ``text_input.py``):

1. **Hard-coded UTF-8** — replaced by a three-stage encoding detection
   chain: BOM → chardet (if installed) → Latin-1 last resort.
2. **BOM character leaking** — BOM bytes are stripped before returning
   text by using ``utf-8-sig`` when a UTF-8 BOM is detected.
3. **No file-size guard** — ``max_file_bytes`` parameter allows callers
   to reject unexpectedly large files before reading into memory.
4. **No section_type in yielded dict** — every chunk now includes
   ``section_type = SectionType.TEXT``.
5. **Single extension** — ``TextReader`` handles ``.txt``; subclasses
   can trivially register ``.md`` / ``.rst`` by setting ``file_type``.

Python compatibility:

Python 3.8-3.15. Zero external runtime dependencies (``chardet`` is
optional and probed at call time).
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Generator, Optional  # noqa: F401

from .._base import DocumentReader
from .._schema import SectionType

logger = logging.getLogger(__name__)

__all__ = [
    "MarkdownReader",
    "ReSTReader",
    "TextReader",
]

# Default upper bound for files read into memory in one shot.
# 500 MB should be generous for any realistic text corpus file.
_DEFAULT_MAX_FILE_BYTES: int = 500 * 1024 * 1024  # 500 MB

# Minimum confidence threshold for accepting chardet's encoding guess.
_CHARDET_MIN_CONFIDENCE: float = 0.80

# Number of bytes read for encoding sniffing (avoids loading huge files
# just to determine encoding).
_SNIFF_BYTES: int = 8192


def _detect_encoding(raw: bytes) -> str:  # noqa: PLR0911
    r"""
    Detect the encoding of ``raw`` bytes using a priority chain.

    Priority order
    --------------
    1. BOM markers (definitive, no false positives).
    2. Strict UTF-8 decode (most common case for modern corpora).
    3. chardet library (if installed, confidence >= 0.80).
    4. Latin-1 fallback (accepts every byte value; never raises).

    Parameters
    ----------
    raw : bytes
        Raw bytes to inspect (need not be the entire file; a head
        sample of several kilobytes is sufficient).

    Returns
    -------
    str
        Python codec name suitable for ``bytes.decode(enc)``.

    Notes
    -----
    Latin-1 as the final fallback is intentional: every byte in
    [0x00, 0xFF] is a valid Latin-1 code point, so it never raises
    ``UnicodeDecodeError``. The decoded text may contain mojibake for
    genuinely non-Latin-1 files, but that is preferable to a crash.

    Examples
    --------
    >>> _detect_encoding(b"\xef\xbb\xbfHello")
    'utf-8-sig'
    >>> _detect_encoding(b"Hello world")
    'utf-8'
    """
    # --- Stage 1: BOM markers ---
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith(b"\xff\xfe\x00\x00"):
        return "utf-32-le"
    if raw.startswith(b"\x00\x00\xfe\xff"):
        return "utf-32-be"
    if raw.startswith(b"\xff\xfe"):
        return "utf-16-le"
    if raw.startswith(b"\xfe\xff"):
        return "utf-16-be"

    # --- Stage 2: Strict UTF-8 ---
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # --- Stage 3: chardet (optional dependency) ---
    try:
        import chardet  # type: ignore[] # noqa: PLC0415

        result = chardet.detect(raw[:_SNIFF_BYTES])
        confidence = result.get("confidence") or 0.0
        enc = result.get("encoding")
        if enc and confidence >= _CHARDET_MIN_CONFIDENCE:
            logger.debug(
                "TextReader: chardet detected encoding %r (confidence=%.2f).",
                enc,
                confidence,
            )
            return enc
    except ImportError:
        pass  # chardet not installed; continue to fallback

    # --- Stage 4: Latin-1 fallback (never raises) ---
    logger.warning(
        "TextReader: could not confidently detect encoding; "
        "falling back to 'latin-1'. Install chardet for better detection: "
        "pip install chardet"
    )
    return "latin-1"


@dataclass
class TextReader(DocumentReader):
    """
    Plain-text document reader.

    Reads a single text file, detects its encoding automatically, and
    yields the entire file content as one raw text chunk. Downstream
    chunking (sentence, paragraph, fixed-window) is handled by the
    injected :class:`~scikitplot.corpus._base.ChunkerBase`.

    Registers for the ``.txt`` extension. To handle ``.md`` or ``.rst``
    files, subclass and set ``file_type`` accordingly::

        @dataclass
        class MarkdownReader(TextReader):
            file_type = ".md"

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.txt`` file.
    encoding : str or None, optional
        Explicit encoding override. When ``None`` (default), encoding is
        detected automatically via BOM → chardet → Latin-1.
    max_file_bytes : int, optional
        Maximum file size in bytes. Files larger than this limit raise
        ``ValueError`` before any bytes are read. Default: 500 MB.
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
        Class variable. Always ``".txt"``.
    file_types : list of str
        Class variable. Registered extensions:
        ``[".txt"]``.

    Raises
    ------
    ValueError
        If the file exceeds ``max_file_bytes``.
    UnicodeDecodeError
        Should never occur (Latin-1 fallback handles all byte values),
        but may be raised if an explicit ``encoding`` override is wrong.

    See Also
    --------
    scikitplot.corpus._readers.XMLReader : XML/TEI document reader.
    scikitplot.corpus._readers.PDFReader : PDF document reader.

    Notes
    -----
    **Encoding detection** uses the ``_detect_encoding()`` helper which
    reads at most ``_SNIFF_BYTES`` (8 KB) for the chardet probe, then
    reads the full file with the detected codec. This means large files
    are read twice from disk only in the chardet path; the UTF-8 BOM and
    strict-UTF-8 paths read the full file only once.

    **Memory:** The entire file is read into a single Python ``str``. For
    files larger than a few hundred MB, consider splitting before ingestion
    or increasing ``max_file_bytes`` deliberately to confirm the intent.

    Examples
    --------
    Default usage (encoding auto-detected):

    >>> from pathlib import Path
    >>> reader = TextReader(input_path=Path("corpus.txt"))
    >>> docs = list(reader.get_documents())

    Explicit encoding:

    >>> reader = TextReader(input_path=Path("corpus.txt"), encoding="utf-8")

    Subclass for Markdown:

    >>> @dataclass
    ... class MarkdownReader(TextReader):
    ...     file_type = ".md"
    >>> reader = MarkdownReader(input_path=Path("notes.md"))
    """

    file_type: ClassVar[str] = ".txt"
    file_types: ClassVar[list[str] | None] = [".txt"]

    encoding: Optional[str] = field(default=None)  # noqa: UP045
    """
    Explicit encoding override. ``None`` triggers automatic detection.
    Example values: ``"utf-8"``, ``"latin-1"``, ``"windows-1252"``.
    """

    max_file_bytes: int = field(default=_DEFAULT_MAX_FILE_BYTES)
    """
    Maximum file size in bytes before rejecting. Default: 500 MB.
    """

    def __post_init__(self) -> None:
        """
        Validate parameters and delegate to parent ``__post_init__``.
        """
        super().__post_init__()
        if self.max_file_bytes <= 0:
            raise ValueError(
                f"TextReader: max_file_bytes must be > 0; got {self.max_file_bytes!r}."
            )

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Read the text file and yield a single raw chunk dict.

        Yields
        ------
        dict
            Single dict with keys ``"text"`` (full file content as str)
            and ``"section_type"`` (``SectionType.TEXT``).

        Raises
        ------
        ValueError
            If the file size exceeds ``max_file_bytes``.
        OSError
            If the file cannot be opened or read.
        UnicodeDecodeError
            If an explicit ``encoding`` is set and the file is not valid
            in that encoding.

        Notes
        -----
        Only one chunk is yielded regardless of file content. All
        sub-chunking into sentences/paragraphs/windows is delegated to
        the injected :class:`~scikitplot.corpus._base.ChunkerBase`.
        """
        file_size = self.input_path.stat().st_size

        # Guard: reject oversized files before reading
        if file_size > self.max_file_bytes:
            raise ValueError(
                f"TextReader: {self.file_name} is {file_size:,} bytes, which"
                f" exceeds max_file_bytes={self.max_file_bytes:,}."
                f" Increase max_file_bytes or split the file before ingestion."
            )

        # Determine encoding
        if self.encoding is not None:
            # Caller supplied explicit encoding — use directly
            enc = self.encoding
            logger.debug(
                "TextReader: using explicit encoding %r for %s.", enc, self.file_name
            )
        else:
            # Auto-detect from head bytes
            with self.input_path.open("rb") as fh:
                head_bytes = fh.read(_SNIFF_BYTES)
            enc = _detect_encoding(head_bytes)
            logger.debug(
                "TextReader: detected encoding %r for %s.", enc, self.file_name
            )

        # Read full file with detected/explicit encoding
        text = self.input_path.read_text(encoding=enc)

        if not text.strip():
            logger.warning(
                "TextReader: %s is empty or contains only whitespace.",
                self.file_name,
            )
            return

        logger.info(
            "TextReader: read %s (%d chars, encoding=%r).",
            self.file_name,
            len(text),
            enc,
        )

        yield {
            "text": text,
            "section_type": SectionType.TEXT.value,
        }


# ---------------------------------------------------------------------------
# Extension aliases — register additional text-like extensions without
# duplicating any logic. Each subclass inherits all TextReader behaviour
# verbatim; only the registered extension differs.
# ---------------------------------------------------------------------------


@dataclass
class MarkdownReader(TextReader):
    """
    Markdown document reader.

    Identical to :class:`TextReader` in every respect — Markdown files
    are read as plain text without any Markdown-specific parsing.
    Downstream :class:`~scikitplot.corpus._base.ChunkerBase` subclasses
    (e.g. :class:`~scikitplot.corpus._chunkers.ParagraphChunker`) operate
    on the raw Markdown source, which is suitable for most NLP corpus
    use cases.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.md`` file.

    Attributes
    ----------
    file_type : str
        Class variable. Always ``".md"``.
    file_types : list of str
        Class variable. Registered extensions:
        ``[".md"]``.

    See Also
    --------
    TextReader : Base class — all parameters and behaviour inherited.
    ReSTReader : For ``.rst`` files.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = MarkdownReader(input_path=Path("README.md"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".md"
    file_types: ClassVar[list[str] | None] = [".md"]


@dataclass
class ReSTReader(TextReader):
    """
    reStructuredText document reader.

    Identical to :class:`TextReader` — ``.rst`` files are treated as
    plain text. RST markup (``::`` blocks, ``.. directive::``, etc.)
    is preserved in the raw text yielded to the chunker.

    Parameters
    ----------
    input_path : pathlib.Path
        Path to the ``.rst`` file.

    Attributes
    ----------
    file_type : str
        Class variable. Always ``".rst"``.
    file_types : list of str
        Class variable. Registered extensions:
        ``[".rst"]``.

    See Also
    --------
    TextReader : Base class — all parameters and behaviour inherited.
    MarkdownReader : For ``.md`` files.

    Examples
    --------
    >>> from pathlib import Path
    >>> reader = ReSTReader(input_path=Path("CHANGES.rst"))
    >>> docs = list(reader.get_documents())
    """

    file_type: ClassVar[str] = ".rst"
    file_types: ClassVar[list[str] | None] = [".rst"]
