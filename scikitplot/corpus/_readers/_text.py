# scikitplot/corpus/_readers/_text.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._readers.text
===============================

Plain-text document reader for the scikitplot corpus pipeline.

Handles ``.txt``, ``.md``, and ``.rst`` files (and any other extension
registered by subclassing) with a robust multi-stage encoding detection
chain so that documents written in any encoding — UTF-8, UTF-16,
Latin-1, Windows-1252, and more — are read correctly without manual
configuration.

Original issues fixed (from remarx ``text_input.py``):

1. **Hard-coded UTF-8** -- replaced by an encoding detection chain over
   the whole file: byte-order mark, UTF-8, chardet (if installed), cp1252,
   latin-1; a binary file raises ``ValueError``.
2. **BOM character leaking** -- the byte-order mark is not kept in the
   text: ``utf-8-sig``, ``utf-16`` and ``utf-32`` consume it.
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

import codecs
import logging
from dataclasses import dataclass, field
from typing import (  # noqa: F401
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Optional,
    Sequence,
)

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
# BUG-05 fix: renamed from _TEXT_DEFAULT_MAX_FILE_BYTES to be format-specific.
_TEXT_DEFAULT_MAX_FILE_BYTES: int = 500 * 1024 * 1024  # 500 MB

# Minimum confidence threshold for accepting chardet's encoding guess.
_CHARDET_MIN_CONFIDENCE: float = 0.80

# Number of bytes read for encoding sniffing (avoids loading huge files
# just to determine encoding).
_SNIFF_BYTES: int = 8192


def _detect_encoding(raw: bytes, *, use_chardet: bool = False) -> str | None:
    r"""
    Return the codec that decodes ``raw``, or ``None`` if it is not text.

    Priority order
    --------------
    1. A byte-order mark: ``utf-8-sig``, ``utf-32`` or ``utf-16``. These
       codecs read the byte order from the mark and drop it.
    2. ``None`` when ``raw`` holds a NUL byte: without a byte-order mark,
       text files do not contain NUL; binary files nearly always do. This
       comes before UTF-8 because NUL is valid UTF-8 (U+0000), and UTF-16
       text without a mark is valid UTF-8 full of NULs.
    3. Strict UTF-8 of all of ``raw``.
    4. With ``use_chardet``: chardet's guess, when its confidence is at
       least 0.80 and the codec it names decodes all of ``raw``.
    5. ``cp1252`` when it decodes all of ``raw``, else ``latin-1``.

    Parameters
    ----------
    raw : bytes
        The whole file. A prefix is not enough: a multi-byte character cut
        at its end fails strict UTF-8, and a non-UTF-8 byte after it is
        missed.
    use_chardet : bool, optional
        Ask chardet before the fallback. Default: ``False``.

    Returns
    -------
    str or None
        Python codec name suitable for ``bytes.decode(enc)``, or ``None``
        for binary content.

    Notes
    -----
    **cp1252 before latin-1:** the two agree except on bytes 0x80 to 0x9F,
    which latin-1 maps to C1 control characters that real text does not
    use and cp1252 maps to punctuation such as the dash and the euro sign.
    The WHATWG Encoding Standard decodes text labelled ISO-8859-1 as
    windows-1252 for the same reason. latin-1 remains for the five bytes
    cp1252 leaves undefined; it decodes every byte, so step 5 never fails.

    **chardet** is a statistical guess, so it is used only when asked for:
    without ``use_chardet`` the result depends on the bytes alone, not on
    what is installed.

    Raises
    ------
    ImportError
        If ``use_chardet`` is true, chardet is needed and not installed.

    Examples
    --------
    >>> _detect_encoding(b"\xef\xbb\xbfHello")
    'utf-8-sig'
    >>> _detect_encoding(b"Hello world")
    'utf-8'
    >>> _detect_encoding(b"caf\xe9 \x97 \x805")  # cafe, em dash, euro sign
    'cp1252'
    >>> _detect_encoding(b"\x89PNG\r\n\x1a\n\x00\x00") is None
    True
    """
    # --- Stage 1: byte-order mark ---
    bom = _bom_encoding(raw)
    if bom is not None:
        return bom

    # --- Stage 2: binary (before UTF-8: NUL is valid UTF-8, U+0000) ---
    if b"\x00" in raw:
        return None

    # --- Stage 3: Strict UTF-8 ---
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        pass

    # --- Stage 4: chardet, only when asked for ---
    if use_chardet:
        try:
            import chardet  # type: ignore[] # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "TextReader(encoding='chardet') requires chardet. Install it"
                " with:\n  pip install chardet"
            ) from exc
        result = chardet.detect(raw[:_SNIFF_BYTES])
        confidence = result.get("confidence") or 0.0
        enc = result.get("encoding")
        if enc and confidence >= _CHARDET_MIN_CONFIDENCE and _decodes(raw, enc):
            logger.debug(
                "TextReader: chardet detected encoding %r (confidence=%.2f).",
                enc,
                confidence,
            )
            return enc

    # --- Stage 5: cp1252, then latin-1 (never fails) ---
    fallback = "cp1252" if _decodes(raw, "cp1252") else "latin-1"
    logger.warning(
        "TextReader: the file is not UTF-8 and its encoding is unknown;"
        " decoding as %r. Pass encoding=... (a codec, a list of candidate"
        " codecs, 'chardet', or a callable) if that is wrong.",
        fallback,
    )
    return fallback


def _bom_encoding(raw: bytes) -> str | None:
    """
    Return the codec a byte-order mark names, or ``None`` without one.

    The codecs read the byte order from the mark and drop it. UTF-32 is
    tested first: its little-endian mark begins with UTF-16's.
    """
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32"
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    return None


def _is_codec(name: Any) -> bool:
    """Return whether *name* is a ``str`` naming a codec Python knows."""
    if not isinstance(name, str):
        return False
    try:
        codecs.lookup(name)
    except LookupError:
        return False
    return True


def _decodes(raw: bytes, encoding: str) -> bool:
    """Return whether ``raw`` decodes strictly with the codec ``encoding``."""
    try:
        raw.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return False
    return True


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
    encoding : str, sequence of str, callable or None, optional
        The encoding policy:

        - ``None`` (default): detected from the whole file by fixed rules,
          the same on every machine: byte-order mark, binary check, UTF-8,
          cp1252, latin-1. See the Notes.
        - a codec name (``"utf-8"``, ``"shift_jis"``): the file is decoded
          with it, strictly; ``UnicodeDecodeError`` if it does not fit.
        - a list or tuple of codec names (``["utf-8", "gb18030",
          "cp1252"]``): the first that decodes the whole file. A
          byte-order mark comes first. ``ValueError`` if none does; end the
          list with ``"latin-1"`` to always succeed.
        - ``"chardet"``: as ``None``, with chardet's guess before the
          fallback (``pip install chardet``). The guess is statistical.
        - a callable ``policy(raw) -> codec`` given the file's bytes when
          they carry no byte-order mark, for example
          ``lambda raw: charset_normalizer.from_bytes(raw).best().encoding``.
          The codec it returns must decode the whole file.

        The codec used is recorded in each chunk's ``encoding`` metadata.
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
    ValueError
        If ``encoding`` is ``None`` and the file is binary (a NUL byte and
        no byte-order mark).
    UnicodeDecodeError
        If an explicit ``encoding`` is set and the file is not valid in it.
        Detected encodings always decode the file.

    See Also
    --------
    scikitplot.corpus._readers.XMLReader : XML/TEI document reader.
    scikitplot.corpus._readers.PDFReader : PDF document reader.

    Notes
    -----
    **Encoding detection** (``encoding=None``) inspects the whole file,
    read once: a byte-order mark (UTF-8, UTF-16, UTF-32; the mark is not
    kept in the text), then a NUL byte (binary: ``ValueError``), then
    strict UTF-8, then ``cp1252``, then ``latin-1``. The result depends on
    the bytes alone. For a corpus in a known legacy encoding, list it:
    ``encoding=["utf-8", "shift_jis"]``; to guess, ``encoding="chardet"``.

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

    # BUG-03 fix: registration now uses file_types (plural) so that a single
    # reader can cover multiple extensions.  file_type (singular) is retained
    # as a read-only backward-compat alias (first element of file_types).
    file_type: ClassVar[str] = ".txt"
    file_types: ClassVar[list[str] | None] = [".txt"]

    encoding: str | Sequence[str] | Callable[[bytes], str] | None = field(default=None)
    """
    How the file is decoded: ``None`` (detect, see the Notes), a codec
    name, a list of candidate codecs, ``"chardet"``, or a callable
    ``(raw: bytes) -> codec``. Default: ``None``.
    """

    max_file_bytes: int = field(default=_TEXT_DEFAULT_MAX_FILE_BYTES)
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
        policy = self.encoding
        if policy is None or callable(policy) or policy == "chardet":
            return
        if isinstance(policy, str):
            if not _is_codec(policy):
                raise ValueError(
                    f"TextReader: encoding {policy!r} is not a codec Python knows."
                )
            return
        if isinstance(policy, (list, tuple)) and policy:
            unknown = [name for name in policy if not _is_codec(name)]
            if unknown:
                raise ValueError(
                    f"TextReader: encoding candidates {unknown!r} are not codecs"
                    " Python knows."
                )
            return
        raise TypeError(
            "TextReader: encoding must be None, a codec name, a non-empty list"
            " or tuple of codec names, 'chardet', or a callable (raw) -> codec;"
            f" got {policy!r}."
        )

    def _choose_encoding(self, raw: bytes) -> str:  # noqa: PLR0911
        """
        Return the codec that decodes *raw* under :attr:`encoding`.

        Raises
        ------
        ValueError
            If the file is binary under ``None`` or ``"chardet"``, no
            candidate decodes it, or a callable names a codec that does not.
        TypeError
            If a callable returns something other than a codec name.
        ImportError
            If ``encoding="chardet"`` and chardet is not installed.
        """
        policy = self.encoding
        if isinstance(policy, str) and policy != "chardet":
            return policy
        if policy is None or policy == "chardet":
            detected = _detect_encoding(raw, use_chardet=policy == "chardet")
            if detected is None:
                offset = raw.index(b"\x00")
                raise ValueError(
                    f"TextReader: {self.file_name} is not a text file: it holds"
                    f" a NUL byte at offset {offset} and no byte-order mark. If"
                    " it is UTF-16 or UTF-32 without a byte-order mark, pass"
                    " encoding='utf-16-le' (or the matching codec)."
                )
            return detected
        bom = _bom_encoding(raw)
        if bom is not None:
            return bom
        if callable(policy):
            chosen = policy(raw)
            if not _is_codec(chosen):
                raise TypeError(
                    "TextReader: an encoding callable must return a codec name;"
                    f" got {chosen!r} for {self.file_name}."
                )
            if not _decodes(raw, chosen):
                raise ValueError(
                    f"TextReader: the encoding callable chose {chosen!r}, which"
                    f" does not decode {self.file_name}."
                )
            return chosen
        for candidate in policy:
            if _decodes(raw, candidate):
                return candidate
        raise ValueError(
            f"TextReader: none of the encoding candidates {list(policy)!r}"
            f" decodes {self.file_name}. Add 'latin-1' last to accept any bytes."
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
        ValueError
            If ``encoding`` is ``None`` and the file is binary.
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

        raw = self.input_path.read_bytes()
        # The whole file decides: a prefix can end inside a multi-byte
        # character or miss a later non-UTF-8 byte.
        enc = self._choose_encoding(raw)
        logger.debug("TextReader: decoding %s as %r.", self.file_name, enc)

        # Universal newlines, as ``Path.read_text`` gives: "\r\n" and "\r"
        # become "\n", so line endings do not depend on the platform that
        # wrote the file.
        text = raw.decode(enc).replace("\r\n", "\n").replace("\r", "\n")

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
            # raw_text: full file content as read from disk before any
            # downstream chunking or NLP — no pre-processing occurs in
            # TextReader, so raw_text equals text by definition.
            "raw_text": text,
            "section_type": SectionType.TEXT.value,
            # The codec used, so a decoding can be traced and repeated.
            "encoding": enc,
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

    # BUG-03 fix: registration now uses file_types (plural).
    # file_type (singular) retained as backward-compat alias.
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

    # BUG-03 fix: registration now uses file_types (plural).
    # file_type (singular) retained as backward-compat alias.
    file_type: ClassVar[str] = ".rst"
    file_types: ClassVar[list[str] | None] = [".rst"]
