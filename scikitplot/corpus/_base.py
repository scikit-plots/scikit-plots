# scikitplot/corpus/_base.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._base
======================
Abstract base classes for all scikitplot.corpus pipeline components.

Three abstract contracts are defined here, corresponding to the three
independent concerns of the pipeline:

:class:`DocumentReader`
    Reads a source file of a specific format and yields
    :class:`~scikitplot.corpus._schema.CorpusDocument` instances.
    Concrete subclasses handle individual formats (``TextReader``,
    ``XMLReader``, ``ALTOReader``, ``PDFReader``, etc.).
    A class-level registry maps file extensions to reader classes and
    is populated automatically by subclass definition — no manual
    registration required.

:class:`ChunkerBase`
    Transforms a block of raw text into a list of ``(char_start, text)``
    tuples. Concrete subclasses implement different granularities:
    sentence-level (spaCy), paragraph-level (blank-line split),
    fixed sliding window (configurable tokens/chars + overlap), etc.

:class:`FilterBase`
    Decides whether a :class:`~scikitplot.corpus._schema.CorpusDocument`
    should be included in the output corpus. The default implementation
    rejects documents below a minimum word count and those whose text
    contains no Unicode letter characters (punctuation/digit-only noise).

Design invariants
-----------------
* ``DocumentReader`` subclasses are ``dataclass``-decorated so that
  construction is type-safe and consistent with remarx's proven pattern.
* ``get_raw_chunks()`` is the **only** abstract method readers must
  implement. The concrete ``get_documents()`` method handles conversion
  from raw chunks to validated ``CorpusDocument`` instances, delegating
  chunking and filtering to collaborator objects passed at construction.
* Chunkers and filters are **stateless pure functions** wrapped in
  classes — same input always produces same output; no mutation of
  shared state.
* All failure modes raise ``ValueError`` with actionable messages.
  ``OSError`` and ``IOError`` from file I/O propagate uncaught so the
  caller can decide the recovery strategy.

Python compatibility
--------------------
Supports Python 3.8 through 3.15. Uses ``from __future__ import
annotations`` for PEP-604/585 annotation syntax. Uses ``TypeVar`` instead
of ``Self`` (Python 3.11+). Uses ``(str, Enum)`` mixin instead of
``StrEnum`` (Python 3.11+).
"""  # noqa: D205, D400

from __future__ import annotations

import abc
import logging
import pathlib
import re
import sys  # noqa: F401
import types
from dataclasses import dataclass, field
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from typing_extensions import Self  # noqa: F401

from ._schema import (
    _PROMOTED_RAW_KEYS,
    ChunkingStrategy,
    CorpusDocument,
    SectionType,
    SourceType,
)

# if TYPE_CHECKING:
#     pass  # reserved for future static-analysis-only imports

logger = logging.getLogger(__name__)

__all__ = [
    "ChunkerBase",
    # Concrete built-in filter
    "DefaultFilter",
    # Abstract bases
    "DocumentReader",
    # Utility readers
    "DummyReader",
    "FilterBase",
    # Pipeline resilience
    "PipelineGuard",
    # Multi-source adapter (context manager)
    "_MultiSourceReader",
    # URL detection helper
    "_is_url",
]

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Matches strings that have no Unicode letter — used by DefaultFilter
_NO_LETTER_RE: re.Pattern[str] = re.compile(r"^[^\w]*$", re.UNICODE)
# More precisely: "no word character that is also a letter"
# \w matches letters, digits, and underscore; we want letter-only check
_LETTER_RE: re.Pattern[str] = re.compile(r"[^\W\d_]", re.UNICODE)

# TypeVar for Self-like returns in classmethods (Python 3.8+ compatible)
_R = TypeVar("_R", bound="DocumentReader")  # noqa: PYI018
_C = TypeVar("_C", bound="ChunkerBase")  # noqa: PYI018
_F = TypeVar("_F", bound="FilterBase")  # noqa: PYI018


# ===========================================================================
# ChunkerBase — text segmentation contract
# ===========================================================================


# ---------------------------------------------------------------------------
# URL detection helper — used by DocumentReader.create() to auto-route
# string inputs that look like URLs without the caller needing from_url().
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://", re.IGNORECASE)


def _is_url(s: object) -> bool:
    """
    Return ``True`` if *s* is a string that looks like an HTTP(S) URL.

    Parameters
    ----------
    s : object
        Value to test.

    Returns
    -------
    bool
        ``True`` when *s* is a ``str`` matching ``^https?://``
        (case-insensitive).  ``False`` for ``pathlib.Path`` objects and
        non-string types.
    """
    return isinstance(s, str) and bool(_URL_RE.match(s))


class ChunkerBase(abc.ABC):
    """
    Abstract base class for all text chunkers.

    A chunker receives a block of raw text (one logical unit from the
    source document — a page, paragraph block, section, etc.) and returns
    a list of ``(char_start, chunk_text)`` tuples. The ``char_start``
    offset is relative to the beginning of the input text block, enabling
    downstream code to reconstruct absolute character positions.

    Parameters
    ----------
    None — subclasses define their own parameters.

    Attributes
    ----------
    strategy : ChunkingStrategy
        Class variable. Identifies which :class:`ChunkingStrategy` enum
        member this chunker implements. Must be defined by every concrete
        subclass.

    See Also
    --------
    scikitplot.corpus._chunkers.SentenceChunker : spaCy sentence segmentation.
    scikitplot.corpus._chunkers.ParagraphChunker : Blank-line paragraph split.
    scikitplot.corpus._chunkers.FixedWindowChunker : Sliding-window with overlap.

    Notes
    -----
    Chunkers must be **stateless** between ``chunk()`` calls. Any state
    required for a single call (e.g. a loaded language model) must be
    initialised inside ``chunk()`` or cached as an instance attribute that
    is never mutated after first assignment.

    Examples
    --------
    Implementing a trivial single-chunk chunker (no splitting):

    >>> class NullChunker(ChunkerBase):
    ...     strategy = ChunkingStrategy.NONE
    ...
    ...     def chunk(self, text, metadata=None):
    ...         return [(0, text)] if text.strip() else []
    """

    strategy: ClassVar[ChunkingStrategy]
    """
    Identifies which :class:`~scikitplot.corpus._schema.ChunkingStrategy`
    this implementation provides. **Must** be defined on every concrete subclass.
    """

    @abc.abstractmethod
    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, str]]:
        """
        Segment ``text`` into a list of ``(char_start, chunk_text)`` tuples.

        Parameters
        ----------
        text : str
            Raw text to segment. Must not be ``None``. Empty string input
            must return an empty list (never raise).
        metadata : dict or None, optional
            Chunk-level metadata from the reader (e.g. page number, section
            type). Made available so chunkers that need context — e.g. a
            semantic chunker deciding boundaries based on section label —
            can access it. Default: ``None``.

        Returns
        -------
        list of (int, str)
            Ordered list of ``(char_start, chunk_text)`` pairs.
            ``char_start`` is the character offset of ``chunk_text`` within
            the input ``text`` string. Must be non-negative and monotonically
            non-decreasing across the list.

        Raises
        ------
        ValueError
            If ``text`` is ``None`` (not just empty).

        Notes
        -----
        The return type is a list (not a generator) so callers can inspect
        length without consuming the iterator. For very large texts, chunkers
        should still return incrementally-built lists rather than loading
        everything into memory at once.
        """

    def __init_subclass__(cls, **kwargs: dict) -> None:
        """
        Enforce that every concrete subclass declares ``strategy``.
        """
        super().__init_subclass__(**kwargs)
        # Only enforce on concrete classes (those not still abstract)
        if not getattr(cls, "__abstractmethods__", None):  # noqa: SIM102
            if not hasattr(cls, "strategy"):
                raise TypeError(
                    f"Concrete ChunkerBase subclass {cls.__name__!r} must"
                    f" define a class-level 'strategy' attribute"
                    f" (ChunkingStrategy member)."
                )


# ===========================================================================
# FilterBase — document acceptance contract
# ===========================================================================


class FilterBase(abc.ABC):
    """
    Abstract base class for corpus document filters.

    A filter receives a fully-constructed :class:`~scikitplot.corpus._schema.CorpusDocument`
    and returns ``True`` if it should be included in the output corpus,
    ``False`` if it should be discarded.

    Filters are applied after chunking and before embedding, so they
    operate on already-segmented text. This is the correct place to
    discard noise tokens, very short fragments, duplicate content, etc.

    See Also
    --------
    scikitplot.corpus._filters.DefaultFilter : Standard word-count + letter filter.
    scikitplot.corpus._filters.CompositeFilter : Chain multiple filters with AND logic.
    scikitplot.corpus._filters.SectionFilter : Filter by SectionType membership.

    Notes
    -----
    Filters must be **side-effect free** — calling ``include()`` must not
    modify the document or any shared state.

    Examples
    --------
    Implementing a length filter:

    >>> class LengthFilter(FilterBase):
    ...     def __init__(self, min_chars: int = 10):
    ...         self.min_chars = min_chars
    ...
    ...     def include(self, doc):
    ...         return len(doc.text) >= self.min_chars
    """

    @abc.abstractmethod
    def include(self, doc: CorpusDocument) -> bool:
        """
        Return ``True`` if ``doc`` should be included in the corpus.

        Parameters
        ----------
        doc : CorpusDocument
            Document to evaluate. Must be a valid, validated instance.

        Returns
        -------
        bool
            ``True`` to include; ``False`` to discard.

        Notes
        -----
        Must never raise for a valid ``CorpusDocument``. Unexpected inputs
        should return ``False`` defensively rather than raising, unless the
        error indicates a programming error (e.g. ``None`` passed instead of
        a document).
        """


class DefaultFilter(FilterBase):
    r"""
    Standard noise filter ported and improved from remarx's ``include_sentence``.

    Rejects a document when **any** of the following is true:

    1. The text contains no Unicode letter characters (punctuation/digit-only).
    2. The whitespace-delimited token count is less than ``min_words``.
    3. The character count (after stripping) is less than ``min_chars``.

    Parameters
    ----------
    min_words : int, optional
        Minimum number of whitespace-delimited tokens. Default: ``3``.
    min_chars : int, optional
        Minimum number of non-whitespace characters. Default: ``10``.

    Notes
    -----
    The letter check uses ``re.compile(r'[^\\W\\d_]', re.UNICODE)`` which
    matches any Unicode letter (including accented and non-Latin characters)
    while excluding digits and underscore. This is more robust than
    remarx's original ``^[\\W\\d]+$`` which could pass on some Unicode inputs.

    Examples
    --------
    >>> f = DefaultFilter(min_words=3, min_chars=10)
    >>> doc_ok = CorpusDocument.create("f.txt", 0, "Hello world test.")
    >>> doc_noise = CorpusDocument.create("f.txt", 1, "p. 56, 57.")
    >>> f.include(doc_ok)
    True
    >>> f.include(doc_noise)
    False
    """

    def __init__(
        self,
        min_words: int = 3,
        min_chars: int = 10,
    ) -> None:
        """
        Filter.

        Parameters
        ----------
        min_words : int, optional
            Minimum whitespace-delimited tokens.  Default: 3.
        min_chars : int, optional
            Minimum character count.  Default: 10.
        """
        if min_words < 0:
            raise ValueError(f"DefaultFilter.min_words must be >= 0; got {min_words!r}")
        if min_chars < 0:
            raise ValueError(f"DefaultFilter.min_chars must be >= 0; got {min_chars!r}")
        self.min_words: int = min_words
        self.min_chars: int = min_chars

    def include(self, doc: CorpusDocument) -> bool:
        """
        Return ``True`` if ``doc`` passes all noise checks.

        Parameters
        ----------
        doc : CorpusDocument
            Document to evaluate.

        Returns
        -------
        bool
            ``True`` to include; ``False`` to discard.

        Notes
        -----
        Character count is measured on the stripped text to avoid counting
        surrounding whitespace as content.
        """
        text = doc.text
        stripped = text.strip()

        # Guard: no letter characters → noise (punctuation, digits only)
        if not _LETTER_RE.search(stripped):
            return False

        # Guard: too few whitespace tokens
        if len(stripped.split()) < self.min_words:
            return False

        # Guard: too few non-whitespace characters
        if len(stripped) < self.min_chars:  # noqa: SIM103
            return False

        return True

    def __repr__(self) -> str:  # noqa: D105
        return f"DefaultFilter(min_words={self.min_words}, min_chars={self.min_chars})"


# ===========================================================================
# DocumentReader — file ingestion contract
# ===========================================================================


@dataclass
class DocumentReader(abc.ABC):
    """
    Abstract base class for all format-specific document readers.

    A ``DocumentReader`` reads a single source file of a known format and
    yields a stream of :class:`~scikitplot.corpus._schema.CorpusDocument`
    instances. Subclasses implement :meth:`get_raw_chunks` to produce raw
    ``{text, ...metadata}`` dicts; the concrete :meth:`get_documents` method
    handles chunking, filtering, and schema construction.

    Parameters
    ----------
    input_file : pathlib.Path
        Absolute or relative path to the source file. Must exist and be
        readable when :meth:`get_documents` is called.
    chunker : ChunkerBase or None, optional
        Chunking strategy to apply to each raw text block. When ``None``,
        no sub-chunking is performed — each raw chunk becomes exactly one
        ``CorpusDocument``. Default: ``None``.
    filter_ : FilterBase or None, optional
        Filter to apply after chunking. When ``None``, the
        :class:`DefaultFilter` with its default parameters is used.
        Pass a :class:`FilterBase` subclass to override; pass a no-op
        filter (``lambda doc: True``) to disable filtering entirely.
        Default: ``None`` (uses ``DefaultFilter``).
    filename_override : str or None, optional
        Override the ``source_file`` field in generated documents.
        Useful when the reader receives a temporary file but should label
        documents with the original filename. Default: ``None``.
    default_language : str or None, optional
        ISO 639-1 language code to assign to documents when the reader
        cannot detect language from the source. Default: ``None``.

    Attributes
    ----------
    file_type : str
        Single file extension this reader handles (lowercase, including leading
        dot). E.g. ``".txt"``, ``".xml"``, ``".zip"``.

        For readers that handle multiple extensions, define ``file_types``
        (plural) instead.  **Exactly one** of ``file_type`` or ``file_types``
        must be defined on every concrete subclass.
    file_types : list of str
        List of file extensions this reader handles (lowercase, leading dot).
        Use instead of ``file_type`` when a single reader class should be
        registered for several extensions — e.g. an image reader for
        ``[".png", ".jpg", ".jpeg", ".gif", ".webp"]``.

        When both ``file_type`` and ``file_types`` are defined on the same
        class, ``file_types`` takes precedence and ``file_type`` is ignored.

    Raises
    ------
    ValueError
        If the input file does not exist (:meth:`validate_input`).
    ValueError
        If ``file_type`` is not defined on a concrete subclass.

    See Also
    --------
    scikitplot.corpus._readers.TextReader : Plain-text file reader.
    scikitplot.corpus._readers.XMLReader : TEI/generic XML reader.
    scikitplot.corpus._readers.ALTOReader : ALTO-XML-in-ZIP reader.
    scikitplot.corpus._readers.PDFReader : PDF reader (pdfminer / pypdf).

    Notes
    -----
    **Registry pattern:** Every concrete ``DocumentReader`` subclass is
    automatically registered by file extension via ``__init_subclass__``.
    The ``create()`` factory uses this registry to instantiate the correct
    reader for a given file. Subclasses do not need to call any registration
    function explicitly.

    **Chunker injection:** The reader receives its chunker at construction
    time, not as a global default. This means different reader instances can
    use different chunkers for the same file type — e.g. sentence chunking for
    body text and paragraph chunking for footnotes.

    **Raw chunk contract:** :meth:`get_raw_chunks` must yield dicts that
    always contain a ``"text"`` key. All other keys are treated as metadata
    and are merged into ``CorpusDocument.metadata``.

    Examples
    --------
    Creating a reader via the factory:

    >>> from pathlib import Path
    >>> reader = DocumentReader.create(Path("corpus.txt"))
    >>> for doc in reader.get_documents():
    ...     print(doc.doc_id, doc.word_count)

    Implementing a minimal reader:

    >>> from dataclasses import dataclass
    >>> from typing import Generator, Dict, Any
    >>> @dataclass
    ... class MyReader(DocumentReader):
    ...     file_type = ".myext"
    ...
    ...     def get_raw_chunks(self):
    ...         yield {"text": self.input_file.read_text("utf-8")}
    """

    # ------------------------------------------------------------------
    # Class-level registry — populated automatically by __init_subclass__
    # ------------------------------------------------------------------

    _registry: ClassVar[dict[str, type[DocumentReader]]] = {}
    """
    Internal mapping of ``file_extension → DocumentReader subclass``.
    Populated automatically when a concrete subclass is defined.
    Do not modify directly.
    """

    # ------------------------------------------------------------------
    # Required class variable — subclasses must define ONE of these
    # ------------------------------------------------------------------

    file_type: ClassVar[str | None]
    """
    Single file extension this reader handles (lowercase, including leading
    dot). E.g. ``".txt"``, ``".xml"``, ``".zip"``.

    For readers that handle multiple extensions, define ``file_types``
    (plural) instead.  **Exactly one** of ``file_type`` or ``file_types``
    must be defined on every concrete subclass.
    """

    file_types: ClassVar[list[str] | None]
    """
    List of file extensions this reader handles (lowercase, leading dot).
    Use instead of ``file_type`` when a single reader class should be
    registered for several extensions — e.g. an image reader for
    ``[".png", ".jpg", ".jpeg", ".gif", ".webp"]``.

    When both ``file_type`` and ``file_types`` are defined on the same
    class, ``file_types`` takes precedence and ``file_type`` is ignored.
    """

    # ------------------------------------------------------------------
    # Instance fields
    # ------------------------------------------------------------------

    input_file: pathlib.Path
    """
    Path to the source file.

    For URL-based readers (:class:`WebReader`, :class:`YouTubeReader`),
    pass ``pathlib.Path(url_string)`` here and set ``source_uri`` to the
    original URL string.  ``validate_input()`` is overridden in those
    subclasses to skip the file-existence check.
    """

    chunker: ChunkerBase | None = field(default=None)
    """
    Chunker to apply to each raw text block. ``None`` means each raw chunk
    is used as-is (one CorpusDocument per raw chunk).
    """

    filter_: FilterBase | None = field(default=None)
    """
    Filter applied after chunking. ``None`` triggers the :class:`DefaultFilter`.
    """

    filename_override: str | None = field(default=None)
    """Override for the ``source_file`` label in generated documents."""

    default_language: str | None = field(default=None)
    """ISO 639-1 language code to assign when the source has no language info."""

    source_uri: str | None = field(default=None)
    """
    Original URI for URL-based readers (web pages, YouTube videos).

    Set this to the full URL string when ``input_file`` is a synthetic
    ``pathlib.Path`` wrapping a URL.  File-based readers leave this
    ``None``.

    Examples
    --------
    >>> reader = WebReader(
    ...     input_file=Path("https://example.com/article"),
    ...     source_uri="https://example.com/article",
    ... )
    """

    source_provenance: dict[str, Any] = field(default_factory=dict)
    """
    Provenance overrides propagated into every yielded ``CorpusDocument``.

    Keys may include ``"source_type"``, ``"source_title"``,
    ``"source_author"``, and ``"collection_id"``.
    Populated by :meth:`create` / :meth:`from_url` from their keyword
    arguments.
    """

    # ------------------------------------------------------------------
    # Post-init: resolve defaults
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """
        Resolve ``filter_`` default after dataclass ``__init__``.

        Notes
        -----
        We cannot use a mutable :class:`DefaultFilter` as a field default
        directly (Python dataclass restriction). ``__post_init__`` is the
        canonical place to set mutable defaults.
        """
        if self.filter_ is None:
            object.__setattr__(self, "filter_", DefaultFilter())
        # Coerce input_file to pathlib.Path if a string was passed
        if not isinstance(self.input_file, pathlib.Path):
            object.__setattr__(self, "input_file", pathlib.Path(self.input_file))

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def file_name(self) -> str:
        """
        Effective filename used in document labels.

        Returns ``filename_override`` when set; otherwise returns
        ``input_file.name``.

        Returns
        -------
        str
            File name string (not a full path).

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = TextReader(input_file=Path("/data/corpus.txt"))
        >>> reader.file_name
        'corpus.txt'
        """
        return self.filename_override or self.input_file.name

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_input(self) -> None:
        """
        Assert that the input file exists and is readable.

        Raises
        ------
        ValueError
            If ``input_file`` does not exist or is not a regular file.

        Notes
        -----
        Called automatically by :meth:`get_documents` before iterating.
        Can also be called eagerly after construction to fail fast.

        Examples
        --------
        >>> reader = DocumentReader.create(Path("missing.txt"))
        >>> reader.validate_input()
        Traceback (most recent call last):
            ...
        ValueError: Input file does not exist: missing.txt
        """
        if not self.input_file.exists():
            raise ValueError(f"Input file does not exist: {self.input_file}")
        if not self.input_file.is_file():
            raise ValueError(f"Input path is not a regular file: {self.input_file}")

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Yield raw text chunks with associated metadata from the source file.

        Every yielded dict **must** contain a ``"text"`` key mapping to a
        non-empty string. All other keys are treated as metadata and merged
        into ``CorpusDocument.metadata``.

        Yields
        ------
        dict
            Mapping with at minimum ``{"text": str}`` plus any format-specific
            metadata fields (e.g. ``"page_number"``, ``"section_type"``,
            ``"author"``, ``"title"``).

        Raises
        ------
        ValueError
            For recoverable format errors (e.g. no valid pages found in
            a ZIP archive).
        OSError
            Propagated from file I/O on unrecoverable read errors.

        Notes
        -----
        Implementations should yield lazily (as a generator, not building
        a list) to keep memory usage proportional to chunk size, not file size.

        The ``"section_type"`` key, if present, must be a :class:`SectionType`
        value or a plain string coercible to one. The ``"text"`` value must
        be a plain string — no XML nodes or bytes.
        """

    # ------------------------------------------------------------------
    # Concrete pipeline method — builds CorpusDocuments from raw chunks
    # ------------------------------------------------------------------

    def get_documents(self) -> Generator[CorpusDocument, None, None]:
        """
        Yield validated :class:`~scikitplot.corpus._schema.CorpusDocument`
        instances for the input file.

        Orchestrates the full per-file pipeline:

        1. :meth:`validate_input` — fail fast if file is missing.
        2. :meth:`get_raw_chunks` — format-specific text extraction.
        3. Chunker (if set) — sub-segments each raw block.
        4. :class:`CorpusDocument` construction with validated schema.
        5. Filter — discards noise documents.

        Yields
        ------
        CorpusDocument
            Validated documents that passed the filter.

        Raises
        ------
        ValueError
            If the input file is missing or the format is invalid.

        Notes
        -----
        The global ``chunk_index`` counter is monotonically increasing across
        **all** raw chunks and sub-chunks for a single file, ensuring that
        ``(source_file, chunk_index)`` is a unique key within one reader run.

        Omitted-document statistics are logged at INFO level after processing
        each file.

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = DocumentReader.create(Path("corpus.txt"))
        >>> docs = list(reader.get_documents())
        >>> all(isinstance(d, CorpusDocument) for d in docs)
        True
        """  # noqa: D205
        self.validate_input()

        chunk_index: int = 0
        omitted: int = 0
        included: int = 0

        for raw_chunk in self.get_raw_chunks():
            raw_text: str = raw_chunk.get("text", "")

            # Determine chunking strategy label
            chunking_strategy = (
                self.chunker.strategy
                if self.chunker is not None
                else ChunkingStrategy.NONE
            )

            # Determine section type from raw chunk metadata
            raw_section = raw_chunk.get("section_type", SectionType.TEXT)
            try:
                section_type = SectionType(raw_section)
            except ValueError:
                logger.warning(
                    "%s: unknown section_type %r in raw chunk; defaulting to TEXT.",
                    self.file_name,
                    raw_section,
                )
                section_type = SectionType.TEXT

            # Split raw chunk keys: promoted → first-class fields,
            # rest → metadata. Keys "text" and "section_type" are consumed.
            promoted: dict[str, Any] = {}
            chunk_metadata: dict[str, Any] = {}
            for k, v in raw_chunk.items():
                if k in ("text", "section_type"):
                    continue
                if k in _PROMOTED_RAW_KEYS:
                    promoted[k] = v
                else:
                    chunk_metadata[k] = v

            # Merge source_provenance (reader-level) under chunk-level values
            provenance: dict[str, Any] = {**self.source_provenance, **promoted}

            # Sub-chunk the raw text (or use it as-is when no chunker)
            if self.chunker is not None and raw_text.strip():
                sub_chunks = self.chunker.chunk(raw_text, metadata=raw_chunk)
            else:
                sub_chunks = [(0, raw_text)] if raw_text.strip() else []

            for char_start, chunk_text in sub_chunks:
                if not chunk_text.strip():
                    omitted += 1
                    continue

                # Coerce source_type from string to enum if needed
                raw_st = provenance.get("source_type", SourceType.UNKNOWN)
                try:
                    resolved_source_type = (
                        raw_st if isinstance(raw_st, SourceType) else SourceType(raw_st)
                    )
                except ValueError:
                    logger.warning(
                        "%s: unknown source_type %r in raw chunk; using UNKNOWN.",
                        self.file_name,
                        raw_st,
                    )
                    resolved_source_type = SourceType.UNKNOWN

                doc = CorpusDocument.create(
                    source_file=self.file_name,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    section_type=section_type,
                    chunking_strategy=chunking_strategy,
                    language=self.default_language,
                    char_start=char_start,
                    char_end=char_start + len(chunk_text),
                    metadata=dict(chunk_metadata),
                    source_type=resolved_source_type,
                    source_title=provenance.get("source_title"),
                    source_author=provenance.get("source_author"),
                    source_date=provenance.get("source_date"),
                    collection_id=provenance.get("collection_id"),
                    url=provenance.get("url"),
                    doi=provenance.get("doi"),
                    isbn=provenance.get("isbn"),
                    page_number=provenance.get("page_number"),
                    paragraph_index=provenance.get("paragraph_index"),
                    line_number=provenance.get("line_number"),
                    parent_doc_id=provenance.get("parent_doc_id"),
                    act=provenance.get("act"),
                    scene_number=provenance.get("scene_number"),
                    timecode_start=provenance.get("timecode_start"),
                    timecode_end=provenance.get("timecode_end"),
                    confidence=provenance.get("confidence"),
                    ocr_engine=provenance.get("ocr_engine"),
                    bbox=provenance.get("bbox"),
                    normalized_text=provenance.get("normalized_text"),
                    tokens=provenance.get("tokens"),
                    lemmas=provenance.get("lemmas"),
                    stems=provenance.get("stems"),
                    keywords=provenance.get("keywords"),
                )
                chunk_index += 1

                # Apply filter
                assert (  # noqa: S101
                    self.filter_ is not None
                )  # always set in __post_init__  # noqa: S101
                if not self.filter_.include(doc):
                    omitted += 1
                    continue

                included += 1
                yield doc

        logger.info(
            "%s: yielded %d documents, omitted %d (filter/empty).",
            self.file_name,
            included,
            omitted,
        )
        # This stores the counts as instance attributes that the pipeline can read.
        self._last_n_included = included
        self._last_n_omitted = omitted

    # ------------------------------------------------------------------
    # Registry classmethods
    # ------------------------------------------------------------------

    @classmethod
    def supported_types(cls) -> list[str]:
        """
        Return a sorted list of file extensions supported by registered readers.

        Returns
        -------
        list of str
            Lowercase file extensions, each including the leading dot.
            E.g. ``['.pdf', '.txt', '.xml', '.zip']``.

        Examples
        --------
        >>> DocumentReader.supported_types()
        ['.pdf', '.txt', '.xml', '.zip']
        """
        return sorted(cls._registry.keys())

    @classmethod
    def subclass_by_type(cls) -> dict[str, type[DocumentReader]]:
        """
        Return a copy of the extension → reader class registry.

        Returns
        -------
        dict
            Mapping of file extension (str) → reader class. Returns a
            shallow copy so callers cannot accidentally mutate the registry.

        Examples
        --------
        >>> registry = DocumentReader.subclass_by_type()
        >>> ".txt" in registry
        True
        """
        return dict(cls._registry)

    # Downloadable URL extensions — used by from_url() stage-1 check.
    _DOWNLOADABLE_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {
            ".pdf",
            ".doc",
            ".docx",
            ".odt",
            ".rtf",
            ".csv",
            ".tsv",
            ".xlsx",
            ".xls",
            ".ods",
            ".txt",
            ".md",
            ".rst",
            ".xml",
            ".json",
            ".jsonl",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".webp",
            ".tiff",
            ".tif",
            ".bmp",
            ".svg",
            ".mp3",
            ".wav",
            ".flac",
            ".ogg",
            ".m4a",
            ".wma",
            ".aac",
            ".aiff",
            ".opus",
            ".mp4",
            ".avi",
            ".mkv",
            ".mov",
            ".webm",
            ".m4v",
            ".wmv",
            ".flv",
            ".zip",
            ".tar",
            ".gz",
            ".bz2",
            ".xz",
            ".7z",
            ".py",
            ".js",
            ".ts",
            ".java",
            ".c",
            ".cpp",
            ".go",
            ".rs",
        }
    )

    @classmethod
    def _build_prov(
        cls,
        *,
        source_type: SourceType | None = None,
        source_title: str | None = None,
        source_author: str | None = None,
        source_date: str | None = None,
        collection_id: str | None = None,
        doi: str | None = None,
        isbn: str | None = None,
    ) -> dict[str, Any]:
        """Build a clean source_provenance dict, excluding None values."""
        prov: dict[str, Any] = {}
        if source_type is not None:
            prov["source_type"] = source_type
        if source_title is not None:
            prov["source_title"] = source_title
        if source_author is not None:
            prov["source_author"] = source_author
        if source_date is not None:
            prov["source_date"] = source_date
        if collection_id is not None:
            prov["collection_id"] = collection_id
        if doi is not None:
            prov["doi"] = doi
        if isbn is not None:
            prov["isbn"] = isbn
        return prov

    @classmethod
    def create(
        cls,
        *inputs: pathlib.Path | str,
        chunker: ChunkerBase | None = None,
        filter_: FilterBase | None = None,
        filename_override: str | None = None,
        default_language: str | None = None,
        source_type: SourceType | list[SourceType | None] | None = None,
        source_title: str | None = None,
        source_author: str | None = None,
        source_date: str | None = None,
        collection_id: str | None = None,
        doi: str | None = None,
        isbn: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Instantiate the appropriate reader for one or more sources.

        Accepts any mix of file paths, URL strings, and
        :class:`pathlib.Path` objects — in any order.  URL strings (those
        starting with ``http://`` or ``https://``) are automatically
        detected and routed to :meth:`from_url`; everything else is treated
        as a local file path and dispatched by extension via the registry.

        Parameters
        ----------
        *inputs : pathlib.Path or str
            One or more source paths or URL strings.  Each element is
            classified independently:

            * ``str`` matching ``^https?://`` (case-insensitive) — treated
              as a URL and routed to :meth:`from_url`.  **Must be passed as
              a plain ``str``, not wrapped in** ``pathlib.Path``; wrapping
              collapses the double-slash (``https://`` → ``https:/``) and
              breaks URL detection.
            * ``str`` not matching the URL pattern — treated as a local
              file path and converted to :class:`pathlib.Path` internally.
            * :class:`pathlib.Path` — always treated as a local file path
              and dispatched by extension via the reader registry.

            Pass a single value for the common case; pass multiple values
            to get a :class:`_MultiSourceReader` that chains all their
            documents in order.
        chunker : ChunkerBase or None, optional
            Chunker injected into every reader. Default: ``None``.
        filter_ : FilterBase or None, optional
            Filter injected into every reader. Default: ``None``
            (:class:`DefaultFilter`).
        filename_override : str or None, optional
            Override the ``source_file`` label.  Only applied when
            *inputs* contains exactly one source.  Default: ``None``.
        default_language : str or None, optional
            ISO 639-1 language code applied to all sources.
            Default: ``None``.
        source_type : SourceType, list[SourceType or None], or None, optional
            Semantic label for the source kind.  When *inputs* has more
            than one element you may pass a list of the same length to
            assign a distinct type per source; ``None`` entries in the
            list mean "infer from extension / URL".  A single value is
            broadcast to all sources.  Default: ``None``.
        source_title : str or None, optional
            Title propagated into every yielded document. Default: ``None``.
        source_author : str or None, optional
            Author propagated into every yielded document. Default: ``None``.
        source_date : str or None, optional
            ISO 8601 publication date. Default: ``None``.
        collection_id : str or None, optional
            Corpus collection identifier. Default: ``None``.
        doi : str or None, optional
            Digital Object Identifier (file sources only). Default: ``None``.
        isbn : str or None, optional
            ISBN (file sources only). Default: ``None``.
        **kwargs : Any
            Extra keyword arguments forwarded verbatim to each concrete
            reader constructor (e.g. ``transcribe=True`` for
            :class:`AudioReader`, ``backend="easyocr"`` for
            :class:`ImageReader`).

        Returns
        -------
        DocumentReader
            A single reader when *inputs* has exactly one element (backward
            compatible with every existing call site).  A
            :class:`_MultiSourceReader` when *inputs* has more than one
            element — it implements the same ``get_documents()`` interface
            and chains documents from all sub-readers in order.

        Raises
        ------
        ValueError
            If *inputs* is empty, or if a source URL is invalid, or if no
            reader is registered for a file's extension.
        TypeError
            If any element of *inputs* is not a ``str`` or
            :class:`pathlib.Path`.

        Notes
        -----
        **URL auto-detection:** A ``str`` element is treated as a URL when
        it matches ``^https?://`` (case-insensitive).  All other strings
        and all :class:`pathlib.Path` objects are treated as local file
        paths.  This means you no longer need to call :meth:`from_url`
        explicitly — just pass the URL string to :meth:`create`.

        **Per-source source_type:** When passing multiple inputs with
        different media types, supply a list::

            DocumentReader.create(
                Path("podcast.mp3"),
                "report.pdf",
                "https://iris.who.int/.../content",  # returns image/jpeg
                source_type=[SourceType.PODCAST, SourceType.RESEARCH, SourceType.IMAGE],
            )

        **Reader-specific kwargs** (forwarded via ``**kwargs``):

        - ``transcribe=True``, ``whisper_model="small"`` → :class:`AudioReader`,
          :class:`VideoReader`
        - ``backend="easyocr"`` → :class:`ImageReader`
        - ``prefer_backend="pypdf"`` → :class:`PDFReader`
        - ``classify=True``, ``classifier=fn`` → :class:`AudioReader`

        Examples
        --------
        Single file (backward-compatible):

        >>> reader = DocumentReader.create(Path("hamlet.txt"))
        >>> docs = list(reader.get_documents())

        URL string auto-detected — no from_url() call required:

        >>> reader = DocumentReader.create(
        ...     "https://en.wikipedia.org/wiki/Python_(programming_language)"
        ... )

        Mixed multi-source batch:

        >>> reader = DocumentReader.create(
        ...     Path("podcast.mp3"),
        ...     "report.pdf",
        ...     "https://iris.who.int/api/bitstreams/abc/content",
        ...     source_type=[SourceType.PODCAST, SourceType.RESEARCH, SourceType.IMAGE],
        ... )
        >>> docs = list(reader.get_documents())  # chained stream from all three
        """
        if not inputs:
            raise ValueError("DocumentReader.create: at least one source is required.")

        # Validate and normalise source_type to a list aligned with inputs.
        n = len(inputs)
        if isinstance(source_type, list):
            if len(source_type) != n:
                raise ValueError(
                    f"DocumentReader.create: source_type list length "
                    f"({len(source_type)}) must match the number of inputs ({n})."
                )
            type_list: list[Any] = list(source_type)
        else:
            type_list = [source_type] * n

        # Build one sub-reader per input.
        readers: list[DocumentReader] = []
        for idx, raw in enumerate(inputs):
            if not isinstance(raw, (str, pathlib.Path)):
                raise TypeError(
                    f"DocumentReader.create: inputs[{idx}] must be str or "
                    f"pathlib.Path; got {type(raw).__name__!r}."
                )
            st = type_list[idx]
            if _is_url(raw):
                # URL string — delegate to from_url()
                readers.append(
                    cls.from_url(
                        str(raw),
                        chunker=chunker,
                        filter_=filter_,
                        default_language=default_language,
                        source_type=st,
                        source_title=source_title,
                        source_author=source_author,
                        source_date=source_date,
                        collection_id=collection_id,
                        doi=doi,
                        isbn=isbn,
                        **kwargs,
                    )
                )
            else:
                # Local file path — extension-based dispatch.
                readers.append(
                    cls._create_one(
                        raw,
                        chunker=chunker,
                        filter_=filter_,
                        filename_override=filename_override if n == 1 else None,
                        default_language=default_language,
                        source_type=st,
                        source_title=source_title,
                        source_author=source_author,
                        source_date=source_date,
                        collection_id=collection_id,
                        doi=doi,
                        isbn=isbn,
                        **kwargs,
                    )
                )

        if len(readers) == 1:
            return readers[0]
        return _MultiSourceReader(readers)

    @classmethod
    def _create_one(
        cls,
        input_file: pathlib.Path | str,
        *,
        chunker: ChunkerBase | None = None,
        filter_: FilterBase | None = None,
        filename_override: str | None = None,
        default_language: str | None = None,
        source_type: SourceType | None = None,
        source_title: str | None = None,
        source_author: str | None = None,
        source_date: str | None = None,
        collection_id: str | None = None,
        doi: str | None = None,
        isbn: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Instantiate and return the reader for a single local file.

        Internal helper called by :meth:`create`.  Dispatches by extension
        via the class registry.

        Parameters
        ----------
        input_file : pathlib.Path or str
            Path to the source file.
        chunker : ChunkerBase or None, optional
            Chunker to inject. Default: ``None``.
        filter_ : FilterBase or None, optional
            Filter to inject. Default: ``None`` (:class:`DefaultFilter`).
        filename_override : str or None, optional
            Override for the ``source_file`` label. Default: ``None``.
        default_language : str or None, optional
            ISO 639-1 language code. Default: ``None``.
        source_type : SourceType or None, optional
            Semantic source label. Default: ``None``.
        source_title : str or None, optional
            Title propagated into documents. Default: ``None``.
        source_author : str or None, optional
            Author propagated into documents. Default: ``None``.
        source_date : str or None, optional
            ISO 8601 date. Default: ``None``.
        collection_id : str or None, optional
            Collection identifier. Default: ``None``.
        doi : str or None, optional
            Digital Object Identifier. Default: ``None``.
        isbn : str or None, optional
            ISBN. Default: ``None``.
        **kwargs : Any
            Forwarded to the reader constructor.

        Returns
        -------
        DocumentReader
            Concrete reader for the file extension.

        Raises
        ------
        ValueError
            If no reader is registered for the extension.

        Developer note
        --------------
        This method replaces the previous body of :meth:`create` and is
        kept separate so :meth:`create` can call it in a loop for the
        multi-source case without duplicating the dispatch logic.
        """
        path = pathlib.Path(input_file)
        ext = path.suffix.lower()
        reader_cls = cls._registry.get(ext)
        if reader_cls is None:
            supported = ", ".join(cls.supported_types()) or "(none registered)"
            raise ValueError(
                f"No DocumentReader registered for extension {ext!r}."
                f" Supported types: {supported}."
                f" To add support for {ext!r}, either:\n"
                f"  1. Use CustomReader directly:\n"
                f"       reader = CustomReader(input_file=path, extractor=my_fn)\n"
                f"  2. Register globally via CustomReader.register():\n"
                f"       CustomReader.register(name='MyReader',"
                f" extensions=[{ext!r}], extractor=my_fn)\n"
                f"  3. Subclass DocumentReader with file_type = {ext!r}."
            )
        prov = cls._build_prov(
            source_type=source_type,
            source_title=source_title,
            source_author=source_author,
            source_date=source_date,
            collection_id=collection_id,
            doi=doi,
            isbn=isbn,
        )
        return reader_cls(
            input_file=path,
            chunker=chunker,
            filter_=filter_,
            filename_override=filename_override,
            default_language=default_language,
            source_provenance=prov,
            **kwargs,
        )

    @classmethod
    def from_manifest(  # noqa: PLR0912
        cls,
        manifest_path: pathlib.Path | str,
        *,
        chunker: ChunkerBase | None = None,
        filter_: FilterBase | None = None,
        default_language: str | None = None,
        source_type: SourceType | None = None,
        source_title: str | None = None,
        source_author: str | None = None,
        source_date: str | None = None,
        collection_id: str | None = None,
        doi: str | None = None,
        isbn: str | None = None,
        encoding: str = "utf-8",
        **kwargs: Any,
    ) -> _MultiSourceReader:
        """
        Build a :class:`_MultiSourceReader` from a manifest file.

        The manifest is a text file with one source per line — either a
        file path or a URL.  Blank lines and lines starting with ``#``
        are ignored.  JSON manifests (a list of strings or objects) are
        also supported.

        Parameters
        ----------
        manifest_path : pathlib.Path or str
            Path to the manifest file.  Supported formats:

            - ``.txt`` / ``.manifest`` — one source per line.
            - ``.json`` — a JSON array of strings (sources) or objects
              with at least a ``"source"`` key (and optional
              ``"source_type"``, ``"source_title"`` per-entry overrides).
        chunker : ChunkerBase or None, optional
            Chunker applied to all sources. Default: ``None``.
        filter_ : FilterBase or None, optional
            Filter applied to all sources. Default: ``None``.
        default_language : str or None, optional
            ISO 639-1 language code. Default: ``None``.
        source_type : SourceType or None, optional
            Override source type for all sources. Default: ``None``.
        source_title : str or None, optional
            Override title for all sources. Default: ``None``.
        source_author : str or None, optional
            Override author for all sources. Default: ``None``.
        source_date : str or None, optional
            Override date for all sources. Default: ``None``.
        collection_id : str or None, optional
            Collection identifier. Default: ``None``.
        doi : str or None, optional
            DOI override. Default: ``None``.
        isbn : str or None, optional
            ISBN override. Default: ``None``.
        encoding : str, optional
            Text encoding for ``.txt`` manifests. Default: ``"utf-8"``.
        **kwargs : Any
            Forwarded to each reader constructor.

        Returns
        -------
        _MultiSourceReader
            Multi-source reader chaining all manifest entries.

        Raises
        ------
        ValueError
            If *manifest_path* does not exist or is empty after filtering
            blank and comment lines.
        ValueError
            If the manifest format is not recognised.

        Notes
        -----
        Per-entry overrides in JSON manifests: each entry may be an
        object with::

            {
                "source": "https://example.com/report.pdf",
                "source_type": "research",
                "source_title": "Annual Report 2024",
            }

        String-level ``source_type`` values are coerced via
        ``SourceType(value)`` and an invalid value raises ``ValueError``.

        Examples
        --------
        Text manifest ``sources.txt``::

            # WHO corpus
            https://www.who.int/europe/news/item/...
            https://youtu.be/rwPISgZcYIk
            WHO-EURO-2025.pdf
            scan.jpg

        Usage::

            reader = DocumentReader.from_manifest(
                Path("sources.txt"),
                collection_id="who-corpus",
            )
            docs = list(reader.get_documents())
        """
        import json as _json  # noqa: PLC0415

        mp = pathlib.Path(manifest_path)
        if not mp.exists():
            raise ValueError(
                f"DocumentReader.from_manifest: manifest not found: {mp!r}"
            )

        entries: list[tuple[str, dict[str, Any]]] = []  # (source, overrides)

        if mp.suffix.lower() == ".json":
            with mp.open(encoding=encoding) as fh:
                data = _json.load(fh)
            if not isinstance(data, list):
                raise ValueError(
                    f"DocumentReader.from_manifest: JSON manifest must be a "
                    f"list; got {type(data).__name__!r}."
                )
            for item in data:
                if isinstance(item, str):
                    entries.append((item, {}))
                elif isinstance(item, dict):
                    source = item.get("source")
                    if not source:
                        raise ValueError(
                            f"DocumentReader.from_manifest: JSON object entry "
                            f"missing 'source' key: {item!r}"
                        )
                    overrides = {k: v for k, v in item.items() if k != "source"}
                    entries.append((source, overrides))
                else:
                    raise TypeError(
                        f"DocumentReader.from_manifest: JSON entries must be "
                        f"str or dict; got {type(item).__name__!r}: {item!r}"
                    )
        else:
            # Plain text: one source per line, # comments, blank lines skipped
            with mp.open(encoding=encoding) as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        entries.append((stripped, {}))

        if not entries:
            raise ValueError(
                f"DocumentReader.from_manifest: manifest {mp!r} is empty "
                f"(no sources after filtering blank lines and comments)."
            )

        # Build one reader per entry
        readers: list[DocumentReader] = []
        for source_str, overrides in entries:
            # Per-entry overrides may specify source_type as a string
            entry_st = source_type
            if "source_type" in overrides:
                raw_st = overrides.pop("source_type")
                try:
                    entry_st = SourceType(raw_st) if isinstance(raw_st, str) else raw_st
                except ValueError as exc:
                    raise ValueError(
                        f"DocumentReader.from_manifest: invalid source_type "
                        f"{raw_st!r} in manifest entry {source_str!r}."
                    ) from exc
            entry_title = overrides.pop("source_title", source_title)
            entry_author = overrides.pop("source_author", source_author)
            readers.append(
                cls.create(
                    source_str,
                    chunker=chunker,
                    filter_=filter_,
                    default_language=default_language,
                    source_type=entry_st,
                    source_title=entry_title,
                    source_author=entry_author,
                    source_date=source_date,
                    collection_id=collection_id,
                    doi=doi,
                    isbn=isbn,
                    **{**kwargs, **overrides},
                )
            )

        # Flatten single-reader cases
        flat: list[DocumentReader] = []
        for r in readers:
            if isinstance(r, _MultiSourceReader):
                flat.extend(r.readers)
            else:
                flat.append(r)
        return _MultiSourceReader(flat)

    @classmethod
    def from_url(  # noqa: PLR0912
        cls,
        url: str,
        *,
        chunker: ChunkerBase | None = None,
        filter_: FilterBase | None = None,
        filename_override: str | None = None,
        default_language: str | None = None,
        source_type: SourceType | None = None,
        source_title: str | None = None,
        source_author: str | None = None,
        source_date: str | None = None,
        collection_id: str | None = None,
        doi: str | None = None,
        isbn: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Instantiate the appropriate reader for a URL source.

        Dispatches to :class:`~scikitplot.corpus._readers.YouTubeReader`
        for YouTube URLs and to
        :class:`~scikitplot.corpus._readers.WebReader` for all other
        ``http://`` / ``https://`` URLs.

        Parameters
        ----------
        url : str
            Full URL string. Must start with ``http://`` or ``https://``.
        chunker : ChunkerBase or None, optional
            Chunker to inject. Default: ``None``.
        filter_ : FilterBase or None, optional
            Filter to inject. Default: ``None`` (:class:`DefaultFilter`).
        filename_override : str or None, optional
            Override for the ``source_file`` label. Default: ``None``.
        default_language : str or None, optional
            ISO 639-1 language code. Default: ``None``.
        source_type : SourceType or None, optional
            Semantic label for the source. Default: ``None``.
        source_title : str or None, optional
            Title of the source work. Default: ``None``.
        source_author : str or None, optional
            Primary author. Default: ``None``.
        source_date : str or None, optional
            Publication date in ISO 8601 format. Default: ``None``.
        collection_id : str or None, optional
            Corpus collection identifier. Default: ``None``.
        doi : str or None, optional
            Digital Object Identifier. Default: ``None``.
        isbn : str or None, optional
            International Standard Book Number. Default: ``None``.
        **kwargs : Any
            Additional kwargs forwarded to the reader constructor (e.g.
            ``include_auto_generated=False`` for :class:`YouTubeReader`).

        Returns
        -------
        DocumentReader
            :class:`~scikitplot.corpus._readers.YouTubeReader` or
            :class:`~scikitplot.corpus._readers.WebReader` instance.

        Raises
        ------
        ValueError
            If ``url`` does not start with ``http://`` or ``https://``.
        ImportError
            If the required reader class is not registered (i.e.
            ``scikitplot.corpus._readers`` has not been imported yet).

        Notes
        -----
        **Prefer :meth:`create` for new code.** Passing a URL string to
        :meth:`create` automatically calls :meth:`from_url` — you rarely
        need to call :meth:`from_url` directly.

        Examples
        --------
        >>> reader = DocumentReader.from_url("https://en.wikipedia.org/wiki/Python")
        >>> docs = list(reader.get_documents())

        >>> yt = DocumentReader.from_url("https://www.youtube.com/watch?v=rwPISgZcYIk")
        >>> docs = list(yt.get_documents())
        """
        import os as _os  # noqa: PLC0415
        import re as _re  # noqa: PLC0415

        if not isinstance(url, str) or not _re.match(r"https?://", url):
            raise ValueError(
                f"DocumentReader.from_url: url must start with 'http://' or"
                f" 'https://'; got {url!r}."
            )

        # ── Stage 1: extension-based classification (no network) ─────────
        # YouTube detection (covers standard, short, embed, shorts, live)
        _YT_RE = _re.compile(  # noqa: N806
            r"https?://(www\.)?(youtube\.com/(watch|shorts|embed|live)|youtu\.be/)",
            _re.IGNORECASE,
        )
        if _YT_RE.match(url):
            reader_key = ":youtube"
            reader_cls = cls._registry.get(reader_key)
            if reader_cls is None:
                raise ImportError(
                    "No reader registered for YouTube URLs. "
                    "Import scikitplot.corpus._readers to register YouTubeReader:"
                    "\n  import scikitplot.corpus._readers"
                )
            prov: dict[str, Any] = cls._build_prov(
                source_type=source_type,
                source_title=source_title,
                source_author=source_author,
                source_date=source_date,
                collection_id=collection_id,
                doi=doi,
                isbn=isbn,
            )
            return reader_cls(
                input_file=pathlib.Path(url),
                source_uri=url,
                chunker=chunker,
                filter_=filter_,
                filename_override=filename_override,
                default_language=default_language,
                source_provenance=prov,
                **kwargs,
            )

        # Check URL path for known downloadable extension
        from urllib.parse import urlparse as _up  # noqa: PLC0415

        path_ext = _os.path.splitext(_up(url).path)[1].lower()
        has_ext = bool(path_ext) and path_ext in cls._DOWNLOADABLE_EXTENSIONS

        # kind is resolved once here and reused in the downloadable block below
        # so that the probe result (from Stage 2) is never discarded.
        _url_kind: Any = None  # populated by Stage 2 when has_ext is False

        # ── Stage 2: probe extensionless URLs with HEAD request ───────────
        if not has_ext:
            try:
                from ._url_handler import (  # noqa: PLC0415
                    URLKind,
                    classify_url,
                    probe_url_kind,
                )

                _url_kind = classify_url(url)
                if _url_kind == URLKind.WEB_PAGE:
                    probed = probe_url_kind(url)
                    if probed == URLKind.DOWNLOADABLE:
                        _url_kind = probed
            except Exception:  # noqa: BLE001 — fail safe: treat as web page
                _url_kind = None
                has_ext = False
            else:
                if _url_kind == URLKind.DOWNLOADABLE:
                    has_ext = True  # treat as downloadable below

        # ── Downloadable (has extension OR probed as DOWNLOADABLE) ────────
        if has_ext:
            import tempfile  # noqa: PLC0415

            from ._url_handler import (  # noqa: PLC0415
                URLKind,
                classify_url,
                download_url,
                resolve_url,
            )

            # Reuse the kind already computed in Stage 2 when available;
            # only call classify_url() for URLs that had a known extension
            # in Stage 1 (those bypassed Stage 2 entirely).
            kind = _url_kind if _url_kind is not None else classify_url(url)
            resolved = resolve_url(url, kind=kind)
            tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="skplt_furl_"))
            try:
                local_path = download_url(resolved, dest_dir=tmp_dir)
                # Infer source_type from downloaded file's extension if not given
                st = source_type
                if st is None:
                    try:
                        from ._schema import (  # noqa: PLC0415
                            SourceType as _ST,  # noqa: N814
                        )

                        st = _ST.infer(local_path)
                    except Exception:  # noqa: BLE001
                        pass
                reader = cls._create_one(
                    local_path,
                    chunker=chunker,
                    filter_=filter_,
                    filename_override=filename_override or url,
                    default_language=default_language,
                    source_type=st,
                    source_title=source_title,
                    source_author=source_author,
                    source_date=source_date,
                    collection_id=collection_id,
                    doi=doi,
                    isbn=isbn,
                    **kwargs,
                )
                # Attach temp_dir for cleanup; caller or gc will remove it
                reader._from_url_tmp_dir = tmp_dir  # type: ignore[attr-defined]
                logger.info(
                    "DocumentReader.from_url: downloaded %s → %s",
                    url,
                    local_path,
                )
                return reader
            except Exception:
                import shutil  # noqa: PLC0415

                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise

        # ── Web page (default) ────────────────────────────────────────────
        reader_cls = cls._registry.get(":url")
        if reader_cls is None:
            raise ImportError(
                "No reader registered for web URLs. "
                "Import scikitplot.corpus._readers to register WebReader:"
                "\n  import scikitplot.corpus._readers"
            )
        prov2: dict[str, Any] = cls._build_prov(
            source_type=source_type,
            source_title=source_title,
            source_author=source_author,
            source_date=source_date,
            collection_id=collection_id,
            doi=doi,
            isbn=isbn,
        )
        return reader_cls(
            input_file=pathlib.Path(url),
            source_uri=url,
            chunker=chunker,
            filter_=filter_,
            filename_override=filename_override,
            default_language=default_language,
            source_provenance=prov2,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Registry auto-population via __init_subclass__
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: dict) -> None:
        """
        Auto-register concrete subclasses in the extension registry.

        Called by Python whenever a class inherits from ``DocumentReader``.
        Inspects both ``file_type`` (``str``) and ``file_types``
        (``list[str]``) class variables. If neither is defined the class
        is treated as abstract and not registered.

        ``file_types`` takes precedence when both are present.

        Notes
        -----
        **Duplicate extension registrations** are logged as warnings. The
        last registered class for a given extension wins, which allows user
        code to override built-in readers.

        **Special keys** such as ``":url"`` and ``":youtube"`` are reserved
        for URL-dispatched readers and must start with ``":"`` rather than
        ``"."``.
        """
        super().__init_subclass__(**kwargs)

        # Resolve the list of extensions/keys to register for this class
        raw_file_types = getattr(cls, "file_types", None)
        raw_file_type = getattr(cls, "file_type", None)

        if raw_file_types is not None:
            # Multi-extension form takes precedence
            if not isinstance(raw_file_types, (list, tuple)):
                raise TypeError(
                    f"DocumentReader subclass {cls.__name__!r}: file_types"
                    f" must be a list of strings; got {raw_file_types!r}."
                )
            extensions = [str(ft).lower() for ft in raw_file_types]
        elif raw_file_type is not None:
            extensions = [str(raw_file_type).lower()]
        else:
            # Abstract base — no registration
            return

        for ext in extensions:
            # Allow "." prefix (file extensions) or ":" prefix (URL schemes)
            if not (ext.startswith((".", ":"))):
                raise TypeError(
                    f"DocumentReader subclass {cls.__name__!r}: extension"
                    f" {ext!r} must start with '.' (file) or ':' (URL scheme)."
                )
            if ext in DocumentReader._registry:
                existing = DocumentReader._registry[ext].__name__
                logger.warning(
                    "DocumentReader: extension %r already registered to %s;"
                    " overriding with %s.",
                    ext,
                    existing,
                    cls.__name__,
                )
            DocumentReader._registry[ext] = cls
            logger.debug(
                "DocumentReader: registered %s for extension %r.",
                cls.__name__,
                ext,
            )


# ===========================================================================
# _MultiSourceReader — chains documents from multiple sub-readers
# ===========================================================================


class _MultiSourceReader:
    """
    Chains multiple :class:`DocumentReader` instances into one stream.

    Returned by :meth:`DocumentReader.create` when more than one source
    is supplied, and by :meth:`DocumentReader.from_manifest`.

    Also acts as a context manager ensuring temporary directories from
    ``from_url()`` downloads are cleaned up on exit.

    Parameters
    ----------
    readers : list[DocumentReader]
        Ordered list of sub-readers. Documents are yielded in order.

    Notes
    -----
    **Context manager usage** (ensures temp-file cleanup)::

        with DocumentReader.create(
            "https://iris.who.int/.../content",
            Path("report.pdf"),
        ) as reader:
            docs = list(reader.get_documents())

    **Duck-typed interface** — exposes ``get_documents()`` matching
    :class:`DocumentReader` so it works anywhere a single reader is
    accepted.

    Examples
    --------
    >>> from pathlib import Path
    >>> import scikitplot.corpus._readers
    >>> reader = DocumentReader.create(Path("a.txt"), Path("b.pdf"))
    >>> type(reader).__name__
    '_MultiSourceReader'
    >>> docs = list(reader.get_documents())
    """

    def __init__(self, readers: list[DocumentReader]) -> None:
        """Build a :class:`_MultiSourceReader` from a list of sub-readers.

        Parameters
        ----------
        readers : list[DocumentReader]
            Ordered list of sub-readers.  Must be non-empty.

        Raises
        ------
        ValueError
            If *readers* is empty.
        """
        if not readers:
            raise ValueError("_MultiSourceReader: readers list must not be empty.")
        self.readers: list[DocumentReader] = readers

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def get_documents(self) -> Generator[Any, None, None]:
        """Yield all documents from all sub-readers in order.

        Yields
        ------
        CorpusDocument
            Documents from each sub-reader, chained sequentially.
        """
        for reader in self.readers:
            yield from reader.get_documents()

    # ------------------------------------------------------------------
    # Context manager — cleans up temp dirs from URL downloads
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns
        -------
        _MultiSourceReader
            Returns ``self`` so the ``with`` target is usable as a reader.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the context manager and delete temporary directories.

        Parameters
        ----------
        exc_type, exc_val, exc_tb
            Standard exception info — unused; cleanup runs unconditionally.

        Notes
        -----
        Removes any ``_from_url_tmp_dir`` directories attached to
        sub-readers by :meth:`~DocumentReader.from_url` after downloading
        remote files.  Safe to call even if ``__enter__`` was never called.
        """
        self.close()

    def close(self) -> None:
        """Release temporary directories created by ``from_url()`` downloads.

        Each sub-reader that downloaded a file has a ``_from_url_tmp_dir``
        attribute set by :meth:`DocumentReader.from_url`.  This method
        deletes those directories.  Called automatically when used as a
        context manager; call manually otherwise.
        """
        import shutil as _shutil  # noqa: PLC0415

        for reader in self.readers:
            tmp = getattr(reader, "_from_url_tmp_dir", None)
            if tmp is not None:
                _shutil.rmtree(tmp, ignore_errors=True)
                reader._from_url_tmp_dir = None  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def n_readers(self) -> int:
        """Number of constituent readers.

        Returns
        -------
        int
            ``len(self.readers)``.
        """
        return len(self.readers)

    def __repr__(self) -> str:
        """Return ``_MultiSourceReader([...], n=N)``."""
        sources = ", ".join(
            getattr(r, "file_name", type(r).__name__) for r in self.readers
        )
        return f"_MultiSourceReader([{sources}], n={len(self.readers)})"


# ===========================================================================
# DummyReader — existence / accessibility check, yields zero documents
# ===========================================================================


class DummyReader(DocumentReader):
    """
    A no-op reader that validates source existence and accessibility.

    Confirms a source can be reached, then yields **zero documents**.
    Use for pre-flight validation before committing to a full ingest
    pipeline.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the file to check, or a synthetic path wrapping a URL.
    source_uri : str or None, optional
        When set, performs an HTTP HEAD request against this URI instead
        of a filesystem existence check.  Default: ``None``.

    Notes
    -----
    **Registration:** Registered under the special key ``":dummy"`` and
    never dispatched automatically by extension.

    **Pipeline pre-flight pattern**::

        ok, errors = DummyReader.check(
            Path("report.pdf"),
            "https://iris.who.int/.../content",
            "https://youtu.be/rwPISgZcYIk",
        )
        for src, exc in errors:
            print(f"UNREACHABLE: {src} — {exc}")

    Examples
    --------
    >>> from pathlib import Path
    >>> import tempfile, os
    >>> with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
    ...     _ = f.write(b"hello")
    ...     tmp = f.name
    >>> reader = DummyReader(input_file=Path(tmp))
    >>> list(reader.get_documents())
    []
    >>> os.unlink(tmp)
    """

    # file_type intentionally not set here — file_types takes precedence
    # per the __init_subclass__ contract ("file_types wins when both present").
    # Defining both caused a silent inconsistency with the documented invariant.
    file_types: ClassVar[list[str] | None] = [":dummy"]

    # ------------------------------------------------------------------
    # Batch class-level check — the primary use-case API
    # ------------------------------------------------------------------

    @classmethod
    def check(
        cls,
        *sources: pathlib.Path | str,
        timeout: int = 10,
        raise_on_first: bool = False,
    ) -> tuple[list[str | pathlib.Path], list[tuple[str | pathlib.Path, Exception]]]:
        """
        Check accessibility of one or more sources without ingesting them.

        Parameters
        ----------
        *sources : pathlib.Path or str
            File paths or URL strings to check.  Accepts the same inputs
            as :meth:`DocumentReader.create`.
        timeout : int, optional
            HTTP timeout in seconds for URL checks. Default: 10.
        raise_on_first : bool, optional
            If ``True``, raise immediately on the first failure instead
            of collecting all results.  Default: ``False``.

        Returns
        -------
        ok : list[str or pathlib.Path]
            Sources that are accessible.
        errors : list[tuple[str or pathlib.Path, Exception]]
            ``(source, exception)`` pairs for inaccessible sources.

        Raises
        ------
        ValueError or OSError
            Only when *raise_on_first* is ``True`` and any source fails.

        Notes
        -----
        File sources use :meth:`validate_input` (existence + is_file).
        URL sources send an HTTP HEAD request (falling back to GET if the
        server returns 405).

        Examples
        --------
        >>> ok, errors = DummyReader.check(
        ...     Path("report.pdf"),
        ...     "https://www.who.int/europe/news/item/...",
        ... )
        >>> print(f"{len(ok)} OK, {len(errors)} failed")
        """
        import re as _re  # noqa: PLC0415
        import urllib.request as _ur  # noqa: PLC0415

        _URL_RE = _re.compile(r"https?://", _re.IGNORECASE)  # noqa: N806
        ok: list = []
        errors: list = []

        for src in sources:
            try:
                src_str = str(src)
                if _URL_RE.match(src_str):
                    # Try HEAD, fall back to GET if 405
                    try:
                        req = _ur.Request(  # noqa: S310
                            src_str,
                            method="HEAD",
                            headers={"User-Agent": "scikitplot-corpus/1.0"},
                        )
                        _ur.urlopen(req, timeout=timeout)  # noqa: S310
                    except Exception as head_err:  # noqa: BLE001
                        # 405 Method Not Allowed → try GET
                        if "405" in str(head_err) or "501" in str(head_err):
                            req2 = _ur.Request(  # noqa: S310
                                src_str,
                                headers={"User-Agent": "scikitplot-corpus/1.0"},
                            )
                            with _ur.urlopen(  # noqa: S310
                                req2, timeout=timeout
                            ) as resp:
                                pass  # opened successfully
                        else:
                            raise
                else:
                    p = pathlib.Path(src)
                    if not p.exists():
                        raise ValueError(f"File does not exist: {p!r}")
                    if not p.is_file():
                        raise ValueError(f"Path is not a regular file: {p!r}")
                ok.append(src)
                logger.debug("DummyReader.check: OK %s", src)
            except Exception as exc:  # noqa: BLE001
                logger.debug("DummyReader.check: FAIL %s — %s", src, exc)
                if raise_on_first:
                    raise
                errors.append((src, exc))

        return ok, errors

    # ------------------------------------------------------------------
    # Reader implementation — always yields zero documents
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """Validate then yield nothing.

        Raises
        ------
        ValueError
            If the source file does not exist or is not accessible.
        urllib.error.URLError
            If ``source_uri`` is set and the URL is unreachable.
        """
        self.validate_input()
        return
        yield  # marks this as a generator

    def validate_input(self) -> None:
        """Check source accessibility without reading content.

        For file-based sources verifies existence and is_file.
        For URL-based sources (``source_uri`` set) sends a HEAD request.

        Raises
        ------
        ValueError
            File does not exist or is not a regular file.
        urllib.error.URLError
            URL is not reachable (network error, DNS failure, etc.).
        urllib.error.HTTPError
            Server returned a 4xx/5xx status code.
        """
        uri = getattr(self, "source_uri", None)
        if uri:
            import urllib.request as _ur  # noqa: PLC0415

            req = _ur.Request(  # noqa: S310
                uri,
                method="HEAD",  # noqa: S310
                headers={"User-Agent": "scikitplot-corpus/1.0"},
            )
            _ur.urlopen(req, timeout=10)  # noqa: S310
        else:
            super().validate_input()


# ===========================================================================
# PipelineGuard — resilient iteration with ErrorPolicy, dedup, checkpoint
# ===========================================================================


class PipelineGuard:
    """
    Wrap any document stream with resilience, deduplication, and checkpointing.

    :class:`PipelineGuard` is a thin, composable layer you place around
    any :meth:`DocumentReader.get_documents` call (or any
    ``Iterable[CorpusDocument]``) to get:

    - **Error isolation** — per-document failures are handled according to
      :class:`~scikitplot.corpus._schema.ErrorPolicy` instead of crashing
      the whole pipeline.
    - **Content deduplication** — documents with identical ``content_hash``
      are dropped after the first occurrence.
    - **Checkpoint / resume** — progress is periodically saved to a JSONL
      file so that a failed pipeline can resume from the last safe point.
    - **Retry with back-off** — transient errors (I/O, network) are retried
      up to ``max_retries`` times with exponential back-off.

    Parameters
    ----------
    policy : ErrorPolicy, optional
        How to handle per-document exceptions.
        Default: :attr:`~ErrorPolicy.LOG` (log and skip).
    dedup : bool, optional
        Drop documents with duplicate ``content_hash``.
        Default: ``True``.
    checkpoint_path : pathlib.Path or None, optional
        Path to a JSONL file for checkpoint/resume.  When set, every
        ``checkpoint_every`` documents are written; on restart, already-seen
        ``doc_id`` values are skipped.  Default: ``None`` (no checkpoint).
    checkpoint_every : int, optional
        Flush checkpoint every N yielded documents.  Default: 500.
    max_retries : int, optional
        Maximum retry attempts for :attr:`~ErrorPolicy.RETRY` policy.
        Default: 3.
    retry_delay : float, optional
        Initial back-off in seconds between retries (doubles each attempt).
        Default: 1.0.

    Notes
    -----
    **Zero-dependency design:** :class:`PipelineGuard` uses only
    ``pathlib``, ``json``, ``time``, and ``hashlib`` from the stdlib.
    It does not require the corpus schema module at import time (it
    reads ``content_hash`` and ``doc_id`` as plain attributes).

    **Thread safety:** Not thread-safe.  Use one guard per thread when
    processing sources in parallel.

    Examples
    --------
    Basic error isolation — skip broken documents:

    >>> guard = PipelineGuard(policy=ErrorPolicy.SKIP)
    >>> docs = list(guard.iter(reader.get_documents()))

    Full pipeline with dedup and checkpoint:

    >>> from pathlib import Path
    >>> guard = PipelineGuard(
    ...     policy=ErrorPolicy.LOG,
    ...     dedup=True,
    ...     checkpoint_path=Path("corpus.ckpt.jsonl"),
    ...     checkpoint_every=200,
    ... )
    >>> for doc in guard.iter(reader.get_documents()):
    ...     process(doc)
    >>> guard.close()

    Context manager (auto-close):

    >>> with PipelineGuard(checkpoint_path=Path("run.ckpt")) as guard:
    ...     docs = list(guard.iter(reader.get_documents()))

    Wrap :class:`_MultiSourceReader`:

    >>> reader = DocumentReader.create(Path("a.mp3"), Path("b.pdf"))
    >>> guard = PipelineGuard(policy=ErrorPolicy.SKIP)
    >>> docs = list(guard.iter(reader.get_documents()))
    """

    def __init__(
        self,
        policy: Any | None = None,
        *,
        dedup: bool = True,
        checkpoint_path: pathlib.Path | None = None,
        checkpoint_every: int = 500,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialise with error policy, dedup, and checkpoint settings.

        Parameters
        ----------
        policy : ErrorPolicy or None, optional
            Per-document error handling.  ``None`` defaults to
            :attr:`~ErrorPolicy.LOG`.  Default: ``None``.
        dedup : bool, optional
            Drop documents with duplicate ``content_hash``.  Default: ``True``.
        checkpoint_path : pathlib.Path or None, optional
            JSONL checkpoint file.  ``None`` disables checkpointing.
            Default: ``None``.
        checkpoint_every : int, optional
            Checkpoint flush frequency in documents.  Default: 500.
        max_retries : int, optional
            Retry attempts for :attr:`~ErrorPolicy.RETRY`.  Default: 3.
        retry_delay : float, optional
            Initial back-off in seconds (doubles each attempt).  Default: 1.0.
        """
        # Import here to avoid circular import at module level
        try:
            from ._schema import (  # noqa: N814, PLC0415
                ErrorPolicy as _EP,
            )

            _default_policy = _EP.LOG
        except ImportError:
            _default_policy = None  # type: ignore[assignment]

        self.policy = policy if policy is not None else _default_policy
        self.dedup = dedup
        self.checkpoint_path = (
            pathlib.Path(checkpoint_path) if checkpoint_path else None
        )
        self.checkpoint_every = max(1, checkpoint_every)
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)

        # Runtime state
        self._seen_hashes: set[str] = set()
        self._seen_ids: set[str] = set()
        # IDs loaded from the checkpoint file — kept separate from _seen_ids
        # so that the checkpoint resume skip does not accidentally trigger on
        # IDs that were added to _seen_ids by the dedup block in the same run.
        self._checkpoint_ids: set[str] = set()
        self._n_yielded: int = 0
        self._n_skipped_dedup: int = 0
        self._n_errors: int = 0
        self._ckpt_file: Any | None = None  # open file handle

        # Load checkpoint if it exists
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter(
        self,
        source: Any,  # Iterable[CorpusDocument]
    ) -> Generator[Any, None, None]:
        """
        Iterate *source* with resilience, dedup, and checkpoint.

        Parameters
        ----------
        source : Iterable[CorpusDocument]
            Any document iterable — typically ``reader.get_documents()``,
            a list, or another ``iter()`` call.

        Yields
        ------
        CorpusDocument
            Documents that passed dedup and error policy filters.

        Notes
        -----
        The guard opens the checkpoint file lazily on the first call to
        :meth:`iter`.  Call :meth:`close` (or use as context manager) to
        ensure the file is flushed and closed.
        """
        import time  # noqa: F401, PLC0415

        try:
            from ._schema import (  # noqa: N814, PLC0415
                ErrorPolicy as _EP,
            )

            _SKIP = _EP.SKIP  # noqa: N806
            _LOG = _EP.LOG  # noqa: N806
            _RETRY = _EP.RETRY  # noqa: N806
        except ImportError:
            _SKIP = _LOG = _RETRY = None  # type: ignore[assignment]  # noqa: N806

        for item in self._safe_iter(source):
            # item is either a CorpusDocument or an Exception
            if isinstance(item, BaseException):
                exc = item
                self._n_errors += 1
                policy_val = getattr(self.policy, "value", str(self.policy))

                if self.policy in (_LOG, _SKIP) or str(self.policy) in ("log", "skip"):
                    if (
                        self.policy in (_LOG,)  # noqa: FURB171
                        or str(self.policy) == "log"
                    ):
                        logger.warning(
                            "PipelineGuard: document error (policy=%s): %s: %s",
                            policy_val,
                            type(exc).__name__,
                            exc,
                        )
                    continue  # skip the bad document

                if (
                    self.policy in (_RETRY,)  # noqa: FURB171
                    or str(self.policy) == "retry"
                ):
                    # Retry is handled in _safe_iter; if we reach here it failed
                    logger.warning(
                        "PipelineGuard: exhausted %d retries: %s",
                        self.max_retries,
                        exc,
                    )
                    continue

                raise exc  # RAISE policy

            doc = item

            # ── Deduplication ─────────────────────────────────────────
            if self.dedup:
                h = getattr(doc, "content_hash", None)
                did = getattr(doc, "doc_id", None)
                if h and h in self._seen_hashes:
                    self._n_skipped_dedup += 1
                    continue
                if h:
                    self._seen_hashes.add(h)
                if did:
                    self._seen_ids.add(did)

            # ── Checkpoint skip (resume) ───────────────────────────────
            did = getattr(doc, "doc_id", None)
            if self._checkpoint_ids and did and did in self._checkpoint_ids:
                # This doc was already processed in a previous run — skip it.
                # _checkpoint_ids is populated exclusively by _load_checkpoint
                # and is disjoint from _seen_ids (which tracks dedup state
                # within the current run only).
                continue

            self._n_yielded += 1
            yield doc

            # ── Write checkpoint ───────────────────────────────────────
            if self.checkpoint_path and self._n_yielded % self.checkpoint_every == 0:
                self._append_checkpoint(doc)

    def close(self) -> None:
        """Flush and close the checkpoint file handle.

        Notes
        -----
        Called automatically when used as a context manager.
        Call manually when using :meth:`iter` without ``with``.
        """
        if self._ckpt_file is not None:
            try:
                self._ckpt_file.flush()
                self._ckpt_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._ckpt_file = None

    @property
    def stats(self) -> dict[str, int]:
        """Runtime statistics: yielded, skipped (dedup), errors."""
        return {
            "n_yielded": self._n_yielded,
            "n_skipped_dedup": self._n_skipped_dedup,
            "n_errors": self._n_errors,
        }

    def __repr__(self) -> str:
        policy = getattr(self.policy, "value", str(self.policy))
        return (
            f"PipelineGuard(policy={policy!r}, dedup={self.dedup}, "
            f"checkpoint={self.checkpoint_path}, "
            f"yielded={self._n_yielded})"
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Enter the context manager.

        Returns
        -------
        PipelineGuard
            Returns ``self`` — enables ``with PipelineGuard(...) as guard:``.
        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager and flush the checkpoint file.

        Calls :meth:`close` unconditionally, even when an exception
        occurred during iteration, to ensure the JSONL checkpoint is
        fully flushed and the file handle is released.
        """
        self.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _safe_iter(self, source: Any) -> Generator[Any, None, None]:
        """Yield documents or exceptions (retry logic for RETRY policy)."""
        import time  # noqa: PLC0415

        try:
            from ._schema import (  # noqa: N814, PLC0415
                ErrorPolicy as _EP,
            )

            _RETRY = _EP.RETRY  # noqa: N806
        except ImportError:
            _RETRY = None  # type: ignore[assignment]  # noqa: N806

        try:
            for doc in source:
                yield doc
        except Exception as exc:  # noqa: BLE001
            if self.policy in (_RETRY,) or str(self.policy) == "retry":  # noqa: FURB171
                for attempt in range(self.max_retries):
                    delay = self.retry_delay * (2**attempt)
                    logger.debug(
                        "PipelineGuard: retry %d/%d after %.1fs: %s",
                        attempt + 1,
                        self.max_retries,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
                    try:
                        for doc in source:  # type: ignore[assignment]
                            yield doc
                        return  # success
                    except Exception as retry_exc:  # noqa: BLE001
                        exc = retry_exc
            yield exc  # propagate as exception-item

    def _load_checkpoint(self) -> None:
        """Load seen doc_ids from a checkpoint JSONL file."""
        import json  # noqa: PLC0415

        if not self.checkpoint_path:
            return
        try:
            with self.checkpoint_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()  # noqa: PLW2901
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        did = obj.get("doc_id")
                        h = obj.get("content_hash")
                        if did:
                            self._seen_ids.add(did)
                            # _checkpoint_ids is the authoritative set for
                            # resume skip — kept separate from _seen_ids so
                            # the dedup block cannot pollute it.
                            self._checkpoint_ids.add(did)
                        if h:
                            self._seen_hashes.add(h)
                    except json.JSONDecodeError:
                        pass
            logger.info(
                "PipelineGuard: loaded checkpoint with %d seen doc_ids from %s",
                len(self._seen_ids),
                self.checkpoint_path,
            )
        except OSError as exc:
            logger.warning(
                "PipelineGuard: could not load checkpoint %s: %s",
                self.checkpoint_path,
                exc,
            )

    def _append_checkpoint(self, doc: Any) -> None:
        """Append a doc_id + content_hash line to the checkpoint file."""
        import json  # noqa: PLC0415

        if not self.checkpoint_path:
            return
        try:
            if self._ckpt_file is None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                self._ckpt_file = self.checkpoint_path.open("a", encoding="utf-8")
            record = {
                "doc_id": getattr(doc, "doc_id", None),
                "content_hash": getattr(doc, "content_hash", None),
                "source_file": getattr(doc, "source_file", None),
            }
            self._ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._ckpt_file.flush()
        except OSError as exc:
            logger.warning("PipelineGuard: checkpoint write failed: %s", exc)
