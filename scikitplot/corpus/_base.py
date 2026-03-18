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

from scikitplot.corpus._schema import (
    _PROMOTED_RAW_KEYS,
    ChunkingStrategy,
    CorpusDocument,
    SectionType,
    SourceType,
)

if TYPE_CHECKING:
    pass  # reserved for future static-analysis-only imports

logger = logging.getLogger(__name__)

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
        Class variable. The file extension this reader handles, including
        the leading dot (e.g. ``".txt"``, ``".xml"``, ``".zip"``). Must be
        lowercase. **Must** be defined on every concrete subclass.

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

    file_type: ClassVar[str]
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

    @classmethod
    def create(
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
        **kwargs: dict,
    ) -> DocumentReader:
        """
        Instantiate and return the appropriate reader for ``input_file``.

        Selects the reader class based on the file extension (lowercase),
        then constructs it with the supplied keyword arguments.

        Parameters
        ----------
        input_file : pathlib.Path or str
            Path to the source file.
        chunker : ChunkerBase or None, optional
            Chunker to inject. Default: ``None`` (no sub-chunking).
        filter_ : FilterBase or None, optional
            Filter to inject. Default: ``None`` (:class:`DefaultFilter`).
        filename_override : str or None, optional
            Override for the source file label. Default: ``None``.
        default_language : str or None, optional
            Default ISO 639-1 language code. Default: ``None``.
        source_type : SourceType or None, optional
            Kind of source (``BOOK``, ``MOVIE``, ``RESEARCH``, …). Propagated
            into every yielded ``CorpusDocument``. Default: ``None``.
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
        **kwargs : dict
            Additional kwargs for pass to type[DocumentReader].

        Returns
        -------
        DocumentReader
            Concrete reader instance appropriate for the given file type.

        Raises
        ------
        ValueError
            If no reader is registered for the file's extension. The error
            message includes the full list of supported types.

        Examples
        --------
        >>> from pathlib import Path
        >>> reader = DocumentReader.create(
        ...     Path("hamlet.xml"),
        ...     source_type=SourceType.PLAY,
        ...     source_title="Hamlet",
        ...     source_author="Shakespeare",
        ... )
        >>> type(reader).__name__
        'XMLReader'
        """
        path = pathlib.Path(input_file)
        ext = path.suffix.lower()
        reader_cls = cls._registry.get(ext)
        if reader_cls is None:
            supported = ", ".join(cls.supported_types()) or "(none registered)"
            raise ValueError(
                f"No DocumentReader registered for extension {ext!r}."
                f" Supported types: {supported}."
                f" Register a custom reader by subclassing DocumentReader"
                f" with file_type = {ext!r}."
            )
        # Build provenance dict; only include non-None values so that
        # get_documents() provenance merge stays clean.
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
    def from_url(
        cls,
        url: str,
        *,
        chunker: ChunkerBase | None = None,
        filter_: FilterBase | None = None,
        default_language: str | None = None,
        source_type: SourceType | None = None,
        source_title: str | None = None,
        source_author: str | None = None,
        source_date: str | None = None,
        collection_id: str | None = None,
        **kwargs: dict,
    ) -> DocumentReader:
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
        default_language : str or None, optional
            ISO 639-1 language code. Default: ``None``.
        source_type : SourceType or None, optional
            Kind of source (``BOOK``, ``MOVIE``, ``RESEARCH``, …). Propagated
            into every yielded ``CorpusDocument``. Default: ``None``.
        source_title : str or None, optional
            Title of the source work. Default: ``None``.
        source_author : str or None, optional
            Primary author. Default: ``None``.
        source_date : str or None, optional
            Publication date in ISO 8601 format. Default: ``None``.
        collection_id : str or None, optional
            Corpus collection identifier. Default: ``None``.
        **kwargs : dict
            Additional kwargs for pass to type[DocumentReader].

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

        Examples
        --------
        >>> reader = DocumentReader.from_url("https://en.wikipedia.org/wiki/Python")
        >>> docs = list(reader.get_documents())

        >>> yt = DocumentReader.from_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        >>> docs = list(yt.get_documents())
        """
        import re as _re  # noqa: PLC0415

        if not isinstance(url, str) or not _re.match(r"https?://", url):
            raise ValueError(
                f"DocumentReader.from_url: url must start with 'http://' or"
                f" 'https://'; got {url!r}."
            )

        # YouTube detection
        _YT_RE = _re.compile(  # noqa: N806
            r"https?://(www\.)?(youtube\.com/watch|youtu\.be/)"
        )
        is_youtube = bool(_YT_RE.match(url))
        reader_key = ":youtube" if is_youtube else ":url"

        reader_cls = cls._registry.get(reader_key)
        if reader_cls is None:
            hint = "YouTubeReader" if is_youtube else "WebReader"
            raise ImportError(
                f"No reader registered for {'YouTube' if is_youtube else 'web'}"
                f" URLs. Import scikitplot.corpus._readers to register {hint}:"
                f"\n  import scikitplot.corpus._readers"
            )

        # Build provenance dict
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

        return reader_cls(
            input_file=pathlib.Path(url),
            source_uri=url,
            chunker=chunker,
            filter_=filter_,
            default_language=default_language,
            source_provenance=prov,
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
# Public API
# ===========================================================================


__all__ = [  # noqa: RUF022
    # Abstract bases
    "DocumentReader",
    "ChunkerBase",
    "FilterBase",
    # Concrete built-in filter
    "DefaultFilter",
]
