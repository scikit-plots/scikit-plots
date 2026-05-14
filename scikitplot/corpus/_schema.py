# scikitplot/corpus/_schema.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._schema
=========================
Canonical data contracts for the scikitplot corpus pipeline.

This module is the **single source of truth** for every data type that flows
through the pipeline. It has **zero runtime dependencies** outside the Python
standard library so that it is safe to import anywhere — in readers, chunkers,
filters, embedders, exporters, and tests — without pulling in heavy optional
packages (numpy, pandas, polars, etc.).

Heavy-dependency imports are deferred to method bodies and guarded under
``TYPE_CHECKING`` for static analysis only.

Design invariants:

* All public classes are immutable-friendly: mutating state returns a new
  instance via :meth:`CorpusDocument.replace`.
* ``CorpusDocument.validate()`` must succeed before a document leaves any
  pipeline stage. Fail fast with actionable ``ValueError`` messages.
* ``doc_id`` is a deterministic 16-character SHA-1 prefix derived from
  ``(source_type, input_path, chunk_index, text[:64])``. Identical inputs
  always yield the same id, making deduplication and caching reliable.
* The ``embedding`` field is typed ``Optional[Any]`` at runtime so that numpy
  is **not** imported here. The companion ``_schema.pyi`` stub provides full
  ``numpy.typing.NDArray`` typing for static type checkers.
* ``_PROMOTED_RAW_KEYS`` is the authoritative frozenset of dict keys that are
  promoted from ``get_raw_chunks()`` dicts to first-class ``CorpusDocument``
  fields. Every key in a raw-chunk dict that is **not** in this set and is not
  ``"text"`` or ``"section_type"`` flows into ``metadata``.

Python compatibility:

Supports Python 3.8 through 3.15. All PEP-604 (``X | Y``) and PEP-585
(``dict[str, X]``) annotations are valid under ``from __future__ import
annotations``, which makes all annotations lazy strings at runtime.
"""  # noqa: D205, D400

from __future__ import annotations

import copy
import hashlib
import logging
import pathlib  # noqa: F401
import re
import warnings
from dataclasses import asdict, dataclass, field, fields  # noqa: F401
from dataclasses import replace as _dc_replace

# Only imports when type checking
from typing import (  # noqa: F401
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Final,
    Generator,
    Iterator,
    Protocol,
    Sequence,
    TypeVar,
)

# ---------------------------------------------------------------------------
# StrEnum compatibility shim
# MEDIUM-07 fix: import from the single centralised shim in _compat.py so
# the backport is not duplicated across modules.
# ---------------------------------------------------------------------------
from ._compat import StrEnum as _StrEnumBase  # noqa: E402

if TYPE_CHECKING:
    from typing_extensions import Self  # noqa: F401

if TYPE_CHECKING:
    # Imported only for static analysis; never executed at runtime.
    import numpy as np  # noqa: F401
    import numpy.typing as npt  # noqa: F401
    import pandas as pd
    import polars as pl

logger = logging.getLogger(__name__)

__all__ = [  # noqa: RUF022
    # Enumerations
    "SectionType",
    "ChunkingStrategy",
    "ExportFormat",
    "SourceType",
    "MatchMode",
    # New enumerations
    "Modality",
    "ErrorPolicy",
    # Core document type
    "CorpusDocument",
    # Bulk helpers
    "documents_to_pandas",
    "documents_to_polars",
    # Module-level constants (importable by tests and downstream consumers)
    "_PROMOTED_RAW_KEYS",  # frozenset of raw-chunk keys promoted to first-class fields
    "_SOURCE_EXT_MAP",  # MEDIUM-02: extension → SourceType lookup dict
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Matches strings that contain no Unicode letter characters — used by the
# default sentence filter to discard punctuation-only noise tokens.
# _NO_LETTER_RE: re.Pattern[str] = re.compile(r"^[^\w]*$", re.UNICODE)

# Basic DOI prefix pattern — DOIs always start with "10." followed by the
# registrant code. Used in validate() to warn on suspicious values.
_DOI_PREFIX_RE: re.Pattern[str] = re.compile(r"^10\.\d{4,}")

# ---------------------------------------------------------------------------
# TypeVar for Self-returning classmethods (Python 3.8+ compatible)
# ---------------------------------------------------------------------------
_T = TypeVar("_T", bound="CorpusDocument")  # noqa: PYI018

# ===========================================================================
# Modality — primary content kind of a CorpusDocument
# ===========================================================================


class Modality(_StrEnumBase):
    """
    Primary content modality of a :class:`CorpusDocument`.

    Notes
    -----
    **TEXT** — document carries non-empty ``text``; ``raw_tensor`` is
    ``None`` (the common, existing case).

    **IMAGE / AUDIO / VIDEO** — document carries ``raw_tensor`` and/or
    ``raw_bytes``; ``text`` may be ``None`` (raw-only) or non-None (e.g.
    OCR transcript alongside pixel array).

    **MULTIMODAL** — document carries both ``text`` and a ``raw_tensor``
    (e.g. an image with its OCR transcript).

    Reader-to-modality mapping:

    - ``TextReader / PDFReader / XMLReader / WebReader / YouTubeReader``
      → ``TEXT``
    - ``ImageReader(yield_raw=False)`` → ``TEXT``
    - ``ImageReader(yield_raw=True)`` → ``IMAGE`` or ``MULTIMODAL``
    - ``AudioReader(yield_waveform=False)`` → ``TEXT``
    - ``AudioReader(yield_waveform=True)`` → ``AUDIO``
    - ``VideoReader(yield_frames=False)`` → ``TEXT``
    - ``VideoReader(yield_frames=True)`` → ``VIDEO`` or ``MULTIMODAL``

    Examples
    --------
    >>> Modality.TEXT == "text"
    True
    >>> Modality("image")
    <Modality.IMAGE: 'image'>
    """

    TEXT = "text"
    """Plain text content — ``raw_tensor`` is ``None``."""

    IMAGE = "image"
    """Raster image tensor — ``raw_tensor`` shape is ``(H, W, C)`` uint8."""

    AUDIO = "audio"
    """Audio waveform — ``raw_tensor`` shape is ``(samples,)`` float32."""

    VIDEO = "video"
    """Video frame tensor — ``raw_tensor`` shape is ``(T, H, W, C)`` uint8."""

    MULTIMODAL = "multimodal"
    """Both ``text`` and ``raw_tensor`` are present (e.g. image + OCR)."""


# ===========================================================================
# ErrorPolicy — how the pipeline handles per-document errors
# ===========================================================================


class ErrorPolicy(_StrEnumBase):
    """
    Per-document error handling strategy for :class:`PipelineGuard`.

    Notes
    -----
    **RAISE** (default) — exceptions propagate immediately; caller handles.
    **SKIP** — broken documents are silently discarded; pipeline continues.
    **LOG** — exception is logged at WARNING level; document is discarded.
    **RETRY** — the document is retried up to ``max_retries`` times
    (for transient I/O errors); falls back to LOG on exhaustion.

    Examples
    --------
    >>> ErrorPolicy.SKIP == "skip"
    True
    """

    RAISE = "raise"
    """Propagate exceptions immediately (default, strictest)."""

    SKIP = "skip"
    """Discard failing documents silently."""

    LOG = "log"
    """Log failures at WARNING level and discard."""

    RETRY = "retry"
    """Retry transient failures up to ``max_retries`` times, then LOG."""


class SectionType(_StrEnumBase):
    r"""
    Semantic label for the role of a text chunk within its source document.

    Notes
    -----
    The ``UNKNOWN`` value is the safe default for readers that cannot determine
    section type. Downstream consumers (filters, exporters) should treat
    ``UNKNOWN`` as ``TEXT`` unless they have specific logic for it.

    Every value is a plain lowercase string so that it round-trips safely
    through CSV, JSON, and database storage without loss.

    Literary and dramatic values (``VERSE``, ``DIALOGUE``,
    ``STAGE_DIRECTION``) allow corpus consumers to exclude non-content sections
    from semantic matching without reprocessing the source.

    Research-paper values (``ABSTRACT``, ``REFERENCES``,
    ``ACKNOWLEDGEMENTS``) allow consumers to restrict or exclude specific paper
    sections from citation-matching pipelines.

    Examples
    --------
    >>> SectionType.TEXT == "text"
    True
    >>> SectionType("footnote") is SectionType.FOOTNOTE
    True
    >>> SectionType("abstract") is SectionType.ABSTRACT
    True
    """

    TEXT = "text"
    """Body text — the primary content of the document."""

    FOOTNOTE = "footnote"
    """Footnote content extracted below the main body."""

    TITLE = "title"
    """Document or section title / heading."""

    TABLE = "table"
    """Tabular data rendered as a text chunk."""

    HEADER = "header"
    """Page or section header (running head, masthead, etc.)."""

    FIGURE = "figure"
    """Figure caption or alt-text associated with an image."""

    CODE = "code"
    """Source-code or pre-formatted block."""

    CAPTION = "caption"
    """Caption attached to a non-figure element (table caption, etc.)."""

    METADATA = "metadata"
    """Document-level metadata (author, date, abstract, etc.)."""

    UNKNOWN = "unknown"
    """Section type could not be determined."""

    # ------------------------------------------------------------------
    # Literary / dramatic / research-paper values (Issue S-1)
    # ------------------------------------------------------------------

    ABSTRACT = "abstract"
    """Research-paper abstract — distinguishable from body for citation matching."""

    REFERENCES = "references"
    """Reference list — should typically be excluded from semantic matching."""

    STAGE_DIRECTION = "stage_direction"
    """Dramatic stage direction, e.g. ``[Enter Hamlet]`` — not narrative content."""

    DIALOGUE = "dialogue"
    """Speaker turn in a dramatic or screenplay source."""

    VERSE = "verse"
    """Poetic line or stanza — metric/phonetic matching differs from prose."""

    ACKNOWLEDGEMENTS = "acknowledgements"
    """Acknowledgements section — typically excluded from content matching."""

    LIST_ITEM = "list_item"
    """Bullet-list or numbered-list item — distinct from paragraph prose."""

    SIDEBAR = "sidebar"
    """Editorial pull-quote, sidebar, or callout box."""

    # ------------------------------------------------------------------
    # Audio / music values (Scenarios 11, 12)
    # ------------------------------------------------------------------

    LYRICS = "lyrics"
    """Song lyrics line from an LRC or lyrics file — distinct from prose."""

    TRANSCRIPT = "transcript"
    """Machine-generated ASR transcription (Whisper, etc.) — not human-written."""


class ChunkingStrategy(_StrEnumBase):
    """
    Describes how a :class:`CorpusDocument` was segmented from raw text.

    Notes
    -----
    This value is written into every ``CorpusDocument`` so that downstream
    consumers can reproduce or verify segmentation without re-reading the
    source document.

    Examples
    --------
    >>> ChunkingStrategy.SENTENCE == "sentence"
    True
    >>> ChunkingStrategy.NONE == "none"
    True
    """

    SENTENCE = "sentence"
    """One sentence per chunk, produced by a language-model segmenter."""

    PARAGRAPH = "paragraph"
    """One paragraph per chunk, split on blank lines (``\\n\\n``)."""

    FIXED_WINDOW = "fixed_window"
    """Sliding window of fixed token/character count with configurable overlap."""

    SEMANTIC = "semantic"
    """Boundary detected by semantic similarity shift (topic segmentation)."""

    PAGE = "page"
    """One page per chunk, as determined by the source format (PDF, ALTO)."""

    BLOCK = "block"
    """Format-native block unit (ALTO TextBlock, TEI <p>, HTML <div>, etc.)."""

    CUSTOM = "custom"
    """User-supplied chunking logic; opaque to the standard pipeline."""

    NONE = "none"
    """No chunking applied — the whole document is one chunk."""


class ExportFormat(_StrEnumBase):
    """
    Supported serialisation targets for a completed corpus.

    Notes
    -----
    Not all targets are available in all environments. The pipeline checks
    availability at export time and raises ``ImportError`` with the required
    package name if the target is unavailable.

    Examples
    --------
    >>> ExportFormat.PARQUET == "parquet"
    True
    """

    CSV = "csv"
    """Comma-separated values; universal but loses numpy embedding arrays."""

    PARQUET = "parquet"
    """Column-oriented binary format via ``pyarrow`` or ``polars``."""

    JSON = "json"
    """JSON-lines (one document per line) for maximum interoperability."""

    JSONL = "jsonl"
    """Alias for JSON-lines format."""

    HUGGINGFACE = "huggingface"
    """HuggingFace ``datasets.Dataset`` object or saved dataset directory."""

    MLFLOW = "mlflow"
    """MLflow artifact (logs corpus + metadata to active or specified run)."""

    PICKLE = "pickle"
    """Python pickle — fastest round-trip; not portable across Python versions."""

    JOBLIB = "joblib"
    """joblib dump — efficient for large numpy arrays inside documents."""

    NUMPY = "numpy"
    """``numpy.savez_compressed`` — embeddings only, loses text/metadata."""

    POLARS = "polars"
    """In-memory ``polars.DataFrame``; returned, not written to disk."""

    PANDAS = "pandas"
    """In-memory ``pandas.DataFrame``; returned, not written to disk."""


class SourceType(_StrEnumBase):
    """
    Semantic label for the kind of source from which a document was read.

    Notes
    -----
    Used as a first-class typed column so that pre-filters on large corpora
    (e.g. 50 million documents) can use predicate pushdown rather than an O(n)
    scan of a ``metadata`` dict.

    Every value is a plain lowercase string that round-trips through CSV,
    JSON, Parquet, and database storage without loss.

    Reader affinity, Each value maps to the reader most likely to handle it:

    ==========================  ========================  ===========================
    SourceType value            Default reader            Typical file extension
    ==========================  ========================  ===========================
    ``BOOK / ARTICLE / ...``    ``TextReader``            ``.txt .md .rst``
    ``RESEARCH``                ``PDFReader``             ``.pdf``
    ``IMAGE``                   ``ImageReader``           ``.png .jpg .tiff …``
    ``AUDIO / PODCAST / ...``   ``AudioReader``           ``.mp3 .wav .flac …``
    ``VIDEO / MOVIE / ...``     ``VideoReader``           ``.mp4 .mkv …``
    ``WEB / WIKI / BLOG``       ``WebReader``             ``http(s)://``
    ``VIDEO`` (YouTube)         ``YouTubeReader``         ``youtu.be / youtube.com``
    ``UNKNOWN``                 inferred from extension   any
    ==========================  ========================  ===========================

    Examples
    --------
    >>> SourceType.BOOK == "book"
    True
    >>> SourceType("wiki") is SourceType.WIKI
    True
    >>> SourceType.UNKNOWN == "unknown"
    True
    """

    # ------------------------------------------------------------------
    # Written / textual
    # ------------------------------------------------------------------

    BOOK = "book"
    """Printed or digital book (novel, monograph, anthology, etc.)."""

    ARTICLE = "article"
    """Magazine or journal article (non-peer-reviewed)."""

    RESEARCH = "research"
    """Peer-reviewed research paper (arXiv, ResearchGate, DOI-bearing PDF)."""

    BIOGRAPHY = "biography"
    """Biography or autobiography."""

    PLAY = "play"
    """Dramatic play text (Shakespeare, modern theatre, screenplays)."""

    POEM = "poem"
    """Poem or collection of poems."""

    # ------------------------------------------------------------------
    # Journalism / media / broadcast
    # ------------------------------------------------------------------

    NEWS = "news"
    """News article from a news outlet or wire service."""

    BLOG = "blog"
    """Personal or corporate blog post."""

    NEWSLETTER = "newsletter"
    """Email or web newsletter (Substack, Revue, Mailchimp, etc.)."""

    PRESS_RELEASE = "press_release"
    """Official press release or public statement."""

    # ------------------------------------------------------------------
    # Audio / spoken word
    # ------------------------------------------------------------------

    AUDIO = "audio"
    """Generic audio source (transcript extracted via ASR / Whisper)."""

    PODCAST = "podcast"
    """Podcast episode — audio file with associated RSS/feed metadata."""

    LECTURE = "lecture"
    """Academic or conference lecture (audio or video recording)."""

    INTERVIEW = "interview"
    """Interview recording or transcript (audio, video, or text)."""

    # ------------------------------------------------------------------
    # Video / visual
    # ------------------------------------------------------------------

    VIDEO = "video"
    """Generic video source (transcript extracted from video file or stream)."""

    MOVIE = "movie"
    """Feature film or short film source (subtitle file or OCR transcript)."""

    SUBTITLE = "subtitle"
    """Subtitle / caption file (.srt, .vtt, .sbv, .sub)."""

    # ------------------------------------------------------------------
    # Web / online
    # ------------------------------------------------------------------

    WEB = "web"
    """General web page (HTML scraped from an http/https URL)."""

    WIKI = "wiki"
    """Wikipedia or MediaWiki article."""

    SOCIAL_MEDIA = "social_media"
    """Post, thread, or profile from a social media platform."""

    FORUM = "forum"
    """Online forum post, thread, or discussion (Reddit, Stack Overflow, etc.)."""

    FAQ = "faq"
    """Frequently-asked-questions page or document."""

    # ------------------------------------------------------------------
    # Reference / professional
    # ------------------------------------------------------------------

    DOCUMENTATION = "documentation"
    """Technical or product documentation (API docs, user manuals, etc.)."""

    TUTORIAL = "tutorial"
    """Step-by-step guide or how-to article."""

    MANUAL = "manual"
    """Instruction manual, operator guide, or maintenance handbook."""

    REPORT = "report"
    """Formal report (annual report, white paper, government report, etc.)."""

    # ------------------------------------------------------------------
    # Domain-specific
    # ------------------------------------------------------------------

    LEGAL = "legal"
    """Legal document (contract, court ruling, legislation, terms of service)."""

    MEDICAL = "medical"
    """Medical or clinical document (case study, clinical trial, drug insert)."""

    PATENT = "patent"
    """Patent application or granted patent document."""

    # ------------------------------------------------------------------
    # Structured / data
    # ------------------------------------------------------------------

    SPREADSHEET = "spreadsheet"
    """Spreadsheet source (.xlsx, .csv, .ods)."""

    DATASET = "dataset"
    """Structured dataset (JSON, JSONL, Parquet, database export)."""

    CODE = "code"
    """Source-code file or repository."""

    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------

    EMAIL = "email"
    """Email message or mailing-list post."""

    CHAT = "chat"
    """Chat or messaging log (Slack, Teams, WhatsApp, IRC export)."""

    # ------------------------------------------------------------------
    # Visual / image
    # ------------------------------------------------------------------

    IMAGE = "image"
    """Image source (OCR'd text from a raster image file)."""

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    UNKNOWN = "unknown"
    """Source type could not be determined."""

    # ------------------------------------------------------------------
    # Class-level inference
    # ------------------------------------------------------------------

    #: Extension → SourceType mapping used by :meth:`infer`.
    #: Only canonical extensions (lower-case, with leading dot).
    _EXT_MAP: ClassVar[dict[str, SourceType]]

    @classmethod
    def infer(  # noqa: PLR0911, PLR0912
        cls,
        input_path: str | pathlib.Path | None = None,
        *,
        mime_type: str | None = None,
    ) -> Self:
        """
        Infer the most likely :class:`SourceType` from a file path or MIME type.

        Parameters
        ----------
        input_path : str, pathlib.Path, or None, optional
            File path or URL string.  The extension (lower-case) is
            extracted and looked up in the internal extension map.
            ``None`` falls through to *mime_type* lookup.  Default: ``None``.
        mime_type : str or None, optional
            MIME type string (e.g. ``"application/pdf"``).  Used when
            *input_path* has no recognisable extension (e.g. extensionless
            API URLs).  Default: ``None``.

        Returns
        -------
        SourceType
            Inferred type, or :attr:`SourceType.UNKNOWN` when neither
            *input_path* nor *mime_type* yields a match.

        Notes
        -----
        **Priority:** extension (from *input_path*) wins over *mime_type*.

        **Use case — ZipReader:** Each member is passed to
        :meth:`infer` so that ``source_type`` is never ``UNKNOWN``
        unless truly ambiguous::

            st = SourceType.infer(member_path)
            # → SourceType.RESEARCH for .pdf, SourceType.IMAGE for .jpg …

        **Use case — from_url probe:** After downloading an extensionless
        URL, the server's ``Content-Type`` header is passed as *mime_type*::

            st = SourceType.infer(mime_type="audio/mpeg")
            # → SourceType.AUDIO

        Examples
        --------
        >>> SourceType.infer("report.pdf")
        <SourceType.RESEARCH: 'research'>
        >>> SourceType.infer("podcast.mp3")
        <SourceType.AUDIO: 'audio'>
        >>> SourceType.infer(mime_type="image/jpeg")
        <SourceType.IMAGE: 'image'>
        >>> SourceType.infer("mystery.bin")
        <SourceType.UNKNOWN: 'unknown'>
        """
        import os as _os  # noqa: PLC0415

        # Extension lookup
        if input_path is not None:
            path_str = str(input_path)
            name_lower = _os.path.basename(path_str).lower()
            # Compound extensions first
            for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
                if name_lower.endswith(compound):
                    return cls.UNKNOWN  # archive, ambiguous content
            _, ext = _os.path.splitext(name_lower)
            if ext and ext in cls._EXT_MAP:
                return cls._EXT_MAP[ext]

        # MIME type lookup
        if mime_type is not None:
            mime_lower = mime_type.split(";")[0].strip().lower()
            if mime_lower.startswith("audio/"):
                return cls.AUDIO
            if mime_lower.startswith("video/"):
                return cls.VIDEO
            if mime_lower.startswith("image/"):
                return cls.IMAGE
            if mime_lower in ("application/pdf",):  # noqa: FURB171
                return cls.RESEARCH
            if mime_lower in ("text/plain", "text/markdown", "text/x-rst"):
                return cls.ARTICLE
            if mime_lower in (
                "application/zip",
                "application/x-tar",
                "application/gzip",
                "application/x-bzip2",
            ):
                return cls.UNKNOWN
            if mime_lower in ("text/html",):  # noqa: FURB171
                return cls.WEB
            if mime_lower in (
                "application/json",
                "text/csv",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ):
                return cls.DATASET

        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# MEDIUM-02: Extension → SourceType lookup table.
#
# Previously this was a bare post-class attribute mutation:
#     SourceType._EXT_MAP = { ... }
# That pattern is fragile (import-order dependent), untypeable by static
# analysis tools, and not idiomatic Python.
#
# Fix: declare as a named, ``Final``-typed module-level constant so it is
# visible to mypy/pyright, importable in tests without going through the
# class, and free from post-class mutation.  SourceType._EXT_MAP is then
# wired to the same object — existing callers that reference ``cls._EXT_MAP``
# (i.e., SourceType.infer()) continue to work without any change.
# ---------------------------------------------------------------------------

_SOURCE_EXT_MAP: Final[dict[str, SourceType]] = {
    # Text / document
    ".txt": SourceType.ARTICLE,
    ".md": SourceType.ARTICLE,
    ".markdown": SourceType.ARTICLE,
    ".rst": SourceType.ARTICLE,
    ".rtf": SourceType.ARTICLE,
    ".pdf": SourceType.RESEARCH,
    ".doc": SourceType.ARTICLE,
    ".docx": SourceType.ARTICLE,
    ".odt": SourceType.ARTICLE,
    # Markup / data
    ".xml": SourceType.ARTICLE,
    ".html": SourceType.WEB,
    ".htm": SourceType.WEB,
    ".json": SourceType.DATASET,
    ".jsonl": SourceType.DATASET,
    ".csv": SourceType.DATASET,
    ".tsv": SourceType.DATASET,
    ".xlsx": SourceType.SPREADSHEET,
    ".xls": SourceType.SPREADSHEET,
    ".ods": SourceType.SPREADSHEET,
    # Images
    ".jpg": SourceType.IMAGE,
    ".jpeg": SourceType.IMAGE,
    ".png": SourceType.IMAGE,
    ".gif": SourceType.IMAGE,
    ".webp": SourceType.IMAGE,
    ".tiff": SourceType.IMAGE,
    ".tif": SourceType.IMAGE,
    ".bmp": SourceType.IMAGE,
    ".svg": SourceType.IMAGE,
    # Audio
    ".mp3": SourceType.AUDIO,
    ".wav": SourceType.AUDIO,
    ".flac": SourceType.AUDIO,
    ".ogg": SourceType.AUDIO,
    ".m4a": SourceType.AUDIO,
    ".aac": SourceType.AUDIO,
    ".wma": SourceType.AUDIO,
    ".aiff": SourceType.AUDIO,
    ".opus": SourceType.AUDIO,
    ".wv": SourceType.AUDIO,
    # Video
    ".mp4": SourceType.VIDEO,
    ".avi": SourceType.VIDEO,
    ".mkv": SourceType.VIDEO,
    ".mov": SourceType.VIDEO,
    ".webm": SourceType.VIDEO,
    ".m4v": SourceType.VIDEO,
    ".wmv": SourceType.VIDEO,
    ".flv": SourceType.VIDEO,
    # Subtitle
    ".srt": SourceType.SUBTITLE,
    ".vtt": SourceType.SUBTITLE,
    ".sbv": SourceType.SUBTITLE,
    ".sub": SourceType.SUBTITLE,
    ".lrc": SourceType.SUBTITLE,
    # Code
    ".py": SourceType.CODE,
    ".js": SourceType.CODE,
    ".ts": SourceType.CODE,
    ".java": SourceType.CODE,
    ".c": SourceType.CODE,
    ".cpp": SourceType.CODE,
    ".go": SourceType.CODE,
    ".rs": SourceType.CODE,
    ".rb": SourceType.CODE,
    ".sh": SourceType.CODE,
}

# Wire into the class variable so cls._EXT_MAP lookups in SourceType.infer()
# resolve to the same object as _SOURCE_EXT_MAP.
# The ClassVar declaration in SourceType provides the type annotation;
# this line performs the single, explicit assignment.
SourceType._EXT_MAP = _SOURCE_EXT_MAP  # type: ignore[assignment]


class MatchMode(_StrEnumBase):
    """
    Search mode for intertextual matching queries against a corpus index.

    Notes
    -----
    ``MatchMode`` belongs on search *query* objects, not on
    :class:`CorpusDocument`. It is defined here so it can be imported by the
    similarity layer, RAG/MCP consumers, and tests without depending on any
    heavy package.

    Mechanism summary:

    * ``STRICT``   — exact substring scan of ``doc.text``
    * ``KEYWORD``  — BM25 / TF-IDF on ``doc.tokens`` / ``doc.keywords``
    * ``SEMANTIC`` — ANN cosine search on ``doc.embedding``
    * ``HYBRID``   — Reciprocal Rank Fusion of KEYWORD and SEMANTIC scores

    Examples
    --------
    >>> MatchMode.SEMANTIC == "semantic"
    True
    >>> MatchMode("hybrid") is MatchMode.HYBRID
    True
    """

    STRICT = "strict"
    """Exact substring match within ``text``."""

    KEYWORD = "keyword"
    """BM25 / TF-IDF match on ``tokens`` or ``keywords``."""

    SEMANTIC = "semantic"
    """Approximate nearest-neighbour search on ``embedding``."""

    HYBRID = "hybrid"
    """Reciprocal Rank Fusion of KEYWORD and SEMANTIC scores."""


# ---------------------------------------------------------------------------
# _PROMOTED_RAW_KEYS
# ---------------------------------------------------------------------------

_PROMOTED_RAW_KEYS: frozenset[str] = frozenset(
    {
        # Provenance
        "source_type",
        "source_title",
        "source_author",
        "source_date",
        "collection_id",
        "url",
        "doi",
        "isbn",
        # Position
        "page_number",
        "paragraph_index",
        "line_number",
        "parent_doc_id",
        # Dramatic position
        "act",
        "scene_number",
        # Media-specific
        "timecode_start",
        "timecode_end",
        "confidence",
        "ocr_engine",
        "bbox",
        # NLP enrichment
        "normalized_text",
        "raw_text",  # pre-processing form: HTML/tag-stripped, WS-joined, or ASR output before NLP (all readers)
        "tokens",
        "lemmas",
        "stems",
        "keywords",
        # Raw media (new)
        "modality",
        "raw_bytes",
        "raw_tensor",
        "raw_shape",
        "raw_dtype",
        "frame_index",
        "content_hash",
        # Multilang / multi-script (populated from MultilangMixin via bridge)
        "script",
        "script_direction",
        "grapheme_count",
        "codepoint_count",
        "is_mixed_script",
        "script_spans",
        "chunking_unit",
        "semanteme_count",
        "morphemes",
        "determinative_groups",
        "script_model_version",
    }
)
"""
Frozenset of raw-chunk dict keys that are promoted to first-class
:class:`CorpusDocument` fields by :meth:`DocumentReader.get_documents`.

Any key yielded by :meth:`DocumentReader.get_raw_chunks` that is **not** in
this set, and is not ``"text"`` or ``"section_type"``, flows into
``CorpusDocument.metadata`` instead.

This constant is the single authoritative list that keeps
``get_raw_chunks()`` routing consistent. Readers must use exactly these
key names when yielding promoted fields.
"""


# ===========================================================================
# CorpusDocument — the canonical output unit
# ===========================================================================


@dataclass(frozen=True)
class CorpusDocument:
    """
    Canonical representation of a single text chunk in a processed corpus.

    A ``CorpusDocument`` is the unit of data that flows between every stage of
    the pipeline: readers produce them, chunkers subdivide them, filters
    accept or reject them, embedders enrich them, and exporters serialise them.

    Parameters
    ----------
    doc_id : str
        Stable 16-character hex identifier. Generated deterministically from
        ``(source_type, input_path, chunk_index, text[:64])`` via
        :meth:`make_doc_id` if not supplied. Must be non-empty.
    input_path : str
        Name or relative path of the original source file. Must be non-empty.
        Set from ``input_path.name`` by readers; do **not** include absolute
        paths to keep corpora portable across machines.
    chunk_index : int
        Zero-based ordinal of this chunk within the source document. Must be
        >= 0. Unique per ``(input_path, chunking_strategy)`` pair.
    text : str
        Cleaned, segmented text content of this chunk. Must be non-empty after
        stripping whitespace.
    section_type : SectionType, optional
        Semantic role of this chunk within its source document. Default:
        ``SectionType.TEXT``.
    chunking_strategy : ChunkingStrategy, optional
        Strategy used to produce this chunk. Default:
        ``ChunkingStrategy.NONE`` (whole document, no splitting applied).
    language : str or None, optional
        ISO 639-1 language code. Default: ``None``.
    char_start : int or None, optional
        Character offset of chunk start within the original document. Default:
        ``None``.
    char_end : int or None, optional
        Character offset of chunk end (exclusive). Default: ``None``.
    embedding : array-like or None, optional
        Dense vector representation of ``text``. Stored as ``Any`` at runtime;
        the ``.pyi`` stub provides ``NDArray[float32]`` for type checkers.
        Default: ``None``.
    metadata : dict, optional
        Open-ended key-value store for truly ad-hoc or format-specific fields
        (ISBN edition, translator, speaker, etc.). All keys must be strings.
        Default: empty dict.
    source_type : SourceType, optional
        Kind of source (BOOK, MOVIE, RESEARCH, WIKI, …). Used as a typed
        pre-filter column. Default: ``SourceType.UNKNOWN``.
    source_title : str or None, optional
        Title of the source work. Default: ``None``.
    source_author : str or None, optional
        Primary author. Default: ``None``.
    source_date : str or None, optional
        Publication date in ISO 8601 format. Default: ``None``.
    collection_id : str or None, optional
        Identifier grouping related sources into one corpus. Default: ``None``.
    url : str or None, optional
        Source URL for web-fetched documents. Default: ``None``.
    doi : str or None, optional
        Digital Object Identifier. Default: ``None``.
    isbn : str or None, optional
        International Standard Book Number. Default: ``None``.
    page_number : int or None, optional
        Zero-based page index. Default: ``None``.
    paragraph_index : int or None, optional
        Zero-based paragraph index within the page or document. Default:
        ``None``.
    line_number : int or None, optional
        Zero-based line number. Default: ``None``.
    parent_doc_id : str or None, optional
        doc_id of the parent chunk when this is a sub-division. Default:
        ``None``.
    act : int or None, optional
        Act number (one-based) in a dramatic source. Default: ``None``.
    scene_number : int or None, optional
        Scene number (one-based) within an act. Default: ``None``.
    timecode_start : float or None, optional
        Start timecode in seconds (>= 0). Default: ``None``.
    timecode_end : float or None, optional
        End timecode in seconds (>= timecode_start). Default: ``None``.
    confidence : float or None, optional
        OCR or ASR confidence in [0.0, 1.0]. Default: ``None``.
    ocr_engine : str or None, optional
        Name of the OCR engine used. Default: ``None``.
    bbox : tuple of float or None, optional
        Bounding box (x0, y0, x1, y1). Must be a 4-tuple of floats. Default:
        ``None``.
    normalized_text : str or None, optional
        Normalised text used by the embedding engine. Default: ``None``.
    tokens : list of str or None, optional
        Tokenised word list (not included in repr or equality). Default:
        ``None``.
    lemmas : list of str or None, optional
        Lemmatised tokens (not included in repr or equality). Default: ``None``.
    stems : list of str or None, optional
        Stemmed tokens (not included in repr or equality). Default: ``None``.
    keywords : list of str or None, optional
        Extracted keyphrases (not included in repr or equality). Default:
        ``None``.

    Attributes
    ----------
    REQUIRED_FIELDS : tuple of str
        Class-level tuple of field names that must be non-empty/non-negative
        for :meth:`validate` to pass.

    Raises
    ------
    ValueError
        If :meth:`validate` is called and any invariant is violated.

    See Also
    --------
    scikitplot.corpus._base.DocumentReader : Produces CorpusDocuments.
    scikitplot.corpus._pipeline.CorpusPipeline : Orchestrates the full flow.

    Notes
    -----
    **Immutability convention:** ``CorpusDocument`` is a mutable dataclass for
    performance, but pipeline stages must not mutate documents in-place after
    yielding them. Use :meth:`replace` to create modified copies.

    **Embedding storage:** When exporting to CSV or JSON, the embedding array
    is serialised as a flat list of floats. When exporting to Parquet or
    HuggingFace format, the array is stored natively.

    **NLP list fields** (``tokens``, ``lemmas``, ``stems``, ``keywords``) are
    excluded from ``__repr__`` and equality comparisons because they are large
    derived views of ``text``.

    Examples
    --------
    Creating from factory with auto-generated id:

    >>> doc = CorpusDocument.create(
    ...     input_path="corpus.xml",
    ...     chunk_index=3,
    ...     text="Das Kapital ist ein Werk von Marx.",
    ...     source_type=SourceType.BOOK,
    ...     source_author="Marx, Karl",
    ...     source_title="Das Kapital",
    ...     language="de",
    ...     page_number=42,
    ... )
    >>> len(doc.doc_id)
    16

    Round-tripping to dict and back:

    >>> d = doc.to_dict()
    >>> restored = CorpusDocument.from_dict(d)
    >>> restored.doc_id == doc.doc_id
    True
    """

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------

    REQUIRED_FIELDS: ClassVar[tuple[str, ...]] = (
        "doc_id",
        "input_path",
    )
    """Fields that must be non-empty strings for :meth:`validate` to pass.

    Notes
    -----
    ``text`` is intentionally excluded from this tuple.  For TEXT-modality
    documents, ``validate()`` enforces non-empty text directly.  For
    raw-media documents (``modality`` is IMAGE, AUDIO, or VIDEO), ``text``
    may legitimately be ``None`` — the document carries its content in
    ``raw_tensor`` or ``raw_bytes`` instead.
    """

    # ------------------------------------------------------------------
    # Core fields — positional in __init__
    # ------------------------------------------------------------------

    doc_id: str
    """Stable 16-character hex identifier for this chunk."""

    input_path: str
    """Name of the original source file (not an absolute path)."""

    chunk_index: int
    """Zero-based position of this chunk within the source document."""

    text: str
    """Cleaned, segmented text content."""

    section_type: SectionType = field(default=SectionType.TEXT)
    """Semantic role of this chunk."""

    chunking_strategy: ChunkingStrategy = field(default=ChunkingStrategy.NONE)
    """Strategy used to produce this chunk. Default: :attr:`ChunkingStrategy.NONE`.

    Set explicitly by chunkers when they produce sub-chunks from a raw document.
    ``NONE`` means no segmentation was applied — the whole document is one chunk.
    Chunkers must always override this to their corresponding strategy value
    so that downstream consumers can reproduce or verify segmentation.
    """

    language: str | None = field(default=None)
    """ISO 639-1 language code, or ``None`` if unknown."""

    char_start: int | None = field(default=None)
    """Character offset of chunk start in source, or ``None``."""

    char_end: int | None = field(default=None)
    """Character offset of chunk end (exclusive) in source, or ``None``."""

    embedding: Any | None = field(default=None, repr=False, compare=False)
    """Dense vector embedding, or ``None`` if not yet computed."""

    # ------------------------------------------------------------------
    # Raw tensor / multimodal fields — any-media-to-any-model path
    # ------------------------------------------------------------------

    modality: Modality = field(default_factory=lambda: Modality.TEXT)
    """Primary content modality. Default: :attr:`Modality.TEXT`."""

    raw_bytes: bytes | None = field(default=None, repr=False, compare=False)
    """Raw encoded media bytes (e.g. JPEG bytes).  ``None`` for text-only."""

    raw_tensor: Any = field(default=None, repr=False, compare=False)
    """Decoded media array ready for model input. Shape conventions:
    image ``(H,W,C)`` uint8; audio ``(samples,)`` float32;
    video ``(T,H,W,C)`` uint8.  ``None`` for text-only."""

    raw_shape: tuple[int, ...] | None = field(default=None)
    """Shape of ``raw_tensor`` as a plain Python tuple.  Default: ``None``."""

    raw_dtype: str | None = field(default=None)
    """String dtype of ``raw_tensor`` (e.g. ``"uint8"``).  Default: ``None``."""

    frame_index: int | None = field(default=None)
    """Zero-based frame index in a video or multi-frame image.  Default: ``None``."""

    content_hash: str | None = field(default=None)
    """SHA-256 hex digest (32 chars) of canonical content.  Dedup key."""

    metadata: dict[str, Any] = field(default_factory=dict, compare=False)
    """Truly ad-hoc format-specific metadata."""

    # ------------------------------------------------------------------
    # Provenance fields (Issue S-4)
    # ------------------------------------------------------------------

    source_type: SourceType = field(default=SourceType.UNKNOWN)
    """Kind of source (BOOK, MOVIE, RESEARCH, WIKI, …)."""

    source_title: str | None = field(default=None)
    """Title of the source work."""

    source_author: str | None = field(default=None)
    """Primary author of the source."""

    source_date: str | None = field(default=None)
    """Publication or creation date in ISO 8601 format."""

    collection_id: str | None = field(default=None)
    """Identifier grouping related sources into one corpus."""

    url: str | None = field(default=None)
    """Source URL for web-fetched documents."""

    doi: str | None = field(default=None)
    """Digital Object Identifier of the source."""

    isbn: str | None = field(default=None)
    """International Standard Book Number of the source."""

    # ------------------------------------------------------------------
    # Position fields (Issue S-4)
    # ------------------------------------------------------------------

    page_number: int | None = field(default=None)
    """Zero-based page index within the source document."""

    paragraph_index: int | None = field(default=None)
    """Zero-based paragraph index within the page or document."""

    line_number: int | None = field(default=None)
    """Zero-based line number within the document."""

    parent_doc_id: str | None = field(default=None)
    """doc_id of the parent chunk when this is a sub-division."""

    # ------------------------------------------------------------------
    # Dramatic position fields (Issue S-4)
    # ------------------------------------------------------------------

    act: int | None = field(default=None)
    """Act number (one-based) in a dramatic source."""

    scene_number: int | None = field(default=None)
    """Scene number (one-based) within an act."""

    # ------------------------------------------------------------------
    # Media-specific fields (Issue S-4)
    # ------------------------------------------------------------------

    timecode_start: float | None = field(default=None)
    """Start timecode in seconds for subtitle / video / audio sources."""

    timecode_end: float | None = field(default=None)
    """End timecode in seconds."""

    confidence: float | None = field(default=None)
    """OCR or ASR confidence score in [0.0, 1.0]."""

    ocr_engine: str | None = field(default=None)
    """Name of the OCR engine used."""

    bbox: tuple[float, float, float, float] | None = field(default=None)
    """Bounding box ``(x0, y0, x1, y1)`` of the text region in page coordinates.

    All four values are floats. Invariants enforced by :meth:`validate`:
    ``x0 < x1`` (non-zero width) and ``y0 < y1`` (non-zero height).
    ``None`` for documents without a spatial layout (plain text, audio, etc.).
    """

    # ------------------------------------------------------------------
    # NLP enrichment fields (Issue S-4)
    # ------------------------------------------------------------------

    raw_text: str | None = field(default=None)
    """Verbatim source text before any normalisation or NLP processing.

    Populated by every reader in the corpus pipeline so that the
    before/after transformation can always be compared at the document
    level:

    * :class:`~._readers._image.ImageReader` — exact Tesseract / easyocr
      output bytes before any chunker or NLP step.
    * :class:`~._readers._audio.AudioReader` — pre-LRC-inline-tag-strip
      or pre-VTT-HTML-strip cue text; verbatim Whisper/NeMo ASR output;
      classifier label text (no pre-processing, equals ``text``).
    * :class:`~._readers._video.VideoReader` — pre-HTML-strip SRT/SBV/VTT
      cue text; verbatim Whisper ASR output.
    * :class:`~._readers._pdf.PDFReader` — backend extraction result
      before ``.strip()`` (preserves original page boundary whitespace).
    * :class:`~._readers._text.TextReader` — full file content as read;
      no pre-processing occurs so ``raw_text == text``.
    * :class:`~._readers._xml.XMLReader` /
      :class:`~._readers._xml.TEIReader` — ``itertext()`` join before
      ``_WS_RE`` whitespace collapsing.
    * :class:`~._readers._alto.ALTOReader` — verbatim ALTO ``CONTENT``
      attribute tokens; no additional normalisation, so ``raw_text == text``.
    * :class:`~._readers._web.WebReader` — inner HTML of the matched
      element (tags included) before ``get_text()`` strips them.
    * :class:`~._readers._web.YouTubeReader` — pre-HTML-strip cue text
      from the transcript API (may contain ``<c>`` tags or HTML entities).

    Use this field to compare what each reader returned against:

    * :attr:`text`            — the chunked form (post-chunker, no NLP)
    * :attr:`normalized_text` — the post-:class:`TextNormalizer` form used
                                for embedding

    Three-tier comparison for quality audit::

        raw_text        →  verbatim reader output before any cleaning
        text            →  cleaned / chunked form
        normalized_text →  NFKC + ligature expansion + hyphen-join + whitespace collapse

    Notes
    -----
    For multilingual images, accuracy requires Tesseract to be invoked with
    the correct ``ocr_lang`` string (e.g. ``"eng+deu+ara+heb+tur+ell"``).
    With ``ocr_lang=None`` (the default), Tesseract uses English-only and
    silently transliterates Arabic / Hebrew / Greek glyphs into Latin
    lookalikes. ``raw_text`` then reflects that garbled output, NOT the
    original script — the problem belongs to the pipeline caller, not here.
    """

    normalized_text: str | None = field(default=None)
    """Normalised text used by the embedding engine."""

    tokens: list[str] | None = field(default=None, repr=False, compare=False)
    """Whitespace-tokenised word list for STRICT / KEYWORD matching."""

    lemmas: list[str] | None = field(default=None, repr=False, compare=False)
    """Lemmatised token list."""

    stems: list[str] | None = field(default=None, repr=False, compare=False)
    """Stemmed token list."""

    keywords: list[str] | None = field(default=None, repr=False, compare=False)
    """Extracted keyphrases for topic-level matching."""

    # ------------------------------------------------------------------
    # Multilang / Multi-script fields (Layer 0-3 output)
    # ------------------------------------------------------------------

    script: str | None = field(default=None)
    """Dominant script of this chunk.

    Set to a :class:`~._chunkers._custom_tokenizer.ScriptType` value string
    (e.g. ``"latin"``, ``"arabic"``, ``"han"``). ``None`` if the chunker was
    script-unaware or no script was detected.
    """

    script_direction: str | None = field(default=None)
    """Writing direction of the dominant script.

    One of ``"ltr"`` (left-to-right), ``"rtl"`` (right-to-left), or
    ``"ttb"`` (top-to-bottom, traditional Mongolian). ``None`` if not detected.
    """

    grapheme_count: int | None = field(default=None)
    """Number of grapheme clusters in :attr:`text`.

    This is the correct user-perceived character count as defined by
    Unicode UAX #29.  Always ``<= len(text)`` because each grapheme
    cluster is at least one codepoint. ``None`` if
    :class:`~._normalizers._normalizer.GraphemeClusterNormalizer` has not
    been applied.
    """

    codepoint_count: int | None = field(default=None)
    """Number of Unicode codepoints in :attr:`text`.

    Equal to ``len(text)``. Stored explicitly so downstream consumers can
    compare grapheme vs. codepoint lengths without re-reading the text.
    ``None`` if not computed.
    """

    is_mixed_script: bool | None = field(default=None)
    """``True`` if the chunk contains codepoints from more than one Unicode
    script block above a noise threshold. ``None`` if not analysed.
    """

    script_spans: list | None = field(default=None, repr=False, compare=False)
    """For mixed-script chunks: list of ScriptSpan dicts.

    Each element is a dict::

        {
            "text": str,        # span text (NFC)
            "script": str,      # ScriptType value string
            "direction": str,   # "ltr" | "rtl" | "ttb"
            "start": int,       # grapheme cluster index (inclusive)
            "end": int,         # grapheme cluster index (exclusive)
        }

    Integer indices refer to the grapheme cluster list produced by
    :class:`~._normalizers._normalizer.GraphemeClusterNormalizer`.
    ``None`` for single-script chunks or when script analysis was skipped.
    """

    chunking_unit: str | None = field(default=None)
    """Granularity at which this chunk was produced.

    One of ``"sentence"``, ``"paragraph"``, ``"word"``,
    ``"grapheme_cluster"``, ``"semanteme"``, ``"morpheme"``,
    ``"character"``, ``"fixed_window"``.  ``None`` for legacy chunks
    produced before this field was introduced.
    """

    semanteme_count: int | None = field(default=None)
    """Number of semantemes identified in this chunk.

    Set by :class:`~._chunkers._semantic.SemanticChunker` only.
    ``None`` if semantic chunking was not used.
    """

    morphemes: list[str] | None = field(default=None, repr=False, compare=False)
    """Morpheme list if ``MORPHOLOGICAL`` or ``HYBRID`` backend was used.

    Excluded from ``repr`` and equality comparisons (like ``tokens`` /
    ``lemmas``). ``None`` if semantic chunking was not applied or a
    non-morphological backend was selected.
    """

    determinative_groups: list | None = field(default=None, repr=False, compare=False)
    """For Egyptian hieroglyphic chunks: list of determinative group dicts.

    Each element is a dict::

        {
            "glyphs": str,          # raw glyph codepoints
            "determinative": str,   # semantic category glyph
            "category": str,        # human-readable category label
        }

    ``None`` for all non-hieroglyphic scripts.
    """

    script_model_version: str | None = field(default=None)
    """Version of the embedding or dictionary model used during semantic
    chunking.

    Required for idempotency verification on pipeline re-runs.
    Format: ``"<model_name>@<version>"``, e.g.
    ``"paraphrase-multilingual-mpnet-base-v2@1.2.0"``.
    ``None`` when the ``MORPHOLOGICAL`` backend was used (always idempotent)
    or when semantic chunking was not applied.
    """

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def has_embedding(self) -> bool:
        """
        Return ``True`` if an embedding has been attached to this document.

        Returns
        -------
        bool
            ``True`` when ``embedding`` is not ``None``.

        Examples
        --------
        >>> doc = CorpusDocument.create("f.txt", 0, "Hello.")
        >>> doc.has_embedding
        False
        """
        return self.embedding is not None

    @property
    def word_count(self) -> int:
        """
        Number of whitespace-delimited tokens in :attr:`text`.

        Returns
        -------
        int
            Token count; 0 for empty text.

        Examples
        --------
        >>> doc = CorpusDocument.create("f.txt", 0, "One two three.")
        >>> doc.word_count
        3
        """
        return len(self.text.split()) if self.text is not None else 0

    @property
    def char_count(self) -> int:
        """
        Length of :attr:`text` in characters.

        Returns
        -------
        int
            Character count.
        """
        return len(self.text) if self.text is not None else 0

    def __post_init__(self) -> None:
        """Coerce string-typed enum fields to their proper enum members.

        Called automatically by the dataclass machinery immediately after
        ``__init__``.  Coercion is done here (not in :meth:`validate`) so
        that it works correctly with ``frozen=True`` — ``object.__setattr__``
        is permitted inside ``__post_init__`` even on frozen dataclasses.

        Notes
        -----
        CRITICAL-03 fix: ``CorpusDocument`` is now ``frozen=True``.
        All mutation (including enum coercion) must happen here.
        :meth:`validate` is now pure read-only validation with no side effects.

        Raises
        ------
        ValueError
            If a string value cannot be coerced to its target enum type.
        """
        _coerce_pairs: list[tuple[str, type]] = [
            ("section_type", SectionType),
            ("chunking_strategy", ChunkingStrategy),
            ("source_type", SourceType),
            ("modality", Modality),
        ]
        for field_name, enum_cls in _coerce_pairs:
            val = getattr(self, field_name)
            if val is not None and not isinstance(val, enum_cls):
                try:
                    object.__setattr__(self, field_name, enum_cls(val))
                except ValueError as exc:
                    valid = ", ".join(m.value for m in enum_cls)
                    raise ValueError(
                        f"CorpusDocument.{field_name} must be a"
                        f" {enum_cls.__name__} member; got {val!r}."
                        f" Valid values: {valid}"
                    ) from exc

    # ------------------------------------------------------------------
    # Validation (Issues S-5)
    # ------------------------------------------------------------------

    def validate(self) -> None:  # noqa: PLR0912
        """
        Assert that all invariants hold. Raises on the first violation.

        Raises
        ------
        ValueError
            With an actionable message identifying the violated invariant and
            the offending value.

        Warns
        -----
        UserWarning
            When ``doi`` does not match the ``10.XXXX/`` prefix pattern.
            A warning (not a raise) is used because real-world DOIs are not
            always well-formed, and hard rejection would discard valid papers.

        Notes
        -----
        Call ``validate()`` explicitly after constructing a document via the
        dataclass constructor. The :meth:`create` factory calls it
        automatically.

        Examples
        --------
        >>> doc = CorpusDocument.create("f.txt", 0, "Hello world.")
        >>> doc.validate()  # no exception

        >>> bad = CorpusDocument(
        ...     doc_id="", input_path="f.txt", chunk_index=0, text="Hello."
        ... )
        >>> bad.validate()
        Traceback (most recent call last):
            ...
        ValueError: CorpusDocument.doc_id must be a non-empty string; got ''
        """
        # --- Required string fields must be non-empty ------------------
        for fname in self.REQUIRED_FIELDS:
            val = getattr(self, fname)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(
                    f"CorpusDocument.{fname} must be a non-empty string; got {val!r}"
                )

        # --- text is required for TEXT modality; optional for raw-media ----
        # Raw-media documents (IMAGE, AUDIO, VIDEO) carry content in
        # raw_tensor / raw_bytes and may have text=None (e.g. a video frame
        # before any OCR or transcription has been applied).  MULTIMODAL
        # documents carry both text and a tensor, so text is also required.
        _text_modality = self.modality if isinstance(self.modality, Modality) else None
        _requires_text = _text_modality in (None, Modality.TEXT, Modality.MULTIMODAL)
        if _requires_text:  # noqa: SIM102
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError(
                    f"CorpusDocument.text must be a non-empty string for"
                    f" modality={getattr(self.modality, 'value', self.modality)!r};"
                    f" got {self.text!r}"
                )

        # --- chunk_index must be non-negative --------------------------
        if not isinstance(self.chunk_index, int) or self.chunk_index < 0:
            raise ValueError(
                f"CorpusDocument.chunk_index must be a non-negative int;"
                f" got {self.chunk_index!r}"
            )

        # --- section_type must be a SectionType member -----------------
        if not isinstance(self.section_type, SectionType):
            valid = ", ".join(m.value for m in SectionType)
            raise ValueError(  # noqa: TRY004
                f"CorpusDocument.section_type must be a SectionType"
                f" member; got {self.section_type!r}."
                f" Valid values: {valid}"
            )

        # --- chunking_strategy must be a ChunkingStrategy member -------
        if not isinstance(self.chunking_strategy, ChunkingStrategy):
            valid = ", ".join(m.value for m in ChunkingStrategy)
            raise ValueError(  # noqa: TRY004
                f"CorpusDocument.chunking_strategy must be a"
                f" ChunkingStrategy member; got"
                f" {self.chunking_strategy!r}. Valid values: {valid}"
            )

        # --- source_type must be a SourceType member -------------------
        if not isinstance(self.source_type, SourceType):
            valid = ", ".join(m.value for m in SourceType)
            raise ValueError(  # noqa: TRY004
                f"CorpusDocument.source_type must be a SourceType"
                f" member; got {self.source_type!r}."
                f" Valid values: {valid}"
            )

        # --- char offsets must be consistent when both are present -----
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_start > self.char_end
        ):
            raise ValueError(
                f"CorpusDocument.char_start ({self.char_start}) must be"
                f" <= char_end ({self.char_end})"
            )

        # --- language must be a non-empty string when provided ---------
        if self.language is not None:  # noqa: SIM102
            if not isinstance(self.language, str) or not self.language.strip():
                raise ValueError(
                    f"CorpusDocument.language must be a non-empty string or"
                    f" None; got {self.language!r}"
                )

        # --- metadata keys must all be strings -------------------------
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"CorpusDocument.metadata must be a dict; got"
                f" {type(self.metadata).__name__}"
            )
        bad_keys = [k for k in self.metadata if not isinstance(k, str)]
        if bad_keys:
            raise ValueError(
                f"CorpusDocument.metadata keys must all be strings;"
                f" found non-string keys: {bad_keys!r}"
            )

        # --- page_number, paragraph_index, line_number must be >= 0 ---
        for _int_field in ("page_number", "paragraph_index", "line_number"):
            val = getattr(self, _int_field)
            if val is not None and (not isinstance(val, int) or val < 0):
                raise ValueError(
                    f"CorpusDocument.{_int_field} must be a non-negative"
                    f" int or None; got {val!r}"
                )

        # --- act and scene_number must be >= 1 when set ----------------
        for _one_based in ("act", "scene_number"):
            val = getattr(self, _one_based)
            if val is not None and (not isinstance(val, int) or val < 1):
                raise ValueError(
                    f"CorpusDocument.{_one_based} must be a positive int"
                    f" (>= 1) or None; got {val!r}"
                )

        # --- timecode_start must be >= 0.0 -----------------------------
        if self.timecode_start is not None:  # noqa: SIM102
            if (
                not isinstance(self.timecode_start, (int, float))
                or self.timecode_start < 0.0
            ):
                raise ValueError(
                    f"CorpusDocument.timecode_start must be a non-negative"
                    f" float or None; got {self.timecode_start!r}"
                )

        # --- timecode_end must be >= timecode_start when both set ------
        if self.timecode_end is not None:
            if (
                not isinstance(self.timecode_end, (int, float))
                or self.timecode_end < 0.0
            ):
                raise ValueError(
                    f"CorpusDocument.timecode_end must be a non-negative"
                    f" float or None; got {self.timecode_end!r}"
                )
            if (
                self.timecode_start is not None
                and self.timecode_end < self.timecode_start
            ):
                raise ValueError(
                    f"CorpusDocument.timecode_end ({self.timecode_end})"
                    f" must be >= timecode_start ({self.timecode_start})"
                )

        # --- confidence must be in [0.0, 1.0] when set -----------------
        if self.confidence is not None:  # noqa: SIM102
            if not isinstance(self.confidence, (int, float)) or not (
                0.0 <= self.confidence <= 1.0
            ):
                raise ValueError(
                    f"CorpusDocument.confidence must be a float in"
                    f" [0.0, 1.0] or None; got {self.confidence!r}"
                )

        # --- bbox must be a 4-tuple of floats with x0<x1 and y0<y1 ----------
        if self.bbox is not None:
            if (
                not isinstance(self.bbox, tuple)
                or len(self.bbox) != 4  # noqa: PLR2004
                or not all(isinstance(v, (int, float)) for v in self.bbox)
            ):
                raise ValueError(
                    f"CorpusDocument.bbox must be a 4-tuple of floats"
                    f" (x0, y0, x1, y1) or None; got {self.bbox!r}"
                )
            x0, y0, x1, y1 = self.bbox
            if x0 >= x1:
                raise ValueError(
                    f"CorpusDocument.bbox invariant violated: x0 ({x0}) must"
                    f" be < x1 ({x1}); got bbox={self.bbox!r}"
                )
            if y0 >= y1:
                raise ValueError(
                    f"CorpusDocument.bbox invariant violated: y0 ({y0}) must"
                    f" be < y1 ({y1}); got bbox={self.bbox!r}"
                )

        # --- doi: warn (not raise) on suspicious format ----------------
        if self.doi is not None and not _DOI_PREFIX_RE.match(self.doi):
            warnings.warn(
                f"CorpusDocument.doi {self.doi!r} does not start with"
                f" '10.XXXX/' — this may not be a valid DOI.",
                UserWarning,
                stacklevel=2,
            )

        # MEDIUM-06 fix: warn when input_path is absolute — absolute paths break
        # corpus portability across machines and environments.
        # We use PurePosixPath for the check so Windows absolute paths like
        # "C:\\..." are also caught (they don't start with "/", but
        # pathlib.Path.is_absolute() is cross-platform and handles both).
        if self.input_path and pathlib.Path(self.input_path).is_absolute():
            warnings.warn(
                f"CorpusDocument.input_path {self.input_path!r} is an absolute "
                f"path. Absolute paths break corpus portability across machines. "
                f"Use input_path.name or a relative path instead.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def make_doc_id(
        cls,
        input_path: str,
        chunk_index: int,
        text: str,
        source_type: SourceType = SourceType.UNKNOWN,
    ) -> str:
        """
        Compute a deterministic 16-character hex document identifier.

        The id is a SHA-1 prefix of
        ``"{source_type}:{input_path}:{chunk_index}:{text[:64]}"``.
        Identical inputs always produce the same id.

        Parameters
        ----------
        input_path : str
            Name of the source file (not a full path).
        chunk_index : int
            Zero-based chunk position within the document.
        text : str
            Raw text content of the chunk (only the first 64 characters
            are used to keep hashing fast).
        source_type : SourceType, optional
            Source kind. Including this in the hash preimage prevents
            collisions when a BOOK chapter and a MOVIE subtitle share the
            same filename, chunk index, and opening text (Issue S-7).
            Default: ``SourceType.UNKNOWN``.

        Returns
        -------
        str
            16-character lowercase hexadecimal string.

        Notes
        -----
        Adding ``source_type`` to the hash preimage is a one-time breaking
        change for corpora built before this version. Existing corpora must
        be re-indexed when upgrading.

        Examples
        --------
        >>> CorpusDocument.make_doc_id("file.txt", 0, "Hello world.")
        '...'  # deterministic 16-char hex
        >>> (
        ...     CorpusDocument.make_doc_id("f.txt", 0, "Hi", SourceType.BOOK)
        ...     != CorpusDocument.make_doc_id("f.txt", 0, "Hi", SourceType.MOVIE)
        ... )
        True
        """
        st_val = (
            source_type.value
            if isinstance(source_type, SourceType)
            else str(source_type)
        )
        # Guard against text=None: raw-media documents (IMAGE, VIDEO, AUDIO
        # modalities) legitimately have no text.  Use an empty string prefix
        # so the hash is still deterministic and unique per (source, chunk).
        text_prefix = text[:64] if text is not None else ""
        raw = f"{st_val}:{input_path}:{chunk_index}:{text_prefix}"
        return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]

    @staticmethod
    def make_content_hash(
        text: str | None = None,
        raw_bytes: bytes | None = None,
    ) -> str:
        """
        Compute a 32-char SHA-256 hex digest for deduplication.

        Parameters
        ----------
        text : str or None
            Text content. Used when ``raw_bytes`` is ``None``.
        raw_bytes : bytes or None
            Raw media bytes. Preferred over ``text`` when set.

        Returns
        -------
        str
            32-character hex SHA-256 prefix.

        Notes
        -----
        Empty / ``None`` inputs return a fixed sentinel value
        ``"0" * 32`` (32 zeros) to ensure ``content_hash`` is always
        populated and the dedup logic is deterministic.
        """
        if raw_bytes is not None:
            return hashlib.sha256(raw_bytes).hexdigest()[:32]
        if text is not None:
            return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[
                :32
            ]
        return "0" * 32

    @classmethod
    def create(  # noqa: D417
        cls,
        input_path: str,
        chunk_index: int,
        text: str | None,
        # Core classification
        section_type: SectionType = SectionType.TEXT,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.NONE,
        language: str | None = None,
        # Character offsets
        char_start: int | None = None,
        char_end: int | None = None,
        # Embedding
        embedding: Any | None = None,
        # Ad-hoc metadata
        metadata: dict[str, Any] | None = None,
        # Explicit doc_id override
        doc_id: str | None = None,
        # Provenance (Issue S-4)
        source_type: SourceType = SourceType.UNKNOWN,
        source_title: str | None = None,
        source_author: str | None = None,
        source_date: str | None = None,
        collection_id: str | None = None,
        url: str | None = None,
        doi: str | None = None,
        isbn: str | None = None,
        # Position (Issue S-4)
        page_number: int | None = None,
        paragraph_index: int | None = None,
        line_number: int | None = None,
        parent_doc_id: str | None = None,
        # Dramatic position (Issue S-4)
        act: int | None = None,
        scene_number: int | None = None,
        # Media-specific (Issue S-4)
        timecode_start: float | None = None,
        timecode_end: float | None = None,
        confidence: float | None = None,
        ocr_engine: str | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        # NLP enrichment (Issue S-4)
        normalized_text: str | None = None,
        raw_text: str | None = None,
        tokens: list[str] | None = None,
        lemmas: list[str] | None = None,
        stems: list[str] | None = None,
        keywords: list[str] | None = None,
        # Raw media fields (new)
        modality: Modality | None = None,
        raw_bytes: bytes | None = None,
        raw_tensor: Any = None,
        raw_shape: tuple[int, ...] | None = None,
        raw_dtype: str | None = None,
        frame_index: int | None = None,
        content_hash: str | None = None,
        # Multilang / multi-script fields (from MultilangMixin)
        script: str | None = None,
        script_direction: str | None = None,
        grapheme_count: int | None = None,
        codepoint_count: int | None = None,
        is_mixed_script: bool | None = None,
        script_spans: list | None = None,
        chunking_unit: str | None = None,
        semanteme_count: int | None = None,
        morphemes: list[str] | None = None,
        determinative_groups: list | None = None,
        script_model_version: str | None = None,
    ) -> CorpusDocument:
        """
        Validate factory constructor for :class:`CorpusDocument`.

        Preferred over direct dataclass instantiation because it
        auto-generates ``doc_id`` when not supplied and calls
        :meth:`validate` before returning.

        Parameters
        ----------
        input_path : str
            Name of the source file.
        chunk_index : int
            Zero-based chunk position.
        text : str
            Text content of the chunk.
        section_type : SectionType, optional
            Semantic section label. Default: ``SectionType.TEXT``.
        chunking_strategy : ChunkingStrategy, optional
            Segmentation strategy used. Default: ``ChunkingStrategy.NONE``.
        language : str or None, optional
            ISO 639-1 language code. Default: ``None``.
        char_start : int or None, optional
            Character start offset. Default: ``None``.
        char_end : int or None, optional
            Character end offset (exclusive). Default: ``None``.
        embedding : array-like or None, optional
            Pre-computed embedding vector. Default: ``None``.
        metadata : dict or None, optional
            Ad-hoc metadata. ``None`` is treated as empty dict. Default:
            ``None``.
        doc_id : str or None, optional
            Explicit document id. Auto-generated if ``None``. Default: ``None``.
        source_type : SourceType, optional
            Kind of source. Default: ``SourceType.UNKNOWN``.
        source_title : str or None, optional
            Title of the source work. Default: ``None``.
        source_author : str or None, optional
            Primary author. Default: ``None``.
        source_date : str or None, optional
            Publication date (ISO 8601). Default: ``None``.
        collection_id : str or None, optional
            Corpus collection identifier. Default: ``None``.
        url : str or None, optional
            Source URL. Default: ``None``.
        doi : str or None, optional
            Digital Object Identifier. Default: ``None``.
        isbn : str or None, optional
            International Standard Book Number. Default: ``None``.
        page_number : int or None, optional
            Zero-based page index. Default: ``None``.
        paragraph_index : int or None, optional
            Zero-based paragraph index. Default: ``None``.
        line_number : int or None, optional
            Zero-based line number. Default: ``None``.
        parent_doc_id : str or None, optional
            doc_id of parent chunk. Default: ``None``.
        act : int or None, optional
            Act number (one-based). Default: ``None``.
        scene_number : int or None, optional
            Scene number (one-based). Default: ``None``.
        timecode_start : float or None, optional
            Start timecode in seconds (>= 0). Default: ``None``.
        timecode_end : float or None, optional
            End timecode in seconds. Default: ``None``.
        confidence : float or None, optional
            OCR/ASR confidence in [0.0, 1.0]. Default: ``None``.
        ocr_engine : str or None, optional
            OCR engine name. Default: ``None``.
        bbox : tuple of float or None, optional
            Bounding box (x0, y0, x1, y1). Default: ``None``.
        normalized_text : str or None, optional
            Pre-normalised text. Default: ``None``.
        tokens : list of str or None, optional
            Tokenised words. Default: ``None``.
        lemmas : list of str or None, optional
            Lemmatised tokens. Default: ``None``.
        stems : list of str or None, optional
            Stemmed tokens. Default: ``None``.
        keywords : list of str or None, optional
            Extracted keyphrases. Default: ``None``.

        Returns
        -------
        CorpusDocument
            Validated document instance.

        Raises
        ------
        ValueError
            If any invariant from :meth:`validate` is violated.

        Examples
        --------
        >>> doc = CorpusDocument.create(
        ...     input_path="corpus.txt",
        ...     chunk_index=0,
        ...     text="Hello world.",
        ...     source_type=SourceType.BOOK,
        ...     language="en",
        ... )
        >>> doc.validate()
        >>> doc.has_embedding
        False
        """
        resolved_id = doc_id or cls.make_doc_id(
            input_path, chunk_index, text, source_type
        )
        instance = cls(
            doc_id=resolved_id,
            input_path=input_path,
            chunk_index=chunk_index,
            text=text,
            section_type=section_type,
            chunking_strategy=chunking_strategy,
            language=language,
            char_start=char_start,
            char_end=char_end,
            embedding=embedding,
            metadata=metadata if metadata is not None else {},
            source_type=source_type,
            source_title=source_title,
            source_author=source_author,
            source_date=source_date,
            collection_id=collection_id,
            url=url,
            doi=doi,
            isbn=isbn,
            page_number=page_number,
            paragraph_index=paragraph_index,
            line_number=line_number,
            parent_doc_id=parent_doc_id,
            act=act,
            scene_number=scene_number,
            timecode_start=timecode_start,
            timecode_end=timecode_end,
            confidence=confidence,
            ocr_engine=ocr_engine,
            bbox=bbox,
            normalized_text=normalized_text,
            raw_text=raw_text,
            tokens=tokens,
            lemmas=lemmas,
            stems=stems,
            keywords=keywords,
            # Raw media
            modality=modality
            or (
                Modality.TEXT
                if raw_tensor is None and raw_bytes is None
                else Modality.IMAGE
            ),
            raw_bytes=raw_bytes,
            raw_tensor=raw_tensor,
            raw_shape=(
                raw_shape
                if raw_shape is not None
                else (
                    tuple(raw_tensor.shape)
                    if raw_tensor is not None and hasattr(raw_tensor, "shape")
                    else None
                )
            ),
            raw_dtype=(
                raw_dtype
                if raw_dtype is not None
                else (
                    str(raw_tensor.dtype)
                    if raw_tensor is not None and hasattr(raw_tensor, "dtype")
                    else None
                )
            ),
            frame_index=frame_index,
            content_hash=content_hash
            or cls.make_content_hash(text=text, raw_bytes=raw_bytes),
            # Multilang / multi-script fields
            script=script,
            script_direction=script_direction,
            grapheme_count=grapheme_count,
            codepoint_count=codepoint_count,
            is_mixed_script=is_mixed_script,
            script_spans=script_spans,
            chunking_unit=chunking_unit,
            semanteme_count=semanteme_count,
            morphemes=morphemes,
            determinative_groups=determinative_groups,
            script_model_version=script_model_version,
        )
        instance.validate()
        return instance

    # ------------------------------------------------------------------
    # Mutation (copy-on-write)
    # ------------------------------------------------------------------

    def replace(self, **changes: Any) -> Self:
        """
        Return a new :class:`CorpusDocument` with the specified fields
        replaced.

        Parameters
        ----------
        **changes : Any
            Field names and new values. Only fields defined on
            :class:`CorpusDocument` are accepted.

        Returns
        -------
        CorpusDocument
            New instance with changed fields; original is unchanged.

        Raises
        ------
        ValueError
            If an unknown field name is given.

        Examples
        --------
        >>> import numpy as np
        >>> doc = CorpusDocument.create("f.txt", 0, "Hello.")
        >>> enriched = doc.replace(embedding=np.zeros(768, dtype=np.float32))
        >>> enriched.has_embedding
        True
        >>> doc.has_embedding  # original unchanged
        False
        """  # noqa: D205
        valid_fields = {f.name for f in fields(self)}
        unknown = set(changes) - valid_fields
        if unknown:
            raise ValueError(
                f"CorpusDocument.replace() received unknown field(s):"
                f" {unknown!r}."
                f" Valid fields: {sorted(valid_fields)}"
            )
        # CRITICAL-03: now that CorpusDocument is frozen=True, use
        # _dc_replace (dataclasses.replace) which correctly creates a new
        # frozen instance and re-runs __post_init__ for enum coercion.
        # Deep-copy mutable containers not being replaced so each call site
        # gets an independent copy with no shared-reference bugs.
        if "metadata" not in changes:
            changes["metadata"] = copy.copy(self.metadata)
        for _list_field in ("tokens", "lemmas", "stems", "keywords"):
            if _list_field not in changes:
                existing = getattr(self, _list_field)
                if existing is not None:
                    changes[_list_field] = list(existing)
        return _dc_replace(self, **changes)

    # ------------------------------------------------------------------
    # Serialisation / conversion (Issues S-6)
    # ------------------------------------------------------------------

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """
        Serialise to a plain Python dictionary.

        Parameters
        ----------
        include_embedding : bool, optional
            When ``True``, include the ``embedding`` field serialised as a
            flat list of floats (if present). Default: ``False`` — embeddings
            are excluded to keep dicts JSON-safe by default.

        Returns
        -------
        dict
            Shallow copy of all fields. Enum fields serialised as string
            values. ``bbox`` serialised as a list (JSON-compatible).
            ``metadata`` is a shallow copy.

        Notes
        -----
        This method does **not** call :meth:`validate` — it is designed to
        be fast and usable even on partially-constructed documents during
        debugging.

        Examples
        --------
        >>> doc = CorpusDocument.create("f.txt", 0, "Hello.")
        >>> d = doc.to_dict()
        >>> isinstance(d["section_type"], str)
        True
        >>> d["source_type"]
        'unknown'
        """
        d: dict[str, Any] = {
            # Core
            "doc_id": self.doc_id,
            "input_path": self.input_path,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "section_type": self.section_type.value,
            "chunking_strategy": self.chunking_strategy.value,
            "language": self.language,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": copy.copy(self.metadata),
            # Provenance
            "source_type": self.source_type.value,
            "source_title": self.source_title,
            "source_author": self.source_author,
            "source_date": self.source_date,
            "collection_id": self.collection_id,
            "url": self.url,
            "doi": self.doi,
            "isbn": self.isbn,
            # Position
            "page_number": self.page_number,
            "paragraph_index": self.paragraph_index,
            "line_number": self.line_number,
            "parent_doc_id": self.parent_doc_id,
            # Dramatic position
            "act": self.act,
            "scene_number": self.scene_number,
            # Media-specific
            "timecode_start": self.timecode_start,
            "timecode_end": self.timecode_end,
            "confidence": self.confidence,
            "ocr_engine": self.ocr_engine,
            # bbox → list for JSON/CSV compatibility; None stays None
            "bbox": list(self.bbox) if self.bbox is not None else None,
            # NLP enrichment — lists are copied for isolation
            "normalized_text": self.normalized_text,
            "raw_text": self.raw_text,
            "tokens": list(self.tokens) if self.tokens is not None else None,
            "lemmas": list(self.lemmas) if self.lemmas is not None else None,
            "stems": list(self.stems) if self.stems is not None else None,
            "keywords": list(self.keywords) if self.keywords is not None else None,
            # Multilang / multi-script fields (Layer 0-3)
            "script": self.script,
            "script_direction": self.script_direction,
            "grapheme_count": self.grapheme_count,
            "codepoint_count": self.codepoint_count,
            "is_mixed_script": self.is_mixed_script,
            "script_spans": (
                list(self.script_spans) if self.script_spans is not None else None
            ),
            "chunking_unit": self.chunking_unit,
            "semanteme_count": self.semanteme_count,
            "morphemes": list(self.morphemes) if self.morphemes is not None else None,
            "determinative_groups": (
                list(self.determinative_groups)
                if self.determinative_groups is not None
                else None
            ),
            "script_model_version": self.script_model_version,
            # Raw media — modality and lightweight scalar fields are always
            # serialised; raw_tensor and raw_bytes are excluded (too large /
            # not JSON-safe; callers that need them work with the object directly).
            "modality": (
                self.modality.value
                if isinstance(self.modality, Modality)
                else self.modality
            ),
            "frame_index": self.frame_index,
            "raw_shape": list(self.raw_shape) if self.raw_shape is not None else None,
            "raw_dtype": self.raw_dtype,
            "content_hash": self.content_hash,
        }
        if include_embedding and self.embedding is not None:
            try:
                d["embedding"] = [  # type: ignore[union-attr]
                    float(v) for v in self.embedding
                ]
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "CorpusDocument.to_dict: could not serialise embedding"
                    " to list; omitting. Cause: %s",
                    exc,
                )
        return d

    def to_flat_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """
        Serialise to a flat dictionary with metadata fields promoted to the
        top level.

        Unlike :meth:`to_dict`, the ``metadata`` sub-dict is merged into
        the top level. Core fields take precedence over metadata fields with
        the same key name.

        Parameters
        ----------
        include_embedding : bool, optional
            When ``True``, include ``embedding`` as a list of floats.
            Default: ``False``.

        Returns
        -------
        dict
            Flat dict suitable for a single row in a tabular export.

        Notes
        -----
        Metadata key collisions with core fields are logged as warnings.

        Examples
        --------
        >>> doc = CorpusDocument.create(
        ...     "f.txt", 0, "Hello.", metadata={"custom_key": "v"}
        ... )
        >>> flat = doc.to_flat_dict()
        >>> flat["custom_key"]
        'v'
        """  # noqa: D205
        core = self.to_dict(include_embedding=include_embedding)
        meta = core.pop("metadata", {})
        overlap = set(meta) & set(core)
        if overlap:
            logger.warning(
                "CorpusDocument.to_flat_dict: metadata keys %r overlap with"
                " core fields; core values take precedence.",
                sorted(overlap),
            )
        return {**meta, **core}

    def to_pandas_row(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """
        Return a dict formatted for a single row in a ``pandas.DataFrame``.

        Parameters
        ----------
        include_embedding : bool, optional
            When ``True``, include the embedding as a numpy array (not a
            list), allowing ``pandas`` to store it as an object column.
            Default: ``False``.

        Returns
        -------
        dict
            Row dict with enums as strings. Embedding kept as-is when
            present and ``include_embedding=True``.

        Examples
        --------
        >>> import pandas as pd
        >>> doc = CorpusDocument.create("f.txt", 0, "Hello.")
        >>> row = doc.to_pandas_row()
        >>> pd.DataFrame([row])["text"][0]
        'Hello.'
        """
        row = self.to_flat_dict(include_embedding=False)
        if include_embedding and self.embedding is not None:
            row["embedding"] = self.embedding
        return row

    def to_polars_row(self, *, include_embedding: bool = False) -> dict[str, Any]:
        """
        Return a dict formatted for a single row in a ``polars.DataFrame``.

        Parameters
        ----------
        include_embedding : bool, optional
            When ``True``, include the embedding as a list of floats (polars
            does not accept numpy arrays directly in dict-based construction).
            Default: ``False``.

        Returns
        -------
        dict
            Row dict. Embedding serialised as ``list[float]`` when present.

        Examples
        --------
        >>> import polars as pl
        >>> doc = CorpusDocument.create("f.txt", 0, "Hello.")
        >>> pl.DataFrame([doc.to_polars_row()])["text"][0]
        'Hello.'
        """
        return self.to_flat_dict(include_embedding=include_embedding)

    # ------------------------------------------------------------------
    # Classmethods for bulk construction
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """
        Reconstruct a :class:`CorpusDocument` from a plain dictionary.

        Parameters
        ----------
        data : dict
            Dictionary as returned by :meth:`to_dict`. Enum fields are
            coerced from string values. ``bbox`` is restored from list to
            tuple. ``metadata`` defaults to empty dict if absent.

        Returns
        -------
        CorpusDocument
            Validated reconstructed document.

        Raises
        ------
        ValueError
            If required fields are missing or values are invalid.

        Examples
        --------
        >>> doc = CorpusDocument.create("f.txt", 0, "Hello.")
        >>> d = doc.to_dict()
        >>> restored = CorpusDocument.from_dict(d)
        >>> restored.doc_id == doc.doc_id
        True
        """
        # text is optional for raw-media documents (modality != TEXT).
        # The required-keys check only enforces the fields that must always
        # be present regardless of modality.
        required_keys = {"doc_id", "input_path", "chunk_index"}
        missing = required_keys - set(data)
        if missing:
            raise ValueError(
                f"CorpusDocument.from_dict: missing required keys: {sorted(missing)}"
            )

        # Restore bbox from list/tuple → tuple[float, ...]
        raw_bbox = data.get("bbox")
        bbox: tuple[float, float, float, float] | None = (
            tuple(float(v) for v in raw_bbox) if raw_bbox is not None else None  # type: ignore[assignment]
        )

        # Restore raw_shape from list → tuple[int, ...]
        raw_shape_raw = data.get("raw_shape")
        raw_shape: tuple[int, ...] | None = (
            tuple(int(v) for v in raw_shape_raw) if raw_shape_raw is not None else None
        )

        # Restore modality enum from stored string value (None → auto-detected
        # by create() from raw_tensor / raw_bytes presence).
        raw_modality = data.get("modality")
        modality: Modality | None = (
            Modality(raw_modality) if raw_modality is not None else None
        )

        # Helper: restore optional list[str] fields
        def _restore_str_list(key: str) -> list[str] | None:
            """Restore an optional ``list[str]`` field from a raw dict."""
            val = data.get(key)
            return [str(s) for s in val] if val is not None else None

        return cls.create(
            input_path=data["input_path"],
            chunk_index=int(data["chunk_index"]),
            text=data.get("text"),
            section_type=SectionType(data.get("section_type", SectionType.TEXT.value)),
            chunking_strategy=ChunkingStrategy(
                data.get("chunking_strategy", ChunkingStrategy.NONE.value)
            ),
            language=data.get("language"),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            embedding=data.get("embedding"),
            metadata=data.get("metadata") or {},
            doc_id=data["doc_id"],
            # Provenance
            source_type=SourceType(data.get("source_type", SourceType.UNKNOWN.value)),
            source_title=data.get("source_title"),
            source_author=data.get("source_author"),
            source_date=data.get("source_date"),
            collection_id=data.get("collection_id"),
            url=data.get("url"),
            doi=data.get("doi"),
            isbn=data.get("isbn"),
            # Position
            page_number=data.get("page_number"),
            paragraph_index=data.get("paragraph_index"),
            line_number=data.get("line_number"),
            parent_doc_id=data.get("parent_doc_id"),
            # Dramatic position
            act=data.get("act"),
            scene_number=data.get("scene_number"),
            # Media-specific
            timecode_start=data.get("timecode_start"),
            timecode_end=data.get("timecode_end"),
            confidence=data.get("confidence"),
            ocr_engine=data.get("ocr_engine"),
            bbox=bbox,
            # NLP enrichment
            normalized_text=data.get("normalized_text"),
            raw_text=data.get("raw_text"),
            tokens=_restore_str_list("tokens"),
            lemmas=_restore_str_list("lemmas"),
            stems=_restore_str_list("stems"),
            keywords=_restore_str_list("keywords"),
            # Multilang / multi-script fields (Layer 0-3)
            script=data.get("script"),
            script_direction=data.get("script_direction"),
            grapheme_count=data.get("grapheme_count"),
            codepoint_count=data.get("codepoint_count"),
            is_mixed_script=data.get("is_mixed_script"),
            script_spans=data.get("script_spans"),
            chunking_unit=data.get("chunking_unit"),
            semanteme_count=data.get("semanteme_count"),
            morphemes=_restore_str_list("morphemes"),
            determinative_groups=data.get("determinative_groups"),
            script_model_version=data.get("script_model_version"),
            # Raw media
            modality=modality,
            frame_index=data.get("frame_index"),
            raw_shape=raw_shape,
            raw_dtype=data.get("raw_dtype"),
            content_hash=data.get("content_hash"),
        )

    # ------------------------------------------------------------------
    # Dunder overrides
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise human-readable representation.

        Returns
        -------
        str
            ``CorpusDocument(input_path=..., chunk_index=..., words=...[, emb=...])``.
        """
        emb_info = (
            f", embedding=<array shape={getattr(self.embedding, 'shape', '?')}>"
            if self.has_embedding
            else ""
        )
        return (
            f"CorpusDocument("
            f"doc_id={self.doc_id!r}, "
            f"input_path={self.input_path!r}, "
            f"chunk_index={self.chunk_index}, "
            f"source_type={self.source_type.value!r}, "
            f"section_type={self.section_type.value!r}, "
            f"words={self.word_count}"
            f"{emb_info}"
            f")"
        )


# ===========================================================================
# Module-level helpers
# ===========================================================================


def documents_to_pandas(
    docs: list[CorpusDocument],
    *,
    include_embedding: bool = False,
) -> pd.DataFrame:
    """
    Convert a list of :class:`CorpusDocument` instances to a
    ``pandas.DataFrame``.

    Parameters
    ----------
    docs : list of CorpusDocument
        Documents to convert. Must be non-empty.
    include_embedding : bool, optional
        When ``True``, include a column ``"embedding"`` with numpy arrays.
        Default: ``False``.

    Returns
    -------
    pandas.DataFrame
        One row per document. Metadata fields are promoted to columns.
        An empty DataFrame with schema columns is returned when ``docs`` is
        empty rather than raising — an empty corpus is a valid pipeline result.

    Raises
    ------
    ImportError
        If ``pandas`` is not installed.

    Examples
    --------
    >>> docs = [CorpusDocument.create("f.txt", i, f"Sentence {i}.") for i in range(3)]
    >>> df = documents_to_pandas(docs)
    >>> len(df)
    3
    """  # noqa: D205
    # Validate non-empty BEFORE attempting any library import so the
    # error is deterministic regardless of whether pandas is installed.
    if not docs:
        raise ValueError(
            "documents_to_pandas() requires a non-empty list of CorpusDocument "
            "objects.  An empty corpus has no schema to infer — pass at least one "
            "document, or check len(documents) before calling."
        )
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pandas is required for documents_to_pandas()."
            " Install it with: pip install pandas"
        ) from exc
    # Legacy dead-code guard (unreachable after the ValueError above) kept
    # for grep-safety — remove in 0.6.0 together with the surrounding refactor.
    if not docs:  # pragma: no cover  # noqa: RET505
        _empty_columns = [
            "doc_id",
            "input_path",
            "chunk_index",
            "text",
            "raw_text",
            "normalized_text",
            "section_type",
            "chunking_strategy",
            "language",
            "source_type",
            "source_title",
            "source_author",
        ]
        return pd.DataFrame(columns=_empty_columns)
    rows = [d.to_pandas_row(include_embedding=include_embedding) for d in docs]
    return pd.DataFrame(rows)


def documents_to_polars(
    docs: list[CorpusDocument],
    *,
    include_embedding: bool = False,
) -> pl.DataFrame:
    """
    Convert a list of :class:`CorpusDocument` instances to a
    ``polars.DataFrame``.

    Parameters
    ----------
    docs : list of CorpusDocument
        Documents to convert. Must be non-empty.
    include_embedding : bool, optional
        When ``True``, include a column ``"embedding"`` with list-of-float
        values. Default: ``False``.

    Returns
    -------
    polars.DataFrame
        One row per document. Metadata fields are promoted to columns.
        An empty DataFrame with schema columns is returned when ``docs`` is
        empty rather than raising — an empty corpus is a valid pipeline result.

    Raises
    ------
    ImportError
        If ``polars`` is not installed.

    Examples
    --------
    >>> docs = [CorpusDocument.create("f.txt", i, f"Sentence {i}.") for i in range(3)]
    >>> df = documents_to_polars(docs)
    >>> len(df)
    3
    """  # noqa: D205
    # Validate non-empty BEFORE attempting the polars import so the error
    # is deterministic regardless of whether polars is installed.
    if not docs:
        raise ValueError(
            "documents_to_polars() requires a non-empty list of CorpusDocument "
            "objects.  An empty corpus has no schema to infer — pass at least one "
            "document, or check len(documents) before calling."
        )
    try:
        import polars as pl  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "polars is required for documents_to_polars()."
            " Install it with: pip install polars"
        ) from exc
    # Legacy dead-code guard (unreachable after the ValueError above) kept
    # for grep-safety — remove in 0.6.0.
    if not docs:  # pragma: no cover  # noqa: RET505
        _empty_columns = [
            "doc_id",
            "input_path",
            "chunk_index",
            "text",
            "raw_text",
            "normalized_text",
            "section_type",
            "chunking_strategy",
            "language",
            "source_type",
            "source_title",
            "source_author",
        ]
        return pl.DataFrame(schema=dict.fromkeys(_empty_columns, pl.Utf8))
    rows = [d.to_polars_row(include_embedding=include_embedding) for d in docs]
    return pl.DataFrame(rows)
