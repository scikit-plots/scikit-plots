# scikitplot/corpus/_readers/_custom.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._readers._custom
===================================
Fully user-customizable document reader for the corpus pipeline.

This module provides two public objects:

:func:`normalize_extractor_output`
    A pure utility function that coerces the return value of any user-supplied
    extractor callable into the canonical ``list[dict[str, Any]]`` format
    expected by the corpus pipeline.  Shared by :class:`CustomReader` and the
    ``custom_extractor`` hooks in all built-in readers.

:class:`CustomReader`
    A :class:`~scikitplot.corpus._base.DocumentReader` subclass that accepts
    *any* file extension and a caller-supplied extraction callable.  Enables
    users to integrate arbitrary third-party or proprietary extraction
    libraries — ``pdfplumber``, ``surya``, ``docling``, ``whisperX``,
    proprietary ASR/OCR APIs, in-memory buffers — without subclassing the
    full :class:`DocumentReader` contract.

Extractor callable contract
----------------------------
Every extractor function passed to :class:`CustomReader` or to the
``custom_extractor`` parameter of the built-in readers **must** accept a
:class:`pathlib.Path` as its first positional argument and may accept
additional keyword arguments.  It must return one of:

* ``str`` — entire resource as a single text chunk.
* ``list[str]`` — one string per logical segment (page, frame, cue …).
* ``dict`` — single chunk with text **and** metadata.  Must contain a
  ``"text"`` key mapping to a ``str``.
* ``list[dict]`` — multiple chunks with text and metadata.  Every dict
  must contain a ``"text"`` key.

Optional metadata keys recognised by the corpus schema
(``page_number``, ``timecode_start``, ``confidence``, ``ocr_engine``, …)
are promoted to first-class :class:`~scikitplot.corpus._schema.CorpusDocument`
fields by the downstream pipeline.  Unrecognised keys land in
``CorpusDocument.metadata``.

Python compatibility
--------------------
Python 3.8-3.15.  Zero runtime imports at module level.  All optional
dependencies are imported lazily inside ``get_raw_chunks``.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401
from typing import (  # noqa: F401
    Any,
    Callable,
    ClassVar,
    Dict,
    Generator,
    List,
    Optional,
    Union,
)

from .._base import DocumentReader
from .._schema import SectionType, SourceType

logger = logging.getLogger(__name__)

__all__ = [
    "CustomReader",
    "normalize_extractor_output",
]

# ---------------------------------------------------------------------------
# Type alias for the extractor return value
# ---------------------------------------------------------------------------

#: Union type accepted by :func:`normalize_extractor_output`.
ExtractorOutput = Union[
    str,
    "list[str]",
    "dict[str, Any]",
    "list[dict[str, Any]]",
]


# ===========================================================================
# normalize_extractor_output — shared pure-function utility
# ===========================================================================


def normalize_extractor_output(
    raw: Any,
    *,
    source_type: SourceType = SourceType.UNKNOWN,
    section_type: SectionType = SectionType.TEXT,
) -> list[dict[str, Any]]:
    r"""
    Coerce an extractor return value to a list of raw chunk dicts.

    Every dict in the returned list is guaranteed to contain a ``"text"``
    key.  Missing ``"section_type"`` and ``"source_type"`` keys are filled
    with the supplied defaults.

    Parameters
    ----------
    raw : str, list[str], dict, or list[dict]
        Value returned by a user-supplied extractor callable.  Supported
        types:

        ``str``
            Entire resource as a single text chunk.
        ``list[str]``
            Multiple text chunks.  All elements must be ``str``.
        ``dict``
            Single chunk with text and optional metadata.  Must contain a
            ``"text"`` key whose value is a ``str``.
        ``list[dict]``
            Multiple chunks.  Every element must be a ``dict`` with a
            ``"text"`` key.

    source_type : SourceType, optional
        Default source type injected into chunks that do not specify
        ``"source_type"``.
        Default: :attr:`~scikitplot.corpus._schema.SourceType.UNKNOWN`.
    section_type : SectionType, optional
        Default section type injected into chunks that do not specify
        ``"section_type"``.
        Default: :attr:`~scikitplot.corpus._schema.SectionType.TEXT`.

    Returns
    -------
    list of dict
        Normalised list of raw chunk dicts, each containing at minimum
        ``{"text": str}``.  The list may be empty when ``raw`` is an empty
        list.

    Raises
    ------
    TypeError
        If ``raw`` is not one of the four supported types, or if a list
        contains a mix of ``str`` and non-``str`` elements, or if a list
        element is neither ``str`` nor ``dict``.
    ValueError
        If any ``dict`` in ``raw`` is missing a ``"text"`` key, or if the
        ``"text"`` value is not a ``str``.

    Notes
    -----
    This function is intentionally *pure* (no side effects, deterministic).
    It is called by :class:`CustomReader` and by the ``custom_extractor``
    dispatch branches of :class:`~scikitplot.corpus._readers.PDFReader`,
    :class:`~scikitplot.corpus._readers.ImageReader`,
    :class:`~scikitplot.corpus._readers.AudioReader`, and
    :class:`~scikitplot.corpus._readers.VideoReader`.

    Examples
    --------
    Single string → one-element list:

    >>> from scikitplot.corpus._readers._custom import normalize_extractor_output
    >>> normalize_extractor_output("Hello world")
    [{'text': 'Hello world', 'section_type': 'text', 'source_type': 'unknown'}]

    List of strings → list of chunk dicts:

    >>> normalize_extractor_output(["Page one", "Page two"])
    [{'text': 'Page one', ...}, {'text': 'Page two', ...}]

    Dict with extra metadata preserved:

    >>> normalize_extractor_output({"text": "Hello", "page_number": 0})
    [{'text': 'Hello', 'page_number': 0, 'section_type': 'text', 'source_type': 'unknown'}]
    """
    _st_val = (
        source_type.value if isinstance(source_type, SourceType) else str(source_type)
    )
    _sec_val = (
        section_type.value
        if isinstance(section_type, SectionType)
        else str(section_type)
    )

    # ── str → single-element list ─────────────────────────────────────
    if isinstance(raw, str):
        return [
            {
                "text": raw,
                "section_type": _sec_val,
                "source_type": _st_val,
            }
        ]

    # ── dict → single-element list ────────────────────────────────────
    if isinstance(raw, dict):
        _validate_chunk_dict(raw)
        chunk: dict[str, Any] = dict(raw)
        chunk.setdefault("section_type", _sec_val)
        chunk.setdefault("source_type", _st_val)
        return [chunk]

    # ── list → dispatch on element type ──────────────────────────────
    if isinstance(raw, list):
        if not raw:
            return []
        first = raw[0]

        if isinstance(first, str):
            # Validate all elements are str before building output.
            for i, item in enumerate(raw):
                if not isinstance(item, str):
                    raise TypeError(
                        f"normalize_extractor_output: list element {i} is "
                        f"{type(item).__name__!r}, expected str.  When the "
                        f"first element is str, all elements must be str."
                    )
            return [
                {
                    "text": t,
                    "section_type": _sec_val,
                    "source_type": _st_val,
                }
                for t in raw
            ]

        if isinstance(first, dict):
            result: list[dict[str, Any]] = []
            for i, item in enumerate(raw):
                if not isinstance(item, dict):
                    raise TypeError(
                        f"normalize_extractor_output: list element {i} is "
                        f"{type(item).__name__!r}, expected dict.  When the "
                        f"first element is dict, all elements must be dict."
                    )
                _validate_chunk_dict(item, index=i)
                chunk = dict(item)
                chunk.setdefault("section_type", _sec_val)
                chunk.setdefault("source_type", _st_val)
                result.append(chunk)
            return result

        raise TypeError(
            f"normalize_extractor_output: list elements must be str or dict; "
            f"got {type(first).__name__!r} at index 0."
        )

    # ── unsupported type ──────────────────────────────────────────────
    raise TypeError(
        f"normalize_extractor_output: extractor must return str, dict, "
        f"list[str], or list[dict]; got {type(raw).__name__!r}."
    )


def _validate_chunk_dict(chunk: dict[str, Any], *, index: int | None = None) -> None:
    """
    Assert that a raw chunk dict has a ``"text"`` key with a ``str`` value.

    Parameters
    ----------
    chunk : dict
        Dict to validate.
    index : int or None, optional
        Position in the parent list (for error messages). Default: ``None``.

    Raises
    ------
    ValueError
        If ``"text"`` is absent or not a ``str``.
    """
    loc = f" (list element {index})" if index is not None else ""
    if "text" not in chunk:
        raise ValueError(
            f"normalize_extractor_output: chunk dict{loc} is missing "
            f"the required 'text' key. Got keys: {sorted(chunk.keys())!r}."
        )
    if not isinstance(chunk["text"], str):
        raise TypeError(
            f"normalize_extractor_output: chunk dict{loc} 'text' must be "
            f"a str; got {type(chunk['text']).__name__!r}."
        )


# ===========================================================================
# CustomReader
# ===========================================================================


@dataclass
class CustomReader(DocumentReader):
    """
    Fully user-customizable reader for any file extension and resource type.

    :class:`CustomReader` accepts *any* file extension and a caller-supplied
    *extractor* callable as its text-extraction engine.  This lets users
    integrate arbitrary third-party or proprietary extraction libraries —
    ``pdfplumber``, ``surya``, ``docling``, proprietary ASR/OCR APIs,
    in-memory streams — without writing a full :class:`DocumentReader`
    subclass.

    Two usage modes are supported:

    **Direct use** (bypass the extension registry):

    .. code-block:: python

        reader = CustomReader(
            input_file=Path("report.xyz"),
            extractor=my_extractor_fn,
        )
        docs = list(reader.get_documents())

    **Registered use** (wire into ``DocumentReader.create()``):

    .. code-block:: python

        CustomReader.register(
            name="XYZReader",
            extensions=[".xyz"],
            extractor=my_extractor_fn,
        )
        # DocumentReader.create(Path("report.xyz")) now works automatically.

    Parameters
    ----------
    input_file : pathlib.Path
        Path to the source file (or a synthetic path for non-filesystem
        resources — set ``validate_file=False`` in that case).
    extractor : callable or None, optional
        User-supplied extraction function.  Signature::

            def extractor(path: pathlib.Path, **kwargs) -> ExtractorOutput

        where ``ExtractorOutput`` is one of:

        * ``str`` — full-file text as one chunk.
        * ``list[str]`` — one string per logical segment.
        * ``dict`` — single chunk with ``"text"`` key and optional metadata.
        * ``list[dict]`` — multiple chunks, each with a ``"text"`` key.

        ``None`` is accepted so that :meth:`register`-produced subclasses
        can be instantiated without explicitly passing an extractor (the
        bound extractor is injected by ``__post_init__`` in the subclass).
        Raises :class:`ValueError` at extraction time if still ``None``.
        Default: ``None``.
    extensions : list of str or None, optional
        File extensions this instance handles (e.g. ``[".abc"]``). Used
        only by :meth:`register` to label the generated subclass; has no
        effect in single-instance usage.  Default: ``None``.
    reader_kwargs : dict, optional
        Extra keyword arguments forwarded to ``extractor`` on every call.
        Default: ``{}`` (empty).
    default_source_type : SourceType, optional
        Fallback source type for chunks where the extractor does not set
        ``"source_type"``.
        Default: :attr:`~scikitplot.corpus._schema.SourceType.UNKNOWN`.
    default_section_type : SectionType, optional
        Fallback section type for chunks where the extractor does not set
        ``"section_type"``.
        Default: :attr:`~scikitplot.corpus._schema.SectionType.TEXT`.
    validate_file : bool, optional
        When ``True`` (default), :meth:`validate_input` checks that
        ``input_file`` exists and is a regular file before extraction.
        Set to ``False`` for non-filesystem sources (network streams,
        in-memory paths) where ``input_file`` is a synthetic path.
        Default: ``True``.
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
    file_type : ClassVar[None]
        Always ``None``.  :class:`CustomReader` does not auto-register for
        any extension.  Use :meth:`register` to create a registered subclass.

    Raises
    ------
    TypeError
        If ``extractor`` is not callable (and not ``None``).
    ValueError
        If any element of ``extensions`` does not start with ``'.'`` or
        ``':'``.
    ValueError
        If ``extractor`` is ``None`` when :meth:`get_raw_chunks` is called.

    See Also
    --------
    CustomReader.register : Dynamically register a named subclass.
    normalize_extractor_output : Coerce extractor return values.
    scikitplot.corpus._readers.PDFReader : Built-in PDF reader with
        ``prefer_backend="custom"`` option.
    scikitplot.corpus._readers.ImageReader : Built-in image reader with
        ``backend="custom"`` option.
    scikitplot.corpus._base.DocumentReader : Abstract base class.

    Notes
    -----
    **Extractor kwargs** — ``reader_kwargs`` is forwarded as
    ``**reader_kwargs`` to the extractor.  Use it to pass library-specific
    options (e.g. ``{"password": "hunter2"}`` for an encrypted PDF extractor,
    or ``{"language": "en"}`` for an ASR extractor).

    **Thread safety** — :class:`CustomReader` instances are not thread-safe.
    Create one instance per thread when parallelising.

    **Empty chunks** — the downstream :class:`DefaultFilter` discards
    whitespace-only chunks, consistent with all other readers.  Empty strings
    returned by the extractor are silently skipped.

    Examples
    --------
    Plug in ``pdfplumber`` as a custom PDF backend:

    >>> import pdfplumber
    >>> from pathlib import Path
    >>> from scikitplot.corpus._readers._custom import CustomReader
    >>>
    >>> def pdfplumber_extract(path, **kw):
    ...     with pdfplumber.open(path) as pdf:
    ...         return [
    ...             {"text": page.extract_text() or "", "page_number": i}
    ...             for i, page in enumerate(pdf.pages)
    ...         ]
    >>>
    >>> reader = CustomReader(
    ...     input_file=Path("report.pdf"),
    ...     extractor=pdfplumber_extract,
    ... )
    >>> docs = list(reader.get_documents())

    Register globally and use via factory:

    >>> CustomReader.register(
    ...     name="PdfPlumberReader",
    ...     extensions=[".pdf"],
    ...     extractor=pdfplumber_extract,
    ...     default_source_type=SourceType.RESEARCH,
    ... )
    >>> reader = DocumentReader.create(Path("report.pdf"))
    >>> docs = list(reader.get_documents())

    Custom audio transcription (e.g. a proprietary ASR API):

    >>> def my_asr(path, language="en", **kw):
    ...     result = my_asr_client.transcribe(path, lang=language)
    ...     return [
    ...         {"text": seg.text, "timecode_start": seg.start, "timecode_end": seg.end}
    ...         for seg in result.segments
    ...     ]
    >>>
    >>> CustomReader.register(
    ...     name="MyASRReader",
    ...     extensions=[".mp3", ".wav", ".flac"],
    ...     extractor=my_asr,
    ...     reader_kwargs={"language": "de"},
    ...     default_source_type=SourceType.PODCAST,
    ... )

    Non-filesystem source (validate_file=False):

    >>> def stream_extractor(path, **kw):
    ...     # path is a synthetic Path wrapping a stream identifier
    ...     data = fetch_from_stream(str(path))
    ...     return data.decode("utf-8")
    >>>
    >>> reader = CustomReader(
    ...     input_file=Path("stream://channel/42"),
    ...     extractor=stream_extractor,
    ...     validate_file=False,
    ... )
    """

    # ------------------------------------------------------------------
    # Class-level registry key — None means no auto-registration.
    # Subclasses created by register() override file_types to trigger
    # __init_subclass__ registration.
    # ------------------------------------------------------------------

    file_type: ClassVar[str | None] = None
    # file_types intentionally not set → getattr returns None → no auto-register

    # ------------------------------------------------------------------
    # Instance fields
    # ------------------------------------------------------------------

    extractor: Callable[..., Any] | None = field(default=None, repr=False)
    """
    User-supplied extraction callable.  Accepts :class:`pathlib.Path` plus
    any ``**reader_kwargs`` and must return a value normalizable by
    :func:`normalize_extractor_output`.  ``None`` is allowed here so that
    :meth:`register`-generated subclasses can be instantiated through the
    :meth:`~scikitplot.corpus._base.DocumentReader.create` factory without
    explicitly passing an extractor.  Raises :class:`ValueError` at
    extraction time if still ``None``.
    """

    extensions: list[str] | None = field(default=None)
    """
    Extensions this instance handles.  Informational only for single-instance
    usage; meaningful for :meth:`register` where it controls which extensions
    are wired into the :class:`~scikitplot.corpus._base.DocumentReader`
    registry.
    """

    reader_kwargs: dict[str, Any] = field(default_factory=dict)
    """Extra keyword arguments forwarded to :attr:`extractor` on every call."""

    default_source_type: SourceType = field(default=SourceType.UNKNOWN)
    """
    Fallback :class:`~scikitplot.corpus._schema.SourceType` for chunks
    where the extractor does not set ``"source_type"``.
    """

    default_section_type: SectionType = field(default=SectionType.TEXT)
    """
    Fallback :class:`~scikitplot.corpus._schema.SectionType` for chunks
    where the extractor does not set ``"section_type"``.
    """

    validate_file: bool = field(default=True)
    """
    When ``False``, skip the filesystem existence check in
    :meth:`validate_input`.  Use for non-filesystem resources where
    :attr:`input_file` is a synthetic path.
    """

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """
        Validate constructor arguments.

        Raises
        ------
        TypeError
            If ``extractor`` is provided but not callable.
        ValueError
            If any element of ``extensions`` does not start with ``'.'``
            (file extension) or ``':'`` (URL-scheme key).
        """
        super().__post_init__()
        # extractor=None is allowed (filled in by register()-generated
        # subclasses). Only validate when a value is provided.
        if self.extractor is not None and not callable(self.extractor):
            raise TypeError(
                f"CustomReader: extractor must be callable or None; "
                f"got {type(self.extractor).__name__!r}."
            )
        if self.extensions is not None:
            bad = [
                e
                for e in self.extensions
                if not (isinstance(e, str) and e.startswith((".", ":")))
            ]
            if bad:
                raise ValueError(
                    f"CustomReader: extensions must start with '.' (file) "
                    f"or ':' (URL scheme); invalid: {bad!r}."
                )

    # ------------------------------------------------------------------
    # validate_input override
    # ------------------------------------------------------------------

    def validate_input(self) -> None:
        """
        Check source accessibility.

        Delegates to the parent implementation when :attr:`validate_file`
        is ``True``; skips the filesystem check entirely when it is
        ``False`` (for non-filesystem sources).

        Raises
        ------
        ValueError
            If :attr:`validate_file` is ``True`` and the file does not
            exist or is not a regular file.
        """
        if self.validate_file:
            super().validate_input()

    # ------------------------------------------------------------------
    # DocumentReader contract
    # ------------------------------------------------------------------

    def get_raw_chunks(self) -> Generator[dict[str, Any], None, None]:
        """
        Call the user-supplied extractor and yield normalised raw chunk dicts.

        Calls ``self.extractor(self.input_file, **self.reader_kwargs)``
        and normalises the return value with :func:`normalize_extractor_output`.

        Yields
        ------
        dict
            Each dict has at least ``{"text": str}``, with ``"section_type"``
            and ``"source_type"`` defaults filled in, plus any metadata
            returned by the extractor.

        Raises
        ------
        ValueError
            If :attr:`extractor` is ``None`` at call time.
        TypeError
            If the extractor returns an unsupported type.
        ValueError
            If any dict returned by the extractor lacks a ``"text"`` key.
        RuntimeError
            If the extractor raises an unexpected exception.  The original
            exception is chained via ``from``.

        Notes
        -----
        Logging at ``INFO`` level records the extractor name, file name,
        and chunk count. ``DEBUG`` records the kwargs forwarded.
        """
        if self.extractor is None:
            raise ValueError(
                "CustomReader: extractor is not set. "
                "Provide an 'extractor' callable when constructing "
                "CustomReader, or use CustomReader.register() to create a "
                "bound subclass that is wired into DocumentReader.create()."
            )

        extractor_name = getattr(self.extractor, "__name__", repr(self.extractor))
        logger.info(
            "CustomReader: calling extractor %r on %s (kwargs=%r).",
            extractor_name,
            self.file_name,
            list(self.reader_kwargs.keys()),
        )
        logger.debug("CustomReader: reader_kwargs=%r.", self.reader_kwargs)

        try:
            raw = self.extractor(self.input_file, **self.reader_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"CustomReader: extractor {extractor_name!r} raised an error "
                f"processing {self.file_name!r}: {exc}"
            ) from exc

        chunks = normalize_extractor_output(
            raw,
            source_type=self.default_source_type,
            section_type=self.default_section_type,
        )

        logger.info(
            "CustomReader: extractor returned %d chunk(s) from %s.",
            len(chunks),
            self.file_name,
        )

        yield from chunks

    # ------------------------------------------------------------------
    # Dynamic registration classmethod
    # ------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        *,
        name: str,
        extensions: list[str],
        extractor: Callable[..., Any],
        reader_kwargs: dict[str, Any] | None = None,
        default_source_type: SourceType = SourceType.UNKNOWN,
        default_section_type: SectionType = SectionType.TEXT,
        validate_file: bool = True,
    ) -> type[CustomReader]:
        """
        Create a named :class:`CustomReader` subclass and register it by
        extension.

        After calling :meth:`register`, :meth:`DocumentReader.create
        <scikitplot.corpus._base.DocumentReader.create>` automatically
        dispatches files with any of the given ``extensions`` to
        ``extractor``.

        Parameters
        ----------
        name : str
            Class name for the generated subclass (e.g.
            ``"PdfPlumberReader"``).  Must be a valid Python identifier.
        extensions : list of str
            File extensions to register (e.g. ``[".pdf"]``). Each must
            start with ``'.'`` (file extension) or ``':'`` (URL-scheme
            key).  Existing registrations for these extensions emit a
            warning and are replaced, consistent with the base-class
            registry behaviour.
        extractor : callable
            Extraction callable. Signature::

                def extractor(path: pathlib.Path, **kwargs) -> ExtractorOutput

        reader_kwargs : dict or None, optional
            Default keyword arguments forwarded to ``extractor``.  Instance-
            level ``reader_kwargs`` (passed directly to the constructor)
            are merged on top: instance kwargs override registered defaults.
            Default: ``{}`` (empty).
        default_source_type : SourceType, optional
            Source type applied to chunks that do not set ``"source_type"``.
            Default: :attr:`~scikitplot.corpus._schema.SourceType.UNKNOWN`.
        default_section_type : SectionType, optional
            Section type applied to chunks that do not set
            ``"section_type"``.
            Default: :attr:`~scikitplot.corpus._schema.SectionType.TEXT`.
        validate_file : bool, optional
            When ``False``, skip the filesystem existence check.
            Default: ``True``.

        Returns
        -------
        type[CustomReader]
            The newly created and registered subclass.  The caller can
            keep a reference to it for type-checking or documentation,
            but it is not required — the subclass is also stored in
            :attr:`~scikitplot.corpus._base.DocumentReader._registry`.

        Raises
        ------
        ValueError
            If ``name`` is not a valid Python identifier.
        ValueError
            If ``extensions`` is empty or any element has an invalid prefix.
        TypeError
            If ``extractor`` is not callable.

        Notes
        -----
        **Subclass lifetime** — each call to :meth:`register` creates a
        *new* class object.  Calling :meth:`register` again with the same
        ``name`` produces a distinct class object.  The last registration
        for a given extension wins (matching the general registry policy).

        **reader_kwargs merging** — instance-level kwargs (passed when
        constructing the reader) are merged on top of the registered
        defaults::

            # Registered defaults: {"language": "en"}
            # Instance override: {"language": "de"}
            reader = DocumentReader.create(
                Path("file.mp3"),
                reader_kwargs={"language": "de"},  # forwarded via **kwargs
            )
            # extractor receives language="de"

        **Type annotation** — the returned class is typed as
        ``type[CustomReader]``.  If you need the precise subclass type,
        assign it directly::

            MyReader = CustomReader.register(name="MyReader", ...)

        Examples
        --------
        Register a ``pdfplumber``-based PDF reader:

        >>> import pdfplumber
        >>> from pathlib import Path
        >>> from scikitplot.corpus._readers._custom import CustomReader
        >>> from scikitplot.corpus._schema import SourceType
        >>>
        >>> def pdfplumber_extract(path, **kw):
        ...     with pdfplumber.open(path) as pdf:
        ...         return [
        ...             {"text": p.extract_text() or "", "page_number": i}
        ...             for i, p in enumerate(pdf.pages)
        ...         ]
        >>>
        >>> PdfPlumberReader = CustomReader.register(
        ...     name="PdfPlumberReader",
        ...     extensions=[".pdf"],
        ...     extractor=pdfplumber_extract,
        ...     default_source_type=SourceType.RESEARCH,
        ... )
        >>> docs = list(DocumentReader.create(Path("paper.pdf")).get_documents())

        Register a multi-extension audio reader using a proprietary API:

        >>> MyASRReader = CustomReader.register(
        ...     name="MyASRReader",
        ...     extensions=[".mp3", ".wav", ".flac"],
        ...     extractor=my_asr_fn,
        ...     reader_kwargs={"model": "large-v3", "language": "en"},
        ...     default_source_type=SourceType.PODCAST,
        ... )
        """  # noqa: D205
        # Validate inputs before creating the subclass
        if not name.isidentifier():
            raise ValueError(
                f"CustomReader.register: name must be a valid Python "
                f"identifier; got {name!r}."
            )
        if not extensions:
            raise ValueError(
                "CustomReader.register: extensions must be a non-empty list."
            )
        bad_exts = [
            e
            for e in extensions
            if not (isinstance(e, str) and e.startswith((".", ":")))
        ]
        if bad_exts:
            raise ValueError(
                f"CustomReader.register: extensions must start with '.' or "
                f"':'; invalid: {bad_exts!r}."
            )
        if not callable(extractor):
            raise TypeError(
                f"CustomReader.register: extractor must be callable; "
                f"got {type(extractor).__name__!r}."
            )

        # Capture closure values (avoids late-binding issues in the inner fn).
        _extractor = extractor
        _reader_kwargs: dict[str, Any] = dict(reader_kwargs or {})
        _default_source_type = default_source_type
        _default_section_type = default_section_type
        _validate_file = validate_file
        _parent_cls = cls  # bind to CustomReader, not a future subclass

        def _post_init_bound(self: CustomReader) -> None:
            """
            Inject bound extractor and defaults before parent validation.

            This override is installed on the dynamically generated subclass.
            Execution order:
            1. Dataclass ``__init__`` sets all fields (extractor defaults to None).
            2. Dataclass ``__init__`` calls ``self.__post_init__()``, which
               dispatches here.
            3. We fill in extractor + defaults from the closure.
            4. We delegate to ``CustomReader.__post_init__`` for validation.
            """
            # Fill in extractor from closure when the caller did not provide
            # one explicitly (the common case via DocumentReader.create()).
            if self.extractor is None:
                object.__setattr__(self, "extractor", _extractor)
            # Merge reader_kwargs: instance-level (more specific) wins over
            # registered defaults (less specific).
            merged_kw = {**_reader_kwargs, **self.reader_kwargs}
            object.__setattr__(self, "reader_kwargs", merged_kw)
            # Set registered defaults for source/section type and validation.
            # These are overridable at instance level by passing them to
            # the constructor explicitly.
            object.__setattr__(self, "default_source_type", _default_source_type)
            object.__setattr__(self, "default_section_type", _default_section_type)
            object.__setattr__(self, "validate_file", _validate_file)
            # Delegate to CustomReader.__post_init__ for type checks.
            _parent_cls.__post_init__(self)

        # Create the dynamic subclass.
        #
        # Setting file_types as a plain class attribute (not a ClassVar
        # annotation) is sufficient for __init_subclass__ to detect and
        # register the extensions. The parent CustomReader's file_type=None
        # ClassVar is shadowed by the subclass-level attribute.
        #
        # The subclass is NOT decorated with @dataclass, so it inherits
        # CustomReader's __init__ (which accepts all fields including extractor
        # with its default=None). The __post_init__ override above runs because
        # the inherited __init__ calls self.__post_init__() at the end.
        subclass: type[CustomReader] = type(  # type: ignore[assignment]
            name,
            (cls,),
            {
                "file_type": None,  # shadow parent ClassVar
                "file_types": list(extensions),  # triggers __init_subclass__
                "__post_init__": _post_init_bound,
                "__doc__": (
                    f"Auto-generated CustomReader subclass for "
                    f"extension(s) {extensions!r}.\n\n"
                    f"Registered extractor: {extractor!r}\n"
                    f"Default source type: {default_source_type!r}\n"
                    f"Default section type: {default_section_type!r}"
                ),
            },
        )
        subclass.__name__ = name
        subclass.__qualname__ = name

        logger.info(
            "CustomReader.register: registered %r for extension(s) %r.",
            name,
            extensions,
        )
        return subclass
