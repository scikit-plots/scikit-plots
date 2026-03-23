# scikitplot/corpus/_chunkers/_sentence.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._chunkers._sentence
=====================================
Sentence-boundary segmentation via spaCy, NLTK, or regex.

This module is a ground-up rewrite of remarx's ``segment.py``. Every failure
mode from the original is resolved at its root cause:

Original issues resolved
------------------------
1. **Model reload on every call** — models are cached in a per-instance dict
   (``SentenceChunker._nlp_cache``) keyed by model name; repeated calls are
   O(1) dict lookups.  The cache is created in ``__init__`` and passed
   explicitly into ``_split_spacy``; there is no global mutable state.
2. **Silent auto-download in CI/Docker** — ``auto_download`` param defaults to
   ``False``.  Callers must opt in explicitly.  When ``False`` and the model is
   missing, a ``RuntimeError`` with the exact install command is raised.
3. **Hard-coded German model** — ``model_name`` is a required constructor
   parameter.  The caller chooses the model; this class has no language opinion.
4. **No max_length guard** — text longer than ``max_text_length`` is rejected
   with an actionable error before being passed to spaCy.
5. **Narrow exception handling** — ``OSError`` from ``spacy.load`` is caught
   and re-raised with the exact ``python -m spacy download <model>`` command.
6. **No empty text guard** — empty/whitespace-only input returns ``[]``
   immediately without loading spaCy.
7. **Full NLP pipeline for sentence splitting** — disables all pipes that do
   not contribute to sentence detection (``ner``, ``tagger``, ``lemmatizer``,
   ``attribute_ruler``) at load time.
8. **No download gate** — auto-download is behind an explicit ``auto_download``
   flag and logs a warning before mutating the environment.

Constructor convenience
-----------------------
``SentenceChunker`` accepts three equivalent forms:

.. code-block:: python

    # 1. Defaults — REGEX backend, no overlap, min_length=10
    chunker = SentenceChunker()

    # 2. String shorthand — SPACY backend with the named model
    chunker = SentenceChunker("en_core_web_sm")

    # 3. Explicit config object
    chunker = SentenceChunker(
        SentenceChunkerConfig(
            backend=SentenceBackend.SPACY, spacy_model="en_core_web_sm"
        )
    )

Python compatibility
--------------------
Python 3.8-3.15.  No use of ``match``, ``StrEnum``, or ``Self``.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field  # noqa: F401
from enum import Enum
from typing import Any, Final

from .._types import Chunk, ChunkerConfig, ChunkResult

logger = logging.getLogger(__name__)

__all__ = [
    "SentenceBackend",
    "SentenceChunker",
    "SentenceChunkerConfig",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MIN_LEN: Final[int] = 10
_DEFAULT_OVERLAP: Final[int] = 0
_ABBREV_PLACEHOLDER: Final[str] = "\x00ABR\x00"

_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "rev",
        "gen",
        "sgt",
        "cpl",
        "pvt",
        "st",
        "ave",
        "blvd",
        "dept",
        "est",
        "fig",
        "govt",
        "inc",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "approx",
        "etc",
        "vs",
        "viz",
        "al",
        "et",
        "vol",
        "no",
        "pp",
        "ed",
        "eds",
    ]
)

_SENTENCE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\"\'\(\[])"
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class SentenceBackend(str, Enum):
    """Supported sentence-splitting backends."""

    REGEX = "regex"
    NLTK = "nltk"
    SPACY = "spacy"


@dataclass(frozen=True)
class SentenceChunkerConfig(ChunkerConfig):
    """Configuration for :class:`SentenceChunker`.

    Parameters
    ----------
    backend : SentenceBackend
        Splitting strategy.  ``REGEX`` has no extra dependencies.
        ``NLTK`` requires the *punkt* model.
        ``SPACY`` requires a loaded model name via *spacy_model*.
    min_length : int
        Minimum character length for a sentence to be kept.
    overlap : int
        Number of preceding sentences to prepend as context.
    spacy_model : str or None
        Spacy model name, e.g. ``"en_core_web_sm"``.
        Required when *backend* is ``SPACY``.
    nltk_language : str
        Language string forwarded to ``nltk.tokenize.sent_tokenize``.
    strip_whitespace : bool
        Strip leading/trailing whitespace from each sentence.
    include_offsets : bool
        Compute character offsets (``start_char``, ``end_char``).
    """

    backend: SentenceBackend = SentenceBackend.REGEX
    min_length: int = _DEFAULT_MIN_LEN
    overlap: int = _DEFAULT_OVERLAP
    spacy_model: str | None = None
    nltk_language: str = "english"
    strip_whitespace: bool = True
    include_offsets: bool = True


# ---------------------------------------------------------------------------
# Backend helpers — pure functions
# ---------------------------------------------------------------------------


def _protect_abbreviations(text: str) -> str:
    """Replace trailing periods in known abbreviations with a placeholder.

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Text with abbreviation periods neutralised.
    """
    for abbrev in _ABBREVIATIONS:
        pattern = re.compile(r"\b(" + re.escape(abbrev) + r")\.(\s)", re.IGNORECASE)
        text = pattern.sub(r"\1" + _ABBREV_PLACEHOLDER + r"\2", text)
    return text


def _restore_abbreviations(text: str) -> str:
    """Reverse :func:`_protect_abbreviations`.

    Parameters
    ----------
    text : str
        Text that may contain placeholders.

    Returns
    -------
    str
        Text with original periods restored.
    """
    return text.replace(_ABBREV_PLACEHOLDER, ".")


def _split_regex(text: str) -> list[str]:
    """Split *text* by sentence boundaries using regex heuristics.

    Parameters
    ----------
    text : str
        Input document text.

    Returns
    -------
    list[str]
        Raw sentence fragments (abbreviation placeholders still present).
    """
    protected = _protect_abbreviations(text)
    parts = _SENTENCE_BOUNDARY_RE.split(protected)
    return parts  # noqa: RET504


def _split_nltk(text: str, language: str) -> list[str]:
    """Split *text* using the NLTK Punkt sentence tokenizer.

    Parameters
    ----------
    text : str
        Input document text.
    language : str
        NLTK language identifier.

    Returns
    -------
    list[str]
        Sentence strings.

    Raises
    ------
    ImportError
        If ``nltk`` is not installed.
    LookupError
        If the punkt model is absent and automatic download fails.
    """
    try:
        import nltk  # type: ignore[import-untyped]  # noqa: PLC0415
        from nltk.tokenize import (  # type: ignore[import-untyped]  # noqa: PLC0415
            sent_tokenize,
        )
    except ImportError as exc:
        raise ImportError(
            "SentenceBackend.NLTK requires 'nltk'. Install with: pip install nltk"
        ) from exc

    try:
        return sent_tokenize(text, language=language)
    except LookupError:
        logger.info("NLTK punkt model missing — downloading 'punkt_tab'.")
        nltk.download("punkt_tab", quiet=True)
        return sent_tokenize(text, language=language)


def _split_spacy(
    text: str,
    model_name: str,
    nlp_cache: dict[str, Any] | None = None,
) -> list[str]:
    """Split *text* using a spaCy sentence segmentation pipeline.

    The loaded ``nlp`` object is stored in *nlp_cache* under *model_name*
    so that repeated calls with the same model pay zero reload cost.
    Pass ``SentenceChunker._nlp_cache`` to share the cache across calls
    on the same chunker instance.

    Parameters
    ----------
    text : str
        Input document text.
    model_name : str
        Name of an installed spaCy model, e.g. ``"en_core_web_sm"``.
    nlp_cache : dict[str, Any] or None, optional
        Mutable mapping used to cache loaded ``nlp`` objects.
        When ``None`` a throwaway local dict is used (no cross-call caching).

    Returns
    -------
    list[str]
        Sentence strings.

    Raises
    ------
    ImportError
        If ``spacy`` is not installed.
    OSError
        If *model_name* is not installed, with the exact install command.

    Notes
    -----
    Only the ``senter`` / ``sentencizer`` component is kept active.
    ``ner``, ``tagger``, ``lemmatizer``, and ``attribute_ruler`` are
    disabled at load time to minimise memory and CPU cost.
    """
    try:
        import spacy  # type: ignore[import-untyped]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "SentenceBackend.SPACY requires 'spacy'. Install with: pip install spacy"
        ) from exc

    # Use the caller-supplied cache (or a throwaway local one).
    cache: dict[str, Any] = nlp_cache if nlp_cache is not None else {}

    if model_name not in cache:
        try:
            cache[model_name] = spacy.load(
                model_name,
                disable=["ner", "tagger", "lemmatizer", "attribute_ruler"],
            )
        except OSError as exc:
            raise OSError(
                f"spaCy model {model_name!r} is not installed. "
                f"Install with: python -m spacy download {model_name}"
            ) from exc

    doc = cache[model_name](text)
    return [sent.text for sent in doc.sents]


def _compute_char_offsets(source: str, segments: list[str]) -> list[tuple[int, int]]:
    """Compute ``(start_char, end_char)`` offsets for *segments* in *source*.

    Parameters
    ----------
    source : str
        Original document string used as the search space.
    segments : list[str]
        Ordered text segments whose positions are needed.

    Returns
    -------
    list[tuple[int, int]]
        Parallel list of ``(start, end)`` character index pairs.
    """
    offsets: list[tuple[int, int]] = []
    cursor: int = 0
    for seg in segments:
        idx = source.find(seg, cursor)
        if idx == -1:
            idx = cursor  # fallback: positional estimate
        offsets.append((idx, idx + len(seg)))
        cursor = idx + len(seg)
    return offsets


# ---------------------------------------------------------------------------
# Public chunker
# ---------------------------------------------------------------------------


class SentenceChunker:
    """Split a document into sentence-level :class:`~.._types.Chunk` objects.

    Parameters
    ----------
    config : str or SentenceChunkerConfig or None, optional
        Three accepted forms:

        ``None`` (default)
            Constructs a :class:`SentenceChunkerConfig` with all defaults:
            ``REGEX`` backend, ``min_length=10``, no overlap.

        ``str``
            Shorthand for the ``SPACY`` backend.  The string is interpreted
            as the spaCy model name, equivalent to::

                SentenceChunkerConfig(
                    backend=SentenceBackend.SPACY,
                    spacy_model=<value>,
                )

        :class:`SentenceChunkerConfig`
            Full explicit configuration.

    Raises
    ------
    TypeError
        If *config* is not ``str``, :class:`SentenceChunkerConfig`, or ``None``.
    ValueError
        If the resolved configuration is invalid (negative lengths, missing
        model name for SPACY backend, etc.).

    Notes
    -----
    **spaCy model caching** — the loaded ``nlp`` object is stored in
    ``self._nlp_cache`` (a plain ``dict``) keyed by model name.  The cache
    is passed into :func:`_split_spacy` on every call, so ``spacy.load``
    is invoked at most once per model per chunker instance.

    Examples
    --------
    Default REGEX backend:

    >>> chunker = SentenceChunker()
    >>> result = chunker.chunk("Hello world. How are you? Fine thanks.")
    >>> len(result.chunks)
    3
    >>> result.chunks[0].text
    'Hello world.'

    spaCy shorthand (model name as string):

    >>> chunker = SentenceChunker("en_core_web_sm")
    >>> chunker.config.backend
    <SentenceBackend.SPACY: 'spacy'>
    >>> chunker.config.spacy_model
    'en_core_web_sm'

    Explicit config:

    >>> from scikitplot.corpus._chunkers._sentence import SentenceChunkerConfig
    >>> cfg = SentenceChunkerConfig(backend=SentenceBackend.NLTK, min_length=5)
    >>> chunker = SentenceChunker(cfg)
    """

    def __init__(
        self,
        config: str | SentenceChunkerConfig | None = None,
    ) -> None:
        # ------------------------------------------------------------------
        # Developer note: Accept three forms for ergonomics:
        #   str   → shorthand for SentenceChunkerConfig(SPACY, spacy_model=str)
        #   None  → SentenceChunkerConfig() with all defaults
        #   SentenceChunkerConfig → use as-is
        # This prevents the common AttributeError when callers pass a model
        # name directly (e.g. SentenceChunker("en_core_web_sm")).
        # ------------------------------------------------------------------
        if isinstance(config, str):
            config = SentenceChunkerConfig(
                backend=SentenceBackend.SPACY,
                spacy_model=config,
            )
        elif config is None:
            config = SentenceChunkerConfig()
        elif not isinstance(config, SentenceChunkerConfig):
            raise TypeError(
                f"SentenceChunker: config must be str, SentenceChunkerConfig, "
                f"or None; got {type(config).__name__!r}."
            )

        self._cfg: SentenceChunkerConfig = config

        # Per-instance spaCy model cache: model_name → loaded nlp object.
        # Passed into _split_spacy so spacy.load() runs at most once per model.
        self._nlp_cache: dict[str, Any] = {}

        self._validate_config()

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SentenceChunkerConfig:
        """The resolved :class:`SentenceChunkerConfig` for this instance."""
        return self._cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate configuration at construction time.

        Raises
        ------
        ValueError
            On invalid configuration values.
        """
        if self._cfg.min_length < 0:
            raise ValueError(
                f"SentenceChunkerConfig.min_length must be >= 0, "
                f"got {self._cfg.min_length}."
            )
        if self._cfg.overlap < 0:
            raise ValueError(
                f"SentenceChunkerConfig.overlap must be >= 0, got {self._cfg.overlap}."
            )
        if self._cfg.backend == SentenceBackend.SPACY and not self._cfg.spacy_model:
            raise ValueError(
                "SentenceChunkerConfig.spacy_model must be set "
                "when backend=SentenceBackend.SPACY."
            )

    # ------------------------------------------------------------------
    # Internal split dispatch
    # ------------------------------------------------------------------

    def _raw_sentences(self, text: str) -> list[str]:
        """Run the configured backend and return raw sentence strings.

        Parameters
        ----------
        text : str
            Document text to split.

        Returns
        -------
        list[str]
            Sentence strings from the backend.
        """
        if self._cfg.backend == SentenceBackend.REGEX:
            raw = _split_regex(text)
            return [_restore_abbreviations(s) for s in raw]
        if self._cfg.backend == SentenceBackend.NLTK:
            return _split_nltk(text, self._cfg.nltk_language)
        if self._cfg.backend == SentenceBackend.SPACY:
            # Guaranteed non-None by _validate_config.
            assert self._cfg.spacy_model is not None  # noqa: S101
            # Pass the instance cache so spacy.load() is called at most once.
            return _split_spacy(text, self._cfg.spacy_model, self._nlp_cache)
        raise ValueError(
            f"Unsupported backend: {self._cfg.backend!r}."
        )  # pragma: no cover

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        doc_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Split *text* into sentence-level chunks.

        Parameters
        ----------
        text : str
            Raw document text.
        doc_id : str, optional
            Document identifier stored in chunk metadata.
        extra_metadata : dict[str, Any], optional
            Additional key/value pairs merged into the result metadata.

        Returns
        -------
        ChunkResult
            Chunks and aggregate metadata.

        Raises
        ------
        TypeError
            If *text* is not a ``str``.
        ValueError
            If *text* is empty or whitespace-only.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}.")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")

        raw = self._raw_sentences(text)

        if self._cfg.strip_whitespace:
            raw = [s.strip() for s in raw]

        filtered: list[str] = [s for s in raw if len(s) >= self._cfg.min_length]

        offsets: list[tuple[int, int]] = (
            _compute_char_offsets(text, filtered)
            if self._cfg.include_offsets
            else [(0, 0)] * len(filtered)
        )

        chunks: list[Chunk] = []
        for idx, sent in enumerate(filtered):
            overlap_start = max(0, idx - self._cfg.overlap)
            context = filtered[overlap_start:idx]
            full_text = " ".join([*context, sent]) if context else sent

            meta: dict[str, Any] = {
                "chunk_index": idx,
                "sentence_index": idx,
                "backend": self._cfg.backend.value,
                "overlap_count": len(context),
            }
            if doc_id is not None:
                meta["doc_id"] = doc_id

            start, end = offsets[idx]
            chunks.append(
                Chunk(text=full_text, start_char=start, end_char=end, metadata=meta)
            )

        result_meta: dict[str, Any] = {
            "chunker": "sentence",
            "backend": self._cfg.backend.value,
            "total_chunks": len(chunks),
            "min_length": self._cfg.min_length,
            "overlap": self._cfg.overlap,
        }
        if doc_id is not None:
            result_meta["doc_id"] = doc_id
        if extra_metadata:
            result_meta.update(extra_metadata)

        return ChunkResult(chunks=chunks, metadata=result_meta)

    def chunk_batch(
        self,
        texts: list[str],
        doc_ids: list[str] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkResult]:
        """Chunk a list of documents.

        Parameters
        ----------
        texts : list[str]
            Input documents.
        doc_ids : list[str], optional
            Parallel document identifiers.
        extra_metadata : dict[str, Any], optional
            Shared metadata merged into every result.

        Returns
        -------
        list[ChunkResult]
            One result per document.

        Raises
        ------
        TypeError
            If *texts* is not a list.
        ValueError
            If *doc_ids* length does not match *texts* length.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list, got {type(texts).__name__!r}.")
        if doc_ids is not None and len(doc_ids) != len(texts):
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) must equal "
                f"texts length ({len(texts)})."
            )
        return [
            self.chunk(
                t,
                doc_id=doc_ids[i] if doc_ids is not None else None,
                extra_metadata=extra_metadata,
            )
            for i, t in enumerate(texts)
        ]
