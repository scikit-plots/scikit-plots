"""
scikitplot.corpus._chunkers._sentence
=====================================
Sentence-boundary segmentation via spaCy.

This module is a ground-up rewrite of remarx's ``segment.py``. Every failure
mode from the original is resolved at its root cause:

Original issues resolved
------------------------
1. **Model reload on every call** — models are cached in a per-instance dict
   keyed by model name; repeated calls are O(1) dict lookups.
2. **Silent auto-download in CI/Docker** — ``auto_download`` param defaults to
   ``False``. Callers must opt in explicitly. When ``False`` and the model is
   missing, a ``RuntimeError`` with the exact install command is raised.
3. **Hard-coded German model** — ``model_name`` is a required constructor
   parameter. The caller chooses the model; this class has no language opinion.
4. **No max_length guard** — text longer than ``max_text_length`` is rejected
   with an actionable error before being passed to spaCy.
5. **Narrow exception handling** — both ``OSError`` and ``IOError`` (which is
   ``OSError`` on Py3) are handled in the model-load path.
6. **No empty text guard** — empty/whitespace-only input returns ``[]``
   immediately without loading spaCy.
7. **Full NLP pipeline for sentence splitting** — disables all pipes that do
   not contribute to sentence detection (``ner``, ``tagger``, ``lemmatizer``,
   ``attribute_ruler``) at load time.
8. **No download gate** — auto-download is behind an explicit ``auto_download``
   flag and logs a warning before mutating the environment.

Python compatibility
--------------------
Python 3.8-3.15. No use of ``match``, ``StrEnum``, or ``Self``.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import threading
from typing import Any, ClassVar, Dict, List, Optional, Tuple  # noqa: F401

from scikitplot.corpus._base import ChunkerBase
from scikitplot.corpus._schema import ChunkingStrategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipes that are safe to disable when only sentence detection is needed.
# We intentionally keep 'tok2vec' enabled: it feeds the senter/sentencizer.
# The senter or sentencizer component itself is never disabled.
# ---------------------------------------------------------------------------
_SAFE_TO_DISABLE: tuple[str, ...] = (
    "ner",
    "tagger",
    "lemmatizer",
    "attribute_ruler",
    "morphologizer",
)

# Default upper bound for text passed to spaCy in a single call.
# spaCy's own default is 1_000_000; we match it and allow override.
_DEFAULT_MAX_TEXT_LENGTH: int = 1_000_000


class SentenceChunker(ChunkerBase):
    """
    Sentence-boundary segmentation via spaCy.

    Wraps spaCy's sentence detection pipeline component (``senter`` or
    ``sentencizer``) and returns ``(char_start, sentence_text)`` pairs that
    match the :class:`~scikitplot.corpus._base.ChunkerBase` contract.

    The loaded spaCy ``Language`` object is cached in ``_model_cache`` so
    that repeated calls with the same model name pay the load cost only once.
    The cache is instance-level, not module-level, so different
    ``SentenceChunker`` instances with different configurations do not share
    state.

    Parameters
    ----------
    model_name : str
        spaCy model to load (e.g. ``"de_core_news_sm"``,
        ``"en_core_web_sm"``, ``"xx_sent_ud_sm"``). No default is provided
        because the correct model is corpus-dependent; callers must choose.
    auto_download : bool, optional
        When ``True``, automatically download the model if it is not
        installed. Logs a warning before downloading. Set to ``False`` in
        CI/Docker environments where network access is restricted or where
        environment mutation is undesirable. Default: ``False``.
    max_text_length : int, optional
        Maximum number of characters accepted in a single ``chunk()`` call.
        Text exceeding this limit raises ``ValueError`` rather than passing
        silently to spaCy and risking a ``MemoryError``. Default:
        ``1_000_000``.
    extra_disable : list of str, optional
        Additional spaCy pipe names to disable beyond the default set
        ``("ner", "tagger", "lemmatizer", "attribute_ruler", "morphologizer")``.
        Useful when a custom model includes pipes not in the default list.
        Default: ``None``.

    Attributes
    ----------
    strategy : ChunkingStrategy
        Class variable. Always ``ChunkingStrategy.SENTENCE``.

    Raises
    ------
    RuntimeError
        If the spaCy model is not installed and ``auto_download=False``.
    ValueError
        If ``text`` exceeds ``max_text_length`` characters.
    ImportError
        If ``spacy`` is not installed at all.

    See Also
    --------
    scikitplot.corpus._chunkers.ParagraphChunker : Blank-line splitting.
    scikitplot.corpus._chunkers.FixedWindowChunker : Sliding window.

    Notes
    -----
    **Thread safety:** ``_model_cache`` is protected by a ``threading.Lock``
    so that concurrent pipeline workers loading the same model for the first
    time do not trigger multiple downloads or race on dict mutation.

    **Pipe disabling:** Pipes are disabled at ``spacy.load()`` time, not via
    ``nlp.select_pipes()``, because the latter creates a context manager and
    is incompatible with persistent cached models.

    **Model selection guidance:**
    - German: ``de_core_news_sm`` (small, fast) or ``de_core_news_lg`` (large)
    - English: ``en_core_web_sm``
    - Multilingual: ``xx_sent_ud_sm`` (no tagger/NER, sentence-only)

    Examples
    --------
    >>> chunker = SentenceChunker("en_core_web_sm")
    >>> chunks = chunker.chunk("Hello world. Second sentence here.")
    >>> [(start, text) for start, text in chunks]
    [(0, 'Hello world.'), (13, 'Second sentence here.')]

    Using auto-download (opt-in):

    >>> chunker = SentenceChunker("en_core_web_sm", auto_download=True)
    >>> chunks = chunker.chunk("Hello world.")
    """

    strategy: ClassVar[ChunkingStrategy] = ChunkingStrategy.SENTENCE

    def __init__(
        self,
        model_name: str,
        *,
        auto_download: bool = False,
        max_text_length: int = _DEFAULT_MAX_TEXT_LENGTH,
        extra_disable: list[str] | None = None,
    ) -> None:
        if not model_name or not model_name.strip():
            raise ValueError(
                "SentenceChunker: model_name must be a non-empty string."
                " Example: 'de_core_news_sm' or 'en_core_web_sm'."
            )
        if max_text_length <= 0:
            raise ValueError(
                f"SentenceChunker: max_text_length must be > 0;"
                f" got {max_text_length!r}."
            )

        self.model_name: str = model_name
        self.auto_download: bool = auto_download
        self.max_text_length: int = max_text_length
        self.extra_disable: list[str] = list(extra_disable or [])

        # Per-instance model cache keyed by (model_name, disable_tuple)
        # Using instance-level cache (not class-level) so that different
        # configurations do not share loaded models.
        self._model_cache: dict[str, Any] = {}
        self._cache_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Private: model loading with cache and download gate
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """
        Load and cache the spaCy model for this chunker.

        Returns the cached model on subsequent calls. Thread-safe via lock.

        Returns
        -------
        spacy.language.Language
            Loaded (and optionally pipe-disabled) spaCy pipeline.

        Raises
        ------
        ImportError
            If spaCy is not installed.
        RuntimeError
            If the model is not installed and ``auto_download=False``.
        """
        cache_key = self.model_name

        # Fast path: already cached (no lock needed for read after first set)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        with self._cache_lock:
            # Double-check after acquiring lock
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]

            try:
                import spacy  # noqa: PLC0415
            except ImportError as exc:
                raise ImportError(
                    "spacy is required for SentenceChunker."
                    " Install it with: pip install spacy"
                ) from exc

            # Determine which pipes to disable at load time
            disable_pipes = list(_SAFE_TO_DISABLE) + self.extra_disable

            def _try_load() -> Any:
                """Attempt to load the model, disabling only pipes that exist."""
                all_names = spacy.info().get("pipelines", {})
                # spacy.load with exclude silently skips missing pipes
                return spacy.load(self.model_name, exclude=disable_pipes)

            try:
                nlp = _try_load()
                logger.debug(
                    "SentenceChunker: loaded model %r (pipes: %s).",
                    self.model_name,
                    nlp.pipe_names,
                )
            except OSError as e:
                # Model not installed
                if not self.auto_download:
                    raise RuntimeError(
                        f"spaCy model {self.model_name!r} is not installed"
                        f" and auto_download=False.\n"
                        f"Install it with:\n"
                        f"  python -m spacy download {self.model_name}\n"
                        f"Or set auto_download=True to allow automatic download"
                        f" (not recommended for CI/Docker)."
                    ) from e
                logger.warning(
                    "SentenceChunker: model %r not found; downloading."
                    " Set auto_download=False to suppress this behaviour.",
                    self.model_name,
                )
                try:
                    from spacy.cli import download as _spacy_download  # noqa: PLC0415

                    _spacy_download(self.model_name)
                except Exception as dl_exc:
                    raise RuntimeError(
                        f"Failed to download spaCy model {self.model_name!r}."
                        f" Cause: {dl_exc}\n"
                        f"Install it manually with:"
                        f" python -m spacy download {self.model_name}"
                    ) from dl_exc
                nlp = _try_load()
                logger.info(
                    "SentenceChunker: model %r downloaded and loaded.",
                    self.model_name,
                )

            self._model_cache[cache_key] = nlp
            return nlp

    # ------------------------------------------------------------------
    # ChunkerBase contract
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, str]]:
        """
        Segment ``text`` into sentences using the configured spaCy model.

        Parameters
        ----------
        text : str
            Raw text to segment. Empty or whitespace-only input returns
            ``[]`` immediately without invoking spaCy.
        metadata : dict or None, optional
            Chunk-level metadata (not used by ``SentenceChunker`` but
            accepted to satisfy the base-class contract). Default: ``None``.

        Returns
        -------
        list of (int, str)
            Ordered list of ``(char_start, sentence_text)`` pairs.
            ``char_start`` is the character offset of the sentence within
            the input ``text``.

        Raises
        ------
        ValueError
            If ``text`` is ``None`` or exceeds ``max_text_length``.
        RuntimeError
            If the spaCy model is not installed and ``auto_download=False``.
        ImportError
            If spaCy is not installed.

        Notes
        -----
        Character offsets are taken directly from spaCy's ``sent.start_char``
        attribute, which is the offset within the input ``text`` string.

        Examples
        --------
        >>> chunker = SentenceChunker("en_core_web_sm")
        >>> chunker.chunk("One sentence. Two sentences.")
        [(0, 'One sentence.'), (14, 'Two sentences.')]

        >>> chunker.chunk("")
        []

        >>> chunker.chunk("   ")
        []
        """
        if text is None:
            raise ValueError("SentenceChunker.chunk: text must not be None.")

        # Fast exit for empty / whitespace-only text
        if not text.strip():
            return []

        # Guard against texts that would exhaust spaCy's memory
        if len(text) > self.max_text_length:
            raise ValueError(
                f"SentenceChunker.chunk: text length {len(text):,} exceeds"
                f" max_text_length {self.max_text_length:,}."
                f" Split the text before chunking, or increase max_text_length."
            )

        nlp = self._load_model()
        doc = nlp(text)

        return [(sent.start_char, sent.text) for sent in doc.sents if sent.text.strip()]

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SentenceChunker("
            f"model_name={self.model_name!r},"
            f" auto_download={self.auto_download},"
            f" max_text_length={self.max_text_length:,})"
        )


__all__ = ["SentenceChunker"]
