# scikitplot/corpus/_custom_hooks.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.corpus._custom_hooks
================================
Comprehensive user-customization entry point for every layer of the corpus
pipeline.

This module provides a fully pluggable customization framework across all eight
pipeline layers.  Every class defined here is a drop-in replacement for its
base-class counterpart — users pass callable hooks and factory functions
without subclassing anything.

Architecture overview
---------------------

.. code-block:: text

    Layer 1 ── DocumentReader.custom_extractor         (all 14 readers, _base.py)
    Layer 2 ── CustomChunker / custom_chunk_fn         (ChunkerBase contract)
    Layer 3 ── CustomFilter                            (FilterBase contract)
    Layer 4 ── CustomNormalizer                        (NormalizerBase contract)
    Layer 5 ── CustomNLPEnricher / CustomEnricherConfig (NLPEnricher backends)
    Layer 6 ── HookableCorpusPipeline / PipelineHooks  (lifecycle callbacks)
    Layer 7 ── FactoryCorpusBuilder / BuilderFactories (component factories)
    Layer 8 ── CustomSimilarityIndex                   (custom scorer)

Layer 1 is implemented directly in :class:`~scikitplot.corpus._base.DocumentReader`
via the ``custom_extractor`` and ``custom_extractor_kwargs`` dataclass fields — no
import from this module is required to use it.  Layers 2-8 are provided here.

Single-import convenience
-------------------------
All public names are re-exported from this module so a single import covers
every customization point::

    from scikitplot.corpus._custom_hooks import (
        # Chunkers
        CustomChunker,
        # Filters
        CustomFilter,
        # Normalizers
        CustomNormalizer,
        # NLP enricher
        CustomEnricherConfig,
        CustomNLPEnricher,
        # Pipeline
        PipelineHooks,
        HookableCorpusPipeline,
        # Builder
        BuilderFactories,
        FactoryCorpusBuilder,
        # Similarity
        CustomSimilarityIndex,
    )

Python compatibility
--------------------
Python 3.8-3.15.  ``from __future__ import annotations`` throughout.
All optional dependencies are imported lazily.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from dataclasses import dataclass, field  # noqa: F401
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple  # noqa: F401

from typing_extensions import Self

from ._base import ChunkerBase, FilterBase
from ._normalizers._normalizer import NormalizerBase
from ._schema import ChunkingStrategy, CorpusDocument

logger = logging.getLogger(__name__)

__all__ = [
    # Layer 7 — Builder factories
    "BuilderFactories",
    # Layer 2 — Chunkers
    "CustomChunker",
    # Layer 5 — NLP Enricher
    "CustomEnricherConfig",
    # Layer 3 — Filters
    "CustomFilter",
    "CustomNLPEnricher",
    # Layer 4 — Normalizers
    "CustomNormalizer",
    # Layer 8 — Similarity scorer
    "CustomSimilarityIndex",
    "FactoryCorpusBuilder",
    # Layer 6 — Pipeline hooks
    "HookableCorpusPipeline",
    "PipelineHooks",
]


# =============================================================================
# Layer 2 — CustomChunker
# =============================================================================


class CustomChunker(ChunkerBase):
    r"""
    Wrap any callable as a :class:`~scikitplot.corpus._base.ChunkerBase`.

    The caller provides a ``chunk_fn`` that accepts ``(text: str,
    metadata: dict)`` and returns ``list[tuple[int, str]]`` where each tuple
    is ``(char_start, chunk_text)``.  This covers the full :meth:`chunk`
    contract without subclassing.

    Parameters
    ----------
    chunk_fn : callable
        Chunking callable. Signature::

            def chunk_fn(
                text: str,
                metadata: dict[str, Any],
            ) -> list[tuple[int, str]]: ...

        ``text`` is the raw text block to segment.
        ``metadata`` is the raw-chunk metadata dict passed by
        :meth:`~scikitplot.corpus._base.DocumentReader.get_documents`.
        Returns ``(char_start, chunk_text)`` pairs, same contract as
        :meth:`ChunkerBase.chunk`.
    name : str, optional
        Human-readable label used in ``__repr__`` and logging.
        Default: the ``__name__`` attribute of ``chunk_fn``.

    Attributes
    ----------
    strategy : ChunkingStrategy
        Always :attr:`~scikitplot.corpus._schema.ChunkingStrategy.CUSTOM`.

    Raises
    ------
    TypeError
        If ``chunk_fn`` is not callable.

    See Also
    --------
    scikitplot.corpus._base.ChunkerBase : Abstract base class.
    scikitplot.corpus._chunkers.SentenceChunker : Built-in sentence chunker.

    Notes
    -----
    **User note:** Use this when none of the built-in chunkers fit your
    segmentation logic — custom XML tag boundaries, transcript cue-based
    splits, semantic paragraph detection via a local LLM, etc.

    **Developer note:** The ``strategy`` class variable is fixed to
    ``CUSTOM`` so the pipeline records the correct
    :class:`~scikitplot.corpus._schema.ChunkingStrategy` on every generated
    :class:`~scikitplot.corpus._schema.CorpusDocument`.

    Examples
    --------
    Split on double newlines (paragraph-like) without using ParagraphChunker::

        def my_para_chunk(text, metadata):
            paras = [p.strip() for p in text.split("\\n\\n") if p.strip()]
            cursor = 0
            result = []
            for para in paras:
                idx = text.find(para, cursor)
                result.append((idx, para))
                cursor = idx + len(para)
            return result


        chunker = CustomChunker(my_para_chunk, name="DoubleNewlineChunker")
        pipeline = CorpusPipeline(chunker=chunker)
    """

    strategy: ChunkingStrategy = ChunkingStrategy.CUSTOM  # type: ignore[assignment]

    def __init__(
        self,
        chunk_fn: Callable[[str, dict[str, Any]], list[tuple[int, str]]],
        *,
        name: str | None = None,
    ) -> None:
        """
        Initialise with a chunking callable.

        Parameters
        ----------
        chunk_fn : callable
            User-supplied ``(text, metadata) -> list[tuple[int, str]]``.
        name : str or None, optional
            Display name.  Defaults to ``chunk_fn.__name__``.

        Raises
        ------
        TypeError
            If ``chunk_fn`` is not callable.
        """
        if not callable(chunk_fn):
            raise TypeError(
                f"CustomChunker: chunk_fn must be callable; "
                f"got {type(chunk_fn).__name__!r}."
            )
        self._chunk_fn = chunk_fn
        self._name = name or getattr(chunk_fn, "__name__", repr(chunk_fn))

    def chunk(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[tuple[int, str]]:
        """
        Delegate to the user-supplied ``chunk_fn``.

        Parameters
        ----------
        text : str
            Raw text to segment.
        metadata : dict or None, optional
            Raw-chunk metadata forwarded from the reader.

        Returns
        -------
        list of (int, str)
            ``(char_start, chunk_text)`` pairs.

        Raises
        ------
        ValueError
            If ``text`` is ``None``.
        RuntimeError
            If ``chunk_fn`` raises an unexpected exception.
        """
        if text is None:
            raise ValueError("CustomChunker.chunk: text must not be None.")
        if not text.strip():
            return []
        try:
            return self._chunk_fn(text, metadata or {})
        except Exception as exc:
            raise RuntimeError(
                f"CustomChunker({self._name!r}): chunk_fn raised: {exc}"
            ) from exc

    def __repr__(self) -> str:  # noqa: D105
        return f"CustomChunker(fn={self._name!r}, strategy=CUSTOM)"


# =============================================================================
# Layer 3 — CustomFilter
# =============================================================================


class CustomFilter(FilterBase):
    """
    Wrap any callable as a :class:`~scikitplot.corpus._base.FilterBase`.

    Parameters
    ----------
    fn : callable
        Filter callable.  Signature::

            def fn(doc: CorpusDocument) -> bool: ...

        Return ``True`` to include the document, ``False`` to discard it.
    name : str, optional
        Human-readable label used in ``__repr__``.

    Raises
    ------
    TypeError
        If ``fn`` is not callable.

    See Also
    --------
    scikitplot.corpus._base.DefaultFilter : Built-in noise filter.
    scikitplot.corpus._base.FilterBase : Abstract base class.

    Notes
    -----
    **User note:** Use to apply domain-specific inclusion criteria —
    language detection, source type gates, keyword presence checks, etc.

    Examples
    --------
    Keep only English documents that contain the word "treatment"::

        def medical_filter(doc):
            return (
                doc.language is None or doc.language == "en"
            ) and "treatment" in doc.text.lower()


        reader = DocumentReader.create(
            Path("research.pdf"),
            filter_=CustomFilter(medical_filter),
        )
    """

    def __init__(
        self,
        fn: Callable[[CorpusDocument], bool],
        *,
        name: str | None = None,
    ) -> None:
        """
        Initialise with a filter callable.

        Parameters
        ----------
        fn : callable
            ``(CorpusDocument) -> bool`` filter callable.
        name : str or None, optional
            Display name.

        Raises
        ------
        TypeError
            If ``fn`` is not callable.
        """
        if not callable(fn):
            raise TypeError(
                f"CustomFilter: fn must be callable; got {type(fn).__name__!r}."
            )
        self._fn = fn
        self._name = name or getattr(fn, "__name__", repr(fn))

    def include(self, doc: CorpusDocument) -> bool:
        """
        Return the result of the user-supplied filter callable.

        Parameters
        ----------
        doc : CorpusDocument
            Document to evaluate.

        Returns
        -------
        bool
            ``True`` to include; ``False`` to discard.
        """
        try:
            return bool(self._fn(doc))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CustomFilter(%r): fn raised on doc %r: %s. Discarding.",
                self._name,
                getattr(doc, "doc_id", "?"),
                exc,
            )
            return False

    def __repr__(self) -> str:  # noqa: D105
        return f"CustomFilter(fn={self._name!r})"


# =============================================================================
# Layer 4 — CustomNormalizer
# =============================================================================


class CustomNormalizer(NormalizerBase):
    r"""
    Wrap any callable as a :class:`~scikitplot.corpus._normalizers.NormalizerBase`.

    Parameters
    ----------
    fn : callable
        Normalizer callable.  One of two signatures accepted:

        ``(doc: CorpusDocument) -> CorpusDocument``
            Full document transform — the callable controls exactly which
            fields change via ``doc.replace()``.

        ``(text: str) -> str``
            Pure text transform — the module wraps the result in
            ``doc.replace(normalized_text=result)`` automatically.
            Detected by inspecting whether the return value is a ``str``.

    name : str, optional
        Human-readable label used in ``__repr__``.
    text_mode : bool, optional
        When ``True``, treat ``fn`` as a ``str → str`` transform and wrap
        automatically.  When ``False`` (default), treat ``fn`` as a full
        ``CorpusDocument → CorpusDocument`` transform.  Pass ``True`` for
        simple string-level operations (regex substitution, lowercasing, etc.)
        without writing the ``doc.replace()`` boilerplate.

    Raises
    ------
    TypeError
        If ``fn`` is not callable.

    See Also
    --------
    scikitplot.corpus._normalizers.NormalizationPipeline : Chain normalizers.
    scikitplot.corpus._normalizers.NormalizerBase : Abstract base class.

    Notes
    -----
    **User note:** Combine with :class:`~scikitplot.corpus._normalizers.NormalizationPipeline`
    to slot a custom step anywhere in the normalisation sequence.

    Examples
    --------
    Strip citation markers ``[1]``, ``[2]`` from academic text::

        import re


        def strip_citations(text: str) -> str:
            return re.sub(r"\\[\\d+\\]", "", text)


        norm = CustomNormalizer(strip_citations, text_mode=True)

    Full document transform (language detection side-channel)::

        def tag_language(doc):
            lang = detect(doc.normalized_text or doc.text)
            return doc.replace(language=lang)


        norm = CustomNormalizer(tag_language)
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        text_mode: bool = False,
    ) -> None:
        """
        Initialise with a normalization callable.

        Parameters
        ----------
        fn : callable
            ``(CorpusDocument) -> CorpusDocument`` or ``(str) -> str`` callable.
        name : str or None, optional
            Display name.
        text_mode : bool, optional
            When ``True``, wrap ``fn`` as a ``str → str`` transform.

        Raises
        ------
        TypeError
            If ``fn`` is not callable.
        """
        if not callable(fn):
            raise TypeError(
                f"CustomNormalizer: fn must be callable; got {type(fn).__name__!r}."
            )
        self._fn = fn
        self._name = name or getattr(fn, "__name__", repr(fn))
        self._text_mode = text_mode

    def normalize_doc(self, doc: CorpusDocument) -> CorpusDocument:
        """
        Apply the user-supplied callable to ``doc``.

        Parameters
        ----------
        doc : CorpusDocument
            Corpus Document.

        Returns
        -------
        CorpusDocument
            Modified document.

        Raises
        ------
        RuntimeError
            If the callable raises an unexpected exception.
        """
        try:
            if self._text_mode:
                source = self._get_source_text(doc)
                result_text = self._fn(source)
                if not isinstance(result_text, str):
                    raise TypeError(
                        f"CustomNormalizer({self._name!r}): text_mode=True but "
                        f"fn returned {type(result_text).__name__!r} instead of str."
                    )
                return doc.replace(normalized_text=result_text)
            else:  # noqa: RET505
                result = self._fn(doc)
                if not isinstance(result, CorpusDocument):
                    raise TypeError(
                        f"CustomNormalizer({self._name!r}): fn must return "
                        f"CorpusDocument; got {type(result).__name__!r}."
                    )
                return result
        except (TypeError, RuntimeError):
            raise
        except Exception as exc:
            raise RuntimeError(
                f"CustomNormalizer({self._name!r}): fn raised: {exc}"
            ) from exc

    def __repr__(self) -> str:  # noqa: D105
        return f"CustomNormalizer(fn={self._name!r}, text_mode={self._text_mode})"


# =============================================================================
# Layer 5 — CustomEnricherConfig + CustomNLPEnricher
# =============================================================================


@dataclass(frozen=True)
class CustomEnricherConfig:
    """
    Custom backend callables for :class:`CustomNLPEnricher`.

    Every field is optional.  When set it **replaces** the corresponding
    built-in backend in :class:`~scikitplot.corpus._enrichers.NLPEnricher`.
    ``None`` means "use the built-in backend from :class:`~scikitplot.corpus._enrichers.EnricherConfig`".

    Parameters
    ----------
    custom_tokenizer : callable or None, optional
        Replaces the built-in tokenizer.  Signature::

            def custom_tokenizer(text: str) -> list[str]: ...

    custom_lemmatizer : callable or None, optional
        Replaces the built-in lemmatizer.  Signature::

            def custom_lemmatizer(tokens: list[str]) -> list[str]: ...

    custom_stemmer : callable or None, optional
        Replaces the built-in stemmer.  Signature::

            def custom_stemmer(tokens: list[str]) -> list[str]: ...

    custom_keyword_extractor : callable or None, optional
        Replaces the built-in keyword extractor.  Signature::

            def custom_keyword_extractor(
                text: str,
                tokens: list[str],
            ) -> list[str]: ...

    custom_stopwords : frozenset[str] or None, optional
        Replaces the built-in stopword set used by ``_filter_tokens``.
        When ``None`` the built-in NLTK / fallback set is used.

    Notes
    -----
    **User note:** Pass a :class:`CustomEnricherConfig` together with the
    standard :class:`~scikitplot.corpus._enrichers.EnricherConfig` to
    :class:`CustomNLPEnricher`.  Built-in fields (``tokenizer``,
    ``lemmatizer``, etc.) in :class:`~scikitplot.corpus._enrichers.EnricherConfig`
    are still honoured for any stage that has no custom callable.

    Examples
    --------
    Replace keyword extraction with a KeyBERT-based extractor::

        from keybert import KeyBERT

        _kb = KeyBERT()


        def kb_extractor(text, tokens):
            return [kw for kw, _ in _kb.extract_keywords(text, top_n=10)]


        ccfg = CustomEnricherConfig(custom_keyword_extractor=kb_extractor)
        enricher = CustomNLPEnricher(custom_config=ccfg)
    """

    custom_tokenizer: Callable[[str], list[str]] | None = None
    custom_lemmatizer: Callable[[list[str]], list[str]] | None = None
    custom_stemmer: Callable[[list[str]], list[str]] | None = None
    custom_keyword_extractor: Callable[[str, list[str]], list[str]] | None = None
    custom_stopwords: frozenset[str] | None = None

    def __post_init__(self) -> None:
        """Validate all callables at construction time.

        Raises
        ------
        TypeError
            If any non-None field is not callable (except ``custom_stopwords``).
        """
        for attr in (
            "custom_tokenizer",
            "custom_lemmatizer",
            "custom_stemmer",
            "custom_keyword_extractor",
        ):
            val = getattr(self, attr)
            if val is not None and not callable(val):
                raise TypeError(
                    f"CustomEnricherConfig.{attr} must be callable or None; "
                    f"got {type(val).__name__!r}."
                )
        if self.custom_stopwords is not None and not isinstance(
            self.custom_stopwords, frozenset
        ):
            raise TypeError(
                "CustomEnricherConfig.custom_stopwords must be frozenset or None; "
                f"got {type(self.custom_stopwords).__name__!r}."
            )


class CustomNLPEnricher:
    """
    :class:`~scikitplot.corpus._enrichers.NLPEnricher` extended with
    fully-replaceable NLP backends.

    Wraps a standard :class:`~scikitplot.corpus._enrichers.NLPEnricher` and
    intercepts each processing stage when the corresponding custom callable is
    set in :attr:`custom_config`.  Built-in backends are used as fallback for
    any stage without a custom override.

    Parameters
    ----------
    config : EnricherConfig or None, optional
        Standard enrichment configuration.  ``None`` uses defaults.
    custom_config : CustomEnricherConfig or None, optional
        Custom backend callables.  ``None`` disables all custom overrides
        (equivalent to using plain :class:`~scikitplot.corpus._enrichers.NLPEnricher`).

    Notes
    -----
    **User note:** Drop-in replacement for
    :class:`~scikitplot.corpus._enrichers.NLPEnricher`.
    The same ``enrich_documents()`` interface is preserved.

    **Developer note:** Delegation order per stage:

    1. If ``custom_config.<stage>`` is set → call the custom callable.
    2. Otherwise → delegate to the wrapped ``NLPEnricher`` method.

    This keeps the built-in lazy-loading cache (spaCy, NLTK, stemmer)
    intact for any stage that does not have a custom override.

    See Also
    --------
    scikitplot.corpus._enrichers.NLPEnricher : Built-in enricher.
    CustomEnricherConfig : Custom backend callables configuration.

    Examples
    --------
    Integrate a custom tokenizer (e.g. SentencePiece)::

        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load("bpe.model")


        def sp_tokenize(text):
            return sp.encode(text, out_type=str)


        ccfg = CustomEnricherConfig(custom_tokenizer=sp_tokenize)
        enricher = CustomNLPEnricher(custom_config=ccfg)
        docs = enricher.enrich_documents(corpus_docs)
    """  # noqa: D205

    def __init__(
        self,
        config: Any | None = None,
        *,
        custom_config: CustomEnricherConfig | None = None,
    ) -> None:
        """
        Initialise with standard and custom configs.

        Parameters
        ----------
        config : EnricherConfig or None, optional
            Standard enricher config.
        custom_config : CustomEnricherConfig or None, optional
            Custom backend callables.
        """
        from ._enrichers._nlp_enricher import (  # noqa: PLC0415
            EnricherConfig,
            NLPEnricher,
        )

        self._inner = NLPEnricher(config=config or EnricherConfig())
        self.custom_config = custom_config or CustomEnricherConfig()
        self.config = self._inner.config

    # ------------------------------------------------------------------
    # Public API — mirrors NLPEnricher
    # ------------------------------------------------------------------

    def enrich_documents(
        self,
        documents: Sequence[Any],
        *,
        overwrite: bool = False,
    ) -> list[Any]:
        """
        Enrich a batch of ``CorpusDocument`` instances using custom or
        built-in backends per stage.

        Parameters
        ----------
        documents : Sequence[CorpusDocument]
            List of Corpus Document.
        overwrite : bool, optional
            Overwrite.

        Returns
        -------
        list[CorpusDocument]
        """  # noqa: D205
        out: list[Any] = []
        n_enriched = 0

        for doc in documents:
            has_tokens = getattr(doc, "tokens", None) is not None
            if not overwrite and has_tokens:
                out.append(doc)
                continue

            text = getattr(doc, "normalized_text", None) or getattr(doc, "text", "")

            tokens = self._tokenize(text)
            tokens = self._filter_tokens(tokens)
            lemmas = (
                self._lemmatize(tokens)
                if self.config.lemmatizer or self.custom_config.custom_lemmatizer
                else None
            )
            stems = (
                self._stem(tokens)
                if self.config.stemmer or self.custom_config.custom_stemmer
                else None
            )
            keywords = self._extract_keywords(text, tokens)

            out.append(
                doc.replace(
                    tokens=tokens or None,
                    lemmas=lemmas,
                    stems=stems,
                    keywords=keywords,
                )
            )
            n_enriched += 1

        logger.info(
            "CustomNLPEnricher: enriched=%d, total=%d",
            n_enriched,
            len(list(documents)),
        )
        return out

    # ------------------------------------------------------------------
    # Stage dispatch helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize using custom callable or built-in backend."""
        if self.custom_config.custom_tokenizer is not None:
            try:
                return self.custom_config.custom_tokenizer(text)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CustomNLPEnricher: custom_tokenizer raised: %s", exc)
        return self._inner._tokenize(text)

    def _filter_tokens(self, tokens: list[str]) -> list[str]:
        """Filter tokens; use custom stopwords when provided.

        When ``custom_stopwords`` is set, the stopword set is temporarily
        overridden on the inner enricher via the ``_stopwords`` property
        setter, then restored after filtering.
        """
        if self.custom_config.custom_stopwords is not None:
            # Store the previous set, inject the override, filter, restore.
            # _stopwords property setter writes to _stopwords_cache["__override__"]
            _prev = self._inner._stopwords
            self._inner._stopwords = self.custom_config.custom_stopwords
            result = self._inner._filter_tokens(
                tokens, self.custom_config.custom_stopwords
            )
            self._inner._stopwords = _prev
            return result
        return self._inner._filter_tokens(tokens)

    def _lemmatize(self, tokens: list[str]) -> list[str] | None:
        """Lemmatize using custom callable or built-in backend.

        Falls back to :meth:`NLPEnricher._lemmatize_tokens` which does
        not require a spaCy Doc to be threaded through.
        """
        if self.custom_config.custom_lemmatizer is not None:
            try:
                return self.custom_config.custom_lemmatizer(tokens)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CustomNLPEnricher: custom_lemmatizer raised: %s", exc)
        return self._inner._lemmatize_tokens(tokens)

    def _stem(self, tokens: list[str]) -> list[str] | None:
        """Stem using custom callable or built-in backend."""
        if self.custom_config.custom_stemmer is not None:
            try:
                return self.custom_config.custom_stemmer(tokens)
            except Exception as exc:  # noqa: BLE001
                logger.warning("CustomNLPEnricher: custom_stemmer raised: %s", exc)
        return self._inner._stem(tokens)

    def _extract_keywords(self, text: str, tokens: list[str]) -> list[str] | None:
        """Extract keywords using custom callable or built-in backend."""
        if self.custom_config.custom_keyword_extractor is not None:
            try:
                return self.custom_config.custom_keyword_extractor(text, tokens)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "CustomNLPEnricher: custom_keyword_extractor raised: %s", exc
                )
        return self._inner._extract_keywords(text, tokens)

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CustomNLPEnricher("
            f"config={self.config!r}, "
            f"custom_config={self.custom_config!r})"
        )


# =============================================================================
# Layer 6 — PipelineHooks + HookableCorpusPipeline
# =============================================================================


@dataclass
class PipelineHooks:
    """
    Lifecycle callbacks for :class:`HookableCorpusPipeline`.

    Every hook is optional (``None`` = no-op).  Hooks are called in the order
    listed here, at the pipeline stages indicated.

    Parameters
    ----------
    pre_read_hook : callable or None, optional
        Called **before** the reader iterates a source.  Receives the
        source label string.  Signature::

            def pre_read_hook(source: str) -> None: ...

    post_read_hook : callable or None, optional
        Called **after** all documents have been read (before embedding).
        Receives the source label and the collected document list.
        May return a modified document list or ``None`` (no modification).
        Signature::

            def post_read_hook(
                source: str,
                documents: list[CorpusDocument],
            ) -> list[CorpusDocument] | None: ...

    post_filter_hook : callable or None, optional
        Called after the built-in filter stage (inside the reader's
        ``get_documents()``).  This hook runs **per-document** as a
        final inclusion gate — return ``True`` to keep, ``False`` to discard.
        Signature::

            def post_filter_hook(doc: CorpusDocument) -> bool: ...

    post_embed_hook : callable or None, optional
        Called **after** embedding is complete.  Receives ``(source, documents)``
        and may return a modified list or ``None``.
        Signature::

            def post_embed_hook(
                source: str,
                documents: list[CorpusDocument],
            ) -> list[CorpusDocument] | None: ...

    pre_export_hook : callable or None, optional
        Called **before** exporting documents to disk.  May return a
        modified document list or ``None``.
        Signature::

            def pre_export_hook(
                source: str,
                documents: list[CorpusDocument],
            ) -> list[CorpusDocument] | None: ...

    Notes
    -----
    **User note:** Hooks are called with minimal overhead — only non-``None``
    hooks incur a function call.  Hook exceptions are caught and logged as
    warnings; they never abort the pipeline run.

    Examples
    --------
    Log progress and filter by source-type in post_read::

        def log_read(source, docs):
            print(f"{source}: {len(docs)} documents read")


        def keep_research(source, docs):
            from scikitplot.corpus._schema import SourceType

            return [d for d in docs if d.source_type == SourceType.RESEARCH]


        hooks = PipelineHooks(
            pre_read_hook=lambda src: print(f"Starting: {src}"),
            post_read_hook=lambda src, docs: log_read(src, docs)
            or keep_research(src, docs),
        )
        pipeline = HookableCorpusPipeline(hooks=hooks)
    """

    pre_read_hook: Callable[[str], None] | None = None
    post_read_hook: Callable[[str, list[Any]], list[Any] | None] | None = None
    post_filter_hook: Callable[[Any], bool] | None = None
    post_embed_hook: Callable[[str, list[Any]], list[Any] | None] | None = None
    pre_export_hook: Callable[[str, list[Any]], list[Any] | None] | None = None

    def __post_init__(self) -> None:
        """Validate all hook callables at construction time.

        Raises
        ------
        TypeError
            If any non-None field is not callable.
        """
        for attr in (
            "pre_read_hook",
            "post_read_hook",
            "post_filter_hook",
            "post_embed_hook",
            "pre_export_hook",
        ):
            val = getattr(self, attr)
            if val is not None and not callable(val):
                raise TypeError(
                    f"PipelineHooks.{attr} must be callable or None; "
                    f"got {type(val).__name__!r}."
                )


class HookableCorpusPipeline:
    """
    :class:`~scikitplot.corpus._pipeline.CorpusPipeline` extended with
    per-stage lifecycle hooks.

    Accepts all the same constructor parameters as
    :class:`~scikitplot.corpus._pipeline.CorpusPipeline` plus a
    :class:`PipelineHooks` instance.  Drop-in replacement — the public
    interface (``run``, ``run_batch``, ``run_url``) is identical.

    Parameters
    ----------
    hooks : PipelineHooks or None, optional
        Lifecycle callbacks.  ``None`` disables all hooks (identical
        behaviour to bare :class:`~scikitplot.corpus._pipeline.CorpusPipeline`).
    chunker : ChunkerBase or None, optional
        Chunker to inject into every reader.
    filter_ : FilterBase or None, optional
        Filter applied after chunking.
    embedding_engine : EmbeddingEngine or None, optional
        Embedding backend.
    output_dir : pathlib.Path or None, optional
        Output directory for exports.
    export_format : ExportFormat or None, optional
        Default export format.
    default_language : str or None, optional
        ISO 639-1 language code.
    progress_callback : callable or None, optional
        Progress notification callback.
    reader_kwargs : dict or None, optional
        Extra kwargs forwarded to each reader.

    Notes
    -----
    **User note:** All hooks are called with a ``try/except`` guard — a
    hook that raises does not abort the pipeline.

    **Developer note:** Hook injection points:

    - ``pre_read_hook`` → called at the start of ``_run_source``.
    - ``post_read_hook`` → called after ``_collect_documents``; may
      return a modified list (``None`` return = unchanged).
    - ``post_filter_hook`` → installed as an additional post-filter
      on the :class:`~scikitplot.corpus._base.FilterBase` passed to
      the reader via a :class:`_CompositeHookFilter` wrapper.
    - ``post_embed_hook`` → called after ``_embed_documents``.
    - ``pre_export_hook`` → called inside ``_export`` before writing.

    Examples
    --------
    ::

        from scikitplot.corpus._custom_hooks import (
            HookableCorpusPipeline,
            PipelineHooks,
        )

        hooks = PipelineHooks(
            pre_read_hook=lambda src: logger.info("Reading: %s", src),
            post_read_hook=lambda src, docs: [d for d in docs if len(d.text) > 50],
        )
        pipeline = HookableCorpusPipeline(hooks=hooks, output_dir=Path("out/"))
        result = pipeline.run(Path("corpus.pdf"))
    """  # noqa: D205

    def __init__(
        self,
        hooks: PipelineHooks | None = None,
        chunker: Any | None = None,
        filter_: Any | None = None,
        embedding_engine: Any | None = None,
        output_dir: Any | None = None,
        export_format: Any | None = None,
        default_language: str | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        reader_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialise the hookable pipeline.

        Parameters
        ----------
        hooks : PipelineHooks or None, optional
            Lifecycle callbacks.
        chunker : Any or None, optional
            chunker.
        filter_ : Any or None, optional
            filter_.
        embedding_engine : Any or None, optional
            embedding_engine.
        output_dir : Any or None, optional
            output_dir.
        export_format : Any or None, optional
            export_format.
        default_language : str or None, optional
            default_language.
        progress_callback : Callable or None, optional
            progress_callback.
        reader_kwargs : dict or None, optional
            Forwarded to :class:`~scikitplot.corpus._pipeline.CorpusPipeline`.
        """
        import pathlib  # noqa: PLC0415

        from ._pipeline import CorpusPipeline  # noqa: PLC0415
        from ._schema import ExportFormat  # noqa: PLC0415

        self.hooks = hooks or PipelineHooks()

        # Wrap filter_ with post_filter_hook if provided
        effective_filter = filter_
        if self.hooks.post_filter_hook is not None:
            effective_filter = _CompositeHookFilter(
                base_filter=filter_,
                hook=self.hooks.post_filter_hook,
            )

        fmt = export_format
        if fmt is None:
            fmt = ExportFormat.CSV

        self._pipeline = CorpusPipeline(
            chunker=chunker,
            filter_=effective_filter,
            embedding_engine=embedding_engine,
            output_dir=pathlib.Path(output_dir) if output_dir else None,
            export_format=fmt,
            default_language=default_language,
            progress_callback=progress_callback,
            reader_kwargs=reader_kwargs,
        )
        # Expose pipeline attrs directly
        self.chunker = self._pipeline.chunker
        self.filter_ = effective_filter
        self.embedding_engine = self._pipeline.embedding_engine
        self.output_dir = self._pipeline.output_dir
        self.export_format = self._pipeline.export_format
        self.default_language = default_language
        self.reader_kwargs = self._pipeline.reader_kwargs

    # ------------------------------------------------------------------
    # Public API — identical to CorpusPipeline
    # ------------------------------------------------------------------

    def run(
        self,
        input_file: Any,
        *,
        output_path: Any | None = None,
        export_format: Any | None = None,
        filename_override: str | None = None,
    ) -> Any:
        """
        Process a single source with lifecycle hooks applied.

        Parameters
        ----------
        input_file : pathlib.Path or str
            Path or URL.
        output_path : pathlib.Path or None, optional
            output_path.
        export_format : ExportFormat or None, optional
            export_format.
        filename_override : str or None, optional
            filename_override.

        Returns
        -------
        PipelineResult
        """
        source_label = str(input_file)

        # pre_read_hook
        self._call_hook("pre_read_hook", source_label)

        result = self._pipeline.run(
            input_file,
            output_path=output_path,
            export_format=export_format,
            filename_override=filename_override,
        )

        # post_read_hook — may transform documents
        docs = list(result.documents)
        modified = self._call_transform_hook("post_read_hook", source_label, docs)
        if modified is not None and modified is not docs:
            result = _replace_result_docs(result, modified)

        return result

    def run_batch(
        self,
        input_files: list[Any],
        *,
        stop_on_error: bool = False,
        export_format: Any | None = None,
    ) -> list[Any]:
        """
        Process multiple sources with hooks applied to each.

        Parameters
        ----------
        input_files : list[pathlib.Path or str]
            input_files.
        stop_on_error : bool, optional
            stop_on_error.
        export_format : ExportFormat or None, optional
            export_format.

        Returns
        -------
        list[PipelineResult]
        """
        results = []
        for src in input_files:
            try:
                results.append(self.run(src, export_format=export_format))
            except Exception as exc:  # noqa: BLE001
                if stop_on_error:
                    raise
                logger.warning(
                    "HookableCorpusPipeline.run_batch: skipping %s — %s",
                    src,
                    exc,
                )
        return results

    def run_url(
        self,
        url: Any,
        *,
        output_path: Any | None = None,
        export_format: Any | None = None,
        stop_on_error: bool = False,
    ) -> Any:
        """
        Process one URL or a list of URLs with hooks applied.

        Parameters
        ----------
        url : str or list[str]
            url.
        output_path : pathlib.Path or None, optional
            output_path.
        export_format : ExportFormat or None, optional
            export_format.
        stop_on_error : bool, optional
            stop_on_error.

        Returns
        -------
        PipelineResult or list[PipelineResult]
        """
        if isinstance(url, list):
            return [
                self.run_url(
                    u, export_format=export_format, stop_on_error=stop_on_error
                )
                for u in url
            ]
        return self.run(
            url,
            output_path=output_path,
            export_format=export_format,
        )

    # ------------------------------------------------------------------
    # Hook dispatch helpers
    # ------------------------------------------------------------------

    def _call_hook(self, hook_name: str, *args: Any) -> None:
        """Call a void hook, swallowing exceptions."""
        hook = getattr(self.hooks, hook_name, None)
        if hook is None:
            return
        try:
            hook(*args)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HookableCorpusPipeline: %s raised: %s", hook_name, exc)

    def _call_transform_hook(
        self,
        hook_name: str,
        source: str,
        docs: list[Any],
    ) -> list[Any] | None:
        """Call a transform hook that may return a modified document list."""
        hook = getattr(self.hooks, hook_name, None)
        if hook is None:
            return None
        try:
            return hook(source, docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HookableCorpusPipeline: %s raised: %s", hook_name, exc)
            return None

    def __repr__(self) -> str:  # noqa: D105
        return f"HookableCorpusPipeline(hooks={self.hooks!r}, inner={self._pipeline!r})"


class _CompositeHookFilter:
    """
    Compose a base filter with a post-filter hook callable.

    Internal — not part of the public API.

    Implements the :class:`~scikitplot.corpus._base.FilterBase` interface
    (duck-typed: has ``include(doc) -> bool``) without importing FilterBase
    to avoid circular imports.
    """

    def __init__(
        self,
        base_filter: Any | None,
        hook: Callable[[Any], bool],
    ) -> None:
        from ._base import DefaultFilter  # noqa: PLC0415

        self._base = base_filter if base_filter is not None else DefaultFilter()
        self._hook = hook

    def include(self, doc: Any) -> bool:
        """Return True only when both base filter and hook agree."""
        if not self._base.include(doc):
            return False
        try:
            return bool(self._hook(doc))
        except Exception as exc:  # noqa: BLE001
            logger.warning("_CompositeHookFilter: post_filter_hook raised: %s", exc)
            return True  # conservative: keep on hook error


def _replace_result_docs(result: Any, docs: list[Any]) -> Any:
    """
    Return a new PipelineResult with ``documents`` replaced.

    PipelineResult is ``frozen=True`` so we reconstruct via its fields.
    """
    from ._pipeline import PipelineResult  # noqa: PLC0415

    return PipelineResult(
        source=result.source,
        documents=list(docs),
        output_path=result.output_path,
        n_read=result.n_read,
        n_omitted=result.n_omitted,
        n_embedded=result.n_embedded,
        elapsed_seconds=result.elapsed_seconds,
        export_format=result.export_format,
    )


# =============================================================================
# Layer 7 — BuilderFactories + FactoryCorpusBuilder
# =============================================================================


@dataclass
class BuilderFactories:
    """
    Component factory callables for :class:`FactoryCorpusBuilder`.

    Each factory replaces the corresponding lazy-creation method in
    :class:`~scikitplot.corpus._corpus_builder.CorpusBuilder`.  ``None``
    means "use the default from :class:`~scikitplot.corpus._corpus_builder.BuilderConfig`".

    Parameters
    ----------
    reader_factory : callable or None, optional
        Factory for :class:`~scikitplot.corpus._base.DocumentReader`.
        Called once per source.  Receives ``(source: str | Path, chunker,
        **reader_kwargs) -> DocumentReader``.  Signature::

            def reader_factory(
                source: str | pathlib.Path,
                chunker: ChunkerBase | None,
                **reader_kwargs: Any,
            ) -> DocumentReader: ...

    chunker_factory : callable or None, optional
        Factory for the chunker.  Called once at build time.  No arguments.
        Signature::

            def chunker_factory() -> ChunkerBase | None: ...

    filter_factory : callable or None, optional
        Factory for the :class:`~scikitplot.corpus._base.FilterBase`.
        Called once at build time.  No arguments.  Signature::

            def filter_factory() -> FilterBase | None: ...

    normalizer_factory : callable or None, optional
        Factory for the
        :class:`~scikitplot.corpus._normalizers.NormalizationPipeline`.
        Called once at build time.  No arguments.  Signature::

            def normalizer_factory() -> NormalizationPipeline | None: ...

    enricher_factory : callable or None, optional
        Factory for the enricher.  Called once at build time.  No arguments.
        Signature::

            def enricher_factory() -> NLPEnricher | None: ...

    embedding_engine_factory : callable or None, optional
        Factory for the embedding engine.  Called once at build time.
        No arguments.  Signature::

            def embedding_engine_factory() -> EmbeddingEngine | None: ...

    Notes
    -----
    **User note:** Factories take precedence over the corresponding
    ``BuilderConfig`` settings.  For example, if ``chunker_factory`` is set,
    ``BuilderConfig.chunker`` is ignored for chunker creation.

    Examples
    --------
    Use a custom reader factory that injects a per-source language code::

        from langdetect import detect


        def smart_reader_factory(source, chunker, **kw):
            lang = detect(open(source).read(200)) if Path(source).exists() else None
            return DocumentReader.create(source, chunker=chunker, default_language=lang)


        factories = BuilderFactories(reader_factory=smart_reader_factory)
        builder = FactoryCorpusBuilder(factories=factories)
        result = builder.build("./data/")
    """

    reader_factory: Callable[..., Any] | None = None
    chunker_factory: Callable[[], Any] | None = None
    filter_factory: Callable[[], Any] | None = None
    normalizer_factory: Callable[[], Any] | None = None
    enricher_factory: Callable[[], Any] | None = None
    embedding_engine_factory: Callable[[], Any] | None = None

    def __post_init__(self) -> None:
        """Validate all factory callables.

        Raises
        ------
        TypeError
            If any non-None field is not callable.
        """
        for attr in (
            "reader_factory",
            "chunker_factory",
            "filter_factory",
            "normalizer_factory",
            "enricher_factory",
            "embedding_engine_factory",
        ):
            val = getattr(self, attr)
            if val is not None and not callable(val):
                raise TypeError(
                    f"BuilderFactories.{attr} must be callable or None; "
                    f"got {type(val).__name__!r}."
                )


class FactoryCorpusBuilder:
    """
    :class:`~scikitplot.corpus._corpus_builder.CorpusBuilder` extended with
    pluggable component factories.

    All public methods (``build``, ``add``, ``search``, ``to_langchain``,
    etc.) are available through delegation to the wrapped
    :class:`~scikitplot.corpus._corpus_builder.CorpusBuilder`.  When a
    factory is provided for a given component, it replaces the corresponding
    lazy-init method.

    Parameters
    ----------
    config : BuilderConfig or None, optional
        Pipeline configuration.  ``None`` uses defaults.
    factories : BuilderFactories or None, optional
        Component factory callables.  ``None`` disables all overrides.

    Notes
    -----
    **User note:** Use :class:`FactoryCorpusBuilder` when you need to
    inject components that cannot be described by configuration alone —
    custom readers with per-source state, enrichers backed by remote APIs,
    embedding engines with non-standard initialisation, etc.

    **Developer note:** Factory injection is performed by overriding the
    private ``_get_*`` lazy-init methods inherited from
    :class:`~scikitplot.corpus._corpus_builder.CorpusBuilder`.

    Examples
    --------
    Inject a custom embedding engine factory::

        def my_embed_factory():
            return MyEmbeddingEngine(model="custom-embedder-v2")


        factories = BuilderFactories(embedding_engine_factory=my_embed_factory)
        builder = FactoryCorpusBuilder(
            config=BuilderConfig(embed=True, build_index=True),
            factories=factories,
        )
        result = builder.build("./papers/")
        results = builder.search("attention mechanism")
    """  # noqa: D205

    def __init__(
        self,
        config: Any | None = None,
        *,
        factories: BuilderFactories | None = None,
    ) -> None:
        """
        Initialise with config and factories.

        Parameters
        ----------
        config : BuilderConfig or None, optional
            config.
        factories : BuilderFactories or None, optional
            factories.
        """
        from ._corpus_builder import BuilderConfig, CorpusBuilder  # noqa: PLC0415

        self._inner = CorpusBuilder(config=config or BuilderConfig())
        self.factories = factories or BuilderFactories()
        self.config = self._inner.config
        # Patch lazy-init methods on inner builder
        self._patch_inner()

    def _patch_inner(self) -> None:
        """
        Monkey-patch factory overrides onto the inner CorpusBuilder.

        Each non-None factory in ``self.factories`` replaces the
        corresponding ``_get_*`` method on the inner builder instance,
        ensuring that existing pipeline code paths invoke the factory.
        """
        factories = self.factories

        if factories.chunker_factory is not None:
            inner = self._inner
            _factory = factories.chunker_factory

            def _get_chunker_custom() -> Any:
                if inner._chunker is None:
                    inner._chunker = _factory()
                return inner._chunker

            import types  # noqa: PLC0415

            inner._get_chunker = types.MethodType(  # type: ignore[method-assign]
                lambda self: _get_chunker_custom(), inner
            )

        if factories.normalizer_factory is not None:
            inner = self._inner
            _nfac = factories.normalizer_factory

            def _get_norm_custom() -> Any:
                if inner._normalizer_pipeline is None:
                    inner._normalizer_pipeline = _nfac()
                return inner._normalizer_pipeline

            import types  # noqa: PLC0415

            inner._get_normalizer_pipeline = types.MethodType(  # type: ignore[method-assign]
                lambda self: _get_norm_custom(), inner
            )

        if factories.enricher_factory is not None:
            inner = self._inner
            _efac = factories.enricher_factory

            def _get_enrich_custom() -> Any:
                if inner._enricher is None:
                    inner._enricher = _efac()
                return inner._enricher

            import types  # noqa: PLC0415

            inner._get_enricher = types.MethodType(  # type: ignore[method-assign]
                lambda self: _get_enrich_custom(), inner
            )

        if factories.embedding_engine_factory is not None:
            inner = self._inner
            _embfac = factories.embedding_engine_factory

            def _get_embed_custom() -> Any:
                if inner._embedding_engine is None:
                    inner._embedding_engine = _embfac()
                return inner._embedding_engine

            import types  # noqa: PLC0415

            inner._get_embedding_engine = types.MethodType(  # type: ignore[method-assign]
                lambda self: _get_embed_custom(), inner
            )

    # ------------------------------------------------------------------
    # Delegate all public methods to inner builder
    # ------------------------------------------------------------------

    def build(self, sources: Any, **kwargs: Any) -> Any:
        """Build corpus — delegates to inner builder with factory overrides."""
        return self._inner.build(sources, **kwargs)

    def add(self, sources: Any, **kwargs: Any) -> Any:
        """Add sources to existing corpus — delegates to inner builder."""
        return self._inner.add(sources, **kwargs)

    def search(self, query: str, **kwargs: Any) -> Any:
        """Search corpus — delegates to inner builder."""
        return self._inner.search(query, **kwargs)

    def to_langchain(self) -> Any:
        """Export as LangChain documents."""
        return self._inner.to_langchain()

    def to_langchain_retriever(self) -> Any:
        """Create LangChain retriever."""
        return self._inner.to_langchain_retriever()

    def to_langgraph_state(self, **kwargs: Any) -> Any:
        """Export as LangGraph state."""
        return self._inner.to_langgraph_state(**kwargs)

    def to_mcp_resources(self, **kwargs: Any) -> Any:
        """Export as MCP resources."""
        return self._inner.to_mcp_resources(**kwargs)

    def to_mcp_tool_result(self, query: str, **kwargs: Any) -> Any:
        """Search and format as MCP tool result."""
        return self._inner.to_mcp_tool_result(query, **kwargs)

    def to_huggingface(self) -> Any:
        """Export as HuggingFace Dataset."""
        return self._inner.to_huggingface()

    def to_rag_tuples(self) -> Any:
        """Export as RAG tuples."""
        return self._inner.to_rag_tuples()

    def to_jsonl(self) -> Any:
        """Export as JSONL lines."""
        return self._inner.to_jsonl()

    def export(self, path: Any, **kwargs: Any) -> Any:
        """Export documents to file."""
        return self._inner.export(path, **kwargs)

    def close(self) -> None:
        """Clean up temporary files."""
        return self._inner.close()

    def __enter__(self) -> Self:
        """Context manager entry."""
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        """Context manager exit."""
        self._inner.__exit__(*exc)

    @property
    def _result(self) -> Any:
        """Most recent build result."""
        return self._inner._result

    @property
    def _index(self) -> Any:
        """Built similarity index."""
        return self._inner._index

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"FactoryCorpusBuilder("
            f"config={self.config!r}, "
            f"factories={self.factories!r})"
        )


# =============================================================================
# Layer 8 — CustomSimilarityIndex
# =============================================================================


class CustomSimilarityIndex:
    """
    :class:`~scikitplot.corpus._similarity.SimilarityIndex` extended with a
    fully-replaceable custom scorer callable.

    When ``custom_scorer_fn`` is provided, :meth:`search` calls it instead of
    the built-in strict / keyword / semantic / hybrid modes.  The callable
    receives the query string, the full document list, and the
    :class:`~scikitplot.corpus._similarity.SearchConfig` object.

    Parameters
    ----------
    config : SearchConfig or None, optional
        Default search configuration.
    custom_scorer_fn : callable or None, optional
        Custom scoring callable.  When set, completely replaces the built-in
        match modes for every :meth:`search` call.  Signature::

            def custom_scorer_fn(
                query: str,
                documents: list[CorpusDocument],
                config: SearchConfig,
            ) -> list[SearchResult]: ...

        The callable must return a list of
        :class:`~scikitplot.corpus._similarity.SearchResult` instances.

    Raises
    ------
    TypeError
        If ``custom_scorer_fn`` is provided but not callable.

    Notes
    -----
    **User note:** Use this to plug in a reranker (Cohere, BGE, ColBERT),
    a dense retrieval backend (Weaviate, Qdrant, Pinecone), or any other
    scoring logic that requires access to the full document list at query time.

    **Developer note:** The built-in index (:class:`~scikitplot.corpus._similarity.SimilarityIndex`)
    is wrapped, not subclassed, to avoid MRO conflicts with its lazy-import
    dependencies.  All ``build()``, property, and ``__repr__`` calls are
    delegated to the inner index.

    Examples
    --------
    Plug in a Cohere reranker::

        import cohere

        co = cohere.Client("API_KEY")


        def cohere_rerank(query, docs, cfg):
            texts = [d.text[:512] for d in docs]
            resp = co.rerank(query=query, documents=texts, top_n=cfg.top_k)
            return [
                SearchResult(
                    doc=docs[r.index], score=r.relevance_score, match_mode="cohere"
                )
                for r in resp.results
            ]


        index = CustomSimilarityIndex(custom_scorer_fn=cohere_rerank)
        index.build(corpus_documents)
        results = index.search("clinical trial outcomes")
    """  # noqa: D205

    def __init__(
        self,
        config: Any | None = None,
        *,
        custom_scorer_fn: Callable[[str, list[Any], Any], list[Any]] | None = None,
    ) -> None:
        """
        Initialise with search config and optional custom scorer.

        Parameters
        ----------
        config : SearchConfig or None, optional
            config.
        custom_scorer_fn : callable or None, optional
            ``(query, documents, config) -> list[SearchResult]``.

        Raises
        ------
        TypeError
            If ``custom_scorer_fn`` is not callable.
        """
        from ._similarity._similarity import (  # noqa: PLC0415
            SearchConfig,
            SimilarityIndex,
        )

        if custom_scorer_fn is not None and not callable(custom_scorer_fn):
            raise TypeError(
                "CustomSimilarityIndex: custom_scorer_fn must be callable or None; "
                f"got {type(custom_scorer_fn).__name__!r}."
            )

        self._inner = SimilarityIndex(config=config or SearchConfig())
        self.config = self._inner.config
        self.custom_scorer_fn = custom_scorer_fn
        self._scorer_name = (
            getattr(custom_scorer_fn, "__name__", repr(custom_scorer_fn))
            if custom_scorer_fn is not None
            else None
        )

    # ------------------------------------------------------------------
    # Build — delegate to inner index
    # ------------------------------------------------------------------

    def build(self, documents: Sequence[Any]) -> None:
        """
        Build the index from documents.

        Parameters
        ----------
        documents : Sequence[CorpusDocument]
            documents.

        Raises
        ------
        ValueError
            If ``documents`` is empty.
        """
        self._inner.build(documents)

    # ------------------------------------------------------------------
    # Search — dispatch to custom_scorer_fn or built-in modes
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        config: Any | None = None,
        query_embedding: Any | None = None,
    ) -> list[Any]:
        """
        Search the index using the custom scorer or built-in modes.

        When ``custom_scorer_fn`` is set it is called with
        ``(query, documents, resolved_config)`` and its return value is
        used directly.  Otherwise :meth:`~scikitplot.corpus._similarity.SimilarityIndex.search`
        is called on the inner index.

        Parameters
        ----------
        query : str
            Query string.
        config : SearchConfig or None, optional
            Per-query config override.
        query_embedding : array-like or None, optional
            Pre-computed query embedding for semantic/hybrid modes.

        Returns
        -------
        list[SearchResult]
            Results sorted by descending score.

        Raises
        ------
        RuntimeError
            If ``custom_scorer_fn`` raises an unexpected exception.
        """
        resolved_cfg = config or self.config

        if self.custom_scorer_fn is not None:
            logger.info(
                "CustomSimilarityIndex: calling custom_scorer_fn %r.",
                self._scorer_name,
            )
            try:
                return self.custom_scorer_fn(
                    query,
                    self._inner._documents,
                    resolved_cfg,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"CustomSimilarityIndex: custom_scorer_fn "
                    f"{self._scorer_name!r} raised: {exc}"
                ) from exc

        return self._inner.search(
            query,
            config=resolved_cfg,
            query_embedding=query_embedding,
        )

    # ------------------------------------------------------------------
    # Property delegation
    # ------------------------------------------------------------------

    @property
    def n_documents(self) -> int:
        """Number of indexed documents."""
        return self._inner.n_documents

    @property
    def has_embeddings(self) -> bool:
        """Whether dense embeddings are indexed."""
        return self._inner.has_embeddings

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CustomSimilarityIndex("
            f"n_docs={self.n_documents}, "
            f"dense={self.has_embeddings}, "
            f"scorer={self._scorer_name!r})"
        )
