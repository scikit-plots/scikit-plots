# scikitplot/corpus/_chunkers/_semantic.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
scikitplot.corpus._chunkers._semantic
=====================================

Layer 3 — :class:`SemanticChunker`: embedding-aware, morphology-aware,
and writing-system-aware text chunking.

Architecture position::

    Layer 0: GraphemeClusterNormalizer    (in _normalizers/)
    Layer 1: ScriptSegmenter              (in _custom_tokenizer.py)
    Layer 2: WritingSystemAdapter         (in _writing_system.py)
    Layer 3: SemanticChunker              ← this file

Three backends (selectable via :class:`SemanticBackend`):

MORPHOLOGICAL
    Runs Layer 1 → Layer 2 only.  Linguistic quality, fully deterministic.
    No embeddings.  For CAG/BM25 retrieval, offline annotation, or any
    pipeline where reproducibility is required.

EMBEDDING
    Embeds candidate splits and merges adjacent splits whose cosine
    similarity exceeds ``similarity_threshold``.  Uses
    ``sentence-transformers`` (``pip install sentence-transformers``).
    Non-deterministic (model weights may change across releases).

HYBRID (default — Q1 decision)
    Runs MORPHOLOGICAL first to get high-quality linguistic splits, then
    applies EMBEDDING boundary refinement.  Both paths are mandatory.
    Provides the best RAG retrieval quality with linguistic grounding.

Design decisions (from final review):

* Q1 → HYBRID default.
* Q3 → All-language coverage via ScriptSegmenter ``\\p{Script=X}`` + fallback chain.
* Q4 → JapaneseStrategy probe chain in WritingSystemAdapter.
* Q5 → WritingSystemAdapter injected as constructor parameter (DI pattern).

Python compatibility: 3.8-3.15.
``from __future__ import annotations`` for all annotations.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
import threading
import traceback
from dataclasses import dataclass, field, replace  # noqa: F401
from enum import Enum, unique
from typing import Any, Callable, Final, Optional, Sequence  # noqa: F401

from .._types import (  # noqa: F401
    Chunk,
    ChunkerConfig,
    ChunkResult,
    EmbeddingVector,
    MetadataDict,
    MultilangChunkMeta,
    PreprocessingTrace,
    SemantemeInfo,
)
from ._custom_tokenizer import ScriptSegmenter, ScriptType, detect_script  # noqa: F401
from ._multilang_mixin import MultilangConfig, MultilangMixin
from ._sentence import _validate_text_input
from ._spans import cover, locate
from ._writing_system import (
    WritingSystemAdapter,
    WritingSystemAdapterConfig,
    check_adapter_unit,
)

logger = logging.getLogger(__name__)

__all__: Final[list[str]] = [
    "SemanticBackend",
    "SemanticChunker",
    "SemanticChunkerConfig",
]


# ===========================================================================
# Section 1 — SemanticBackend enum
# ===========================================================================


@unique
class SemanticBackend(str, Enum):
    """Semantic chunking backend selector.

    Attributes
    ----------
    MORPHOLOGICAL
        Layer 1 + Layer 2 only.  No embeddings.  Fully deterministic.
        Best for CAG, BM25 retrieval, offline annotation.
    EMBEDDING
        Pure embedding-based boundary detection.  Requires
        ``sentence-transformers``.  Non-deterministic.
    HYBRID
        MORPHOLOGICAL first, then EMBEDDING boundary refinement.
        Default — provides linguistic grounding + RAG quality.
    """

    MORPHOLOGICAL = "morphological"
    EMBEDDING = "embedding"
    HYBRID = "hybrid"


# ===========================================================================
# Section 2 — SemanticChunkerConfig
# ===========================================================================


@dataclass(frozen=True)
class SemanticChunkerConfig(ChunkerConfig):
    """Configuration for :class:`SemanticChunker`.

    Parameters
    ----------
    backend : SemanticBackend
        Chunking backend.  Default :attr:`SemanticBackend.HYBRID`.
    model_name : str
        Sentence-transformers model name used for EMBEDDING / HYBRID
        backends.  Default ``"paraphrase-multilingual-mpnet-base-v2"``
        (supports 50+ languages, 768-dim).
    model_version : str or None
        Model version tag for ``script_model_version`` provenance field.
        Default ``None`` (auto-populated from installed package version
        when available).
    similarity_threshold : float
        Cosine similarity threshold for EMBEDDING / HYBRID boundary merging.
        Adjacent chunks with similarity >= this value are merged.
        Range ``[0.0, 1.0]``.  Default ``0.85``.
    max_chunk_tokens : int or None
        Hard upper bound on tokens per semantic chunk.  Chunks exceeding
        this are forcibly split by the Layer 2 adapter.
        Default ``None`` (no limit).
    min_chunk_tokens : int
        Minimum token count.  Chunks below this are merged into the
        adjacent larger chunk.  Default ``1``.
    include_offsets : bool
        Compute and store character offsets.  Default ``True``.
    adapter_unit : str
        Layer 2 segmentation granularity: ``"word"``, ``"sentence"``,
        ``"grapheme_cluster"``.  Default ``"word"``.
    multilang_config : MultilangConfig or None
        Multilang feature flags for the mixin.  Default ``None``
        → ``MultilangConfig(include_semantemes=True)``.
    device : str
        Torch device for embeddings: ``"cpu"``, ``"cuda"``, ``"mps"``.
        Default ``"cpu"``.
    allow_fallback : bool
        When ``sentence-transformers`` is unavailable, the model fails to
        load, or the model fails while refining boundaries, EMBEDDING and
        HYBRID fall back to MORPHOLOGICAL (``True``, default) or raise
        (``False``): ``ImportError`` for the missing package, ``RuntimeError``
        (chained from the cause) for a load failure, the model's own exception
        for an encode failure. Either way the result reports what actually
        ran: ``metadata["backend"]`` is the *effective* backend,
        ``metadata["requested_backend"]`` the configured one, and
        ``metadata["fallback_reason"]`` why they differ
        (``"sentence_transformers_unavailable"``, ``"model_load_failed"`` or
        ``"embedding_failed"``).

    Notes
    -----
    **Developer note:** ``model_name`` and ``model_version`` are recorded
    only when the model was called for the text: in the result metadata
    (``model_name``, ``model_version``, and ``model_invoked``) and in each
    chunk's ``metadata["multilang"]`` (``model_name``,
    ``script_model_version`` = ``"name@version"`` or ``"name"``).
    ``script_model_version`` is the idempotency verification field -- set
    ``model_version`` when running EMBEDDING or HYBRID to detect
    model-version drift across pipeline re-runs. A MORPHOLOGICAL run,
    requested or fallen back to, calls no model, and neither does a text
    with a single unit (nothing to refine: the requested backend is
    reported, ``model_invoked`` is ``False``). A model you declare yourself in
    ``multilang_config.embedding_model_name`` is recorded on every chunk as
    declared, as in every chunker with multilang metadata; when the semantic
    model is called, its name replaces the declared one.
    """

    backend: SemanticBackend = SemanticBackend.HYBRID
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    model_version: str | None = None
    similarity_threshold: float = 0.85
    max_chunk_tokens: int | None = None
    min_chunk_tokens: int = 1
    include_offsets: bool = True
    adapter_unit: str = "word"
    multilang_config: Any = None  # MultilangConfig or None
    device: str = "cpu"
    allow_fallback: bool = True

    def __post_init__(self) -> None:
        """Coerce enum fields (R3-2) and refuse an unknown ``adapter_unit``."""
        super().__post_init__()
        check_adapter_unit(self.adapter_unit, "SemanticChunkerConfig.adapter_unit")


# ===========================================================================
# Section 3 — SemanticChunker
# ===========================================================================


class SemanticChunker(MultilangMixin):
    r"""Embedding-aware, writing-system-aware semantic chunker (Layer 3).

    Combines Layer 1 (:class:`~._custom_tokenizer.ScriptSegmenter`) and
    Layer 2 (:class:`~._writing_system.WritingSystemAdapter`) for
    linguistically correct segmentation across all Unicode scripts, then
    optionally refines boundaries with sentence-transformers embeddings.

    Parameters
    ----------
    config : SemanticChunkerConfig, optional
        Full configuration.  Default: HYBRID backend, multilingual mpnet.
    adapter : WritingSystemAdapter or None, optional
        Layer 2 adapter injected via Dependency Injection (Q5).
        When ``None``, a default adapter is constructed from
        ``config.adapter_unit``.  Pass a custom adapter to swap
        per-script strategies without subclassing.

    Notes
    -----
    **User note:** Install ``sentence-transformers`` and ``regex`` for
    full HYBRID functionality::

        pip install sentence-transformers regex

    For MORPHOLOGICAL-only mode (no embeddings, fully offline), set
    ``backend=SemanticBackend.MORPHOLOGICAL``.

    **User note (raw text):** Set
    ``multilang_config=MultilangConfig(include_raw_text=True,
    include_preprocessing_trace=True)`` to preserve the pre-NFC raw text
    alongside the normalised text in each chunk for comparison.

    **Developer note (Q5 — Dependency Injection):** ``adapter`` is a
    constructor parameter, not an internal attribute.  This makes
    :class:`SemanticChunker` fully testable without touching the global
    strategy registry, and composable with custom strategies.

    **Developer note (embedding hook):** Call
    :meth:`attach_embedding_batch` after chunking to add vectors to
    existing chunks without re-running the full pipeline.

    Examples
    --------
    >>> chunker = SemanticChunker()
    >>> result = chunker.chunk("Hello world. مرحبا بالعالم。")
    >>> result.chunks[0].metadata["multilang"]["script"]
    'latin'

    >>> # MORPHOLOGICAL only — no network, no GPU
    >>> from ._semantic import SemanticBackend, SemanticChunkerConfig
    >>> cfg = SemanticChunkerConfig(backend=SemanticBackend.MORPHOLOGICAL)
    >>> chunker_morph = SemanticChunker(cfg)

    >>> # Inject custom adapter
    >>> from ._writing_system import WritingSystemAdapter, WritingSystemAdapterConfig
    >>> adapter = WritingSystemAdapter(WritingSystemAdapterConfig(unit="sentence"))
    >>> chunker_sent = SemanticChunker(adapter=adapter)
    """

    #: The configuration dataclass this chunker takes as ``config=``; the
    #: component registry pairs them by it (R3-1).
    config_class = SemanticChunkerConfig

    #: Whether each chunk's text is ``source[context_span]`` (R3-3).
    text_is_source_slice = True

    def __init__(
        self,
        config: SemanticChunkerConfig | None = None,
        *,
        adapter: WritingSystemAdapter | None = None,
    ) -> None:
        if config is not None and not isinstance(config, SemanticChunkerConfig):
            raise TypeError(
                f"SemanticChunker: config must be a SemanticChunkerConfig or None, "
                f"got {type(config).__name__}."
            )
        self._cfg = config if config is not None else SemanticChunkerConfig()
        self._validate_config()
        # Accept the enum or its value ("hybrid"): a string used to pass
        # construction and then fail at ``backend.value`` inside ``chunk``.
        try:
            self._backend = SemanticBackend(self._cfg.backend)
        except ValueError:
            raise ValueError(
                f"SemanticChunkerConfig.backend must be one of "
                f"{[b.value for b in SemanticBackend]}; got {self._cfg.backend!r}."
            ) from None

        # Layer 2 adapter — DI pattern (Q5)
        if adapter is not None:
            self._adapter = adapter
        else:
            self._adapter = WritingSystemAdapter(
                WritingSystemAdapterConfig(unit=self._cfg.adapter_unit)
            )

        # Multilang mixin init. The boundary model is deliberately not written
        # into this config: whether it runs is known only inside ``chunk``
        # (Step 6), which records it on each chunk when, and only when, it
        # ran. Writing it here stamped every chunk -- MORPHOLOGICAL and
        # fallen-back runs included -- with a model that never ran.
        ml_cfg = (
            self._cfg.multilang_config
            if isinstance(getattr(self._cfg, "multilang_config", None), MultilangConfig)
            else MultilangConfig(include_semantemes=True)
        )
        self._ml_init(ml_cfg)

        # Embedding model — lazy loaded on first use
        self._embed_model: Any = None
        self._embed_model_loaded: bool = False
        # Why the model is not loaded: (fallback_reason, cause). The cause is
        # None when the package is missing, else the exception the model raised.
        self._model_failure: tuple[str, BaseException | None] | None = None
        # Serialises the first load: a second thread waits for its outcome
        # instead of reading a half-set state.
        self._model_lock = threading.Lock()
        self._fallback_reported: bool = False

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> SemanticChunkerConfig:
        """The resolved :class:`SemanticChunkerConfig` for this instance (read-only)."""
        return self._cfg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_config(self) -> None:
        """Validate SemanticChunkerConfig at construction time.

        Raises
        ------
        ValueError
            On invalid threshold or token count values.
        """
        if not (0.0 <= self._cfg.similarity_threshold <= 1.0):
            raise ValueError(
                f"SemanticChunkerConfig.similarity_threshold must be in "
                f"[0.0, 1.0], got {self._cfg.similarity_threshold!r}."
            )
        if self._cfg.min_chunk_tokens < 1:
            raise ValueError(
                f"SemanticChunkerConfig.min_chunk_tokens must be >= 1, "
                f"got {self._cfg.min_chunk_tokens!r}."
            )
        if (
            self._cfg.max_chunk_tokens is not None
            and self._cfg.max_chunk_tokens < self._cfg.min_chunk_tokens
        ):
            raise ValueError(
                "SemanticChunkerConfig.max_chunk_tokens must be >= "
                "min_chunk_tokens or None. "
                f"Got max={self._cfg.max_chunk_tokens}, "
                f"min={self._cfg.min_chunk_tokens}."
            )

    # ------------------------------------------------------------------
    # Section A — Embedding model (lazy, probe-once)
    # ------------------------------------------------------------------

    def _load_embed_model(self) -> bool:
        """Load the sentence-transformers model once.

        Returns
        -------
        bool
            ``True`` if the model is loaded. ``False`` if it is not, with the
            reason in ``self._model_failure``: ``sentence-transformers`` is not
            installed, or the model itself failed to load (a wrong name, no
            network for the first download, ...).

        Notes
        -----
        **Developer.** Both failures are remembered for the life of the
        chunker, so a fallback does not retry a download on every ``chunk``.
        ``chunk`` reports them: the reason is part of the result, and with
        ``allow_fallback=False`` they raise. A load failure used to escape on
        the first call only; later calls then fell back under the reason
        "sentence_transformers_unavailable", with the package installed.

        The outcome is published only once it is known, under a lock: the
        "loaded" flag used to be set first, so an import failing with another
        exception, an interrupted first download, or a second thread arriving
        mid-load left a chunker whose every later call raised ``TypeError``.
        ``KeyboardInterrupt`` and ``SystemExit`` are not outcomes: the next
        call tries again. A remembered exception keeps its traceback but not
        the frames' locals, which would hold a half-loaded model.
        """
        with self._model_lock:
            if self._embed_model_loaded:
                return self._embed_model is not None
            try:
                from sentence_transformers import (  # type: ignore[import-not-found]  # noqa: PLC0415
                    SentenceTransformer,
                )
            except ImportError:
                self._model_failure = ("sentence_transformers_unavailable", None)
            except Exception as exc:  # noqa: BLE001 - a broken install; see ``chunk``
                traceback.clear_frames(exc.__traceback__)
                self._model_failure = ("model_load_failed", exc)
            else:
                try:
                    self._embed_model = SentenceTransformer(
                        self._cfg.model_name,
                        device=self._cfg.device,
                    )
                except Exception as exc:  # noqa: BLE001 - reported by ``chunk``
                    traceback.clear_frames(exc.__traceback__)
                    self._model_failure = ("model_load_failed", exc)
                else:
                    logger.debug(
                        "SemanticChunker: loaded model %r on device %r.",
                        self._cfg.model_name,
                        self._cfg.device,
                    )
            self._embed_model_loaded = True
            return self._embed_model is not None

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, returning a parallel list of vectors.

        Parameters
        ----------
        texts : list[str]
            Non-empty texts.

        Returns
        -------
        list[list[float]]
            One vector per text.

        Raises
        ------
        RuntimeError
            If the embedding model is not loaded.
        """
        if self._embed_model is None:
            raise RuntimeError(
                "SemanticChunker._embed_texts: model not loaded. "
                "Call _load_embed_model() first."
            )
        vectors = self._embed_model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=False,
        )
        return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

    # ------------------------------------------------------------------
    # Section B — Cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two dense vectors.

        Parameters
        ----------
        a : list[float]
            Vector A.
        b : list[float]
            Vector B.

        Returns
        -------
        float
            Cosine similarity in ``[-1.0, 1.0]``.
        """
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ------------------------------------------------------------------
    # Section C — Boundary refinement (EMBEDDING / HYBRID)
    # ------------------------------------------------------------------

    def _refine_boundaries(
        self,
        chunks: list[Chunk],
        source: str,
    ) -> list[Chunk]:
        """Merge adjacent chunks with high embedding similarity.

        Adjacent chunks whose cosine similarity >= ``similarity_threshold``
        are merged into a single chunk.  This reduces over-segmentation
        from the morphological layer while preserving semantically distinct
        boundaries.

        Parameters
        ----------
        chunks : list[Chunk]
            Morphological chunks from Layer 2.
        source : str
            The text the chunks' spans index; a merged chunk's text is its
            slice (R3-3), not the parts joined with ``" "``.

        Returns
        -------
        list[Chunk]
            Boundary-refined list (always non-empty for non-empty input).

        Raises
        ------
        Exception
            Whatever the embedding model raises. :meth:`chunk` decides
            between falling back (and reporting it) and re-raising; this
            method used to return the unrefined chunks itself, so a run whose
            model failed was still reported as EMBEDDING/HYBRID.
        """
        if len(chunks) <= 1:
            return chunks

        vectors = self._embed_texts([c.text for c in chunks])

        merged: list[Chunk] = [chunks[0]]
        merged_vecs: list[list[float]] = [vectors[0]]

        for i in range(1, len(chunks)):
            sim = self._cosine_similarity(merged_vecs[-1], vectors[i])
            if sim >= self._cfg.similarity_threshold:
                # Merge: the source from the start of the first to the end of
                # the last (R3-3; a " " join is not the source in CJK or
                # across newlines).
                prev = merged[-1]
                curr = chunks[i]
                new_text = source[prev.start_char : curr.end_char]
                new_meta = {
                    **dict(prev.metadata),
                    "merged_from": [
                        prev.metadata.get("chunk_index", -1),
                        curr.metadata.get("chunk_index", i),
                    ],
                    "merge_similarity": round(sim, 4),
                }
                merged[-1] = Chunk(
                    text=new_text,
                    start_char=prev.start_char,
                    end_char=curr.end_char,
                    metadata=new_meta,
                )
                # Update the merged vector as the average
                merged_vecs[-1] = [
                    (a + b) / 2 for a, b in zip(merged_vecs[-1], vectors[i])
                ]
            else:
                merged.append(chunks[i])
                merged_vecs.append(vectors[i])

        return merged

    # ------------------------------------------------------------------
    # Section D — Public chunk() API
    # ------------------------------------------------------------------

    def chunk(  # noqa: PLR0912
        self,
        text: str,
        doc_id: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ChunkResult:
        """Segment *text* using Layer 1 → Layer 2 → optional Layer 3.

        Parameters
        ----------
        text : str
            NFC-normalised (or raw, will be normalised internally) text.
        doc_id : str, optional
            Document identifier stored in each chunk's metadata.
        extra_metadata : dict[str, Any], optional
            Extra key/value pairs merged into result metadata.

        Returns
        -------
        ChunkResult
            Ordered chunks with multilang metadata.

        Raises
        ------
        TypeError
            If *text* is not a ``str``.
        ValueError
            If *text* is empty or whitespace-only, or contains NUL bytes
            or lone surrogates.
        ImportError
            If EMBEDDING or HYBRID is configured, ``sentence-transformers``
            is not installed and ``allow_fallback=False``.
        RuntimeError
            If the model failed to load and ``allow_fallback=False`` (chained
            from the load error).
        Exception
            Whatever the embedding model raised while encoding, when
            ``allow_fallback=False``.

        Notes
        -----
        **Algorithm:**

        1. Input validation (NUL / surrogate guard).
        2. Multilang preprocessing trace (BOM strip → control strip → NFC).
        3. Layer 1: ScriptSegmenter → ScriptSpan list.
        4. Layer 2: WritingSystemAdapter → per-script Chunk list.
        5. Min/max token filtering and enforcement.
        6. EMBEDDING / HYBRID: boundary refinement via cosine similarity; a
           missing package, a model that fails to load or a model that fails
           to encode falls back to MORPHOLOGICAL (see
           ``SemanticChunkerConfig.allow_fallback``).
        7. MultilangMixin: enrich each chunk with MultilangChunkMeta; the
           model is recorded only when it was called for this text.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__!r}.")
        _validate_text_input(text, "SemanticChunker.chunk")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace-only.")

        # ── Step 2: Preprocessing trace ───────────────────────────────────
        raw_text = text
        preprocessing_trace: PreprocessingTrace | None = None
        if self._ml_cfg.enabled and (
            self._ml_cfg.include_preprocessing_trace or self._ml_cfg.include_raw_text
        ):
            text, preprocessing_trace = self._ml_build_preprocessing_trace(raw_text)
        # R3-3: spans index the text that was split. When preprocessing
        # (NFC, BOM/control stripping) changed it, they index that text, not
        # the input: say so, and slice no "raw" text with them (review F2).
        span_basis = "input" if text == raw_text else "preprocessed"

        # ── Step 3+4: Layer 1 → Layer 2 ──────────────────────────────────
        layer2_chunks = self._adapter.adapt_text(text)

        # ── Step 5: Token count filtering ────────────────────────────────
        if self._cfg.min_chunk_tokens > 1 or self._cfg.max_chunk_tokens is not None:
            layer2_chunks = self._enforce_token_limits(layer2_chunks, text)

        # ── Step 6: Boundary refinement ───────────────────────────────────
        # ``requested`` is configuration; ``backend`` is what actually ran.
        # Results used to report the requested backend (and a model name)
        # after falling back -- for a missing package and for a model that
        # failed at encode time alike -- so a morphological run read as
        # "semantic_hybrid".
        requested = self._backend
        backend = requested
        fallback_reason: str | None = None
        model_invoked = False
        if requested in (SemanticBackend.EMBEDDING, SemanticBackend.HYBRID):
            if not self._load_embed_model():
                fallback_reason, cause = self._model_failure
                if cause is None:
                    problem = (
                        "sentence-transformers is not installed. Install it"
                        " with: pip install sentence-transformers"
                    )
                else:
                    problem = (
                        f"model {self._cfg.model_name!r} failed to load"
                        f" ({type(cause).__name__}: {cause})"
                    )
                if not self._cfg.allow_fallback:
                    message = (
                        f"SemanticChunker: backend {requested.value!r} cannot"
                        f" run and allow_fallback=False: {problem}"
                    )
                    if cause is None:
                        raise ImportError(message)
                    raise RuntimeError(message) from cause
                if not self._fallback_reported:
                    logger.warning(
                        "SemanticChunker: backend %r runs as 'morphological'"
                        " (reported in result metadata): %s",
                        requested.value,
                        problem,
                    )
                    self._fallback_reported = True
            elif len(layer2_chunks) > 1:
                # With one unit there is nothing to refine: the model is not
                # called, so it is not named either (see ``model_invoked``).
                try:
                    layer2_chunks = self._refine_boundaries(layer2_chunks, text)
                    model_invoked = True
                except Exception as exc:  # noqa: BLE001 - any model failure
                    if not self._cfg.allow_fallback:
                        raise
                    fallback_reason = "embedding_failed"
                    logger.warning(
                        "SemanticChunker: the embedding model failed (%s: %s);"
                        " backend %r ran as 'morphological' for this text"
                        " (reported in result metadata).",
                        type(exc).__name__,
                        exc,
                        requested.value,
                    )
            if fallback_reason is not None:
                backend = SemanticBackend.MORPHOLOGICAL
        # A model is provenance only when it was called for this text.
        model_name = self._cfg.model_name if model_invoked else None
        model_version = self._cfg.model_version if model_invoked else None
        model_tag = (
            f"{model_name}@{model_version}"
            if model_name and model_version
            else model_name
        )

        # ── Step 7: Multilang enrichment ──────────────────────────────────
        final_chunks: list[Chunk] = []
        for idx, ch in enumerate(layer2_chunks):
            chunk_meta = dict(ch.metadata)
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunking_strategy"] = f"semantic_{backend.value}"
            if doc_id is not None:
                chunk_meta["doc_id"] = doc_id

            # Honour include_offsets (review F7): (0, 0) when disabled.
            start, end = (
                (ch.start_char, ch.end_char) if self._cfg.include_offsets else (0, 0)
            )
            enriched = Chunk(
                text=ch.text,
                start_char=start,
                end_char=end,
                metadata=chunk_meta,
            )

            if self._ml_cfg.enabled:
                chunk_raw = (
                    raw_text[ch.start_char : ch.end_char]
                    if self._ml_cfg.include_raw_text
                    and span_basis == "input"
                    and 0 <= ch.start_char < ch.end_char <= len(raw_text)
                    else None
                )
                # Build morpheme_map from multilang metadata if available
                tokens = ch.text.split()
                ml_meta = self._ml_build_meta(
                    ch.text,
                    chunking_unit="semanteme",
                    tokens=tokens,
                    raw_text=chunk_raw,
                    preprocessing_trace=preprocessing_trace,
                    # Propagate Layer 2 strategy name for provenance tracking.
                    # WritingSystemAdapter.adapt() stores this in ch.metadata.
                    layer2_strategy=(
                        ch.metadata.get("layer2_strategy") if ch.metadata else None
                    ),
                )
                if model_invoked:
                    # The boundary model ran: it is this chunk's provenance.
                    # (A per-chunk ``encode`` used to run here, one call per
                    # chunk, only to set these two fields: ``_ml_enrich_chunk``
                    # drops the vector. Vectors are attached on request with
                    # ``attach_embedding_batch``.)
                    ml_meta = replace(
                        ml_meta,
                        model_name=model_name,
                        script_model_version=model_tag,
                    )

                enriched = self._ml_enrich_chunk(enriched, ml_meta)

            final_chunks.append(enriched)

        result_meta: dict[str, Any] = {
            "span_basis": span_basis,
            "chunker": "semantic",
            "backend": backend.value,
            "requested_backend": requested.value,
            "fallback_reason": fallback_reason,
            "model_name": model_name,
            "model_version": model_version,
            "model_invoked": model_invoked,
            "similarity_threshold": self._cfg.similarity_threshold,
            "total_chunks": len(final_chunks),
        }
        if doc_id is not None:
            result_meta["doc_id"] = doc_id
        if extra_metadata:
            result_meta.update(extra_metadata)

        return ChunkResult(chunks=final_chunks, metadata=result_meta)

    # ------------------------------------------------------------------
    # Section E — Token limit enforcement
    # ------------------------------------------------------------------

    def _enforce_token_limits(self, chunks: list[Chunk], source: str) -> list[Chunk]:
        """Enforce min/max token limits on Layer 2 chunks.

        Parameters
        ----------
        chunks : list[Chunk]
            Input chunk list.
        source : str
            The text the chunks' spans index (merged and split chunks are
            slices of it, R3-3).

        Returns
        -------
        list[Chunk]
            Filtered / split chunk list.  Always non-empty for non-empty input.
        """
        result: list[Chunk] = []
        merge_buf: list[Chunk] = []

        for ch in chunks:
            token_count = len(ch.text.split())

            # Split oversized chunks
            if (
                self._cfg.max_chunk_tokens is not None
                and token_count > self._cfg.max_chunk_tokens
            ):
                sub_chunks = self._split_oversized(ch, source)
                for sub in sub_chunks:
                    self._maybe_flush_buf(source, merge_buf, result)
                    merge_buf = []
                    result.append(sub)
                continue

            # Buffer undersized chunks for merging
            if token_count < self._cfg.min_chunk_tokens:
                merge_buf.append(ch)
            else:  # noqa: PLR5501
                if merge_buf:
                    result.append(self._merge_buf(source, merge_buf, ch))
                    merge_buf = []
                else:
                    result.append(ch)

        # Flush remaining buffer into the last chunk
        if merge_buf:
            if result:
                result[-1] = self._merge_buf(source, [result[-1]], *merge_buf)
            else:
                result.append(self._merge_buf(source, merge_buf))

        return result or chunks

    def _split_oversized(self, chunk: Chunk, source: str) -> list[Chunk]:
        """Split an oversized chunk into pieces of at most max_chunk_tokens.

        Each piece is the source from its first token to its last (R3-3):
        pieces used to be the tokens joined with ``" "`` and every piece
        carried the whole chunk's span.
        """
        max_t = self._cfg.max_chunk_tokens or 9999
        start, end = chunk.start_char, chunk.end_char
        region = source[start:end]
        words = region.split()
        spans = [
            (start + s, start + e)
            for s, e in locate(region, words, where="SemanticChunker")
        ]
        result: list[Chunk] = []
        for i in range(0, len(spans), max_t):
            piece_start, piece_end = cover(spans[i : i + max_t])
            result.append(
                Chunk(
                    text=source[piece_start:piece_end],
                    start_char=piece_start,
                    end_char=piece_end,
                    metadata={**dict(chunk.metadata), "oversized_split": True},
                )
            )
        return result or [chunk]

    @staticmethod
    def _merge_buf(source: str, buf: list[Chunk], *extra: Chunk) -> Chunk:
        """Merge a buffer of chunks into one: the source slice they cover."""
        all_chunks = list(buf) + list(extra)
        start, end = all_chunks[0].start_char, all_chunks[-1].end_char
        return Chunk(
            text=source[start:end],
            start_char=start,
            end_char=end,
            metadata={**dict(all_chunks[0].metadata), "undersized_merge": True},
        )

    @staticmethod
    def _maybe_flush_buf(source: str, buf: list[Chunk], result: list[Chunk]) -> None:
        if buf:
            result.append(SemanticChunker._merge_buf(source, buf))

    # ------------------------------------------------------------------
    # Section F — Batch API
    # ------------------------------------------------------------------

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
            Shared metadata for every result.

        Returns
        -------
        list[ChunkResult]
            One result per document.

        Raises
        ------
        TypeError
            If *texts* is not a list.
        ValueError
            If *doc_ids* length mismatches *texts*.
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
                doc_id=doc_ids[i] if doc_ids else None,
                extra_metadata=extra_metadata,
            )
            for i, t in enumerate(texts)
        ]
