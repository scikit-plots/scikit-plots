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
from dataclasses import dataclass, field  # noqa: F401
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
from ._writing_system import (
    WritingSystemAdapter,
    WritingSystemAdapterConfig,
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

    Notes
    -----
    **Developer note:** ``model_name`` and ``model_version`` populate
    ``CorpusDocument.script_model_version`` via ``MultilangChunkMeta``.
    This is the idempotency verification field — always set it when
    running EMBEDDING or HYBRID to detect model-version drift across
    pipeline re-runs.
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

    def __init__(
        self,
        config: SemanticChunkerConfig | None = None,
        *,
        adapter: WritingSystemAdapter | None = None,
    ) -> None:
        self._cfg = config if config is not None else SemanticChunkerConfig()
        self._validate_config()

        # Layer 2 adapter — DI pattern (Q5)
        if adapter is not None:
            self._adapter = adapter
        else:
            self._adapter = WritingSystemAdapter(
                WritingSystemAdapterConfig(unit=self._cfg.adapter_unit)
            )

        # Multilang mixin init
        ml_cfg = (
            self._cfg.multilang_config
            if isinstance(getattr(self._cfg, "multilang_config", None), MultilangConfig)
            else MultilangConfig(include_semantemes=True)
        )
        # Override model fields from config
        if ml_cfg.embedding_model_name is None and self._cfg.model_name:
            ml_cfg = MultilangConfig(
                enabled=ml_cfg.enabled,
                include_raw_text=ml_cfg.include_raw_text,
                include_preprocessing_trace=ml_cfg.include_preprocessing_trace,
                include_semantemes=ml_cfg.include_semantemes,
                include_script_spans=ml_cfg.include_script_spans,
                include_grapheme_counts=ml_cfg.include_grapheme_counts,
                include_embedding_hook=ml_cfg.include_embedding_hook,
                embedding_model_name=self._cfg.model_name,
                embedding_model_version=self._cfg.model_version,
                language_hint=ml_cfg.language_hint,
            )
        self._ml_init(ml_cfg)

        # Embedding model — lazy loaded on first use
        self._embed_model: Any = None
        self._embed_model_loaded: bool = False

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
            ``True`` if model is available and loaded; ``False`` if
            ``sentence-transformers`` is not installed.
        """
        if self._embed_model_loaded:
            return self._embed_model is not None
        self._embed_model_loaded = True
        try:
            from sentence_transformers import (  # type: ignore[import-not-found]  # noqa: PLC0415
                SentenceTransformer,
            )

            self._embed_model = SentenceTransformer(
                self._cfg.model_name,
                device=self._cfg.device,
            )
            logger.debug(
                "SemanticChunker: loaded model %r on device %r.",
                self._cfg.model_name,
                self._cfg.device,
            )
            return True
        except ImportError:
            logger.warning(
                "SemanticChunker: sentence-transformers not installed. "
                "EMBEDDING and HYBRID backends unavailable. "
                "Install with: pip install sentence-transformers\n"
                "Falling back to MORPHOLOGICAL backend."
            )
            return False

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

        Returns
        -------
        list[Chunk]
            Boundary-refined list (always non-empty for non-empty input).
        """
        if len(chunks) <= 1:
            return chunks

        texts = [c.text for c in chunks]
        try:
            vectors = self._embed_texts(texts)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SemanticChunker: embedding failed during boundary refinement "
                "(%s). Returning unrefined morphological chunks.",
                exc,
            )
            return chunks

        merged: list[Chunk] = [chunks[0]]
        merged_vecs: list[list[float]] = [vectors[0]]

        for i in range(1, len(chunks)):
            sim = self._cosine_similarity(merged_vecs[-1], vectors[i])
            if sim >= self._cfg.similarity_threshold:
                # Merge: concatenate text, keep start of first / end of last
                prev = merged[-1]
                curr = chunks[i]
                new_text = prev.text + " " + curr.text
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

        Notes
        -----
        **Algorithm:**

        1. Input validation (NUL / surrogate guard).
        2. Multilang preprocessing trace (BOM strip → control strip → NFC).
        3. Layer 1: ScriptSegmenter → ScriptSpan list.
        4. Layer 2: WritingSystemAdapter → per-script Chunk list.
        5. Min/max token filtering and enforcement.
        6. EMBEDDING / HYBRID: boundary refinement via cosine similarity.
        7. MultilangMixin: enrich each chunk with MultilangChunkMeta.
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

        # ── Step 3+4: Layer 1 → Layer 2 ──────────────────────────────────
        layer2_chunks = self._adapter.adapt_text(text)

        # ── Step 5: Token count filtering ────────────────────────────────
        if self._cfg.min_chunk_tokens > 1 or self._cfg.max_chunk_tokens is not None:
            layer2_chunks = self._enforce_token_limits(layer2_chunks)

        # ── Step 6: Boundary refinement ───────────────────────────────────
        backend = self._cfg.backend
        if backend in (SemanticBackend.EMBEDDING, SemanticBackend.HYBRID):
            model_ok = self._load_embed_model()
            if model_ok:
                layer2_chunks = self._refine_boundaries(layer2_chunks)
            else:
                logger.warning(
                    "SemanticChunker: falling back to MORPHOLOGICAL "
                    "because sentence-transformers is unavailable."
                )

        # ── Step 7: Multilang enrichment ──────────────────────────────────
        final_chunks: list[Chunk] = []
        for idx, ch in enumerate(layer2_chunks):
            chunk_meta = dict(ch.metadata)
            chunk_meta["chunk_index"] = idx
            chunk_meta["chunking_strategy"] = f"semantic_{backend.value}"
            if doc_id is not None:
                chunk_meta["doc_id"] = doc_id

            enriched = Chunk(
                text=ch.text,
                start_char=ch.start_char,
                end_char=ch.end_char,
                metadata=chunk_meta,
            )

            if self._ml_cfg.enabled:
                chunk_raw = (
                    raw_text[ch.start_char : ch.end_char]
                    if self._ml_cfg.include_raw_text
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
                # Attach embedding if EMBEDDING / HYBRID backend was used
                if (
                    backend in (SemanticBackend.EMBEDDING, SemanticBackend.HYBRID)
                    and self._embed_model is not None
                ):
                    try:
                        vec = self._embed_texts([ch.text])[0]
                        ml_meta = ml_meta.with_embedding(
                            vec,
                            model_name=self._cfg.model_name,
                            model_version=self._cfg.model_version,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "SemanticChunker: per-chunk embedding failed (%s).", exc
                        )

                enriched = self._ml_enrich_chunk(enriched, ml_meta)

            final_chunks.append(enriched)

        result_meta: dict[str, Any] = {
            "chunker": "semantic",
            "backend": backend.value,
            "model_name": self._cfg.model_name,
            "model_version": self._cfg.model_version,
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

    def _enforce_token_limits(self, chunks: list[Chunk]) -> list[Chunk]:
        """Enforce min/max token limits on Layer 2 chunks.

        Parameters
        ----------
        chunks : list[Chunk]
            Input chunk list.

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
                sub_chunks = self._split_oversized(ch)
                for sub in sub_chunks:
                    self._maybe_flush_buf(merge_buf, result)
                    merge_buf = []
                    result.append(sub)
                continue

            # Buffer undersized chunks for merging
            if token_count < self._cfg.min_chunk_tokens:
                merge_buf.append(ch)
            else:  # noqa: PLR5501
                if merge_buf:
                    result.append(self._merge_buf(merge_buf, ch))
                    merge_buf = []
                else:
                    result.append(ch)

        # Flush remaining buffer into the last chunk
        if merge_buf:
            if result:
                result[-1] = self._merge_buf([result[-1]], *merge_buf)
            else:
                result.append(self._merge_buf(merge_buf))

        return result or chunks

    def _split_oversized(self, chunk: Chunk) -> list[Chunk]:
        """Split an oversized chunk by max_chunk_tokens."""
        max_t = self._cfg.max_chunk_tokens or 9999
        words = chunk.text.split()
        result: list[Chunk] = []
        for i in range(0, len(words), max_t):
            part = " ".join(words[i : i + max_t])
            result.append(
                Chunk(
                    text=part,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={**dict(chunk.metadata), "oversized_split": True},
                )
            )
        return result or [chunk]

    @staticmethod
    def _merge_buf(buf: list[Chunk], *extra: Chunk) -> Chunk:
        """Merge a buffer of chunks into one."""
        all_chunks = list(buf) + list(extra)
        merged_text = " ".join(c.text for c in all_chunks)
        return Chunk(
            text=merged_text,
            start_char=all_chunks[0].start_char,
            end_char=all_chunks[-1].end_char,
            metadata={**dict(all_chunks[0].metadata), "undersized_merge": True},
        )

    @staticmethod
    def _maybe_flush_buf(buf: list[Chunk], result: list[Chunk]) -> None:
        if buf:
            result.append(SemanticChunker._merge_buf(buf))

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
