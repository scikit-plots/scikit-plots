# scikitplot/corpus/_similarity/_similarity.py
#
# Flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Multi-mode similarity search over ``CorpusDocument`` collections.

Supports four match modes defined in
:class:`~scikitplot.corpus._schema.MatchMode`:

- **STRICT** — exact substring / n-gram matching in ``text``
- **KEYWORD** — stemmed/lemmatised keyword overlap (Jaccard or BM25)
- **SEMANTIC** — dense vector cosine similarity via a pluggable ANN backend
  (Annoy by default; FAISS or Voyager when installed; exact brute-force floor)
- **HYBRID** — reciprocal rank fusion of BM25 sparse + dense vector

.. admonition:: Backend requirements

   - ``STRICT``/``KEYWORD`` — zero external deps (pure Python)
   - ``SEMANTIC`` — requires ``numpy``; uses ``scikitplot.annoy`` by default
     and optionally ``faiss-cpu`` or ``voyager`` for ANN indexing. Selection is
     centralised in :mod:`scikitplot.corpus._similarity._backends`; the exact
     pure-``numpy`` brute-force backend is the always-available floor.
   - ``HYBRID`` — requires both ``numpy`` and a keyword index

   All backends return scores on one scale: cosine similarity in ``[-1, 1]``.

Supports Python 3.8 through 3.15.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter  # noqa: F401
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "SearchConfig",
    "SearchResult",
    "SimilarityIndex",
]


# =====================================================================
# Result types
# =====================================================================


@dataclass(frozen=True)
class SearchResult:
    """A single search result.

    Parameters
    ----------
    doc : CorpusDocument
        The matched document.
    score : float
        Relevance score (higher is better).  Scale depends on
        match mode:

        - STRICT: 1.0 if match, 0.0 otherwise
        - KEYWORD: Jaccard similarity [0, 1] or BM25 score
        - SEMANTIC: cosine similarity [-1, 1]
        - HYBRID: reciprocal rank fusion score
    match_mode : str
        The mode that produced this result.
    backend : str or None
        Name of the dense ANN backend that produced this result
        (SEMANTIC/HYBRID), or ``None`` for STRICT/KEYWORD.  Provenance only:
        excluded from equality and hashing.
    index_generation : int or None
        The :class:`SimilarityIndex` build generation that produced this
        result.  Increments on every :meth:`SimilarityIndex.build`, so a caller
        can detect results computed against a since-rebuilt index.  Provenance
        only: excluded from equality and hashing.

    Notes
    -----
    **Developer note:** ``backend`` and ``index_generation`` describe *how* the
    result was produced, not *what* it is, so they use ``compare=False`` — two
    results for the same document/score/mode remain equal regardless of
    provenance.  Embedding-model identity is out of scope here (it travels with
    the document embeddings; see the embedding-cache identity contract).
    """

    doc: Any
    score: float
    match_mode: str
    backend: str | None = field(default=None, compare=False)
    index_generation: int | None = field(default=None, compare=False)


@dataclass(frozen=True)
class SearchConfig:
    """Configuration for similarity search.

    Parameters
    ----------
    top_k : int
        Maximum results to return.
    match_mode : str
        One of ``"strict"``, ``"keyword"``, ``"semantic"``,
        ``"hybrid"``.
    semantic_threshold : float
        Minimum cosine similarity for SEMANTIC results.
    keyword_threshold : float
        Minimum keyword overlap for KEYWORD results.
    hybrid_alpha : float
        Weight for semantic scores in HYBRID mode (0 = pure keyword,
        1 = pure semantic).  Default 0.5 (equal weight).
    rrf_k : int
        Reciprocal rank fusion constant.  Default 60 (standard).
    use_normalized_text : bool
        Use ``normalized_text`` for matching when available.
    case_sensitive : bool
        Case-sensitive matching in STRICT mode.
    backend : str
        Dense ANN backend selector for SEMANTIC/HYBRID modes. One of
        ``"auto"`` (default; resolves to Annoy when available, else FAISS,
        Voyager, or exact brute-force), ``"annoy"``, ``"faiss"``,
        ``"voyager"``, ``"bruteforce"``. An explicitly named backend that is
        not installed raises at build time rather than silently degrading.
    annoy_n_trees : int
        Annoy tree count (accuracy/size trade-off) when the Annoy backend is
        used.  Higher is more accurate and larger.  Default 10.
    annoy_metric : str
        Annoy distance metric.  Default ``"angular"`` (cosine-like); scores are
        always reported as cosine similarity regardless of metric.
    annoy_search_k : int
        Annoy query-time node budget.  ``-1`` (default) lets Annoy choose.
    annoy_impl : str
        Which Annoy index class to use: ``"auto"`` (default; high-level
        ``scikitplot.annoy.Index`` first, else the Cython
        ``scikitplot.annoy._annoy.Index``), ``"highlevel"``, or ``"cython"``.
    annoy_dtype : str or None
        Embedding precision for the Cython Annoy class (e.g. ``"float32"``,
        ``"float64"``).  Ignored by the high-level class.  Default ``None``.
    annoy_index_dtype : str or None
        Item-id integer width for the Cython Annoy class (e.g. ``"int32"``,
        ``"uint64"``) for very large corpora.  Ignored otherwise.  Default
        ``None``.

    Notes
    -----
    **User note:** For RAG pipelines, ``match_mode="hybrid"`` with
    default settings provides a good balance.  For exact citation
    matching, use ``match_mode="strict"``.  To force a specific ANN
    library, set e.g. ``backend="annoy"`` and tune ``annoy_n_trees``.
    """

    top_k: int = 10
    match_mode: str = "semantic"
    semantic_threshold: float = 0.0
    keyword_threshold: float = 0.0
    hybrid_alpha: float = 0.5
    rrf_k: int = 60
    use_normalized_text: bool = True
    case_sensitive: bool = False
    backend: str = "auto"
    annoy_n_trees: int = 10
    annoy_metric: str = "angular"
    annoy_search_k: int = -1
    annoy_impl: str = "auto"
    annoy_dtype: str | None = None
    annoy_index_dtype: str | None = None

    def __post_init__(self) -> None:
        valid = ("strict", "keyword", "semantic", "hybrid")
        if self.match_mode not in valid:
            raise ValueError(
                f"match_mode must be one of {valid}, got {self.match_mode!r}"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError(f"hybrid_alpha must be in [0, 1], got {self.hybrid_alpha}")
        valid_backends = ("auto", "annoy", "faiss", "voyager", "bruteforce", "brute")
        if self.backend not in valid_backends:
            raise ValueError(
                f"backend must be one of {valid_backends}, got {self.backend!r}"
            )
        if self.annoy_n_trees < 1:
            raise ValueError(f"annoy_n_trees must be >= 1, got {self.annoy_n_trees}")
        valid_impls = ("auto", "highlevel", "cython")
        if self.annoy_impl not in valid_impls:
            raise ValueError(
                f"annoy_impl must be one of {valid_impls}, got {self.annoy_impl!r}"
            )


# =====================================================================
# Tokenisation helpers (no external deps)
# =====================================================================

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize_simple(text: str) -> list[str]:
    """Simpler regex tokeniser for keyword matching."""
    return [w.lower() for w in _WORD_RE.findall(text)]


def _get_text(doc: Any, use_normalized: bool) -> str:
    """Extract text from a document, preferring normalized_text."""
    if use_normalized:
        nt = getattr(doc, "normalized_text", None)
        if nt:
            return nt
    return getattr(doc, "text", "")


# =====================================================================
# BM25 sparse index (pure Python, no deps)
# =====================================================================


class _BM25Index:
    """Okapi BM25 index for keyword search.

    Parameters
    ----------
    k1 : float
        Term frequency saturation.
    b : float
        Length normalisation factor.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_freqs: dict[str, int] = {}
        self._doc_lens: list[int] = []
        self._doc_tfs: list[dict[str, int]] = []
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def build(self, token_lists: Sequence[list[str]]) -> None:
        """Build index from pre-tokenised document lists."""
        self._n_docs = len(token_lists)
        self._doc_freqs = {}
        self._doc_lens = []
        self._doc_tfs = []

        for tokens in token_lists:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._doc_tfs.append(tf)
            self._doc_lens.append(len(tokens))
            for term in set(tokens):
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        total = sum(self._doc_lens)
        self._avgdl = total / self._n_docs if self._n_docs else 1.0

    def query(
        self,
        query_tokens: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return ``(doc_index, bm25_score)`` pairs, sorted desc."""
        scores: list[float] = [0.0] * self._n_docs
        n = self._n_docs

        for term in query_tokens:
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
            for i in range(n):
                tf = self._doc_tfs[i].get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * num / den

        # Sort by score descending, take top_k
        indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]


# =====================================================================
# SimilarityIndex
# =====================================================================


class SimilarityIndex:
    """Multi-mode similarity index over ``CorpusDocument`` collections.

    Parameters
    ----------
    config : SearchConfig or None, optional
        Default search configuration.  Can be overridden per query.

    Notes
    -----
    **User note:** Build the index once, query many times::

        index = SimilarityIndex()
        index.build(documents)
        results = index.search("What did Hamlet say about death?")

    **Developer note:** The index stores references to the original
    documents.  If documents are mutated after building, results
    are undefined.

    See Also
    --------
    scikitplot.corpus._schema.MatchMode : Enum of match modes.
    scikitplot.corpus._adapters : Convert results to LangChain /
        MCP format.

    Examples
    --------
    >>> index = SimilarityIndex()
    >>> # index.build(corpus_documents)
    >>> # results = index.search("quantum computing")
    """

    def __init__(
        self,
        config: SearchConfig | None = None,
    ) -> None:
        self.config = config or SearchConfig()
        self._documents: list[Any] = []
        self._bm25: _BM25Index | None = None
        self._token_lists: list[list[str]] = []
        self._embeddings: Any = None  # np.ndarray or None
        self._backend: Any = None  # ANNBackend or None (dense index)
        self._generation: int = 0  # bumped on every successful build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, documents: Sequence[Any]) -> None:
        """Build the index from ``CorpusDocument`` instances.

        Parameters
        ----------
        documents : Sequence[CorpusDocument]
            Documents to index.  Must have ``text`` (and optionally
            ``embedding``, ``tokens``, ``normalized_text``).

        Raises
        ------
        ValueError
            If *documents* is empty.
        """
        if not documents:
            raise ValueError("Cannot build index from empty documents.")

        self._documents = list(documents)
        use_norm = self.config.use_normalized_text

        # Build keyword index (always — needed for KEYWORD and HYBRID)
        self._token_lists = []
        for doc in self._documents:
            # Prefer pre-computed tokens
            tokens = getattr(doc, "tokens", None)
            if tokens is None:
                text = _get_text(doc, use_norm)
                tokens = _tokenize_simple(text)
            self._token_lists.append(tokens)

        self._bm25 = _BM25Index()
        self._bm25.build(self._token_lists)

        # Build dense index if embeddings are available
        self._embeddings = None
        self._backend = None

        embs = []
        for doc in self._documents:
            e = getattr(doc, "embedding", None)
            if e is not None:
                embs.append(e)

        # ``embs`` is a Python list here (not an ndarray); ``if embs`` is a
        # correct emptiness check.
        if embs and len(embs) == len(self._documents):
            try:
                import numpy as np  # noqa: PLC0415
            except ImportError:
                logger.warning("NumPy not available; SEMANTIC mode disabled.")
            else:
                try:
                    stacked = np.vstack(embs).astype(np.float32)
                except Exception as exc:  # noqa: BLE001 - tolerate malformed data
                    logger.warning(
                        "Failed to stack embeddings; SEMANTIC disabled: %s", exc
                    )
                else:
                    self._embeddings = stacked
                    # Config errors (unknown/unavailable *explicit* backend)
                    # propagate here by design — fail fast, do not silently
                    # degrade a deliberately requested backend.
                    self._build_ann_backend(self._embeddings)

        self._generation += 1
        logger.info(
            "SimilarityIndex: built with %d documents "
            "(dense=%s, backend=%s, sparse=True, generation=%d)",
            len(self._documents),
            self._embeddings is not None,
            self.backend_name,
            self._generation,
        )

    def _build_ann_backend(self, embeddings: Any) -> None:
        """Build the dense ANN index via the centralized backend selector.

        The backend is chosen from :class:`SearchConfig.backend`
        (default ``"auto"`` → Annoy when available, else FAISS, Voyager, or
        exact brute-force).  Selecting an explicitly named backend that is not
        installed raises :class:`RuntimeError` here — a deliberately requested
        backend must not be silently downgraded.
        """
        from ._backends import select_backend  # noqa: PLC0415

        cfg = self.config
        # Config errors (unknown/unavailable explicit backend) propagate.
        backend = select_backend(
            cfg.backend,
            annoy_metric=cfg.annoy_metric,
            annoy_n_trees=cfg.annoy_n_trees,
            annoy_search_k=cfg.annoy_search_k,
            annoy_impl=cfg.annoy_impl,
            annoy_dtype=cfg.annoy_dtype,
            annoy_index_dtype=cfg.annoy_index_dtype,
        )
        try:
            backend.build(embeddings)
        except ValueError as exc:
            # Data-level problem (e.g. non-finite embeddings): disable the dense
            # index and degrade to sparse rather than failing the whole build.
            logger.warning("Dense index disabled (invalid embeddings): %s", exc)
            self._backend = None
            self._embeddings = None
            return
        self._backend = backend
        logger.debug(
            "SimilarityIndex: dense backend=%r (dim=%d)",
            backend.name,
            int(embeddings.shape[1]),
        )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        config: SearchConfig | None = None,
        query_embedding: Any | None = None,
    ) -> list[SearchResult]:
        """Search the index.

        Parameters
        ----------
        query : str
            Query text.
        config : SearchConfig or None, optional
            Override default config for this query.
        query_embedding : array-like or None, optional
            Pre-computed query embedding.  Required for SEMANTIC
            mode if no embedding engine is attached.

        Returns
        -------
        list[SearchResult]
            Results sorted by descending score.
        """
        cfg = config or self.config

        if cfg.match_mode == "strict":
            return self._search_strict(query, cfg)
        elif cfg.match_mode == "keyword":  # noqa: RET505
            return self._search_keyword(query, cfg)
        elif cfg.match_mode == "semantic":
            return self._search_semantic(query, query_embedding, cfg)
        elif cfg.match_mode == "hybrid":
            return self._search_hybrid(query, query_embedding, cfg)
        else:
            raise ValueError(f"Unknown match_mode: {cfg.match_mode!r}")

    # ------------------------------------------------------------------
    # STRICT search
    # ------------------------------------------------------------------

    def _search_strict(
        self,
        query: str,
        cfg: SearchConfig,
    ) -> list[SearchResult]:
        """Exact substring search."""
        results: list[SearchResult] = []
        use_norm = cfg.use_normalized_text
        q = query if cfg.case_sensitive else query.lower()

        for doc in self._documents:
            text = _get_text(doc, use_norm)
            t = text if cfg.case_sensitive else text.lower()
            if q in t:
                results.append(
                    SearchResult(
                        doc=doc,
                        score=1.0,
                        match_mode="strict",
                        backend=None,
                        index_generation=self._generation,
                    )
                )
                if len(results) >= cfg.top_k:
                    break

        return results

    # ------------------------------------------------------------------
    # KEYWORD search (BM25)
    # ------------------------------------------------------------------

    def _search_keyword(
        self,
        query: str,
        cfg: SearchConfig,
    ) -> list[SearchResult]:
        """BM25-based keyword search."""
        if self._bm25 is None:
            return []

        query_tokens = _tokenize_simple(query)
        if not query_tokens:
            return []

        bm25_results = self._bm25.query(query_tokens, top_k=cfg.top_k)
        results: list[SearchResult] = []
        for idx, score in bm25_results:
            if score < cfg.keyword_threshold:
                continue
            results.append(
                SearchResult(
                    doc=self._documents[idx],
                    score=score,
                    match_mode="keyword",
                    backend=None,
                    index_generation=self._generation,
                )
            )

        return results

    # ------------------------------------------------------------------
    # SEMANTIC search
    # ------------------------------------------------------------------

    def _search_semantic(
        self,
        query: str,
        query_embedding: Any | None,
        cfg: SearchConfig,
    ) -> list[SearchResult]:
        """Dense vector cosine similarity search via the unified backend.

        All numeric handling — normalisation, dimension and finiteness
        validation, cosine conversion, and deterministic tie ordering — is
        owned by the selected :class:`~._backends.ANNBackend`, so this method
        only applies the semantic threshold and wraps hits in
        :class:`SearchResult`.  Scores are cosine similarity in ``[-1, 1]``.
        """
        if self._embeddings is None or self._backend is None:
            logger.warning(
                "No dense index available for SEMANTIC search. "
                "Build the index with embedded documents."
            )
            return []

        if query_embedding is None:
            raise ValueError(
                "query_embedding is required for SEMANTIC mode. "
                "Pass it directly or use CorpusBuilder.search() "
                "which auto-embeds the query."
            )

        results: list[SearchResult] = []
        for idx, score in self._backend.query(query_embedding, cfg.top_k):
            if score < cfg.semantic_threshold:
                continue
            results.append(
                SearchResult(
                    doc=self._documents[idx],
                    score=float(score),
                    match_mode="semantic",
                    backend=self.backend_name,
                    index_generation=self._generation,
                )
            )
        return results

    # ------------------------------------------------------------------
    # HYBRID search (reciprocal rank fusion)
    # ------------------------------------------------------------------

    def _search_hybrid(
        self,
        query: str,
        query_embedding: Any | None,
        cfg: SearchConfig,
    ) -> list[SearchResult]:
        """Reciprocal rank fusion of BM25 + dense vector."""
        # Fetch more candidates for fusion
        fetch_k = min(cfg.top_k * 3, len(self._documents))

        kw_cfg = SearchConfig(
            top_k=fetch_k,
            match_mode="keyword",
            use_normalized_text=cfg.use_normalized_text,
        )
        keyword_results = self._search_keyword(query, kw_cfg)

        semantic_results: list[SearchResult] = []
        if query_embedding is not None and self._embeddings is not None:
            sem_cfg = SearchConfig(
                top_k=fetch_k,
                match_mode="semantic",
                semantic_threshold=0.0,
                use_normalized_text=cfg.use_normalized_text,
            )
            semantic_results = self._search_semantic(query, query_embedding, sem_cfg)

        # Reciprocal rank fusion
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, Any] = {}
        k = cfg.rrf_k

        for rank, res in enumerate(keyword_results):
            doc_id = getattr(res.doc, "doc_id", id(res.doc))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                (1.0 - cfg.hybrid_alpha) / (k + rank + 1)
            )
            doc_map[doc_id] = res.doc

        for rank, res in enumerate(semantic_results):
            doc_id = getattr(res.doc, "doc_id", id(res.doc))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (
                cfg.hybrid_alpha / (k + rank + 1)
            )
            doc_map[doc_id] = res.doc

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        results = []
        for doc_id in sorted_ids[: cfg.top_k]:
            results.append(
                SearchResult(
                    doc=doc_map[doc_id],
                    score=rrf_scores[doc_id],
                    match_mode="hybrid",
                    backend=self.backend_name,
                    index_generation=self._generation,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def n_documents(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    @property
    def has_embeddings(self) -> bool:
        """Whether dense embeddings are indexed."""
        return self._embeddings is not None

    @property
    def backend_name(self) -> str | None:
        """Name of the active dense ANN backend, or ``None`` if unbuilt."""
        return self._backend.name if self._backend is not None else None

    @property
    def index_generation(self) -> int:
        """Build generation, incremented on every :meth:`build`.

        Zero before the first build.  Every :class:`SearchResult` produced by
        :meth:`search` carries the generation active at query time, so a caller
        can detect results computed against a since-rebuilt index.
        """
        return self._generation

    def query(
        self,
        vector: Any,
        k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Vector-level ANN query returning ``(doc_id, score)`` pairs.

        This is the vector-index seam consumed by
        :mod:`scikitplot.mcp` (the ``VectorIndex`` protocol): it takes a query
        **vector** (already embedded) rather than a query string, and returns
        stable document identities instead of :class:`SearchResult` objects.

        Parameters
        ----------
        vector : array-like
            Query embedding of the same dimension as the indexed vectors.
        k : int or None, optional
            Number of neighbours to return.  Defaults to ``config.top_k``.

        Returns
        -------
        list of (str, float)
            ``(doc_id, cosine_score)`` pairs, best first.  ``doc_id`` is the
            document's ``doc_id`` attribute when present, else its stringified
            index.  Empty if no dense index was built or the query is zero-norm.

        Raises
        ------
        ValueError
            If *vector* dimension mismatches the index or is non-finite.
        """
        if self._backend is None or self._embeddings is None:
            return []
        top = int(k) if k is not None else self.config.top_k
        out: list[tuple[str, float]] = []
        for idx, score in self._backend.query(vector, top):
            doc = self._documents[idx]
            doc_id = getattr(doc, "doc_id", None)
            out.append((doc_id if doc_id is not None else str(idx), float(score)))
        return out

    def __repr__(self) -> str:
        return (
            f"SimilarityIndex("
            f"n_docs={self.n_documents}, "
            f"dense={self.has_embeddings}, "
            f"backend={self.backend_name!r}, "
            f"mode={self.config.match_mode!r})"
        )
