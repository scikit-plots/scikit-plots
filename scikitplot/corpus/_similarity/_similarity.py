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
- **SEMANTIC** — dense vector cosine similarity via FAISS or Voyager
- **HYBRID** — reciprocal rank fusion of BM25 sparse + dense vector

.. admonition:: Backend requirements

   - ``STRICT``/``KEYWORD`` — zero external deps (pure Python)
   - ``SEMANTIC`` — requires ``numpy``; optionally ``faiss-cpu`` or
     ``voyager`` for ANN indexing (falls back to brute-force cosine)
   - ``HYBRID`` — requires both ``numpy`` and a keyword index

Supports Python 3.8 through 3.15.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter  # noqa: F401
from dataclasses import dataclass, field  # noqa: F401
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
    """

    doc: Any
    score: float
    match_mode: str


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

    Notes
    -----
    **User note:** For RAG pipelines, ``match_mode="hybrid"`` with
    default settings provides a good balance.  For exact citation
    matching, use ``match_mode="strict"``.
    """

    top_k: int = 10
    match_mode: str = "semantic"
    semantic_threshold: float = 0.0
    keyword_threshold: float = 0.0
    hybrid_alpha: float = 0.5
    rrf_k: int = 60
    use_normalized_text: bool = True
    case_sensitive: bool = False

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
        self._faiss_index: Any = None

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
        self._faiss_index = None

        embs = []
        for doc in self._documents:
            e = getattr(doc, "embedding", None)
            if e is not None:
                embs.append(e)

        if embs and len(embs) == len(self._documents):
            try:
                import numpy as np  # noqa: PLC0415

                self._embeddings = np.vstack(embs).astype(np.float32)
                self._build_faiss_index(self._embeddings)
            except ImportError:
                logger.warning("NumPy not available; SEMANTIC mode disabled.")
            except Exception as exc:
                logger.warning("Failed to build dense index: %s", exc)

        logger.info(
            "SimilarityIndex: built with %d documents (dense=%s, sparse=True)",
            len(self._documents),
            self._embeddings is not None,
        )

    def _build_faiss_index(self, embeddings: Any) -> None:
        """Build a FAISS index, falling back to brute-force."""
        dim = embeddings.shape[1]
        try:
            import faiss  # type: ignore[import]  # noqa: PLC0415

            self._faiss_index = faiss.IndexFlatIP(dim)
            # Normalise for cosine similarity via inner product
            import numpy as np  # noqa: PLC0415

            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = embeddings / norms
            self._faiss_index.add(normed)
            logger.debug("Built FAISS IndexFlatIP (dim=%d)", dim)
        except ImportError:
            try:
                import voyager  # type: ignore[import]  # noqa: PLC0415

                self._faiss_index = voyager.Index(
                    voyager.Space.Cosine,
                    num_dimensions=dim,
                )
                self._faiss_index.add_items(embeddings)
                logger.debug("Built Voyager index (dim=%d)", dim)
            except ImportError:
                # Brute-force fallback — no external deps needed
                self._faiss_index = None
                logger.debug("No ANN library; using brute-force cosine.")

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
                )
            )

        return results

    # ------------------------------------------------------------------
    # SEMANTIC search
    # ------------------------------------------------------------------

    def _search_semantic(  # noqa: PLR0912
        self,
        query: str,
        query_embedding: Any | None,
        cfg: SearchConfig,
    ) -> list[SearchResult]:
        """Dense vector cosine similarity search."""
        if self._embeddings is None:
            logger.warning(
                "No embeddings available for SEMANTIC search. "
                "Build index with embedded documents."
            )
            return []

        if query_embedding is None:
            raise ValueError(
                "query_embedding is required for SEMANTIC mode. "
                "Pass it directly or use CorpusBuilder.search() "
                "which auto-embeds the query."
            )

        import numpy as np  # noqa: PLC0415

        qe = np.asarray(query_embedding, dtype=np.float32).flatten()

        if self._faiss_index is not None:
            # FAISS or Voyager
            try:
                import faiss  # type: ignore[import]  # noqa: F401, PLC0415

                qe_norm = qe / (np.linalg.norm(qe) or 1.0)
                D, I = self._faiss_index.search(  # noqa: N806
                    qe_norm.reshape(1, -1), cfg.top_k
                )
                results = []
                for score, idx in zip(D[0], I[0]):
                    if idx == -1:
                        continue
                    if score < cfg.semantic_threshold:
                        continue
                    results.append(
                        SearchResult(
                            doc=self._documents[idx],
                            score=float(score),
                            match_mode="semantic",
                        )
                    )
                return results
            except (ImportError, AttributeError):
                pass

            # Try Voyager
            try:
                ids, distances = self._faiss_index.query(qe, k=cfg.top_k)
                results = []
                for idx, dist in zip(ids, distances):
                    score = 1.0 - dist  # Voyager cosine = distance
                    if score < cfg.semantic_threshold:
                        continue
                    results.append(
                        SearchResult(
                            doc=self._documents[idx],
                            score=score,
                            match_mode="semantic",
                        )
                    )
                return results
            except Exception:
                pass

        # Brute-force cosine
        embs = self._embeddings
        norms_e = np.linalg.norm(embs, axis=1)
        norm_q = np.linalg.norm(qe)
        if norm_q == 0:
            return []

        scores = embs @ qe / (norms_e * norm_q + 1e-10)
        top_indices = np.argsort(scores)[::-1][: cfg.top_k]

        results = []
        for idx in top_indices:
            s = float(scores[idx])
            if s < cfg.semantic_threshold:
                break
            results.append(
                SearchResult(
                    doc=self._documents[idx],
                    score=s,
                    match_mode="semantic",
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

    def __repr__(self) -> str:
        return (
            f"SimilarityIndex("
            f"n_docs={self.n_documents}, "
            f"dense={self.has_embeddings}, "
            f"mode={self.config.match_mode!r})"
        )
