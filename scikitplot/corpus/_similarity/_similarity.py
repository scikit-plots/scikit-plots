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
from typing import Any, Iterable, Mapping, Sequence

from .._diagnostics import ErrorCategory, ErrorRecord
from .._embedding_manifest import IncompatibleEmbeddingsError
from .._generation import IndexGeneration, derive_generation
from .._retrieval import (
    LegKind,
    LegOutcome,
    LegStatus,
    RetrievalResponse,
)

logger = logging.getLogger(__name__)

__all__ = [
    "RetrievalConfig",
    "RetrievalHit",
    "RetrievalIndex",
]


# =====================================================================
# Result types
# =====================================================================


@dataclass(frozen=True)
class RetrievalHit:
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
    index_generation : IndexGeneration or None
        The :class:`RetrievalIndex` build generation that produced this
        result.  Increments on every :meth:`RetrievalIndex.build`, so a caller
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
    index_generation: IndexGeneration | None = field(default=None, compare=False)
    native_score: float | None = field(default=None, compare=False)
    """The score the backend actually returned, before any fusion.

    ``score`` is the value this hit was *ranked* by, which for a fused result is
    not any backend's output.  Keeping them separate is what makes a fused score
    explainable rather than merely a number (finding F-R07-03).
    """

    native_metric: str | None = field(default=None, compare=False)
    """Scale of :attr:`native_score`, e.g. ``"cosine_similarity"``.

    §19 forbids comparing cosine, Euclidean, inner-product and backend-specific
    relevance scores as though they shared one scale.  A hit that records its own
    scale lets a consumer verify no such comparison happened.
    """

    contributions: tuple = field(default=(), compare=False)
    """Per-leg provenance for a fused hit, empty for a single-leg hit.

    Each entry records which leg found this document, at what rank, and with
    what native score and metric.  A hit that ranked #1 in both legs and one
    that ranked #1 and #40 used to be indistinguishable -- very different
    confidence signals collapsed into one fused float (finding F-R09-02).
    """

    rank: int | None = field(default=None, compare=False)
    """Zero-based position within the leg that produced this hit.

    §19 names rank as the fallback ordering when no validated normalization
    exists -- which, per R06, is the case for every non-cosine metric today.
    """


@dataclass(frozen=True)
class RetrievalConfig:
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
    backend : str or VectorIndexBackend subclass
        Dense ANN backend selector for SEMANTIC/HYBRID modes. Built-in string
        names include ``"auto"``, ``"annoy"``, ``"faiss"``, ``"voyager"``
        and ``"bruteforce"``. A custom ``VectorIndexBackend`` subclass may be
        supplied directly for an application-local backend without adding
        backend-specific fields to this dataclass.
    index_kwargs : mapping, optional
        Generic constructor keyword arguments for the selected vector-index
        backend. Prefer this for new backend-specific tuning, for example
        ``index_kwargs={"n_trees": 20, "metric": "angular"}`` with Annoy.
        Existing ``annoy_*`` fields remain supported for compatibility.
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
    backend: Any = "auto"
    index_kwargs: Mapping[str, Any] = field(default_factory=dict)
    annoy_n_trees: int = 10
    annoy_metric: str = "angular"
    annoy_search_k: int = -1
    annoy_impl: str = "auto"
    annoy_dtype: str | None = None
    annoy_index_dtype: str | None = None

    def __post_init__(self) -> None:  # ruff: ignore[too-many-branches]
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
        if isinstance(self.backend, str):
            if self.backend not in valid_backends:
                raise ValueError(
                    f"backend must be one of {valid_backends} or a "
                    f"VectorIndexBackend subclass, got {self.backend!r}"
                )
        elif isinstance(self.backend, type):
            from ._backends import VectorIndexBackend  # noqa: PLC0415

            if not issubclass(self.backend, VectorIndexBackend):
                raise TypeError(
                    "backend classes must subclass VectorIndexBackend; "
                    f"got {self.backend!r}"
                )
        else:
            raise TypeError(
                "backend must be a built-in backend name or VectorIndexBackend subclass"
            )
        if not isinstance(self.index_kwargs, Mapping):
            raise TypeError(
                f"index_kwargs must be a mapping, got {type(self.index_kwargs).__name__!r}"
            )
        if any(not isinstance(key, str) for key in self.index_kwargs):
            raise TypeError(
                "index_kwargs keys must be strings because they are constructor kwargs"
            )
        # Canonicalise keyword order so equivalent mappings produce the same
        # CorpusPlan fingerprint regardless of caller insertion order.
        object.__setattr__(
            self,
            "index_kwargs",
            dict(sorted(self.index_kwargs.items())),
        )
        if isinstance(self.backend, str) and self.backend in {"auto", "annoy"}:
            compatibility = {
                "n_trees": (self.annoy_n_trees, 10, "annoy_n_trees"),
                "metric": (self.annoy_metric, "angular", "annoy_metric"),
                "search_k": (self.annoy_search_k, -1, "annoy_search_k"),
                "impl": (self.annoy_impl, "auto", "annoy_impl"),
                "dtype": (self.annoy_dtype, None, "annoy_dtype"),
                "index_dtype": (self.annoy_index_dtype, None, "annoy_index_dtype"),
            }
            for key, (
                legacy_value,
                legacy_default,
                legacy_name,
            ) in compatibility.items():
                if (
                    key in self.index_kwargs
                    and legacy_value != legacy_default
                    and self.index_kwargs[key] != legacy_value
                ):
                    raise ValueError(
                        f"conflicting Annoy configuration: {legacy_name}={legacy_value!r} "
                        f"but index_kwargs[{key!r}]={self.index_kwargs[key]!r}"
                    )
        if self.annoy_n_trees < 1:
            raise ValueError(f"annoy_n_trees must be >= 1, got {self.annoy_n_trees}")
        valid_impls = ("auto", "highlevel", "cython")
        if self.annoy_impl not in valid_impls:
            raise ValueError(
                f"annoy_impl must be one of {valid_impls}, got {self.annoy_impl!r}"
            )

    def _legacy_annoy_selector_kwargs(self) -> dict[str, Any]:
        """Return legacy selector kwargs in one compatibility mapping."""
        return {
            "annoy_metric": self.annoy_metric,
            "annoy_n_trees": self.annoy_n_trees,
            "annoy_search_k": self.annoy_search_k,
            "annoy_impl": self.annoy_impl,
            "annoy_dtype": self.annoy_dtype,
            "annoy_index_dtype": self.annoy_index_dtype,
        }


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
# RetrievalIndex
# =====================================================================


def _skipped(kind: LegKind) -> LegOutcome:
    """Return an outcome for a leg this query did not request."""
    return LegOutcome(leg=kind, status=LegStatus.SKIPPED)


class RetrievalIndex:
    """Multi-mode similarity index over ``CorpusDocument`` collections.

    Parameters
    ----------
    config : RetrievalConfig or None, optional
        Default search configuration.  Can be overridden per query.

    Notes
    -----
    **User note:** Build the index once, query many times::

        index = RetrievalIndex()
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
    >>> index = RetrievalIndex()
    >>> # index.build(corpus_documents)
    >>> # results = index.search("quantum computing")
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._documents: list[Any] = []
        self._bm25: _BM25Index | None = None
        self._token_lists: list[list[str]] = []
        self._embeddings: Any = None  # np.ndarray or None
        self._backend: Any = None  # VectorIndexBackend or None (dense index)
        #: Content-derived identity of the built index, ``None`` before the
        #: first build.  Replaces the former per-instance counter, which
        #: carried no information across a process boundary (F-R01-06,
        #: F-R04-01).
        self._generation: IndexGeneration | None = None

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
        # F-R05-01: refuse to build one index over incompatible embedding
        # generations.  Same-dimension vectors from different models used to be
        # stacked and searched as though they shared a space, returning ranked
        # results that nothing detected as meaningless.
        self._reject_mixed_manifests()
        self._reject_ragged_dimensions(embs)

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

        self._generation = derive_generation(self._documents, backend=self.backend_name)
        logger.info(
            "RetrievalIndex: built with %d documents "
            "(dense=%s, backend=%s, sparse=True, generation=%s)",
            len(self._documents),
            self._embeddings is not None,
            self.backend_name,
            self._generation,
        )

    def _check_threshold_scale(self, cfg: RetrievalConfig) -> None:
        """Reject a threshold whose scale does not match the active metric.

        Raises
        ------
        ValueError
            If ``semantic_threshold`` cannot be valid on the scale the backend
            declares it will return.

        Notes
        -----
        **User-focused.**  ``semantic_threshold`` means different things on
        different metrics.  On a cosine backend it is a similarity in
        ``[-1, 1]``; on a non-cosine Annoy metric the backend returns
        ``1 / (1 + d)`` in ``(0, 1]``, where a *negative* threshold is
        meaningless and the default ``0.0`` excludes nothing at all.

        **Developer-focused.**  Finding F-R06-02: ``semantic_threshold`` and
        ``annoy_metric`` were independent flat fields, so a threshold tuned on
        one metric silently mis-filtered on another, and the default ``0.0``
        behaved differently on each scale.  Now the backend *declares* its score
        semantics (P-I1-15) and the threshold is validated against that
        declaration rather than against an assumption.
        """
        backend = self._backend
        if backend is None:
            return
        semantics = getattr(backend, "score_semantics", "cosine_similarity")
        threshold = cfg.semantic_threshold

        if semantics == "bounded_inverse_distance" and threshold < 0.0:
            raise ValueError(
                f"semantic_threshold={threshold} is invalid for backend "
                f"{getattr(backend, 'name', '?')!r} with metric "
                f"{getattr(backend, 'metric', '?')!r}: its scores are a bounded "
                "inverse distance in (0, 1], so a negative threshold cannot "
                "match. Thresholds are not portable across metrics."
            )
        if semantics == "cosine_similarity" and not (-1.0 <= threshold <= 1.0):
            raise ValueError(
                f"semantic_threshold={threshold} is outside the cosine "
                f"similarity range [-1, 1] returned by backend "
                f"{getattr(backend, 'name', '?')!r}."
            )

    def _reject_mixed_manifests(self) -> None:
        """Refuse a document set spanning more than one embedding generation.

        Raises
        ------
        IncompatibleEmbeddingsError
            Naming every distinct generation found.

        Notes
        -----
        **User.**  Re-embed the corpus with a single model, or build one index
        per generation.

        **Developer.**  Documents predating manifests carry
        ``embedding_manifest_id is None``.  Those are *not* treated as a
        generation of their own -- doing so would reject every existing corpus.
        But a corpus mixing tagged and untagged embeddings **is** rejected,
        because the untagged ones cannot be shown compatible with anything.
        """
        tagged, untagged = set(), 0
        for doc in self._documents:
            if getattr(doc, "embedding", None) is None:
                continue
            manifest_id = getattr(doc, "embedding_manifest_id", None)
            if manifest_id is None:
                untagged += 1
            else:
                tagged.add(manifest_id)

        if len(tagged) > 1:
            raise IncompatibleEmbeddingsError(
                f"corpus contains embeddings from {len(tagged)} different "
                f"generations ({sorted(tagged)}); vectors from different "
                "embedding generations occupy different spaces, so distances "
                "between them are meaningless. Re-embed with one model, or "
                "build one index per generation."
            )
        if tagged and untagged:
            raise IncompatibleEmbeddingsError(
                f"corpus mixes {untagged} embedding(s) with no manifest and "
                f"{len(tagged)} tagged generation ({sorted(tagged)}); untagged "
                "vectors cannot be shown compatible with tagged ones. Re-embed "
                "the untagged documents."
            )

    def _reject_ragged_dimensions(self, embs: list) -> None:
        """Refuse embeddings of differing length.

        Raises
        ------
        IncompatibleEmbeddingsError
            Naming the dimensions found.

        Notes
        -----
        **Developer.**  This used to be caught only as a ``vstack`` failure,
        which was logged and then silently disabled dense search for the
        *entire* corpus -- one mis-dimensioned document cost semantic search
        over everything, and ``build()`` still reported success (finding
        F-R05-01, case G1).
        """
        sizes = {len(e) for e in embs if e is not None}
        if len(sizes) > 1:
            raise IncompatibleEmbeddingsError(
                f"corpus contains embeddings of differing dimensions "
                f"{sorted(sizes)}; a single index requires one dimension. "
                "This previously disabled dense search for the whole corpus "
                "without reporting why."
            )

    def _build_ann_backend(self, embeddings: Any) -> None:
        """Build the dense ANN index via the centralized backend selector.

        The backend is chosen from :class:`RetrievalConfig.backend`
        (default ``"auto"`` → Annoy when available, else FAISS, Voyager, or
        exact brute-force).  Selecting an explicitly named backend that is not
        installed raises :class:`RuntimeError` here — a deliberately requested
        backend must not be silently downgraded.
        """
        from ._backends import select_backend  # noqa: PLC0415

        cfg = self.config
        # Config errors (unknown/unavailable explicit backend) propagate.
        # Backend-specific constructor tuning travels through one generic
        # ``index_kwargs`` mapping; legacy Annoy fields are folded in by the
        # selector for backward compatibility.
        backend = select_backend(
            cfg.backend,
            index_kwargs=cfg.index_kwargs,
            **cfg._legacy_annoy_selector_kwargs(),
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
            "RetrievalIndex: dense backend=%r (dim=%d)",
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
        config: RetrievalConfig | None = None,
        query_embedding: Any | None = None,
    ) -> RetrievalResponse:
        """Search the index.

        Parameters
        ----------
        query : str
            Query text.
        config : RetrievalConfig or None, optional
            Override default config for this query.
        query_embedding : array-like or None, optional
            Pre-computed query embedding.  Required for SEMANTIC
            mode if no embedding engine is attached.

        Returns
        -------
        RetrievalResponse
            Hits sorted by descending score, plus a per-leg account of how the
            search went.  The response iterates, indexes and lens like the list
            of hits it replaced, so ``for hit in response`` is unchanged; consult
            :attr:`RetrievalResponse.status` to distinguish a complete result
            from a partial one.

        Notes
        -----
        **Developer.**  Before this returned an envelope, a hybrid query without
        a query embedding silently dropped its dense leg and returned fused
        lexical-only results still labelled ``match_mode="hybrid"`` -- fewer
        hits, every score halved by the missing ``hybrid_alpha`` contribution,
        and no signal (finding F-R09-01).  That outcome is now ``DEGRADED`` with
        the dense leg marked ``FAILED``.
        """
        cfg = config or self.config

        if cfg.match_mode == "strict":
            hits = self._search_strict(query, cfg)
            legs = [
                self._leg(LegKind.LEXICAL, hits, backend=None),
                _skipped(LegKind.DENSE),
            ]
        elif cfg.match_mode == "keyword":  # noqa: RET505
            hits = self._search_keyword(query, cfg)
            legs = [
                self._leg(LegKind.LEXICAL, hits, backend=None),
                _skipped(LegKind.DENSE),
            ]
        elif cfg.match_mode == "semantic":
            hits, dense_leg = self._run_dense(
                query, query_embedding, cfg, sole_leg=True
            )
            legs = [_skipped(LegKind.LEXICAL), dense_leg]
        elif cfg.match_mode == "hybrid":
            hits, legs = self._search_hybrid_with_legs(query, query_embedding, cfg)
        else:
            raise ValueError(f"Unknown match_mode: {cfg.match_mode!r}")

        return RetrievalResponse(hits=hits, legs=legs, query=query)

    # ------------------------------------------------------------------
    # Leg construction (F-R09-01)
    # ------------------------------------------------------------------

    def _leg(
        self,
        kind: LegKind,
        hits: list[RetrievalHit],
        *,
        backend: str | None,
    ) -> LegOutcome:
        """Build a successful/empty outcome for a leg that ran."""
        return LegOutcome(
            leg=kind,
            status=LegStatus.SUCCESS if hits else LegStatus.EMPTY,
            hit_count=len(hits),
            generation=self._generation,
            backend=backend,
        )

    def _dense_unavailable_reason(
        self, query_embedding: Any | None
    ) -> ErrorRecord | None:
        """Explain why the dense leg cannot run, or ``None`` if it can.

        Notes
        -----
        **Developer.**  Each branch here used to be an unreported early
        ``return []``.  Findings F-R01-08, F-R02-04 and F-R09-01 are all
        instances of that: a caller could not distinguish "no semantic match"
        from "semantic search never ran".
        """
        # Order matters: an absent dense index is a *corpus-state* condition and
        # must degrade, whereas a missing embedding when an index DOES exist is a
        # caller error.  Checking the caller first would misreport an
        # embedding-free corpus as the caller's fault.
        if self._embeddings is None or self._backend is None:
            return ErrorRecord(
                code="NO_DENSE_INDEX",
                category=ErrorCategory.CAPABILITY,
                message=(
                    "no dense index is available; the corpus may have no "
                    "embeddings, incomplete embedding coverage, or ragged "
                    "embedding dimensions"
                ),
                stage="retrieve",
                details={"documents": len(self._documents)},
            )
        if query_embedding is None:
            return ErrorRecord(
                code="NO_QUERY_EMBEDDING",
                category=ErrorCategory.CAPABILITY,
                message=(
                    "dense retrieval requires a query embedding; none was "
                    "supplied and no embedding engine is attached"
                ),
                stage="retrieve",
            )
        return None

    def _run_dense(  # ruff: ignore[undocumented-param]
        self,
        query: str,
        query_embedding: Any | None,
        cfg: RetrievalConfig,
        *,
        sole_leg: bool = False,
    ) -> tuple[list[RetrievalHit], LegOutcome]:
        """Run the dense leg, reporting *why* it could not run when it cannot.

        Parameters
        ----------
        sole_leg : bool, optional
            ``True`` when dense is the only requested evidence path, i.e. an
            explicit ``match_mode="semantic"`` query.

        Raises
        ------
        ValueError
            If ``sole_leg`` is ``True`` and the caller supplied no query
            embedding.

        Notes
        -----
        **Developer.**  The raise/degrade split is deliberate and follows the
        distinction the review drew for backend selection (disproof D-5): an
        explicitly requested capability that the *caller* failed to supply is a
        caller error and raises, while a capability unavailable because of
        *corpus state* degrades and is reported through the envelope.

        So ``match_mode="semantic"`` with no embedding raises -- the caller asked
        for semantic search and provided nothing to search with -- whereas the
        same missing embedding inside a hybrid query degrades that one leg,
        because the lexical leg can still answer.
        """
        reason = self._dense_unavailable_reason(query_embedding)
        if sole_leg and reason is not None and reason.code == "NO_QUERY_EMBEDDING":
            raise ValueError(
                "match_mode='semantic' requires query_embedding; none supplied."
            )
        if reason is not None:
            return [], LegOutcome(
                leg=LegKind.DENSE,
                status=LegStatus.FAILED,
                generation=self._generation,
                error=reason,
            )
        self._check_threshold_scale(cfg)
        hits = self._search_semantic(query, query_embedding, cfg)
        return hits, self._leg(LegKind.DENSE, hits, backend=self.backend_name)

    # ------------------------------------------------------------------
    # STRICT search
    # ------------------------------------------------------------------

    def _search_strict(
        self,
        query: str,
        cfg: RetrievalConfig,
    ) -> list[RetrievalHit]:
        """Exact substring search."""
        results: list[RetrievalHit] = []
        use_norm = cfg.use_normalized_text
        q = query if cfg.case_sensitive else query.lower()

        for doc in self._documents:
            text = _get_text(doc, use_norm)
            t = text if cfg.case_sensitive else text.lower()
            if q in t:
                results.append(
                    RetrievalHit(
                        doc=doc,
                        score=1.0,
                        match_mode="strict",
                        backend=None,
                        index_generation=self._generation,
                        native_score=1.0,
                        native_metric="exact_match",
                        rank=len(results),
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
        cfg: RetrievalConfig,
    ) -> list[RetrievalHit]:
        """BM25-based keyword search."""
        if self._bm25 is None:
            return []

        query_tokens = _tokenize_simple(query)
        if not query_tokens:
            return []

        bm25_results = self._bm25.query(query_tokens, top_k=cfg.top_k)
        results: list[RetrievalHit] = []
        for idx, score in bm25_results:
            if score < cfg.keyword_threshold:
                continue
            results.append(
                RetrievalHit(
                    doc=self._documents[idx],
                    score=score,
                    match_mode="keyword",
                    backend=None,
                    index_generation=self._generation,
                    native_score=score,
                    native_metric="bm25",
                    rank=len(results),
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
        cfg: RetrievalConfig,
    ) -> list[RetrievalHit]:
        """Dense vector cosine similarity search via the unified backend.

        All numeric handling — normalisation, dimension and finiteness
        validation, cosine conversion, and deterministic tie ordering — is
        owned by the selected :class:`~._backends.VectorIndexBackend`, so this method
        only applies the semantic threshold and wraps hits in
        :class:`RetrievalHit`.  Scores are cosine similarity in ``[-1, 1]``.
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

        results: list[RetrievalHit] = []
        for idx, score in self._backend.query(query_embedding, cfg.top_k):
            if score < cfg.semantic_threshold:
                continue
            results.append(
                RetrievalHit(
                    doc=self._documents[idx],
                    score=float(score),
                    match_mode="semantic",
                    backend=self.backend_name,
                    index_generation=self._generation,
                    native_score=float(score),
                    native_metric=getattr(
                        self._backend, "score_semantics", "cosine_similarity"
                    ),
                    rank=len(results),
                )
            )
        return results

    # ------------------------------------------------------------------
    # HYBRID search (reciprocal rank fusion)
    # ------------------------------------------------------------------

    def _search_hybrid_with_legs(
        self,
        query: str,
        query_embedding: Any | None,
        cfg: RetrievalConfig,
    ) -> tuple[list[RetrievalHit], list[LegOutcome]]:
        """Run both legs, record each outcome, then fuse.

        Notes
        -----
        **Developer.**  This is the F-R09-01 fix.  The dense leg's outcome is
        recorded *before* fusion, so a hybrid query whose dense leg could not run
        yields ``DEGRADED`` with a naming ``ErrorRecord`` instead of silently
        fused lexical-only results still labelled ``hybrid``.
        """
        fetch_k = min(cfg.top_k * 3, max(len(self._documents), 1))

        kw_cfg = RetrievalConfig(
            top_k=fetch_k,
            match_mode="keyword",
            use_normalized_text=cfg.use_normalized_text,
        )
        keyword_results = self._search_keyword(query, kw_cfg)
        lexical_leg = self._leg(LegKind.LEXICAL, keyword_results, backend=None)

        sem_cfg = RetrievalConfig(
            top_k=fetch_k,
            match_mode="semantic",
            backend=cfg.backend,
            semantic_threshold=cfg.semantic_threshold,
        )
        semantic_results, dense_leg = self._run_dense(query, query_embedding, sem_cfg)

        hits = self._fuse_hybrid(query, keyword_results, semantic_results, cfg)
        return hits, [lexical_leg, dense_leg]

    @staticmethod
    def check_score_fusion_allowed(
        hits: Iterable[RetrievalHit],
    ) -> str | None:
        """Whether score-space fusion is defensible for these hits.

        Parameters
        ----------
        hits : iterable of RetrievalHit
            Hits from every leg that would be combined.

        Returns
        -------
        str or None
            The shared ``native_metric`` when score fusion is permissible, or
            ``None`` when it is not and rank fusion must be used.

        Notes
        -----
        **User-focused.**  Rank fusion is the default combiner.  Score-space
        fusion is available only when every leg reports the *same*
        ``native_metric``, because adding a BM25 score to a cosine similarity
        produces a number with no meaning.

        **Developer-focused.**  ADR-R07-003 inverts the dangerous default
        deliberately: the failure mode of unnecessary rank fusion is slightly
        worse ranking, while the failure mode of unjustified score fusion is
        *confidently wrong* ranking.  §19 states the rule directly -- "do not
        compare cosine, Euclidean, inner product and backend-specific relevance
        scores as if they share one scale."

        Note that a shared metric is necessary but not sufficient in general: a
        validated normalization is also required.  R06 established none exists
        for any non-cosine metric today, which is why the cosine case is the only
        one this returns for.
        """
        metrics = {hit.native_metric for hit in hits if hit.native_metric is not None}
        if len(metrics) != 1:
            return None
        metric = next(iter(metrics))
        return metric if metric == "cosine_similarity" else None

    def _fuse_hybrid(
        self,
        query: str,
        keyword_results: list[RetrievalHit],
        semantic_results: list[RetrievalHit],
        cfg: RetrievalConfig,
    ) -> list[RetrievalHit]:
        """Reciprocal rank fusion of the supplied leg results.

        Notes
        -----
        The fusion arithmetic is unchanged from the reviewed implementation --
        standard-form ``1/(k + rank + 1)`` weighted by ``hybrid_alpha``, keyed on
        ``doc_id`` rather than row offset.  Review run R09 verified it correct
        (disproof D-17); only the *leg gating* around it moved out.
        """
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
                RetrievalHit(
                    doc=doc_map[doc_id],
                    score=rrf_scores[doc_id],
                    match_mode="hybrid",
                    backend=self.backend_name,
                    index_generation=self._generation,
                    # A fused score belongs to no backend, so `native_score` is
                    # deliberately None: claiming one would assert a scale the
                    # value does not have (ADR-R07-003).
                    native_score=None,
                    native_metric="reciprocal_rank_fusion",
                    rank=len(results),
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
    def index_generation(self) -> IndexGeneration | None:
        """Content-derived identity of the built index.

        Returns
        -------
        IndexGeneration or None
            ``None`` before the first build.

        Notes
        -----
        **User-focused.**  Every :class:`RetrievalHit` carries the generation
        active at query time, so a caller can detect a result computed against a
        different index -- including one built in another process, which a
        counter could not express.

        **Developer-focused.**  Because the value is derived from content rather
        than incremented, rebuilding the same documents with the same
        configuration yields the *same* generation.  That makes ``build()``
        idempotent, removing the only ``NON_IDEMPOTENT`` operation R04 found in
        the package, and it turns rebuild-detection into the question a caller
        actually has: *does this index match this content?*
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
        stable document identities instead of :class:`RetrievalHit` objects.

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
            f"RetrievalIndex("
            f"n_docs={self.n_documents}, "
            f"dense={self.has_embeddings}, "
            f"backend={self.backend_name!r}, "
            f"mode={self.config.match_mode!r})"
        )
