# scikitplot/corpus/_similarity/_backends.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Pluggable approximate-nearest-neighbour (ANN) backends for dense semantic search.

This module centralises every vector-index implementation behind a single,
uniform contract so that :class:`~scikitplot.corpus._similarity.SimilarityIndex`
(and any consumer such as :mod:`scikitplot.mcp`) never has to branch on the
concrete backend, and so that *scores mean the same thing regardless of backend*.

Unified score contract
-----------------------
Every backend's :meth:`ANNBackend.query` returns ``(row_index, score)`` pairs
where ``score`` is **cosine similarity in the closed interval ``[-1.0, 1.0]``**
(higher is better), sorted in descending score order with deterministic,
index-ascending tie breaking. This makes ``semantic_threshold`` comparisons and
hybrid fusion identical across backends.

Backend selection order
------------------------
``select_backend("auto")`` resolves the first *available* backend in
:data:`DEFAULT_BACKEND_ORDER`, which is ``annoy`` first (it is an internal
dependency of scikit-plots), then ``faiss``, then ``voyager``, then the always
available pure-``numpy`` ``bruteforce`` floor. An explicitly named backend that
is unavailable raises :class:`RuntimeError` with an actionable message rather
than silently degrading.

Notes
-----
**Developer note:** All semantic backends require ``numpy``. Native ANN
libraries (Annoy, FAISS, Voyager) are optional; their absence never removes
semantic capability because ``bruteforce`` is the guaranteed floor.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BACKEND_ORDER",
    "ANNBackend",
    "AnnoyBackend",
    "BruteForceBackend",
    "FaissBackend",
    "VoyagerBackend",
    "list_available_backends",
    "select_backend",
]

# Backends tried, in order, when ``backend="auto"``. Annoy first: it is an
# internal scikit-plots dependency and provides a persistent, memory-mapped
# vector index; ``bruteforce`` is the always-available numpy floor.
DEFAULT_BACKEND_ORDER = ("annoy", "faiss", "voyager", "bruteforce")


# =====================================================================
# Shared numeric helpers
# =====================================================================


def _require_numpy() -> Any:
    """Import numpy or raise an actionable ImportError."""
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Semantic search requires numpy. Install it with "
            "`pip install numpy` or `pip install scikit-plots[corpus]`."
        ) from exc
    return np


def _validate_query_vector(np: Any, vector: Any, dim: int) -> Any:
    """Coerce, dimension-check, and finiteness-check a query vector.

    Returns a contiguous 1-D ``float32`` array of length ``dim``.

    Raises
    ------
    ValueError
        If the flattened vector length does not match the index dimension, or
        if the vector contains non-finite (NaN/Inf) values.
    """
    qe = np.asarray(vector, dtype=np.float32).ravel()
    if qe.shape[0] != dim:
        raise ValueError(
            f"query vector has dimension {qe.shape[0]}, "
            f"but the index was built with dimension {dim}"
        )
    if not np.all(np.isfinite(qe)):
        raise ValueError("query vector contains NaN or infinite values")
    return qe


def _validate_embeddings(np: Any, embeddings: Any) -> Any:
    """Coerce to a finite 2-D ``float32`` matrix or raise.

    Centralised so every backend enforces the same build-time contract:
    a non-empty ``(n_docs, dim)`` matrix with no NaN/Inf values.

    Raises
    ------
    ValueError
        If *embeddings* is not 2-D, is empty, or contains non-finite values.
    """
    embs = np.ascontiguousarray(np.asarray(embeddings, dtype=np.float32))
    if embs.ndim != 2:  # ruff: ignore[magic-value-comparison]
        raise ValueError(
            f"embeddings must be 2-D (n_docs, dim), got shape {embs.shape}"
        )
    if embs.shape[0] == 0:
        raise ValueError("cannot build an index from zero embeddings")
    if not np.all(np.isfinite(embs)):
        raise ValueError("embeddings contain NaN or infinite values")
    return embs


def _resolve_annoy_index_cls(impl: str) -> tuple[Any, str]:
    """Resolve the requested Annoy ``Index`` class.

    Both implementations share the same ``add_item`` / ``build`` /
    ``get_nns_by_vector`` contract; the Cython class additionally accepts
    ``dtype`` / ``index_dtype``.

    Parameters
    ----------
    impl : str
        ``"auto"`` (high-level first, else Cython), ``"highlevel"``
        (``scikitplot.annoy.Index``), or ``"cython"``
        (``scikitplot.annoy._annoy.Index``).

    Returns
    -------
    (type, str)
        The resolved class and the concrete implementation name.
    """
    impl = (impl or "auto").strip().lower()
    if impl == "cython":
        from scikitplot.annoy._annoy import Index  # noqa: PLC0415

        return Index, "cython"
    if impl == "highlevel":
        from scikitplot.annoy import Index  # noqa: PLC0415

        return Index, "highlevel"
    if impl != "auto":
        raise ValueError(
            f"annoy impl must be 'auto', 'highlevel', or 'cython', got {impl!r}"
        )
    try:
        from scikitplot.annoy import Index  # noqa: PLC0415

        return Index, "highlevel"
    except Exception:  # noqa: BLE001 - fall back to the Cython class
        from scikitplot.annoy._annoy import Index  # noqa: PLC0415

        return Index, "cython"


# =====================================================================
# Backend contract
# =====================================================================


class ANNBackend:
    """Uniform contract for a dense vector index.

    Subclasses build an index from a 2-D embedding matrix and answer
    top-``k`` cosine-similarity queries. Every subclass guarantees the
    *unified score contract* described in the module docstring.

    Attributes
    ----------
    name : str
        Stable backend identifier (e.g. ``"annoy"``).
    """

    name: str = "base"

    @classmethod
    def is_available(cls) -> bool:
        """Whether this backend's runtime dependencies are importable."""
        raise NotImplementedError

    def build(self, embeddings: Any) -> None:
        """Build the index from an ``(n_docs, dim)`` embedding matrix.

        Parameters
        ----------
        embeddings : numpy.ndarray
            Row-major ``float32`` matrix. Rows are *not* required to be
            unit-normalised; each backend normalises internally as needed
            for cosine similarity.
        """
        raise NotImplementedError

    def query(self, vector: Any, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` ``(row_index, cosine_score)`` pairs.

        Results are sorted by descending ``cosine_score`` in ``[-1, 1]`` with
        deterministic, index-ascending tie breaking. A zero-norm query returns
        an empty list (cosine is undefined). Non-finite queries raise.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r})"


# =====================================================================
# Brute-force (always-available numpy floor)
# =====================================================================


class BruteForceBackend(ANNBackend):
    """Exact cosine similarity via a normalised dot product.

    Deterministic and dependency-free beyond ``numpy``. Suited to small and
    medium corpora and used as the guaranteed fallback for ``backend="auto"``.
    """

    name = "bruteforce"

    def __init__(self) -> None:
        self._normed: Any = None
        self._dim: int = 0

    @classmethod
    def is_available(cls) -> bool:
        try:
            import numpy  # noqa: F401, PLC0415  # ruff: ignore[unconventional-import-alias]

            return True
        except ImportError:  # pragma: no cover - environment dependent
            return False

    def build(self, embeddings: Any) -> None:
        np = _require_numpy()
        embs = _validate_embeddings(np, embeddings)
        self._dim = int(embs.shape[1])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        # Preserve zero rows as zero (they can never be a cosine match).
        norms = np.where(norms == 0.0, 1.0, norms)
        self._normed = embs / norms

    def query(self, vector: Any, k: int) -> list[tuple[int, float]]:
        np = _require_numpy()
        qe = _validate_query_vector(np, vector, self._dim)
        norm_q = float(np.linalg.norm(qe))
        if norm_q == 0.0:
            return []
        sims = self._normed @ (qe / norm_q)
        sims = np.clip(sims, -1.0, 1.0)
        k = max(0, min(int(k), sims.shape[0]))
        if k == 0:
            return []
        # Descending score, index-ascending ties: stable argsort on -sims.
        order = np.argsort(-sims, kind="stable")[:k]
        return [(int(i), float(sims[i])) for i in order]


# =====================================================================
# Annoy (default when available)
# =====================================================================


class AnnoyBackend(ANNBackend):
    """ANN backend over a scikit-plots Annoy ``Index`` (default backend).

    Works with **both** shipped Annoy index classes, which expose the same
    ``add_item`` / ``build`` / ``get_nns_by_vector`` contract:

    * ``scikitplot.annoy.Index`` — the high-level, mixin-composed class
      (adds a validated bulk ``add_items`` path).
    * ``scikitplot.annoy._annoy.Index`` — the Cython class, which additionally
      accepts ``dtype`` (embedding precision) and ``index_dtype`` (id width).

    Parameters
    ----------
    metric : str, optional
        Annoy distance metric. ``"angular"`` (default) yields cosine-like
        ranking; the angular distance is converted back to exact cosine
        similarity for the unified score contract.
    n_trees : int, optional
        Number of trees built (accuracy/size trade-off).
    search_k : int, optional
        Query-time node budget passed to Annoy; ``-1`` lets Annoy choose.
    impl : str, optional
        Which index class to use: ``"auto"`` (default; high-level first, else
        Cython), ``"highlevel"``, or ``"cython"``.
    dtype : str or None, optional
        Embedding precision (e.g. ``"float32"``, ``"float64"``). Forwarded only
        to implementations that accept it (the Cython class); silently ignored
        otherwise.
    index_dtype : str or None, optional
        Item-id integer width (e.g. ``"int32"``, ``"uint64"``) for very large
        corpora. Cython class only; ignored otherwise.

    Notes
    -----
    **Developer note:** Annoy's ``"angular"`` metric operates on internally
    normalised vectors, so for two vectors with cosine similarity ``c`` the
    reported distance is ``d = sqrt(2 * (1 - c))``. Cosine is recovered exactly
    as ``c = 1 - d**2 / 2`` — the correct inverse (a plain ``1 - d/2`` is not).
    """

    name = "annoy"

    def __init__(
        self,
        *,
        metric: str = "angular",
        n_trees: int = 10,
        search_k: int = -1,
        impl: str = "auto",
        dtype: str | None = None,
        index_dtype: str | None = None,
    ) -> None:
        if n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {n_trees}")
        self._metric = metric
        self._n_trees = int(n_trees)
        self._search_k = int(search_k)
        self._impl = impl
        self._dtype = dtype
        self._index_dtype = index_dtype
        self._index: Any = None
        self._dim: int = 0
        self._resolved_impl: str | None = None

    @classmethod
    def is_available(cls) -> bool:
        try:
            _resolve_annoy_index_cls("auto")
            return True
        except Exception:  # noqa: BLE001 - any import/native failure => unavailable
            return False

    def _construct(self, index_cls: Any) -> Any:
        """Construct an index, forwarding dtype kwargs only when accepted."""
        kwargs: dict[str, Any] = {}
        if self._dtype is not None:
            kwargs["dtype"] = self._dtype
        if self._index_dtype is not None:
            kwargs["index_dtype"] = self._index_dtype
        try:
            return index_cls(self._dim, self._metric, **kwargs)
        except TypeError:
            if kwargs:
                logger.debug(
                    "annoy impl %r does not accept %s; constructing without them",
                    self._resolved_impl,
                    sorted(kwargs),
                )
            return index_cls(self._dim, self._metric)

    def build(self, embeddings: Any) -> None:
        np = _require_numpy()
        index_cls, impl_name = _resolve_annoy_index_cls(self._impl)
        self._resolved_impl = impl_name
        embs = _validate_embeddings(np, embeddings)
        self._dim = int(embs.shape[1])
        index = self._construct(index_cls)
        n = int(embs.shape[0])
        # Prefer the validated, finiteness-checked bulk path when present
        # (high-level Index); otherwise loop add_item (Cython Index).
        if hasattr(index, "add_items"):
            index.add_items(embs, ids=list(range(n)))
        else:
            for i in range(n):
                index.add_item(i, embs[i].tolist())
        index.build(self._n_trees)
        self._index = index

    def query(self, vector: Any, k: int) -> list[tuple[int, float]]:
        np = _require_numpy()
        qe = _validate_query_vector(np, vector, self._dim)
        if float(np.linalg.norm(qe)) == 0.0:
            return []
        k = max(1, int(k))
        ids, dists = self._nns(qe.tolist(), k)
        return [(int(i), self._distance_to_score(float(d))) for i, d in zip(ids, dists)]

    def _nns(self, vec: Sequence[float], k: int) -> tuple[Any, Any]:
        """Call ``get_nns_by_vector`` tolerating both known signatures."""
        get = self._index.get_nns_by_vector
        try:
            return get(vec, k, self._search_k, include_distances=True)
        except TypeError:  # variant without a positional search_k
            return get(vec, k, include_distances=True)

    def _distance_to_score(self, d: float) -> float:
        """Convert an Annoy distance to the unified cosine score in [-1, 1]."""
        if self._metric in ("angular", "cosine"):
            # angular distance on normalised vectors: d = sqrt(2*(1 - cos))
            cos = 1.0 - (d * d) / 2.0
            return max(-1.0, min(1.0, cos))
        # Non-cosine metrics have no cosine; expose a bounded higher-is-better
        # score in (0, 1] so ordering and thresholds still behave.
        return 1.0 / (1.0 + max(0.0, d))


# =====================================================================
# FAISS
# =====================================================================


class FaissBackend(ANNBackend):
    """ANN backend over a FAISS ``IndexFlatIP`` on normalised vectors.

    Inner product on unit-normalised vectors equals cosine similarity, so the
    raw FAISS score already satisfies the unified contract.
    """

    name = "faiss"

    def __init__(self) -> None:
        self._index: Any = None
        self._dim: int = 0

    @classmethod
    def is_available(cls) -> bool:
        try:
            import faiss  # type: ignore[import]  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False

    def build(self, embeddings: Any) -> None:
        np = _require_numpy()
        import faiss  # type: ignore[import]  # noqa: PLC0415

        embs = _validate_embeddings(np, embeddings)
        self._dim = int(embs.shape[1])
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        normed = np.ascontiguousarray(embs / norms, dtype=np.float32)
        index = faiss.IndexFlatIP(self._dim)
        index.add(normed)
        self._index = index

    def query(self, vector: Any, k: int) -> list[tuple[int, float]]:
        np = _require_numpy()
        qe = _validate_query_vector(np, vector, self._dim)
        norm_q = float(np.linalg.norm(qe))
        if norm_q == 0.0:
            return []
        q = np.ascontiguousarray((qe / norm_q).reshape(1, -1), dtype=np.float32)
        k = max(1, min(int(k), self._index.ntotal))
        scores, idxs = self._index.search(q, k)
        out: list[tuple[int, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if int(idx) == -1:
                continue
            out.append((int(idx), max(-1.0, min(1.0, float(score)))))
        return out


# =====================================================================
# Voyager
# =====================================================================


class VoyagerBackend(ANNBackend):
    """ANN backend over a Voyager cosine-space index.

    Voyager's cosine *distance* is ``1 - cosine``; cosine similarity is
    recovered as ``1 - distance`` for the unified contract.
    """

    name = "voyager"

    def __init__(self) -> None:
        self._index: Any = None
        self._dim: int = 0

    @classmethod
    def is_available(cls) -> bool:
        try:
            import voyager  # type: ignore[import]  # noqa: F401, PLC0415

            return True
        except ImportError:
            return False

    def build(self, embeddings: Any) -> None:
        np = _require_numpy()
        import voyager  # type: ignore[import]  # noqa: PLC0415

        embs = _validate_embeddings(np, embeddings)
        self._dim = int(embs.shape[1])
        index = voyager.Index(voyager.Space.Cosine, num_dimensions=self._dim)
        index.add_items(embs)
        self._index = index

    def query(self, vector: Any, k: int) -> list[tuple[int, float]]:
        np = _require_numpy()
        qe = _validate_query_vector(np, vector, self._dim)
        if float(np.linalg.norm(qe)) == 0.0:
            return []
        k = max(1, int(k))
        ids, distances = self._index.query(qe, k=k)
        return [
            (int(i), max(-1.0, min(1.0, 1.0 - float(dist))))
            for i, dist in zip(ids, distances)
        ]


# =====================================================================
# Selection
# =====================================================================

_BACKENDS: dict[str, type[ANNBackend]] = {
    "annoy": AnnoyBackend,
    "faiss": FaissBackend,
    "voyager": VoyagerBackend,
    "bruteforce": BruteForceBackend,
    "brute": BruteForceBackend,  # convenience alias
}


def list_available_backends() -> list[str]:
    """Return the canonical names of currently importable backends."""
    seen: dict[str, None] = {}
    for name in DEFAULT_BACKEND_ORDER:
        if _BACKENDS[name].is_available():
            seen[name] = None
    return list(seen)


def select_backend(
    name: str = "auto",
    *,
    annoy_metric: str = "angular",
    annoy_n_trees: int = 10,
    annoy_search_k: int = -1,
    annoy_impl: str = "auto",
    annoy_dtype: str | None = None,
    annoy_index_dtype: str | None = None,
) -> ANNBackend:
    """Construct an ANN backend by name.

    Parameters
    ----------
    name : str, optional
        ``"auto"`` (default) resolves the first available backend in
        :data:`DEFAULT_BACKEND_ORDER`. An explicit name
        (``"annoy"``, ``"faiss"``, ``"voyager"``, ``"bruteforce"``/``"brute"``)
        is honoured or raises if that backend is unavailable.
    annoy_metric, annoy_n_trees, annoy_search_k, annoy_impl, annoy_dtype, annoy_index_dtype
        Forwarded to :class:`AnnoyBackend` (ignored by other backends).
        ``annoy_impl`` selects the high-level or Cython index class; the
        ``dtype`` options apply only to the Cython class.

    Returns
    -------
    ANNBackend
        An unbuilt backend instance.

    Raises
    ------
    ValueError
        If *name* is not a known backend or ``"auto"``.
    RuntimeError
        If an explicitly named backend is unavailable, or if ``"auto"`` finds
        no available backend (numpy missing).
    """
    key = (name or "auto").strip().lower()

    def _make(cls: type[ANNBackend]) -> ANNBackend:
        if cls is AnnoyBackend:
            return AnnoyBackend(
                metric=annoy_metric,
                n_trees=annoy_n_trees,
                search_k=annoy_search_k,
                impl=annoy_impl,
                dtype=annoy_dtype,
                index_dtype=annoy_index_dtype,
            )
        return cls()

    if key == "auto":
        for candidate in DEFAULT_BACKEND_ORDER:
            cls = _BACKENDS[candidate]
            if cls.is_available():
                logger.debug("select_backend: auto-selected %r", candidate)
                return _make(cls)
        raise RuntimeError(
            "No vector backend is available. Semantic search requires numpy; "
            "install it with `pip install scikit-plots[corpus]`."
        )

    if key not in _BACKENDS:
        valid = sorted(set(_BACKENDS) | {"auto"})
        raise ValueError(f"unknown backend {name!r}; choose from {valid}")

    cls = _BACKENDS[key]
    if not cls.is_available():
        raise RuntimeError(
            f"backend {key!r} is not available in this environment. "
            f"Install the corresponding package or use backend='auto'."
        )
    return _make(cls)
