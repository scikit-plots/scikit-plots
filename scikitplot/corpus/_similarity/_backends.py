# scikitplot/corpus/_similarity/_backends.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

r"""
Pluggable approximate-nearest-neighbour (ANN) backends for dense semantic search.

This module centralises every vector-index implementation behind a single,
uniform contract so that :class:`~scikitplot.corpus._similarity.RetrievalIndex`
(and any consumer such as :mod:`scikitplot.mcp`) never has to branch on the
concrete backend, and so that *scores mean the same thing regardless of backend*.

Unified score contract
-----------------------
Every backend's :meth:`VectorIndexBackend.query` returns ``(row_index, score)`` pairs
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

So for this patch:

BruteForce → generic dependency probe
FAISS      → generic dependency probe
Voyager    → generic dependency probe
Annoy      → special capability probe for now
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, ClassVar, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BACKEND_ORDER",
    "AnnoyBackend",
    "BruteForceBackend",
    "FaissBackend",
    "VectorIndexBackend",
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


class VectorIndexBackend:
    """Uniform contract for a dense vector index.

    Subclasses build an index from a 2-D embedding matrix and answer
    top-``k`` cosine-similarity queries. Every subclass guarantees the
    *unified score contract* described in the module docstring.

    Attributes
    ----------
    name : str
        Stable backend identifier (e.g. ``"annoy"``).
    """

    name: ClassVar[str] = "base"
    dependency_modules: ClassVar[str | tuple[str, ...] | None] = None

    # @classmethod
    # def is_available(cls) -> bool:
    #     """Whether this backend's runtime dependencies are importable."""
    #     raise NotImplementedError
    @classmethod
    def is_available(cls) -> bool:
        """Return whether this backend's optional dependencies are discoverable."""
        dependency = cls.dependency_modules
        if dependency is None:
            return True
        modules = (dependency,) if isinstance(dependency, str) else dependency
        if not modules:
            return False
        try:
            return all(
                importlib.util.find_spec(module) is not None for module in modules
            )
        except (ImportError, ValueError):
            return False

    # -- declarative capability members (P-I1-15 / finding F-R06-01) ---------
    #
    # The contract previously exposed three of the eleven properties §18
    # requires: availability, build and query.  The other eight existed only as
    # private attributes, implicit conventions, or not at all -- which is why a
    # caller could not ask what score scale a backend returns, whether it can
    # persist, or what it costs.  Each has a base default describing today's
    # behaviour, so every existing subclass keeps working unchanged.

    metric: str = "cosine"
    """Distance/similarity metric this backend computes."""

    score_semantics: str = "cosine_similarity"
    """Meaning of the ``score`` in :meth:`query` results.

    ``"cosine_similarity"`` -- range ``[-1, 1]``, higher is better.
    ``"bounded_inverse_distance"`` -- range ``(0, 1]``, higher is better, but
    **not** a similarity: it is ``1 / (1 + d)`` and is not comparable with a
    cosine score.  Declaring this is what lets a threshold be validated against
    the scale it will actually be applied to (finding F-R06-02).
    """

    dtype: str = "float32"
    """Element type of the indexed vectors."""

    supports_persistence: bool = True
    """Whether this backend's index can be persisted and reloaded.

    ``True`` since :class:`~scikitplot.corpus.ANNIndexArtifact` landed: an
    artifact writes the native payload alongside a versioned ordinal->doc_id
    sidecar and the embedding manifest, so a reloaded index can prove its rows
    still name the documents they named at write time.  Before that, R06 found
    zero persistence surface in this module and the declaration was ``False``
    -- accurately.
    """

    thread_safety: str = "concurrent_read"
    """``"single_thread"``, ``"concurrent_read"`` or ``"full"``."""

    memory_profile: str = "in_memory"
    """``"in_memory"``, ``"mmap"`` or ``"hybrid"``."""

    @property
    def dimension(self) -> int | None:
        """Vector length this index was built with, ``None`` before build."""
        return getattr(self, "_dim", None)

    def capabilities(self) -> dict[str, Any]:
        """Return every declared capability as a mapping.

        Returns
        -------
        dict
            Suitable for a capability report or a build manifest.
        """
        return {
            "name": self.name,
            "metric": self.metric,
            "score_semantics": self.score_semantics,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "supports_persistence": self.supports_persistence,
            "thread_safety": self.thread_safety,
            "memory_profile": self.memory_profile,
        }

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


class BruteForceBackend(VectorIndexBackend):
    """Exact cosine similarity via a normalised dot product.

    Deterministic and dependency-free beyond ``numpy``. Suited to small and
    medium corpora and used as the guaranteed fallback for ``backend="auto"``.
    """

    name = "bruteforce"
    dependency_modules = "numpy"

    def __init__(self) -> None:
        self._normed: Any = None
        self._dim: int = 0

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


class AnnoyBackend(VectorIndexBackend):
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
    dependency_modules = (
        "scikitplot.annoy",
        "scikitplot.annoy._annoy",
    )

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
        # The instance declares what it ACTUALLY computes, which may differ from
        # the class default.  Annoy's non-cosine metrics return a bounded
        # inverse distance, not a similarity -- and a threshold tuned on one
        # scale silently mis-filters on the other (finding F-R06-02).
        self.metric = metric
        self.score_semantics = (
            "cosine_similarity"
            if metric in ("angular", "cosine")
            else "bounded_inverse_distance"
        )
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


class FaissBackend(VectorIndexBackend):
    """ANN backend over a FAISS ``IndexFlatIP`` on normalised vectors.

    Inner product on unit-normalised vectors equals cosine similarity, so the
    raw FAISS score already satisfies the unified contract.
    """

    name = "faiss"
    dependency_modules = "faiss"

    def __init__(self) -> None:
        self._index: Any = None
        self._dim: int = 0

    def build(self, embeddings: Any) -> None:
        import faiss  # type: ignore[import]  # noqa: F401, PLC0415

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


class VoyagerBackend(VectorIndexBackend):
    """ANN backend over a Voyager cosine-space index.

    Voyager's cosine *distance* is ``1 - cosine``; cosine similarity is
    recovered as ``1 - distance`` for the unified contract.
    """

    name = "voyager"
    dependency_modules = "voyager"

    def __init__(self) -> None:
        self._index: Any = None
        self._dim: int = 0

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

#: Canonical backend identities.  Exactly one entry per implementation, so
#: anything that iterates this mapping counts each backend once.
#:
#: Finding F-R02-06 / F-R06-05: ``"brute"`` used to live here as a second entry
#: for :class:`BruteForceBackend`, which made ``capability_snapshot()`` report
#: five backends for four classes -- with contradictory versions, because the
#: distribution map had no entry for the alias.  Aliases now resolve through
#: :data:`_BACKEND_ALIASES` instead.
_BACKENDS: dict[str, type[VectorIndexBackend]] = {
    "annoy": AnnoyBackend,
    "bruteforce": BruteForceBackend,
    "faiss": FaissBackend,
    "voyager": VoyagerBackend,
}

#: Accepted alternative spellings, mapped to canonical identities.  Callers may
#: pass either; anything that *enumerates* backends sees only the canonical set.
_BACKEND_ALIASES: dict[str, str] = {
    "brute": "bruteforce",
}


def canonical_backend_name(name: str) -> str:
    """Resolve ``name`` to its canonical backend identity.

    Parameters
    ----------
    name : str
        A canonical name or a known alias.

    Returns
    -------
    str
        The canonical identity.  Unknown names are returned unchanged so the
        caller's own error handling reports them.

    Examples
    --------
    >>> canonical_backend_name("brute")
    'bruteforce'
    >>> canonical_backend_name("annoy")
    'annoy'
    """
    return _BACKEND_ALIASES.get(name, name)


def backend_aliases(name: str) -> list[str]:
    """Return the aliases pointing at canonical backend ``name``."""
    return sorted(a for a, c in _BACKEND_ALIASES.items() if c == name)


def list_available_backends() -> list[str]:
    """Return the canonical names of currently importable backends."""
    seen: dict[str, None] = {}
    for name in DEFAULT_BACKEND_ORDER:
        if _BACKENDS[canonical_backend_name(name)].is_available():
            seen[name] = None
    return list(seen)


def select_backend(
    name: Any = "auto",
    *,
    index_kwargs: Mapping[str, Any] | None = None,
    annoy_metric: str = "angular",
    annoy_n_trees: int = 10,
    annoy_search_k: int = -1,
    annoy_impl: str = "auto",
    annoy_dtype: str | None = None,
    annoy_index_dtype: str | None = None,
    **kwargs: Any,
) -> VectorIndexBackend:
    """Construct an ANN backend by name.

    Parameters
    ----------
    name : str or VectorIndexBackend subclass, optional
        ``"auto"`` (default) resolves the first available backend in
        :data:`DEFAULT_BACKEND_ORDER`. An explicit built-in name is honoured
        or raises if unavailable. A custom :class:`VectorIndexBackend` subclass
        may be supplied directly.
    index_kwargs : mapping, optional
        Generic constructor keyword arguments for the selected backend. New
        code should prefer this mapping instead of adding backend-specific
        parameters to :class:`RetrievalConfig`.
    annoy_metric, annoy_n_trees, annoy_search_k, annoy_impl, annoy_dtype, annoy_index_dtype
        Backward-compatible Annoy constructor settings. They are merged only
        when the resolved backend is :class:`AnnoyBackend`; generic
        ``index_kwargs`` win over default legacy values.
    **kwargs : Any
        Compatibility form for generic backend constructor keyword arguments.
        Conflicting keys between ``index_kwargs`` and ``kwargs`` raise.

    Returns
    -------
    VectorIndexBackend
        An unbuilt backend instance.

    Raises
    ------
    ValueError
        If *name* is not a known backend or ``"auto"``.
    RuntimeError
        If an explicitly named backend is unavailable, or if ``"auto"`` finds
        no available backend (numpy missing).
    """
    constructor_kwargs = dict(index_kwargs or {})
    if any(not isinstance(key, str) for key in constructor_kwargs):
        raise TypeError(
            "index_kwargs keys must be strings because they are constructor kwargs"
        )
    for key_name, value in kwargs.items():
        if key_name in constructor_kwargs and constructor_kwargs[key_name] != value:
            raise ValueError(
                f"conflicting backend constructor option {key_name!r}: "
                f"index_kwargs has {constructor_kwargs[key_name]!r}, "
                f"keyword argument has {value!r}"
            )
        constructor_kwargs[key_name] = value

    def _make(cls: type[VectorIndexBackend], resolved_name: str) -> VectorIndexBackend:
        options = dict(constructor_kwargs)
        if cls is AnnoyBackend:
            legacy = {
                "metric": (annoy_metric, "angular", "annoy_metric"),
                "n_trees": (annoy_n_trees, 10, "annoy_n_trees"),
                "search_k": (annoy_search_k, -1, "annoy_search_k"),
                "impl": (annoy_impl, "auto", "annoy_impl"),
                "dtype": (annoy_dtype, None, "annoy_dtype"),
                "index_dtype": (annoy_index_dtype, None, "annoy_index_dtype"),
            }
            for option, (legacy_value, legacy_default, legacy_name) in legacy.items():
                if option in options:
                    if (
                        legacy_value != legacy_default
                        and options[option] != legacy_value
                    ):
                        raise ValueError(
                            f"conflicting Annoy configuration: {legacy_name}="
                            f"{legacy_value!r} but index_kwargs[{option!r}]="
                            f"{options[option]!r}"
                        )
                else:
                    options[option] = legacy_value
        try:
            return cls(**options)
        except TypeError as exc:
            raise TypeError(
                f"invalid index_kwargs for backend {resolved_name!r}: {exc}"
            ) from exc

    if isinstance(name, type):
        if not issubclass(name, VectorIndexBackend):
            raise TypeError(
                f"backend classes must subclass VectorIndexBackend; got {name!r}"
            )
        if not name.is_available():
            raise RuntimeError(
                f"backend class {name.__module__}.{name.__qualname__} is not "
                "available in this environment."
            )
        return _make(name, getattr(name, "name", name.__name__))

    if not isinstance(name, str):
        raise TypeError("backend must be a backend name or VectorIndexBackend subclass")
    key = (name or "auto").strip().lower()

    if key == "auto":
        for candidate in DEFAULT_BACKEND_ORDER:
            cls = _BACKENDS[candidate]
            if cls.is_available():
                logger.debug("select_backend: auto-selected %r", candidate)
                return _make(cls, candidate)
        raise RuntimeError(
            "No vector backend is available. Semantic search requires numpy; "
            "install it with `pip install scikit-plots[corpus]`."
        )

    key = canonical_backend_name(key)
    if key not in _BACKENDS:
        valid = sorted(set(_BACKENDS) | set(_BACKEND_ALIASES) | {"auto"})
        raise ValueError(f"unknown backend {name!r}; choose from {valid}")

    cls = _BACKENDS[key]
    if not cls.is_available():
        raise RuntimeError(
            f"backend {key!r} is not available in this environment. "
            f"Install the corresponding package or use backend='auto'."
        )
    return _make(cls, key)
