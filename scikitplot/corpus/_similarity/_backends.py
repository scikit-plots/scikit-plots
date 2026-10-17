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

Unified score contract:

Every backend's :meth:`VectorIndexBackend.query` returns ``(row_index, score)`` pairs
where ``score`` is **cosine similarity in the closed interval ``[-1.0, 1.0]``**
(higher is better), sorted in descending score order with deterministic,
index-ascending tie breaking. This makes ``semantic_threshold`` comparisons and
hybrid fusion identical across backends.

Backend selection order:

``select_backend("auto")`` resolves the first *available* backend in
:data:`DEFAULT_BACKEND_ORDER`, which is ``annoy`` first (it is an internal
dependency of scikit-plots), then ``faiss``, then ``voyager``, then the always
available pure-``numpy`` ``bruteforce`` floor. An explicitly named backend that
is unavailable raises :class:`RuntimeError` with an actionable message rather
than silently degrading.

Fallback vocabulary (R2-5):

Every backend instance reports :meth:`VectorIndexBackend.provenance`: the
``requested_backend`` (``"auto"``, a name, or a class name), the ``backend``
that runs, the ``fallback_reason`` when they differ for a reason other than
plain absence, and the ``allow_fallback`` switch. Under ``"auto"`` an absent
backend is not a fallback (``auto`` means "the first one installed"); one that
is installed but cannot be loaded is (``"annoy_broken"``), and
``allow_fallback=False`` refuses it instead. :class:`AnnoyBackend` reports the
same names for its binding: ``impl="auto"`` picks a binding that accepts every
requested option, and an option the chosen binding cannot honour is reported
as ``"option_unsupported"`` (or refused when ``allow_fallback=False``), never
dropped silently.

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
import sys
from typing import Any, ClassVar, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BACKEND_ORDER",
    "AnnoyBackend",
    "BruteForceBackend",
    "FaissBackend",
    "FallbackRefusedError",
    "VectorIndexBackend",
    "VoyagerBackend",
    "list_available_backends",
    "select_backend",
]

# Backends tried, in order, when ``backend="auto"``. Annoy first: it is an
# internal scikit-plots dependency and provides a persistent, memory-mapped
# vector index; ``bruteforce`` is the always-available numpy floor.
DEFAULT_BACKEND_ORDER = ("annoy", "faiss", "voyager", "bruteforce")


class FallbackRefusedError(RuntimeError):
    """A fallback was needed and ``allow_fallback=False`` refused it (R2-5).

    Raised when the dense backend (or an Annoy binding) cannot run as
    configured: an installed backend fails to load under ``"auto"``, a
    preferred binding cannot load, or the running binding cannot honour a
    requested option. It is a :class:`RuntimeError`, so it is never mistaken
    for the invalid-data ``ValueError`` that disables a dense index.
    """


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


#: Constructor options each Annoy binding accepts beyond ``(dim, metric)``.
#: Declared rather than discovered by trial construction: ``impl="auto"`` must
#: choose a binding *before* it builds (F-40). ``test__fallback_vocabulary.py``
#: checks the table against both classes.
_ANNOY_IMPL_OPTIONS: dict[str, frozenset[str]] = {
    "highlevel": frozenset(),
    "cython": frozenset({"dtype", "index_dtype"}),
}

#: Binding preference under ``impl="auto"`` when every binding accepts the
#: requested options: the high-level class first (validated bulk ``add_items``).
_ANNOY_IMPL_ORDER: tuple[str, ...] = ("highlevel", "cython")


def _import_annoy_impl(impl: str) -> Any:
    """Import and return the Annoy ``Index`` class of binding ``impl``."""
    if impl == "cython":
        from scikitplot.annoy._annoy import Index  # noqa: PLC0415

        return Index
    from scikitplot.annoy import Index  # noqa: PLC0415

    return Index


def _resolve_annoy_impl(
    impl: str,
    options: frozenset[str],
    *,
    allow_fallback: bool,
) -> tuple[Any, str, frozenset[str], str | None]:
    """Choose the Annoy binding for ``impl`` and the requested ``options``.

    Parameters
    ----------
    impl : str
        ``"auto"``, ``"highlevel"`` (``scikitplot.annoy.Index``) or
        ``"cython"`` (``scikitplot.annoy._annoy.Index``).
    options : frozenset of str
        Names of the binding options requested (``"dtype"``,
        ``"index_dtype"``).
    allow_fallback : bool
        Whether a binding that cannot honour every option (or a second choice
        under ``"auto"``) may run instead of raising.

    Returns
    -------
    tuple
        ``(index_cls, impl_name, unsupported_options, fallback_reason)``;
        ``fallback_reason`` is ``None``, ``"option_unsupported"`` or
        ``"<binding>_unavailable"``.

    Raises
    ------
    ValueError
        If ``impl`` is unknown.
    FallbackRefusedError
        If ``allow_fallback`` is ``False`` and an option cannot be honoured,
        or the preferred binding under ``"auto"`` cannot load.
    ImportError
        If the requested binding (or, under ``"auto"``, every binding) cannot
        be loaded.

    Notes
    -----
    **Developer.** Under ``"auto"`` the bindings that accept every requested
    option come first, in :data:`_ANNOY_IMPL_ORDER`; the first that loads
    runs. A binding that loads but cannot honour an option is used only when
    none that can is loadable (F-40: ``dtype="float64"`` used to run on the
    high-level class, which discarded it, while the Cython class that honours
    it was installed).
    """
    key = (impl or "auto").strip().lower()
    if key != "auto" and key not in _ANNOY_IMPL_OPTIONS:
        raise ValueError(
            f"annoy impl must be 'auto', 'highlevel', or 'cython', got {impl!r}"
        )
    candidates = (
        sorted(
            _ANNOY_IMPL_ORDER,
            key=lambda name: not options <= _ANNOY_IMPL_OPTIONS[name],
        )
        if key == "auto"
        else [key]
    )
    first_error: BaseException | None = None
    for position, name in enumerate(candidates):
        try:
            index_cls = _import_annoy_impl(name)
        except Exception as exc:  # noqa: BLE001 - classified below
            if key != "auto":
                raise
            first_error = first_error or exc
            continue
        unsupported = options - _ANNOY_IMPL_OPTIONS[name]
        if unsupported:
            if not allow_fallback:
                raise FallbackRefusedError(
                    f"annoy binding {name!r} does not accept {sorted(unsupported)}"
                    + (
                        " and no binding that does could be loaded"
                        if key == "auto"
                        else ""
                    )
                    + "; allow_fallback=False refuses to build without them"
                ) from first_error
            return index_cls, name, unsupported, "option_unsupported"
        if position == 0:
            return index_cls, name, unsupported, None
        if not allow_fallback:
            raise FallbackRefusedError(
                f"annoy binding {candidates[0]!r} cannot be loaded "
                f"({type(first_error).__name__}: {first_error}); "
                "allow_fallback=False refuses another binding"
            ) from first_error
        return index_cls, name, unsupported, f"{candidates[0]}_unavailable"
    raise ImportError(
        f"no Annoy binding could be loaded ({type(first_error).__name__}: "
        f"{first_error})"
    ) from first_error


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

    # Selection provenance (R2-5), set by :func:`select_backend`; a backend
    # constructed directly requested itself.
    _requested_backend: str | None = None
    _selection_reason: str | None = None
    _allow_fallback: bool = True

    #: Whether answering :meth:`is_available` imports the backend's packages
    #: (a native library); ``capability_snapshot(load=False)`` then asks only
    #: once the modules in :attr:`probe_modules` are already loaded (F-50).
    loads_on_probe: ClassVar[bool] = False
    probe_modules: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def is_available(cls) -> bool:
        """Return whether this backend can be used now: installed and loading."""
        return cls.is_installed() and cls.load_error() is None

    @classmethod
    def load_error(cls) -> BaseException | None:
        """Return the exception that stops this backend loading, or ``None``.

        Imports :attr:`dependency_modules` (when installed) and returns the
        first failure. An installed backend with a load error is *broken*,
        never *absent*.
        """
        dependency = cls.dependency_modules
        if dependency is None or not cls.is_installed():
            return None
        modules = (dependency,) if isinstance(dependency, str) else dependency
        try:
            for module in modules:
                importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return exc
        return None

    @classmethod
    def is_installed(cls) -> bool:
        """Return whether this backend's packages can be found, importing none."""
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

    @property
    def requested_backend(self) -> str:
        """What the configuration asked for: ``"auto"``, a name or a class."""
        return self._requested_backend or self.name

    @property
    def fallback_reason(self) -> str | None:
        """Why the backend that runs is not the one requested, or ``None``."""
        return self._selection_reason

    @property
    def allow_fallback(self) -> bool:
        """Whether a fallback was allowed (``False`` raises instead)."""
        return self._allow_fallback

    def provenance(self) -> dict[str, Any]:
        """Return what was requested, what runs, and why they differ (R2-5).

        Returns
        -------
        dict
            ``requested_backend``, ``backend``, ``fallback_reason`` and
            ``allow_fallback``; JSON-compatible.
        """
        return {
            "requested_backend": self.requested_backend,
            "backend": self.name,
            "fallback_reason": self.fallback_reason,
            "allow_fallback": self.allow_fallback,
        }

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
        Which index class to use: ``"auto"`` (default: a binding that accepts
        every requested option, the high-level class first), ``"highlevel"``,
        or ``"cython"``.
    dtype : str or None, optional
        Embedding precision (e.g. ``"float32"``, ``"float64"``). Accepted by
        the Cython class only; ``impl="auto"`` therefore picks it.
    index_dtype : str or None, optional
        Item-id integer width (e.g. ``"int32"``, ``"uint64"``) for very large
        corpora. Cython class only, like ``dtype``.
    allow_fallback : bool, optional
        What happens when the binding that runs cannot honour ``dtype`` or
        ``index_dtype`` (``impl="highlevel"``, or no accepting binding
        loads), or the preferred binding under ``"auto"`` cannot load:
        ``True`` (default) builds anyway and reports it
        (:attr:`fallback_reason`, :attr:`discarded_options`, one warning);
        ``False`` raises :class:`FallbackRefusedError` when :meth:`build` is
        called. ``dtype`` and ``index_dtype`` spellings are validated by the
        binding that honours them; a value dropped by a fallback is reported
        verbatim.

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
    # Probing availability imports the bindings (and scikit-learn, SciPy).
    loads_on_probe = True
    probe_modules = ("scikitplot.annoy",)

    def __init__(
        self,
        *,
        metric: str = "angular",
        n_trees: int = 10,
        search_k: int = -1,
        impl: str = "auto",
        dtype: str | None = None,
        index_dtype: str | None = None,
        allow_fallback: bool = True,
    ) -> None:
        if n_trees < 1:
            raise ValueError(f"n_trees must be >= 1, got {n_trees}")
        if not isinstance(allow_fallback, bool):
            raise TypeError(
                f"allow_fallback must be a bool, got {type(allow_fallback).__name__}"
            )
        if impl is None:
            impl = "auto"  # accepted before R2-5; kept
        if not isinstance(impl, str):
            raise TypeError(f"impl must be a str or None, got {type(impl).__name__}")
        for option, value in (("dtype", dtype), ("index_dtype", index_dtype)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise TypeError(
                    f"{option} must be a non-empty str or None, got {value!r}; the "
                    "binding that honours it validates the spelling"
                )
        key = impl.strip().lower() or "auto"
        if key != "auto" and key not in _ANNOY_IMPL_OPTIONS:
            raise ValueError(
                f"annoy impl must be 'auto', 'highlevel', or 'cython', got {impl!r}"
            )
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
        self._impl = key
        self._dtype = dtype
        self._index_dtype = index_dtype
        self._allow_fallback = allow_fallback
        self._index: Any = None
        self._dim: int = 0
        self._resolved_impl: str | None = None
        self._impl_reason: str | None = None
        self._discarded_kwargs: dict[str, Any] = {}

    @classmethod
    def is_installed(cls) -> bool:
        """Return whether ``scikitplot.annoy`` can be found, importing nothing.

        Notes
        -----
        **Developer.** Only the package is looked up: finding a submodule
        (``scikitplot.annoy._annoy``) would import the package, whose mixins
        import scikit-learn and SciPy (F-50).
        """
        if "scikitplot.annoy" in sys.modules:
            return sys.modules["scikitplot.annoy"] is not None
        try:
            return importlib.util.find_spec("scikitplot.annoy") is not None
        except (ImportError, ValueError):
            return False

    @classmethod
    def load_error(cls) -> BaseException | None:
        """Return why no Annoy binding loads, or ``None`` when one does.

        Notes
        -----
        **Developer.** This imports the bindings, so it costs what using them
        costs; :func:`~scikitplot.corpus.capability_snapshot` calls it only
        when asked to load (``load=True``) or once Annoy is already loaded.
        """
        error: BaseException | None = None
        for name in _ANNOY_IMPL_ORDER:
            try:
                _import_annoy_impl(name)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                error = error or exc
                continue
            return None
        return error

    def _requested_options(self) -> dict[str, Any]:
        """Return the binding options this backend was asked for."""
        options: dict[str, Any] = {}
        if self._dtype is not None:
            options["dtype"] = self._dtype
        if self._index_dtype is not None:
            options["index_dtype"] = self._index_dtype
        return options

    def build(self, embeddings: Any) -> None:
        np = _require_numpy()
        options = self._requested_options()
        index_cls, impl_name, unsupported, reason = _resolve_annoy_impl(
            self._impl, frozenset(options), allow_fallback=self._allow_fallback
        )
        embs = _validate_embeddings(np, embeddings)
        if reason is not None and reason != self._impl_reason:
            logger.warning(
                "AnnoyBackend: binding %r runs (requested %r), fallback_reason=%r%s;"
                " see provenance().",
                impl_name,
                self._impl,
                reason,
                f", built without {sorted(unsupported)}" if unsupported else "",
            )
        self._resolved_impl = impl_name
        self._impl_reason = reason
        # The precision this index holds (F-40 made float64 real).
        self.dtype = (
            options["dtype"]
            if "dtype" in options and "dtype" not in unsupported
            else VectorIndexBackend.dtype
        )
        self._discarded_kwargs = {name: options[name] for name in sorted(unsupported)}
        self._dim = int(embs.shape[1])
        kwargs = {k: v for k, v in options.items() if k not in unsupported}
        index = index_cls(self._dim, self._metric, **kwargs)
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

    @property
    def discarded_options(self) -> dict[str, Any]:
        """Options this backend requested and the binding that ran cannot honour.

        Returns
        -------
        dict
            Empty when everything requested was honoured. Non-empty only when
            ``allow_fallback`` is ``True`` and :attr:`fallback_reason` is
            ``"option_unsupported"``.
        """
        return dict(self._discarded_kwargs)

    @property
    def fallback_reason(self) -> str | None:
        """Why what runs differs from what was requested, or ``None``.

        The selection reason (``"<backend>_broken"``) first, then the
        binding's (``"option_unsupported"``, ``"<binding>_unavailable"``).
        """
        return self._selection_reason or self._impl_reason

    def provenance(self) -> dict[str, Any]:
        """Return :meth:`VectorIndexBackend.provenance` plus the binding.

        Returns
        -------
        dict
            Adds ``requested_impl``, ``impl`` (``None`` before :meth:`build`)
            and ``discarded_options``.
        """
        record = super().provenance()
        record.update(
            requested_impl=self._impl,
            impl=self._resolved_impl,
            discarded_options=self.discarded_options,
        )
        return record

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
    loads_on_probe = True
    probe_modules = ("faiss",)

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
    loads_on_probe = True
    probe_modules = ("voyager",)

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


def _load_failure(cls: type[VectorIndexBackend]) -> BaseException | None:
    """Return why an installed backend cannot load, or ``None``.

    ``None`` also when it is not installed, or when nothing is known: a
    backend is called broken only on evidence of a failed load.
    """
    return cls.load_error() if cls.is_installed() else None


def _unavailable(cls: type[VectorIndexBackend], label: str) -> str:
    """Say whether an unavailable backend is absent or failing to load."""
    error = _load_failure(cls)
    if error is not None:
        return (
            f"backend {label} is installed but cannot be loaded "
            f"({type(error).__name__}: {error})."
        )
    return f"backend {label} is not available in this environment."


def _first_usable_backend(*, allow_fallback: bool) -> tuple[str, str | None]:
    """Resolve ``"auto"``: the first usable backend and the fallback reason.

    An absent backend is passed over silently (``"auto"`` means the first one
    installed); an installed one that cannot load is a fallback, reported as
    ``"<name>_broken"`` with a warning, or refused when ``allow_fallback`` is
    ``False``.
    """
    reason: str | None = None
    for candidate in DEFAULT_BACKEND_ORDER:
        cls = _BACKENDS[candidate]
        if cls.is_available():
            logger.debug("select_backend: auto-selected %r", candidate)
            return candidate, reason
        error = _load_failure(cls)
        if error is None:
            continue  # absent: "auto" means the first installed one
        message = _unavailable(cls, repr(candidate))
        if not allow_fallback:
            raise FallbackRefusedError(
                f"{message} allow_fallback=False refuses another backend."
            ) from error
        logger.warning("select_backend: %s Trying the next one.", message)
        reason = reason or f"{candidate}_broken"
    raise RuntimeError(
        "No vector backend is available. Semantic search requires numpy; "
        "install it with `pip install scikit-plots[corpus]`."
    )


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
    allow_fallback: bool = True,
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
    allow_fallback : bool, optional
        Under ``"auto"``, whether a backend that is installed but cannot be
        loaded is skipped (``True``, default; the result reports
        ``fallback_reason="<name>_broken"``) or refused (``False``). Also
        passed to :class:`AnnoyBackend`. An explicitly named backend is never
        replaced, whatever this says.
    **kwargs : Any
        Compatibility form for generic backend constructor keyword arguments.
        Conflicting keys between ``index_kwargs`` and ``kwargs`` raise.

    Returns
    -------
    VectorIndexBackend
        An unbuilt backend instance; :meth:`VectorIndexBackend.provenance`
        records what was requested and why the result differs.

    Raises
    ------
    ValueError
        If *name* is not a known backend or ``"auto"``.
    RuntimeError
        If an explicitly named backend is unavailable (the message says
        whether it is absent or installed but failing to load), or if
        ``"auto"`` finds no available backend (numpy missing).
    FallbackRefusedError
        If ``"auto"`` meets an installed backend that cannot load and
        ``allow_fallback`` is ``False`` (a :class:`RuntimeError`).
    """
    if not isinstance(allow_fallback, bool):
        raise TypeError(
            f"allow_fallback must be a bool, got {type(allow_fallback).__name__}"
        )
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
        if issubclass(cls, AnnoyBackend):
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
            # Same rule as the legacy fields: a non-default selector value
            # and a different index_kwargs value conflict; otherwise the
            # index_kwargs value (when given) wins.
            if (
                "allow_fallback" in options
                and allow_fallback is False
                and options["allow_fallback"] is not False
            ):
                raise ValueError(
                    "conflicting Annoy configuration: allow_fallback=False but "
                    f"index_kwargs['allow_fallback']={options['allow_fallback']!r}"
                )
            options.setdefault("allow_fallback", allow_fallback)
        try:
            return cls(**options)
        except TypeError as exc:
            raise TypeError(
                f"invalid index_kwargs for backend {resolved_name!r}: {exc}"
            ) from exc

    def _selected(
        backend: VectorIndexBackend, requested: str, reason: str | None
    ) -> VectorIndexBackend:
        # Provenance (R2-5) is recorded on the instance the caller receives.
        backend._requested_backend = requested
        backend._selection_reason = reason
        if not isinstance(backend, AnnoyBackend):
            # An AnnoyBackend holds its own switch (it also governs its binding).
            backend._allow_fallback = allow_fallback
        return backend

    if isinstance(name, type):
        if not issubclass(name, VectorIndexBackend):
            raise TypeError(
                f"backend classes must subclass VectorIndexBackend; got {name!r}"
            )
        label = f"{name.__module__}.{name.__qualname__}"
        if not name.is_available():
            raise RuntimeError(_unavailable(name, f"class {label}"))
        return _selected(_make(name, getattr(name, "name", name.__name__)), label, None)

    if not isinstance(name, str):
        raise TypeError("backend must be a backend name or VectorIndexBackend subclass")
    key = (name or "auto").strip().lower()

    if key == "auto":
        candidate, reason = _first_usable_backend(allow_fallback=allow_fallback)
        return _selected(_make(_BACKENDS[candidate], candidate), "auto", reason)

    key = canonical_backend_name(key)
    if key not in _BACKENDS:
        valid = sorted(set(_BACKENDS) | set(_BACKEND_ALIASES) | {"auto"})
        raise ValueError(f"unknown backend {name!r}; choose from {valid}")

    cls = _BACKENDS[key]
    if not cls.is_available():
        raise RuntimeError(
            f"{_unavailable(cls, repr(key))} Install the corresponding package "
            "or use backend='auto'."
        )
    return _selected(_make(cls, key), key, None)
