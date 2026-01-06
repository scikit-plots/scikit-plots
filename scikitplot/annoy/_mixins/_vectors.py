# scikitplot/annoy/_mixins/_vectors.py
"""
Vector neighbor utilities for Annoy-style indexes.

This module defines :class:`~scikitplot.annoy._mixins._vectors.VectorOpsMixin`, a
**deterministic**, **explicit** mixin providing user-facing neighbor queries.

Required backend surface:

A backend returned by :func:`~scikitplot.annoy._utils.backend_for` is expected to
implement:

- ``get_nns_by_item(item: int, n: int, search_k: int = -1, include_distances: bool = False)``
- ``get_nns_by_vector(vector: Sequence[float], n: int, search_k: int = -1, include_distances: bool = False)``

- ``get_item_vector(item: int) -> Sequence[float]`` (only for ``*_vectors`` helpers)

Optional backend surface:

- ``get_n_trees() -> int`` (built-state detection)
- ``get_n_items() -> int`` (graph sizing / defensive checks)
- attribute/property ``f`` (dimension)

Notes
-----
All validated query inputs are converted to contiguous ``float32`` arrays before
dispatching to the backend. Returned distances are exposed as ``float32`` arrays.

See Also
--------
scikitplot.annoy._mixins._ndarray.NDArrayMixin
    Batch add / export utilities.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import numpy as np
from sklearn.exceptions import NotFittedError

# from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (  # noqa: F401
    FLOAT_DTYPES,
    check_array,
    validate_data,
)

from .._utils import backend_for, lock_for

__all__: tuple[str, ...] = ("VectorOpsMixin",)


# ------------------------------------------------------------------
# Small, deterministic validators (no implicit behavior changes).
# ------------------------------------------------------------------
def _as_positive_int(name: str, value: Any) -> int:
    """Cast ``value`` to ``int`` and ensure it is strictly positive."""
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer; got {ivalue}.")
    return ivalue


def _as_int(name: str, value: Any) -> int:
    """Cast ``value`` to ``int`` (for explicit public parameters)."""
    try:
        return int(value)
    except Exception as e:
        raise TypeError(f"{name} must be an int; got {type(value)!r}.") from e


def _normalize_exclude_ids(exclude_item_ids: Iterable[int] | None) -> frozenset[int]:
    """Normalize excluded ids to a ``frozenset[int]`` for stable membership tests."""
    if exclude_item_ids is None:
        return frozenset()
    try:
        return frozenset(int(i) for i in exclude_item_ids)
    except Exception as e:
        raise TypeError("exclude_item_ids must be an iterable of ints or None.") from e


def _raise_if_not_built(backend: Any) -> None:
    """Raise ``NotFittedError`` if the backend can report that it is unbuilt."""
    get_n_trees = getattr(backend, "get_n_trees", None)
    if callable(get_n_trees):
        try:
            n_trees = int(get_n_trees())
        except Exception:
            return
        if n_trees <= 0:
            raise NotFittedError(
                "This Annoy index does not appear to be built. Call build(...) before querying."
            )


def _validate_query_matrix(
    est: Any,
    X: Any,
    *,
    ensure_all_finite: bool | Literal["allow-nan"],
    copy: bool,
) -> np.ndarray:
    """
    Validate a query matrix with scikit-learn utilities.

    This is deterministic and does not infer or impute values. It either accepts
    the input (after dtype/shape validation) or raises an error.

    Returns
    -------
    Xv : numpy.ndarray
        Contiguous ``float32`` array of shape ``(n_samples, n_features)``.
    """
    # Prefer validate_data when available to respect scikit-learn estimator
    # conventions (e.g., n_features_in_). Fall back to check_array otherwise.
    try:
        Xv = validate_data(  # noqa: N806
            est,
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            ensure_all_finite=ensure_all_finite,
            copy=copy,
            reset=False,
        )
    except Exception:
        Xv = check_array(  # noqa: N806
            X,
            accept_sparse=False,
            dtype=FLOAT_DTYPES,
            ensure_all_finite=ensure_all_finite,
            copy=copy,
        )
    # Annoy backends typically expect contiguous float arrays.
    return np.ascontiguousarray(np.asarray(Xv), dtype=np.float32)


def _vectors_validate_1d(
    est: Any,
    x: Any,
    *,
    ensure_all_finite: bool | Literal["allow-nan"],
    copy: bool,
) -> np.ndarray:
    """
    Validate a single query vector as a 1D ``float32`` array.

    Accepts ``(f,)`` or ``(1, f)`` and rejects anything else deterministically.

    Returns
    -------
    v : numpy.ndarray of shape (f,)
        Contiguous ``float32`` query vector.
    """
    arr = np.asarray(x, dtype=np.float32)

    if arr.ndim == 2:  # noqa: PLR2004
        if int(arr.shape[0]) != 1:
            raise ValueError(
                "vector must be 1D of shape (f,) or 2D with a single row (1, f); "
                f"got shape={arr.shape}."
            )
        arr = arr.reshape(-1)
    elif arr.ndim != 1:
        raise ValueError(f"vector must be 1D of shape (f,); got ndim={int(arr.ndim)}.")

    Xv = _validate_query_matrix(  # noqa: N806
        est,
        arr.reshape(1, -1),
        ensure_all_finite=ensure_all_finite,
        copy=copy,
    )
    v = Xv.ravel()

    f = int(getattr(est, "f", 0) or 0)
    if f > 0 and int(v.shape[0]) != f:
        raise ValueError(
            f"Query vector has {int(v.shape[0])} features, but index dimension f={f}."
        )
    return v


def _vectors_validate_2d(
    est: Any,
    X: Any,
    *,
    ensure_all_finite: bool | Literal["allow-nan"],
    copy: bool,
) -> np.ndarray:
    """
    Validate multiple query vectors as a 2D ``float32`` array.

    Returns
    -------
    Xv : numpy.ndarray of shape (n_queries, f)
        Contiguous ``float32`` query matrix.
    """
    Xv = _validate_query_matrix(  # noqa: N806
        est,
        X,
        ensure_all_finite=ensure_all_finite,
        copy=copy,
    )
    if Xv.ndim != 2:  # noqa: PLR2004
        raise ValueError(
            f"X must be 2D of shape (n_queries, n_features); got ndim={Xv.ndim}."
        )
    f = int(getattr(est, "f", 0) or 0)
    if f > 0 and int(Xv.shape[1]) != f:
        raise ValueError(
            f"X has {int(Xv.shape[1])} features, but index dimension f={f}."
        )
    return Xv


def _filter_and_slice_neighbors(
    idx: Sequence[int] | np.ndarray,
    dists: Sequence[float] | np.ndarray,
    *,
    n_neighbors: int,
    exclude_ids: frozenset[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter neighbors by id, preserving order, and slice to ``n_neighbors``.

    Parameters
    ----------
    idx, dists
        Sequences returned by the backend (must have equal length).
    n_neighbors
        Number of results required after filtering.
    exclude_ids
        Ids to exclude.

    Returns
    -------
    idx_out, dists_out
        Filtered arrays (both sliced to ``n_neighbors``).

    Raises
    ------
    ValueError
        If fewer than ``n_neighbors`` results remain after filtering.
    """
    idx_arr = np.asarray(idx, dtype=np.intp)
    dists_arr = np.asarray(dists, dtype=np.float32)

    if idx_arr.shape[0] != dists_arr.shape[0]:
        raise RuntimeError(
            "Backend returned indices and distances with different lengths: "
            f"len(idx)={idx_arr.shape[0]} len(dists)={dists_arr.shape[0]}."
        )

    if exclude_ids:
        keep_mask = np.fromiter(
            (int(i) not in exclude_ids for i in idx_arr), dtype=bool, count=idx_arr.size
        )
        idx_arr = idx_arr[keep_mask]
        dists_arr = dists_arr[keep_mask]

    if idx_arr.shape[0] < n_neighbors:
        raise ValueError(
            "Backend did not return enough neighbors after applying exclusions. "
            f"requested={n_neighbors}, returned={int(idx_arr.shape[0])}."
        )

    return idx_arr[:n_neighbors], dists_arr[:n_neighbors]


class VectorOpsMixin:
    """
    User-facing neighbor queries for Annoy-like backends.

    This mixin exposes explicit per-query helpers (:meth:`query_by_item`,
    :meth:`query_by_vector`) and scikit-learn style batch helpers
    (:meth:`kneighbors`, :meth:`kneighbors_graph`).

    Notes
    -----
    Output ordering for :meth:`kneighbors` is ``(neighbors, distances)`` when
    ``include_distances=True`` (neighbors first). This is *not* the same as
    ``sklearn.neighbors.NearestNeighbors.kneighbors`` (which returns distances
    first). The order is intentional and documented.
    """

    # ------------------------------------------------------------------
    # Public API: explicit queries
    # ------------------------------------------------------------------
    def query_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = False,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Query neighbors by stored item id.

        Parameters
        ----------
        item : int
            Stored item id.
        n_neighbors : int
            Number of neighbors to return *after* applying exclusions.
        search_k : int, default=-1
            Search parameter forwarded to the backend.
        include_distances : bool, default=False
            If True, also return distances.
        exclude_self : bool, default=False
            If True, exclude ``item`` from the returned neighbors.
        exclude_item_ids : iterable of int, optional
            Additional item ids to exclude.
        ensure_all_finite : bool or 'allow-nan', default=True
            Input validation option forwarded to scikit-learn.
        copy : bool, default=False
            Input validation option forwarded to scikit-learn.

        Returns
        -------
        indices : numpy.ndarray of shape (n_neighbors,)
            Neighbor ids.
        (indices, distances) : tuple of numpy.ndarray
            Returned when ``include_distances=True``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the backend reports that the index is unbuilt.
        ValueError
            If ``n_neighbors <= 0`` or not enough neighbors remain after exclusions.

        Notes
        -----
        Exclusions are applied deterministically in the order returned by the backend.

        See Also
        --------
        query_by_vector : Query neighbors by an explicit vector.
        kneighbors : Batch neighbor queries (sklearn-like).
        """
        n_neighbors_i = _as_positive_int("n_neighbors", n_neighbors)
        search_k_i = _as_int("search_k", search_k)

        backend = backend_for(self)
        _raise_if_not_built(backend)

        item_i = _as_int("item", item)
        # vector = _vectors_validate_1d(
        #     self,
        #     vector,
        #     ensure_all_finite=ensure_all_finite,
        #     copy=copy,
        # )

        exclude_ids = _normalize_exclude_ids(exclude_item_ids)
        if exclude_self:
            exclude_ids = exclude_ids | frozenset({item_i})

        # Request enough candidates to account for exclusions.
        n_request = n_neighbors_i + len(exclude_ids)
        get_n_items = getattr(backend, "get_n_items", None)
        if callable(get_n_items):
            try:
                n_items = int(get_n_items())
            except Exception:
                n_items = 0
            if n_items > 0:
                n_request = min(n_request, n_items)

        with lock_for(self):
            try:
                idx, dists = backend.get_nns_by_item(  # type: ignore[attr-defined]
                    item_i,
                    int(n_request),
                    search_k=search_k_i,
                    include_distances=True,
                )
            except Exception as e:  # pragma: no cover
                raise RuntimeError("Backend get_nns_by_item failed.") from e

            if list(exclude_ids):
                idx, dists = _filter_and_slice_neighbors(
                    idx,
                    dists,
                    n_neighbors=n_neighbors_i,
                    exclude_ids=exclude_ids,
                )

        if include_distances:
            return idx, dists
        return idx

    def query_vectors_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = False,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
        dtype: Any = np.float32,
        output_type: Literal["item", "vector"] = "vector",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Query neighbor vectors by stored item id.

        This is a convenience wrapper over :meth:`query_by_item` that materializes
        vectors using the backend's ``get_item_vector``.

        Parameters
        ----------
        item, n_neighbors, search_k, include_distances, exclude_self, exclude_item_ids
            See :meth:`query_by_item`.
        ensure_all_finite, copy
            See :meth:`query_by_vector`.
        dtype : numpy dtype, default=numpy.float32
            Output dtype for the returned vectors.
        output_type : {'item', 'vector'}, default='vector'
            If 'vector', return neighbor vectors. If 'item', return neighbor ids.

        Returns
        -------
        vectors : numpy.ndarray of shape (n_neighbors, f)
            Neighbor vectors.
        (vectors, distances) : tuple
            Returned when ``include_distances=True``.

        See Also
        --------
        query_vectors_by_vector : Vector query returning vectors (or ids).
        """
        backend = backend_for(self)

        with lock_for(self):
            idx, dist = cast(
                tuple[np.ndarray, np.ndarray],
                self.query_by_item(
                    item,
                    n_neighbors,
                    search_k=search_k,
                    include_distances=True,
                    exclude_self=exclude_self,
                    exclude_item_ids=exclude_item_ids,
                    ensure_all_finite=ensure_all_finite,
                    copy=copy,
                ),
            )

        with lock_for(self):
            if output_type == "vector":
                idx = np.asarray(
                    [backend.get_item_vector(int(i)) for i in idx],
                    dtype=dtype,
                )
            else:
                idx = np.asarray(idx, dtype=np.intp)

        if include_distances:
            return idx, dist
        return idx

    def query_by_vector(
        self,
        vector: Any,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = False,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Query neighbors by an explicit vector.

        Parameters
        ----------
        vector : array-like of shape (f,)
            Query vector.
        n_neighbors : int
            Number of neighbors to return after exclusions.
        search_k : int, default=-1
            Search parameter forwarded to the backend.
        include_distances : bool, default=False
            If True, also return distances.
        exclude_self : bool, default=False
            If True, exclude the first returned candidate whose distance
            is exactly ``0.0``. This is intended for queries where ``vector`` comes
            from the index itself.
        exclude_item_ids : iterable of int, optional
            Additional item ids to exclude.
        ensure_all_finite : bool or 'allow-nan', default=True
            Input validation option forwarded to scikit-learn.
        copy : bool, default=False
            Input validation option forwarded to scikit-learn.

        Returns
        -------
        indices : numpy.ndarray of shape (n_neighbors,)
            Neighbor ids.
        (indices, distances) : tuple of numpy.ndarray
            Returned when ``include_distances=True``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the backend reports that the index is unbuilt.
        ValueError
            If ``n_neighbors <= 0``, vector dimension mismatches ``f``, or not
            enough neighbors remain after exclusions.

        Notes
        -----
        Exclusions are applied deterministically in the order returned by the backend.
        If ``exclude_self=True`` and no exact ``0.0`` distance candidate is returned
        in the first position, no additional self-exclusion is applied.

        See Also
        --------
        query_by_item : Query neighbors by stored item id.
        kneighbors : Batch neighbor queries (sklearn-like).
        """
        n_neighbors_i = _as_positive_int("n_neighbors", n_neighbors)
        search_k_i = _as_int("search_k", search_k)

        backend = backend_for(self)
        _raise_if_not_built(backend)

        # vector = _vectors_validate_1d(
        #     self,
        #     vector,
        #     ensure_all_finite=ensure_all_finite,
        #     copy=copy,
        # )

        exclude_ids = _normalize_exclude_ids(exclude_item_ids)

        # Request enough candidates to account for exclusions and possible self-drop.
        n_request = n_neighbors_i + len(exclude_ids) + int(bool(exclude_self))

        with lock_for(self):
            try:
                idx, dists = backend.get_nns_by_vector(  # type: ignore[attr-defined]
                    vector,
                    int(n_request),
                    search_k=search_k_i,
                    include_distances=True,
                )
            except Exception as e:  # pragma: no cover
                raise RuntimeError("Backend get_nns_by_vector failed.") from e

            idx = np.asarray(idx, dtype=np.intp)
            dists = np.asarray(dists, dtype=np.float32)

            # Deterministic self-exclusion rule for vector queries.
            if exclude_self and dists.size and float(dists[0]) == 0.0:
                idx = idx[1:]
                dists = dists[1:]

            if list(exclude_ids):
                idx, dists = _filter_and_slice_neighbors(
                    idx,
                    dists,
                    n_neighbors=n_neighbors_i,
                    exclude_ids=exclude_ids,
                )

        if include_distances:
            return idx, dists
        return idx

    def query_vectors_by_vector(
        self,
        vector: Any,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = False,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
        dtype: Any = np.float32,
        output_type: Literal["item", "vector"] = "vector",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Query neighbor vectors by an explicit vector.

        Convenience wrapper over :meth:`query_by_vector`. By default it returns
        vectors; set ``output_type='item'`` to return neighbor ids instead.

        Parameters
        ----------
        vector, n_neighbors, search_k, include_distances, exclude_self, exclude_item_ids,
            See :meth:`query_by_item`.
        ensure_all_finite, copy
            See :meth:`query_by_vector`.
        dtype : numpy dtype, default=numpy.float32
            Output dtype for the returned vectors.
        output_type : {'item', 'vector'}, default='vector'
            If 'vector', return neighbor vectors. If 'item', return neighbor ids.

        Returns
        -------
        neighbors : numpy.ndarray
            If ``output_type='vector'``, an array of shape ``(n_neighbors, f)``.
            If ``output_type='item'``, an array of shape ``(n_neighbors,)``.
        (neighbors, distances) : tuple
            Returned when ``include_distances=True``.

        See Also
        --------
        query_vectors_by_item : Item id query returning vectors.
        query_by_vector : Per-query id interface.
        """
        backend = backend_for(self)

        with lock_for(self):
            idx, dist = cast(
                tuple[np.ndarray, np.ndarray],
                self.query_by_vector(
                    vector,
                    n_neighbors,
                    search_k=search_k,
                    include_distances=True,
                    exclude_self=exclude_self,
                    exclude_item_ids=exclude_item_ids,
                    ensure_all_finite=ensure_all_finite,
                    copy=copy,
                ),
            )

        with lock_for(self):
            if output_type == "vector":
                if output_type == "vector":
                    idx = np.asarray(
                        [backend.get_item_vector(int(i)) for i in idx],
                        dtype=dtype,
                    )
                else:
                    idx = np.asarray(idx, dtype=np.intp)

        if include_distances:
            return idx, dist
        return idx

    # ------------------------------------------------------------------
    # scikit-learn style APIs (batch)
    # ------------------------------------------------------------------
    def kneighbors(  # noqa: D417
        self,
        X: Any,
        n_neighbors: int = 5,
        *,
        search_k: int = -1,
        include_distances: bool = True,
        exclude_self: bool = False,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
        output_type: Literal["item", "vector"] = "vector",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for one or more query vectors.

        This is a sklearn-like convenience wrapper that returns rectangular arrays.

        Parameters
        ----------
        X : array-like of shape (f,) or (n_queries, f)
            Query vector(s).
        n_neighbors : int, default=5
            Number of neighbors to return per query.
        search_k : int, default=-1
            Search parameter forwarded to the backend.
        include_distances : bool, default=True
            If True, return ``(neighbors, distances)``. Otherwise return neighbors.
        exclude_self : bool, default=False
            If True, apply the same deterministic self-exclusion rule as
            :meth:`query_by_vector` for each query row.
        exclude_item_ids : iterable of int, optional
            Exclude these ids for every query.
        ensure_all_finite : bool or 'allow-nan', default=True
            Input validation option forwarded to scikit-learn.
        copy : bool, default=False
            Input validation option forwarded to scikit-learn.
        output_type : {'item', 'vector'}, default='vector'
            If 'item', return neighbor ids. If 'vector', return neighbor vectors.

        Returns
        -------
        neighbors : numpy.ndarray
            If ``output_type='item'``, shape is ``(n_queries, n_neighbors)``.
            If ``output_type='vector'``, shape is ``(n_queries, n_neighbors, f)``.
        distances : numpy.ndarray of shape (n_queries, n_neighbors)
            Neighbor distances. Returned when ``include_distances=True``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the backend reports that the index is unbuilt.
        ValueError
            If ``n_neighbors <= 0`` or any query yields too few neighbors after exclusions.

        See Also
        --------
        query_by_vector : Per-query 1D interface.
        kneighbors_graph : CSR kNN graph.
        """
        n_neighbors_i = _as_positive_int("n_neighbors", n_neighbors)
        search_k_i = _as_int("search_k", search_k)

        Xv = np.asarray(X)  # noqa: N806
        if Xv.ndim == 1:
            Xv = Xv.reshape(1, -1)  # noqa: N806

        # Xv = _vectors_validate_2d(
        #     self,
        #     Xv,
        #     ensure_all_finite=ensure_all_finite,
        #     copy=copy,
        # )

        distances = np.empty((Xv.shape[0], n_neighbors_i), dtype=np.float32)

        if output_type == "item":
            neighbors: np.ndarray = np.empty(
                (Xv.shape[0], n_neighbors_i), dtype=np.intp
            )
        else:
            neighbors = np.empty(
                (Xv.shape[0], n_neighbors_i, Xv.shape[1]), dtype=np.float32
            )

        for i in range(int(Xv.shape[0])):
            neigh_i, dist_i = cast(
                tuple[np.ndarray, np.ndarray],
                self.query_vectors_by_vector(
                    Xv[i],
                    n_neighbors_i,
                    search_k=search_k_i,
                    include_distances=True,
                    exclude_self=exclude_self,
                    exclude_item_ids=exclude_item_ids,
                    ensure_all_finite=ensure_all_finite,
                    copy=copy,
                    dtype=np.float32,
                    output_type=output_type,
                ),
            )
            neighbors[i] = neigh_i
            distances[i] = dist_i

        if include_distances:
            return neighbors, distances
        return neighbors

    def kneighbors_graph(
        self,
        X: Any,
        n_neighbors: int = 5,
        *,
        search_k: int = -1,
        mode: Literal["connectivity", "distance"] = "connectivity",
        exclude_self: bool = False,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
        output_type: Literal["item", "vector"] = "item",
    ) -> Any:
        """
        Compute the k-neighbors graph (CSR) for query vectors.

        Parameters
        ----------
        X : array-like of shape (f,) or (n_queries, f)
            Query vector(s).
        n_neighbors : int, default=5
            Number of neighbors per query.
        search_k : int, default=-1
            Search parameter forwarded to the backend.
        mode : {'connectivity', 'distance'}, default='connectivity'
            If 'connectivity', graph entries are 1. If 'distance', entries are
            backend distances.
        exclude_self : bool, default=False
            If True, apply the same deterministic self-exclusion rule as
            :meth:`kneighbors` for each query row.
        exclude_item_ids : iterable of int, optional
            Exclude these ids for every query.
        ensure_all_finite : bool or 'allow-nan', default=True
            Input validation option forwarded to scikit-learn.
        copy : bool, default=False
            Input validation option forwarded to scikit-learn.
        output_type : {'item'}, default='item'
            Must be 'item' for CSR construction.

        Returns
        -------
        graph : scipy.sparse.csr_matrix
            CSR matrix of shape ``(n_queries, n_items)``.

        Raises
        ------
        ImportError
            If SciPy is not installed.
        ValueError
            If ``mode`` is invalid or ``output_type != 'item'``.
        RuntimeError
            If the backend returns an out-of-range neighbor id.

        See Also
        --------
        kneighbors : Dense kNN results.
        """
        if mode not in {"connectivity", "distance"}:
            raise ValueError("mode must be 'connectivity' or 'distance'")

        if output_type != "item":
            raise ValueError("kneighbors_graph requires output_type='item'")

        try:
            import scipy.sparse as sp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError("SciPy is required for kneighbors_graph") from e

        backend = backend_for(self)

        indices, distances = cast(
            tuple[np.ndarray, np.ndarray],
            self.kneighbors(
                X,
                n_neighbors=n_neighbors,
                search_k=search_k,
                include_distances=True,
                exclude_self=exclude_self,
                exclude_item_ids=exclude_item_ids,
                ensure_all_finite=ensure_all_finite,
                copy=copy,
                output_type="item",
            ),
        )

        if mode == "distance":
            data = distances.ravel()
        else:
            data = np.ones(indices.size, dtype=np.float32)

        n_queries = int(indices.shape[0])
        max_id = int(indices.max()) if indices.size else -1

        get_n_items = getattr(backend, "get_n_items", None)
        if callable(get_n_items):
            n_items = int(get_n_items())
            if indices.size and max_id >= n_items:
                raise RuntimeError(
                    "Backend returned a neighbor id outside the valid range "
                    f"[0, n_items). max_id={max_id}, n_items={n_items}."
                )
        else:
            n_items = max_id + 1

        rows = np.repeat(
            np.arange(n_queries, dtype=np.intp),
            _as_positive_int("n_neighbors", n_neighbors),
        )
        cols = indices.ravel().astype(np.intp, copy=False)
        return sp.csr_matrix((data, (rows, cols)), shape=(n_queries, n_items))
