# scikitplot/annoy/_mixins/_vectors.py
"""
Vector neighbor utilities for Annoy-style indexes.

Defines :class:`~scikitplot.annoy._mixins._vectors.VectorOpsMixin`, a
**deterministic**, **explicit** mixin providing user-facing neighbor queries.

Key rules
---------
- No heuristics, no implicit dispatch. Every behavior is controlled by explicit
  parameters.
- Mixin independence: no ``__init__`` and no reliance on other mixins being
  present in the MRO.
- Supports inheritance or composition via :func:`~scikitplot.annoy._utils.backend_for`.

Required backend surface
------------------------
- ``get_nns_by_item(item: int, n: int, search_k: int = -1, include_distances: bool = False)``
- ``get_nns_by_vector(vector: Sequence[float], n: int, search_k: int = -1, include_distances: bool = False)``
- ``get_item_vector(item: int) -> Sequence[float]`` (only for *_vectors helpers)
- Optional: ``get_n_trees() -> int`` (built-state detection)
- Optional: ``get_n_items() -> int`` (graph sizing)
- Optional: attribute/property ``f`` (dimension)

See Also
--------
scikitplot.annoy._mixins._ndarray.NDArrayMixin
    Batch add / export utilities.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

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


def _validate_query_matrix(
    est: Any,
    X: Any,
    *,
    ensure_all_finite: bool | Literal["allow-nan"],
    copy: bool,
) -> np.ndarray:
    """Validate query matrix with scikit-learn utilities (deterministic)."""
    # Check that X and y have correct shape, set n_features_in_, etc.
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
    return np.ascontiguousarray(np.asarray(Xv))


class VectorOpsMixin:
    """User-facing neighbor queries for Annoy-like backends."""

    # ------------------------------------------------------------------
    # State checks
    # ------------------------------------------------------------------
    def _vectors_require_built(self) -> None:
        """Raise ``NotFittedError`` if the backend indicates the index is not built."""
        backend = backend_for(self)
        get_n_trees = getattr(backend, "get_n_trees", None)
        if callable(get_n_trees):
            try:
                if int(get_n_trees()) <= 0:
                    raise NotFittedError(
                        "This Annoy index is not built yet. Call `build(...)` before querying."
                    )
            except NotFittedError:
                raise
            except Exception:
                # If the backend doesn't support the call reliably, do not guess.
                pass

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _vectors_validate_vector(
        self,
        x: Any,
        *,
        ensure_all_finite: bool | Literal["allow-nan"],
        copy: bool,
    ) -> np.ndarray:
        """Validate a single query vector as a 1D float array."""
        X = _validate_query_matrix(
            self,
            np.asarray(x).reshape(1, -1),
            ensure_all_finite=ensure_all_finite,
            copy=copy,
        )
        v = X.ravel()
        f = int(getattr(self, "f", 0) or 0)
        if f > 0 and int(v.shape[0]) != f:
            raise ValueError(
                f"Query vector has {int(v.shape[0])} features, but index dimension f={f}."
            )
        return v

    def _vectors_validate_vectors(
        self,
        X: Any,
        *,
        ensure_all_finite: bool | Literal["allow-nan"],
        copy: bool,
    ) -> np.ndarray:
        """Validate multiple query vectors as a 2D float array."""
        Xv = _validate_query_matrix(  # noqa: N806
            self,
            X,
            ensure_all_finite=ensure_all_finite,
            copy=copy,
        )  # noqa: N806
        if Xv.ndim != 2:  # noqa: PLR2004
            raise ValueError(
                f"X must be 2D of shape (n_queries, n_features); got ndim={Xv.ndim}."
            )
        f = int(getattr(self, "f", 0) or 0)
        if f > 0 and int(Xv.shape[1]) != f:
            raise ValueError(
                f"X has {int(Xv.shape[1])} features, but index dimension f={f}."
            )
        return Xv

    # ------------------------------------------------------------------
    # Core filtering logic (single-pass, deterministic)
    # ------------------------------------------------------------------
    @staticmethod
    def _vectors_filter_topk(
        ids: Any,
        dists: Any | None,
        *,
        k: int,
        exclude: set[int],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Filter ids (and optional distances) and return up to k results."""
        out_ids: list[int] = []
        out_dists: list[float] | None = [] if dists is not None else None

        if dists is not None and len(ids) != len(dists):
            raise RuntimeError(
                "Backend returned ids and distances of different lengths; this indicates a backend error."
            )

        if dists is None:
            for i in ids:
                ii = int(i)
                if ii in exclude:
                    continue
                out_ids.append(ii)
                if len(out_ids) >= k:
                    break
            return np.asarray(out_ids, dtype=np.intp), None

        assert out_dists is not None  # noqa: S101
        for i, d in zip(ids, dists):
            ii = int(i)
            if ii in exclude:
                continue
            out_ids.append(ii)
            out_dists.append(float(d))
            if len(out_ids) >= k:
                break
        return np.asarray(out_ids, dtype=np.intp), np.asarray(
            out_dists, dtype=np.float32
        )

    # ------------------------------------------------------------------

    def _vectors_maybe_exclude_first_zero_distance(
        self,
        ids: Any,
        dists: Any,
        *,
        exclude: set[int],
    ) -> set[int]:
        """
        Exclude the first non-excluded candidate if its distance is exactly 0.

        This implements deterministic ``exclude_self`` behavior for by-vector queries
        without guessing based on vector equality. It relies on the backend's distance
        computation (which may include internal normalization depending on the metric).

        Rules
        -----
        1. The caller requests one extra neighbor.
        2. We scan the returned candidates in order, skipping ids already in ``exclude``.
        3. If the first remaining candidate has distance ``0.0`` (exact), we exclude it.

        Notes
        -----
        - If multiple items are true duplicates under the metric (distance 0), this rule
          will exclude the first such candidate. If you need to exclude a specific id,
          pass it explicitly via ``exclude_item_ids``.
        - This method does not attempt approximate comparisons.
        """
        if dists is None:
            return exclude

        # Defensive: backend should return aligned sequences.
        try:
            it = zip(ids, dists)
        except Exception:
            return exclude

        for cand, dist in it:
            ii = int(cand)
            if ii in exclude:
                continue
            if float(dist) == 0.0:
                new_exclude = set(exclude)
                new_exclude.add(ii)
                return new_exclude
            return exclude
        return exclude

    # ------------------------------------------------------------------
    # Internal: validated by-vector query (no re-validation in loops)
    # ------------------------------------------------------------------
    def _vectors_query_by_vector_validated(
        self,
        v: np.ndarray,
        n_neighbors: int,
        *,
        search_k: int,
        include_distances: bool,
        exclude_self: bool,
        exclude: set[int],
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        backend = backend_for(self)
        exclude_local = exclude
        n_req = int(n_neighbors) + len(exclude_local) + (1 if exclude_self else 0)
        q = v.tolist()  # predictable sequence interface for the C-extension

        # If we need to exclude self, we must inspect distances deterministically.
        need_distances = bool(include_distances) or bool(exclude_self)

        if need_distances:
            ids, dists = backend.get_nns_by_vector(  # type: ignore[attr-defined]
                q,
                n_req,
                search_k=int(search_k),
                include_distances=True,
            )
            if exclude_self:
                exclude_local = self._vectors_maybe_exclude_first_zero_distance(
                    ids, dists, exclude=exclude_local
                )

            idx, dist = self._vectors_filter_topk(
                ids, dists, k=int(n_neighbors), exclude=exclude_local
            )
            if include_distances:
                return idx, (
                    dist if dist is not None else np.empty((0,), dtype=np.float32)
                )
            return idx

        ids = backend.get_nns_by_vector(  # type: ignore[attr-defined]
            q,
            n_req,
            search_k=int(search_k),
            include_distances=False,
        )
        idx, _ = self._vectors_filter_topk(
            ids, None, k=int(n_neighbors), exclude=exclude_local
        )
        return idx

    # ------------------------------------------------------------------
    # Public API: canonical explicit queries
    # ------------------------------------------------------------------
    def query_by_item(  # noqa: D417
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
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
        exclude_self : bool, default=True
            If True (default), apply the same deterministic self-exclusion rule as
            :meth:`kneighbors` for each query row.
        exclude_item_ids : iterable of int, optional
            Additional item ids to exclude.

        Returns
        -------
        indices : numpy.ndarray of shape (n_neighbors,)
            Neighbor ids.
        (indices, distances) : tuple of numpy.ndarray
            Returned when ``include_distances=True``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the index appears to be unbuilt.
        ValueError
            If ``n_neighbors <= 0``.

        Notes
        -----
        This method is deterministic given the underlying backend.

        See Also
        --------
        query_by_vector : Query neighbors by an explicit vector.
        kneighbors : scikit-learn style batch query.
        """
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        self._vectors_require_built()
        if int(item) < 0:
            raise ValueError("item must be a non-negative integer")

        backend = backend_for(self)
        exclude: set[int] = {int(x) for x in (exclude_item_ids or [])}
        if exclude_self:
            exclude.add(int(item))

        n_req = int(n_neighbors) + len(exclude)
        with lock_for(self):
            if include_distances:
                ids, dists = backend.get_nns_by_item(  # type: ignore[attr-defined]
                    int(item),
                    n_req,
                    search_k=int(search_k),
                    include_distances=True,
                )
                idx, dist = self._vectors_filter_topk(
                    ids, dists, k=int(n_neighbors), exclude=exclude
                )
                return idx, (
                    dist if dist is not None else np.empty((0,), dtype=np.float32)
                )

            ids = backend.get_nns_by_item(  # type: ignore[attr-defined]
                int(item),
                n_req,
                search_k=int(search_k),
                include_distances=False,
            )
            idx, _ = self._vectors_filter_topk(
                ids, None, k=int(n_neighbors), exclude=exclude
            )
            return idx

    def query_by_vector(  # noqa: D417
        self,
        vector: Any,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = True,
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
        exclude_self : bool, default=True
            If True (default), exclude the first returned candidate whose distance is exactly ``0.0``.
            This is intended for queries where ``vector`` comes from the index itself.
        exclude_item_ids : iterable of int, optional
            Additional item ids to exclude.

        Returns
        -------
        indices : numpy.ndarray of shape (n_neighbors,)
            Neighbor ids.
        (indices, distances) : tuple of numpy.ndarray
            Returned when ``include_distances=True``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the index appears to be unbuilt.
        ValueError
            If ``n_neighbors <= 0`` or the vector dimension mismatches ``f``.

        Notes
        -----
        This API is **explicit**.

        - For strict exclusion by id, pass ``exclude_item_ids``.
        - If ``exclude_self=True`` and no explicit id is provided, the method performs a
          deterministic, exact check against the *first* returned candidate: if that
          candidate has distance exactly ``0.0``, it is excluded.

        See Also
        --------
        query_by_item : Query neighbors by stored item id.
        kneighbors : scikit-learn style batch query.
        """
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        self._vectors_require_built()

        v = self._vectors_validate_vector(
            vector, ensure_all_finite=ensure_all_finite, copy=copy
        )
        exclude: set[int] = {int(x) for x in (exclude_item_ids or [])}

        with lock_for(self):
            return self._vectors_query_by_vector_validated(
                v,
                int(n_neighbors),
                search_k=int(search_k),
                include_distances=bool(include_distances),
                exclude_self=bool(exclude_self),
                exclude=exclude,
            )

    # ------------------------------------------------------------------
    # Convenience: return vectors instead of ids
    # ------------------------------------------------------------------
    def _vectors_materialize_vectors(
        self, ids: np.ndarray, *, dtype: Any
    ) -> np.ndarray:
        """
        Materialize backend vectors for a 1D array of ids.

        Parameters
        ----------
        ids : numpy.ndarray of shape (n_ids,)
            Item ids to materialize.
        dtype : numpy dtype
            Output dtype.

        Returns
        -------
        vectors : numpy.ndarray of shape (n_ids, f)
            Materialized vectors in the same order as ``ids``.
        """
        backend = backend_for(self)
        with lock_for(self):
            vecs = [backend.get_item_vector(int(i)) for i in ids]  # type: ignore[attr-defined]
        return np.asarray(vecs, dtype=dtype)

    def query_vectors_by_item(
        self,
        item: int,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        dtype: Any = np.float32,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Query neighbor vectors by stored item id.

        This is a convenience wrapper over :meth:`query_by_item` that materializes
        vectors using the backend's ``get_item_vector``.

        Parameters
        ----------
        item, n_neighbors, search_k, include_distances, exclude_self, exclude_item_ids
            See :meth:`query_by_item`.
        dtype : numpy dtype, default=numpy.float32
            Output dtype.

        Returns
        -------
        vectors : numpy.ndarray of shape (n_neighbors, f)
            Neighbor vectors.
        (vectors, distances) : tuple
            Returned when ``include_distances=True``.

        See Also
        --------
        query_vectors_by_vector : Vector query returning vectors.
        """
        backend = backend_for(self)
        if include_distances:
            idx, dist = self.query_by_item(
                item,
                n_neighbors,
                search_k=search_k,
                include_distances=True,
                exclude_self=exclude_self,
                exclude_item_ids=exclude_item_ids,
            )
            vecs = self._vectors_materialize_vectors(idx, dtype=dtype)
            return vecs, dist

        idx = self.query_by_item(
            item,
            n_neighbors,
            search_k=search_k,
            include_distances=False,
            exclude_self=exclude_self,
            exclude_item_ids=exclude_item_ids,
        )
        return self._vectors_materialize_vectors(idx, dtype=dtype)

    def query_vectors_by_vector(  # noqa: D417
        self,
        vector: Any,
        n_neighbors: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
        dtype: Any = np.float32,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Query neighbor vectors by an explicit vector.

        Convenience wrapper over :meth:`query_by_vector`.

        Parameters
        ----------
        vector, n_neighbors, search_k, include_distances, exclude_self, exclude_item_ids,
        ensure_all_finite, copy
            See :meth:`query_by_vector`.
        dtype : numpy dtype, default=numpy.float32
            Output dtype.

        Returns
        -------
        vectors : numpy.ndarray of shape (n_neighbors, f)
            Neighbor vectors.
        (vectors, distances) : tuple
            Returned when ``include_distances=True``.

        See Also
        --------
        query_vectors_by_item : Item id query returning vectors.
        """
        backend = backend_for(self)
        if include_distances:
            idx, dist = self.query_by_vector(
                vector,
                n_neighbors,
                search_k=search_k,
                include_distances=True,
                exclude_self=exclude_self,
                exclude_item_ids=exclude_item_ids,
                ensure_all_finite=ensure_all_finite,
                copy=copy,
            )
            vecs = self._vectors_materialize_vectors(idx, dtype=dtype)
            return vecs, dist

        idx = self.query_by_vector(
            vector,
            n_neighbors,
            search_k=search_k,
            include_distances=False,
            exclude_self=exclude_self,
            exclude_item_ids=exclude_item_ids,
            ensure_all_finite=ensure_all_finite,
            copy=copy,
        )
        return self._vectors_materialize_vectors(idx, dtype=dtype)

    # ------------------------------------------------------------------
    # scikit-learn compatible APIs
    # ------------------------------------------------------------------
    def kneighbors(  # noqa: D417
        self,
        X: Any,
        n_neighbors: int = 5,
        *,
        search_k: int = -1,
        include_distances: bool = True,
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors for one or more query vectors.

        This is a scikit-learn compatible convenience wrapper that returns
        rectangular (2D) arrays.

        Parameters
        ----------
        X : array-like of shape (f,) or (n_queries, f)
            Query vector(s).
        n_neighbors : int, default=5
            Number of neighbors to return per query.
        search_k : int, default=-1
            Search parameter forwarded to the backend.
        include_distances : bool, default=True
            If True, return ``(indices, distances)``. Otherwise return indices.
        exclude_self : bool, default=True
            If True (default), apply the same deterministic self-exclusion rule as
            :meth:`query_by_vector` for each query row.
        exclude_item_ids : iterable of int, optional
            Exclude these ids for every query.

        Returns
        -------
        indices : numpy.ndarray of shape (n_queries, n_neighbors)
            Neighbor ids.
        distances : numpy.ndarray of shape (n_queries, n_neighbors)
            Neighbor distances. Returned when ``include_distances=True``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the index appears to be unbuilt.
        ValueError
            If any query returns fewer than ``n_neighbors`` neighbors.

        See Also
        --------
        query_by_vector : Per-query 1D interface.
        kneighbors_graph : CSR kNN graph.
        """
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be a positive integer")
        self._vectors_require_built()

        Xv = np.asarray(X)  # noqa: N806
        if Xv.ndim == 1:
            Xv = Xv.reshape(1, -1)  # noqa: N806
        Xv = self._vectors_validate_vectors(  # noqa: N806
            Xv, ensure_all_finite=ensure_all_finite, copy=copy
        )  # noqa: N806

        n_queries = int(Xv.shape[0])
        indices = np.empty((n_queries, int(n_neighbors)), dtype=np.intp)
        distances = (
            np.empty((n_queries, int(n_neighbors)), dtype=np.float32)
            if include_distances
            else None
        )

        exclude: set[int] = {int(x) for x in (exclude_item_ids or [])}

        with lock_for(self):
            for i in range(n_queries):
                if include_distances:
                    idx, dist = self._vectors_query_by_vector_validated(
                        Xv[i],
                        int(n_neighbors),
                        search_k=int(search_k),
                        include_distances=True,
                        exclude_self=exclude_self,
                        exclude=exclude,
                    )
                else:
                    idx = self._vectors_query_by_vector_validated(
                        Xv[i],
                        int(n_neighbors),
                        search_k=int(search_k),
                        include_distances=False,
                        exclude_self=exclude_self,
                        exclude=exclude,
                    )
                if int(idx.size) != int(n_neighbors):
                    raise ValueError(
                        f"Backend returned {int(idx.size)} neighbors for query row {i}, expected {int(n_neighbors)}. "
                        "Reduce n_neighbors or add more items to the index."
                    )
                indices[i] = idx
                if distances is not None:
                    if int(dist.size) != int(n_neighbors):
                        raise ValueError(
                            f"Backend returned {int(dist.size)} distances for query row {i}, expected {int(n_neighbors)}."
                        )
                    distances[i] = dist

        if include_distances:
            assert distances is not None  # noqa: S101
            # always use indices, distances
            return indices, distances
        return indices

    def kneighbors_graph(
        self,
        X: Any,
        n_neighbors: int = 5,
        *,
        search_k: int = -1,
        mode: Literal["connectivity", "distance"] = "connectivity",
        exclude_self: bool = True,
        exclude_item_ids: Iterable[int] | None = None,
        ensure_all_finite: bool | Literal["allow-nan"] = True,
        copy: bool = False,
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
        exclude_self : bool, default=True
            If True (default), apply the same deterministic self-exclusion rule as
            :meth:`kneighbors` for each query row.
        exclude_item_ids : iterable of int, optional
            Exclude these ids for every query.
        ensure_all_finite : bool or 'allow-nan', default=True
            Input validation option forwarded to scikit-learn.
        copy : bool, default=False
            Input validation option forwarded to scikit-learn.

        Returns
        -------
        graph : scipy.sparse.csr_matrix
            CSR matrix of shape (n_queries, n_items).

        Raises
        ------
        ImportError
            If SciPy is not installed.
        ValueError
            If ``mode`` is invalid.

        See Also
        --------
        kneighbors : Dense kNN results.
        """
        if mode not in {"connectivity", "distance"}:
            raise ValueError("mode must be 'connectivity' or 'distance'")

        try:
            import scipy.sparse as sp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError("SciPy is required for kneighbors_graph") from e

        backend = backend_for(self)

        if mode == "distance":
            indices, distances = self.kneighbors(
                X,
                n_neighbors=n_neighbors,
                search_k=search_k,
                include_distances=True,
                exclude_self=exclude_self,
                exclude_item_ids=exclude_item_ids,
                ensure_all_finite=ensure_all_finite,
                copy=copy,
            )
            data = distances.ravel()
        else:
            indices = self.kneighbors(
                X,
                n_neighbors=n_neighbors,
                search_k=search_k,
                include_distances=False,
                exclude_self=exclude_self,
                exclude_item_ids=exclude_item_ids,
                ensure_all_finite=ensure_all_finite,
                copy=copy,
            )
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
            # If the backend cannot report n_items, infer the minimal valid column
            # dimension from the returned ids (deterministic).
            n_items = max_id + 1

        rows = np.repeat(np.arange(n_queries, dtype=np.intp), int(n_neighbors))
        cols = indices.ravel().astype(np.intp, copy=False)
        return sp.csr_matrix((data, (rows, cols)), shape=(n_queries, n_items))
