# scikitplot/annoy/_mixins/_vectors.py
"""
Vector (neighbor) utilities for Annoy-style indexes.

This module provides a **thin, deterministic** Python layer on top of the
low-level Annoy API exposed by the C-extension. It is implemented as a mixin so
it can be composed into multiple high-level index classes without duplicating
logic.

Design goals (non-negotiable)
-----------------------------
* **No C-API changes** required.
* **Deterministic semantics** (no randomness, no numeric tolerance heuristics).
* **Loyal behavior**: we forward to the underlying Annoy query methods and only
  apply explicit, documented post-processing (e.g., excluding a self id).
* **User-friendly surface**: consistent ``include_self`` handling for both
  by-item and by-vector queries, optional NumPy output, and clear errors.

Required low-level methods
--------------------------
A class mixing this in MUST provide these methods (from the C-API wrapper):

* ``get_nns_by_item(item: int, n: int, search_k: int = -1,
  include_distances: bool = False)``
* ``get_nns_by_vector(vector: Sequence[float], n: int, search_k: int = -1,
  include_distances: bool = False)``
* ``get_item_vector(item: int) -> Sequence[float]``

Notes
-----
Annoy query order can be implementation-dependent when there are exact ties.
This mixin does not change the underlying ordering; it only filters results
deterministically based on explicit rules.

See Also
--------
scikitplot.cexternals._annoy.annoylib.Annoy
    Low-level C-extension wrapper providing the core neighbor queries.
scikitplot.cexternals._annoy._plotting
    Utilities that can visualize neighbor structure (kNN edges) for debugging.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator, List, Sequence, Tuple, Union

try:  # optional dependency
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


__all__ = [
    "Dists",
    "Ids",
    "NeighborIdsReturn",
    "NeighborVectorsMatrix",
    "NeighborVectorsReturn",
    "Vector",
    "VectorOpsMixin",
]

Ids = List[int]
Dists = List[float]
Vector = Sequence[float]

NeighborIdsReturn = Union[Ids, Tuple[Ids, Dists]]
NeighborVectorsMatrix = Union[List[Sequence[float]], "np.ndarray"]
NeighborVectorsReturn = Union[
    NeighborVectorsMatrix, Tuple[NeighborVectorsMatrix, Dists]
]


class VectorOpsMixin:
    """
    High-level vector operations for Annoy-like objects.

    This mixin provides a consistent, user-friendly API around the core Annoy
    neighbor query primitives.

    Notes
    -----
    * All methods here are deterministic wrappers around the low-level API.
    * ``include_self`` is implemented in Python by filtering ids.
    * For by-vector queries, "self" is defined as a *stored vector that is
      strictly element-wise equal* to the query vector (no tolerance).

    See Also
    --------
    scikitplot.cexternals._annoy.annoylib.Annoy.get_nns_by_item
    scikitplot.cexternals._annoy.annoylib.Annoy.get_nns_by_vector
    scikitplot.cexternals._annoy.annoylib.Annoy.get_item_vector
        The required low-level methods provided by the Annoy C-extension wrapper.
    """

    # This mixin supports both inheritance-style (Index subclasses Annoy)
    # and composition-style (Index wraps a low-level Annoy instance).
    #
    # IMPORTANT: the methods in this mixin call ``self.get_nns_*`` and
    # ``self.get_item_vector``. For composition-style wrappers, provide those
    # methods as explicit proxies that forward to ``self._annoy``.
    #
    # We keep this calling convention intentionally to remain loyal to existing
    # high-level overrides (e.g., validation/caching) while still enabling
    # composition when proxies are defined.

    def _low_level(self) -> Any:
        """
        Return the low-level Annoy object.

        Preference order is explicit and deterministic:

        1) ``self._annoy`` when present (composition style)
        2) ``self`` (inheritance style)
        """
        ll = getattr(self, "_annoy", None)
        return ll if ll is not None else self

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _filter_ids(
        self,
        ids: Ids,
        dists: Dists | None = None,
        *,
        exclude_ids: set[int] | None = None,
    ) -> tuple[Ids, Dists | None]:
        """Filter ``ids`` (and optionally ``dists``) by a set of excluded ids."""
        if not exclude_ids:
            return ids, dists

        if dists is None:
            return [i for i in ids if i not in exclude_ids], None

        filtered_ids: Ids = []
        filtered_dists: Dists = []
        for i, d in zip(ids, dists):
            if i in exclude_ids:
                continue
            filtered_ids.append(i)
            filtered_dists.append(d)

        return filtered_ids, filtered_dists

    def _require_numpy(self) -> Any:
        """
        Return the imported NumPy module or raise if unavailable.

        This project treats NumPy as an optional dependency at import time.
        Methods that expose ``as_numpy=True`` must fail with a clear, deterministic
        error if NumPy is not installed.

        Returns
        -------
        numpy : module
            The imported NumPy module.

        Raises
        ------
        ImportError
            If NumPy is not available in the current environment.
        """
        if np is None:  # pragma: no cover
            raise ImportError("NumPy is required when as_numpy=True")
        return np

    def _vectors_equal_strict(self, a: Vector, b: Vector) -> bool:
        """Strict element-wise equality (no tolerance, no coercion)."""
        if len(a) != len(b):
            return False
        # for x, y in zip(a, b):
        #     if x != y:
        #         return False
        # return True
        return all(x == y for x, y in zip(a, b))

    def _find_first_exact_match_id(self, vector: Vector, ids: Ids) -> int | None:
        """
        Return the first id whose stored vector equals ``vector`` strictly.

        Notes
        -----
        This intentionally does **not** swallow errors from ``get_item_vector``.
        If the low-level index cannot return a stored vector for an id that it
        itself produced, that is a correctness issue and should surface to the
        caller.
        """
        for i in ids:
            cand = self.get_item_vector(int(i))
            if self._vectors_equal_strict(vector, cand):
                return int(i)
        return None

    # ---------------------------------------------------------------------
    # By-item strict neighbor IDs
    # ---------------------------------------------------------------------
    def _neighbor_ids_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: bool = False,
    ) -> NeighborIdsReturn:
        if n <= 0:
            raise ValueError("n must be a positive integer")

        # Fast path: no filtering.
        if include_self:
            return self.get_nns_by_item(
                item,
                n,
                search_k=search_k,
                include_distances=include_distances,
            )

        exclude = {int(item)}

        # Content-aware strict path:
        # - request n first
        # - only request n+1 if the self id appears (to preserve 'n' outputs)
        if include_distances:
            ids, dists = self.get_nns_by_item(
                item, n, search_k=search_k, include_distances=True
            )
            if item not in ids:
                return ids[:n], dists[:n]

            ids2, dists2 = self.get_nns_by_item(
                item, n + 1, search_k=search_k, include_distances=True
            )
            ids_f, dists_f = self._filter_ids(ids2, dists2, exclude_ids=exclude)
            return ids_f[:n], (dists_f or [])[:n]

        ids = self.get_nns_by_item(item, n, search_k=search_k, include_distances=False)
        if item not in ids:
            return ids[:n]

        ids2 = self.get_nns_by_item(
            item, n + 1, search_k=search_k, include_distances=False
        )
        ids_f, _ = self._filter_ids(ids2, None, exclude_ids=exclude)
        return ids_f[:n]

    # ---------------------------------------------------------------------
    # Public by-item API
    # ---------------------------------------------------------------------
    def get_neighbor_ids_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: bool = False,
    ) -> NeighborIdsReturn:
        """
        Return neighbor ids for a stored item id.

        Parameters
        ----------
        item : int
            Stored item id.
        n : int
            Number of neighbors to return (after applying ``include_self``).
        search_k : int, default=-1
            Forwarded to the underlying Annoy query.
        include_self : bool, default=False
            If False (default), filter ``item`` out if it appears in the result.
        include_distances : bool, default=False
            If True, return ``(ids, distances)``.

        Returns
        -------
        ids : list[int] or (list[int], list[float])
            Neighbor ids, optionally with distances.

        Notes
        -----
        This method is deterministic given the underlying Annoy index.

        See Also
        --------
        get_neighbor_ids_by_vector
            Same semantics for by-vector queries.
        scikitplot.cexternals._annoy.annoylib.Annoy.get_nns_by_item
            Low-level Annoy query primitive.
        """
        return self._neighbor_ids_by_item(
            item,
            n,
            search_k=search_k,
            include_self=include_self,
            include_distances=include_distances,
        )

    def get_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: bool = False,
        as_numpy: bool = False,
        dtype: str = "float32",
    ) -> NeighborVectorsReturn:
        """
        Return neighbor vectors for a stored item id.

        Parameters
        ----------
        item : int
            Stored item id.
        n : int
            Number of neighbors to return (after filtering).
        search_k : int, default=-1
            Forwarded to the underlying Annoy query.
        include_self : bool, default=False
            If False (default), filter ``item`` out if it appears in the result.
        include_distances : bool, default=False
            If True, return ``(vectors, distances)``.
        as_numpy : bool, default=False
            If True, return a NumPy array. Requires NumPy to be installed.
        dtype : str, default="float32"
            NumPy dtype used when ``as_numpy=True``.

        Returns
        -------
        vectors : list[Sequence[float]] or numpy.ndarray
            Matrix of vectors (row-major).
        (vectors, distances) : tuple
            When ``include_distances=True``, returns ``(vectors, distances)``.

        Raises
        ------
        ImportError
            If ``as_numpy=True`` but NumPy is not installed.

        See Also
        --------
        iter_neighbor_vectors_by_item
            Streaming interface for neighbor vectors.
        scikitplot.cexternals._annoy.annoylib.Annoy.get_item_vector
            Low-level vector access primitive.
        """
        np_mod = self._require_numpy() if as_numpy else None

        if include_distances:
            ids, dists = self._neighbor_ids_by_item(
                item,
                n,
                search_k=search_k,
                include_self=include_self,
                include_distances=True,
            )
        else:
            ids = self._neighbor_ids_by_item(
                item,
                n,
                search_k=search_k,
                include_self=include_self,
                include_distances=False,
            )
            dists = None

        vectors = [self.get_item_vector(int(i)) for i in ids]
        mat = np_mod.asarray(vectors, dtype=dtype) if as_numpy else vectors  # type: ignore[union-attr]

        return (mat, dists) if include_distances else mat

    def iter_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
    ) -> Iterator[Sequence[float]]:
        """
        Yield neighbor vectors for a stored item id (streaming).

        Notes
        -----
        This avoids materializing all vectors at once.
        """
        ids = self._neighbor_ids_by_item(
            item,
            n,
            search_k=search_k,
            include_self=include_self,
            include_distances=False,
        )
        for i in ids:
            yield self.get_item_vector(int(i))

    # ---------------------------------------------------------------------
    # By-vector strict neighbor IDs
    # ---------------------------------------------------------------------
    def _neighbor_ids_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
    ) -> NeighborIdsReturn:
        """
        Strict by-vector semantics (symmetric with by-item).

        Rules (deterministic)
        ---------------------
        * ``exclude_item_ids`` are always excluded.
        * If ``exclude_item`` is provided and ``include_self=False``, that id is
          excluded.
        * If ``include_self=False`` and ``exclude_item is None``, we perform an
          **exact-match self detection**:

          - Query Annoy for neighbors.
          - Find the first returned id whose stored vector is strictly equal to
            the query vector (element-wise equality).
          - Exclude that id.

        This avoids metric-specific assumptions (e.g., "distance == 0") and does
        not use any float tolerance heuristics.

        Notes
        -----
        If your index contains duplicate identical vectors, "self" is ambiguous.
        In that case, this method excludes **the first exact match in Annoy's
        returned order** (deterministic).

        See Also
        --------
        get_neighbor_ids_by_item
            Symmetric behavior for by-item queries.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")

        exclude_set: set[int] = {int(x) for x in (exclude_item_ids or [])}

        # Explicit self-id support
        if exclude_item is not None and not include_self:
            exclude_set.add(int(exclude_item))

        want_auto_self = (not include_self) and (exclude_item is None)

        # 1) Probe
        if include_distances:
            ids, dists = self.get_nns_by_vector(
                vector, n, search_k=search_k, include_distances=True
            )
        else:
            ids = self.get_nns_by_vector(
                vector, n, search_k=search_k, include_distances=False
            )
            dists = None

        # 2) Deterministic auto-self detection by strict vector equality
        if want_auto_self and ids:
            match = self._find_first_exact_match_id(vector, ids)
            if match is not None:
                exclude_set.add(int(match))

        # 3) If nothing to exclude, return directly
        if not exclude_set:
            return (ids[:n], dists[:n]) if include_distances else ids[:n]

        hits = sum(1 for i in ids if int(i) in exclude_set)
        if hits == 0:
            return (ids[:n], dists[:n]) if include_distances else ids[:n]

        # 4) Single deterministic retry to fill n after exclusions
        # Request enough slack to compensate for all excluded ids that may appear.
        n2 = n + len(exclude_set)
        if include_distances:
            ids2, dists2 = self.get_nns_by_vector(
                vector, n2, search_k=search_k, include_distances=True
            )
            ids_f, dists_f = self._filter_ids(ids2, dists2, exclude_ids=exclude_set)
            return ids_f[:n], (dists_f or [])[:n]
        ids2 = self.get_nns_by_vector(
            vector, n2, search_k=search_k, include_distances=False
        )
        ids_f, _ = self._filter_ids(ids2, None, exclude_ids=exclude_set)
        return ids_f[:n]

    # ---------------------------------------------------------------------
    # Public by-vector API
    # ---------------------------------------------------------------------
    def get_neighbor_ids_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
    ) -> NeighborIdsReturn:
        """
        Return neighbor ids for a query vector.

        Parameters
        ----------
        vector : Sequence[float]
            Query vector (same dimension as the index).
        n : int
            Number of neighbors to return (after filtering).
        search_k : int, default=-1
            Forwarded to the underlying Annoy query.
        include_distances : bool, default=False
            If True, return ``(ids, distances)``.
        include_self : bool, default=False
            If False (default), attempt to exclude an exact-match stored vector.
        exclude_item : int, optional
            Explicit stored id to exclude when ``include_self=False``.
            Use this when you know the corresponding id of the query vector.
        exclude_item_ids : Iterable[int], optional
            Additional ids to exclude.

        Returns
        -------
        ids : list[int] or (list[int], list[float])
            Neighbor ids, optionally with distances.

        See Also
        --------
        get_neighbor_vectors_by_vector
            Convenience method returning vectors instead of ids.
        scikitplot.cexternals._annoy.annoylib.Annoy.get_nns_by_vector
            Low-level Annoy query primitive.
        """
        return self._neighbor_ids_by_vector(
            vector,
            n,
            search_k=search_k,
            include_distances=include_distances,
            include_self=include_self,
            exclude_item=exclude_item,
            exclude_item_ids=exclude_item_ids,
        )

    def get_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
        as_numpy: bool = False,
        dtype: str = "float32",
    ) -> NeighborVectorsReturn:
        """
        Return neighbor vectors for a query vector.

        Parameters
        ----------
        vector : Sequence[float]
            Query vector.
        n : int
            Number of neighbors to return (after filtering).
        search_k : int, default=-1
            Forwarded to the underlying Annoy query.
        include_distances : bool, default=False
            If True, return ``(vectors, distances)``.
        include_self : bool, default=False
            If False (default), attempt to exclude an exact-match stored vector.
        exclude_item : int, optional
            Explicit stored id to exclude when ``include_self=False``.
        exclude_item_ids : Iterable[int], optional
            Additional ids to exclude.
        as_numpy : bool, default=False
            If True, return a NumPy array. Requires NumPy.
        dtype : str, default="float32"
            NumPy dtype used when ``as_numpy=True``.

        Raises
        ------
        ImportError
            If ``as_numpy=True`` but NumPy is not installed.

        See Also
        --------
        get_neighbor_ids_by_vector
            Same query returning ids.
        iter_neighbor_vectors_by_vector
            Streaming interface.
        """
        np_mod = self._require_numpy() if as_numpy else None

        if include_distances:
            ids, dists = self._neighbor_ids_by_vector(
                vector,
                n,
                search_k=search_k,
                include_distances=True,
                include_self=include_self,
                exclude_item=exclude_item,
                exclude_item_ids=exclude_item_ids,
            )
        else:
            ids = self._neighbor_ids_by_vector(
                vector,
                n,
                search_k=search_k,
                include_distances=False,
                include_self=include_self,
                exclude_item=exclude_item,
                exclude_item_ids=exclude_item_ids,
            )
            dists = None

        vectors = [self.get_item_vector(int(i)) for i in ids]
        mat = np_mod.asarray(vectors, dtype=dtype) if as_numpy else vectors  # type: ignore[union-attr]
        return (mat, dists) if include_distances else mat

    def iter_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        exclude_item: int | None = None,
        exclude_item_ids: Iterable[int] | None = None,
    ) -> Iterator[Sequence[float]]:
        """Yield neighbor vectors for a query vector (streaming)."""
        ids = self._neighbor_ids_by_vector(
            vector,
            n,
            search_k=search_k,
            include_distances=False,
            include_self=include_self,
            exclude_item=exclude_item,
            exclude_item_ids=exclude_item_ids,
        )
        for i in ids:
            yield self.get_item_vector(int(i))
