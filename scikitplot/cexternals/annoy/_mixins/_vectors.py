# _vectors.py
"""
VectorOpsMixin for scikitplot.cexternals.annoy

High-level vector utilities built strictly on the existing low-level API:
- get_nns_by_item
- get_nns_by_vector
- get_item_vector

Goals:
- No C-API changes.
- Strict, deterministic behavior.
- Content-aware handling of "self" for by-item queries.
- Content-aware, deterministic self-candidate handling for by-vector queries.
- Symmetric include_self parameter across public by-item/by-vector APIs.
- Optional NumPy output without forcing a hard dependency.
"""

from __future__ import annotations

from typing import (
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

try:
    import numpy as np  # optional dependency
except Exception:
    np = None


Ids = List[int]
Dists = List[float]
Vector = Sequence[float]

NeighborIdsReturn = Union[Ids, Tuple[Ids, Dists]]
NeighborVectorsMatrix = Union[List[Sequence[float]], "np.ndarray"]
NeighborVectorsReturn = Union[
    NeighborVectorsMatrix,
    Tuple[NeighborVectorsMatrix, Dists],
]


class VectorOpsMixin:
    """
    High-level vector operations for Annoy-like objects.

    A class mixing this in MUST provide:
    - get_nns_by_item(item: int, n: int, *, search_k: int = -1,
                     include_distances: bool = False)
    - get_nns_by_vector(vector: Sequence[float], n: int, *, search_k: int = -1,
                        include_distances: bool = False)
    - get_item_vector(item: int) -> Sequence[float]

    This mixin does not assume any extra C-API features.
    """

    # -----------------------------
    # Generic filtering
    # -----------------------------
    def _filter_ids(
        self,
        ids: Ids,
        dists: Optional[Dists] = None,
        *,
        exclude_ids: Optional[Set[int]] = None,
    ) -> Tuple[Ids, Optional[Dists]]:
        if not exclude_ids:
            return ids, dists

        if dists is None:
            filtered_ids = [i for i in ids if i not in exclude_ids]
            return filtered_ids, None

        filtered_ids: Ids = []
        filtered_dists: Dists = []
        for i, d in zip(ids, dists):
            if i in exclude_ids:
                continue
            filtered_ids.append(i)
            filtered_dists.append(d)

        return filtered_ids, filtered_dists

    # -----------------------------
    # Strict vector equality
    # -----------------------------
    def _vectors_equal_strict(self, a: Vector, b: Vector) -> bool:
        if len(a) != len(b):
            return False
        # strict element-wise equality (no tolerance)
        for x, y in zip(a, b):
            if x != y:
                return False
        return True

    # -----------------------------
    # By-vector self-candidate detection
    # -----------------------------
    def _detect_self_candidate_id(
        self,
        vector: Vector,
        ids: Ids,
        dists: Optional[Dists] = None,
    ) -> Optional[int]:
        """
        Deterministic "self candidate" detection for by-vector queries.

        Rule:
        - If distances are available, scan ids where distance == 0.0.
          For each, confirm strict vector equality with get_item_vector(id).
        - If distances are not available, only check the first id.

        Returns:
        - candidate item id to exclude
        - or None if no strict match is found
        """
        if not ids:
            return None

        if dists is not None:
            for i, d in zip(ids, dists):
                if d != 0.0:
                    continue
                try:
                    cand_vec = self.get_item_vector(i)
                except Exception:
                    continue
                if self._vectors_equal_strict(vector, cand_vec):
                    return int(i)
            return None

        # no distances: check only the top candidate
        top = ids[0]
        try:
            cand_vec = self.get_item_vector(top)
        except Exception:
            return None
        if self._vectors_equal_strict(vector, cand_vec):
            return int(top)
        return None

    # -----------------------------
    # By-item strict neighbor IDs
    # -----------------------------
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

        if include_self:
            return self.get_nns_by_item(
                item,
                n,
                search_k=search_k,
                include_distances=include_distances,
            )

        exclude = {int(item)}

        # Content-aware strict path:
        # 1) request n
        # 2) only request n+1 if self appeared
        if include_distances:
            ids, dists = self.get_nns_by_item(
                item,
                n,
                search_k=search_k,
                include_distances=True,
            )

            if item not in ids:
                return ids[:n], dists[:n]

            ids2, dists2 = self.get_nns_by_item(
                item,
                n + 1,
                search_k=search_k,
                include_distances=True,
            )
            filtered_ids, filtered_dists = self._filter_ids(
                ids2, dists2, exclude_ids=exclude
            )
            return filtered_ids[:n], (filtered_dists or [])[:n]

        # no distances
        ids = self.get_nns_by_item(
            item,
            n,
            search_k=search_k,
            include_distances=False,
        )

        if item not in ids:
            return ids[:n]

        ids2 = self.get_nns_by_item(
            item,
            n + 1,
            search_k=search_k,
            include_distances=False,
        )
        filtered_ids, _ = self._filter_ids(ids2, None, exclude_ids=exclude)
        return filtered_ids[:n]

    # Public alias (IDs)
    def get_neighbor_ids_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        include_distances: bool = False,
    ) -> NeighborIdsReturn:
        return self._neighbor_ids_by_item(
            item,
            n,
            search_k=search_k,
            include_self=include_self,
            include_distances=include_distances,
        )

    # Public (vectors matrix)
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
        if as_numpy and np is None:
            raise RuntimeError("as_numpy=True requires numpy to be installed")

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

        vectors = [self.get_item_vector(i) for i in ids]
        mat = np.asarray(vectors, dtype=dtype) if as_numpy else vectors

        if include_distances:
            return mat, dists
        return mat

    def iter_neighbor_vectors_by_item(
        self,
        item: int,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
    ):
        ids = self._neighbor_ids_by_item(
            item,
            n,
            search_k=search_k,
            include_self=include_self,
            include_distances=False,
        )
        for i in ids:
            yield self.get_item_vector(i)

    # -----------------------------
    # By-vector strict neighbor IDs
    # -----------------------------
    def _neighbor_ids_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        include_self: bool = False,
        exclude_item: Optional[int] = None,
        exclude_item_ids: Optional[Iterable[int]] = None,
    ) -> NeighborIdsReturn:
        """
        Strict by-vector semantics (symmetric with by-item):

        - include_self=False attempts to drop a "self candidate" even when
        exclude_item is not provided.
        - For by-vector, the strict self-candidate signal is:
            leading neighbors with distance == 0.0
        (deterministic and content-aware; no float tolerance heuristics).
        - exclude_item is treated as an explicit self-id only when include_self=False.
        - exclude_item_ids are always excluded.

        NOTE:
        If your index contains multiple identical vectors, multiple ids may have
        distance==0.0. This rule will exclude the leading zero-distance ids
        deterministically.
        """
        if n <= 0:
            raise ValueError("n must be a positive integer")

        exclude_set: Set[int] = set(int(x) for x in (exclude_item_ids or []))

        # Explicit self-id support
        if exclude_item is not None and not include_self:
            exclude_set.add(int(exclude_item))

        want_auto_self = (not include_self and exclude_item is None)

        # We need distances for strict auto-self detection
        probe_with_distances = want_auto_self or include_distances

        if probe_with_distances:
            ids, dists = self.get_nns_by_vector(
                vector,
                n,
                search_k=search_k,
                include_distances=True,
            )

            # Auto self-detection by zero distance (leading block)
            if want_auto_self and ids and dists:
                for i, d in zip(ids, dists):
                    if d == 0.0:
                        exclude_set.add(int(i))
                    else:
                        break

            # If nothing to exclude, return directly
            hits = sum(1 for i in ids if i in exclude_set)
            if hits == 0:
                return (ids[:n], dists[:n]) if include_distances else ids[:n]

            # Single deterministic retry to fill n after exclusions
            ids2, dists2 = self.get_nns_by_vector(
                vector,
                n + hits,
                search_k=search_k,
                include_distances=True,
            )
            filtered_ids, filtered_dists = self._filter_ids(
                ids2, dists2, exclude_ids=exclude_set
            )

            if include_distances:
                return filtered_ids[:n], (filtered_dists or [])[:n]
            return filtered_ids[:n]

        # No distances needed and no auto-self requested
        ids = self.get_nns_by_vector(
            vector,
            n,
            search_k=search_k,
            include_distances=False,
        )

        if not exclude_set:
            return ids[:n]

        hits = sum(1 for i in ids if i in exclude_set)
        if hits == 0:
            return ids[:n]

        ids2 = self.get_nns_by_vector(
            vector,
            n + hits,
            search_k=search_k,
            include_distances=False,
        )
        filtered_ids, _ = self._filter_ids(ids2, None, exclude_ids=exclude_set)
        return filtered_ids[:n]

    # Public alias (IDs)
    def get_neighbor_ids_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        include_self: bool = False,
        exclude_item: Optional[int] = None,
        exclude_item_ids: Optional[Iterable[int]] = None,
    ) -> NeighborIdsReturn:
        return self._neighbor_ids_by_vector(
            vector,
            n,
            search_k=search_k,
            include_distances=include_distances,
            include_self=include_self,
            exclude_item=exclude_item,
            exclude_item_ids=exclude_item_ids,
        )

    # Public (vectors matrix)
    def get_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_distances: bool = False,
        include_self: bool = False,
        exclude_item: Optional[int] = None,
        exclude_item_ids: Optional[Iterable[int]] = None,
        as_numpy: bool = False,
        dtype: str = "float32",
    ) -> NeighborVectorsReturn:
        if as_numpy and np is None:
            raise RuntimeError("as_numpy=True requires numpy to be installed")

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

        vectors = [self.get_item_vector(i) for i in ids]
        mat = np.asarray(vectors, dtype=dtype) if as_numpy else vectors

        if include_distances:
            return mat, dists
        return mat

    def iter_neighbor_vectors_by_vector(
        self,
        vector: Vector,
        n: int,
        *,
        search_k: int = -1,
        include_self: bool = False,
        exclude_item: Optional[int] = None,
        exclude_item_ids: Optional[Iterable[int]] = None,
    ):
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
            yield self.get_item_vector(i)
