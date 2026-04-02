# # scikitplot/annoy/_annoy.py — deterministic in-memory stub (no C extension needed)
# # Implements exactly the surface used by ANNImputer so tests run without the
# # real Annoy C++ library.
# from __future__ import annotations
# import os
# import pickle
# from pathlib import Path

# import numpy as np


# class Index:
#     """
#     Pure-Python stub that mirrors the AnnoyIndex public API used by ANNImputer.

#     Supports metrics: angular, euclidean, manhattan, dot, hamming.
#     Nearest-neighbour search is exact (brute-force) — acceptable for small
#     unit-test datasets.
#     """

#     # ------------------------------------------------------------------ #
#     _SUPPORTED_METRICS = {"angular", "euclidean", "manhattan", "dot", "hamming"}

#     def __init__(self, f: int, metric: str = "angular"):
#         if metric not in self._SUPPORTED_METRICS:
#             raise ValueError(f"Unsupported metric: {metric!r}")
#         self.f      = f          # number of dimensions
#         self.metric = metric
#         self._items: dict[int, np.ndarray] = {}
#         self._built = False
#         self._seed  = None
#         # on_disk_build path (stored but ignored in stub)
#         self.on_disk_path: str | None = None

#     # ------------------------------------------------------------------ #
#     # Build API
#     # ------------------------------------------------------------------ #
#     def set_seed(self, seed: int) -> None:
#         self._seed = seed

#     def on_disk_build(self, path: str) -> None:
#         """Record the path; stub does nothing else (no actual file streaming)."""
#         self.on_disk_path = str(path)

#     def add_item(self, i: int, vector) -> None:
#         if self._built:
#             raise RuntimeError("Cannot add items after build() has been called.")
#         self._items[i] = np.asarray(vector, dtype=float)

#     def build(self, n_trees: int = 10, n_jobs: int = -1) -> None:
#         self._built = True
#         # If on_disk_build() was called, write index to that path now so that
#         # _store_index (which detects same-path and skips save()) can find it.
#         if self.on_disk_path is not None:
#             self.save(self.on_disk_path)

#     def get_n_items(self) -> int:
#         return len(self._items)

#     # ------------------------------------------------------------------ #
#     # Query API
#     # ------------------------------------------------------------------ #
#     def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
#         if self.metric in {"angular", "cosine"}:
#             na, nb = np.linalg.norm(a), np.linalg.norm(b)
#             if na == 0 or nb == 0:
#                 return 1.0
#             return float(1.0 - np.dot(a, b) / (na * nb))
#         if self.metric in {"euclidean", "l2"}:
#             return float(np.linalg.norm(a - b))
#         if self.metric in {"manhattan", "l1"}:
#             return float(np.sum(np.abs(a - b)))
#         if self.metric == "dot":
#             return float(-np.dot(a, b))
#         if self.metric == "hamming":
#             return float(np.mean(a != b))
#         raise ValueError(self.metric)

#     def get_nns_by_vector(
#         self,
#         vector,
#         n: int,
#         search_k: int = -1,
#         include_distances: bool = False,
#     ):
#         vec = np.asarray(vector, dtype=float)
#         scored = [(self._distance(vec, v), idx) for idx, v in self._items.items()]
#         scored.sort()
#         top = scored[:n]
#         ids   = [t[1] for t in top]
#         dists = [t[0] for t in top]
#         if include_distances:
#             return ids, dists
#         return ids

#     def get_item(self, i: int) -> list:
#         return self._items[i].tolist()

#     # ------------------------------------------------------------------ #
#     # Persistence
#     # ------------------------------------------------------------------ #
#     def save(self, path: str) -> None:
#         Path(path).parent.mkdir(parents=True, exist_ok=True)
#         with open(path, "wb") as fh:
#             pickle.dump({"f": self.f, "metric": self.metric,
#                          "items": self._items, "built": self._built}, fh)

#     def load(self, path: str) -> None:
#         with open(path, "rb") as fh:
#             data = pickle.load(fh)
#         self.f      = data["f"]
#         self.metric = data["metric"]
#         self._items = data["items"]
#         self._built = data["built"]

#     def __len__(self):
#         return len(self._items)
