# test_vectors.py
# Pytest tests for VectorOpsMixin
#
# These tests are pure-Python and do NOT require the Annoy C-extension.
# They validate strict, content-aware behavior for:
# - by-item neighbor id selection with include_self handling
# - by-vector neighbor id selection with explicit exclusion
# - neighbor vector matrix construction (list + optional numpy)
#
# Expected module path:
#   scikitplot.annoy._vectors import VectorOpsMixin
#
# If your package layout differs, adjust the import accordingly.

from __future__ import annotations

import pytest

try:
    from .._mixins._vectors import VectorOpsMixin
except Exception as e:
    # Fail fast with a clear message.
    raise ImportError(
        "Cannot import VectorOpsMixin. "
        "Expected path: scikitplot.annoy._mixins._vectors"
    ) from e


class DummyIndex(VectorOpsMixin):
    """
    A deterministic fake index that records calls to
    get_nns_by_item and get_nns_by_vector.

    We control returned ids/dists by (kind, n, include_distances) keys.
    """

    def __init__(self, item_map=None, vector_map=None, vectors=None, f=3, metric="angular"):
        self._item_map = item_map or {}
        self._vector_map = vector_map or {}
        self._vectors = vectors or {}
        self._calls = []
        self.f = f
        self.metric = metric

    @property
    def calls(self):
        return list(self._calls)

    def get_item_vector(self, item: int):
        if item not in self._vectors:
            raise KeyError(f"Vector for item {item} not provided in test dummy.")
        return self._vectors[item]

    def get_nns_by_item(self, item, n, *, search_k=-1, include_distances=False):
        self._calls.append(("item", int(item), int(n), int(search_k), bool(include_distances)))

        key = (int(item), int(n), bool(include_distances))
        if key not in self._item_map:
            raise KeyError(f"Dummy item_map missing key {key}")

        value = self._item_map[key]
        return value

    def get_nns_by_vector(self, vector, n, *, search_k=-1, include_distances=False):
        # Vector itself is not hashed as key; tests use a name token.
        token = vector  # tests will pass a simple string token
        self._calls.append(("vector", token, int(n), int(search_k), bool(include_distances)))

        key = (token, int(n), bool(include_distances))
        if key not in self._vector_map:
            raise KeyError(f"Dummy vector_map missing key {key}")

        value = self._vector_map[key]
        return value
