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
#   scikitplot.cexternals.annoy._vectors import VectorOpsMixin
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
        "Expected path: scikitplot.cexternals.annoy._mixins._vectors"
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


# -----------------------------
# By-item ID behavior
# -----------------------------

def test_by_item_excludes_self_no_retry_when_self_absent():
    item_map = {
        # First call for n=3 returns no self
        (0, 3, False): [1, 2, 3],
    }
    idx = DummyIndex(item_map=item_map)

    ids = idx.get_neighbor_ids_by_item(0, 3, include_self=False)
    assert ids == [1, 2, 3]
    assert idx.calls == [("item", 0, 3, -1, False)]


def test_by_item_excludes_self_triggers_retry():
    item_map = {
        # First call returns self in top-n
        (0, 2, False): [0, 1],
        # Second call with n+1
        (0, 3, False): [0, 1, 2],
    }
    idx = DummyIndex(item_map=item_map)

    ids = idx.get_neighbor_ids_by_item(0, 2, include_self=False)
    assert ids == [1, 2]

    assert idx.calls == [
        ("item", 0, 2, -1, False),
        ("item", 0, 3, -1, False),
    ]


def test_by_item_include_self_true_single_call():
    item_map = {
        (0, 2, False): [0, 1],
    }
    idx = DummyIndex(item_map=item_map)

    ids = idx.get_neighbor_ids_by_item(0, 2, include_self=True)
    assert ids == [0, 1]
    assert idx.calls == [("item", 0, 2, -1, False)]


def test_by_item_excludes_self_with_distances_triggers_retry():
    item_map = {
        (0, 2, True): ([0, 1], [0.0, 0.2]),
        (0, 3, True): ([0, 1, 2], [0.0, 0.2, 0.3]),
    }
    idx = DummyIndex(item_map=item_map)

    ids, dists = idx.get_neighbor_ids_by_item(0, 2, include_self=False, include_distances=True)
    assert ids == [1, 2]
    assert dists == [0.2, 0.3]

    assert idx.calls == [
        ("item", 0, 2, -1, True),
        ("item", 0, 3, -1, True),
    ]


# -----------------------------
# By-vector ID behavior
# -----------------------------

def test_by_vector_no_exclusions_single_call():
    vector_map = {
        ("q", 3, False): [10, 11, 12],
    }
    idx = DummyIndex(vector_map=vector_map)

    ids = idx.get_neighbor_ids_by_vector("q", 3)
    assert ids == [10, 11, 12]
    assert idx.calls == [("vector", "q", 3, -1, False)]


def test_by_vector_exclude_ids_no_retry_when_no_hits():
    vector_map = {
        ("q", 3, False): [10, 11, 12],
    }
    idx = DummyIndex(vector_map=vector_map)

    ids = idx.get_neighbor_ids_by_vector("q", 3, exclude_item_ids=[99, 100])
    assert ids == [10, 11, 12]
    assert idx.calls == [("vector", "q", 3, -1, False)]


def test_by_vector_exclude_ids_triggers_retry_with_hits_count():
    vector_map = {
        # First call returns 2 excluded hits
        ("q", 3, False): [10, 11, 12],
        # Second call requests n + hits = 5
        ("q", 5, False): [10, 11, 12, 13, 14],
    }
    idx = DummyIndex(vector_map=vector_map)

    ids = idx.get_neighbor_ids_by_vector("q", 3, exclude_item_ids=[10, 12])
    assert ids == [11, 13, 14]

    assert idx.calls == [
        ("vector", "q", 3, -1, False),
        ("vector", "q", 5, -1, False),
    ]


def test_by_vector_exclude_ids_with_distances_triggers_retry():
    vector_map = {
        ("q", 2, True): ([10, 11], [0.1, 0.2]),
        # hits=1 if exclude {10}
        ("q", 3, True): ([10, 11, 12], [0.1, 0.2, 0.3]),
    }
    idx = DummyIndex(vector_map=vector_map)

    ids, dists = idx.get_neighbor_ids_by_vector(
        "q", 2, include_distances=True, exclude_item_ids=[10]
    )
    assert ids == [11, 12]
    assert dists == [0.2, 0.3]

    assert idx.calls == [
        ("vector", "q", 2, -1, True),
        ("vector", "q", 3, -1, True),
    ]


# -----------------------------
# Vector matrix construction
# -----------------------------

def test_neighbor_vectors_by_item_list_matrix():
    item_map = {
        (0, 2, False): [0, 1],
        (0, 3, False): [0, 1, 2],
    }
    vectors = {
        1: [1.0, 1.0, 1.0],
        2: [2.0, 2.0, 2.0],
    }
    idx = DummyIndex(item_map=item_map, vectors=vectors)

    mat = idx.get_neighbor_vectors_by_item(0, 2, include_self=False, as_numpy=False)
    assert mat == [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]


def test_neighbor_vectors_by_vector_list_matrix():
    vector_map = {
        ("q", 2, False): [10, 11],
    }
    vectors = {
        10: [10.0, 0.0, 0.0],
        11: [11.0, 0.0, 0.0],
    }
    idx = DummyIndex(vector_map=vector_map, vectors=vectors)

    mat = idx.get_neighbor_vectors_by_vector("q", 2, as_numpy=False)
    assert mat == [
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
    ]


def test_neighbor_vectors_by_item_with_distances():
    item_map = {
        (0, 2, True): ([0, 1], [0.0, 0.2]),
        (0, 3, True): ([0, 1, 2], [0.0, 0.2, 0.3]),
    }
    vectors = {
        1: [1.0, 1.0, 1.0],
        2: [2.0, 2.0, 2.0],
    }
    idx = DummyIndex(item_map=item_map, vectors=vectors)

    mat, dists = idx.get_neighbor_vectors_by_item(
        0, 2, include_self=False, include_distances=True, as_numpy=False
    )
    assert mat == [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ]
    assert dists == [0.2, 0.3]


# -----------------------------
# Optional NumPy output
# -----------------------------

@pytest.mark.skipif(
    pytest.importorskip("numpy", reason="numpy not installed") is None,
    reason="numpy not installed",
)
def test_neighbor_vectors_numpy_output_shape():
    import numpy as np

    item_map = {
        (0, 2, False): [0, 1],
        (0, 3, False): [0, 1, 2],
    }
    vectors = {
        1: [1.0, 1.0, 1.0],
        2: [2.0, 2.0, 2.0],
    }
    idx = DummyIndex(item_map=item_map, vectors=vectors)

    mat = idx.get_neighbor_vectors_by_item(0, 2, include_self=False, as_numpy=True)
    assert isinstance(mat, np.ndarray)
    assert mat.shape == (2, 3)
