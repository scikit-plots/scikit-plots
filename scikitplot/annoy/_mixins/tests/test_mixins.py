# scikitplot/annoy/_mixins/tests/test_mixins.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the composable ``scikitplot.annoy._mixins`` package.

These cover the public behaviour each mixin contributes to the high-level
:class:`scikitplot.annoy.Index` (which composes all six): NumPy batch/array ops
(``NDArrayMixin``), sklearn-style queries (``VectorOpsMixin``), native persistence
(``IndexIOMixin``), metadata JSON export (``MetaMixin``), pickling
(``PickleMixin``), and the plotting surface (``PlottingMixin``). They also lock
the composition (MRO) so a mixin can't silently drop out of the class.
"""
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scikitplot.annoy import Index

DIM = 4


def _built(n=10):
    idx = Index(DIM, "euclidean")
    for i in range(n):
        idx.add_item(i, [float(i)] * DIM)
    idx.build(5)
    return idx


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def test_index_composes_all_six_mixins():
    mro = {c.__name__ for c in Index.__mro__}
    for mixin in (
        "MetaMixin", "IndexIOMixin", "PickleMixin",
        "VectorOpsMixin", "NDArrayMixin", "PlottingMixin",
    ):
        assert mixin in mro, f"{mixin} missing from Index MRO"


# --------------------------------------------------------------------------- #
# NDArrayMixin
# --------------------------------------------------------------------------- #
def test_ndarray_add_items_batch():
    idx = Index(DIM, "euclidean")
    data = np.arange(20, dtype=float).reshape(5, DIM)
    idx.add_items(data)
    idx.build(3)
    assert idx.get_n_items() == 5


def test_ndarray_get_item_vectors_and_to_numpy():
    idx = _built(10)
    got = np.asarray(idx.get_item_vectors([0, 1, 2]))
    assert got.shape == (3, DIM)
    assert np.allclose(got[1], [1.0] * DIM)
    full = np.asarray(idx.to_numpy())
    assert full.shape == (10, DIM)


# --------------------------------------------------------------------------- #
# VectorOpsMixin
# --------------------------------------------------------------------------- #
def test_vectors_query_by_item_returns_nearest():
    idx = _built(10)
    nn = idx.query_by_item(0, 3)
    assert list(nn)[0] == 0            # an item is its own nearest neighbour


def test_vectors_kneighbors_sklearn_style():
    idx = _built(10)
    vecs, dists = idx.kneighbors([[0.0, 0.0, 0.0, 0.0]], n_neighbors=3)
    vecs = np.asarray(vecs)
    dists = np.asarray(dists)
    assert vecs.shape == (1, 3, DIM)
    assert dists.shape == (1, 3)
    assert dists[0, 0] <= dists[0, 1] <= dists[0, 2]   # sorted ascending


# --------------------------------------------------------------------------- #
# IndexIOMixin
# --------------------------------------------------------------------------- #
def test_io_save_load_index_roundtrip():
    src = _built(10)
    fn = str(Path(tempfile.mkdtemp()) / "idx.ann")
    src.save_index(fn)
    # load_index is a classmethod: (f, metric, path)
    loaded = Index.load_index(DIM, "euclidean", fn)
    assert loaded.get_n_items() == src.get_n_items()
    for i in range(src.get_n_items()):
        assert loaded.get_item(i) == src.get_item(i)


# --------------------------------------------------------------------------- #
# MetaMixin
# --------------------------------------------------------------------------- #
def test_meta_to_json_is_valid_and_describes_index():
    idx = _built(10)
    payload = idx.to_json()
    assert isinstance(payload, str)
    obj = json.loads(payload)                  # must be valid JSON
    assert isinstance(obj, dict)
    # metadata should carry the structural identity of the index
    flat = json.dumps(obj)
    assert "euclidean" in flat


# --------------------------------------------------------------------------- #
# PickleMixin
# --------------------------------------------------------------------------- #
def test_pickle_roundtrip_preserves_data():
    idx = _built(10)
    restored = pickle.loads(pickle.dumps(idx))
    assert restored.get_n_items() == idx.get_n_items()
    assert restored.get_item(3) == idx.get_item(3)
    assert restored.get_nns_by_item(0, 3) == idx.get_nns_by_item(0, 3)


@pytest.mark.xfail(
    reason="Known issue: once a lock-using method populates self._lock "
    "(threading.RLock), pickle.dumps fails with 'cannot pickle _thread.RLock'. "
    "A fresh built index pickles fine; the reduce/metadata path must exclude "
    "the lazily-created _lock. Tracked for a focused follow-up.",
    strict=False,
)
def test_pickle_after_lock_use_known_lock_issue():
    idx = _built(10)
    idx._get_lock()                      # any of to_json/to_numpy/kneighbors/... does this
    restored = pickle.loads(pickle.dumps(idx))
    assert restored.get_n_items() == idx.get_n_items()


def test_pickle_requires_built_index_in_byte_mode():
    # byte mode (default 'auto' with no on-disk path) needs a built index
    idx = Index(DIM, "euclidean")
    idx.add_item(0, [1.0] * DIM)   # added but not built
    with pytest.raises((RuntimeError, TypeError)):
        pickle.dumps(idx)


# --------------------------------------------------------------------------- #
# PlottingMixin
# --------------------------------------------------------------------------- #
def test_plotting_surface_exposed_and_callable():
    idx = _built(10)
    # the plotting surface is present and callable (actual rendering may require
    # optional deps like matplotlib, which are imported lazily inside the calls)
    exposed = [n for n in ("plot_index", "plot_knn_edges") if hasattr(idx, n)]
    assert exposed, "PlottingMixin exposed no plotting methods"
    for name in exposed:
        assert callable(getattr(idx, name))
