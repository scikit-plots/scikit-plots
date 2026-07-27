# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CY-010 (guide 34).

State restore must reconstruct the exact concrete dispatch type. ``set_state``
restored the type STRINGS (``index_dtype``/``dtype``) but not the dispatch ENUMS
(``index_type_id``/``data_type_id``) that ``_ensure_index`` uses, so a non-default
``index_dtype`` (e.g. ``int64``) rebuilt a default int32 backing and then failed to
deserialize the non-default blob ("n_nodes < n_items"). Now the enums are derived
from the strings on restore, matching ``__init__``.
"""
import pickle

import pytest

from scikitplot.annoy._annoy import annoylib as A

DIM = 4


def _build(**kw):
    idx = A.Index(DIM, **kw)
    for i in range(6):
        idx.add_item(i, [float(i) + 0.1 * j for j in range(DIM)])
    idx.build(5)
    return idx


def _snapshot(idx):
    p = idx.get_params()
    return (
        p["index_dtype"], p["dtype"], p["metric"],
        idx.get_n_items(),
        idx.get_item(3),
        idx.get_nns_by_item(0, 3),
    )


@pytest.mark.parametrize("index_dtype", ["int32", "int64", "uint64"])
def test_index_dtype_state_roundtrip(index_dtype):
    idx = _build(metric="euclidean", index_dtype=index_dtype)
    before = _snapshot(idx)
    restored = pickle.loads(pickle.dumps(idx))
    after = _snapshot(restored)
    assert before == after
    # type metadata is preserved exactly, not silently coerced to a default
    assert restored.get_params()["index_dtype"] == index_dtype


@pytest.mark.parametrize("metric", ["euclidean", "angular", "manhattan", "dot", "hamming"])
def test_metric_state_roundtrip(metric):
    idx = _build(metric=metric)
    assert _snapshot(pickle.loads(pickle.dumps(idx))) == _snapshot(idx)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_data_dtype_state_roundtrip(dtype):
    idx = _build(metric="euclidean", dtype=dtype)
    assert _snapshot(pickle.loads(pickle.dumps(idx))) == _snapshot(idx)


def test_combined_non_default_types_roundtrip():
    idx = _build(metric="euclidean", index_dtype="int64", dtype="float64")
    before = _snapshot(idx)
    restored = pickle.loads(pickle.dumps(idx))
    assert _snapshot(restored) == before
    assert restored.get_params()["index_dtype"] == "int64"
    assert restored.get_params()["dtype"] == "float64"


def test_restored_index_is_queryable_and_consistent():
    # boundary/numerical semantics survive: same nns for a fresh query vector
    idx = _build(metric="euclidean", index_dtype="int64")
    q = [0.5, 0.5, 0.5, 0.5]
    before = idx.get_nns_by_vector(q, 4)
    restored = pickle.loads(pickle.dumps(idx))
    assert restored.get_nns_by_vector(q, 4) == before
