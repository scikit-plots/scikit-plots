# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUILD-WARN-001.

`annoy_build_portable_blob` assembles the portable serialization blob used by
`__getstate__` (pickle) and the portable save path. The payload append was
rewritten from `vector::insert(end(), it, it)` to an equivalent
`reserve()+resize()+memcpy` to clear a spurious GCC 13 `-Wstringop-overflow`.
This test pins that the rewrite is byte-for-byte behavior-preserving: an index
pickled and restored must answer queries identically.
"""
import pickle
import random

import pytest

from scikitplot.cexternals._annoy import annoylib as A


@pytest.mark.parametrize("metric", ["euclidean", "angular", "manhattan", "dot"])
def test_pickle_portable_blob_roundtrip(metric):
    random.seed(1234)
    dim = 6
    idx = A.AnnoyIndex(dim, metric)
    for i in range(80):
        idx.add_item(i, [random.random() for _ in range(dim)])
    idx.build(12)

    queries = [[random.random() for _ in range(dim)] for _ in range(10)]
    before = [idx.get_nns_by_vector(q, 10, include_distances=True) for q in queries]

    blob = pickle.dumps(idx)          # exercises annoy_build_portable_blob
    restored = pickle.loads(blob)

    assert restored.get_n_items() == idx.get_n_items()
    assert restored.get_n_trees() == idx.get_n_trees()
    after = [restored.get_nns_by_vector(q, 10, include_distances=True) for q in queries]

    for (ids_b, d_b), (ids_a, d_a) in zip(before, after):
        assert ids_b == ids_a
        assert d_b == pytest.approx(d_a)


def test_pickle_blob_stable_across_two_dumps():
    # The blob is deterministic: two dumps of the same index are identical bytes.
    random.seed(7)
    idx = A.AnnoyIndex(4, "euclidean")
    for i in range(40):
        idx.add_item(i, [random.random() for _ in range(4)])
    idx.build(8)
    assert pickle.dumps(idx) == pickle.dumps(idx)
