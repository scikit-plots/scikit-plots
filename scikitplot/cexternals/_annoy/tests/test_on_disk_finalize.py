# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for ANNOY-SAVE-002 (guide 6.7).

If the final truncate during `on_disk_build` finalization fails, the partially
finalized, header-less file must NOT be left on disk where `load()` could accept
it as a (corrupt) valid index. `build()` now removes it on that failure.

The truncate-failure path itself needs filesystem fault injection (ENOSPC / a
bad fd) and lives in CI. These tests lock the happy-path contract: a successful
on-disk build leaves a complete, loadable file, and a normal build does not
remove it.
"""
import os
import random
import tempfile

import pytest

from scikitplot.cexternals._annoy import annoylib as A

DIM = 6


def _on_disk_index(path, items=80, seed=1):
    idx = A.AnnoyIndex(DIM, "euclidean")
    idx.on_disk_build(path)
    r = random.Random(seed)
    for i in range(items):
        idx.add_item(i, [r.random() for _ in range(DIM)])
    idx.build(12)
    return idx


def test_on_disk_finalize_leaves_complete_loadable_file(tmp_path):
    p = str(tmp_path / "od.ann")
    idx = _on_disk_index(p)
    q = [random.random() for _ in range(DIM)]
    expected = idx.get_nns_by_vector(q, 8)
    idx.unload()

    assert os.path.exists(p), "successful on-disk build must keep the file"
    other = A.AnnoyIndex(DIM, "euclidean")
    other.load(p)
    assert other.get_nns_by_vector(q, 8) == expected


def test_on_disk_build_reload_roundtrip(tmp_path):
    p = str(tmp_path / "od2.ann")
    idx = _on_disk_index(p, items=50, seed=7)
    q = [random.random() for _ in range(DIM)]
    a = idx.get_nns_by_vector(q, 10, include_distances=True)
    idx.unload()
    other = A.AnnoyIndex(DIM, "euclidean")
    other.load(p)
    b = other.get_nns_by_vector(q, 10, include_distances=True)
    assert a[0] == b[0]
    assert a[1] == pytest.approx(b[1])
