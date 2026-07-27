# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CY-017 (guide 27.2).

The C++ interface is a strict no-throw core: every virtual method is ``noexcept``
and reports failure via ``bool`` + ``char** error``. The Cython ``.pxd`` now
declares those interface methods ``noexcept`` (was ``except +``), so there is a
SINGLE fault model — failures surface as Python exceptions raised by the wrapper
from the error channel, never via C++ exception translation. These tests lock
that: every failing interface call raises a clear Python exception (no silent
success, no crash), and success paths are unaffected.
"""
import os
import tempfile

import pickle
import pytest

from scikitplot.annoy._annoy import annoylib as A

DIM = 4


def _built():
    idx = A.Index(DIM, "euclidean")
    for i in range(5):
        idx.add_item(i, [float(i)] * DIM)
    idx.build(3)
    return idx


def test_get_item_failure_raises_via_channel():
    with pytest.raises(IndexError):
        _built().get_item(99)          # out-of-range -> wrapper raises


def test_load_failure_raises_via_channel():
    d = tempfile.mkdtemp()
    with pytest.raises((OSError, IOError)):
        A.Index(DIM, "euclidean").load(os.path.join(d, "missing.ann"))


def test_on_disk_build_failure_raises_via_channel():
    d = tempfile.mkdtemp()
    with pytest.raises((OSError, IOError)):
        A.Index(DIM, "euclidean").on_disk_build(os.path.join(d, "no", "x.ann"))


def test_success_paths_unaffected_by_noexcept_decls():
    idx = _built()
    assert idx.get_item(2) == [2.0] * DIM
    assert len(idx.get_nns_by_item(0, 3)) == 3
    assert idx.get_distance(0, 1) > 0
    # serialize/deserialize (both noexcept in the core) still round-trip
    restored = pickle.loads(pickle.dumps(idx))
    assert restored.get_n_items() == 5
    assert restored.get_item(2) == [2.0] * DIM


def test_no_failure_is_silent():
    # a failing call must never quietly "succeed" (return a bogus value); it
    # must raise. This guards the single-fault-model contract end to end.
    idx = _built()
    raised = False
    try:
        idx.get_item(10_000)
    except Exception:
        raised = True
    assert raised
