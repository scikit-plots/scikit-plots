# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CY-004 (guide 28).

The deallocation path must be no-fail: `_destroy_index` (called from
`__dealloc__`) only invokes `unload()` and `del self.ptr`, both of which are
`noexcept` in the native layer, so no Python exception can be established while
an index is being garbage-collected — across partial init, double unload, and
mmap-backed (loaded) state. These exercises must never crash or raise.
"""
import gc
import os
import tempfile

from scikitplot.annoy._annoy import annoylib as A

DIM = 4


def _built():
    idx = A.Index(DIM, "euclidean")
    for i in range(5):
        idx.add_item(i, [float(i)] * DIM)
    idx.build(3)
    return idx


def test_dealloc_partial_init():
    idx = A.Index(DIM, "euclidean")   # constructed, never built
    del idx
    gc.collect()


def test_dealloc_built():
    idx = _built()
    del idx
    gc.collect()


def test_dealloc_after_double_unload():
    idx = _built()
    idx.unload()
    idx.unload()      # idempotent
    del idx
    gc.collect()


def test_dealloc_unload_before_build():
    idx = A.Index(DIM, "euclidean")
    idx.unload()      # nothing constructed yet
    del idx
    gc.collect()


def test_dealloc_loaded_mmap_backed():
    src = _built()
    d = tempfile.mkdtemp()
    fn = os.path.join(d, "x.ann")
    src.save(fn)
    loaded = A.Index(DIM, "euclidean")
    loaded.load(fn)                    # mmap-backed
    assert loaded.get_n_items() == 5
    del loaded
    gc.collect()


def test_many_create_destroy_cycles_stable():
    # repeated construct/build/destroy must stay stable (no leak-driven failure)
    for _ in range(200):
        idx = _built()
        del idx
    gc.collect()
