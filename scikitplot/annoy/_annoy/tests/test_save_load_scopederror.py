# scikitplot/annoy/_annoy/tests/test_save_load_scopederror.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for save/load native-error ownership (R14/R19 follow-up).

save() and load() now own the native ``char*`` error via ``ScopedError`` (RAII)
instead of a manual ``free()`` that a future early-return/exception could bypass.
These tests lock the observable contract: round-trip works, and failures raise a
clear ``IOError`` carrying the native message — repeatedly, so any leak/double-free
regression would surface under iteration.
"""
import os
import tempfile

import pytest

from scikitplot.annoy._annoy import annoylib as A


def _built(n=10):
    idx = A.Index(4, "euclidean")
    for i in range(n):
        idx.add_item(i, [float(i)] * 4)
    idx.build(5)
    return idx


def test_save_load_roundtrip():
    src = _built(10)
    fn = os.path.join(tempfile.mkdtemp(), "idx.ann")
    src.save(fn)
    dst = A.Index(4, "euclidean")
    dst.load(fn)
    assert dst.get_n_items() == 10
    for i in range(10):
        assert dst.get_item(i) == src.get_item(i)


def test_load_missing_file_raises_ioerror():
    missing = os.path.join(tempfile.mkdtemp(), "does_not_exist.ann")
    with pytest.raises(IOError) as ei:
        A.Index(4, "euclidean").load(missing)
    assert "load failed" in str(ei.value)


def test_save_to_bad_path_raises_ioerror():
    idx = _built(5)
    with pytest.raises(IOError) as ei:
        idx.save("/nonexistent_dir_xyzzy/idx.ann")
    assert "save failed" in str(ei.value)


def test_repeated_failures_do_not_corrupt(tmp_path):
    # exercise the error path many times: a leak/double-free in the RAII holder
    # would tend to surface as a crash or malloc error under repetition
    idx = _built(5)
    for _ in range(200):
        with pytest.raises(IOError):
            A.Index(4, "euclidean").load(str(tmp_path / "nope.ann"))
        with pytest.raises(IOError):
            idx.save("/nonexistent_dir_xyzzy/idx.ann")
    # still fully functional afterward
    fn = str(tmp_path / "ok.ann")
    idx.save(fn)
    r = A.Index(4, "euclidean")
    r.load(fn)
    assert r.get_n_items() == 5


def test_unbuild_and_serialize_roundtrip_repeated():
    # Exercise the newly ScopedError-migrated paths (unbuild, get_state/set_state
    # serialize/deserialize) under repetition; a leak/double-free would surface.
    import pickle
    for _ in range(100):
        idx = _built(8)
        blob = pickle.dumps(idx)              # get_state -> serialize
        restored = pickle.loads(blob)         # set_state -> deserialize
        assert restored.get_n_items() == 8
        idx.unbuild()                         # unbuild
    # still functional
    final = _built(8)
    assert pickle.loads(pickle.dumps(final)).get_n_items() == 8
