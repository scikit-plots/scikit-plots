# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CY-005 (guide 28).

The public ``on_disk_build`` failure path decoded a native error string and
raised without freeing it, leaking one heap allocation per failure (the alternate
internal path freed it). It now owns the string via ``ScopedError``, so the free
is automatic on every exit including the raise.

Leak-freedom itself is structurally guaranteed by ``ScopedError`` (RAII); it was
additionally checked with an RSS proxy (0 KB growth over 40k failing builds, see
the RUN13 evidence). These tests lock the observable contract: failures raise,
success still works, and the error path is stable under repetition.
"""
import os
import tempfile

import pytest

from scikitplot.annoy._annoy import annoylib as A

DIM = 4


def _bad_path():
    # a path under a non-existent directory -> native open fails fast
    d = tempfile.mkdtemp()
    return os.path.join(d, "does_not_exist", "sub", "idx.ann")


def test_on_disk_build_failure_raises_ioerror():
    idx = A.Index(DIM, "euclidean")
    with pytest.raises(IOError):
        idx.on_disk_build(_bad_path())


def test_on_disk_build_success_still_works():
    d = tempfile.mkdtemp()
    ok = os.path.join(d, "ok.ann")
    idx = A.Index(DIM, "euclidean")
    idx.on_disk_build(ok)
    for i in range(5):
        idx.add_item(i, [float(i)] * DIM)
    idx.build(3)
    assert idx.get_n_items() == 5


def test_repeated_failures_are_stable():
    # exercises the hardened error-ownership path many times; each must raise
    # cleanly and leave a fresh index fully usable (no corruption/instability).
    bad = _bad_path()
    for _ in range(500):
        idx = A.Index(DIM, "euclidean")
        with pytest.raises(IOError):
            idx.on_disk_build(bad)
    # a fresh index after the failure loop still builds normally
    d = tempfile.mkdtemp()
    ok = os.path.join(d, "after.ann")
    idx = A.Index(DIM, "euclidean")
    idx.on_disk_build(ok)
    idx.add_item(0, [1.0] * DIM)
    idx.build(2)
    assert idx.get_n_items() == 1
