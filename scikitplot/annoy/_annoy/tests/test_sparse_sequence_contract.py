# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CY-007 (guide 30).

The Index exposes a low-level extent/array sequence protocol over annoy's sparse
ID domain (Model C). ``__len__``/``__getitem__``/``__contains__``/``__iter__``
are all defined over the ID EXTENT ``[0, len)`` and must be mutually consistent:
membership is ID-range validity (not occupancy), iteration walks the extent, and
``len`` is the extent. These tests lock that self-consistent contract so it can
never silently drift back to promising dense-occupancy semantics.
"""
import pytest

from scikitplot.annoy._annoy import annoylib as A

DIM = 3


def _sparse_index():
    # add IDs 0 and 5 -> extent 6, with gaps 1..4
    idx = A.Index(DIM, "euclidean")
    idx.add_item(0, [1.0, 0.0, 0.0])
    idx.add_item(5, [0.0, 1.0, 0.0])
    idx.build(5)
    return idx


def test_len_is_extent_not_count():
    idx = _sparse_index()
    assert len(idx) == 6            # extent (max id + 1), not the 2 present items
    assert idx.get_n_items() == 6


def test_contains_is_extent_range_membership():
    idx = _sparse_index()
    n = len(idx)
    # membership is exactly ID-range validity over the extent
    for i in range(n):
        assert i in idx            # every in-range ID (incl. gaps) is "in"
    assert n not in idx            # == len is out of range
    assert (n + 10) not in idx
    assert -1 not in idx


def test_contains_matches_getitem_accessibility():
    idx = _sparse_index()
    # Over the extent domain (non-negative IDs), `i in idx` iff `idx[i]` is
    # accessible — the two must agree exactly. (Negative indices are excluded:
    # __getitem__ applies Python negative-index wraparound while __contains__ is
    # strict [0, len) range membership; that divergence is a separate concern.)
    for i in range(0, len(idx) + 3):
        in_idx = i in idx
        try:
            idx[i]
            accessible = True
        except (IndexError, OverflowError):
            accessible = False
        assert in_idx == accessible


def test_iter_walks_extent_and_matches_getitem():
    idx = _sparse_index()
    via_iter = list(idx)
    via_index = [idx[i] for i in range(len(idx))]
    assert len(via_iter) == len(idx)        # one yield per extent slot
    assert via_iter == via_index            # iteration == extent-based getitem
    # a gap slot yields the native zero/placeholder vector (not skipped)
    assert via_iter[1] == [0.0] * DIM


def test_empty_and_unbuilt_extent():
    idx = A.Index(DIM, "euclidean")
    # nothing added yet -> extent 0, empty iteration, nothing in range
    assert len(idx) == 0
    assert list(idx) == []
    assert 0 not in idx
