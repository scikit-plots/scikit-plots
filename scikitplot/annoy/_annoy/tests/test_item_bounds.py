# scikitplot/annoy/_annoy/tests/test_item_bounds.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CY-008 (guide 31).

Every item-taking operation must reject a Python-visible invalid id (negative,
over dtype-capacity, or >= n_items) with the documented exception BEFORE any
unchecked native read. Previously ids in ``[n_items, dtype_max]`` reached the
core and returned garbage instead of raising.
"""
import pytest

from scikitplot.annoy._annoy import annoylib as A

DIM = 4


def _index(n=10):
    idx = A.Index(DIM, "euclidean")
    for i in range(n):
        idx.add_item(i, [float(i)] * DIM)
    idx.build(5)
    return idx


@pytest.mark.parametrize("bad", [10, 11, 100, 10_000])
def test_get_item_out_of_range_raises(bad):
    idx = _index(10)
    with pytest.raises(IndexError):
        idx.get_item(bad)


@pytest.mark.parametrize("bad", [10, 100])
def test_get_nns_by_item_out_of_range_raises(bad):
    idx = _index(10)
    with pytest.raises(IndexError):
        idx.get_nns_by_item(bad, 3)


@pytest.mark.parametrize("pair", [(0, 10), (10, 0), (50, 60)])
def test_get_distance_out_of_range_raises(pair):
    idx = _index(10)
    with pytest.raises(IndexError):
        idx.get_distance(*pair)


def test_negative_still_rejected():
    idx = _index(10)
    with pytest.raises(IndexError):
        idx.get_item(-1)


def test_valid_ids_still_work():
    idx = _index(10)
    assert idx.get_item(0) == [0.0] * DIM
    assert idx.get_item(9) == [9.0] * DIM
    assert len(idx.get_nns_by_item(0, 3)) == 3


def test_holes_within_n_items_are_not_rejected():
    # annoy permits gaps; n_items == max_id + 1. An id inside [0, n_items) that
    # was never populated is a valid "hole" (native returns zeros) and must NOT
    # be treated as out-of-range.
    idx = A.Index(DIM, "euclidean")
    idx.add_item(0, [1.0] * DIM)
    idx.add_item(5, [2.0] * DIM)   # creates holes 1..4; n_items == 6
    idx.build(5)
    assert idx.get_n_items() == 6
    idx.get_item(3)                # hole, in range -> must not raise
    with pytest.raises(IndexError):
        idx.get_item(6)            # == n_items -> out of range
