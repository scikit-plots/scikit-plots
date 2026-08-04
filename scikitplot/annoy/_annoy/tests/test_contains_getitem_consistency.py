# scikitplot/annoy/_annoy/tests/test_contains_getitem_consistency.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression: __contains__ must agree with __getitem__ accessibility (R15 follow-up).

Previously ``self[-1]`` worked (negative indexing) but ``-1 in self`` was False,
contradicting ``__contains__``'s own documented contract ("True iff self[item] is
accessible"). ``__contains__`` now honors negative indexing, so membership is True
iff the same-key ``__getitem__`` would succeed, over ``[-len, len)``.
"""
import pytest

from scikitplot.annoy._annoy import annoylib as A


def _built(n=10):
    idx = A.Index(4, "euclidean")
    for i in range(n):
        idx.add_item(i, [float(i)] * 4)
    idx.build(5)
    return idx


def _accessible(idx, k):
    try:
        idx[k]
        return True
    except IndexError:
        return False


@pytest.mark.parametrize("k", list(range(-15, 15)))
def test_contains_matches_getitem_accessibility(k):
    idx = _built(10)
    assert (k in idx) == _accessible(idx, k), f"mismatch at key {k}"


def test_negative_membership_specifics():
    idx = _built(10)
    assert -1 in idx and idx[-1] == idx[9]      # last item
    assert -10 in idx and idx[-10] == idx[0]    # first item via wraparound
    assert -11 not in idx                        # just past the negative extent
    assert 9 in idx and 10 not in idx            # positive extent boundary


def test_empty_and_missing_ptr_membership():
    idx = A.Index(4, "euclidean")  # not built / no items
    assert 0 not in idx
    assert -1 not in idx
