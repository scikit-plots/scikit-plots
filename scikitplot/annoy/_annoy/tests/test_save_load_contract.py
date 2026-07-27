# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Regression tests for CY-018 (guide 39).

Native ``save``/``load`` persist the NATIVE index only (dimension ``f``, metric,
item vectors, built trees) and validate the structural metadata on load. Wrapper
query parameters (``n_neighbors``) and the build ``seed`` are not part of the
``.ann`` format and remain at the loading instance's configuration. These tests
lock that contract: structural mismatches are rejected, the backing (data + query
results) mirrors exactly, and the documented non-mirrored params behave as stated.
"""
import os
import tempfile

import pytest

from scikitplot.annoy._annoy import annoylib as A

DIM = 6
METRIC = "angular"


def _saved_index():
    idx = A.Index(DIM, metric=METRIC)
    idx.set_params(n_neighbors=9, seed=123)
    for i in range(8):
        idx.add_item(i, [float(i) + 0.1 * j for j in range(DIM)])
    idx.build(7)
    d = tempfile.mkdtemp()
    fn = os.path.join(d, "idx.ann")
    idx.save(fn)
    return idx, fn


def test_load_rejects_dimension_mismatch():
    _, fn = _saved_index()
    wrong = A.Index(DIM + 4, metric=METRIC)
    with pytest.raises(IOError):          # OSError/IOError, not RuntimeError
        wrong.load(fn)


def test_load_rejects_metric_mismatch():
    _, fn = _saved_index()
    wrong = A.Index(DIM, metric="manhattan")
    with pytest.raises(IOError):
        wrong.load(fn)


def test_backing_data_mirrors_after_load():
    src, fn = _saved_index()
    loaded = A.Index(DIM, metric=METRIC)
    loaded.load(fn)
    # structural + data state mirrors the saved index exactly
    assert loaded.get_n_items() == src.get_n_items()
    assert loaded.get_params()["f"] == src.get_params()["f"]
    assert loaded.get_params()["metric"] == src.get_params()["metric"]
    for i in range(src.get_n_items()):
        assert loaded.get_item(i) == src.get_item(i)
    assert loaded.get_nns_by_item(0, 5) == src.get_nns_by_item(0, 5)


def test_wrapper_query_params_are_not_restored_by_native_load():
    # documented contract: n_neighbors/seed are NOT in the .ann format; the
    # loading instance keeps its own configuration (they do not mirror).
    _, fn = _saved_index()   # saved with n_neighbors=9, seed=123
    loaded = A.Index(DIM, metric=METRIC)
    loaded.set_params(n_neighbors=3, seed=999)
    loaded.load(fn)
    assert loaded.get_params()["n_neighbors"] == 3   # loader's, not saved 9
    assert loaded.get_params()["seed"] == 999        # loader's, not saved 123


def test_load_docstring_documents_the_contract():
    doc = A.Index.load.__doc__ or ""
    # the contract (what is / isn't restored) must be documented
    assert "n_neighbors" in doc and "pickle" in doc
    # and the accurate exception type, not the old RuntimeError claim
    assert "RuntimeError" not in doc
