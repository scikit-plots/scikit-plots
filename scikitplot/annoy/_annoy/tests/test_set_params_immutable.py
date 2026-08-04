# scikitplot/annoy/_annoy/tests/test_set_params_immutable.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CY-006 (guide 29.3).

`set_params` must not let structural type parameters diverge from the concrete
backing: `index_dtype`, `dtype`, `wrapper_dtype`, and `random_dtype` fix the
native type/dispatch at construction, so — like `f`/`metric` — they are immutable
once constructed. Unknown parameters are rejected instead of silently ignored.
Mutable params still work, and `set_params(**get_params())` round-trips.
"""
import pytest

from scikitplot.annoy._annoy import annoylib as A


def _index():
    return A.Index(4, "euclidean")


@pytest.mark.parametrize(
    "param,value",
    [
        ("index_dtype", "int64"),
        ("dtype", "float64"),
        ("wrapper_dtype", "float64"),
        ("random_dtype", "uint64"),
        ("metric", "angular"),
        ("f", 8),
    ],
)
def test_structural_params_immutable_after_construction(param, value):
    idx = _index()  # constructed -> ptr != NULL
    with pytest.raises(ValueError, match="Cannot modify"):
        idx.set_params(**{param: value})


def test_index_dtype_no_longer_silently_diverges():
    idx = _index()
    before = idx.get_params()["index_dtype"]
    with pytest.raises(ValueError):
        idx.set_params(index_dtype="int64")
    # metadata unchanged after the rejected mutation
    assert idx.get_params()["index_dtype"] == before


def test_unknown_parameter_rejected():
    idx = _index()
    with pytest.raises(ValueError, match="Invalid parameter"):
        idx.set_params(not_a_real_param=1)


def test_mutable_params_still_settable():
    idx = _index()
    idx.set_params(n_neighbors=20, seed=42)
    assert idx.get_params()["n_neighbors"] == 20


def test_get_params_roundtrips_through_set_params():
    idx = _index()
    idx.set_params(n_neighbors=7)
    params = idx.get_params()
    # a fresh index takes structural params via __init__; the mutable subset must
    # round-trip through set_params without spurious "unknown parameter" errors.
    fresh = _index()
    mutable = {
        k: v
        for k, v in params.items()
        if k not in ("f", "metric", "index_dtype", "dtype",
                     "wrapper_dtype", "random_dtype")
    }
    fresh.set_params(**mutable)  # must not raise
    assert fresh.get_params()["n_neighbors"] == 7
