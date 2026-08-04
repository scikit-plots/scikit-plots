# scikitplot/annoy/_annoy/tests/test_doc_parity.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CY-019 (guide 39).

Documentation parity: the ``Index.__init__`` docstring had drifted from the code
— it claimed ``index_dtype`` default ``"int64"`` (actual ``"int32"``) and that
only ``int32``/``int64`` (and ``float32``/``float64``) were supported with the
rest "future", when in fact all eight integer types and four float types work.
These tests assert the documented defaults and supported sets match reality so
the docs cannot silently drift again.
"""
import re

import pytest

from scikitplot.annoy._annoy import annoylib as A

DOC = A.Index.__init__.__doc__ or ""


def _doc_default(param):
    m = re.search(rf'{param} : str, default="(\w+)"', DOC)
    assert m, f"no documented default for {param}"
    return m.group(1)


def test_documented_index_dtype_default_matches_actual():
    assert _doc_default("index_dtype") == "int32"
    assert A.Index(4, "euclidean").get_params()["index_dtype"] == "int32"


def test_documented_dtype_default_matches_actual():
    assert _doc_default("dtype") == "float32"
    assert A.Index(4, "euclidean").get_params()["dtype"] == "float32"


@pytest.mark.parametrize(
    "idx_dtype",
    ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"],
)
def test_every_documented_index_dtype_actually_works(idx_dtype):
    # each type named in the docstring must construct AND build
    assert idx_dtype in DOC
    idx = A.Index(4, "euclidean", index_dtype=idx_dtype)
    idx.add_item(0, [1.0] * 4)
    idx.build(2)


@pytest.mark.parametrize("data_dtype", ["float16", "float32", "float64", "float128"])
def test_every_documented_dtype_actually_works(data_dtype):
    assert data_dtype in DOC
    idx = A.Index(4, "euclidean", dtype=data_dtype)
    idx.add_item(0, [1.0] * 4)
    idx.build(2)


def test_no_stale_future_claims_for_supported_types():
    # the drifted text claimed working types were "Future" — must be gone
    assert 'Future: "bool", "int8", "uint8", "float16", "float128"' not in DOC
    assert 'only "int32", "int64" supported' not in DOC
