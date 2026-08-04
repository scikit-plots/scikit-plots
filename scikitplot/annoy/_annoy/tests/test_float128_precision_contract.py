# scikitplot/annoy/_annoy/tests/test_float128_precision_contract.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CY-012 (guide 39).

All embeddings pass through a double-precision (float64) ``_w`` bridge, so
``float128`` provides no more input/output precision than ``float64`` — its only
benefit is higher-precision internal distance arithmetic. These tests lock that
contract (float128 I/O == float64 I/O) so the public precision claim can't drift
above what the bridge delivers, and confirm the contract is documented.
"""
import pytest

from scikitplot.annoy._annoy import annoylib as A

DIM = 3
# a value that differs from 1.0 only within double precision
V = 1.0 + 2 ** -40


def _io(dtype):
    idx = A.Index(DIM, "euclidean", dtype=dtype)
    idx.add_item(0, [V] * DIM)
    idx.add_item(1, [V + 1.0] * DIM)
    idx.build(4)
    return idx.get_item(0)


def test_float128_io_equals_float64_io():
    # identical I/O -> float128 gives no extra input/output precision (double cap)
    assert _io("float128") == _io("float64")


def test_float128_builds_and_queries():
    idx = A.Index(DIM, "euclidean", dtype="float128")
    for i in range(6):
        idx.add_item(i, [float(i)] * DIM)
    idx.build(5)
    assert idx.get_n_items() == 6
    assert idx.get_nns_by_item(0, 3)[0] == 0


def test_float16_is_narrowed_through_bridge():
    # float16 round-trips through double as a (narrowed) double, not raising
    idx = A.Index(DIM, "euclidean", dtype="float16")
    idx.add_item(0, [1.0] * DIM)
    idx.build(2)
    got = idx.get_item(0)
    assert len(got) == DIM


def test_precision_contract_is_documented():
    doc = A.Index.add_item.__doc__ or ""
    assert "double-precision" in doc and "float128" in doc
    # class docstring keeps the canonical precision note too
    cls_doc = ""
    for c in A.Index.__mro__:
        cls_doc += c.__doc__ or ""
    assert "gains no input precision" in cls_doc
