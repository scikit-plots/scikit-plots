# scikitplot/annoy/_annoy/tests/test_rng_noexcept_decls.py
#
# flake8: noqa: D213
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for CY-017 (guide 27.2).

The Cython declarations must mirror the strict no-throw core. The
``Kiss64Random`` operational methods (``reset``/``set_seed``/``kiss``/``flip``/
``index``) are non-allocating pure arithmetic and cannot throw, so they are
declared ``noexcept`` rather than ``except +`` (allocating constructors keep
``except +``). This guards that the RNG integrates correctly under the no-throw
declarations, and — when the build input is available — that the declarations
have not drifted back to ``except +``.
"""
import os

import pytest

from scikitplot.annoy._annoy import annoylib as A


def _harder_index(seed):
    # 1 tree over 20-D gaussian data -> approximate, so tree-construction RNG
    # actually influences results (exercises the Kiss64Random path).
    import random
    random.seed(0)
    data = [[random.gauss(0, 1) for _ in range(20)] for _ in range(400)]
    idx = A.Index(20, "euclidean")
    idx.set_params(seed=seed)
    for i, v in enumerate(data):
        idx.add_item(i, v)
    idx.build(1)
    return idx.get_nns_by_item(0, 10)


def test_rng_is_reproducible_under_noexcept_decls():
    # same seed -> identical approximate neighbours (RNG path intact, no-throw)
    assert _harder_index(1) == _harder_index(1)


def test_rng_is_actually_active():
    # different seeds -> different approximate neighbours (RNG really drives it)
    assert _harder_index(1) != _harder_index(999)


def test_pxd_rng_methods_declared_noexcept_not_except_plus():
    # Structural guard on the build input, if present alongside the package.
    here = os.path.dirname(os.path.dirname(__file__))  # annoy/_annoy
    pxd = os.path.join(here, "annoylib.pxd.in")
    if not os.path.exists(pxd):
        pytest.skip("annoylib.pxd.in not present in this install")
    with open(pxd, encoding="utf-8") as f:
        text = f.read()
    # find the Kiss64Random extern block and check its operational methods
    for method in ("uint64_t kiss()", "void set_seed(uint64_t seed)",
                   "int flip()", "size_t index(size_t n)",
                   "void reset(uint64_t seed)", "void reset_default()"):
        line = next((ln for ln in text.splitlines() if method in ln), None)
        assert line is not None, f"decl not found: {method}"
        assert "noexcept" in line and "except +" not in line, (
            f"RNG method must be noexcept (no-throw core): {line.strip()}"
        )
