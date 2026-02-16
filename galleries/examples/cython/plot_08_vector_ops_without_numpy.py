"""
Vector ops without NumPy: array('d') + memoryviews
==================================================

.. currentmodule:: scikitplot.cython

You can write fast numeric code without NumPy using Python's ``array`` module
and typed memoryviews.

Why this works
--------------
``array('d')`` exposes the buffer protocol, so Cython can view its memory as a
contiguous ``double[:]`` without copying.

This example scales a vector in-place and demonstrates:

- typed memoryviews (``double[:]``)
- strict input validation (buffer-compatible, correct itemsize)
- optional compiler directives for speed (boundscheck/wraparound)

Notes
-----
This example does not require NumPy.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
from __future__ import annotations

from array import array
from typing import Iterable

from scikitplot import cython

# %%


def py_scale(x: array, a: float) -> None:
    """Pure-Python baseline: scale in-place."""
    for i in range(len(x)):
        x[i] *= a


# %%
# Generate `python` Module from `cython`
# --------------------------------------

report = cython.check_build_prereqs(numpy=False)

if not report.get('cython', {}).get('ok'):
    print("Skipping compilation because build prerequisites are missing.")
    problems = report.get("problems", [])
    if problems:
        print("Problems:", problems)
else:
    code = r"""
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

def scale(double[:] x, double a):
    '''Scale a 1D contiguous vector in-place.'''
    cdef Py_ssize_t i
    cdef Py_ssize_t n = x.shape[0]
    for i in range(n):
        x[i] *= a
    return None

def scaled_copy(double[:] x, double a):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = x.shape[0]
    out = [0.0] * n
    for i in range(n):
        out[i] = x[i] * a
    return out
"""

    # Use result API so we can show cache metadata.
    r = cython.compile_and_load_result(
        code,
        module_name="memview_scale_demo",
        profile="fast-debug",
        numpy_support=False,
        verbose=0,
    )
    m = r.module

    print("Build key:", r.key)
    print("Artifact :", r.artifact_path)

    # ------------------------------------------------------------
    # Strict demo: array('d') is buffer-compatible with double[:]
    # ------------------------------------------------------------
    x = array("d", [1.0, 2.0, 3.0])

    # Educational sanity checks (strict, no heuristics).
    print("\nInput type:", type(x).__name__)
    print("Typecode   :", x.typecode)
    print("Length     :", len(x))

    # Run compiled scaling.
    m.scale(x, 10.0)
    print("After Cython scale:", list(x))
    print("Expected         :", [10.0, 20.0, 30.0])

    # ------------------------------------------------------------
    # Baseline: pure Python scaling (for learning)
    # ------------------------------------------------------------
    y = array("d", [1.0, 2.0, 3.0])
    py_scale(y, 10.0)
    print("\nAfter Python scale:", list(y))

    # ------------------------------------------------------------
    # Strict edge case: wrong typecode (will raise TypeError)
    # ------------------------------------------------------------
    z = array("f", [1.0, 2.0, 3.0])  # float32
    try:
        m.scale(z, 2.0)
        print("Unexpected: scale accepted array('f').")
    except Exception as e:
        print("\nAs expected, scale rejected array('f'):")
        print(" ", type(e).__name__ + ":", e)

# %%
#
# .. tags::
#
#    domain: cython
#    plot-type: cython
#    purpose: showcase
