"""
Multi-file builds: .pxi includes and external headers
=====================================================

.. currentmodule:: scikitplot.cython

Real projects often split Cython code across multiple files:

- ``.pxi`` for shared Cython fragments (``include "helper.pxi"``)
- ``.pxd`` for shared declarations (usually used with ``cimport`` across modules)
- ``.h`` for C-level declarations used via ``cdef extern from ...``

This single-module example uses ``support_files`` to ship extra files into the
build dir without touching your repository.

Important
---------
A ``.pxd`` file alone does not create a runtime module. Using ``cimport name``
requires that ``name`` is an actual compiled module. To keep this demo as a
single-module build, we include the ``.pxd`` textually instead of ``cimport``.
See the package-build note below for the real multi-module ``cimport`` pattern.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
from __future__ import annotations

from scikitplot import cython

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
    support_files = {
        # Declarations file (used as textual include in this single-module demo).
        "_helper_decl.pxd": "cdef inline int square_d(int x):\n    return x * x\n",
        # Implementation fragment (text include).
        "_helper_square.pxi": "cdef int square_i(int x):\n    return x * x\n",
        # C header (extern usage).
        "_helper_add.h": "static int add_i(int a, int b) { return a + b; }\n",
    }

    code = r"""
# distutils: language = c

# Single-module demo: include .pxd textually for declarations (no runtime import).
from _helper_decl cimport square_d

include "_helper_square.pxi"

cdef extern from "_helper_add.h":
    int add_i(int a, int b)

def squared(int n):
    return square_d(n)

def square(int n):
    return square_i(n)

def add(int a, int b):
    return add_i(a, b)
"""

    r = cython.compile_and_load_result(
        code,
        module_name="multifile_demo",
        profile="fast-debug",
        support_files=support_files,
        compiler_directives={"language_level": 3, "embedsignature": True},
        verbose=0,
    )

    print("squared(7) =", r.module.squared(7))
    print("square(7)  =", r.module.square(7))
    print("add(2,3)   =", r.module.add(2, 3))

# %%
#
# .. tags::
#
#    domain: cython
#    plot-type: cython
#    purpose: showcase
