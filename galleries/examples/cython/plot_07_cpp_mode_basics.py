"""
C++ mode basics: cppclass and libcpp containers
===============================================

.. currentmodule:: scikitplot.cython

This example demonstrates a minimal C++-mode build using:

- ``language="c++"``
- ``libcpp.vector``
- a tiny header-only ``cppclass`` shipped via ``support_files``

No external ``.cpp`` sources are required (header-only usage).

Notes
-----
- ``noexcept`` is intentionally not used in Cython signatures for maximum compatibility.
- Some compilers may require an explicit C++ standard flag; see the comment near
  ``extra_compile_args``.
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
    print(report.get("problems", []))
else:
    support_files = {
        "accum.hpp": r"""
#pragma once

namespace skplt_demo {

class Accumulator {
public:
    Accumulator() : total_(0) {}
    void add(int x) { total_ += x; }
    int value() const { return total_; }

private:
    int total_;
};

}  // namespace skplt_demo
"""
    }

    code = r"""
# distutils: language = c++

from libcpp.vector cimport vector

cdef extern from "accum.hpp" namespace "skplt_demo":
    cdef cppclass Accumulator:
        Accumulator() except +
        void add(int x)
        int value() const

cdef class PyAccumulator:
    cdef Accumulator* _p

    def __cinit__(self):
        self._p = new Accumulator()

    def __dealloc__(self):
        del self._p

    def add(self, int x):
        self._p.add(x)

    def value(self):
        return self._p.value()

def sum_vector():
    cdef vector[int] v
    v.push_back(1)
    v.push_back(2)
    v.push_back(3)
    cdef int s = 0
    cdef int x
    for x in v:
        s += x
    return s

def accumulate_demo():
    a = PyAccumulator()
    a.add(10)
    a.add(20)
    return a.value()
"""

    try:
        r = cython.compile_and_load_result(
            code,
            module_name="cpp_mode_demo",
            profile="fast-debug",
            language="c++",
            support_files=support_files,
            compiler_directives={"language_level": 3, "embedsignature": True},
            # Optional: if you hit "unknown type" or stdlib issues on older toolchains,
            # set an explicit standard. Keep it user-controlled (no guessing).
            # extra_compile_args=["-std=c++11"],
            verbose=0,
        )
    except Exception as e:
        print("C++ build failed:")
        print(type(e).__name__ + ":", e)
    else:
        print("sum_vector() =", r.module.sum_vector(), "(expected 6)")
        print("accumulate_demo() =", r.module.accumulate_demo(), "(expected 30)")

# %%
#
# .. tags::
#
#    domain: cython
#    plot-type: cython
#    purpose: showcase
