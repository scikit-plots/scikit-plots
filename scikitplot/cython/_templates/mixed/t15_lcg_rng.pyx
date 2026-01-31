# cython: language_level=3
"""
t15_lcg_rng — deterministic RNG (LCG) as cdef class.

What it demonstrates
--------------------
- A tiny reproducible PRNG implemented with 32-bit arithmetic.
- cdef class encapsulation for state and speed.

How to run
----------
>>> from scikitplot.cython import compile_template
>>> m = compile_template("t15_lcg_rng")
>>> r = m.LCG(123)
>>> [r.next_u32() for _ in range(3)]
[... deterministic ...]
"""

cdef class LCG:
    cdef unsigned int _state

    def __cinit__(self, unsigned int seed=1):
        if seed == 0:
            seed = 1
        self._state = seed

    cpdef unsigned int next_u32(self):
        # Numerical Recipes LCG constants (deterministic).
        self._state = self._state * 1664525u + 1013904223u
        return self._state

    cpdef double next_float01(self):
        # Convert to [0, 1) using 32-bit scaling.
        return self.next_u32() / 4294967296.0
