# scikitplot/random/_kiss/__init__.py
#
# ruff: noqa: F401,F405
# flake8: noqa: F403
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Use cpdef for public API, cdef for internal helpers
# Always manage memory explicitly (__cinit__/__dealloc__)
# Use nogil when possible for performance
# Avoid .pxi files in modern code
# #
# kissrandom.h           # Pure C++ (logic, constants, inline funcs)
# kiss_random.pxd         # (like C headers) Cython C++ declarations ONLY (cppclass) (no Python-facing logic)
# kiss_random.pxi         # OPTIONAL thin helpers (rare, usually empty) Share Cython code between multiple .pyx for beginners: avoid .pxi
# kiss_random.pyx         # (like C source files) Cython Python implementation code wrapper class ONCE (Python-facing cdef class wrappers logic)
# kiss_random.pyi         # (for Python tooling) Python type hints (typing only for users, IDEs)
#
# C++ (kissrandom.h)
#         ↓
# Cython declarations (kiss_random.pxd)
#         ↓
# Python wrapper (kiss_random.pyx OR annoy_wrapper.pyx)
#
# cdef cppclass CKiss32Random:    # in .pxd
# cdef class Kiss32Random:        # in .pyx
# Never both with the same name (.h -> .pxd -> .pyx)
# Either as a cppclass (C++ side)
# Or as a cdef class (Python wrapper)

"""
Random Number Generation (Numpy-Like :class:`~numpy.random.Generator`).

.. currentmodule:: scikitplot.random

Use ``default_rng()`` to create a `Generator` and call its methods.

=============== =========================================================
Generator
--------------- ---------------------------------------------------------
Generator       Class implementing all of the random number distributions
default_rng     Default constructor for ``Generator``
=============== =========================================================

.. seealso::
  * https://de.wikipedia.org/wiki/KISS_(Zufallszahlengenerator)

References
----------
.. [1] Marsaglia, G. (1999). "Random Number Generators."
       Journal of Modern Applied Statistical Methods, 2(1), 2-13.
.. [2] Jones, D. "Good Practice in (Pseudo) Random Number Generation for
       Bioinformatics Applications."
       https://www0.cs.ucl.ac.uk/staff/d.jones/GoodPracticeRNG.pdf
.. [3] NumPy Development Team. "Random Generator."
       https://numpy.org/doc/stable/reference/random/generator.html

Examples
--------
>>> from scikitplot.random import default_rng, kiss_context
>>> rng = default_rng(42)
>>> data = rng.random(1000)

Context manager

>>> with default_rng(42) as rng:
...     data = rng.random(1000)
>>>
>>> with kiss_context(42) as rng:
...     data = rng.random(1000)

Serialization

>>> import pickle
>>> state = pickle.dumps(rng)
>>> restored = pickle.loads(state)

JSON export

>>> import json
>>> json_str = json.dumps(rng.serialize())
>>> restored = KissGenerator.deserialize(json.loads(json_str))
"""

from ..._utils import set_module
from . import kiss_random
from .kiss_random import *

__all__ = []
__all__ += kiss_random.__all__


Kiss32Random = set_module()(Kiss32Random)
Kiss64Random = set_module()(Kiss64Random)
KissRandom.__doc__ = KissRandom.__doc__
KissSeedSequence = set_module()(KissSeedSequence)
KissBitGenerator = set_module()(KissBitGenerator)
KissGenerator = set_module()(KissGenerator)
KissRandomState = set_module()(KissRandomState)
default_rng.__doc__ = default_rng.__doc__
kiss_context.__doc__ = kiss_context.__doc__
