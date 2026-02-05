# scikitplot/random/_kiss.py
#
# ruff: noqa: B018,F405
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

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

from __future__ import annotations

from ..cexternals._annoy._kissrandom import kissrandom as _kiss
from ..cexternals._annoy._kissrandom.kissrandom import *  # noqa: F403

__all__ = []
__all__ += _kiss.__all__
