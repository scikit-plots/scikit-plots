# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
A "grab bag" of relatively small general-purpose utilities that don't have
a clear module/package to live in.
"""

from __future__ import annotations

import contextlib
import difflib
import inspect
import json
import locale
import os
import re
import sys
import threading
import traceback
import unicodedata
from collections import defaultdict
from contextlib import contextmanager
from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "NumpyRNGContext",
]


class NumpyRNGContext:
    """
    A context manager (for use with the ``with`` statement) that will seed the
    numpy random number generator (RNG) to a specific value, and then restore
    the RNG state back to whatever it was before.

    This is primarily intended for use in the astropy testing suit, but it
    may be useful in ensuring reproducibility of Monte Carlo simulations in a
    science context.

    Parameters
    ----------
    seed : int
        The value to use to seed the numpy RNG

    Examples
    --------
    A typical use case might be::

        with NumpyRNGContext(<some seed value you pick>):
            from numpy import random

            randarr = random.randn(100)
            ... run your test using `randarr` ...

        #Any code using numpy.random at this indent level will act just as it
        #would have if it had been before the with statement - e.g. whatever
        #the default seed is.


    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        from numpy import random

        self.startstate = random.get_state()
        random.seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        from numpy import random

        random.set_state(self.startstate)
