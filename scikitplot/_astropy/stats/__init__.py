# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""This subpackage contains statistical tools provided for or used by Astropy.

While the :py:mod:`scipy.stats` package contains a wide range of statistical
tools, it is a general-purpose package, and is missing some that are
particularly useful to astronomy or are used in an atypical way in
astronomy. This package is intended to provide such functionality, but
*not* to replace :py:mod:`scipy.stats` if its implementation satisfies
astronomers' needs.
"""
from . import (
  _bayesian_blocks as _bb,
  funcs,
  _histogram as _hist,
  info_theory,
)
from ._bayesian_blocks import *
from .funcs import *
from ._histogram import *
from .info_theory import *

__all__ = []
__all__ += _bb.__all__
__all__ += funcs.__all__
__all__ += _hist.__all__
__all__ += info_theory.__all__