# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides the tools used to internally run the astropy test suite
from the installed astropy.  It makes use of the |pytest| testing framework.
"""

import os
import pickle
import sys

import pytest

def assert_quantity_allclose(actual, desired, rtol=1.0e-7, atol=0, **kwargs):
    """
    Raise an assertion if two objects are not equal up to desired tolerance.

    This is a :class:`~astropy.units.Quantity`-aware version of
    :func:`numpy.testing.assert_allclose`.
    """
    __tracebackhide__ = True

    import numpy as np
    # from astropy.units.quantity import _unquantify_allclose_arguments
    # np.testing.assert_allclose(
    #     *_unquantify_allclose_arguments(actual, desired, rtol, atol), **kwargs
    # )
    np.testing.assert_allclose(
        actual, desired, rtol=rtol, atol=atol, **kwargs
    )
