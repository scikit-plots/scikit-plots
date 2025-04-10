"""
Experimental functions and utilities by Python and Cython/Pybind11 Bindings.

This package contains experimental functions and utilities for use with
Cython/Pybind11 Bindings and Python implementations. The functions provided here
are optimized for performance and are implemented using Cython/Pybind11 Bindings
to leverage its speed advantages.

This package is intended for experimental use. Functionality and APIs may
change in future releases.
"""

# scikitplot/experimental/__init__.py

## this module dependent array_api_extra py>=3.9

## Your package/module initialization code goes here
from ._logsumexp import *

## Optionally import other modules if available
try:
    from ._cy_experimental import *
    from ._py_experimental import *
except ImportError:
    # Optional modules not available
    pass
