# scikitplot/nc/_version/__init__.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Module exposing nc::VERSION.
"""

# Cython module (Python side)
from . import _version  # C++ module exposing nc::VERSION

# PYBIND11_MODULE (Python side)
from .__version import __version__  # C++ module exposing nc::VERSION

__all__ = [
    "__version__",
    "_version",
]
