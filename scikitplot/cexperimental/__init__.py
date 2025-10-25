# scikitplot/cexperimental/__init__.py

# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

"""
C-Experimental modules for Scikit-Plots.

This package contains high-performance functions implemented in C, C++, and Cython,
exposed to Python via Cython and Pybind11 bindings. These utilities are intended
for experimental and performance-critical use cases.
"""
# Notes
# -----
# - APIs are experimental and may change without notice.
# - Modules in this package may rely on compiled extensions.
# - Intended for advanced users and contributors exploring low-level optimizations.
# bindings : Python/C++ bridge interfaces
# include  : Header files for C/C++ extensions
# src      : Source implementations
# tests    : Test suite for experimental features
## Your package/module initialization code goes here
from ._logsumexp import *

## Optionally import modules if available
try: from ._cy_cexperimental import *
except ImportError: pass
try: from ._py_cexperimental import *
except ImportError: pass

# __all__ = [
#     "_cy_cexperimental",
#     "_py_cexperimental",
# ]
# ## this module dependent array_api_extra py>=3.9
# import contextlib as _contextlib
# for _m in __all__:
#     with _contextlib.suppress(ImportError):
#         # import importlib
#         # importlib.import_module(f"scikitplot.cexperimental.{_m}")
#         __import__(f"scikitplot.cexperimental.{_m}", globals(), locals())
# del _m
# del _contextlib
