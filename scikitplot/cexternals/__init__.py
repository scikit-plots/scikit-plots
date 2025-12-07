# scikitplot/cexternals/__init__.py

# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# External, bundled dependencies.
"""
C-External libraries vendored for high-performance extensions for Scikit-Plots.

This package contains optimized C/C++ and template-based libraries that are
bundled with Scikit-Plots to provide fast numerical, geometric, and algorithmic
capabilities used in advanced visualization and machine learning workflows.

These modules wrap native backends (e.g. Annoy, NumCpp, F2PY-generated code)
through Python extension interfaces to ensure consistent performance and
availability across environments, without relying on separate system installs.
"""
# Notes
# -----
# - All modules are optional and imported safely when available.
# - Vendored code may be experimental or limited to internal use.
# - APIs may change in future releases without deprecation warnings.
# _annoy    : Approximate nearest neighbor search implementation
# _astropy  : Astronomy utilities and array backends (vendored subset)
# _f2py     : Fortran-Python interface support
# _numcpp   : NumCpp: A Templatized Header Only C++ Library

## Your package/module initialization code goes here
