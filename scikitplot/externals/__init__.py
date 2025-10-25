# scikitplot/externals/__init__.py

# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the scikit-learn project.
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/externals

# External, bundled dependencies.
"""
External dependencies vendored for stability and reproducibility for Scikit-Plots.

This package contains third-party Python modules that Scikit-Plots depends on for
plotting, statistics, array APIs, and numerical utilities. These modules are
bundled directly to ensure consistent behavior across different environments,
even when the original libraries are missing or version-incompatible.
"""
# Notes
# -----
# - These modules are *vendored*, meaning they are included in the source tree
#   and isolated from system-wide installations.
# - They are loaded only when required.
# - Missing optional components are handled gracefully without raising import errors.
# Vendored subpackages include (but are not limited to):
#     _packaging        : Lightweight packaging utilities
#     _probscale        : Probability scale transformations
#     _scipy            : Minimal SciPy functionality for internal use
#     _seaborn          : Core plotting extensions
#     _sphinxext        : Sphinx documentation tools
#     _tweedie          : Tweedie distribution helpers
#     array_api_compat  : Compatibility layer for array API standards
#     array_api_extra   : Extensions to array API compatibility
## Your package/module initialization code goes here

## Optionally import modules if available
try: from . import array_api_compat
except ImportError: pass
try: from . import array_api_extra
except ImportError: pass

__all__ = [
    "array_api_compat",
    "array_api_extra",
]
