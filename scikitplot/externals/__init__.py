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
External Python-level dependencies vendored for stability and reproducibility.

These modules provide optional functionality (plotting, stats, array APIs, etc.)
that Scikit-Plot integrates tightly with, but cannot rely on being installed in
all environments. Vendoring ensures consistent behavior across versions.

Included vendor modules:
    - _packaging, _probscale, _scipy, _seaborn, _sphinxext, _tweedie
    - array_api_compat, array_api_extra

Each submodule may be safely imported; missing ones are silently ignored.
"""
import contextlib as _contextlib

__all__ = [
    "_packaging",
    "_probscale",
    "_scipy",
    "_seaborn",
    "_sphinxext",
    "_tweedie",
    "array_api_compat",
    "array_api_extra",
]

for _m in __all__:
    with _contextlib.suppress(ImportError):
        # import importlib
        # importlib.import_module(f"scikitplot.externals.{_m}")
        __import__(f"scikitplot.externals.{_m}", globals(), locals())
