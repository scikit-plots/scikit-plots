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
External C/C++ or compiled libraries vendored for high-performance extensions.

These components wrap native or template-based libraries (e.g. Annoy, NumCpp)
that enable fast numerical, algorithmic, or geometric operations used by
Scikit-Plot's advanced visualization and machine learning utilities.

Included vendor modules:
    - _astropy, _f2py, annoy, NumCpp

Each submodule is optional and imported safely when available.
"""
import contextlib as _contextlib

__all__ = [
    "_astropy",
    "_f2py",
    "annoy",
    "NumCpp",
]

for _m in __all__:
    with _contextlib.suppress(ImportError):
        # import importlib
        # importlib.import_module(f"scikitplot.externals.{_m}")
        __import__(f"scikitplot.externals.{_m}", globals(), locals())
