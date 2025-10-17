# scikitplot/experimental/__init__.py

# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# Experimental, bundled dependencies.
"""
Experimental submodules for Scikit-Plots.

These vendored modules provide optional functionality (plotting, stats, array APIs, etc.)
that Scikit-Plots can use, but may not be installed in all environments.

Submodules included:
    _clv, _doremi, _entities, _llm_provider, _ui_app, pipeline

Each submodule can be imported safely; missing ones are ignored.
"""
import contextlib as _contextlib

__all__ = [
    "_clv",
    "_doremi",
    "_entities",
    "_llm_provider",
    "_ui_app",
    "pipeline",
]

for _m in __all__:
    with _contextlib.suppress(ImportError):
        # import importlib
        # importlib.import_module(f"scikitplot.experimental.{_m}")
        __import__(f"scikitplot.experimental.{_m}", globals(), locals())
