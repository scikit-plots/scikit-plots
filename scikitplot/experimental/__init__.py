# scikitplot/experimental/__init__.py

# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

"""
Experimental submodules for Scikit-Plots.

This package contains optional or in-development modules that extend
Scikit-Plots functionality in areas such as customer lifetime analytics,
LLM-based components, and UI tooling. These modules are provided on an
experimental basis and may change or be removed in future releases.
"""
# Notes
# -----
# - These modules are not guaranteed to be stable.
# - Importing them is optional; missing dependencies are handled gracefully.
# - APIs may change without notice.
# _clv            : Customer lifetime value analysis
# _doremi         : Domain-specific modeling utilities
# _entities       : Structured data entity representations
# _llm_provider   : Large language model integration utilities
# _snsx           : Extended Seaborn-based plotting tools
# _ui_app         : Experimental user interface components
# pipeline        : Experimental ML pipeline tools
## Your package/module initialization code goes here
from . import pipeline

__all__ = [
    "pipeline",
]
