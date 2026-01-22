# scikitplot/decile/__init__.py

"""
Visualizing predictive model insights for enhanced business decision-making.

The :py:mod:`~scikitplot.decile` module to build nice plots
to explain your modelling efforts easily to business colleagues.
"""

from . import kds  # noqa: I001
from . import modelplotpy
from . import _decile_modelplotpy as _dmpy
from ._decile_modelplotpy import *  # noqa: F403

__all__ = [
    "kds",
    "modelplotpy",  # legacy ModelPlotPy
]
__all__ += _dmpy.__all__  # New ModelPlotPy
