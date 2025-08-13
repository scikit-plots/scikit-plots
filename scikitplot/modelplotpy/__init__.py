# scikitplot/modelplotpy/__init__.py

# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the modelplotpy project.
# https://github.com/modelplot/modelplotpy

"""
Visualizing predictive model insights for enhanced business decision-making.

The :py:mod:`~scikitplot.modelplotpy` module to build nice plots
to explain your modelling efforts easily to business colleagues.

Documentation is available in the docstrings and
online at https://modelplot.github.io/.

.. seealso::

   * https://www.kdnuggets.com/2018/10/evaluating-business-value-predictive-models-modelplotpy.html
"""

# Your package/module initialization code goes here
from .._testing._pytesttester import PytestTester  # Pytest testing
from ._modelplotpy import *

test = PytestTester(__name__)
del PytestTester

# Define the modelplotpy version
# https://github.com/modelplot/modelplotpy/blob/master/setup.py
__version__ = "1.0.0"
__author__ = "Pieter Marcus"
__author_email__ = "pb.marcus@hotmail.com"

# Define the modelplotpy git hash
# from .._build_utils.gitversion import git_remote_version

# __git_hash__ = git_remote_version(url="https://github.com/scikit-plots/modelplotpy")[0]
__git_hash__ = "83ca84e67c357ee3bd98e296b94219c1a0863f68"
# del git_remote_version

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = [
    "ModelPlotPy",
    # "_check_input",
    # "_modelplotpy",
    # "_range01",
    "plot_all",
    "plot_costsrevs",
    "plot_cumgains",
    "plot_cumlift",
    "plot_cumresponse",
    "plot_profit",
    "plot_response",
    "plot_roi",
    "test",
]
