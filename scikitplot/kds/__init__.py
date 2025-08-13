# scikitplot/kds/__init__.py

# fmt: off
# ruff: noqa
# ruff: noqa: PGH004
# flake8: noqa
# pylint: skip-file
# mypy: ignore-errors
# type: ignore

# This module was copied from the kds project.
# https://github.com/tensorbored/kds

"""
Quick report for business analysis

The :py:mod:`~scikitplot.kds` KeyToDataScience module to Plot Decile Table, Lift, Gain
and KS Statistic charts with single line functions

Just input "labels" and "probabilities" to get quick report for analysis

kds is the result of a data scientist's humble effort to provide an easy way of
visualizing metrics. So that one can focus on the analysis rather than hassling
with copy/paste of various visialization functions.
"""

# Your package/module initialization code goes here
from .._testing._pytesttester import PytestTester  # Pytest testing
from ._kds import *

test = PytestTester(__name__)
del PytestTester

# Define the kds version
# https://github.com/tensorbored/kds/blob/master/setup.py
__version__ = "0.1.3"
__author__ = "Prateek Sharma"
__author_email__ = "s.prateek3080@gmail.com"

# Define the kds git hash
# from .._build_utils.gitversion import git_remote_version

# __git_hash__ = git_remote_version(url="https://github.com/scikit-plots/kds")[0]
__git_hash__ = "18a2e90872f0dae8bb92a2eb13f637eeaa196fc4"
# del git_remote_version

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = [
    "decile_table",
    "plot_cumulative_gain",
    "plot_ks_statistic",
    "plot_lift",
    "plot_lift_decile_wise",
    "print_labels",
    "report",
    "test",
]
