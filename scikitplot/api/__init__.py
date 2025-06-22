# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Scikit-plots Functional API module.
"""

# To get sub-modules into your current module's scope or to add global attribute
from ..utils.utils_plot_mpl import stack  # not a module or namespace, global attr
from ..utils.utils_path import remove_path
from .decomposition import *  # into your current module's scope
from .estimators import *
from .metrics import *
from . import plotters

# Remove, filters out private/dunder names.
__all__ = [s for s in sorted(globals().keys() | {*dir()}) if not s.startswith("_")]

from .._testing._pytesttester import PytestTester  # Pytest testing

test = PytestTester(__name__)
del PytestTester
