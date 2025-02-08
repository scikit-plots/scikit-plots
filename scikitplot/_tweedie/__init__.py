r"""
Tweedie Distribution Module
===============================================

This module implements the Tweedie distribution,
a member of the exponential dispersion model (EDM) family,
using SciPy's :py:mod:`~scipy.stats.rv_continuous` class.

It is especially useful for modeling claim amounts in the insurance industry,
where data often exhibit a mixture of zeroes and positive continuous values.

The primary focus of this package is the compound-Poisson behavior
of the Tweedie distribution, particularly in the range `1 < p < 2`.
However, it supports calculations for all valid values of the shape parameter `p`.
"""

# scikitplot/_tweedie/__init__.py

from ._tweedie_dist import *

# Define the tweedie version
# https://pypi.org/project/tweedie/#history
__version__ = "0.0.9"
__author__ = "Peter Quackenbush"
__author_email__ = "pquack@gmail.com"

# Define the tweedie git hash
# scikitplot._build_utils.gitversion.git_remote_version(url='https://github.com/thequackdaddy/tweedie')[0]
__git_hash__ = "f14a189d7cd80d41886041f44f40ae4db27d0067"

# Define __all__ to control what gets imported with 'from module import *'
# Combine global names (explicitly defined in the module) and dynamically available names
__all__ = [
    "tweedie_gen",
    "tweedie",
]
