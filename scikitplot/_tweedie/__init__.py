r"""
Tweedie Distribution Module
===============================================

This module implements the Tweedie distribution, a member of the exponential dispersion model (EDM) family, using SciPy's :py:mod:`~scipy.stats.rv_continuous` class.
It is especially useful for modeling claim amounts in the insurance industry, where data often exhibit a mixture of zeroes and positive continuous values.

The primary focus of this package is the compound-Poisson behavior of the Tweedie distribution, particularly in the range `1 < p < 2`.
However, it supports calculations for all valid values of the shape parameter `p`.

Features
--------
- Supports modeling data with a point mass at zero and a continuous positive domain.
- Parameterized by a mean and a variance function of the form `Var(Y) = \phi \mu^p`, where `p` is a shape parameter.
- Encompasses well-known distributions as special cases for specific values of `p`.
- Implements SciPy's :py:mod:`~scipy.stats.rv_continuous` class for seamless integration with Python scientific libraries.

Notes
-----
The probability density function (PDF) of the Tweedie distribution cannot be expressed in a closed form for most values of `p`.
However, approximations and numerical methods are employed to compute the PDF for practical purposes.
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
__githash__ = 'f14a189d7cd80d41886041f44f40ae4db27d0067'

# Define __all__ to control what gets imported with 'from module import *'
# Combine global names (explicitly defined in the module) and dynamically available names
__all__ = [
  'tweedie_gen',
  'tweedie',
]