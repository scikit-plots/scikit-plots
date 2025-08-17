"""
Elegant statistical tools for intuitive and insightful data visualization and interpretation.

The :py:mod:`~scikitplot.stats` module offers a wide range of probability distributions, summary
and frequency statistics, correlation functions, statistical tests,
masked statistics, and additional tools.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from contextlib import suppress as _suppress

from .. import logger as _logger
from ..exceptions import ScikitplotException as _ScikitplotException

try:
    with _suppress(ImportError):
        from ..cexternals._astropy.stats import *  # noqa: F401,F403
except _ScikitplotException:
    _logger.warning(
        "Failed to import astropy.stats. Some features may not work as expected."
    )

try:
    with _suppress(ImportError):
        from ..externals._tweedie import *  # noqa: F401,F403
except _ScikitplotException:
    _logger.warning("Failed to import tweedie. Some features may not work as expected.")
