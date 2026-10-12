# scikitplot/stats/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.stats
================

Elegant statistical tools for intuitive and insightful data visualization and interpretation.

The :py:mod:`~scikitplot.stats` module offers a wide range of probability distributions, summary
and frequency statistics, correlation functions, statistical tests,
masked statistics, and additional tools.
"""  # noqa: D205, D400

from __future__ import annotations

import logging
from contextlib import suppress as _suppress

from ..exceptions import ScikitplotException as _ScikitplotException

logger = logging.getLogger(__name__)

try:
    with _suppress(ImportError):
        from ..cexternals._astropy.stats import *  # noqa: F401,F403
except _ScikitplotException:
    logger.warning(
        "Failed to import astropy.stats. Some features may not work as expected.",
    )

try:
    with _suppress(ImportError):
        from ..externals._tweedie import *  # noqa: F401,F403
except _ScikitplotException:
    logger.warning(
        "Failed to import tweedie. Some features may not work as expected.",
    )
