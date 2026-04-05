# scikitplot/seaborn/__init__.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
scikitplot.seaborn
==================
Seaborn-style scikit-plots plotting.

This submodule provides a seaborn-like, high-level plotting API for
machine-learning model exploration.
"""  # noqa: D205, D400

from __future__ import annotations

from ._auc import aucplot
from ._confusion_matrix import evalplot
from ._decile import decileplot, print_labels
from ._model import modelplot

__all__ = [
    "aucplot",
    "decileplot",
    "evalplot",
    "modelplot",
    "print_labels",
]
