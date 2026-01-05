"""
Seaborn-style scikit-plots plotting (snsx).

This submodule provides a seaborn-like, high-level plotting API for
machine-learning model exploration.
"""

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
