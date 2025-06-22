# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Scikit-plots Functional API module.
"""

# To get sub-modules into your current module"s scope or to add global attribute
from .._testing._pytesttester import PytestTester  # Pytest testing
from ..utils.utils_plot_mpl import stack  # not a module or namespace, global attr
from ..utils.utils_path import remove_path
from .decomposition import *  # into your current module"s scope
from .estimators import *
from .metrics import *
from . import plotters

test = PytestTester(__name__)
del PytestTester

# __all__ = [s for s in dir() if not s.startswith("_")]  # Remove dunders.
__all__ = [
    "decomposition",
    "estimators",
    "metrics",
    "plot_calibration",
    "plot_classifier_eval",
    "plot_confusion_matrix",
    "plot_elbow",
    "plot_feature_importances",
    "plot_learning_curve",
    "plot_pca_2d_projection",
    "plot_pca_component_variance",
    "plot_precision_recall",
    "plot_precision_recall_curve",
    "plot_residuals_distribution",
    "plot_roc",
    "plot_roc_curve",
    "plot_silhouette",
    "plotters",
    "remove_path",
    "stack",
    "test",
]
