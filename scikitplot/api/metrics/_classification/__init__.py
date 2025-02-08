"""
Classification metrics
========================================================

The :py:mod:`~scikitplot.metrics` module includes plots for machine learning evaluation metrics
e.g. confusion matrix, silhouette scores, etc.
"""

# scikitplot/api/metrics/_classification/__init__.py

# Your package/module initialization code goes here
from ._calibration import (
    plot_calibration as plot_calibration,
)
from ._confusion_matrix import (
    plot_classifier_eval as plot_classifier_eval,
)
from ._confusion_matrix import (
    plot_confusion_matrix as plot_confusion_matrix,
)
from ._precision_recall_curve import (
    plot_precision_recall as plot_precision_recall,
)
from ._precision_recall_curve import (
    plot_precision_recall_curve as plot_precision_recall_curve,
)
from ._roc_curve import (
    plot_roc as plot_roc,
)
from ._roc_curve import (
    plot_roc_curve as plot_roc_curve,
)
