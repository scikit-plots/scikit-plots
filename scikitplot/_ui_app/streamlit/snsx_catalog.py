# snsx/snsx_catalog.py

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=import-error
# pylint: disable=unused-import
# pylint: disable=unused-argument
# pylint: disable=line-too-long

"""
Central registry of ML-EDA plotting functions with metadata.

Module contains metadata for the `snsx` plotting library,
organizing ML-EDA and interpretability functions by:

- Category (e.g., evaluation, features, explanation)
- Task type (classification, regression, unsupervised)
- Plot type (curve, matrix, embedding, etc.)
- Supervised/unsupervised designation
- Explainability level (low, medium, high)

Each function includes a short description and expected input parameters,
enabling registry-driven documentation, Streamlit exploration, and dynamic UIs.

Usage:
    from snsx.snsx_catalog import snsx_catalog
    for plot in snsx_catalog:
        print(plot['function'], plot['description'])
"""

from abc import ABCMeta, abstractmethod  # noqa: F401
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "snsx_catalog",
]


@dataclass
class FunctionMeta(Enum):
    """
    FunctionMeta for scikit-plots any customized (e.g. cli, web) options.

    Inspired scikit-learn metadata routing.

    snsx_catalog = [
        {
            "module": "scikitplot",
            "function": ".snsx.evaluation.roc_curve",
            "fallback_function": ".api.metrics.plot_roc",
            "category": "evaluation",
            "task_type": "classification",
            "plot_type": "diagnostic",
            "supervised": True,
            "explainability_level": "medium",
            "description": "Plot the Receiver Operating Characteristic (ROC) curve to visualize the trade-off between true positive rate and false positive rate.",
            "parameters": ["y_true", "y_score"],
            "optional_parameters": {
                "figsize": (6, 2.85)
            }
        },
    """

    function: str
    module: str
    fallback_function: str
    parameters: list[str]


snsx_catalog = [
    {
        "module": "scikitplot",
        "function": ".snsx.evaluation.plot_classifier_eval",
        "fallback_function": ".api.metrics.plot_classifier_eval",
        "category": "evaluation",
        "task_type": "classification",
        "plot_type": "tabular/matrix",
        "supervised": True,
        "explainability_level": "medium",
        "description": "Plot the.",
        "parameters": ["y_true", "y_pred"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.evaluation.roc_curve",
        "fallback_function": ".api.metrics.plot_roc",
        "category": "evaluation",
        "task_type": "classification",
        "plot_type": "diagnostic",
        "supervised": True,
        "explainability_level": "medium",
        "description": (
            "Plot the Receiver Operating Characteristic (ROC) curve to visualize the trade-off between true positive rate and false positive rate."
        ),
        "parameters": ["y_true", "y_score"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.evaluation.pr_curve",
        "fallback_function": ".api.metrics.plot_precision_recall",
        "category": "evaluation",
        "task_type": "classification",
        "plot_type": "diagnostic",
        "supervised": True,
        "explainability_level": "medium",
        "description": (
            "Plot the Precision-Recall curve for imbalanced classification problems."
        ),
        "parameters": ["y_true", "y_score"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.evaluation.lift_curve",
        "fallback_function": ".kds.plot_lift",
        "category": "evaluation",
        "task_type": "classification",
        "plot_type": "ranking/performance",
        "supervised": True,
        "explainability_level": "medium",
        "description": (
            "Display the lift curve to understand the effectiveness of a predictive model by deciles."
        ),
        "parameters": ["y_true", "y_score"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.evaluation.ks_statistic",
        "fallback_function": ".kds.plot_ks_statistic",
        "category": "evaluation",
        "task_type": "classification",
        "plot_type": "diagnostic",
        "supervised": True,
        "explainability_level": "medium",
        "description": (
            "Plot the Kolmogorov-Smirnov (KS) statistic curve to evaluate the separation between positive and negative classes."
        ),
        "parameters": ["y_true", "y_score"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.representation.pca",
        "fallback_function": ".api.metrics.plot_roc",
        "category": "representation",
        "task_type": "unsupervised",
        "plot_type": "embedding",
        "supervised": False,
        "explainability_level": "low",
        "description": (
            "Visualize the principal component analysis (PCA) 2D projection of input features."
        ),
        "parameters": ["data", "hue"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.explanation.shap_summary",
        "fallback_function": ".api.metrics.plot_roc",
        "category": "explanation",
        "task_type": "general",
        "plot_type": "importance",
        "supervised": True,
        "explainability_level": "high",
        "description": (
            "Show SHAP summary plot to explain global feature importance for a trained model."
        ),
        "parameters": ["model", "X"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.features.mutual_information",
        "fallback_function": ".api.metrics.plot_roc",
        "category": "features",
        "task_type": "general",
        "plot_type": "importance",
        "supervised": True,
        "explainability_level": "medium",
        "description": "Plot mutual information between features and target variable.",
        "parameters": ["X", "y"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.dataset.null_heatmap",
        "fallback_function": ".api.metrics.plot_roc",
        "category": "dataset",
        "task_type": "general",
        "plot_type": "matrix",
        "supervised": False,
        "explainability_level": "low",
        "description": (
            "Visualize missing values in a heatmap to assess data completeness."
        ),
        "parameters": ["df"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    {
        "module": "scikitplot",
        "function": ".snsx.training.loss_curve",
        "fallback_function": ".api.metrics.plot_roc",
        "category": "training",
        "task_type": "general",
        "plot_type": "curve",
        "supervised": True,
        "explainability_level": "medium",
        "description": (
            "Plot model training loss across epochs to monitor convergence and overfitting."
        ),
        "parameters": ["history"],
        "optional_parameters": {"figsize": (6, 2.85)},
    },
    # Additional functions can be appended here following the same structure
]

# Example usage
if __name__ == "__main__":
    from pprint import pprint

    pprint(snsx_catalog)  # noqa: T203
