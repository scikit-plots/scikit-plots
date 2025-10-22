# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Transformers for missing value imputation."""

import typing

# from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute._base import MissingIndicator, SimpleImputer
from sklearn.impute._iterative import IterativeImputer
from sklearn.impute._knn import KNNImputer

if typing.TYPE_CHECKING:
    # Avoid errors in type checkers (e.g. mypy) for experimental estimators.
    # TODO: remove this check once the estimator is no longer experimental.
    from ._annoy_knn import AnnoyKNNImputer  # noqa: F401

__all__ = [
    "IterativeImputer",
    "KNNImputer",
    "MissingIndicator",
    "SimpleImputer",
]


# TODO: remove this check once the estimator is no longer experimental.
def __getattr__(name):
    msg = {
        "IterativeImputer": (
            "To use it, you need to explicitly import "
            "enable_iterative_imputer:\n"
            "from sklearn.experimental import enable_iterative_imputer"
        ),
        "AnnoyKNNImputer": (
            "To use it, you need to explicitly import "
            "enable_iterative_imputer:\n"
            "from scikitplot.experimental import enable_annoyknn_imputer"
        ),
    }
    if name in ["AnnoyKNNImputer", "IterativeAnnoyKNNImputer"]:
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            f"deprecation cycle. {msg[name]}"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
