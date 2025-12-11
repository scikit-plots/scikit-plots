# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Transformers for missing value imputation.

This submodule re-exports the standard scikit-learn imputers and
exposes additional experimental imputers via the
:mod:`scikitplot.experimental` namespace.

In particular, :py:class:`~.ANNImputer` is an approximate
nearest-neighbours based imputer built on top of the Spotify
Annoy library. It is gated behind the experimental switch

``from scikitplot.experimental import enable_ann_imputer``

to follow scikit-learn's experimental API conventions.
"""

import typing

from sklearn.impute._base import MissingIndicator, SimpleImputer  # noqa: F401

if typing.TYPE_CHECKING:
    # Avoid errors in type checkers (e.g. mypy) for experimental estimators.
    # TODO: remove this check once the estimator is no longer experimental.
    from ._ann import ANNImputer  # noqa: F401

__all__ = []


# TODO: remove this check once the estimator is no longer experimental.
def __getattr__(name):
    """
    Provide lazy, informative error messages for experimental imputers.

    Accessing certain names (for example :py:class:`~.ANNImputer`)
    from :mod:`scikitplot.impute` requires that the corresponding
    experimental feature has been explicitly enabled. This function
    raises an :class:`ImportError` with a clear instruction when a
    guarded name is requested without the appropriate
    ``enable_...`` import.
    """
    msg = {
        "ANNImputer": (
            "To use it, you need to explicitly import "
            "enable_ann_imputer:\n"
            "from scikitplot.experimental import enable_ann_imputer"
        ),
    }
    if name in ["ANNImputer"]:
        raise ImportError(
            f"{name} is experimental and the API might change without any "
            f"deprecation cycle. {msg[name]}"
        )
    raise AttributeError(f"module {__name__} has no attribute {name}")
