# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Enables AnnoyKNNImputer, IterativeAnnoyKNNImputer

The API and results of this estimator might change without any deprecation
cycle.

Importing this file dynamically sets :class:`~scikitplot.impute.AnnoyKNNImputer`
as an attribute of the impute module::

    >>> # explicitly require this experimental feature
    >>> from scikitplot.experimental import enable_annoyknn_imputer  # noqa
    >>> # now you can import normally from impute
    >>> from scikitplot.impute import AnnoyKNNImputer
"""
from .. import impute
from ..impute._annoy_knn import AnnoyKNNImputer
from ..impute._iterative_annoy_knn import IterativeAnnoyKNNImputer

# use settattr to avoid mypy errors when monkeypatching
setattr(impute, "AnnoyKNNImputer", AnnoyKNNImputer)
setattr(impute, "IterativeAnnoyKNNImputer", IterativeAnnoyKNNImputer)
impute.__all__ += ["AnnoyKNNImputer", "IterativeAnnoyKNNImputer"]
