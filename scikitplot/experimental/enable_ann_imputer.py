# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Enables ANNImputer

The API and results of this estimator might change without any deprecation
cycle.

Importing this file dynamically sets :py:class:`~.ANNImputer`
as an attribute of the impute module.

Examples
--------
Explicitly require this experimental feature

>>> from scikitplot.experimental import enable_ann_imputer  # noqa

Now you can import normally from impute

>>> from scikitplot.impute import ANNImputer
"""
from .. import impute
from ..impute._ann import ANNImputer  #, AnnoyKNNImputer

# use settattr to avoid mypy errors when monkeypatching
setattr(impute, "ANNImputer", ANNImputer)
# setattr(impute, "AnnoyKNNImputer", AnnoyKNNImputer)

impute.__all__ += [
    "ANNImputer",
    # "AnnoyKNNImputer",
]

__all__ = []
