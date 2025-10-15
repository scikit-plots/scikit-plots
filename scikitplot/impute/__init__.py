# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Transformers for missing value imputation."""

import contextlib

with contextlib.suppress(ImportError):
    from ._annoy_knn import AnnoyKNNImputer

__all__ = [
    "AnnoyKNNImputer",
]
