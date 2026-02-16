# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Index (cython) python-api with examples
=======================================

An example showing the :py:class:`~scikitplot.annoy._annoy.Index` class.

.. important::

  Some parameters are placeholders only and are not processed::

  * dtype
  * index_dtype
  * wrapper_dtype
  * random_dtype
"""

# %%

import numpy as np
import random; random.seed(0)
from pprint import pprint

# from annoy import Annoy, AnnoyIndex
# from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
# from scikitplot.annoy import Annoy, AnnoyIndex, Index
from scikitplot.annoy._annoy import Index

print(Index.__doc__)

# %%

index = Index()
index

# %%

index.set_params(**index.get_params())

# %%

a = index.clone()
a

# %%

Index(10, metric= '.')

# %%

Index(10, metric= 'l1')

# %%

Index(10, metric= 'l2')

# %%

Index(10, metric= 'hamming')

# %%

import numpy as np

# Create index
index = Index(128)

# Add normalized vectors
for i in range(1000):
    v = np.random.randn(128)
    v = v / np.linalg.norm(v)  # Normalize
    index.add_item(i, v)

# Build and query
index.build(10)
neighbors, distances = index.get_nns_by_item(0, 10, include_distances=True)
neighbors, distances

# %%

index.get_params()

# %%

with index.clone() as idx:
    pprint(idx.get_state(), sort_dicts=False)

# %%
#
# .. tags::
#
#    level: beginner
#    purpose: showcase
