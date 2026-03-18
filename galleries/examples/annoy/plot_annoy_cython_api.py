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

import struct; print(struct.calcsize('P')*8)

# %%

from scikitplot.annoy._annoy import Index  # cython Total: 160 concrete index-data types
from scikitplot.annoy import AnnoyIndex  # Cpp 1 concrete index-data type (uint64 index float32 data)

# 32 bit int index
i = Index(10, "angular")
i.load(f"./test.tree")
# This might change in the future if we change the search algorithm, but in that case let's update the test
result1 = i.get_nns_by_item(0, 10)

# 64 bit uint index
j = AnnoyIndex(10, "angular")
for idx in range(i.get_n_items()):
    j.add_item(idx, i.get_item(idx))
j.build(10)
j.save("test64.tree")
result2 = i.get_nns_by_item(0, 10)

i = AnnoyIndex(10, "angular")
i.load(f"./test64.tree")
# This might change in the future if we change the search algorithm, but in that case let's update the test
result3 = i.get_nns_by_item(0, 10)

result1, result2, result3

# %%

# from annoy import Annoy, AnnoyIndex
# from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
# from scikitplot.annoy import Annoy, AnnoyIndex, Index
from scikitplot.annoy._annoy import Index, AnnoyIndex

print(Index.__doc__)

# %%

import sys
import scikitplot

# spotify/annoy Backward compatibility helper
sys.modules["annoy"] = scikitplot.annoy._annoy  # now `import annoy` will resolve to your module

import annoy; print(annoy.AnnoyIndex.__doc__)

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
#    model-workflow: vector-db
#    level: beginner
#    purpose: showcase
