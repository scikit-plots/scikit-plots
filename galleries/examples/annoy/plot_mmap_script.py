# Authors: Spotify AB
# SPDX-License-Identifier: Apache-2.0

"""
Mmap annoy.AnnoyIndex with examples
=====================================

An example showing the :py:class:`~scikitplot.cexternals.annoy.AnnoyIndex` class.
"""

# %%
from __future__ import print_function

import random; random.seed(0)
import time

# from annoy import AnnoyIndex
# from scikitplot.cexternals.annoy import AnnoyIndex
from scikitplot.cexternals.annoy import Index as AnnoyIndex

# %%
a = AnnoyIndex(
    f=3,
    metric='angular',
)
a.add_item(0, [1, 0, 0])
a.add_item(1, [0, 1, 0])
a.add_item(2, [0, 0, 1])
a.build(-1)
a.save('test.annoy')

# %%
b = AnnoyIndex(
    f=3,
    metric='angular',
)
b.load('test.annoy')

# %%
print(b.get_nns_by_item(0, 100))
print(b.get_nns_by_vector([1.0, 0.5, 0.5], 100))

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: impute
#    plot-type: bar
#    level: beginner
#    purpose: showcase
