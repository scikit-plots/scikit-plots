"""
annoy with examples
==================================

An example showing the :py:func:`~scikitplot.cexternals.annoy` function.
"""

# Authors: Spotify AB
# SPDX-License-Identifier: Apache-2.0

# %%
# from annoy import AnnoyIndex
from scikitplot.cexternals.annoy import AnnoyIndex

# %%
a = AnnoyIndex(3, 'angular')
a.add_item(0, [1, 0, 0])
a.add_item(1, [0, 1, 0])
a.add_item(2, [0, 0, 1])
a.build(-1)
a.save('test.tree')

# %%
b = AnnoyIndex(3, 'angular')
b.load('test.tree')

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
