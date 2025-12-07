# Authors: Spotify AB
# SPDX-License-Identifier: Apache-2.0

"""
Simple annoy.AnnoyIndex with examples
=======================================

An example showing the :py:class:`~scikitplot.annoy.AnnoyIndex` class.
"""

# %%
# from annoy import AnnoyIndex
# from scikitplot.annoy import AnnoyIndex
from scikitplot.annoy import Index as AnnoyIndex

a = AnnoyIndex(
    f=3,
    metric='angular',
)
a.add_item(0, [1, 0, 0])
a.add_item(1, [0, 1, 0])
a.add_item(2, [0, 0, 1])
a.build(-1)

print(a.get_nns_by_item(0, 100))
print(a.get_nns_by_vector([1.0, 0.5, 0.5], 100))

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: impute
#    plot-type: bar
#    level: beginner
#    purpose: showcase
