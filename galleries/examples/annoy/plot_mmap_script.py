# Authors: Spotify AB
# SPDX-License-Identifier: Apache-2.0

"""
Mmap annoy.AnnoyIndex with examples
=====================================

An example showing the :py:class:`~scikitplot.annoy.AnnoyIndex` class.
"""

# %%
from __future__ import print_function

import random; random.seed(0)
import time

# from annoy import AnnoyIndex
# from scikitplot.annoy import AnnoyIndex
from scikitplot.annoy import Index as AnnoyIndex

# %%
idx = AnnoyIndex(
    f=3,
    metric='angular',
)
idx.add_item(0, [1, 0, 0])
idx.add_item(1, [0, 1, 0])
idx.add_item(2, [0, 0, 1])
idx.build(-1)
idx.save('test.annoy')

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

def plot(idx, y=None, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    import scikitplot.cexternals._annoy._plotting as utils

    single = np.zeros(idx.get_n_items(), dtype=int)
    if y is None:
        double = np.random.uniform(0, 1, idx.get_n_items()).round()

    # single vs double
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    alpha = kwargs.pop("alpha", 0.8)
    y2 = utils.plot_annoy_index(
        idx,
        dims = list(range(idx.f)),
        plot_kwargs={"draw_legend": False},
        ax=ax[0],
    )[0]
    utils.plot_annoy_knn_edges(
        idx,
        y2,
        k=1,
        line_kwargs={"alpha": alpha},
        ax=ax[1],
    )

# idx.unbuild()
# idx.build(10)
plot(idx)

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: impute
#    plot-type: bar
#    level: beginner
#    purpose: showcase
