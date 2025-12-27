# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
annoy.Annoy legacy c-api with examples
==========================================

An example showing the :py:class:`~scikitplot.annoy.Annoy`, :py:class:`~scikitplot.annoy.AnnoyIndex` class.

.. seealso::
    * :py:obj:`~scikitplot.annoy.Index.from_low_level`
    * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
"""

# %%

import numpy as np
import random; random.seed(0)

# from annoy import Annoy, AnnoyIndex
from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
# from scikitplot.annoy import Annoy, AnnoyIndex, Index

print(AnnoyIndex.__doc__)

# %%

import sys

# TODO: change this import to wherever your modified AnnoyIndex lives
# e.g. scikitplot.cexternals._annoy or similar
import scikitplot.cexternals._annoy as annoy
# from scikitplot import annoy

sys.modules["annoy"] = annoy  # now `import annoy` will resolve to your module

import annoy

print(annoy.__doc__)

# %%

import gc

x = annoy.Annoy()
assert gc.is_tracked(x) in (True, False)  # must NOT crash
gc.is_tracked(x)

# %%

x = annoy.Annoy()
del x
import gc; gc.collect()

# %%

Annoy(), \
Annoy(None), \
Annoy(None, None), \
Annoy(1), \
Annoy(1,".", seed=1, verbose=1), \
Annoy(1,"@", seed=1, verbose=1).set_verbose(2).set_seed(2)

# %%

Annoy()

# %%

AnnoyIndex()

# %%

print(AnnoyIndex())

# %%

# from IPython.display import display
# from IPython.core.display import HTML
# display(HTML('<h1>Hello, world!</h1>'))

# display(AnnoyIndex())

# %%

AnnoyIndex().info()

# %%

AnnoyIndex().repr_info()

# %%

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex()
idx = AnnoyIndex(None, None)
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx.info())
print(idx)
print(type(idx))
idx
# help(idx.info)


# %%

dir(idx)

# %%

# AttributeError: readonly attribute
# idx._metric_id = 1
idx._f, idx._metric_id, idx._on_disk_path

# %%

idx.f, idx.metric, idx.on_disk_path
idx

# %%

idx.metric = "dot"
idx

# %%

idx.f, idx.metric, idx.on_disk_path

# %%

type(idx)

# %%

# =============================================================
# 1. Construction
# =============================================================
# idx = AnnoyIndex(f=3)
idx.add_item(0, [1, 0, 0])
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx.info())
print(idx)
idx
# idx.on_disk_path is None


# %%

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex(f=3, metric="angular")
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx.info())
print(idx)
idx


# %%

# =============================================================
# 2. Add items
# =============================================================
idx.add_item(0, [0, 0, 0])

idx.add_item(1, [1, 0, 0])
idx.add_item(2, [0, 1, 0])
idx.add_item(3, [0, 0, 1])

idx.add_item(4, [2, 0, 0])
idx.add_item(5, [0, 2, 0])
idx.add_item(6, [0, 0, 2])

idx.add_item(7, [3, 0, 0])
idx.add_item(8, [0, 3, 0])
idx.add_item(9, [0, 0, 3])

idx.add_item(10, [4, 0, 0])
idx.add_item(11, [0, 4, 0])
idx.add_item(12, [0, 0, 4])

idx.add_item(12, [4, 0, 0])
idx.add_item(13, [0, 4, 0])
idx.add_item(14, [0, 0, 4])

print("Number of items:", idx.get_n_items())
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx.info())
print(idx)
idx

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

idx.unbuild()
idx.build(100)
plot(idx)

# %%

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex(100, metric="angular")
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
idx.on_disk_build("annoy_test_1.annoy"), idx#.on_disk_path
# help(idx.on_disk_build)

# %%

# =============================================================
# 2. Add items
# =============================================================
f=100
n=1000
for i in range(n):
    if(i % (n//10) == 0): print(f"{i} / {n} = {1.0 * i / n}")
    # v = []
    # for z in range(f):
    #     v.append(random.gauss(0, 1))
    v = [random.gauss(0, 1) for _ in range(f)]
    idx.add_item(i, v)

print("Number of items:", idx.get_n_items())
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx.info())
print(idx)
idx


# %%

# =============================================================
# 3. Build index
# =============================================================
idx.build(10)
print("Trees:", idx.get_n_trees())
print("Memory usage:", idx.memory_usage(), "bytes")
print(idx.info())
print(idx)
idx
# help(idx.build)

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

print(idx.info())
print(idx)
idx

# %%

import numpy as np
import scikitplot.cexternals._annoy._plotting as utils

# labels = np.random.uniform(0, 1, idx.get_n_items()).round()
labels = np.zeros(idx.get_n_items(), dtype=int)

nth = 1
ids_subset = np.arange(0, idx.get_n_items(), nth)  # every 1th item

y2, ids, ax = utils.plot_annoy_index(
    idx,
    labels=labels[ids_subset],    # must match ids length
    ids=ids_subset,
    projection="pca",
    plot_kwargs={"draw_legend": True},
)

utils.plot_annoy_knn_edges(idx, y2, ids=ids, k=nth)

# %%

import numpy as np
import scikitplot.cexternals._annoy._plotting as utils

# labels = np.random.uniform(0, 1, idx.get_n_items()).round()
labels = np.zeros(idx.get_n_items(), dtype=int)

X, ids = utils.annoy_index_to_array(idx)  # X: (n_items, f), ids: (n_items,)
ax = utils.plot(
    x=X,
    y=labels,  # must match y2 rows labels[ids]
    title="2D embedding",
    figsize=(7, 7),
    draw_legend=True,
    scatter_kwargs={"s": 6, "alpha": 0.7},
)


# %%

idx.unbuild()
print(idx.info())
print(idx)
idx

# %%

idx.build(10)
print(idx.info())
print(idx)
idx


# %%

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex(0, metric="angular")
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx.info())
print(idx)
idx

# %%

# =============================================================
# 2. Add items
# =============================================================
f=100
n=1000
for i in range(n):
    if(i % (n//10) == 0): print(f"{i} / {n} = {1.0 * i / n}")
    # v = []
    # for z in range(f):
    #     v.append(random.gauss(0, 1))
    v = [random.gauss(0, 1) for _ in range(f)]
    idx.add_item(i, v)

print("Number of items:", idx.get_n_items())
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx.info())
print(idx)
idx


# %%

# =============================================================
# 3. Build index
# =============================================================
idx.build(10)
print("Trees:", idx.get_n_trees())
print("Memory usage:", idx.memory_usage(), "bytes")
print(idx.info())
print(idx)
idx
# help(idx.get_n_trees)


# %%

# =============================================================
# 4. Query — return NNSResult
# =============================================================
res = idx.get_nns_by_item(
    0,
    5,
    # search_k = -1,
    include_distances=True,
)

print(res)


# %%

# =============================================================
# 8. Query using vector
# =============================================================
res2 = idx.get_nns_by_vector(
    [random.gauss(0, 1) for _ in range(f)],
    5,
    include_distances=True
)
print("\nQuery by vector:", res2)


# %%

# =============================================================
# 9. Low-level (non-result) mode
# =============================================================
items = idx.get_nns_by_item(0, 2, include_distances=False)
print("\nLow-level items only:", items)

items_low, d_low = idx.get_nns_by_item(0, 2, include_distances=True)
print("Low-level tuple return:", items_low, d_low)


# %%

# =============================================================
# 10. Persistence
# =============================================================
print("\n=== Saving with binary annoy ===")
print(idx)
idx.save("annoy_test_1.annoy")
print(idx)

print("Loading...")
idx2 = AnnoyIndex(100, metric='angular').load("annoy_test_1.annoy")
print("Loaded index:", idx2)


# %%

import joblib

joblib.dump(idx, "test.joblib")
a = joblib.load("test.joblib")
a

# %%

a.info(), a.get_n_items(), a.get_n_trees()

# %%

np.array_equal(a.get_item_vector(0), idx2.get_item_vector(0))

# %%

np.array_equal(a.get_item_vector(0), idx.get_item_vector(0))

# %%

# =============================================================
# 11. Raw serialize / deserialize
# =============================================================
print("\n=== Raw serialize ===")
buf = idx.serialize()
new_idx = AnnoyIndex(100, metric='angular')
new_idx.deserialize(buf)
print("Deserialized index n_items:", new_idx.get_n_items())
print(idx.info())
print(idx)
idx


# %%

idx.unload()
print(idx.info())
print(idx)
idx


# %%

# idx.build(10)
idx.load("annoy_test_1.annoy")
print(idx.info())
print(idx)
idx

# %%

idx.info()

# %%

idx.get_nns_by_item(0, 10), len(idx.get_item_vector(0))

# %%

import random
from scikitplot.utils._time import Timer

n, f = 10_000, 1_000
X = [[random.gauss(0, 1) for _ in range(f)] for _ in range(n)]
q = [[random.gauss(0, 1) for _ in range(f)]]

# %%

with Timer("set_params"):
    for m in ["angular", "l1", "l2", ".", "hamming"]:
        idx = AnnoyIndex().set_params(metric=m).fit(X)
        print(m, idx.transform(q))

# %%

with Timer("rebuild"):
    base = AnnoyIndex(metric="l2").fit(X)

    for m in ["angular", "l1", "l2", "dot", "hamming"]:
        idx_m = base.rebuild(metric=m)          # rebuild-from-index
        print(m, idx_m.transform(q))            # no .fit(X) here


# %%
#
# .. tags::
#
#    level: beginner
#    purpose: showcase
