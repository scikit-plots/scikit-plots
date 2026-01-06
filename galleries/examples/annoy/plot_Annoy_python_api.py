# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
annoy.Index python-api with examples
=====================================

An example showing the :py:class:`~scikitplot.annoy.Index` class.

.. seealso::
    * :py:obj:`~scikitplot.annoy.Index.from_low_level`
    * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
"""

# %%

import numpy as np
import random; random.seed(0)

# from annoy import Annoy, AnnoyIndex
# from scikitplot.cexternals._annoy import Annoy, AnnoyIndex
from scikitplot.annoy import Annoy, AnnoyIndex, Index

print(Annoy.__doc__)

# %%

print(Index.__doc__)

# %%

from scikitplot import annoy

annoy.__version__, dir(annoy), dir(annoy.Annoy)

# %%

import sys

# TODO: change this import to wherever your modified AnnoyIndex lives
# e.g. scikitplot.cexternals._annoy or similar
# import scikitplot.cexternals._annoy as annoy
from scikitplot import annoy

sys.modules["annoy"] = annoy  # now `import annoy` will resolve to your module

import annoy

print(annoy.__doc__)

# %%

AnnoyIndex()

# %%

Index()

# %%

# =============================================================
# 1. Construction
# =============================================================
idx = Index()
idx = Index(None, None)
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
idx = Index(f=3, metric="angular")
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

from scikitplot import annoy as a

print(a.Annoy)          # same
print(a.AnnoyIndex)     # same
print(a.Index)          # should show <class '..._base.Index'>

print(isinstance(idx, a.Annoy))
print(isinstance(idx, a.AnnoyIndex))
print(isinstance(idx, a.Index))

print(type(idx))
print(idx.__class__.__module__)
print(idx.__class__.__mro__)


# %%

# =============================================================
# 1. Construction
# =============================================================
idx = Index(f=3, metric="angular")
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx.info())
print(idx)
idx

# %%

# =============================================================
# 2. Add items
# =============================================================
idx.add_item(0, [1, 0, 0])
idx.add_item(1, [0, 1, 0])
idx.add_item(2, [0, 0, 1])

print("Number of items:", idx.get_n_items())
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)

# %%

# =============================================================
# 1. Construction
# =============================================================
idx = Index(100, metric="angular")
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
idx.on_disk_build("annoy_test_2.annoy")
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
print(idx)


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
idx = Index(0, metric="angular")
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
print(idx)


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
# 4. Query — return
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
print(idx.info())
print(idx)
idx
idx.save("annoy_test_2.annoy")
print(idx.info())
print(idx)
idx

print("Loading...")
idx2 = Index(100, metric='angular').load("annoy_test_2.annoy")
print("Loaded index:", idx2)

# %%

import joblib

joblib.dump(idx2, "test.joblib")
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
new_idx = Index(100, metric='angular')
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
idx.load("annoy_test_2.annoy")
print(idx)
type(idx)

# %%

# joblib
import joblib

joblib.dump(idx, "test.joblib"), joblib.load("test.joblib")

# %%

from scikitplot import annoy as a

f = 10
idx = a.AnnoyIndex(f, "angular")

# Distinct non-zero content so we can see mismatches clearly
for i in range(20):
    idx.add_item(i, [float(i)] * f)
idx.build(10)
type(idx)

# %%

from scikitplot import annoy as a

# Legacy Support
idx = a.Index.from_low_level(idx)

import joblib
joblib.dump(idx, "test.joblib")
type(idx)


# %%

print(idx.info())
print(idx)
idx

# %%

idx.get_nns_by_item(0, 10), len(idx.get_item_vector(0))

# %%

import random
from scikitplot.utils._time import Timer

n, f = 1_000_000, 10
X = [[random.gauss(0, 1) for _ in range(f)] for _ in range(n)]
q = [[random.gauss(0, 1) for _ in range(f)]]

# %%

# idx = Index().fit(X, feature_names=map("feature_{}".format, range(0,10)))
idx = Index().fit(X, feature_names=map("col_{}".format, range(0,10)))
idx

# %%

idx.feature_names_in_

# %%

idx.transform(X[:5], include_distances=True, return_labels=True)

# %%

with Timer("set_params"):
    for m in ["angular", "l1", "l2", ".", "hamming"]:
        idx = Index().set_params(metric=m).fit(X)
        print(m, idx.transform(q))

# %%

with Timer("rebuild"):
    base = Index(metric="l2").fit(X)

    for m in ["angular", "l1", "l2", "dot", "hamming"]:
        idx_m = base.rebuild(metric=m)          # rebuild-from-index
        print(m, idx_m.transform(q))            # no .fit(X) here


# %%
#
# .. tags::
#
#    level: beginner
#    purpose: showcase
