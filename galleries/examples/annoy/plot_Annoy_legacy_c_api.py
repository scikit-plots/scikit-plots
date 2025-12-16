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

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex(0)
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx)
print(idx.info())
type(idx)
# help(idx.info)


# %%

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex(f=3)
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
print(idx)
# idx.on_disk_path is None


# %%

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex(f=3, metric="angular")
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)


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
idx

# %%

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex(100, metric="angular")
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)
idx.on_disk_build("annoy_test1.annoy"), idx#.on_disk_path
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
print(idx)
print(idx.info())
# help(idx.build)

# %%

idx.unbuild()
print(idx)

# %%

idx.build(10)
print(idx)


# %%

# =============================================================
# 1. Construction
# =============================================================
idx = AnnoyIndex(0, metric="angular")
print("Index dimension:", idx.f)
print("Metric         :", idx.metric)


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
print(idx)
print(idx.info())
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
idx.save("annoy_test1.annoy")
print(idx)

print("Loading...")
idx2 = AnnoyIndex(100, metric='angular').load("annoy_test1.annoy")
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
print(idx)
print(new_idx)


# %%

idx.unload()
print(idx)


# %%

# idx.build(10)
idx.load("annoy_test1.annoy")
print(idx)

# %%

idx.info()

# %%

idx.get_nns_by_item(0, 10), len(idx.get_item_vector(0))

# %%
#
# .. tags::
#
#    level: beginner
#    purpose: showcase
