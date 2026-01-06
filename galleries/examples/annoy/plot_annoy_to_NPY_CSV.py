# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""
annoy.Index to NPY or CSV with examples
=======================================

An example showing the :py:class:`~scikitplot.annoy.Index` class.

.. seealso::
    * :py:obj:`~scikitplot.annoy.Index.from_low_level`
    * https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled
"""

# %%
import random; random.seed(0)

# from annoy import Annoy, AnnoyIndex
from scikitplot.annoy import Annoy, AnnoyIndex, Index

print(Index.__doc__)

# %%

import random
from pathlib import Path

random.seed(0)

HERE = Path.cwd().resolve()
OUT = HERE / "../../../scikitplot/annoy/tests" / "test_v2.tree"

f = 10
n = 1000
idx = Index(f, "angular")
for i in range(n):
    v = [random.gauss(0, 1) for _ in range(f)]
    idx.add_item(i, v)

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
idx.build(10)
plot(idx)

# %%

# idx.build(10)
idx.save(str(OUT))
print("Wrote", OUT)
idx

# %%

import random
from scikitplot.utils._time import Timer

n, f = 1_000, 10
X = [[random.gauss(0, 1) for _ in range(f)] for _ in range(n)]
q = [[random.gauss(0, 1) for _ in range(f)]]

q

# %%

# idx = Index().fit(X, feature_names=map("feature_{}".format, range(0,10)))
idx = Index().fit(X, feature_names=map("col_{}".format, range(0,10)))
idx

# %%

idx.feature_names_in_

# %%

idx.get_feature_names_out()

# %%

idx.transform(X[:5])

# %%

idx.transform(X[:5], output_type="item")

# %%

idx.transform(q, output_type="item")

# %%

idx.transform(q, output_type="vector")

# %%

idx.kneighbors(q, n_neighbors=5, output_type="vector")

# %%

idx.kneighbors(X[:5], n_neighbors=5, include_distances=False).shape

# %%

import numpy as np

arr = idx.to_numpy()
arr


# %%

# save, savez
np.save("annoy_vectors.npy", arr)
np.load("annoy_vectors.npy")

# %%

idx.to_scipy_csr()

# %%

idx.to_pandas(id_location="index")

# %%

# Small subset → DataFrame/CSV
df = idx.to_pandas()
df.to_csv("sample.csv", index=False)

# %%
import pandas as pd

pd.read_csv("sample.csv")

# %%

idx.query_by_item(item=999, n_neighbors=10, include_distances=True)

# %%

idx.query_by_vector(v, n_neighbors=10, include_distances=True)

# %%

idx.kneighbors(v, n_neighbors=10, include_distances=True)

# %%

idx.kneighbors_graph(v, n_neighbors=10)

# %%

idx.kneighbors_graph(v, n_neighbors=10).toarray()

# %%
#
# .. tags::
#
#    level: beginner
#    purpose: showcase
