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

print(AnnoyIndex.__doc__)

# %%

import random
from pathlib import Path

random.seed(0)

HERE = Path.cwd().resolve()
OUT = HERE / "../../../scikitplot/annoy/tests" / "test_v2.tree"

f = 10
n = 1000
idx = AnnoyIndex(f, "angular")
for i in range(n):
    idx.add_item(i, [random.gauss(0, 1) for _ in range(f)])

idx.build(10)
idx.save(str(OUT))
print("Wrote", OUT)
idx

# %%
# Small subset → DataFrame/CSV
df = idx.to_dataframe(start=0, stop=1000)
df.to_csv("sample.csv", index=False)

# %%
import pandas as pd

pd.read_csv("sample.csv")

# %%
# Streaming CSV (warning: huge)
idx.to_csv("annoy_vectors.csv", start=0, stop=100_000)

# %%
import pandas as pd

pd.read_csv("annoy_vectors.csv")

# %%
# Large export → memory-safe .npy
# Exports items [0, n_items) into a memmapped .npy
idx.save_vectors_npy("annoy_vectors.npy")

# %%
import numpy as np

np.load("annoy_vectors.npy")

# %%
# Range-only export (strict, sized)
idx.save_vectors_npy("chunk_0_1m.npy", start=0, stop=1_000_000)

# %%
import numpy as np

np.load("chunk_0_1m.npy")


# %%
#
# .. tags::
#
#    level: beginner
#    purpose: showcase
