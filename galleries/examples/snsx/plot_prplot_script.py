"""
plot_prplot_script with examples
==========================================

An example showing the :py:func:`~scikitplot.snsx.prplot` function
used by a scikit-learn regressor.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
np.random.seed(0)  # reproducibility
import pandas as pd

# Import scikit-plot
import scikitplot.snsx as sp

df = pd.DataFrame({
    "true": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]*10,
    "prob": np.random.lognormal(0.0,1.0,100),
    "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"]*10
})

# PR Curve
ax = sp.prplot(x=df.true, y=df.prob, hue=df.group)
# ax = sp.rocplot(df, x="true", y="prob", label=f"classA")
