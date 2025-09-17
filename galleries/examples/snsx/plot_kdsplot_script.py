"""
plot_kdsplot_script with examples
==========================================

An example showing the :py:func:`~scikitplot.snsx.kdsplot` function
used by a scikit-learn regressor.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Import scikit-plot
import scikitplot.snsx as sp


# %%
import numpy as np; np.random.seed(0)  # reproducibility
import pandas as pd

# Create a DataFrame with predictions
df = pd.DataFrame({
    "y_true": np.random.normal(0.5, 0.1, 100).round(),
    "y_score": np.random.normal(0.5, 0.15, 100),
    "hue": np.random.normal(0.5, 0.4, 100).round(),
})


# %%
p = sp.kdsplot(df, x="y_true", y="y_score", kind="df", n_deciles=10, round_digits=2)
p


# %%
p = sp.kdsplot(df, x="y_true", y="y_score",kind="lift")

# %%
p = sp.kdsplot(df, x="y_true", y="y_score",kind="lift_decile_wise")

# %%
p = sp.kdsplot(df, x="y_true", y="y_score",kind="cumulative_gain")

# %%
p = sp.kdsplot(df, x="y_true", y="y_score",kind="ks_statistic")

# %%
p = sp.kdsplot(df, x="y_true", y="y_score",kind="report", verbose=True)
