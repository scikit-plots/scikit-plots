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
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)  # reproducibility
import pandas as pd

from sklearn.datasets import (
    load_breast_cancer as data_2_classes,
    # load_iris as data_3_classes,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# %%
# Load the data
# X, y = data_3_classes(return_X_y=True, as_frame=False)
X, y = data_2_classes(return_X_y=True, as_frame=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
np.unique(y)

# %%
# Create an instance of the LogisticRegression
model = (
    LogisticRegression(max_iter=int(1e5), random_state=0)
    .fit(X_train, y_train)
)
# Perform predictions
y_val_prob = model.predict_proba(X_val)
# Create a DataFrame with predictions
df = pd.DataFrame({
    "y_true": y_val==1,  # target class (0,1,2)
    "y_score": y_val_prob[:, 1],  # target class (0,1,2)
    # "y_true": np.random.normal(0.5, 0.1, 100).round(),
    # "y_score": np.random.normal(0.5, 0.15, 100),
    # "hue": np.random.normal(0.5, 0.4, 100).round(),
})



# %%
p = sp.kdsplot(
    df,
    x="y_true",
    y="y_score",
    kind="df",
    n_deciles=10,
    round_digits=4,
    verbose=True,
)
p
# p.columns.tolist()
# p[["decile", "cnt_resp", "cnt_resp_wiz", "cum_resp_pct", "cum_resp_wiz_pct"]]
p.iloc[:, range(9, 23)]
# p.iloc[:, [11, 12, 12, 14]]


# %%
p = sp.kdsplot(df, x="y_true", y="y_score", kind="cumulative_lift", n_deciles=10)

# %%
p = sp.kdsplot(df, x="y_true", y="y_score", kind="decile_wise_lift", n_deciles=10)

# %%
p = sp.kdsplot(df, x="y_true", y="y_score", kind="cumulative_gain", n_deciles=10)

# %%
p = sp.kdsplot(df, x="y_true", y="y_score", kind="cumulative_response", n_deciles=10)

# %%
p = sp.kdsplot(df, x="y_true", y="y_score", kind="decile_wise_gain", n_deciles=10)

# %%
p = sp.kdsplot(df, x="y_true", y="y_score", kind="ks_statistic", n_deciles=10)

# %%
fig, ax = plt.subplots(figsize=(10, 10))
p = sp.kdsplot(
    df,
    x="y_true",
    y="y_score",
    kind="report",
    n_deciles=10,
    round_digits=6,
    verbose=True,
)


# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model evaluation
#    plot-type: line
#    plot-type: cum-gain curve
#    level: beginner
#    purpose: showcase
