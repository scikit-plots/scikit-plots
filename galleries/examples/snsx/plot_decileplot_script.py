"""
plot_decileplot_script with examples
==========================================

An example showing the :py:func:`~scikitplot.snsx.decileplot` function
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

from sklearn.datasets import make_classification
from sklearn.datasets import (
    load_breast_cancer as data_2_classes,
    load_iris as data_3_classes,
    load_digits as data_10_classes,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# %%
# Load the data
# X, y = data_3_classes(return_X_y=True, as_frame=False)
# X, y = data_2_classes(return_X_y=True, as_frame=False)

# Generate a sample dataset
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15,
                          n_redundant=2, n_classes=2, n_repeated=0,
                          class_sep=1.5, flip_y=0.01, weights=[0.85, 0.15],
                          random_state=0)

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=0
)
np.unique(y)

# %%
# Create an instance of the LogisticRegression
model = (
    LogisticRegression(
        # max_iter=int(1e5),
        # C=10,
        # penalty='l1',
        # solver='liblinear',
        class_weight='balanced',
        random_state=0
    )
    .fit(X_train, y_train)
)
# Perform predictions
y_val_prob = model.predict_proba(X_val)
# Create a DataFrame with predictions
df = pd.DataFrame({
    "y_true": y_val==1,  # target class (0,1,2)
    "y_score": y_val_prob[:, 1],  # target class (0,1,2)
    # np.argmax
    "y_pred": y_val_prob[:, 1] > 0.5,  # target class (0,1,2)
    # "y_true": np.random.normal(0.5, 0.1, 100).round(),
    # "y_score": np.random.normal(0.5, 0.15, 100),
    # "hue": np.random.normal(0.5, 0.4, 100).round(),
})
df



# %%
p = sp.decileplot(
    df,
    x="y_true",
    y="y_score",
    kind="df",
    n_deciles=10,
    digits=4,
    verbose=True,
)
p.T
# p.columns.tolist()
# p[["decile", "cnt_resp", "cnt_resp_wiz", "cum_resp_pct", "cum_resp_wiz_pct"]]
# p.iloc[:, range(9, 23)]
# p.iloc[:, [11, 12, 12, 14]]


# %%
p = sp.decileplot(df, x="y_true", y="y_score", kind="cumulative_lift", n_deciles=10, annot=True)

# %%
p = sp.decileplot(df, x="y_true", y="y_score", kind="decile_wise_lift", n_deciles=10, annot=True)

# %%
p = sp.decileplot(df, x="y_true", y="y_score", kind="cumulative_gain", n_deciles=10, annot=True)

# %%
p = sp.decileplot(df, x="y_true", y="y_score", kind="cumulative_response", n_deciles=10, annot=True)

# %%
p = sp.decileplot(df, x="y_true", y="y_score", kind="decile_wise_gain", n_deciles=10, annot=True)

# %%
p = sp.decileplot(df, x="y_true", y="y_score", kind="ks_statistic", n_deciles=10, annot=True)

# %%
# fig, ax = plt.subplots(figsize=(10, 10))
p = sp.decileplot(
    df,
    x="y_true",
    y="y_score",
    kind="report",
    n_deciles=10,
    digits=4,
    annot=True,
    verbose=True,
)

# %%
# fig, ax = plt.subplots(figsize=(10, 10))
p = sp.decileplot(
    df,
    x="y_true",
    y="y_score",
    kind="report",
    n_deciles=10,
    digits=6,
    fmt='.4g'
)

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model evaluation
#    plot-type: line
#    plot-type: decile
#    level: beginner
#    purpose: showcase
