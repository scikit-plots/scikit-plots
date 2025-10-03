"""
plot_roc_curve with examples
============================

An example showing the :py:func:`~scikitplot.api.metrics.plot_roc_curve` function
used by a scikit-learn classifier.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Import scikit-plots
# ------------------------

from sklearn.datasets import (
    load_digits as data_10_classes,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import numpy as np

np.random.seed(0)  # reproducibility
# importing pylab or pyplot
import matplotlib.pyplot as plt

# Import scikit-plot
import scikitplot as sp

# %%
# Loading the dataset
# ------------------------

# Load the data
X, y = data_10_classes(return_X_y=True, as_frame=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# Model Training
# --------------

# Create an instance of the LogisticRegression
model = LogisticRegression(max_iter=1, random_state=0).fit(X_train, y_train)

# Perform predictions
y_val_prob = model.predict_proba(X_val)

# %%
# Plot!
# ------------------------

# Plot!
ax = sp.metrics.plot_roc(
    y_val,
    y_val_prob,
    save_fig=True,
    save_fig_filename="",
    # overwrite=True,
    add_timestamp=True,
    # verbose=True,
)

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model evaluation
#    plot-type: line
#    plot-type: auc
#    level: beginner
#    purpose: showcase
