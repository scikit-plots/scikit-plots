"""
plot_classifier_eval with examples
==================================

An example showing the :py:func:`~scikitplot.api.metrics.plot_classifier_eval` function
used by a scikit-learn classifier.
"""
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.datasets import (
  make_classification,
  load_breast_cancer as data_2_classes,
  load_iris as data_3_classes,
  load_digits as data_10_classes,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict

import numpy as np; np.random.seed(0)  # reproducibility
# importing pylab or pyplot
# import matplotlib as mpl
import matplotlib.pyplot as plt

# Import scikit-plot
import scikitplot as sp

# Load the data
X, y = data_3_classes(return_X_y=True, as_frame=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)

# Create an instance of the LogisticRegression
model = LogisticRegression(max_iter=int(1e5), random_state=0).fit(X_train, y_train)

# Perform predictions
y_val_pred = model.predict(X_val)
y_train_pred = model.predict(X_train)

fig1 = sp.metrics.plot_classifier_eval(
    y_val, y_val_pred, 
    labels=np.unique(y_train),
    figsize=(8, 2),
    title='Val',
);
# plt.show(block=True)
fig2 = sp.metrics.plot_classifier_eval(
    y_train, y_train_pred, 
    labels=np.unique(y_train),
    figsize=(8, 2),
    title='Train',
);

# Save the combined figures as an simple image file
figs = sp.api._utils.save_figure(
    (fig1, fig2),
    to_save=False
);

# Adjust layout to make sure everything fits
plt.tight_layout()

# Save the plot with a filename based on the current script's name
# sp.api._utils.save_plot()

# Display the plot
plt.show(block=True)

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model evaluation
#    plot-type: matrix
#    plot-type: specialty
#    level: beginner
#    purpose: showcase