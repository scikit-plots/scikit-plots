"""
plot_feature_importances with examples
======================================

An example showing the :py:func:`~scikitplot.api.estimators.plot_feature_importances` function
used by a scikit-learn classifier.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sklearn.datasets import (
    load_iris as data_3_classes,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.random.seed(0)  # reproducibility
# importing pylab or pyplot
import matplotlib.pyplot as plt

# Import scikit-plot
import scikitplot as sp

# Load the data
X, y = data_3_classes(return_X_y=True, as_frame=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)

# Create an instance of the LogisticRegression
model = RandomForestClassifier(random_state=0).fit(X_train, y_train)

# Plot!
ax, features = sp.estimators.plot_feature_importances(
    model,
    feature_names=["petal length", "petal width", "sepal length", "sepal width"],
)

# Adjust layout to make sure everything fits
plt.tight_layout()

# Save the plot with a filename based on the current script's name
# sp.api._utils.save_plot()

# Display the plot
plt.show(block=True)

# %%
#
# .. admonition:: References
#
#    The use of the following functions, methods, classes and modules is shown
#    in this example:
#
#    - https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model evaluation
#    plot-type: bar
#    level: beginner
#    purpose: showcase
