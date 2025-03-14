"""
plot_silhouette with examples
=============================

An example showing the :py:func:`~scikitplot.api.metrics.plot_silhouette` function
used by a scikit-learn clusterer.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import (
    load_iris as data_3_classes,
)
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
model = KMeans(n_clusters=4, random_state=0)

cluster_labels = model.fit_predict(X_train)

# Plot!
ax = sp.metrics.plot_silhouette(X_train, cluster_labels)

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
#    model-type: clustering
#    model-type: k-means
#    model-workflow: model evaluation
#    plot-type: bar
#    plot-type: silhouette plot
#    level: beginner
#    purpose: showcase
