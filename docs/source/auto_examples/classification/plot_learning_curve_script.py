"""
plot_learning_curve with examples
=================================

An example showing the :py:func:`~scikitplot.api.estimators.plot_learning_curve` function
used by a scikit-learn classifier.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from sklearn.datasets import (
    load_digits as data_10_classes,
)
from sklearn.linear_model import LogisticRegression

np.random.seed(0)  # reproducibility
# importing pylab or pyplot
import matplotlib.pyplot as plt

# Import scikit-plot
import scikitplot as sp

# Load the data
X, y = data_10_classes(return_X_y=True, as_frame=False)

# Create an instance of the LogisticRegression
model = LogisticRegression(max_iter=int(1e5), random_state=0)

# Plot!
ax = sp.estimators.plot_learning_curve(model, X, y)

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
#    plot-type: line
#    plot-type: learning curve
#    level: beginner
#    purpose: showcase
