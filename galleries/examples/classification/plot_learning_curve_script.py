"""
plot_learning_curve with examples
=================================

An example showing the :py:func:`~scikitplot.api.estimators.plot_learning_curve` function
used by a scikit-learn classifier.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.datasets import (
    load_digits as data_10_classes,
)
from sklearn.linear_model import LogisticRegression

import numpy as np

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
ax = sp.estimators.plot_learning_curve(
    model,
    X,
    y,
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
#    plot-type: learning curve
#    level: beginner
#    purpose: showcase
