"""
plot_learning_curve with examples
=================================

An example showing the :py:func:`~scikitplot.api.estimators.plot_learning_curve` function
used by a scikit-learn classifier.
"""

# Authors: scikit-plots developers
# License: MIT

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
import numpy as np; np.random.seed(0)
# importing pylab or pyplot
import matplotlib.pyplot as plt

# Import scikit-plot
import scikitplot as skplt

# Load the data
X, y = data_10_classes(return_X_y=True, as_frame=False)

# Create an instance of the LogisticRegression
model = LogisticRegression(max_iter=int(1e5), random_state=0)

# Plot!
ax = skplt.estimators.plot_learning_curve(
    model, X, y
);

# Adjust layout to make sure everything fits
plt.tight_layout()

# Save the plot with a filename based on the current script's name
skplt.utils.save_current_plot()

# Display the plot
plt.show(block=True)

# %%
#
# .. tags::
#
#    model-type: Classification
#    model-workflow: Model Evaluation
#    plot-type: line
#    plot-type: learning-curve
#    level: beginner
#    purpose: showcase