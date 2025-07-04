"""
plot_calibration with examples
====================================

An example showing the :py:func:`~scikitplot.api.metrics.plot_calibration` function
used by a scikit-learn classifier.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# run: Python scripts and shows any outputs directly in the notebook.
# %run ./examples/calibration/plot_calibration_script.py

# %%
# Import scikit-plots
# ------------------------

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

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
X, y = make_classification(
    n_samples=100000,
    n_features=20,
    n_informative=4,
    n_redundant=2,
    n_repeated=0,
    n_classes=3,
    n_clusters_per_class=2,
    random_state=0,
)
X_train, y_train, X_val, y_val = X[:1000], y[:1000], X[1000:], y[1000:]

# %%
# Model Training
# --------------

# Create an instance of the LogisticRegression
lr_probas = (
    LogisticRegression(max_iter=int(1e5), random_state=0)
    .fit(X_train, y_train)
    .predict_proba(X_val)
)
nb_probas = GaussianNB().fit(X_train, y_train).predict_proba(X_val)
svc_scores = LinearSVC().fit(X_train, y_train).decision_function(X_val)
svc_isotonic = (
    CalibratedClassifierCV(LinearSVC(), cv=2, method="isotonic")
    .fit(X_train, y_train)
    .predict_proba(X_val)
)
svc_sigmoid = (
    CalibratedClassifierCV(LinearSVC(), cv=2, method="sigmoid")
    .fit(X_train, y_train)
    .predict_proba(X_val)
)
rf_probas = (
    RandomForestClassifier(random_state=0).fit(X_train, y_train).predict_proba(X_val)
)

probas_dict = {
    LogisticRegression(): lr_probas,
    # GaussianNB(): nb_probas,
    "LinearSVC() + MinMax": svc_scores,
    "LinearSVC() + Isotonic": svc_isotonic,
    "LinearSVC() + Sigmoid": svc_sigmoid,
    # RandomForestClassifier(): rf_probas,
}
probas_dict

# %%
# Plot!
# ------------------------

# Plot!
fig, ax = plt.subplots(figsize=(12, 6))
ax = sp.metrics.plot_calibration(
    y_val,
    y_probas_list=probas_dict.values(),
    estimator_names=probas_dict.keys(),
    ax=ax,
    save_fig=True,
    save_fig_filename="",
    overwrite=True,
    add_timestamp=True,
    verbose=True,
)

# %%
#
# .. admonition:: Interpretation
#
#     Primary Use: Evaluating probabilistic classifiers by comparing predicted probabilities to observed frequencies of the positive class.
#
#     Goal: To assess how well the predicted probabilities align with the actual outcomes, identifying if a model is well-calibrated, overconfident, or underconfident.
#
#     Typical Characteristics:
#
#     - X-axis: Predicted probability (e.g., in bins from 0 to 1).
#     - Y-axis: Observed frequency of the positive class within each bin.
#     - Reference line (diagonal at 45°): Represents perfect calibration, where predicted probabilities match observed frequencies.
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model evaluation
#    component: fitted model
#    plot-type: line
#    plot-type: calibration plot
#    level: beginner
#    purpose: showcase
