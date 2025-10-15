"""
plot_impute with examples
==================================

An example showing the :py:func:`~scikitplot.impute.report` function used
by a scikit-learn classifier.

.. seealso::

    * https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
import time
import numpy as np

from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import (
    load_breast_cancer as data_2_classes,
    load_iris as data_3_classes,
    load_digits as data_10_classes,
)
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split

X_diabetes, y_diabetes = load_diabetes(return_X_y=True)
X_california, y_california = fetch_california_housing(return_X_y=True)

X_diabetes = X_diabetes[:300]
y_diabetes = y_diabetes[:300]
X_california = X_california[:300]
y_california = y_california[:300]

# Generate a sample dataset
X, y = make_regression(n_samples=2500, n_features=20, n_informative=15,
                       n_targets=1, bias=0, noise=0,
                       shuffle=True, coef=False, random_state=0)

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=0
)

def add_missing_values(X_full, y_full, rng):
    n_samples, n_features = X_full.shape

    # Add missing values in 75% of the lines
    missing_rate = 0.75
    n_missing_samples = int(n_samples * missing_rate)

    missing_samples = np.zeros(n_samples, dtype=bool)
    missing_samples[:n_missing_samples] = True

    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)
    X_missing = X_full.copy()
    X_missing[missing_samples, missing_features] = np.nan
    y_missing = y_full.copy()

    return X_missing, y_missing

rng = np.random.RandomState(42)
X_miss_diabetes, y_miss_diabetes = add_missing_values(X_diabetes, y_diabetes, rng)
X_miss_california, y_miss_california = add_missing_values(
    X_california, y_california, rng
)
X_miss_train, y_miss_train = add_missing_values(
    X_train, y_train, rng
)


# %%
from sklearn.ensemble import RandomForestRegressor

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

N_SPLITS = 4

def get_score(X, y, imputer=None):
    regressor = RandomForestRegressor(random_state=0)
    if imputer is not None:
        estimator = make_pipeline(imputer, regressor)
    else:
        estimator = regressor
    scores = cross_val_score(
        estimator, X, y, scoring="neg_mean_squared_error", cv=N_SPLITS
    )
    return scores.mean(), scores.std()

x_labels = []

mses_diabetes = np.zeros(7)
stds_diabetes = np.zeros(7)
mses_california = np.zeros(7)
stds_california = np.zeros(7)
mses_train = np.zeros(7)
stds_train = np.zeros(7)
time_data = np.zeros(7)


# %%
t0 = time.time()
mses_diabetes[0], stds_diabetes[0] = get_score(X_diabetes, y_diabetes)
mses_california[0], stds_california[0] = get_score(X_california, y_california)
mses_train[0], stds_train[0] = get_score(X_train, y_train)
x_labels.append("Full Data")
T = time.time() - t0
print(T)
time_data[0] = T


# %%
t0 = time.time()
imputer = SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)
mses_diabetes[1], stds_diabetes[1] = get_score(
    X_miss_diabetes, y_miss_diabetes, imputer
)
mses_california[1], stds_california[1] = get_score(
    X_miss_california, y_miss_california, imputer
)
mses_train[1], stds_train[1] = get_score(
    X_miss_train, y_miss_train, imputer
)
x_labels.append("Zero Imputation")
T = time.time() - t0
print(T)
time_data[1] = T

# %%
t0 = time.time()
imputer = SimpleImputer(strategy="mean", add_indicator=True)
mses_diabetes[2], stds_diabetes[2] = get_score(
    X_miss_diabetes, y_miss_diabetes, imputer
)
mses_california[2], stds_california[2] = get_score(
    X_miss_california, y_miss_california, imputer
)
mses_train[2], stds_train[2] = get_score(
    X_miss_train, y_miss_train, imputer
)
x_labels.append("Mean Imputation")
T = time.time() - t0
print(T)
time_data[2] = T

# %%
t0 = time.time()
imputer = SimpleImputer(strategy="median", add_indicator=True)
mses_diabetes[3], stds_diabetes[3] = get_score(
    X_miss_diabetes, y_miss_diabetes, imputer
)
mses_california[3], stds_california[3] = get_score(
    X_miss_california, y_miss_california, imputer
)
mses_train[3], stds_train[3] = get_score(
    X_miss_train, y_miss_train, imputer
)
x_labels.append("Median Imputation")
T = time.time() - t0
print(T)
time_data[3] = T

# %%
t0 = time.time()
imputer = IterativeImputer(add_indicator=True)

mses_diabetes[4], stds_diabetes[4] = get_score(
    X_miss_diabetes, y_miss_diabetes, imputer
)
mses_california[4], stds_california[4] = get_score(
    X_miss_california, y_miss_california, make_pipeline(RobustScaler(), imputer)
)
mses_train[4], stds_train[4] = get_score(
    X_miss_train, y_miss_train, make_pipeline(RobustScaler(), imputer)
)
x_labels.append("Iterative Imputation")
T = time.time() - t0
print(T)
time_data[4] = T

# %%
t0 = time.time()
imputer = KNNImputer(add_indicator=True)
mses_diabetes[5], stds_diabetes[5] = get_score(
    X_miss_diabetes, y_miss_diabetes, imputer
)
mses_california[5], stds_california[5] = get_score(
    X_miss_california, y_miss_california, make_pipeline(RobustScaler(), imputer)
)
mses_train[5], stds_train[5] = get_score(
    X_miss_train, y_miss_train, make_pipeline(RobustScaler(), imputer)
)
x_labels.append("KNN Imputation")
T = time.time() - t0
print(T)
time_data[5] = T

# %%
import scikitplot as sp
sp.__version__

# %%
from scikitplot.impute import AnnoyKNNImputer
print(AnnoyKNNImputer.__doc__)

# %%
t0 = time.time()
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, n_neighbors=5, search_k=50)
mses_diabetes[6], stds_diabetes[6] = get_score(
    X_miss_diabetes, y_miss_diabetes, imputer
)
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, n_neighbors=5, search_k=50)
mses_california[6], stds_california[6] = get_score(
    X_miss_california, y_miss_california, make_pipeline(RobustScaler(), imputer)
)
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0)
mses_train[6], stds_train[6] = get_score(
    X_miss_train, y_miss_train, make_pipeline(RobustScaler(), imputer)
)
x_labels.append("AnnoyKNN Imputation")
T = time.time() - t0
print(T)
time_data[6] = T

# %%
mses_diabetes = mses_diabetes * -1
mses_california = mses_california * -1
mses_train = mses_train * -1

# %%
import matplotlib.pyplot as plt

n_bars = len(mses_diabetes)
xval = np.arange(n_bars)

colors = ["r", "g", "b", "orange", "k", "m", "c"]

# plot diabetes results
plt.figure(figsize=(18, 6))
ax1 = plt.subplot(131)
bars1 = []
for j, td in zip(xval, time_data):
    bar = ax1.barh(
        j,
        mses_diabetes[j],
        xerr=stds_diabetes[j],
        color=colors[j],
        alpha=0.6,
        align="center",
        label=str(np.round(td, 3))
    )
    bars1.append(bar)

# Add bar value labels
for bar in bars1:
    ax1.bar_label(bar, fmt="%.3f", label_type='center', padding=-20)

ax1.set_title("Imputation Techniques with Diabetes Data")
ax1.set_xlim(left=np.min(mses_diabetes) * 0.9, right=np.max(mses_diabetes) * 1.1)
ax1.set_yticks(xval)
ax1.set_xlabel("MSE")
ax1.invert_yaxis()
ax1.set_yticklabels(x_labels)

# plot california dataset results
ax2 = plt.subplot(132)
bars2 = []
for j, td in zip(xval, time_data):
    bar = ax2.barh(
        j,
        mses_california[j],
        xerr=stds_california[j],
        color=colors[j],
        alpha=0.6,
        align="center",
        label=str(np.round(td, 3))
    )
    bars2.append(bar)

# Add bar value labels
for bar in bars2:
    ax2.bar_label(bar, fmt="%.3f", label_type='center', padding=-20)

ax2.set_title("Imputation Techniques with California Data")
ax2.set_yticks(xval)
ax2.set_xlabel("MSE")
ax2.invert_yaxis()
ax2.set_yticklabels([""] * n_bars)

# plot train dataset results
ax3 = plt.subplot(133)
bars3 = []
for j, td in zip(xval, time_data):
    bar = ax3.barh(
        j,
        mses_train[j],
        xerr=stds_train[j],
        color=colors[j],
        alpha=0.6,
        align="center",
        label=str(np.round(td, 3))
    )
    bars3.append(bar)

# Add bar value labels
for bar in bars3:
    ax3.bar_label(bar, fmt="%.3f", label_type='center', padding=-20)

ax3.set_title("Imputation Techniques with Train Data")
ax3.set_yticks(xval)
ax3.set_xlabel("MSE")
ax3.invert_yaxis()
ax3.set_yticklabels([""] * n_bars)

plt.legend(title='Time')
plt.tight_layout()
plt.show()

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: impute
#    plot-type: bar
#    level: beginner
#    purpose: showcase
