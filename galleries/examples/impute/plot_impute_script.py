"""
annoy impute with examples
==================================

Examples related to the :py:class:`~scikitplot.impute.AnnoyKNNImputer` class
with a scikit-learn regressor (e.g., :py:class:`~sklearn.linear_model.LinearRegression`) instance.

.. seealso::

    * https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause


# %%
import time
import numpy as np; np.random.seed(0)  # reproducibility

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
def make_realistic_regression(
    n_samples=3000,
    n_features=20,
    n_informative=15,
    noise=10.0,
    bias=10.0,
    corr_strength=0.15,
    nonlinear_strength=0.5,
    outlier_fraction=0.05,
    missing_fraction=0.0,
    categorical_features=6,
    random_state=None,
):
    """
    Generate a more realistic synthetic regression dataset.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of numerical features.
    n_informative : int
        Number of informative (truly predictive) features.
    noise : float
        Standard deviation of Gaussian noise added to the target.
    bias : float
        Constant bias term in target generation.
    corr_strength : float
        Std. dev. of correlated noise added to create redundancy.
    nonlinear_strength : float
        Strength of nonlinear feature transformations (0 disables).
    outlier_fraction : float
        Fraction of samples with outlier targets.
    missing_fraction : float
        Fraction of missing entries (NaN) in X.
    categorical_features : int
        Number of discrete/categorical columns to append.
    random_state : int or None
        Random seed for reproducibility.
    """

    rng = np.random.default_rng(random_state)

    # Step 1: Linear base dataset
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        bias=bias,
        random_state=random_state,
    )

    # Step 2: Add nonlinear transformations
    if nonlinear_strength > 0:
        X_nl = X.copy()
        X_nl[:, 0] = np.exp(nonlinear_strength * 0.01 * X[:, 0])
        X_nl[:, 1] = np.sin(nonlinear_strength * X[:, 1])
        X_nl[:, 2] = (X[:, 2] ** 2) * nonlinear_strength
        X = np.hstack([X, X_nl])

    # Step 3: Add correlated (redundant) features
    if corr_strength > 0:
        corr_features = X[:, :min(5, X.shape[1])] + rng.normal(
            0, corr_strength, size=(n_samples, min(5, X.shape[1]))
        )
        X = np.hstack([X, corr_features])

    # Step 4: Add outliers in target
    if outlier_fraction > 0:
        n_outliers = int(n_samples * outlier_fraction)
        idx = rng.choice(n_samples, n_outliers, replace=False)
        y[idx] += rng.normal(0, noise * 10, size=n_outliers)

    # Step 5: Add missing values
    if missing_fraction > 0:
        missing_mask = rng.random(X.shape) < missing_fraction
        X[missing_mask] = np.nan

    # Step 6: Add categorical/discrete features
    if categorical_features > 0:
        cats = rng.integers(0, 3, size=(n_samples, categorical_features))
        X = np.hstack([X, cats])

    return X, y

X, y = make_realistic_regression(
    random_state=0,
)
print(X.shape, np.isnan(X).mean(), y[:5])


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
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

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

n_size = 7
x_labels = np.zeros(n_size, dtype=object)
mses_diabetes = np.zeros(n_size)
stds_diabetes = np.zeros(n_size)
mses_california = np.zeros(n_size)
stds_california = np.zeros(n_size)
mses_train = np.zeros(n_size)
stds_train = np.zeros(n_size)
time_data = np.zeros(n_size)


# %%
t0 = time.time()
mses_diabetes[0], stds_diabetes[0] = get_score(X_diabetes, y_diabetes)
mses_california[0], stds_california[0] = get_score(X_california, y_california)
mses_train[0], stds_train[0] = get_score(X_train, y_train)
x_labels[0] = "Full Data"
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
x_labels[1] = "Zero\nImputation\n(constant)"
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
x_labels[2] = "Mean\nImputation"
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
x_labels[3] = "Median\nImputation"
T = time.time() - t0
print(T)
time_data[3] = T


# %%
t0 = time.time()
imputer = KNNImputer(add_indicator=True)
mses_diabetes[4], stds_diabetes[4] = get_score(
    X_miss_diabetes, y_miss_diabetes, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
mses_california[4], stds_california[4] = get_score(
    X_miss_california, y_miss_california, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
mses_train[4], stds_train[4] = get_score(
    X_miss_train, y_miss_train, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
x_labels[4] = "KNN\nImputation"
T = time.time() - t0
print(T)
time_data[4] = T


# %%
t0 = time.time()
imputer = IterativeImputer(add_indicator=True)

mses_diabetes[5], stds_diabetes[5] = get_score(
    X_miss_diabetes, y_miss_diabetes, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
mses_california[5], stds_california[5] = get_score(
    X_miss_california, y_miss_california, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
mses_train[5], stds_train[5] = get_score(
    X_miss_train, y_miss_train, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
x_labels[5] = "Iterative\nImputation\n(BayesianRidge)"
T = time.time() - t0
print(T)
time_data[5] = T


# %%
import scikitplot as sp
sp.__version__


# %%
from scikitplot.experimental import enable_annoyknn_imputer
from scikitplot.impute import AnnoyKNNImputer
# print(AnnoyKNNImputer.__doc__)


# %%
t0 = time.time()
# 'angular', 'euclidean', 'manhattan', 'hamming', 'dot'
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, n_trees=5, metric='dot')
mses_diabetes[6], stds_diabetes[6] = get_score(
    X_miss_diabetes, y_miss_diabetes, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, n_trees=10, metric='dot')
mses_california[6], stds_california[6] = get_score(
    X_miss_california, y_miss_california, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, n_trees=20, metric='angular')
mses_train[6], stds_train[6] = get_score(
    X_miss_train, y_miss_train, make_pipeline(RobustScaler(), MaxAbsScaler(), imputer)
)
x_labels[6] = "AnnoyKNN\nImputation\n(Vector Based)"
T = time.time() - t0
print(T)
time_data[6] = T


# %%
import matplotlib.pyplot as plt

mses_diabetes = np.abs(mses_diabetes)  # * -1
mses_california = np.abs(mses_california)
mses_train = np.abs(mses_train)

n_bars = len(mses_diabetes)
xval = np.arange(n_bars)

colors = ["r", "g", "b", "m", "c", "orange", "olive", "gray"]

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
    ax1.bar_label(bar, fmt="%.3f", label_type='center', padding=-45)

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
    ax2.bar_label(bar, fmt="%.3f", label_type='center', padding=-45)

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
    ax3.bar_label(bar, fmt="%.3f", label_type='center', padding=-45)

ax3.set_title("Imputation Techniques with Train Data")
ax3.set_yticks(xval)
ax3.set_xlabel("MSE")
ax3.invert_yaxis()
ax3.set_yticklabels([""] * n_bars)

plt.legend(title='Time')
plt.tight_layout()
plt.show()

# %%
# AnnoyKNNImputer performance and accuracy are highly sensitive to both the
# selected distance metric and the number of trees used to build the Annoy index.
# An inappropriate metric or insufficient number of trees may lead to poor
# neighbor retrieval and degraded imputation quality.

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: impute
#    plot-type: bar
#    level: beginner
#    purpose: showcase
