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
import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.datasets import (
    load_breast_cancer as data_2_classes,
    load_iris as data_3_classes,
    load_digits as data_10_classes,
)
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split

# %%
# https://www.geeksforgeeks.org/machine-learning/ml-credit-card-fraud-detection/
# https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv
# df = pd.read_csv("https://media.geeksforgeeks.org/wp-content/uploads/20240904104950/creditcard.csv")
# df.head()

# %%
Xdi_train, Xdi_val, ydi_train, ydi_val = train_test_split(
    *load_diabetes(return_X_y=True), test_size=0.25, random_state=0
)
Xca_train, Xca_val, yca_train, yca_val = train_test_split(
    *fetch_california_housing(return_X_y=True), test_size=0.25, random_state=0
)
Xbc_train, Xbc_val, ybc_train, ybc_val = train_test_split(
    *data_2_classes(return_X_y=True), test_size=0.25, random_state=0,
    stratify=data_2_classes(return_X_y=True)[1]
)


# %%
def add_missing_values(X_full, y_full, rng=np.random.RandomState(0)):
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

Xdi_train_miss, ydi_train_miss = add_missing_values(Xdi_train, ydi_train)
Xca_train_miss, yca_train_miss = add_missing_values(Xca_train, yca_train)
Xbc_train_miss, ybc_train_miss = add_missing_values(Xbc_train, ybc_train)

Xdi_val_miss, ydi_val_miss = add_missing_values(Xdi_val, ydi_val)
Xca_val_miss, yca_val_miss = add_missing_values(Xca_val, yca_val)
Xbc_val_miss, ybc_val_miss = add_missing_values(Xbc_val, ybc_val)


# %%
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# To use the experimental IterativeImputer, we need to explicitly ask for it:
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler

N_SPLITS = 4

def get_score(Xt, Xv, yt, yv, imputer=None, regresion=True):
    if regresion:
        estimator = RandomForestRegressor(random_state=0)
        scoring="neg_mean_squared_error"
    else:
        estimator = RandomForestClassifier(random_state=0, max_depth=6, class_weight='balanced')
        scoring="neg_log_loss"
    if imputer is not None:
        Xt = imputer.fit_transform(Xt, yt)
        Xv = imputer.transform(Xv)
    estimator.fit(Xt, yt)
    scores = cross_val_score(
        estimator,
        Xv, yv, scoring=scoring, cv=N_SPLITS
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
mses_diabetes[0], stds_diabetes[0] = get_score(Xdi_train, Xdi_val, ydi_train, ydi_val)
mses_california[0], stds_california[0] = get_score(Xca_train, Xca_val, yca_train, yca_val)
mses_train[0], stds_train[0] = get_score(Xbc_train, Xbc_val, ybc_train, ybc_val, regresion=False)
x_labels[0] = "Full Data"
T = time.time() - t0
print(T)
time_data[0] = T


# %%
t0 = time.time()
imputer = SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)
mses_diabetes[1], stds_diabetes[1] = get_score(
    Xdi_train_miss, Xdi_val_miss, ydi_train_miss, ydi_val_miss, imputer
)
mses_california[1], stds_california[1] = get_score(
    Xca_train_miss, Xca_val_miss, yca_train_miss, yca_val_miss, imputer
)
mses_train[1], stds_train[1] = get_score(
    Xbc_train_miss, Xbc_val_miss, ybc_train_miss, ybc_val_miss, imputer,
    regresion=False
)
x_labels[1] = "Zero\nImputation\n(constant)"
T = time.time() - t0
print(T)
time_data[1] = T


# %%
t0 = time.time()
imputer = SimpleImputer(strategy="mean", add_indicator=True)
mses_diabetes[2], stds_diabetes[2] = get_score(
    Xdi_train_miss, Xdi_val_miss, ydi_train_miss, ydi_val_miss, imputer
)
mses_california[2], stds_california[2] = get_score(
    Xca_train_miss, Xca_val_miss, yca_train_miss, yca_val_miss, imputer
)
mses_train[2], stds_train[2] = get_score(
    Xbc_train_miss, Xbc_val_miss, ybc_train_miss, ybc_val_miss, imputer,
    regresion=False
)
x_labels[2] = "Mean\nImputation"
T = time.time() - t0
print(T)
time_data[2] = T


# %%
t0 = time.time()
imputer = SimpleImputer(strategy="median", add_indicator=True)
mses_diabetes[3], stds_diabetes[3] = get_score(
    Xdi_train_miss, Xdi_val_miss, ydi_train_miss, ydi_val_miss, imputer
)
mses_california[3], stds_california[3] = get_score(
    Xca_train_miss, Xca_val_miss, yca_train_miss, yca_val_miss, imputer
)
mses_train[3], stds_train[3] = get_score(
    Xbc_train_miss, Xbc_val_miss, ybc_train_miss, ybc_val_miss, imputer,
    regresion=False
)
x_labels[3] = "Median\nImputation"
T = time.time() - t0
print(T)
time_data[3] = T


# %%
t0 = time.time()
imputer = KNNImputer(add_indicator=True)
mses_diabetes[4], stds_diabetes[4] = get_score(
    Xdi_train_miss, Xdi_val_miss, ydi_train_miss, ydi_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer)
)
mses_california[4], stds_california[4] = get_score(
    Xca_train_miss, Xca_val_miss, yca_train_miss, yca_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer)
)
mses_train[4], stds_train[4] = get_score(
    Xbc_train_miss, Xbc_val_miss, ybc_train_miss, ybc_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer),
    regresion=False
)
x_labels[4] = "KNN\nImputation"
T = time.time() - t0
print(T)
time_data[4] = T


# %%
t0 = time.time()
imputer = IterativeImputer(add_indicator=True)

mses_diabetes[5], stds_diabetes[5] = get_score(
    Xdi_train_miss, Xdi_val_miss, ydi_train_miss, ydi_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer)
)
mses_california[5], stds_california[5] = get_score(
    Xca_train_miss, Xca_val_miss, yca_train_miss, yca_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer)
)
mses_train[5], stds_train[5] = get_score(
    Xbc_train_miss, Xbc_val_miss, ybc_train_miss, ybc_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer),
    regresion=False
)
x_labels[5] = "Iterative\nImputation\n(BayesianRidge)"
T = time.time() - t0
print(T)
time_data[5] = T


# %%
import scikitplot as sp
# sp.get_logger().setLevel(sp.logging.WARNING)  # sp.logging == sp.logger
sp.logger.setLevel(sp.logger.INFO)  # default WARNING
sp.__version__


# %%
from scikitplot.experimental import enable_annoyknn_imputer
from scikitplot.impute import AnnoyKNNImputer
# print(AnnoyKNNImputer.__doc__)


# %%
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler
t0 = time.time()
# 'angular', 'euclidean', 'manhattan', 'hamming', 'dot'
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, metric='angular', n_neighbors=1)
# imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, weights="distance")
mses_diabetes[6], stds_diabetes[6] = get_score(
    Xdi_train_miss, Xdi_val_miss, ydi_train_miss, ydi_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer),
)
# imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, metric='euclidean', n_neighbors=1, initial_strategy="median")
# imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, weights="distance", initial_strategy="median", metric='euclidean', n_neighbors=430, n_trees=-1)
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, weights="distance", initial_strategy="median", metric='euclidean', n_neighbors=484)
mses_california[6], stds_california[6] = get_score(
    Xca_train_miss, Xca_val_miss, yca_train_miss, yca_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer),
)
# imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, metric='euclidean', initial_strategy="median")
imputer = AnnoyKNNImputer(add_indicator=True, random_state=0, metric='angular', initial_strategy="median", n_neighbors=1)
mses_train[6], stds_train[6] = get_score(
    Xbc_train_miss, Xbc_val_miss, ybc_train_miss, ybc_val_miss,
    make_pipeline(MaxAbsScaler(), RobustScaler(), imputer),
    regresion=False
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
        xerr=stds_diabetes[j] / 10,
        color=colors[j],
        alpha=0.6,
        align="center",
        label=str(np.round(td, 3))
    )
    bars1.append(bar)

# Add bar value labels
for bar in bars1:
    ax1.bar_label(bar, fmt="%.3f", label_type='center', padding=-10)

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
    ax2.bar_label(bar, fmt="%.5f", label_type='center', padding=-10)

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
    ax3.bar_label(bar, fmt="%.5f", label_type='center', padding=-10)

ax3.set_title("Imputation Techniques with Breast Cancer Data")
ax3.set_yticks(xval)
ax3.set_xlabel("LogLoss")
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
