"""
plot_evalplot_script with examples
==========================================

An example showing the :py:func:`~scikitplot.snsx.evalplot` function
used by a scikit-learn regressor.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Import scikit-plot
import scikitplot.snsx as sp


# %%
import matplotlib.pyplot as plt
import numpy as np; np.random.seed(0)  # reproducibility
import pandas as pd

from sklearn.datasets import (
    load_breast_cancer as data_2_classes,
    # load_iris as data_3_classes,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# %%
# Load the data
# X, y = data_3_classes(return_X_y=True, as_frame=False)
X, y = data_2_classes(return_X_y=True, as_frame=False)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=0)
np.unique(y)

# %%
# Create an instance of the LogisticRegression
model = (
    LogisticRegression(max_iter=int(1e5), random_state=0)
    .fit(X_train, y_train)
)
# Perform predictions
y_val_prob = model.predict_proba(X_val)
# Create a DataFrame with predictions
df = pd.DataFrame({
    "y_true": y_val==1,  # target class (0,1,2)
    "y_score": y_val_prob[:, 1],  # target class (0,1,2)
    # "y_true": np.random.normal(0.5, 0.1, 100).round(),
    # "y_score": np.random.normal(0.5, 0.15, 100),
    # "hue": np.random.normal(0.5, 0.4, 100).round(),
})



# %%
p = sp.evalplot(
    df,
    x="y_true",
    y="y_score",
    kind="all",
)


# %%
p = sp.evalplot(
    df,
    x="y_true",
    y="y_score",
    kind="classification_report",
    text_kws={'fontsize': 16},
)

# %%
p = sp.evalplot(
    df,
    x="y_true",
    y="y_score",
    kind="confusion_matrix",
)

# %%
fig, ax = plt.subplots(figsize=(8, 6))
p = sp.evalplot(
    df,
    x="y_true",
    y="y_score",
    kind="all",
    # legend=True,
)

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# %%
# Create a synthetic dataset with 15 classes
X, y = make_classification(n_samples=5000, n_features=20,
                           n_classes=15, n_informative=15,
                           n_redundant=5, random_state=0)

# Convert to DataFrame for easier visualization (optional)
data = pd.DataFrame(X)
data['target'] = y

print(data.head())

# %%
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# %%
# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)

# Train the model
rf_model.fit(X_train, y_train)

# %%
# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# %%
fig, ax = plt.subplots(figsize=(8, 8))
p = sp.evalplot(
    x=y_test,
    y=y_pred,
    kind="all",
    # legend=True,
)

# %%
import scikitplot as sp

# Save the combined figure as an image file
figs = sp.stack(  # experimental
    p.figure,
    p.figure,
    orient='x',
    **{'figsize': (12, 8)}
)

# %%
# Generate a classification report
print(classification_report(y_test, y_pred))

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# %%
# import seaborn as sns

# plt.figure(figsize=(12, 7))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
#             xticklabels=np.arange(15), yticklabels=np.arange(15))
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()


# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model evaluation
#    plot-type: line
#    plot-type: eval
#    level: beginner
#    purpose: showcase
