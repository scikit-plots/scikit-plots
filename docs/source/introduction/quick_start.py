# introduction/quick_start.py
# %run: Python scripts and shows any outputs directly in the notebook.
# %run ../docs/source/introduction/quick_start.py

# Import Libraries
import numpy as np
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split

# Loading the dataset
X, y = load_digits(return_X_y=True)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=0)

# Define a simple model
clf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=0)

# Train the model
y_pred = cross_val_predict(clf, X_train, y_train)

# Plot the data
import scikitplot as sp

sp.get_logger().setLevel(sp.sp_logging.WARNING)
train_r = sp.metrics.plot_classifier_eval(
    y_train,
    y_pred,
    labels=np.unique(y),
    figsize=(8, 3.2),
    title="Confusion Matrix",
    save_fig=True,
    save_fig_filename="",
    # overwrite=True,
    add_timestamp=True,
    verbose=True,
)
