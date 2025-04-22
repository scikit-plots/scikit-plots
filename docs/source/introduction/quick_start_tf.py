# introduction/quick_start_tf.py
# %run: Python scripts and shows any outputs directly in the notebook.
# %run ../docs/source/introduction/quick_start_tf.py

# Import Libraries
# Before tf {'0':'All', '1':'Warnings+', '2':'Errors+', '3':'Fatal Only'} if any
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Disable GPU and force TensorFlow to use CPU
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set TensorFlow's logging level to Fatal
import logging

import tensorflow as tf

tf.get_logger().setLevel(logging.CRITICAL)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Loading the dataset
X, y = load_digits(return_X_y=True)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=0)

# Convert labels to one-hot encoding
Y_train = tf.keras.utils.to_categorical(y_train)
Y_val = tf.keras.utils.to_categorical(y_val)

# Define a simple TensorFlow model
tf.keras.backend.clear_session()
model = tf.keras.Sequential(
    [
        # tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input (Functional API)
        tf.keras.layers.InputLayer(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(
    X_train, Y_train, batch_size=32, epochs=2, validation_data=(X_val, Y_val), verbose=0
)

# Predict probabilities on the validation set
y_probas = model.predict(X_val)

# Plot the data
import matplotlib.pyplot as plt

import scikitplot as sp

sp.get_logger().setLevel(sp.sp_logging.WARNING)
# Plot precision-recall curves
sp.metrics.plot_precision_recall(
    y_val,
    y_probas,
    save_fig=True,
    save_fig_filename="",
    # overwrite=True,
    add_timestamp=True,
    verbose=True,
)
