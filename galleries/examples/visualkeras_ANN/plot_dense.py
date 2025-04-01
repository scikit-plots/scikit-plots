"""
visualkeras Spam Dense example
==========================================

An example showing the :py:func:`~scikitplot.visualkeras` function
used by a :py:class:`~tensorflow.keras.Model` model.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Force garbage collection
import gc

gc.collect()

# pip install protobuf==5.29.4

import tensorflow as tf

# Clear any session to reset the state of TensorFlow/Keras
tf.keras.backend.clear_session()

import tensorflow.python as tf_python

# Clear any session to reset the state of TensorFlow/Keras
tf_python.keras.backend.clear_session()

model = tf_python.keras.models.Sequential()
model.add(tf_python.keras.layers.InputLayer(input_shape=(100,)))

# Add Dense layers
model.add(tf_python.keras.layers.Dense(64, activation="relu"))  # input_shape=(100,)
model.add(tf_python.keras.layers.Dense(32, activation="relu"))
model.add(tf_python.keras.layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

from scikitplot import visualkeras

img_spam = visualkeras.layered_view(
    model,
    to_file="../result_images/spam_dense.png",
    min_xy=10,
    min_z=10,
    scale_xy=10,
    scale_z=10,
    one_dim_orientation="x",
)
try:
    import matplotlib.pyplot as plt

    plt.imshow(img_spam)
    plt.axis("off")
    plt.show()
except:
    pass

# %%
#
# .. tags::
#
#    model-type: classification
#    model-workflow: model building
#    plot-type: visualkeras
#    domain: neural network
#    level: beginner
#    purpose: showcase
