"""
Visualkeras Spam Classification Conv1D Dense Example
======================================================================

An example showing Spam the :py:func:`~scikitplot.visualkeras` function
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

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(100,)))

# To convert 2D of input data into a 3D input
# Reshape to a compatible shape for Conv1D as [batch_size, time_steps, input_dimension]
# The Conv1D layer expects a 3D input: (batch_size, steps, channels).
# The Reshape layer now reshapes the input to (n_timesteps,n_features) like (100, 1),
# which matches the expected input of Conv1D.
model.add(
    tf.keras.layers.Reshape((100, 1))
)  # Shape: (batch_size, 100, 1), input_shape=(100,)

# Add Conv1D and other layers
model.add(tf.keras.layers.Conv1D(32, 1, strides=1, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

# Flatten and add Dense layers
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dense(32, activation="relu"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

from scikitplot import visualkeras

img_spam = visualkeras.layered_view(
    model,
    to_file="../result_images/spam_conv.png",
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
#    level: intermediate
#    purpose: showcase
