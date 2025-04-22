"""
Visualkeras: Spam Classification Conv1D Dense Example
======================================================================

An example showing Spam the :py:func:`~scikitplot.visualkeras` function
used by a :py:class:`~tensorflow.keras.Model` model.
"""

# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Force garbage collection

import gc

gc.collect()

# %%

# pip install protobuf==5.29.4
import tensorflow as tf

# Clear any session to reset the state of TensorFlow/Keras
tf.keras.backend.clear_session()

# %%

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
model.summary()

# %%

from scikitplot import visualkeras

img_spam = visualkeras.layered_view(
    model,
    min_z=1,
    min_xy=1,
    max_z=4096,
    max_xy=4096,
    scale_z=6,
    scale_xy=0.2,
    font={"font_size": 14},
    text_callable="default",
    one_dim_orientation="x",
    # to_file="./spam_conv_x.png",
    save_fig=True,
    save_fig_filename="spam_conv_x.png",
)

img_spam = visualkeras.layered_view(
    model,
    min_z=1,
    min_xy=1,
    max_z=4096,
    max_xy=4096,
    scale_z=6,
    scale_xy=0.2,
    font={"font_size": 14},
    text_callable="default",
    one_dim_orientation="y",
    # to_file="./spam_conv_y.png",
    save_fig=True,
    save_fig_filename="spam_conv_y.png",
)

img_spam = visualkeras.layered_view(
    model,
    min_z=1,
    min_xy=1,
    max_z=4096,
    max_xy=4096,
    scale_z=0.2,
    scale_xy=1,
    font={"font_size": 9},
    text_callable="default",
    one_dim_orientation="z",
    # to_file="./spam_conv_z.png",
    save_fig=True,
    save_fig_filename="spam_conv_z.png",
    overwrite=False,
    add_timestamp=True,
    verbose=True,
)

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
