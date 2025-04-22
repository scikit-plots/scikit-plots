"""
visualkeras: Spam Dense example
==========================================

An example showing the :py:func:`~scikitplot.visualkeras` function
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

import tensorflow.python as tf_python

# Clear any session to reset the state of TensorFlow/Keras
tf_python.keras.backend.clear_session()

# %%

model = tf_python.keras.models.Sequential()
model.add(tf_python.keras.layers.InputLayer(input_shape=(100,)))

# Add Dense layers
model.add(tf_python.keras.layers.Dense(64, activation="relu"))  # input_shape=(100,)
model.add(tf_python.keras.layers.Dense(32, activation="relu"))
model.add(tf_python.keras.layers.Dense(1, activation="sigmoid"))

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
    scale_z=1,
    scale_xy=1,
    font={"font_size": 7},
    text_callable="default",
    one_dim_orientation="x",
    # to_file="result_images/spam_dense_x.png",
    save_fig=True,
    save_fig_filename="spam_dense_x.png",
)

img_spam = visualkeras.layered_view(
    model,
    min_z=1,
    min_xy=1,
    max_z=4096,
    max_xy=4096,
    scale_z=1,
    scale_xy=1,
    font={"font_size": 7},
    text_callable="default",
    one_dim_orientation="y",
    # to_file="result_images/spam_dense_y.png",
    save_fig=True,
    save_fig_filename="spam_dense_y.png",
)

img_spam = visualkeras.layered_view(
    model,
    min_z=1,
    min_xy=1,
    max_z=4096,
    max_xy=4096,
    scale_z=1,
    scale_xy=1,
    font={"font_size": 7},
    text_callable="default",
    one_dim_orientation="z",
    # to_file="result_images/spam_dense_z.png",
    save_fig=True,
    save_fig_filename="spam_dense_z.png",
)

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
