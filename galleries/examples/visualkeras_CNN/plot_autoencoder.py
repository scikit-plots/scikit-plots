"""
visualkeras autoencoder example
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

# encoder Model
encoder_input = tf.keras.Input(shape=(28, 28, 1), name="img")
x = tf.keras.layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
x = tf.keras.layers.MaxPooling2D(3)(x)
x = tf.keras.layers.Conv2D(32, 3, activation="relu")(x)
x = tf.keras.layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = tf.keras.layers.GlobalMaxPooling2D()(x)
encoder = tf.keras.Model(encoder_input, encoder_output, name="encoder")

# autoencoder Model
x = tf.keras.layers.Reshape((4, 4, 1))(encoder_output)
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = tf.keras.layers.UpSampling2D(3)(x)
x = tf.keras.layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = tf.keras.layers.Conv2DTranspose(1, 3, activation="relu")(x)
autoencoder = tf.keras.Model(encoder_input, decoder_output, name="autoencoder")

from scikitplot import visualkeras

img_encoder = visualkeras.layered_view(encoder, to_file="../result_images/encoder.png")
img_autoencoder = visualkeras.layered_view(
    autoencoder, to_file="../result_images/autoencoder.png"
)
try:
    import matplotlib.pyplot as plt

    plt.imshow(img_encoder)
    plt.axis("off")
    plt.show()
    plt.imshow(img_autoencoder)
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
