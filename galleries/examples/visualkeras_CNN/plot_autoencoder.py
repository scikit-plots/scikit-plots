"""
visualkeras: autoencoder example
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

# %%

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
autoencoder.summary()

# %%

# Build the model with an explicit input shape
autoencoder.build(
    input_shape=(None, 28, 28, 1)
)  # Batch size of None, shape (28, 28, 1)

# Create a dummy input tensor with a batch size of 1
dummy_input = tf.random.normal([1, 28, 28, 1])  # Batch size of 1, shape (28, 28, 1)
# Run the dummy input through the model to trigger shape calculation
encoder_output = autoencoder(dummy_input)
# Now check the output shape of the encoder
print("Output shape after running model with dummy input:", encoder_output.shape)

# Check each layer's output shape after building the model
for layer in encoder.layers:
    if hasattr(layer, "output_shape"):
        print(f"{layer.name} output shape: {layer.output_shape}")
    if hasattr(layer, "output"):
        print(f"{layer.name} shape: {layer.output.shape}")

# %%

from scikitplot import visualkeras

img_encoder = visualkeras.layered_view(
    encoder,
    text_callable="default",
    # to_file="result_images/encoder.png",
    save_fig=True,
    save_fig_filename="encoder.png",
)

# %%

img_autoencoder = visualkeras.layered_view(
    autoencoder,
    # to_file="result_images/autoencoder.png",
    save_fig=True,
    save_fig_filename="autoencoder.png",
)

# %%

img_autoencoder_text = visualkeras.layered_view(
    autoencoder,
    min_z=1,
    min_xy=1,
    max_z=4096,
    max_xy=4096,
    scale_z=1,
    scale_xy=1,
    # font={"font_size": 14},
    text_callable="default",
    # to_file="result_images/autoencoder_text.png",
    save_fig=True,
    save_fig_filename="autoencoder_text.png",
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
#    level: beginner
#    purpose: showcase
