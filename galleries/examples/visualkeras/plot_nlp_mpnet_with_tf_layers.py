"""
visualkeras transformers example
==========================================

An example showing the :py:func:`~scikitplot.visualkeras` function
used by a :py:mod:`~tf.keras` and :py:mod:`~transformers`.
"""
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Force garbage collection
import gc; gc.collect()
import tensorflow as tf
# Clear any session to reset the state of TensorFlow/Keras
tf.keras.backend.clear_session()

from transformers import TFAutoModel

from scikitplot import visualkeras

# Load the Hugging Face transformer model
transformer_model = TFAutoModel.from_pretrained("microsoft/mpnet-base")

# Define a Keras-compatible wrapper for the Hugging Face model
def wrap_transformer_model(inputs):
    input_ids, attention_mask = inputs
    outputs = transformer_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state  # Return the last hidden state for visualization

# Define Keras model inputs
input_ids = tf.keras.Input(shape=(128,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(128,), dtype=tf.int32, name="attention_mask")

# Pass inputs through the transformer model using a Lambda layer
last_hidden_state = tf.keras.layers.Lambda(
    wrap_transformer_model,
    output_shape=(128, 768),  # Explicitly specify the output shape
    name='microsoft_mpnet-base',
)([input_ids, attention_mask])

# Reshape the output to fit into Conv2D (adding extra channel dimension) inside a Lambda layer
# def reshape_last_hidden_state(x):
#     return tf.reshape(x, (-1, 1, 128, 768))
# reshaped_output = tf.keras.layers.Lambda(reshape_last_hidden_state)(last_hidden_state)
# Use Reshape layer to reshape the output to fit into Conv2D (adding extra channel dimension)
# Reshape to (batch_size, 128, 768, 1) for Conv2D input
reshaped_output = tf.keras.layers.Reshape((-1, 128, 768))(last_hidden_state)

# Add different layers to the model
x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv2d_1')(reshaped_output)
x = tf.keras.layers.BatchNormalization(name='batchnorm_1')(x)
x = tf.keras.layers.Dropout(0.3, name='dropout_1')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(x)

x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv2d_2')(x)
x = tf.keras.layers.BatchNormalization(name='batchnorm_2')(x)
x = tf.keras.layers.Dropout(0.3, name='dropout_2')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(x)

x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2d_3')(x)
x = tf.keras.layers.BatchNormalization(name='batchnorm_3')(x)
x = tf.keras.layers.Dropout(0.4, name='dropout_3')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(x)

# Add GlobalAveragePooling2D before the Dense layers
x = tf.keras.layers.GlobalAveragePooling2D(name='globalaveragepool')(x)

# Add Dense layers
x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(x)
x = tf.keras.layers.Dropout(0.5, name='dropout_4')(x)
x = tf.keras.layers.Dense(128, activation='relu', name='dense_2')(x)

# Add output layer (classification head)
dummy_output = tf.keras.layers.Dense(2, activation='softmax', name="dummy_classification_head")(x)

# Wrap into a Keras model
wrapped_model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=dummy_output)

# Visualize the wrapped model
img_nlp_mpnet_with_tf_layers = visualkeras.layered_view(
    wrapped_model,
    legend=True,
    show_dimension=True,
    scale_xy=1, scale_z=1, max_z=250,
    to_file="../result_images/nlp_mpnet_with_tf_layers.png"
)
try:
    import matplotlib.pyplot as plt
    plt.imshow(img_nlp_mpnet_with_tf_layers)
    plt.axis('off')
    plt.show()
except:
    pass