"""
visualkeras EfficientNetV2 example
==========================================

An example showing the :py:func:`~scikitplot.visualkeras` function
used by a :py:mod:`~tf.keras`.
"""
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Force garbage collection
import gc; gc.collect()
import tensorflow as tf
# Clear any session to reset the state of TensorFlow/Keras
tf.keras.backend.clear_session()

from scikitplot import visualkeras

model = tf.keras.applications.EfficientNetV2B0(
    include_top=True,
    weights=None,  # "imagenet" or 'path/'
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="efficientnetv2-b0",
)
img_efficientnetv2 = visualkeras.layered_view(
  model,
  legend=True,
  show_dimension=True,
  to_file='../result_images/efficientnetv2-b0.png',
)
try:
    import matplotlib.pyplot as plt
    plt.imshow(img_efficientnetv2)
    plt.axis('off')
    plt.show()
except:
    pass

# model = tf.keras.applications.EfficientNetV2B1(
#     include_top=True,
#     weights=None,  # "imagenet" or 'path/'
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     name="efficientnetv2-b1",
# )
# visualkeras.layered_view(
#   model,
#   legend=True,
#   show_dimension=True,
#   to_file='../result_images/efficientnetv2-b1.png',
# )

# model = tf.keras.applications.EfficientNetV2B2(
#     include_top=True,
#     weights=None,  # "imagenet" or 'path/'
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     name="efficientnetv2-b2",
# )
# visualkeras.layered_view(
#   model,
#   legend=True,
#   show_dimension=True,
#   to_file='../result_images/efficientnetv2-b2.png',
# )

# model = tf.keras.applications.EfficientNetV2B3(
#     include_top=True,
#     weights=None,  # "imagenet" or 'path/'
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     name="efficientnetv2-b3",
# )
# visualkeras.layered_view(
#   model,
#   legend=True,
#   show_dimension=True,
#   to_file='../result_images/efficientnetv2-b3.png',
# )

# model = tf.keras.applications.EfficientNetV2S(
#     include_top=True,
#     weights=None,  # "imagenet" or 'path/'
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     name="efficientnetv2-s",
# )
# visualkeras.layered_view(
#   model,
#   legend=True,
#   show_dimension=True,
#   to_file='../result_images/efficientnetv2-s.png',
# )

# model = tf.keras.applications.EfficientNetV2M(
#     include_top=True,
#     weights=None,  # "imagenet" or 'path/'
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     name="efficientnetv2-m",
# )
# visualkeras.layered_view(
#   model,
#   legend=True,
#   show_dimension=True,
#   to_file='../result_images/efficientnetv2-m.png',
# )

# model = tf.keras.applications.EfficientNetV2L(
#     include_top=True,
#     weights=None,  # "imagenet" or 'path/'
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     name="efficientnetv2-l",
# )
# visualkeras.layered_view(
#   model,
#   legend=True,
#   show_dimension=True,
#   to_file='../result_images/efficientnetv2-l.png',
# )