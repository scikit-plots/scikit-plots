"""
visualkeras custom VGG example
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

from scikitplot import visualkeras

# model = tf.keras.applications.VGG16(
#     include_top=True,
#     weights=None,  # "imagenet" or 'path/'
#     input_tensor=None,
#     input_shape=None,
#     pooling=None,
#     classes=1000,
#     classifier_activation="softmax",
#     name="vgg16",
# )
# visualkeras.layered_view(
#   model,
#   legend=True,
#   show_dimension=True,
#   to_file='../result_images/vgg16.png',
# )

model = tf.keras.applications.VGG19(
    include_top=True,
    weights=None,  # "imagenet" or 'path/'
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
    name="vgg19",
)
img_vgg19 = visualkeras.layered_view(
    model,
    legend=True,
    show_dimension=True,
    to_file="../result_images/vgg19.png",
)
try:
    import matplotlib.pyplot as plt

    plt.imshow(img_vgg19)
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
