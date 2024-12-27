"""
visualkeras Dense example
==========================================

An example showing the :py:func:`~scikitplot.visualkeras` function
used by a :py:mod:`~tf.keras`.
"""
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

# Force garbage collection
import gc; gc.collect()
import tensorflow.python as tf_python
# Clear the GPU memory cache
tf_python.keras.backend.clear_session()

model = tf_python.keras.models.Sequential()
model.add(tf_python.keras.layers.Dense(8, activation='relu', input_shape=(4000,)))
model.add(tf_python.keras.layers.Dense(8, activation='relu'))
model.add(tf_python.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

from scikitplot import visualkeras

img_spam = visualkeras.layered_view(
  model,
  to_file='../result_images/spam.png',
  min_xy=10, min_z=10, scale_xy=100, scale_z=100,
  one_dim_orientation='x',
)
try:
    import matplotlib.pyplot as plt
    plt.imshow(img_spam)
    plt.axis('off')
    plt.show()
except:
    pass