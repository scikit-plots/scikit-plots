"""
visualkeras custom vgg16 example
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

# create VGG16
image_size = 224
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(shape=(image_size, image_size, 3)))

model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(64, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(64, activation="relu", kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())

model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(128, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(128, activation="relu", kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())

model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(256, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(256, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(256, activation="relu", kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())

model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(512, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(512, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(512, activation="relu", kernel_size=(3, 3)))
model.add(visualkeras.SpacingDummyLayer())

model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(512, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(512, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.ZeroPadding2D((1, 1)))
model.add(tf.keras.layers.Conv2D(512, activation="relu", kernel_size=(3, 3)))
model.add(tf.keras.layers.MaxPooling2D())
model.add(visualkeras.SpacingDummyLayer())

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(4096, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(4096, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1000, activation="softmax"))

# Now visualize the model!

from collections import defaultdict

color_map = defaultdict(dict)
color_map[tf.keras.layers.Conv2D]["fill"] = "orange"
color_map[tf.keras.layers.ZeroPadding2D]["fill"] = "gray"
color_map[tf.keras.layers.Dropout]["fill"] = "pink"
color_map[tf.keras.layers.MaxPooling2D]["fill"] = "red"
color_map[tf.keras.layers.Dense]["fill"] = "green"
color_map[tf.keras.layers.Flatten]["fill"] = "teal"

from PIL import ImageFont


def get_font():
    import platform

    system_platform = platform.system().lower()
    # Detect platform and select font accordingly
    try:
        if system_platform == "windows":
            return ImageFont.truetype("arial.ttf", 32)
        if system_platform == "darwin":  # macOS
            return ImageFont.truetype(
                "/Library/Fonts/Arial.ttf", 32
            )  # or "/System/Library/Fonts/Helvetica.ttc"
        if system_platform == "linux":
            # Try a more common font path
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
            )
        raise ValueError("Unsupported platform")
    except OSError:
        # Fallback font if the specified font is not found
        print("Font not found, using default font.")
        return ImageFont.load_default()


# Example usage
font = get_font()

img_vgg16 = visualkeras.layered_view(
    model,
    to_file="../result_images/vgg16.png",
    type_ignore=[visualkeras.SpacingDummyLayer],
)
img_vgg16_legend = visualkeras.layered_view(
    model,
    to_file="../result_images/vgg16_legend.png",
    type_ignore=[visualkeras.SpacingDummyLayer],
    legend=True,
    font=font,
)
img_vgg16_spacing_layers = visualkeras.layered_view(
    model,
    to_file="../result_images/vgg16_spacing_layers.png",
    type_ignore=[],
    spacing=0,
)
img_vgg16_type_ignore = visualkeras.layered_view(
    model,
    to_file="../result_images/vgg16_type_ignore.png",
    type_ignore=[
        tf.keras.layers.ZeroPadding2D,
        tf.keras.layers.Dropout,
        tf.keras.layers.Flatten,
        visualkeras.SpacingDummyLayer,
    ],
)
img_vgg16_color_map = visualkeras.layered_view(
    model,
    to_file="../result_images/vgg16_color_map.png",
    type_ignore=[visualkeras.SpacingDummyLayer],
    color_map=color_map,
)
img_vgg16_flat = visualkeras.layered_view(
    model,
    to_file="../result_images/vgg16_flat.png",
    type_ignore=[visualkeras.SpacingDummyLayer],
    draw_volume=False,
)
img_vgg16_scaling = visualkeras.layered_view(
    model,
    to_file="../result_images/vgg16_scaling.png",
    type_ignore=[visualkeras.SpacingDummyLayer],
    scale_xy=1,
    scale_z=1,
    max_z=1000,
)
try:
    import matplotlib.pyplot as plt

    plt.imshow(img_vgg16)
    plt.axis("off")
    plt.show()
    plt.imshow(img_vgg16_legend)
    plt.axis("off")
    plt.show()
    plt.imshow(img_vgg16_spacing_layers)
    plt.axis("off")
    plt.show()
    plt.imshow(img_vgg16_type_ignore)
    plt.axis("off")
    plt.show()
    plt.imshow(img_vgg16_color_map)
    plt.axis("off")
    plt.show()
    plt.imshow(img_vgg16_flat)
    plt.axis("off")
    plt.show()
    plt.imshow(img_vgg16_scaling)
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
