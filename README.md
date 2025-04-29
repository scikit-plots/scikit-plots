<!--
- https://packaging.python.org/en/latest/guides/making-a-pypi-friendly-readme/
- https://github.com/pypa/readme_renderer
-->

# Welcome to Scikit-plots 101

<a href="https://github.com/scikit-plots/scikit-plots/actions/workflows/wheels.yml?query=event%3Aworkflow_dispatch" target="_blank">
<img style="display:auto;width:28.5%;height:auto;" alt="CI Build Wheels" src="https://github.com/scikit-plots/scikit-plots/actions/workflows/wheels.yml/badge.svg?event=workflow_dispatch">
</a>
<a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots" target="_blank">
<img style="display:auto;width:27.5%;height:auto;" alt="Nightly wheels" src="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/badges/latest_release_date.svg">
</a>
<a href="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots" target="_blank">
<img style="display:auto;width:25%;height:auto;" alt="Nightly wheels" src="https://anaconda.org/scikit-plots-wheels-staging-nightly/scikit-plots/badges/version.svg">
</a>
<a href="https://pypi.org/project/scikit-plots" target="_blank">
<img style="display:auto;width:16.5%;height:auto;" alt="pypi" src="https://img.shields.io/pypi/v/scikit-plots">
</a>

<a href="https://app.circleci.com/pipelines/circleci/MzCciwxVsGS9w3PCUFjTaB/TPithCzV9DBEcZUACH7Zij" target="_blank">
<img style="display:auto;width:17.5%;height:auto;" alt="CircleCI" src="https://dl.circleci.com/status-badge/img/circleci/MzCciwxVsGS9w3PCUFjTaB/TPithCzV9DBEcZUACH7Zij/tree/main.svg?style=shield">
</a>
<a href="https://doi.org/10.5281/zenodo.13367000" target="_blank">
<img style="display:auto;width:32.5%;height:auto;" alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.13367000.svg">
</a>
<a href="https://github.com/astral-sh/ruff" target="_blank">
<img style="display:auto;width:16%;height:auto;" alt="Ruff" src="https://img.shields.io/badge/code%20style-ruff-000000.svg">
</a>
<a href="https://pypi.org/project/scikit-plots" target="_blank">
<img style="display:auto;width:31.5%;height:auto;" alt="pyversions" src="https://img.shields.io/pypi/pyversions/scikit-plots.svg">
</a>



## Single line functions for detailed visualizations

The quickest and easiest way to go from analysis...

#### Install|Try Scikit-plots

##### - Doc: https://scikit-plots.github.io/dev/devel/index.html

## User Installation:

### PIP Installation by `pypi` or `github`

The easiest way to set up scikit-plots is to install it using pip with the following command:

- by `pypi`:
  ```sh
  pip install -U scikit-plots
  ```

- by `pypi.anaconda.org`:
  ```sh
  ## (Optionally) Install the lost dependency packages
  ## wget https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
  curl -O https://raw.githubusercontent.com/scikit-plots/scikit-plots/main/requirements/default.txt
  pip install -r default.txt
  ```
  ```sh
  ## Try Ensure all dependencies installed
  pip install -U -i https://pypi.anaconda.org/scikit-plots-wheels-staging-nightly/simple scikit-plots
  ```

- by `GITHUB` to use `@<branches>` or `@<tags>`, If any:
  - Branches:
    ```bash
    ## pip install git+https://github.com/scikit-plots/scikit-plots.git@<branches>
    ## Latest in Development
    pip install git+https://github.com/scikit-plots/scikit-plots.git@main
    ##
    ## Works with standard Python (CPython), Added C, Cpp, Fortran Support
    ##
    pip install git+https://github.com/scikit-plots/scikit-plots.git@maintenance/0.4.x
    ##
    ## Works with standard Python (CPython), Works with PyPy interpreter
    ##
    pip install git+https://github.com/scikit-plots/scikit-plots.git@maintenance/0.3.x
    pip install git+https://github.com/scikit-plots/scikit-plots.git@maintenance/0.3.7
    ```
  <br>

  - Tags:
    ```bash
    ## pip install git+https://github.com/scikit-plots/scikit-plots.git@<tags>
    pip install git+https://github.com/scikit-plots/scikit-plots.git@v0.4.0rc3
    pip install git+https://github.com/scikit-plots/scikit-plots.git@v0.3.9rc3
    pip install git+https://github.com/scikit-plots/scikit-plots.git@v0.3.7
    ```

### Cloned Source Installation (REQUIRED OS/LIB BUILD PACKAGES)

- You can also install scikit-plots from source if you want to take advantage of the latest changes:
  ```sh
  ## Forked repo: https://github.com/scikit-plots/scikit-plots.git
  git clone https://github.com/YOUR-USER-NAME/scikit-plots.git
  cd scikit-plots
  ```
  ```sh
  ## (Optionally) Add safe directories for git
  bash docker/script/safe_dirs.sh
  ## download submodules
  git submodule update --init
  ```
  ```sh
  # pip install -r ./requirements/all.txt
  pip install -r ./requirements/build.txt
  ## Install development version
  pip install --no-cache-dir -e . -v
  ```

- #### It is also possible to include optional dependencies:
  ```sh
  ## (Optionally) Try Development [build,dev,test,doc]
  ## For More in Doc: https://scikit-plots.github.io/
  python -m pip install --no-cache-dir --no-build-isolation -e .[build,dev,test,doc] -v
  ```
  ```sh
  ## [cpu] refer tensorflow-cpu, keras, transformers
  ## [gpu] refer Cupy tensorflow lib require NVIDIA CUDA support
  pip install "scikit-plots[cpu]"
  ```

---

## Sample Plots


<table style="margin-left:auto;margin-right:auto;width:100%;border-collapse:collapse;">
  <tr>
    <th style="width:50%;text-align:center;">Sample Plot 1</th>
    <th style="width:50%;text-align:center;">Sample Plot 2</th>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_learning_curve.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-estimators-plot_learning_curve-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_calibration_curve.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-metrics-plot_calibration-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_classifier_eval.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-metrics-plot_classifier_eval-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_feature_importances.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-estimators-plot_feature_importances-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_roc.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-metrics-plot_roc-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_precision_recall.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-metrics-plot_precision_recall-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_pca_component_variance.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-decomposition-plot_pca_component_variance-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_pca_2d_projection.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-decomposition-plot_pca_2d_projection-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_elbow.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-estimators-plot_elbow-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_silhouette.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-api-metrics-plot_silhouette-1.png">
    </td>
  </tr>
  <tr>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_cumulative_gain.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-kds-plot_cumulative_gain-1.png">
    </td>
    <td style="width:50%;text-align:center;">
      <img style="display:block;width:100%;height:auto;" alt="plot_lift.png" src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/scikitplot-kds-plot_lift-1.png">
    </td>
  </tr>
</table>


Scikit-plots is the result of an unartistic data scientist's dreadful realization that *visualization is one of the most crucial components in the data science process, not just a mere afterthought*.

Gaining insights is simply a lot easier when you're looking at a colored heatmap of a confusion matrix complete with class labels rather than a single-line dump of numbers enclosed in brackets. Besides, if you ever need to present your results to someone (virtually any time anybody hires you to do data science), you show them visualizations, not a bunch of numbers in Excel.

That said, there are a number of visualizations that frequently pop up in machine learning. Scikit-plots is a humble attempt to provide aesthetically-challenged programmers (such as myself) the opportunity to generate quick and beautiful graphs and plots with as little boilerplate as possible.

## Okay then, prove it. Show us an example.

Say we use [Keras Classifier](https://keras.io/api/models/sequential/) in multi-class classification and decide we want to visualize the results of a common classification metric, such as sklearn's [classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) with a [confusion matrix](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html).

Let’s start with a basic example where we use a Keras classifier to evaluate the digits dataset provided by Scikit-learn.

```python
# Import Libraries
# Before tf {'0':'All', '1':'Warnings+', '2':'Errors+', '3':'Fatal Only'} if any
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable GPU and force TensorFlow to use CPU
import os; os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
# Set TensorFlow's logging level to Fatal
import logging; tf.get_logger().setLevel(logging.CRITICAL)

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# Loading the dataset
X, y = load_digits(return_X_y=True)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
  X, y, test_size=0.33, random_state=0
)

# Convert labels to one-hot encoding
Y_train = tf.keras.utils.to_categorical(y_train)
Y_val = tf.keras.utils.to_categorical(y_val)

# Define a simple TensorFlow model
tf.keras.backend.clear_session()
model = tf.keras.Sequential([
    # tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input (Functional API)
    tf.keras.layers.InputLayer(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(
    X_train, Y_train,
    batch_size=32,
    epochs=2,
    validation_data=(X_val, Y_val),
    verbose=0
)

# Predict probabilities on the validation set
y_probas = model.predict(X_val)

# Plot the data
import matplotlib.pyplot as plt
import scikitplot as sp
sp.get_logger().setLevel(sp.sp_logging.WARNING)
# Plot precision-recall curves
sp.metrics.plot_precision_recall(y_val, y_probas)
plt.show()
```

<div align=center>
  <img style="display:block;width:60%;height:auto;align:center;" alt="quick_start"
    src="https://raw.githubusercontent.com/scikit-plots/scikit-plots.github.io/refs/heads/main/dev/_images/quick_start_tf.png">
</div>

Pretty.

## Maximum flexibility. Compatibility with non-scikit-learn objects.

Although Scikit-plot is loosely based around the scikit-learn interface, you don't actually need scikit-learn objects to use the available functions.
As long as you provide the functions what they're asking for, they'll happily draw the plots for you.

The possibilities are endless.

## Release Notes

See the [changelog](https://scikit-plots.github.io/dev/whats_new/index.html)
for a history of notable changes to scikit-plots.

## Documentation and Examples

Explore the full features of Scikit-plot.

## Contributing to scikit-plots

Reporting a bug? Suggesting a feature? Want to add your own plot to the library? Visit our.

## Citing scikit-plots

1. scikit-plots, “scikit-plots: vlatest”. Zenodo, Aug. 23, 2024.
   DOI: [10.5281/zenodo.13367000](https://doi.org/10.5281/zenodo.13367000).

2. scikit-plots, “scikit-plots: v0.3.8dev0”. Zenodo, Aug. 23, 2024.
   DOI: [10.5281/zenodo.13367001](https://doi.org/10.5281/zenodo.13367001).
