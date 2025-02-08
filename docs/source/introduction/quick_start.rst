.. _quick_start:

.. title:: scikit-plots: Machine Learning Visualization with Python

======================================================================
Quick Start Guide
======================================================================

This guide provides a quick introduction to plotting with scikit-plots.

1. **Install Scikit-plots**:

- Use pip to install Scikit-plots::

    >>> pip install scikit-plots


A Simple Example
----------------

Let's start with a basic example where we use a Random Forest classifier to evaluate the digits dataset provided by Scikit-learn.

A common way to assess a classifier's performance is through its confusion matrix. Here’s how we can do it:

1. **Load the Dataset**:
   We'll use the digits dataset, which contains features and labels for classification.

2. **Initialize the Classifier**:
   Create a :class:`~sklearn.ensemble.RandomForestClassifier` with specified parameters.

3. **Generate Predictions**:
   Use :func:`~sklearn.model_selection.cross_val_predict` to obtain predicted labels through cross-validation. This function provides cross-validated estimates for each sample point, which helps in evaluating metrics like accuracy, precision, recall, and the confusion matrix.

4. **Plot the Confusion Matrix**:
   Use :py:func:`~scikitplot.api.metrics.plot_classifier_eval` to visualize the confusion matrix.

5. **Display the Plot**:
   Optionally, use :py:func:`~matplotlib.pyplot.show` to display the plot.

Here’s the code to illustrate the process:

.. plot:: introduction/quick_start.py
   :align: center
   :context: close-figs
   :include-source:

The resulting confusion matrix shows how well the classifier performs. In this case, it struggles with digits 1, 8, and 9. Fine-tuning the Random Forest’s hyperparameters might improve performance.


One More Example
----------------

**Maximum flexibility. Compatibility with non-scikit-learn objects.**

Although Scikit-plot is loosely based around the scikit-learn interface, you don't actually need Scikit-learn objects to use the available functions. As long as you provide the functions what they're asking for, they'll happily draw the plots for you.

Try Deep Learning Models like `Tensorflow <https://www.tensorflow.org>`_ or `Pytorch <https://pytorch.org>`_ or  `🤗 Transformers <https://huggingface.co/docs/transformers/index>`_ etc.

Here's a quick example to generate the precision-recall curves of a :py:class:`~tensorflow.keras.Model` or :py:class:`~torch.nn.Module` or :py:class:`~transformers.TFPreTrainedModel` model on a sample dataset.

.. plot:: introduction/quick_start_tf.py
   :align: center
   :context: close-figs
   :include-source:

Just pass the ground truth labels and predicted probabilities to
:py:func:`~scikitplot.api.metrics.plot_precision_recall` to generate the precision-recall curves.
This method is flexible and works with any classifier that produces predicted probabilities,
from Keras classifiers to NLTK Naive Bayes to XGBoost as long as you pass in the predicted probabilities
in the correct format.


Now what?
---------

The recommended way to start using Scikit-plot is to just go through the documentation for the various modules and choose which plots you think would be useful for your work.

Happy plotting!
