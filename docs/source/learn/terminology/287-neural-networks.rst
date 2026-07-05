:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-neural-networks:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">💡&nbsp;&nbsp;<b>Neural Networks</b></div>`

=================
Neural Networks
=================

*Layered models of interconnected units that learn complex functions from data.*

What it is
----------

A **neural network** is a model of interconnected **neurons** arranged in **layers** — an **input**
layer, one or more **hidden** layers, and an **output** layer. Each neuron computes a **weighted sum** of
its inputs, adds a **bias**, and passes the result through a **non-linear activation function**, letting
the network model complex relationships.

Weights, biases, activations
----------------------------

**Weights** set the strength of each connection and **biases** shift the activation; the **non-linearity**
is what lets stacked layers represent functions a linear model cannot. Common activations are **ReLU**,
**sigmoid** and **tanh**. A single unit computes

.. math::

   a = \phi\!\left( \sum_i w_i x_i + b \right),

where :math:`\phi` is the activation function.

Forward pass and backpropagation
--------------------------------

In the **forward pass**, inputs flow layer by layer to an output, and a **loss function** (MSE for
regression, cross-entropy for classification) measures the error. **Backpropagation** then applies the
**chain rule** to compute the gradient of that loss with respect to every weight, propagating the error
**backward** from output to input; **gradient descent** updates the weights, with the **learning rate**
:math:`\eta` setting the step size:

.. math::

   w \leftarrow w - \eta \, \frac{\partial \mathcal{L}}{\partial w}.

In practice
-----------

Depth and width are chosen for the task, **validation** data guards against overfitting, and frameworks
like **PyTorch**, **TensorFlow** and **Keras** implement backpropagation automatically.

.. code-block:: python

   from sklearn.neural_network import MLPClassifier

   clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=500)
   clf.fit(X_train, y_train)

----

**Mind map — connected ideas**

   :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`Deep Ensembles <335-deep-ensembles>` · :doc:`Autoencoder <171-autoencoder>`

----

**More in AI & ML Concepts**

   :doc:`AI (Artificial Intelligence) <143-ai-artificial-intelligence>` · :doc:`Classification Models <294-classification-models>` · :doc:`Computer Vision (CV) <321-computer-vision-cv>` · :doc:`Decision Trees <340-decision-trees>` · :doc:`Linear Models <341-linear-models>` · :doc:`LLMs (Large Language Models) <158-llms-large-language-models>` · :doc:`Logistic Regression <292-logistic-regression>` · :doc:`Machine Learning (ML) <144-machine-learning-ml>` · :doc:`Medical AI <145-medical-ai>` · :doc:`Natural Language Processing (NLP) <322-natural-language-processing-nlp>` · :doc:`Regression Models <309-regression-models>` · :doc:`Support Vector Machines (SVMs) <282-support-vector-machines-svms>` · :doc:`Target Variable <236-target-variable>`

----

*Theme:* :ref:`AI & ML Concepts <term-theme-concepts>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Neural Networks <https://insightful-data-lab.com/2025/08/21/neural-networks/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: beginner
