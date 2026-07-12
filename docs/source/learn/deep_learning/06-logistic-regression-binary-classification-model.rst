.. _dl-logistic-regression-binary-classification-model:

========================================================================
Logistic Regression (Binary Classification Model)
========================================================================

**Stage 2 · 🔵 Logistic Regression as a Neuron**  ·  Lesson 06 of 17  ·  *beginner*

:doc:`◀ Previous · Binary Classification and Logistic Regression (Neural Network Basics) <05-binary-classification-and-logistic-regression-neural-network-basics>`   ·   :doc:`Next · Logistic Regression – Loss Function and Cost Function ▶ <07-logistic-regression-loss-function-and-cost-function>`   ·   :doc:`↑ Section <index>`


From score to probability
---------------------------

Given an input :math:`x`, logistic regression predicts :math:`\hat{y} = P(y = 1 \mid x)` — the
**probability** the label is 1. A plain linear score :math:`\mathbf{w}^{\!\top}\mathbf{x} + b` can be
any real number, from large negative to large positive, so it **cannot** serve as a probability
directly. It has to be squashed into :math:`[0, 1]`.

The sigmoid
-------------

The squashing function is the **sigmoid** (logistic) function:

.. math::

   \hat{y} = \sigma(\mathbf{w}^{\!\top}\mathbf{x} + b), \qquad \sigma(z) = \frac{1}{1 + e^{-z}},

with parameters :math:`\mathbf{w} \in \mathbb{R}^{n_x}` (a weight per feature) and a bias
:math:`b \in \mathbb{R}`.

Reading it
------------

The sigmoid is an **S-curve** between 0 and 1: as :math:`z \to +\infty` it approaches **1**, as
:math:`z \to -\infty` it approaches **0**, and at :math:`z = 0` it is exactly **0.5**. So a large
positive score means "confidently 1", a large negative score "confidently 0", and a score near zero
an undecided, halfway probability.

One neuron
------------

Put together, logistic regression is precisely a **single neuron**: a linear combination of the
inputs followed by a nonlinear activation — here the sigmoid. That is the exact template from
Lesson 1, and stacking many such units is all a neural network is. The next lesson gives this neuron
a **loss**, so it can learn :math:`\mathbf{w}` and :math:`b` from data.

.. hint::

   **Related lessons:** :doc:`Binary Classification and Logistic Regression (Neural Network Basics) <05-binary-classification-and-logistic-regression-neural-network-basics>`  ·  :doc:`Logistic Regression – Loss Function and Cost Function <07-logistic-regression-loss-function-and-cost-function>`  ·  :doc:`What is a Neural Network? <01-what-is-a-neural-network>`  ·  :doc:`Gradient Descent in Logistic Regression <08-gradient-descent-in-logistic-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/logistic-regression-binary-classification-model/ <https://insightful-data-lab.com/2025/04/07/logistic-regression-binary-classification-model/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: beginner
