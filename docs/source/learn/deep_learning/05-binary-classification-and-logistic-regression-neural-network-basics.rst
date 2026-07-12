.. _dl-binary-classification-and-logistic-regression-neural-network-basics:

========================================================================
Binary Classification and Logistic Regression (Neural Network Basics)
========================================================================

**Stage 2 · 🔵 Logistic Regression as a Neuron**  ·  Lesson 05 of 17  ·  *beginner*

:doc:`◀ Previous · Geoffrey Hinton Interview <04-geoffrey-hinton-interview>`   ·   :doc:`Next · Logistic Regression (Binary Classification Model) ▶ <06-logistic-regression-binary-classification-model>`   ·   :doc:`↑ Section <index>`


The task
----------

**Binary classification** asks a yes/no question: given an input, output a label
:math:`y \in \{0, 1\}`. The running example is a **cat classifier** — an image goes in, and the model
should output **1** for "cat" and **0** for "not cat". Logistic regression, the subject of this
stage, is the simplest model for this — and, read the right way, a **single neuron**.

One example, as a vector
--------------------------

A model needs numbers, so an image is **unrolled into a feature vector**. A 64×64 colour image has
three channels (red, green, blue), giving :math:`n_x = 64 \times 64 \times 3 = 12{,}288` values
stacked into one column :math:`x \in \mathbb{R}^{n_x}`. A single labelled example is the pair
:math:`(x, y)` with :math:`x \in \mathbb{R}^{n_x}` and :math:`y \in \{0, 1\}`.

Stacking the whole set
------------------------

With :math:`m` training examples :math:`(x^{(1)}, y^{(1)}), \dots, (x^{(m)}, y^{(m)})`, Ng stacks
them into matrices. Each example becomes a **column**, so the data form

.. math::

   X \in \mathbb{R}^{n_x \times m}, \qquad Y \in \mathbb{R}^{1 \times m},

with :math:`X` holding one example per column and :math:`Y` the matching row of labels.

Why columns
-------------

Putting examples in **columns** rather than rows is a deliberate convention: it makes the
**vectorised** forward and backward passes later in this stage line up as clean matrix products, with
no transposes to track. A small choice now, much tidier code from Lesson 15 onward. In ``numpy`` the
shapes are ``X.shape == (n_x, m)`` and ``Y.shape == (1, m)``.

.. hint::

   **Related lessons:** :doc:`Logistic Regression (Binary Classification Model) <06-logistic-regression-binary-classification-model>`  ·  :doc:`Logistic Regression – Loss Function and Cost Function <07-logistic-regression-loss-function-and-cost-function>`  ·  :doc:`What is a Neural Network? <01-what-is-a-neural-network>`  ·  :doc:`Vectorizing Logistic Regression <17-vectorizing-logistic-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/binary-classification-and-logistic-regression-neural-network-basics/ <https://insightful-data-lab.com/2025/04/07/binary-classification-and-logistic-regression-neural-network-basics/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: beginner
