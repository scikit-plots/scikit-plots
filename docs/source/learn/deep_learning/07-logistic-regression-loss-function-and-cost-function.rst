.. _dl-logistic-regression-loss-function-and-cost-function:

========================================================================
Logistic Regression – Loss Function and Cost Function
========================================================================

**Stage 2 · 🔵 Logistic Regression as a Neuron**  ·  Lesson 07 of 17  ·  *beginner*

:doc:`◀ Previous · Logistic Regression (Binary Classification Model) <06-logistic-regression-binary-classification-model>`   ·   :doc:`Next · Gradient Descent in Logistic Regression ▶ <08-gradient-descent-in-logistic-regression>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Measuring one prediction
--------------------------

To learn, the neuron needs a **score of how wrong it is**. A **loss function**
:math:`\mathcal{L}(\hat{y}, y)` measures the error on a **single** example — comparing the predicted
probability :math:`\hat{y}` against the true label :math:`y`. Training then adjusts :math:`\mathbf{w}`
and :math:`b` to make that loss small.

Why not squared error
-----------------------

The obvious choice, **squared error** :math:`\tfrac{1}{2}(\hat{y} - y)^2`, works for linear
regression but is a poor fit here. Because :math:`\hat{y}` passes through the **sigmoid**, squared
error makes the cost **non-convex** — a bumpy surface with many local minima where gradient descent
can get **stuck**. We want a loss whose surface is a single smooth bowl.

The cross-entropy loss
------------------------

The right choice is the **cross-entropy** (log) loss:

.. math::

   \mathcal{L}(\hat{y}, y) = -\big(y \log \hat{y} + (1 - y)\log(1 - \hat{y})\big).

Read it by cases. If :math:`y = 1` it reduces to :math:`-\log \hat{y}`, which is small only when
:math:`\hat{y}` is **close to 1**; if :math:`y = 0` it reduces to :math:`-\log(1 - \hat{y})`, small
only when :math:`\hat{y}` is **close to 0**. Confident wrong answers are punished hard — and for
logistic regression this loss is **convex**. (It is exactly the negative log-likelihood of a
Bernoulli label.)

From loss to cost
-------------------

Loss scores **one** example; the **cost function** averages it over **all** :math:`m`:

.. math::

   J(\mathbf{w}, b) = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(\hat{y}^{(i)}, y^{(i)})
     = -\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log \hat{y}^{(i)}
       + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})\Big].

**Loss is per-example; cost is the whole-set average.** Training means finding the
:math:`\mathbf{w}, b` that **minimise** :math:`J` — the job of the next lesson.

.. hint::

   **Related lessons:** :doc:`Logistic Regression (Binary Classification Model) <06-logistic-regression-binary-classification-model>`  ·  :doc:`Gradient Descent in Logistic Regression <08-gradient-descent-in-logistic-regression>`  ·  :doc:`Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`  ·  :doc:`Binary Classification and Logistic Regression (Neural Network Basics) <05-binary-classification-and-logistic-regression-neural-network-basics>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/logistic-regression-loss-function-and-cost-function/ <https://insightful-data-lab.com/2025/04/07/logistic-regression-loss-function-and-cost-function/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: beginner
