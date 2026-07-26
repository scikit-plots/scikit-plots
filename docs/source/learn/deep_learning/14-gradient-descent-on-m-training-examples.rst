.. _dl-gradient-descent-on-m-training-examples:

========================================================================
Gradient Descent on m Training Examples
========================================================================

**Stage 4 · ⚙️ Backprop & Vectorization**  ·  Lesson 14 of 17  ·  *intermediate*

:doc:`◀ Previous · Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`   ·   :doc:`Next · Vectorization in Logistic Regression ▶ <15-vectorization-in-logistic-regression>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Averaging the gradient
------------------------

A single example gives a noisy gradient; training uses the **whole set**. The cost is the average
loss, so — because differentiation is linear — **its gradient is the average of the per-example
gradients**:

.. math::

   J(\mathbf{w}, b) = \frac{1}{m}\sum_{i=1}^{m} \mathcal{L}(a^{(i)}, y^{(i)}), \qquad
   \frac{\partial J}{\partial w_j} = \frac{1}{m}\sum_{i=1}^{m} x_j^{(i)}\,\mathrm{d}z^{(i)}, \qquad
   \frac{\partial J}{\partial b} = \frac{1}{m}\sum_{i=1}^{m} \mathrm{d}z^{(i)}.

The explicit algorithm
------------------------

Written out naively, one gradient-descent step **accumulates** over the examples, then divides by
:math:`m`:

.. code-block:: python

   J = 0; dw1 = 0; dw2 = 0; db = 0
   for i in range(m):                     # loop over the m examples
       z = w1 * x1[i] + w2 * x2[i] + b
       a = sigmoid(z)
       J += -(y[i] * log(a) + (1 - y[i]) * log(1 - a))
       dz = a - y[i]
       dw1 += x1[i] * dz                  # one line per feature ...
       dw2 += x2[i] * dz
       db += dz
   J /= m; dw1 /= m; dw2 /= m; db /= m
   w1 -= alpha * dw1; w2 -= alpha * dw2; b -= alpha * db

Two loops, and slow
---------------------

Look closely and there are **two** nested loops: the visible one over the :math:`m` examples, and a
hidden one over the :math:`n` **features** — with :math:`n` weights you would need
:math:`\mathrm{d}w_1, \mathrm{d}w_2, \dots, \mathrm{d}w_n`. Explicit Python loops like these are
**badly suited to parallel hardware** and crawl on large datasets. Removing them — **vectorization** —
is the subject of the next lesson.

.. hint::

   **Related lessons:** :doc:`Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`  ·  :doc:`Vectorization in Logistic Regression <15-vectorization-in-logistic-regression>`  ·  :doc:`Gradient Descent in Logistic Regression <08-gradient-descent-in-logistic-regression>`  ·  :doc:`Vectorizing Logistic Regression <17-vectorizing-logistic-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/gradient-descent-on-m-training-examples/ <https://insightful-data-lab.com/2025/04/07/gradient-descent-on-m-training-examples/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
