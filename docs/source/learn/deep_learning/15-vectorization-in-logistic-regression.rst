.. _dl-vectorization-in-logistic-regression:

========================================================================
Vectorization in Logistic Regression
========================================================================

**Stage 4 · ⚙️ Backprop & Vectorization**  ·  Lesson 15 of 17  ·  *intermediate*

:doc:`◀ Previous · Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>`   ·   :doc:`Next · More Vectorization Examples ▶ <16-more-vectorization-examples>`   ·   :doc:`↑ Section <index>`


What vectorization is
-----------------------

**Vectorization** is, in Andrew Ng's words, "the art of getting rid of explicit for-loops in your
code". Instead of looping element by element, you express the computation as **whole-array
operations** — matrix and vector products — and let an optimised numerical library do the looping in
fast compiled code.

The forward pass, vectorized
------------------------------

Recall the per-example forward step :math:`z^{(i)} = \mathbf{w}^{\!\top} x^{(i)} + b`. Stacking all
:math:`m` examples as **columns** of :math:`X` (the convention from Lesson 5) collapses the whole
loop into one line:

.. math::

   Z = \mathbf{w}^{\!\top} X + b, \qquad A = \sigma(Z),

where :math:`Z` and :math:`A` are :math:`1 \times m`. In ``numpy`` this is literally
``Z = np.dot(w.T, X) + b`` followed by ``A = sigmoid(Z)`` — the scalar ``b`` is **broadcast** across
all columns automatically.

Why it's faster
-----------------

``np.dot`` hands the arithmetic to routines that use the CPU's (or GPU's) **parallel** vector
instructions, processing many numbers at once. Ng's demonstration times the same dot product both
ways, and the vectorised version runs **hundreds of times faster** than the Python loop. On large
datasets that is the difference between minutes and hours.

The habit
-----------

The rule to carry forward is Ng's programming guideline: **"whenever possible, avoid explicit
for-loops."** The remaining lessons apply it relentlessly — vectorising the **gradients** too, until
an entire step of logistic regression over all :math:`m` examples runs without a single Python loop
over the data.

.. hint::

   **Related lessons:** :doc:`Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>`  ·  :doc:`More Vectorization Examples <16-more-vectorization-examples>`  ·  :doc:`Vectorizing Logistic Regression <17-vectorizing-logistic-regression>`  ·  :doc:`Binary Classification and Logistic Regression (Neural Network Basics) <05-binary-classification-and-logistic-regression-neural-network-basics>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/vectorization-in-logistic-regression/ <https://insightful-data-lab.com/2025/04/07/vectorization-in-logistic-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
