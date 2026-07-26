.. _dl-more-vectorization-examples:

========================================================================
More Vectorization Examples
========================================================================

**Stage 4 · ⚙️ Backprop & Vectorization**  ·  Lesson 16 of 17  ·  *intermediate*

:doc:`◀ Previous · Vectorization in Logistic Regression <15-vectorization-in-logistic-regression>`   ·   :doc:`Next · Vectorizing Logistic Regression ▶ <17-vectorizing-logistic-regression>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Element-wise functions
------------------------

Loops are not only for sums. Any operation applied **to every element** of an array has a vectorised
form. Instead of writing a loop to exponentiate each entry of a vector :math:`v`, call ``np.exp(v)``;
the same holds for ``np.log``, ``np.abs``, ``np.maximum(0, v)`` (a ReLU), powers like ``v ** 2``, and
reciprocals ``1 / v``. Each runs the loop internally in compiled code, over the whole array at once.

Broadcasting
--------------

Vectorised code often combines arrays of **different shapes**, and ``numpy`` reconciles them by
**broadcasting**: the smaller array is **stretched** to match the larger. Add a scalar to a vector and
the scalar is applied to every element; add a :math:`1 \times n` row to an :math:`m \times n` matrix
and the row is **copied down** all :math:`m` rows. It is the mechanism that let the bias ``b`` add
cleanly across every column in the last lesson.

A worked example
------------------

Ng's example computes each food's macronutrient split as a percentage of its calories. With a matrix
:math:`A` of nutrient values, the totals and percentages take **two lines, no loop**:

.. code-block:: python

   cal = A.sum(axis=0)                 # column sums -> total calories per food
   percentage = 100 * A / cal.reshape(1, 4)

The division **broadcasts** the :math:`1 \times 4` totals across every row of :math:`A`. (Here
``axis=0`` sums down columns; ``axis=1`` would sum across rows.)

The guideline
---------------

All of this serves one rule — Ng's programming guideline, **"whenever possible, avoid explicit
for-loops."** Reach first for a whole-array operation or a broadcast; fall back to a Python loop only
when no vectorised form exists. With these tools the entire logistic-regression step vectorises, which
the final lesson assembles.

.. hint::

   **Related lessons:** :doc:`Vectorization in Logistic Regression <15-vectorization-in-logistic-regression>`  ·  :doc:`Vectorizing Logistic Regression <17-vectorizing-logistic-regression>`  ·  :doc:`Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>`  ·  :doc:`Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/more-vectorization-examples/ <https://insightful-data-lab.com/2025/04/07/more-vectorization-examples/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
