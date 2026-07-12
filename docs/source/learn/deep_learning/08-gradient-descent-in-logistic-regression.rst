.. _dl-gradient-descent-in-logistic-regression:

========================================================================
Gradient Descent in Logistic Regression
========================================================================

**Stage 3 · 📉 Derivatives & the Computation Graph**  ·  Lesson 08 of 17  ·  *intermediate*

:doc:`◀ Previous · Logistic Regression – Loss Function and Cost Function <07-logistic-regression-loss-function-and-cost-function>`   ·   :doc:`Next · Derivatives ▶ <09-derivatives>`   ·   :doc:`↑ Section <index>`


The optimization problem
--------------------------

With a cost :math:`J(\mathbf{w}, b)` in hand, learning becomes a search: **find the**
:math:`\mathbf{w}, b` **that make** :math:`J` **smallest**. The cross-entropy cost for logistic
regression is **convex** — a single bowl-shaped surface with **one global minimum** and no misleading
local dips — which is what makes the search reliable.

Rolling downhill
------------------

**Gradient descent** finds that minimum by repeatedly stepping **downhill**. At the current point the
**gradient** — the partial derivatives :math:`\partial J / \partial \mathbf{w}` and
:math:`\partial J / \partial b` — points in the direction of steepest **increase**; moving the
**opposite** way decreases the cost. Start anywhere (for a convex cost, even all-zeros works) and
repeat.

The update rule
-----------------

Each iteration nudges the parameters against the gradient:

.. math::

   \mathbf{w} := \mathbf{w} - \alpha\,\frac{\partial J}{\partial \mathbf{w}}, \qquad
   b := b - \alpha\,\frac{\partial J}{\partial b}.

In code the derivatives are conventionally named ``dw`` and ``db``, so the step reads
``w -= alpha * dw`` and ``b -= alpha * db``. Repeat until the cost stops decreasing.

The learning rate
-------------------

The step size :math:`\alpha` is the **learning rate**. Too **small** and training crawls, needing
many iterations; too **large** and the steps overshoot the minimum and may **diverge**. Choosing
:math:`\alpha` well — and computing those derivatives efficiently — is what the rest of this stage is
about, starting with the calculus itself.

.. hint::

   **Related lessons:** :doc:`Logistic Regression – Loss Function and Cost Function <07-logistic-regression-loss-function-and-cost-function>`  ·  :doc:`Derivatives <09-derivatives>`  ·  :doc:`Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`  ·  :doc:`Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/gradient-descent-in-logistic-regression/ <https://insightful-data-lab.com/2025/04/07/gradient-descent-in-logistic-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
