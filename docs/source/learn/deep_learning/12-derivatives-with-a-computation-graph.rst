.. _dl-derivatives-with-a-computation-graph:

========================================================================
Derivatives with a Computation Graph
========================================================================

**Stage 3 · 📉 Derivatives & the Computation Graph**  ·  Lesson 12 of 17  ·  *intermediate*

:doc:`◀ Previous · Computation Graph <11-computation-graph>`   ·   :doc:`Next · Logistic Regression Gradient Descent ▶ <13-logistic-regression-gradient-descent>`   ·   :doc:`↑ Section <index>`


Walking backward
------------------

With the forward pass done, the **backward pass** computes derivatives by moving **right to left**
through the same graph, applying the **chain rule** at each node. This right-to-left sweep is exactly
what **backpropagation** does.

One step at a time
--------------------

Start at the output. Since :math:`J = 3v`, nudging :math:`v` moves :math:`J` three times as much, so
:math:`\partial J / \partial v = 3`. Step back through :math:`v = a + u`:

.. math::

   \frac{\partial J}{\partial a} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial a}
     = 3 \cdot 1 = 3, \qquad
   \frac{\partial J}{\partial u} = \frac{\partial J}{\partial v}\frac{\partial v}{\partial u}
     = 3 \cdot 1 = 3.

Then back through :math:`u = bc`:

.. math::

   \frac{\partial J}{\partial b} = \frac{\partial J}{\partial u}\frac{\partial u}{\partial b}
     = 3c = 6, \qquad
   \frac{\partial J}{\partial c} = \frac{\partial J}{\partial u}\frac{\partial u}{\partial c}
     = 3b = 9,

using :math:`a = 5, b = 3, c = 2`.

The dvar convention
---------------------

A coding shorthand runs through the whole course: the variable named **``dvar``** always means "the
derivative of the **final output** :math:`J` with respect to :math:`\text{var}`". So ``dv`` is
:math:`\partial J / \partial v`, ``da`` is :math:`\partial J / \partial a`, ``du`` is
:math:`\partial J / \partial u`, and so on — every ``d``-something is a gradient of the same target,
which keeps the code readable.

Reuse is the point
--------------------

Notice that :math:`\partial J / \partial v` was computed **once** and then **reused** for both
:math:`\partial J / \partial a` and :math:`\partial J / \partial u`, and :math:`\partial J / \partial u`
was reused for :math:`b` and :math:`c`. That reuse is why the backward pass is **efficient**: each
node's derivative follows from the one just downstream, so the whole gradient costs about as much as
the forward pass. Applying this to the logistic-regression neuron is the next lesson.

.. hint::

   **Related lessons:** :doc:`Computation Graph <11-computation-graph>`  ·  :doc:`More Derivative Examples <10-more-derivative-examples>`  ·  :doc:`Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`  ·  :doc:`Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/derivatives-with-a-computation-graph/ <https://insightful-data-lab.com/2025/04/07/derivatives-with-a-computation-graph/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
