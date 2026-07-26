.. _dl-logistic-regression-gradient-descent:

========================================================================
Logistic Regression Gradient Descent
========================================================================

**Stage 4 · ⚙️ Backprop & Vectorization**  ·  Lesson 13 of 17  ·  *intermediate*

:doc:`◀ Previous · Derivatives with a Computation Graph <12-derivatives-with-a-computation-graph>`   ·   :doc:`Next · Gradient Descent on m Training Examples ▶ <14-gradient-descent-on-m-training-examples>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The neuron's graph
--------------------

The computation-graph machinery now applies to the neuron itself. For a two-feature example the
forward graph is

.. math::

   z = w_1 x_1 + w_2 x_2 + b, \qquad a = \sigma(z) = \hat{y}, \qquad
   \mathcal{L}(a, y) = -\big(y \log a + (1 - y)\log(1 - a)\big).

Gradient descent needs :math:`\partial \mathcal{L}/\partial w_1`, :math:`\partial \mathcal{L}/\partial w_2`
and :math:`\partial \mathcal{L}/\partial b` — found by walking this graph backward.

Backprop, step by step
------------------------

Start at the loss and step back. The derivative of the loss with respect to the activation is

.. math::

   \mathrm{d}a = \frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1 - y}{1 - a}.

The sigmoid contributes :math:`\partial a / \partial z = a(1 - a)`, so by the chain rule
:math:`\mathrm{d}z = \mathrm{d}a \cdot a(1 - a)`.

The clean result
------------------

Those two pieces multiply out to something remarkably simple:

.. math::

   \mathrm{d}z = \frac{\partial \mathcal{L}}{\partial z} = a - y.

The gradient of the loss with respect to the pre-activation is just **prediction minus truth**. From
there the parameter gradients fall out immediately:

.. math::

   \mathrm{d}w_1 = x_1\,\mathrm{d}z, \qquad \mathrm{d}w_2 = x_2\,\mathrm{d}z, \qquad
   \mathrm{d}b = \mathrm{d}z.

Then the update
-----------------

With the gradients in hand, one gradient-descent step nudges each parameter downhill:

.. math::

   w_1 := w_1 - \alpha\,\mathrm{d}w_1, \qquad w_2 := w_2 - \alpha\,\mathrm{d}w_2, \qquad
   b := b - \alpha\,\mathrm{d}b.

That is a full learning step — for a **single** example. Real training averages over many, which is
the next lesson.

.. hint::

   **Related lessons:** :doc:`Derivatives with a Computation Graph <12-derivatives-with-a-computation-graph>`  ·  :doc:`Logistic Regression – Loss Function and Cost Function <07-logistic-regression-loss-function-and-cost-function>`  ·  :doc:`Gradient Descent on m Training Examples <14-gradient-descent-on-m-training-examples>`  ·  :doc:`Computation Graph <11-computation-graph>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/logistic-regression-gradient-descent/ <https://insightful-data-lab.com/2025/04/07/logistic-regression-gradient-descent/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
