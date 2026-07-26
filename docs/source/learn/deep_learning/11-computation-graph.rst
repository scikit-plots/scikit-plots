.. _dl-computation-graph:

========================================================================
Computation Graph
========================================================================

**Stage 3 · 📉 Derivatives & the Computation Graph**  ·  Lesson 11 of 17  ·  *intermediate*

:doc:`◀ Previous · More Derivative Examples <10-more-derivative-examples>`   ·   :doc:`Next · Derivatives with a Computation Graph ▶ <12-derivatives-with-a-computation-graph>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Breaking it into steps
------------------------

A **computation graph** writes a calculation as a chain of **elementary steps**, each a node feeding
the next. It looks like extra bookkeeping, but it is the structure that makes **backpropagation** —
computing every derivative in one sweep — both possible and efficient.

A small example
-----------------

Ng's example computes :math:`J = 3(a + bc)`. Broken into steps it becomes three nodes:

.. math::

   u = bc, \qquad v = a + u, \qquad J = 3v.

Each intermediate value (:math:`u`, then :math:`v`, then :math:`J`) depends only on ones already
computed — a strict left-to-right flow from inputs :math:`a, b, c` to output :math:`J`.

The forward pass
------------------

Computing the graph **left to right** is the **forward pass**: plug in the inputs and fill in each
node. With :math:`a = 5, b = 3, c = 2`:

.. math::

   u = bc = 6, \qquad v = a + u = 11, \qquad J = 3v = 33.

This is exactly the forward propagation that produces a neuron's prediction — inputs in, output out.

Why bother
------------

The payoff comes next. Because the graph records **how each value was built from the previous ones**,
we can walk it **backward** and apply the chain rule step by step, computing
:math:`\partial J / \partial a`, :math:`\partial J / \partial b`, :math:`\partial J / \partial c`
without ever untangling the whole nested expression at once.

.. hint::

   **Related lessons:** :doc:`Derivatives with a Computation Graph <12-derivatives-with-a-computation-graph>`  ·  :doc:`Derivatives <09-derivatives>`  ·  :doc:`Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`  ·  :doc:`Gradient Descent in Logistic Regression <08-gradient-descent-in-logistic-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/computation-graph/ <https://insightful-data-lab.com/2025/04/07/computation-graph/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
