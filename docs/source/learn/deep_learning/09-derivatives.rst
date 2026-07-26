.. _dl-derivatives:

========================================================================
Derivatives
========================================================================

**Stage 3 · 📉 Derivatives & the Computation Graph**  ·  Lesson 09 of 17  ·  *intermediate*

:doc:`◀ Previous · Gradient Descent in Logistic Regression <08-gradient-descent-in-logistic-regression>`   ·   :doc:`Next · More Derivative Examples ▶ <10-more-derivative-examples>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Derivative means slope
------------------------

Gradient descent runs on **derivatives**, so this lesson pins down what a derivative actually is: a
**slope**. The derivative of :math:`f` at a point measures **how much** :math:`f` **changes when its
input changes by a tiny amount** — the steepness of the function right there.

A worked nudge
----------------

Take the straight line :math:`f(a) = 3a`. At :math:`a = 2`, :math:`f = 6`. Nudge the input a hair, to
:math:`a = 2.001`; then :math:`f = 6.003`. The output moved by :math:`0.003` when the input moved by
:math:`0.001` — **three times as much**. That ratio *is* the derivative: :math:`\frac{df}{da} = 3`.

Slope of a straight line
--------------------------

Geometrically the derivative is **rise over run** — the height of a tiny triangle divided by its
width. For a **straight line** the slope is the **same everywhere**: :math:`f(a) = 3a` has derivative
:math:`3` at every point, whether :math:`a` is 2, 5 or :math:`-10`. (Curved functions have a slope
that *changes* from point to point — the subject of the next lesson.)

Why it matters here
---------------------

Every gradient in this course is built from derivatives like this one. Knowing that the derivative is
just "**how fast the output moves when you wiggle the input**" is enough to follow the **chain rule**
and **backpropagation** ahead — the machinery that computes :math:`\partial J / \partial \mathbf{w}`
and :math:`\partial J / \partial b` for a whole network.

.. hint::

   **Related lessons:** :doc:`More Derivative Examples <10-more-derivative-examples>`  ·  :doc:`Computation Graph <11-computation-graph>`  ·  :doc:`Gradient Descent in Logistic Regression <08-gradient-descent-in-logistic-regression>`  ·  :doc:`Derivatives with a Computation Graph <12-derivatives-with-a-computation-graph>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/04/07/derivatives/ <https://insightful-data-lab.com/2025/04/07/derivatives/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: deep learning, level: intermediate
