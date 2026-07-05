.. _dl-more-derivative-examples:

========================================================================
More Derivative Examples
========================================================================

**Stage 3 · 📉 Derivatives & the Computation Graph**  ·  Lesson 10 of 17  ·  *intermediate*

:doc:`◀ Previous · Derivatives <09-derivatives>`   ·   :doc:`Next · Computation Graph <11-computation-graph> ▶`


When the slope changes
------------------------

The line :math:`f(a) = 3a` had the **same slope everywhere**. Most functions do not — their
**derivative changes from point to point**. This lesson makes that concrete, because a neuron's
sigmoid is exactly such a curve.

A curved example
------------------

Take :math:`f(a) = a^2`. At :math:`a = 2`, :math:`f = 4`; nudge to :math:`a = 2.001` and
:math:`f \approx 4.004` — a slope of about **4**. But at :math:`a = 5`, :math:`f = 25`; nudge to
:math:`5.001` and :math:`f \approx 25.010` — a slope of about **10**. The slope is **twice the
input**, which is exactly the rule

.. math::

   f(a) = a^2 \;\Rightarrow\; \frac{df}{da} = 2a.

The derivative is now a **function of** :math:`a`, not a constant.

A few more
------------

The same pattern holds across the standard functions — :math:`f(a) = a^3` has derivative
:math:`3a^2`, and :math:`f(a) = \ln a` has derivative :math:`1/a`. You need not re-derive these from
nudges each time; they are tabulated in any calculus reference. What matters is reading them the same
way: *how fast does the output move as I wiggle the input, right here?*

The takeaway
--------------

For a **curve**, "the derivative" always means the slope **at a particular point**. That single idea
— a slope that varies — is all the calculus the rest of the course needs. Next we organise a
multi-step computation so these per-point slopes can be combined **mechanically**, through a
**computation graph**.

.. seealso::

   **Related lessons:** :doc:`Derivatives <09-derivatives>`  ·  :doc:`Computation Graph <11-computation-graph>`  ·  :doc:`Derivatives with a Computation Graph <12-derivatives-with-a-computation-graph>`  ·  :doc:`Logistic Regression Gradient Descent <13-logistic-regression-gradient-descent>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2025/04/07/more-derivative-examples/ <https://insightful-data-lab.com/2025/04/07/more-derivative-examples/>`__

.. tags:: purpose: reference, topic: deep learning, level: intermediate
