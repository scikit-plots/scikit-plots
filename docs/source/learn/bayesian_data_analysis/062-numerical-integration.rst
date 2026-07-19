.. _bda-numerical-integration:

========================================================================
Numerical integration
========================================================================

**Part 3 · Stage 8 · 🧰 Simulation Basics**  ·  Lesson 062 of 144  ·  *intermediate*

:doc:`◀ Previous · Personal vs. institutional decision analysis <061-personal-vs-institutional-decision-analysis>`   ·   :doc:`Next · Distributional approximations ▶ <063-distributional-approximations>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Every Bayesian answer is an integral
--------------------------------------

Posterior expectations, marginal distributions, predictive densities, the evidence: all are integrals
of the form :math:`\int h(\theta) \, p(\theta \mid y) \, d\theta`. Part I evaded them with conjugacy;
the bioassay showed how quickly that fails. Part III is about computing them when no formula exists.

Deterministic quadrature
--------------------------

The classical approach evaluates the integrand on a **grid** and sums with weights:

.. math::

   \int h(\theta) \, p(\theta \mid y) \, d\theta \;\approx\;
   \sum_{k=1}^{K} w_k \, h(\theta_k) \, p(\theta_k \mid y) .

Simple rules (trapezoid, Simpson) use equally spaced points; **Gaussian quadrature** places them
cleverly and achieves high accuracy with few evaluations. In **one or two dimensions** this is superb —
it is exactly what the bioassay grid did, and what the conjugate hierarchical model did for
:math:`(\alpha, \beta)`.

.. code-block:: python

   import numpy as np
   from scipy import integrate
   grid = np.linspace(-5, 10, 400)
   dens = np.exp(log_posterior(grid) - log_posterior(grid).max())
   dens /= integrate.trapezoid(dens, grid)              # normalise
   integrate.trapezoid(grid * dens, grid)               # E[theta | y]

The wall
----------

Quadrature dies of **dimensionality**. A grid of :math:`K` points per dimension costs :math:`K^d`
evaluations: 100 points in 10 dimensions is :math:`10^{20}` — impossible. Worse, in high dimensions
almost all of that grid lies where the posterior has **no mass**. The posterior of a modern model
concentrates in a thin, curved shell whose position you do not know in advance; enumerating the space is
hopeless.

Monte Carlo turns the problem around
--------------------------------------

Instead of choosing points and weighting by density, **draw points from the density** and weight
equally:

.. math::

   \int h(\theta) \, p(\theta \mid y) \, d\theta \;\approx\;
   \frac{1}{S} \sum_{s=1}^{S} h\bigl(\theta^{(s)}\bigr), \qquad
   \theta^{(s)} \sim p(\theta \mid y).

The error shrinks like :math:`1/\sqrt{S}` — **regardless of dimension**. That independence from
:math:`d` is the single fact on which all of modern Bayesian computation rests. The catch, of course, is
the premise: how do you draw from a distribution you can only evaluate up to a constant? The remaining
lessons of Part III are answers to that question.

.. hint::

   **Related lessons:** :doc:`Distributional approximations <063-distributional-approximations>`  ·  :doc:`Direct simulation and rejection sampling <064-direct-simulation-and-rejection-sampling>`  ·  :doc:`Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate) <026-example-bayesian-analysis-of-a-bioassay-experiment-logistic-nonconjugate>`  ·  :doc:`How many simulation draws are needed? <066-how-many-simulation-draws-are-needed>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/numerical-integration/ <https://insightful-data-lab.com/2025/11/11/numerical-integration/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
