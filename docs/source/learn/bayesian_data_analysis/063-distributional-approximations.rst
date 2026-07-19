.. _bda-distributional-approximations:

========================================================================
Distributional approximations
========================================================================

**Part 3 · Stage 8 · 🧰 Simulation Basics**  ·  Lesson 063 of 144  ·  *intermediate*

:doc:`◀ Previous · Numerical integration <062-numerical-integration>`   ·   :doc:`Next · Direct simulation and rejection sampling ▶ <064-direct-simulation-and-rejection-sampling>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Replace the posterior with something tractable
------------------------------------------------

Before sampling, consider **approximating**. If a simple distribution stands in for the posterior, every
integral becomes analytic or trivially simulated. The approximation is never exact, but it is fast, and
it is often the right first move — and, as it turns out, a component of the sophisticated methods later.

The normal approximation, revisited
-------------------------------------

The workhorse is the **Laplace** approximation from Stage 4: a normal centred at the posterior mode with
covariance the inverse observed information. Two refinements matter in practice.

**Work on a transformed scale.** A posterior for :math:`\sigma > 0` is skewed and bounded; the posterior
for :math:`\log \sigma` is far closer to normal. Approximate there, then transform the draws back — the
Jacobian handles itself, because functions of draws are draws.

**Use a** :math:`t` **instead of a normal** when the approximation will serve as a proposal or
importance-sampling density: heavier tails guarantee the approximation **covers** the target, which
matters enormously for the stability of the weights.

.. code-block:: python

   import numpy as np
   from scipy import optimize, stats

   fit = optimize.minimize(lambda t: -log_post(t), x0, method="BFGS")
   mode, cov = fit.x, fit.hess_inv
   approx = stats.multivariate_normal(mode, cov)         # Laplace
   # heavier-tailed variant for use as a proposal:
   draws = mode + stats.multivariate_t(shape=cov, df=4).rvs(1000)

Mixtures, and the modern descendants
--------------------------------------

A single normal cannot represent a **multimodal** posterior. Fitting a **mixture** of normals — one
component per mode found by repeated optimisation from dispersed starting points — extends the reach at
modest cost. Push the idea further and you arrive at the methods of Stage 10: **variational inference**,
which fits the best member of a chosen family by minimising a divergence, and **expectation propagation**,
which matches moments locally.

Honest uses
-------------

An approximation is trustworthy when the posterior is smooth, unimodal and not near a boundary — which
large-sample theory says becomes true as :math:`n` grows. It is untrustworthy exactly where Bayesian
methods are most valuable: small samples, hierarchical variance parameters, weakly identified models.
So use approximations for **starting values**, for **proposal distributions**, for a **quick first
look**, and for problems too large to sample — but check them against MCMC where you can, and report
which you used.

.. hint::

   **Related lessons:** :doc:`Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>`  ·  :doc:`Numerical integration <062-numerical-integration>`  ·  :doc:`Variational inference <087-variational-inference>`  ·  :doc:`Importance sampling <065-importance-sampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/11/distributional-approximations/ <https://insightful-data-lab.com/2025/11/11/distributional-approximations/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
