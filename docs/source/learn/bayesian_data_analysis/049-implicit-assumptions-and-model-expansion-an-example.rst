.. _bda-implicit-assumptions-and-model-expansion-an-example:

========================================================================
Implicit assumptions and model expansion: an example
========================================================================

**Part 2 · Stage 6 · 🔍 Model Checking & Comparison**  ·  Lesson 049 of 144  ·  *intermediate*

:doc:`◀ Previous · Continuous model expansion <048-continuous-model-expansion>`   ·   :doc:`Next · Bayesian inference requires a model for data collection ▶ <050-bayesian-inference-requires-a-model-for-data-collection>`   ·   :doc:`↑ Section <index>`


The assumptions you forgot you made
-------------------------------------

Every model states some assumptions **explicitly** — the likelihood, the prior — and smuggles in others
**implicitly**, in the very act of writing it down. Implicit assumptions are dangerous precisely because
they are invisible: nothing in the output flags them, and a posterior predictive check will not test a
feature the model was never asked about. Expansion is how you drag them into the open.

Inventory the silences
------------------------

Take the eight-schools model of Stage 5, an analysis that passed its checks:

.. math::

   y_j \mid \theta_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad
   \theta_j \mid \mu, \tau \sim \mathrm{N}(\mu, \tau^2) .

Four assumptions were never stated. That the school effects are **normally distributed** around
:math:`\mu` — no outlying school with a genuinely different programme. That the standard errors
:math:`\sigma_j` are **known exactly**, rather than estimated from modest samples. That the schools are
**exchangeable** — nothing about size, funding or curriculum predicts the effect. And that each school's
estimate is **unbiased** for its own effect, so no selective reporting occurred.

Expand to test
----------------

Each silent assumption becomes a parameter, and the posterior for that parameter tells you whether it
mattered. Replace the normal population with a :math:`t`:

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfCauchy("tau", 5)
       nu  = pm.Gamma("nu", 2, 0.1)            # was: implicitly infinity (normal)
       eta = pm.StudentT("eta", nu=nu, mu=0, sigma=1, shape=8)
       theta = pm.Deterministic("theta", mu + tau * eta)
       pm.Normal("y", theta, sigma=sigma_j, observed=y)

If the posterior for :math:`\nu` piles up at large values, the normal assumption was harmless and you
have **earned** the right to it. If :math:`\nu` concentrates near 3, one school is an outlier, and the
original analysis over-shrank it. Similarly, letting :math:`\theta_j` depend on a school-level covariate
tests exchangeability; giving :math:`\sigma_j` its own uncertainty tests the known-variance assumption.

The general move
------------------

The pattern is: **name the assumption, embed it in a family, fit, inspect the posterior of the embedding
parameter.** This converts an untestable premise into an estimated quantity. Where the data are
informative the expansion settles the question; where they are not, the posterior for the new parameter
simply reproduces its prior — which is itself the answer: *the data cannot tell*, and the conclusion
rests on judgement rather than evidence.

Two honest limits
-------------------

Not every implicit assumption can be expanded away. The claim that :math:`\sigma_j` describes the right
sampling process, or that no results were suppressed before publication, lies **outside** the
likelihood — no parameter inside the model can detect it. And each expansion costs identifiability:
with :math:`J = 8`, :math:`\nu` will be weakly determined, so the check reports "no evidence against"
rather than "assumption verified". Model expansion widens what the data can speak to; it does not make
them omniscient.

.. hint::

   **Related lessons:** :doc:`Continuous model expansion <048-continuous-model-expansion>`  ·  :doc:`Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>`  ·  :doc:`Model checking for the educational testing example <044-model-checking-for-the-educational-testing-example>`  ·  :doc:`Robust inference for the eight schools <116-robust-inference-for-the-eight-schools>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/10/implicit-assumptions-and-model-expansion-an-example/ <https://insightful-data-lab.com/2025/11/10/implicit-assumptions-and-model-expansion-an-example/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
