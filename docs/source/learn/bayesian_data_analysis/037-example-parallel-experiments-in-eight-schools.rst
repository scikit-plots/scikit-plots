.. _bda-example-parallel-experiments-in-eight-schools:

========================================================================
Example: parallel experiments in eight schools
========================================================================

**Part 1 · Stage 5 · 🏛️ Hierarchical Models**  ·  Lesson 037 of 144  ·  *beginner*

:doc:`◀ Previous · Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>`   ·   :doc:`Next · Hierarchical modeling applied to a meta-analysis ▶ <038-hierarchical-modeling-applied-to-a-meta-analysis>`   ·   :doc:`↑ Section <index>`


Eight coaching programs
-------------------------

The most-analysed dataset in Bayesian statistics. The Educational Testing Service ran **separate
randomised experiments** in eight high schools to measure whether short-term coaching raised SAT-Verbal
scores. Each school reported an estimated effect :math:`y_j` and its standard error :math:`\sigma_j`,
both taken as given:

.. list-table::
   :header-rows: 1

   * - School
     - A
     - B
     - C
     - D
     - E
     - F
     - G
     - H
   * - Effect :math:`y_j`
     - 28
     - 8
     - -3
     - 7
     - -1
     - 1
     - 8
     - 12
   * - Std. error :math:`\sigma_j`
     - 15
     - 10
     - 16
     - 11
     - 9
     - 11
     - 10
     - 18

School A's effect of 28 points looks dramatic. Its standard error is 15. Should we believe it?

Three analyses
----------------

**No pooling** takes each :math:`y_j` at face value: school A's effect is 28. But the standard errors
are large, and the eight estimates are perfectly consistent with the schools being identical. **Complete
pooling** averages everything into one effect (about 8 points) and denies the schools differ at all.
Neither is credible. **Partial pooling** — the hierarchical normal model of the last lesson — lets the
data decide:

.. math::

   y_j \mid \theta_j \sim \mathrm{N}(\theta_j, \sigma_j^2), \qquad
   \theta_j \mid \mu, \tau \sim \mathrm{N}(\mu, \tau^2), \qquad
   \mu \sim \mathrm{N}(0, 5^2), \quad \tau \sim \text{half-Cauchy}(0, 5).

What the data say about :math:`\tau`
--------------------------------------

The pivotal quantity is :math:`\tau`, the **between-school** standard deviation. Its posterior is
concentrated near **zero** — the eight observed spreads are about what you would see if every school
had the *same* true effect and only sampling noise separated them. Consequently the shrinkage factors
:math:`B_j = \sigma_j^2/(\sigma_j^2 + \tau^2)` are large, and school A's estimate is pulled from 28 down
to roughly **10**, close to the pooled mean, with an interval that comfortably includes zero. The
posterior probability that school A's programme is the best is far below what its raw ranking suggests.

.. code-block:: python

   import numpy as np, pymc as pm
   y = np.array([28, 8, -3, 7, -1, 1, 8, 12])
   sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])

   with pm.Model():
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfCauchy("tau", 5)
       eta = pm.Normal("eta", 0, 1, shape=8)        # non-centred: avoids the funnel
       theta = pm.Deterministic("theta", mu + tau * eta)
       pm.Normal("y", theta, sigma=sigma, observed=y)
       idata = pm.sample(target_accept=0.9)
   # (idata.posterior["theta"].sel(theta_dim_0=0) > 28).mean()  -> tiny

The funnel
------------

Fit the **centred** version (:math:`\theta_j \sim \mathrm{N}(\mu, \tau)` directly) and the sampler
reports **divergences**. The reason is geometric: as :math:`\tau \to 0` the :math:`\theta_j` are squeezed
into an ever-narrowing neck, and no single step size explores both the neck and the wide mouth. The
**non-centred** parameterisation above — sampling standard-normal :math:`\eta_j` and rescaling —
decouples them and fixes it. Eight schools is a small dataset that teaches three large lessons:
shrinkage tames dramatic claims, :math:`\tau` is the parameter that matters, and **model geometry**
determines whether your sampler tells the truth.

.. hint::

   **Related lessons:** :doc:`Normal model with exchangeable parameters <036-normal-model-with-exchangeable-parameters>`  ·  :doc:`Hierarchical modeling applied to a meta-analysis <038-hierarchical-modeling-applied-to-a-meta-analysis>`  ·  :doc:`Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>`  ·  :doc:`Model checking for the educational testing example <044-model-checking-for-the-educational-testing-example>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/example-parallel-experiments-in-eight-schools/ <https://insightful-data-lab.com/2025/11/09/example-parallel-experiments-in-eight-schools/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
