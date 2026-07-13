.. _bda-hamiltonian-monte-carlo-for-a-hierarchical-model:

========================================================================
Hamiltonian Monte Carlo for a hierarchical model
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 079 of 144  ·  *intermediate*

:doc:`◀ Previous · Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>`   ·   :doc:`Next · Stan: developing a computing environment ▶ <080-stan-developing-a-computing-environment>`   ·   :doc:`↑ Section <index>`


HMC meets the funnel
----------------------

Hierarchical models are where HMC's power and its fragility both appear. Return to eight schools in its
**centred** form, :math:`\theta_j \sim \mathrm{N}(\mu, \tau)`, and watch the sampler struggle exactly
where Gibbs crawled.

The geometry
--------------

Plot :math:`\log \tau` against any :math:`\theta_j`. The joint posterior is a **funnel**: when
:math:`\tau` is large, the :math:`\theta_j` are spread widely; when :math:`\tau` is small, they are
squeezed into a narrow neck. The posterior's **curvature** therefore changes by orders of magnitude
along :math:`\tau`, and a leapfrog step size that is stable in the wide mouth is catastrophically too
large in the neck.

The symptom is **divergences**, and they cluster at small :math:`\tau` — precisely the region that
decides whether the schools should be pooled. The sampler reports a value for :math:`\tau`; it is biased
away from zero, because the neck was never explored.

The cure is a change of coordinates
-------------------------------------

Write the same model so that the group parameters no longer depend on :math:`\tau`:

.. math::

   \theta_j = \mu + \tau \, \eta_j, \qquad \eta_j \sim \mathrm{N}(0, 1).

This is the **non-centred** parameterisation. The sampler now moves in :math:`(\mu, \tau, \eta)`, where
:math:`\eta` has a fixed standard-normal geometry independent of :math:`\tau`. The funnel is gone; one
step size fits everywhere.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu  = pm.Normal("mu", 0, 5)
       tau = pm.HalfCauchy("tau", 5)
       eta = pm.Normal("eta", 0, 1, shape=8)             # <- flat geometry
       theta = pm.Deterministic("theta", mu + tau * eta) # <- recovered by transformation
       pm.Normal("y", theta, sigma=sigma_j, observed=y)
       idata = pm.sample(target_accept=0.9)
   idata.sample_stats["diverging"].sum()                 # must be 0

Which parameterisation, when
------------------------------

Not always non-centred. When each group has **plenty of data** — large :math:`n_j`, or a large
:math:`\tau` — the likelihood pins each :math:`\theta_j` down and the **centred** form is better
conditioned; forcing the non-centred version then *creates* the correlation it was meant to remove. The
rule of thumb: **weak data per group → non-centred; strong data per group → centred**, and with mixed
groups, some models parameterise them differently.

The general lesson is the one that closes this stage. HMC's efficiency is determined by the posterior's
**geometry**, and geometry is something the modeller controls through parameterisation. A divergence is
not a complaint about the algorithm; it is information about the model.

.. hint::

   **Related lessons:** :doc:`Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>`  ·  :doc:`Example: parallel experiments in eight schools <037-example-parallel-experiments-in-eight-schools>`  ·  :doc:`Eﬃcient Gibbs samplers <075-efficient-gibbs-samplers>`  ·  :doc:`Weakly Informative Priors for Variance Parameters <039-weakly-informative-priors-for-variance-parameters>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/hamiltonian-monte-carlo-for-a-hierarchical-model/ <https://insightful-data-lab.com/2025/11/22/hamiltonian-monte-carlo-for-a-hierarchical-model/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
