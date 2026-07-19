.. _bda-computation-batching-and-transformation:

========================================================================
Computation: batching and transformation
========================================================================

**Part 4 · Stage 12 · 🏗️ Hierarchical Regression**  ·  Lesson 103 of 144  ·  *advanced*

:doc:`◀ Previous · Varying intercepts and slopes <102-varying-intercepts-and-slopes>`   ·   :doc:`Next · Analysis of variance and the batching of coeﬃcients ▶ <104-analysis-of-variance-and-the-batching-of-coefficients>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Making batched models sample
------------------------------

A model with several batches of exchangeable coefficients — varying intercepts, varying slopes, their
covariance — is easy to *write* and can be hard to *fit*. The geometry of hierarchical posteriors, first
met at eight schools, returns in force here, and two transformations make these models tractable.

The funnel, at scale
----------------------

Each batch carries the same pathology: when its scale :math:`\tau_b` is small, the coefficients in that
batch are squeezed into a narrow neck whose curvature the sampler cannot follow, and NUTS reports
divergences (Stage 9). With multiple batches the funnels compound. The **non-centred** parameterisation
is the standard cure — write each coefficient as a standard-normal draw scaled and shifted by the batch
parameters, so the raw parameters have a geometry independent of :math:`\tau_b`:

.. math::

   \beta_j = \mu_b + \tau_b \, \eta_j, \qquad \eta_j \sim \mathrm{N}(0, 1).

Apply it per batch, and — for varying intercepts and slopes together — to the **whole coefficient vector
at once** through the Cholesky factor of the group covariance.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       mu = pm.Normal("mu", 0, 5, shape=2)
       chol, _, _ = pm.LKJCholeskyCov("cov", n=2, eta=2,
                                      sd_dist=pm.HalfNormal.dist(1.0), compute_corr=True)
       z = pm.Normal("z", 0, 1, shape=(n_groups, 2))        # flat geometry
       ab = pm.Deterministic("ab", mu + z @ chol.T)         # correlated, non-centred
       mu_i = ab[grp, 0] + ab[grp, 1] * x
       pm.Normal("y", mu_i, pm.HalfNormal("s", 1), observed=y)

Transformation and scaling
----------------------------

The second lever is the same one that helped Gibbs. **Centre and scale predictors** so intercepts and
slopes decorrelate and share a common metric; a :math:`\mathrm{N}(0, 1)` prior then means the same thing
everywhere, and the sampler sees a roughly spherical posterior. Where a batch is data-rich, the
**centred** parameterisation is actually better conditioned — so the choice of centred versus non-centred
is per batch, decided by how much information each group carries, exactly as in the eight-schools lesson.

Practical workflow
--------------------

Three habits keep batched models honest. **Standardise inputs** before fitting. **Default to
non-centred** for batches with weak data per group, centred for strong. And **read the diagnostics**
per batch — divergences, :math:`\hat{R}`, and bulk/tail ESS for each :math:`\tau_b` — because a single
badly-parameterised batch can stall an otherwise healthy model. Batching organises the model; these
transformations are what let the sampler explore it.

.. hint::

   **Related lessons:** :doc:`Varying intercepts and slopes <102-varying-intercepts-and-slopes>`  ·  :doc:`Hamiltonian Monte Carlo for a hierarchical model <079-hamiltonian-monte-carlo-for-a-hierarchical-model>`  ·  :doc:`Eﬃcient Gibbs samplers <075-efficient-gibbs-samplers>`  ·  :doc:`Analysis of variance and the batching of coeﬃcients <104-analysis-of-variance-and-the-batching-of-coefficients>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/12/06/computation-batching-and-transformation/ <https://insightful-data-lab.com/2025/12/06/computation-batching-and-transformation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
