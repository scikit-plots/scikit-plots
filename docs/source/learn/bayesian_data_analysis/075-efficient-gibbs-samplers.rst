.. _bda-efficient-gibbs-samplers:

========================================================================
Eﬃcient Gibbs samplers
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 075 of 144  ·  *intermediate*

:doc:`◀ Previous · Example: hierarchical normal model <074-example-hierarchical-normal-model>`   ·   :doc:`Next · Eﬃcient Metropolis jumping rules ▶ <076-efficient-metropolis-jumping-rules>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Fixing what slows Gibbs down
------------------------------

Gibbs moves parallel to the coordinate axes. Its efficiency is therefore governed entirely by the
**posterior correlation** among the parameters it updates separately: a narrow diagonal ridge is
traversed in tiny axis-aligned steps. Three standard cures attack that correlation directly.

Reparameterise
----------------

Change coordinates so the posterior is closer to spherical.

* **Centring predictors.** In :math:`y = \alpha + \beta x`, an uncentred :math:`x` makes
  :math:`\alpha` and :math:`\beta` strongly correlated; subtracting :math:`\bar{x}` makes them nearly
  independent, and Gibbs mixes immediately.
* **Non-centred hierarchies.** Replace :math:`\theta_j \sim \mathrm{N}(\mu, \tau^2)` with
  :math:`\theta_j = \mu + \tau \eta_j`, :math:`\eta_j \sim \mathrm{N}(0, 1)`. The funnel of the previous
  lesson vanishes because :math:`\eta` no longer depends on :math:`\tau`.
* **Rescaling.** Put parameters on comparable scales so no single conditional dominates.

Which parameterisation wins depends on the data: **centred** works when groups are informative (large
:math:`n_j`, large :math:`\tau`); **non-centred** when they are not. Hierarchical models with weak data
per group want the non-centred form.

Block
-------

Update correlated parameters **jointly**, drawing from their joint conditional. In a conjugate linear
model the entire coefficient vector :math:`\beta` is drawn in one multivariate normal step rather than
coordinate by coordinate, and correlation among coefficients stops mattering at all.

.. code-block:: python

   import numpy as np
   from scipy import stats
   # blocked draw of the whole coefficient vector, given sigma^2
   V = np.linalg.inv(X.T @ X / sigma**2 + np.linalg.inv(Sigma0))
   m = V @ (X.T @ y / sigma**2 + np.linalg.inv(Sigma0) @ mu0)
   beta = stats.multivariate_normal(m, V).rvs()      # one step, any correlation

Augment
---------

Add **auxiliary variables** that make the conditionals conjugate. A :math:`t` likelihood is a
scale-mixture of normals: introduce per-observation scale parameters and every conditional becomes
standard. Probit regression given latent normal utilities is conjugate throughout. You sample in a
bigger space, then **drop the extra columns** — marginalisation, again.

The residual limit
--------------------

Even a well-tuned Gibbs sampler explores by **random walk**, and its cost grows quadratically with the
distance it must travel. Reparameterising and blocking buy constants, sometimes large ones. Escaping the
random walk itself requires **gradients**, which is Hamiltonian Monte Carlo.

.. hint::

   **Related lessons:** :doc:`Gibbs sampler <069-gibbs-sampler>`  ·  :doc:`Example: hierarchical normal model <074-example-hierarchical-normal-model>`  ·  :doc:`Using Gibbs and Metropolis as building blocks <071-using-gibbs-and-metropolis-as-building-blocks>`  ·  :doc:`Computation: batching and transformation <103-computation-batching-and-transformation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/21/e%ef%ac%83cient-gibbs-samplers/ <https://insightful-data-lab.com/2025/11/21/e%ef%ac%83cient-gibbs-samplers/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
