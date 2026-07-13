.. _bda-using-gibbs-and-metropolis-as-building-blocks:

========================================================================
Using Gibbs and Metropolis as building blocks
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 071 of 144  ·  *intermediate*

:doc:`◀ Previous · Metropolis and Metropolis-Hastings algorithms <070-metropolis-and-metropolis-hastings-algorithms>`   ·   :doc:`Next · Inference and assessing convergence ▶ <072-inference-and-assessing-convergence>`   ·   :doc:`↑ Section <index>`


Mix and match
---------------

Gibbs and Metropolis are not rivals; they are **components**. A realistic model has some parameters with
tidy conjugate conditionals and others without. The natural sampler updates each block by whatever
method suits it, and the composition still targets the correct posterior.

Metropolis-within-Gibbs
-------------------------

Sweep through the parameter blocks as in Gibbs. For a block whose full conditional is a standard
distribution, **draw from it** (acceptance 1). For a block whose conditional is only known up to a
constant, take a **Metropolis step** targeting that conditional. Each update leaves the posterior
invariant, so the composed chain does too.

.. code-block:: python

   import numpy as np
   from scipy import stats
   for t in range(n_iter):
       # block 1: conjugate -> exact Gibbs draw
       sigma2 = stats.invgamma(a + n / 2, scale=b + 0.5 * ((y - X @ beta) ** 2).sum()).rvs()
       # block 2: no closed form -> Metropolis step on the conditional
       prop = beta + step * np.random.standard_normal(len(beta))
       if np.log(np.random.rand()) < log_cond(prop, sigma2) - log_cond(beta, sigma2):
           beta = prop

Blocking and reparameterising
-------------------------------

The two levers that fix slow mixing, both aimed at **posterior correlation**:

* **Blocking** — update strongly dependent parameters **jointly** rather than one at a time. Gibbs
  moves along axes; a correlated pair traversed jointly moves along the ridge. Regression coefficients
  under a conjugate prior can be drawn as a whole vector.
* **Reparameterising** — change coordinates so the posterior is less correlated. Centre predictors;
  replace :math:`\theta_j \sim \mathrm{N}(\mu, \tau^2)` with :math:`\theta_j = \mu + \tau \eta_j`,
  :math:`\eta_j \sim \mathrm{N}(0,1)`. The **non-centred** parameterisation from eight schools is exactly
  this move, and it works for Gibbs and HMC alike.

Auxiliary variables
---------------------

A third trick: **add** parameters to make the conditionals tractable. Data augmentation introduces latent
variables (a :math:`t` distribution written as a scale-mixture of normals; a probit model given latent
normals) whose presence turns an awkward joint into a chain of conjugate conditionals. You sample in a
larger space and **discard the extra columns**, which — per Part I — is marginalisation.

The upshot
------------

These compositions dominated Bayesian computation for two decades, and they still matter: they explain
what BUGS and JAGS do, they remain the right tool for discrete parameters (which gradient methods cannot
touch), and their diagnostics are the subject of the next lesson. Their limitation is universal — all of
them explore by **random walk**, and none escape its quadratic cost.

.. hint::

   **Related lessons:** :doc:`Gibbs sampler <069-gibbs-sampler>`  ·  :doc:`Metropolis and Metropolis-Hastings algorithms <070-metropolis-and-metropolis-hastings-algorithms>`  ·  :doc:`Eﬃcient Gibbs samplers <075-efficient-gibbs-samplers>`  ·  :doc:`Further extensions to Gibbs and Metropolis <077-further-extensions-to-gibbs-and-metropolis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/12/using-gibbs-and-metropolis-as-building-blocks/ <https://insightful-data-lab.com/2025/11/12/using-gibbs-and-metropolis-as-building-blocks/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
