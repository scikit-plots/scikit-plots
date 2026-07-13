.. _bda-effective-number-of-simulation-draws:

========================================================================
Eﬀective number of simulation draws
========================================================================

**Part 3 · Stage 9 · ⛓️ MCMC: Gibbs, Metropolis & HMC**  ·  Lesson 073 of 144  ·  *intermediate*

:doc:`◀ Previous · Inference and assessing convergence <072-inference-and-assessing-convergence>`   ·   :doc:`Next · Example: hierarchical normal model ▶ <074-example-hierarchical-normal-model>`   ·   :doc:`↑ Section <index>`


Correlated draws carry less information
-----------------------------------------

An MCMC chain's draws are **dependent** by construction: each is a perturbation of the last. A thousand
such draws therefore say less about the posterior than a thousand independent ones. The **effective
sample size** measures how much less.

The definition
----------------

If the chain's autocorrelation at lag :math:`t` is :math:`\rho_t`, then

.. math::

   S_{\text{eff}} = \frac{S}{1 + 2 \sum_{t=1}^{\infty} \rho_t} .

Independent draws (:math:`\rho_t = 0`) give :math:`S_{\text{eff}} = S`. Positively autocorrelated draws
— the usual case for random-walk Metropolis and Gibbs — give :math:`S_{\text{eff}} \ll S`. The estimate
truncates the sum once the estimated :math:`\rho_t` becomes noise, and in practice pools information
across chains.

Better than independent
-------------------------

A pleasing surprise: the sum can be **negative**. When successive draws are **anticorrelated**, the
effective sample size **exceeds** the number of iterations, :math:`S_{\text{eff}} > S`. Hamiltonian
Monte Carlo, and the NUTS sampler in particular, routinely achieves this for parameters whose posterior
is close to Gaussian with little dependence on the others — the chain deliberately overshoots, so
consecutive draws sit on opposite sides of the mode. For such problems MCMC beats independent sampling.

Bulk and tail
---------------

One number is not enough. **bulk-ESS** governs the precision of the mean and standard deviation;
**tail-ESS** governs the extreme quantiles, and a chain that mixes briskly through the centre can crawl
in the tails. Report both, and require both above roughly **400** before trusting any summary — including
:math:`\hat{R}` itself, whose reliability depends on adequate ESS.

.. code-block:: python

   import arviz as az
   az.summary(idata)                    # ess_bulk, ess_tail, mcse_mean, mcse_sd, r_hat
   az.ess(idata, method="tail")         # tail-ESS explicitly
   az.plot_autocorr(idata, var_names=["tau"])   # where does rho_t decay to zero?

Don't thin
------------

A persistent habit deserves retiring. **Thinning** — keeping every tenth draw — does not improve
inference. Discarding draws throws away information; the effective sample size of the thinned chain is
at best what the full chain already carried, and if the draws are anticorrelated, thinning actively
**destroys** the advantage. The only legitimate reason to thin is **memory**. If ESS is too low, the
answer is a better sampler, a reparameterisation, or a longer run — never a smaller one.

.. hint::

   **Related lessons:** :doc:`Inference and assessing convergence <072-inference-and-assessing-convergence>`  ·  :doc:`How many simulation draws are needed? <066-how-many-simulation-draws-are-needed>`  ·  :doc:`Eﬃcient Metropolis jumping rules <076-efficient-metropolis-jumping-rules>`  ·  :doc:`Hamiltonian Monte Carlo <078-hamiltonian-monte-carlo>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/12/e%ef%ac%80ective-number-of-simulation-draws/ <https://insightful-data-lab.com/2025/11/12/e%ef%ac%80ective-number-of-simulation-draws/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
