.. _bda-other-approximations:

========================================================================
Other approximations
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 089 of 144  ·  *intermediate*

:doc:`◀ Previous · Expectation propagation <088-expectation-propagation>`   ·   :doc:`Next · Unknown normalizing factors ▶ <090-unknown-normalizing-factors>`   ·   :doc:`↑ Section <index>`


The wider family
------------------

Laplace, EM, variational inference and expectation propagation are the landmarks; several other methods
occupy the space between them, each trading accuracy for speed in a different currency.

INLA
------

**Integrated nested Laplace approximation** (Rue, Martino and Chopin) is the standout for a specific,
large class: **latent Gaussian models**, in which observations are conditionally independent given a
latent Gaussian field, whose precision matrix depends on a few hyperparameters :math:`\theta`. That
class covers most hierarchical regressions, spatial and spatio-temporal models, and smoothers.

INLA implements the conditional/marginal factorisation of two lessons ago, twice over. It approximates
the hyperparameter marginal by evaluating the joint against a Gaussian approximation of the latent field
at its mode,

.. math::

   \tilde{p}(\theta \mid y) \;\propto\;
   \left. \frac{p(x, \theta, y)}{\tilde{p}_G(x \mid \theta, y)} \right|_{x = x^{*}(\theta)},

then recovers each latent marginal by **numerical integration** over a small grid of :math:`\theta`
values, :math:`p(x_i \mid y) \approx \sum_k p(x_i \mid y, \theta_k) \, \tilde{p}(\theta_k \mid y) \,
\Delta_k`. Being deterministic, it has **no mixing to diagnose** and no chains to run — minutes where
MCMC takes hours. Its price is the model class: leave latent-Gaussian territory and INLA does not apply.

Others in brief
-----------------

* **Laplace / nested Laplace** — the building block of INLA; excellent when the integrand is unimodal
  and smooth.
* **Pseudo-marginal and particle methods** — replace an intractable likelihood with an **unbiased
  estimate** inside the Metropolis ratio; remarkably, the chain still targets the exact posterior.
* **Approximate Bayesian computation (ABC)** — when the likelihood cannot even be evaluated but data can
  be **simulated**: accept parameter draws whose simulated summaries land near the observed ones.
* **Stochastic-gradient MCMC** — subsample the data per step; scalable, biased, and increasingly
  understood.

Choosing
----------

The decision rests on two questions. **Is your model in a class with a specialised method?** (Latent
Gaussian → INLA; simulator-only → ABC.) And **what will the answer be used for?** Point estimates
tolerate crude approximations; tail probabilities and hierarchical variance parameters do not.

.. code-block:: python

   import arviz as az
   # whatever the approximation, check it from outside:
   #   PSIS-reweight the approximate draws toward the true posterior
   logw = log_posterior(draws) - approx.logpdf(draws)
   #   k_hat < 0.7 -> the approximation is usable; larger -> do not trust it
   az.psislw(logw)

That last line is the discipline of this whole stage. Every approximation here is silent about its own
error; **importance-reweighting supplies the missing diagnostic**, and where the model is small enough,
so does a run of the sampler you were trying to avoid.

.. hint::

   **Related lessons:** :doc:`Variational inference <087-variational-inference>`  ·  :doc:`Expectation propagation <088-expectation-propagation>`  ·  :doc:`Conditional and marginal posterior approximations <085-conditional-and-marginal-posterior-approximations>`  ·  :doc:`Unknown normalizing factors <090-unknown-normalizing-factors>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/other-approximations/ <https://insightful-data-lab.com/2025/11/23/other-approximations/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
