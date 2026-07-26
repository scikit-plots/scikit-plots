.. _bda-finding-posterior-modes:

========================================================================
Finding posterior modes
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 081 of 144  ·  *intermediate*

:doc:`◀ Previous · Stan: developing a computing environment <080-stan-developing-a-computing-environment>`   ·   :doc:`Next · Boundary-avoiding priors for modal summaries ▶ <082-boundary-avoiding-priors-for-modal-summaries>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Optimisation instead of sampling
----------------------------------

Sampling explores the posterior; **optimisation** finds its peak. The **posterior mode** — the MAP
estimate — is the cheapest possible summary, obtained by maximising :math:`\log p(\theta \mid y)` with
the same machinery used for maximum likelihood, since a prior is just an additive penalty on the log
scale.

How
-----

Because the objective is a sum of log-likelihood and log-prior, any gradient-based optimiser applies.
Newton-type methods use the curvature and give the **observed information** for free, which is exactly
what the Laplace approximation of Stage 4 needs:

.. code-block:: python

   import numpy as np
   from scipy import optimize

   neg_log_post = lambda t: -(log_lik(t, y) + log_prior(t))
   fit = optimize.minimize(neg_log_post, x0, method="BFGS")
   mode = fit.x
   cov = fit.hess_inv            # inverse observed information ~ posterior covariance
   sd = np.sqrt(np.diag(cov))

Practical advice from the sampling lessons carries over: optimise on an **unconstrained** scale
(:math:`\log \sigma`, :math:`\mathrm{logit}\, \theta`), start from several dispersed points to detect
multiple modes, and rescale so parameters are of order one.

Why the mode can mislead
--------------------------

Three failures deserve naming. The mode is **not invariant** to reparameterisation: the mode of
:math:`\sigma` is not the transform of the mode of :math:`\log \sigma`, because densities pick up a
Jacobian — whereas posterior *means* of transformed draws behave sensibly. The mode may sit at a
**boundary** (a variance of exactly zero), which the next lesson addresses. And in **high dimensions**
the mode is not where the probability is: almost all posterior mass lies in a thin shell away from the
peak, so a point estimate at the mode is unrepresentative of the draws you would obtain by sampling.

Where it belongs
------------------

The mode remains useful in three roles, all honest. As a **fast approximate answer** for large, regular
problems, where large-sample theory says the posterior is nearly normal around it. As **starting values
and a metric** for MCMC — the inverse Hessian is a good initial mass matrix. And as the point estimate
that connects Bayesian analysis to **penalised likelihood** (Stage 4): ridge, lasso and multilevel
point-estimation software are all computing posterior modes. What the mode never provides is
**uncertainty**; the curvature around it is an approximation whose adequacy must itself be checked.

.. hint::

   **Related lessons:** :doc:`Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>`  ·  :doc:`Boundary-avoiding priors for modal summaries <082-boundary-avoiding-priors-for-modal-summaries>`  ·  :doc:`Finding marginal posterior modes using EM <084-finding-marginal-posterior-modes-using-em>`  ·  :doc:`Bayesian interpretations of other statistical methods <032-bayesian-interpretations-of-other-statistical-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/finding-posterior-modes/ <https://insightful-data-lab.com/2025/11/22/finding-posterior-modes/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
