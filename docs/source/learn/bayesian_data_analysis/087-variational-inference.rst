.. _bda-variational-inference:

========================================================================
Variational inference
========================================================================

**Part 3 · Stage 10 · 🎛️ Modal & Variational Approximation**  ·  Lesson 087 of 144  ·  *intermediate*

:doc:`◀ Previous · Example: hierarchical normal model (continued) <086-example-hierarchical-normal-model-continued>`   ·   :doc:`Next · Expectation propagation ▶ <088-expectation-propagation>`   ·   :doc:`↑ Section <index>`


Turn inference into optimisation
----------------------------------

MCMC samples the posterior; **variational inference** *approximates* it, by choosing the member of a
tractable family :math:`\mathcal{Q}` that is closest to it. Inference becomes optimisation, which scales
to models and datasets where sampling is hopeless.

The ELBO
----------

Closeness is measured by Kullback–Leibler divergence, and the decomposition is exact:

.. math::

   \log p(y) = \underbrace{\mathrm{E}_{q}\bigl[\log p(\theta, y)\bigr] - \mathrm{E}_{q}\bigl[\log q(\theta)\bigr]}_{\text{ELBO}(q)}
             \;+\; \mathrm{KL}\bigl(q(\theta) \,\|\, p(\theta \mid y)\bigr) .

Since :math:`\log p(y)` is fixed and the KL term is non-negative, **maximising the ELBO minimises the
divergence** — and the intractable evidence never has to be computed. The ELBO's two pieces read
naturally: fit the joint density well, but keep the approximation's entropy high.

Families, and the mean-field default
--------------------------------------

The commonest choice factorises the posterior across parameters — the **mean-field** family,
:math:`q(\theta) = \prod_k q_k(\theta_k)` — usually with Gaussian factors on an unconstrained scale.
**ADVI** automates the whole thing: transform constrained parameters, fit a Gaussian, optimise the ELBO
by stochastic gradients using automatic differentiation.

.. code-block:: python

   import pymc as pm
   with model:
       approx = pm.fit(n=30_000, method="advi")     # optimise the ELBO
       idata = approx.sample(2000)                  # draws from q, not from the posterior

The direction of the KL
-------------------------

Everything that goes wrong follows from **which** KL is minimised. :math:`\mathrm{KL}(q \| p)` is
**mode-seeking**: it heavily penalises putting :math:`q` mass where :math:`p` has none, and barely
penalises the reverse. So :math:`q` prefers to cover one mode and **underestimate variance** and tails.
A mean-field family compounds this: forcing independence across correlated parameters shrinks the
approximation's spread further.

The consequences are systematic, not random. VI point estimates are often excellent; VI **uncertainties
are usually too narrow**; and multimodal posteriors are typically represented by one mode. In a
hierarchical model, VI will underestimate :math:`\tau`'s uncertainty exactly where it matters.

Check it, always
------------------

Unlike MCMC, VI has no :math:`\hat{R}` — the optimiser converging says nothing about whether
:math:`\mathcal{Q}` **contains** anything close to the posterior. Diagnostics exist and should be used:
**PSIS** importance-reweighting of the VI draws (whose :math:`\hat{k}` flags a bad approximation), and
**simulation-based calibration**. Use VI for exploration, for very large models, and as an initialiser —
and never report a VI interval without checking it.

.. hint::

   **Related lessons:** :doc:`Distributional approximations <063-distributional-approximations>`  ·  :doc:`Expectation propagation <088-expectation-propagation>`  ·  :doc:`Normal and related mixture approximations <083-normal-and-related-mixture-approximations>`  ·  :doc:`Importance sampling <065-importance-sampling>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/22/variational-inference/ <https://insightful-data-lab.com/2025/11/22/variational-inference/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: intermediate
