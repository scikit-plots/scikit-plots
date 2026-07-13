.. _bda-example-bayesian-analysis-of-a-bioassay-experiment-logistic-nonconjugate:

==============================================================================
Example: Bayesian analysis of a bioassay experiment (logistic, nonconjugate)
==============================================================================

**Part 1 · Stage 3 · 🧮 Multiparameter Models**  ·  Lesson 026 of 144  ·  *beginner*

:doc:`◀ Previous · Multivariate Normal with Unknown Mean and Variance <025-multivariate-normal-with-unknown-mean-and-variance>`   ·   :doc:`Next · Summary of Elementary Modeling and Computation ▶ <027-summary-of-elementary-modeling-and-computation>`   ·   :doc:`↑ Section <index>`


Where the closed forms end
----------------------------

Every posterior so far has had a formula. This one does not, and that is the point. In a **bioassay**,
groups of animals receive increasing **doses** :math:`x_i` of a compound and the number of **deaths**
:math:`y_i` out of :math:`n_i` is recorded. A typical experiment is tiny — four dose groups, five
animals each.

The logistic dose–response model
----------------------------------

Deaths are binomial with a dose-dependent probability, modelled on the **log-odds** scale:

.. math::

   y_i \sim \mathrm{Binomial}(n_i,\, \theta_i), \qquad
   \mathrm{logit}(\theta_i) = \log\frac{\theta_i}{1 - \theta_i} = \alpha + \beta x_i .

Here :math:`\beta` is the **dose effect** — the scientific question is usually whether
:math:`\beta > 0`. A Beta prior cannot help: the parameters :math:`(\alpha, \beta)` enter through a
nonlinear link, so **no conjugate prior exists** and the posterior

.. math::

   p(\alpha, \beta \mid y) \;\propto\; p(\alpha, \beta)
   \prod_{i} \bigl[\mathrm{logit}^{-1}(\alpha + \beta x_i)\bigr]^{y_i}
             \bigl[1 - \mathrm{logit}^{-1}(\alpha + \beta x_i)\bigr]^{n_i - y_i}

has no closed form. This is the normal situation in applied work; Stage 2's tidy algebra was the
exception.

Compute it on a grid
----------------------

With only **two** parameters, you can simply **evaluate** the unnormalised posterior over a grid,
normalise by summing, and sample from the discrete approximation. It is brute force, and it is exact
enough to be a benchmark for the samplers of Part III.

.. code-block:: python

   import numpy as np
   from scipy.special import expit

   x = np.array([-0.86, -0.30, -0.05, 0.73])       # log dose
   n = np.array([5, 5, 5, 5]); y = np.array([0, 1, 3, 5])

   a, b = np.meshgrid(np.linspace(-5, 10, 400), np.linspace(-10, 40, 400))
   th = expit(a[..., None] + b[..., None] * x)     # broadcast over dose groups
   logpost = (y * np.log(th) + (n - y) * np.log1p(-th)).sum(-1)   # flat prior
   post = np.exp(logpost - logpost.max()); post /= post.sum()
   (post.sum(axis=0) @ np.linspace(-5, 10, 400))   # E[alpha | y], etc.

Reading the answer
--------------------

The posterior for :math:`(\alpha, \beta)` is **skewed and correlated** — no normal approximation would
capture its banana shape well — and :math:`\Pr(\beta > 0 \mid y)` is essentially 1: the dose kills.
Because any function of draws is itself a posterior draw, the **LD50** (the dose at which
:math:`\theta = 0.5`, namely :math:`-\alpha / \beta`) comes free, with a wide and asymmetric interval
that a plug-in estimate would badly misrepresent. Grids work in two dimensions; beyond three or four
they die of dimensionality — which is precisely why Part III exists.

.. hint::

   **Related lessons:** :doc:`Standard generalized linear model likelihoods <106-standard-generalized-linear-model-likelihoods>`  ·  :doc:`Numerical integration <062-numerical-integration>`  ·  :doc:`Normal Approximations to the Posterior Distribution <028-normal-approximations-to-the-posterior-distribution>`  ·  :doc:`Summary of Elementary Modeling and Computation <027-summary-of-elementary-modeling-and-computation>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/09/example-bayesian-analysis-of-a-bioassay-experiment-logistic-nonconjugate/ <https://insightful-data-lab.com/2025/11/09/example-bayesian-analysis-of-a-bioassay-experiment-logistic-nonconjugate/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: beginner
