.. _bda-bayesian-analysis-of-classical-regression:

========================================================================
Bayesian analysis of classical regression
========================================================================

**Part 4 · Stage 11 · 📈 Regression Foundations**  ·  Lesson 092 of 144  ·  *advanced*

:doc:`◀ Previous · Conditional modeling <091-conditional-modeling>`   ·   :doc:`Next · Regression for causal inference: incumbency and voting ▶ <093-regression-for-causal-inference-incumbency-and-voting>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The normal linear model
-------------------------

The workhorse of applied statistics, read Bayesianly. With :math:`n` observations, :math:`k` predictors
in a matrix :math:`X`, coefficients :math:`\beta` and noise variance :math:`\sigma^2`:

.. math::

   y \mid \beta, \sigma^2, X \sim \mathrm{N}(X\beta, \; \sigma^2 I).

Everything from Stage 3's normal model carries over, with :math:`X\beta` in place of a scalar mean.

Noninformative prior
----------------------

Take the standard :math:`p(\beta, \sigma^2) \propto 1/\sigma^2`. The posterior factors exactly as before
— conditional for the coefficients, marginal for the variance:

.. math::

   \beta \mid \sigma^2, y \sim \mathrm{N}\bigl(\hat{\beta}, \; \sigma^2 (X^{\top} X)^{-1}\bigr),
   \qquad
   \sigma^2 \mid y \sim \text{Inv-}\chi^2\bigl(n - k, \; s^2\bigr),

where :math:`\hat{\beta} = (X^{\top}X)^{-1} X^{\top} y` is the **least-squares estimate** and
:math:`s^2 = \frac{1}{n-k}(y - X\hat{\beta})^{\top}(y - X\hat{\beta})`. Marginalising :math:`\sigma^2`
gives a **multivariate** :math:`t` for :math:`\beta` — the heavy tails again the price of not knowing the
noise scale.

The classical results, reinterpreted
--------------------------------------

The posterior mean of :math:`\beta` **is** the least-squares estimate; the posterior covariance **is**
:math:`s^2 (X^{\top}X)^{-1}`; the marginal for each coefficient **is** a :math:`t` on :math:`n - k`
degrees of freedom. Every number in a regression printout is recovered — but reinterpreted: the interval
is now a probability statement about :math:`\beta`, and the :math:`t` arises by **integration** rather
than by a sampling-distribution argument.

Sampling without MCMC
-----------------------

Because the factorisation is exact, joint draws are direct — no chain, no diagnostics:

.. code-block:: python

   import numpy as np
   from scipy import stats
   n, k = X.shape
   XtX_inv = np.linalg.inv(X.T @ X)
   beta_hat = XtX_inv @ X.T @ y
   s2 = ((y - X @ beta_hat) ** 2).sum() / (n - k)

   sigma2 = (n - k) * s2 / stats.chi2(n - k).rvs(4000)                   # marginal draw
   beta = np.array([stats.multivariate_normal(beta_hat, s * XtX_inv).rvs()
                    for s in sigma2])                                     # conditional draw
   np.percentile(beta[:, 1], [2.5, 97.5])       # posterior interval for the 2nd coefficient

What the closed form hides
----------------------------

Three assumptions are doing quiet work: **normal errors** (relaxed in Stage 14), **constant variance and
independence** (relaxed later in this stage), and an :math:`X^{\top}X` that is **invertible** and
well-conditioned. When :math:`k` approaches :math:`n`, or predictors are collinear, the noninformative
posterior is diffuse or improper — and the remedy is a **prior**, which is exactly regularisation.

.. hint::

   **Related lessons:** :doc:`Conditional modeling <091-conditional-modeling>`  ·  :doc:`Normal Data with a Noninformative Prior Distribution <021-normal-data-with-a-noninformative-prior-distribution>`  ·  :doc:`Regularization and dimension reduction <096-regularization-and-dimension-reduction>`  ·  :doc:`Goals of regression analysis <094-goals-of-regression-analysis>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/bayesian-analysis-of-classical-regression/ <https://insightful-data-lab.com/2025/11/23/bayesian-analysis-of-classical-regression/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
