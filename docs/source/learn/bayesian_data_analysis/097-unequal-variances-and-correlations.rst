.. _bda-unequal-variances-and-correlations:

========================================================================
Unequal variances and correlations
========================================================================

**Part 4 · Stage 11 · 📈 Regression Foundations**  ·  Lesson 097 of 144  ·  *advanced*

:doc:`◀ Previous · Regularization and dimension reduction <096-regularization-and-dimension-reduction>`   ·   :doc:`Next · Including numerical prior information ▶ <098-including-numerical-prior-information>`   ·   :doc:`↑ Section <index>`


When the errors are not i.i.d.
--------------------------------

Classical regression assumes :math:`y \sim \mathrm{N}(X\beta, \sigma^2 I)` — errors with **equal
variance** and **no correlation**. Real errors routinely violate both: measurements of differing
precision, variance that grows with the mean, observations clustered in time or space. The general
normal linear model replaces :math:`\sigma^2 I` with a full covariance :math:`\Sigma`:

.. math::

   y \mid \beta, \Sigma \sim \mathrm{N}(X\beta, \; \Sigma).

Known :math:`\Sigma`: reduce to the standard case
---------------------------------------------------

If :math:`\Sigma` is known, a single transformation recovers everything. Factor :math:`\Sigma = L
L^{\top}` (Cholesky) and multiply through by :math:`L^{-1}`:

.. math::

   \underbrace{L^{-1} y}_{y^{*}} = \underbrace{L^{-1} X}_{X^{*}} \beta + \epsilon^{*},
   \qquad \epsilon^{*} \sim \mathrm{N}(0, I).

The transformed model is ordinary regression, so its posterior mean is the **generalized least squares**
estimate :math:`\hat{\beta} = (X^{\top}\Sigma^{-1}X)^{-1} X^{\top}\Sigma^{-1}y`. The **diagonal** case —
uncorrelated but unequal variances, :math:`\Sigma = \mathrm{diag}(\sigma_i^2)` — is **weighted least
squares**, with each observation weighted by its precision :math:`1/\sigma_i^2`. Whitening and weighting
are the same operation, one general and one diagonal.

.. code-block:: python

   import numpy as np
   L = np.linalg.cholesky(Sigma)                    # Sigma = L L^T
   Xs = np.linalg.solve(L, X)                        # L^{-1} X  (whitened design)
   ys = np.linalg.solve(L, y)                        # L^{-1} y
   beta_gls = np.linalg.lstsq(Xs, ys, rcond=None)[0] # OLS on whitened data == GLS

Unknown :math:`\Sigma`: model it
----------------------------------

Usually :math:`\Sigma` is unknown, and here the Bayesian treatment separates cleanly from the classical
one: rather than plug in an estimate, give :math:`\Sigma` a **structure with its own parameters** and
infer everything jointly.

* **Variances a function of a predictor** — :math:`\log \sigma_i = \gamma_0 + \gamma_1 z_i`, letting the
  data learn how the noise scales.
* **Correlation from structure** — an AR(1) covariance for time series, a distance-decaying kernel for
  spatial data (Stage 15), a compound-symmetry block for grouped data.
* **Hierarchical variances** — a prior across the :math:`\sigma_i` when there are many.

The posterior then propagates uncertainty in :math:`\Sigma` into the coefficients — which plug-in GLS
ignores, understating the intervals exactly as empirical Bayes did.

.. code-block:: python

   import pymc as pm
   with pm.Model():
       beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
       g = pm.Normal("g", 0, 1, shape=2)
       sigma = pm.Deterministic("sigma", pm.math.exp(g[0] + g[1] * z))   # variance model
       pm.Normal("y", X @ beta, sigma, observed=y)                       # heteroscedastic

Why it matters
----------------

Ignoring unequal variance does not bias :math:`\hat{\beta}`, but it makes the standard errors **wrong**
and the estimator **inefficient** — the tidy printout is confidently miscalibrated. Correlated errors are
worse: treating clustered or serially dependent observations as independent **overstates** the effective
sample size, producing intervals far too narrow. Modelling :math:`\Sigma` is what makes the uncertainty
honest, and it is the doorway to the mixed models and Gaussian processes ahead.

.. hint::

   **Related lessons:** :doc:`Bayesian analysis of classical regression <092-bayesian-analysis-of-classical-regression>`  ·  :doc:`Including numerical prior information <098-including-numerical-prior-information>`  ·  :doc:`Models for multivariate and multinomial responses <111-models-for-multivariate-and-multinomial-responses>`  ·  :doc:`Gaussian process regression <129-gaussian-process-regression>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2025/11/23/unequal-variances-and-correlations/ <https://insightful-data-lab.com/2025/11/23/unequal-variances-and-correlations/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: data analysis, domain: bayesian, level: advanced
