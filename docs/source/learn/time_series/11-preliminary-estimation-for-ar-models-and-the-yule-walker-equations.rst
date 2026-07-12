.. _ts-preliminary-estimation-for-ar-models-and-the-yule-walker-equations:

========================================================================
Preliminary Estimation for AR Models and the Yule–Walker Equations
========================================================================

**Stage 5 · 🧮 Estimation**  ·  Lesson 11 of 18  ·  *advanced*

:doc:`◀ Previous · Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`   ·   :doc:`Next · Maximum Likelihood Estimation for ARMA Models (Gaussian MLE) ▶ <12-maximum-likelihood-estimation-for-arma-models-gaussian-mle>`   ·   :doc:`↑ Section <index>`


The idea
----------

The **Yule–Walker** method is the simplest way to fit an **AR(p)** model: it is **method of
moments** — match the model's theoretical autocovariances to the ones you measure, and solve for
the coefficients. No optimisation, no starting values.

The equations
--------------

Multiplying the AR recursion by lagged values and taking expectations links the coefficients to the
autocovariances through a **Toeplitz** linear system:

.. math::

   \Gamma_p\, \boldsymbol{\phi} = \boldsymbol{\gamma}_p, \qquad
   \Gamma_p = \big[\gamma(i-j)\big]_{i,j=1}^{p}, \quad
   \boldsymbol{\gamma}_p = \big(\gamma(1), \dots, \gamma(p)\big)^{\!\top}.

Because :math:`\Gamma_p` is a full-rank symmetric Toeplitz matrix, the solution
:math:`\boldsymbol{\phi} = \Gamma_p^{-1}\boldsymbol{\gamma}_p` always exists.

Plugging in the data
----------------------

Replace each :math:`\gamma(\cdot)` with its **sample** counterpart :math:`\hat{\gamma}(\cdot)` to
get :math:`\hat{\boldsymbol{\phi}} = \hat{\Gamma}_p^{-1}\hat{\boldsymbol{\gamma}}_p`, and read off
the noise variance as
:math:`\hat{\sigma}_w^2 = \hat{\gamma}(0) - \hat{\boldsymbol{\phi}}^{\!\top}\hat{\boldsymbol{\gamma}}_p`.
The **Durbin–Levinson recursion** solves the system without an explicit inverse; in ``statsmodels``
this is ``yule_walker``.

Why it's a good start
----------------------

For a genuine **AR(p)**, Yule–Walker estimators are **consistent**, **asymptotically normal**, and
**as efficient as maximum likelihood** — and they always return a **causal (stationary)** model.
That reliability makes them the standard **preliminary estimate** and a natural **starting point**
for the likelihood methods below. (For **ARMA** the same idea works but is **no longer efficient**,
so it is used only to seed the optimiser.)

.. hint::

   **Related lessons:** :doc:`Best Linear Predictor of a Stationary Process <09-best-linear-predictor-of-a-stationary-process>`  ·  :doc:`Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`  ·  :doc:`Maximum Likelihood Estimation for ARMA Models (Gaussian MLE) <12-maximum-likelihood-estimation-for-arma-models-gaussian-mle>`  ·  :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/preliminary-estimation-for-ar-models-and-the-yule-walker-equations/ <https://insightful-data-lab.com/2026/01/17/preliminary-estimation-for-ar-models-and-the-yule-walker-equations/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: advanced
