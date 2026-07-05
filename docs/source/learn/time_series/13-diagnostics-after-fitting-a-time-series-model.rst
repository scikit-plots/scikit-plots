.. _ts-diagnostics-after-fitting-a-time-series-model:

========================================================================
Diagnostics After Fitting a Time Series Model
========================================================================

**Stage 6 · 🏗️ Building & Forecasting Models**  ·  Lesson 13 of 18  ·  *advanced*

:doc:`◀ Previous · Maximum Likelihood Estimation for ARMA Models (Gaussian MLE) <12-maximum-likelihood-estimation-for-arma-models-gaussian-mle>`   ·   :doc:`Next · Order Selection for Time Series Models <14-order-selection-for-time-series-models> ▶`


The goal
----------

A fitted model is only as good as its **residuals**. If the model has captured all the structure,
what remains should be **white noise** — patternless, uncorrelated, constant-variance. Diagnostics
**standardise** the residuals (divide by their standard error) and then test that claim.

What to look at
----------------

Three views, read together:

* **residual plot** over time — should scatter randomly around zero, with **no trend and no
  changing spread** (heteroscedasticity);
* **residual ACF / correlogram** — essentially **no spikes** outside the
  :math:`\pm 1.96/\sqrt{n}` bands (spikes at **seasonal** lags hint at missed seasonality);
* **normality** — a **Q–Q plot** and histogram, since valid prediction intervals lean on roughly
  normal errors.

``statsmodels`` bundles these panels into ``results.plot_diagnostics()``.

The Ljung-Box test
--------------------

Eyeballing the ACF is subjective; the **Ljung–Box** test makes it formal. It is a **portmanteau**
test of the residual autocorrelations **jointly** up to lag :math:`h`:

.. math::

   Q = n(n+2)\sum_{k=1}^{h} \frac{\hat{r}_k^2}{n-k},

compared against a :math:`\chi^2` distribution. The null is **no autocorrelation** (white-noise
residuals), so a **small p-value means the model is inadequate**. The degrees of freedom are reduced
by the number of fitted AR / MA parameters. (``acorr_ljungbox`` in ``statsmodels``.)

When it fails
--------------

Failing diagnostics are a **diagnosis, not a dead end**. Autocorrelation left in the residuals says
the model is **missing structure** — add an **AR or MA** term, or, if the leftover spikes fall at
**seasonal** lags, move to a **seasonal** model. Re-fit and re-check until the residuals look like
noise.

.. seealso::

   **Related lessons:** :doc:`Order Selection for Time Series Models <14-order-selection-for-time-series-models>`  ·  :doc:`Maximum Likelihood Estimation for ARMA Models (Gaussian MLE) <12-maximum-likelihood-estimation-for-arma-models-gaussian-mle>`  ·  :doc:`Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`  ·  :doc:`ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/17/diagnostics-after-fitting-a-time-series-model-standardized-residuals-normality-and-autocorrelation/ <https://insightful-data-lab.com/2026/01/17/diagnostics-after-fitting-a-time-series-model-standardized-residuals-normality-and-autocorrelation/>`__

.. tags:: purpose: reference, topic: time series, level: advanced
