.. _ts-order-selection-for-time-series-models:

========================================================================
Order Selection for Time Series Models
========================================================================

**Stage 6 · 🏗️ Building & Forecasting Models**  ·  Lesson 14 of 18  ·  *advanced*

:doc:`◀ Previous · Diagnostics After Fitting a Time Series Model <13-diagnostics-after-fitting-a-time-series-model>`   ·   :doc:`Next · ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones> ▶`


The trade-off
--------------

Every extra parameter **improves the in-sample fit** but risks **overfitting** — chasing noise that
will not repeat. Order selection is the search for a model that is **complex enough to fit, simple
enough to generalise**: the **parsimony** principle at the heart of Box–Jenkins.

Information criteria
----------------------

The standard tools score fit **against** complexity. Both the **Akaike** and **Bayesian**
information criteria reward the likelihood and **penalise** the parameter count :math:`k`:

.. math::

   \mathrm{AIC} = 2k - 2\ln \hat{L}, \qquad \mathrm{BIC} = k\ln n - 2\ln \hat{L}.

**Lower is better**, and you compare candidates fitted to the **same** data. (The small-sample
correction **AICc** is safer when :math:`n` is not large.)

AIC versus BIC
----------------

The two differ only in the penalty. **BIC**'s per-parameter cost :math:`\ln n` is harsher than
**AIC**'s :math:`2` (once :math:`n > 7`), so **BIC favours simpler models** and is preferred when
**parsimony** matters; **AIC** tends to pick slightly richer models and suits **predictive
accuracy**. When they disagree, the choice is yours to justify.

Putting it together
--------------------

In practice: pick **d** by differencing until stationary (ADF / KPSS), read tentative **p, q** off
the ACF / PACF, then **grid-search** nearby orders and keep the lowest-criterion model **that also
passes diagnostics** — a lower AIC means nothing if the residuals are still autocorrelated. Tools
like ``pmdarima.auto_arima`` automate the search; ``statsmodels`` exposes ``.aic`` and ``.bic``.

.. seealso::

   **Related lessons:** :doc:`Diagnostics After Fitting a Time Series Model <13-diagnostics-after-fitting-a-time-series-model>`  ·  :doc:`Sample ACF and Sample PACF <10-sample-acf-and-sample-pacf>`  ·  :doc:`ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`  ·  :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/17/order-selection-for-time-series-models/ <https://insightful-data-lab.com/2026/01/17/order-selection-for-time-series-models/>`__

.. tags:: purpose: reference, topic: time series, level: advanced
