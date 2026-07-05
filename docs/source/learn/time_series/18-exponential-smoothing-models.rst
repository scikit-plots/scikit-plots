.. _ts-exponential-smoothing-models:

========================================================================
Exponential Smoothing Models
========================================================================

**Stage 6 · 🏗️ Building & Forecasting Models**  ·  Lesson 18 of 18  ·  *advanced*

:doc:`◀ Previous · Beyond One-Step Ahead Predictions <17-beyond-one-step-ahead-predictions>`


The idea
----------

**Exponential smoothing** forecasts with a **weighted average of past observations**, where the
weights **decay exponentially** into the past — recent data counts most, older data fades but never
fully vanishes. It is a different lineage from ARIMA, built around **components** (level, trend,
season) rather than autocorrelations.

Simple smoothing
------------------

The simplest form, **Simple Exponential Smoothing (SES)**, tracks a single **level** and suits
series with **no trend or seasonality**:

.. math::

   \hat{x}_{t+1} = \alpha\, x_t + (1 - \alpha)\, \hat{x}_t, \qquad 0 < \alpha < 1.

The smoothing parameter :math:`\alpha` sets the memory: near 1 reacts fast to recent values, near 0
stays smooth and sluggish. Its forecasts are **flat**.

Adding trend and season
------------------------

Two extensions handle richer data. **Holt's linear** method adds a **trend** component (a second
parameter :math:`\beta`), giving sloped forecasts. **Holt–Winters** adds a **seasonal** component too
(a third parameter :math:`\gamma`), either **additive** or **multiplicative** — the standard choice
for series with **both** trend and seasonality. Together these form the **ETS** (Error–Trend–Seasonal)
family.

Smoothing or ARIMA?
--------------------

The two approaches are **complementary**, not rivals: ARIMA describes a series through its
**autocorrelations**, exponential smoothing through its **trend and seasonal structure**. Which wins
is **empirical** — smoothing often shines on strongly trended, seasonal data, ARIMA on stationary,
mean-reverting data. In ``statsmodels`` these live in ``SimpleExpSmoothing``, ``Holt`` and
``ExponentialSmoothing`` (Holt–Winters).

.. seealso::

   **Related lessons:** :doc:`ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`  ·  :doc:`SARIMA Models: Seasonal ARIMA <16-sarima-models-seasonal-arima>`  ·  :doc:`Beyond One-Step Ahead Predictions <17-beyond-one-step-ahead-predictions>`  ·  :doc:`What Are Time Series, and How Are They Used? <01-what-are-time-series-and-how-are-they-used>`

**Source** (context, re-expressed in our own words): `https://insightful-data-lab.com/2026/01/17/exponential-smoothing-models/ <https://insightful-data-lab.com/2026/01/17/exponential-smoothing-models/>`__

.. tags:: purpose: reference, topic: time series, level: advanced
