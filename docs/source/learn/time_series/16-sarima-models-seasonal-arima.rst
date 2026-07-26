.. _ts-sarima-models-seasonal-arima:

========================================================================
SARIMA Models: Seasonal ARIMA
========================================================================

**Stage 6 · 🏗️ Building & Forecasting Models**  ·  Lesson 16 of 18  ·  *advanced*

:doc:`◀ Previous · ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`   ·   :doc:`Next · Beyond One-Step Ahead Predictions ▶ <17-beyond-one-step-ahead-predictions>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The seasonal problem
----------------------

Plain ARIMA handles **trend** but not **seasonality** — a pattern that repeats every :math:`s` steps
(12 for monthly data, 4 for quarterly). **SARIMA** extends it by adding a **second, seasonal** ARIMA
operating at the seasonal lag, so one model captures **both** short-term and season-to-season
structure.

The notation
--------------

A SARIMA is written **SARIMA(p, d, q)(P, D, Q)** with seasonal period **s**. The first triple is the
ordinary non-seasonal part; the second is its **seasonal mirror** — :math:`P` seasonal AR terms,
:math:`D` seasonal differences, :math:`Q` seasonal MA terms — each acting at multiples of :math:`s`.
In operator form the two multiply:

.. math::

   \Phi_P(B^s)\,\phi_p(B)\,(1-B)^d\,(1-B^s)^D\, x_t
     = \Theta_Q(B^s)\,\theta_q(B)\, w_t.

The seasonal difference
------------------------

The workhorse is the **seasonal difference** :math:`(1 - B^s)`, which subtracts the value from **one
full season ago**:

.. math::

   \nabla_s x_t = x_t - x_{t-s}.

Just as ordinary differencing removes a trend, this removes a **repeating seasonal** pattern. Data
with **both** trend and seasonality may need **both** a regular difference (:math:`d`) and a seasonal
one (:math:`D`). Spikes in the ACF / PACF at lags :math:`s, 2s, \dots` point to the seasonal orders.

Fitting and pitfalls
----------------------

In ``statsmodels`` this is ``SARIMAX(order=(p,d,q), seasonal_order=(P,D,Q,s))``. Two cautions:
seasonal terms **cost parameters**, so imposing them on a **non-seasonal** series adds noise and can
**degrade** forecasts; and for **multiple** overlapping seasonalities (say daily *and* weekly),
specialised tools like **TBATS** or **Prophet** fit better than a single seasonal period.

.. hint::

   **Related lessons:** :doc:`ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`  ·  :doc:`A Gentle Introduction to Stationarity <03-a-gentle-introduction-to-stationarity>`  ·  :doc:`Exponential Smoothing Models <18-exponential-smoothing-models>`  ·  :doc:`Beyond One-Step Ahead Predictions <17-beyond-one-step-ahead-predictions>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/sarima-models-seasonal-arima/ <https://insightful-data-lab.com/2026/01/17/sarima-models-seasonal-arima/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: advanced
