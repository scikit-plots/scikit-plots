.. _ts-beyond-one-step-ahead-predictions:

========================================================================
Beyond One-Step Ahead Predictions
========================================================================

**Stage 6 · 🏗️ Building & Forecasting Models**  ·  Lesson 17 of 18  ·  *advanced*

:doc:`◀ Previous · SARIMA Models: Seasonal ARIMA <16-sarima-models-seasonal-arima>`   ·   :doc:`Next · Exponential Smoothing Models ▶ <18-exponential-smoothing-models>`   ·   :doc:`↑ Section <index>`


.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

Forecasting further
--------------------

A one-step forecast is just the start. Usually you want the value :math:`h` steps ahead — the
**h-step forecast** — which is the **conditional expectation**
:math:`\hat{x}_{n+h} = \mathbb{E}[x_{n+h} \mid x_1, \dots, x_n]`: the mean of the future given
everything observed so far.

Building it up
----------------

The practical route is **recursive** ("plug-in"): apply the one-step predictor, then **feed its
output back in** as if it were observed, and step forward again. Each future value the model needs
but has not seen is replaced by its **own forecast**, so errors from early steps **propagate** into
later ones.

Uncertainty grows
------------------

The crucial consequence: **forecast uncertainty widens with the horizon**. Every step adds a fresh
shock whose variance **accumulates**, so the **prediction intervals fan out** the further you look —
a forecast for next month is far tighter than one for next year. Reporting the **interval**, not just
the point, is essential. (``get_forecast(steps=h)`` in ``statsmodels`` returns both.)

Two limiting cases
--------------------

How the intervals grow depends on the model. For a **stationary ARMA**, forecasts **revert to the
mean** and the interval width **levels off** at the unconditional variance — the process forgets its
current state. For an **ARIMA** with differencing, there is no mean to revert to, so the intervals
**keep widening without bound**, random-walk style. Either way, small **parameter errors compound**
over long horizons.

.. hint::

   **Related lessons:** :doc:`Best Linear Predictor of a Stationary Process <09-best-linear-predictor-of-a-stationary-process>`  ·  :doc:`ARIMA Models: How Nonstationary Models Are Built from Stationary Ones <15-arima-models-how-nonstationary-models-are-built-from-stationary-ones>`  ·  :doc:`Exponential Smoothing Models <18-exponential-smoothing-models>`  ·  :doc:`Diagnostics After Fitting a Time Series Model <13-diagnostics-after-fitting-a-time-series-model>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/beyond-one-step-ahead-predictions/ <https://insightful-data-lab.com/2026/01/17/beyond-one-step-ahead-predictions/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: advanced
