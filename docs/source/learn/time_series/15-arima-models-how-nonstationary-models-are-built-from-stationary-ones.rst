.. _ts-arima-models-how-nonstationary-models-are-built-from-stationary-ones:

========================================================================
ARIMA Models: How Nonstationary Models Are Built from Stationary Ones
========================================================================

**Stage 6 · 🏗️ Building & Forecasting Models**  ·  Lesson 15 of 18  ·  *advanced*

:doc:`◀ Previous · Order Selection for Time Series Models <14-order-selection-for-time-series-models>`   ·   :doc:`Next · SARIMA Models: Seasonal ARIMA ▶ <16-sarima-models-seasonal-arima>`   ·   :doc:`↑ Section <index>`


.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

The core idea
--------------

ARMA needs a **stationary** series, but real data trends and drifts. **ARIMA** bridges the gap with
one move: **difference the series until it is stationary**, fit an ordinary **ARMA** to the
differenced version, and the model inherits ARMA's whole toolkit. The **"I"** stands for
**Integrated** — the series must be *un*-differenced (integrated) to recover the original.

The model
----------

An **ARIMA(p, d, q)** applies the :math:`d`-th difference :math:`(1-B)^d` before the ARMA machinery:

.. math::

   \phi(B)\,(1 - B)^d\, x_t = \theta(B)\, w_t.

Here :math:`(1-B)^d` is the **differencing operator**, :math:`\phi(B)` the AR polynomial and
:math:`\theta(B)` the MA polynomial. A series needing :math:`d` differences to become stationary is
called **integrated of order** :math:`d`, or :math:`I(d)`.

Choosing d
------------

One difference (:math:`d = 1`) removes a **linear** trend; two (:math:`d = 2`) removes a
**quadratic** one; seasonal patterns need a **seasonal** difference (next lesson). Pick the
**smallest** :math:`d` that makes the ADF / KPSS tests read stationary — **over-differencing**
inflates the variance and injects artificial correlation, so more is not better.

Forecasting back
------------------

Fitting happens on the **differenced** scale, but forecasts are wanted on the **original** one. The
model **"integrates"** — cumulatively sums — its differenced-scale predictions back up to the level
of the raw series, carrying the forecast **uncertainty** with it. In ``statsmodels`` this is all
handled by ``ARIMA(y, order=(p, d, q))``.

.. hint::

   **Related lessons:** :doc:`A Gentle Introduction to Stationarity <03-a-gentle-introduction-to-stationarity>`  ·  :doc:`Understanding ARMA Processes <06-understanding-arma-processes>`  ·  :doc:`SARIMA Models: Seasonal ARIMA <16-sarima-models-seasonal-arima>`  ·  :doc:`Order Selection for Time Series Models <14-order-selection-for-time-series-models>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `https://insightful-data-lab.com/2026/01/17/arima-models-how-nonstationary-models-are-built-from-stationary-ones/ <https://insightful-data-lab.com/2026/01/17/arima-models-how-nonstationary-models-are-built-from-stationary-ones/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: time series, level: advanced
