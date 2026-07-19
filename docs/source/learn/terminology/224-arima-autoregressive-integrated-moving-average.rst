:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-arima-autoregressive-integrated-moving-average:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>ARIMA (AutoRegressive Integrated Moving Average)</b></div>`

==================================================
ARIMA (AutoRegressive Integrated Moving Average)
==================================================

*A classic model combining autoregression, differencing and moving-average terms for forecasting.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

**ARIMA (AutoRegressive Integrated Moving Average)** is a classical statistical model for
**time-series forecasting** that predicts future values from a series' **own past**. It fuses three
ideas — **autoregression (AR)**, **integration / differencing (I)** and **moving average (MA)** — and
describes a series by its **autocorrelations** rather than by explicit trend or seasonality. A model
is written **ARIMA(p, d, q)**.

The three parameters
--------------------

Each letter is one parameter. **p** is the **AR order** — how many lagged past *values* the current
value is regressed on. **d** is the **differencing order** — how many times the series is differenced
to make it **stationary** (removing trend). **q** is the **MA order** — how many past *error* terms
feed the forecast. In backshift form,

.. math::

   \phi_p(B)\,(1 - B)^d\, y_t = \theta_q(B)\,\varepsilon_t,

where :math:`B` is the backshift operator (:math:`B y_t = y_{t-1}`), :math:`\phi_p` and
:math:`\theta_q` are the AR and MA polynomials, and :math:`(1 - B)^d` applies the differencing.
Setting parameters to zero recovers the simpler **AR**, **MA** and **ARMA** models.

Building one (Box-Jenkins)
----------------------------

The Box-Jenkins recipe has three stages. First, make the series **stationary** by differencing,
checked with a unit-root test such as the **Dickey-Fuller** test. Next, pick **p** and **q**: the
**ACF** (autocorrelation function) guides **q**, the **PACF** (partial autocorrelation function)
guides **p**, and among candidates you choose the one with the lowest **AIC** (or BIC). Finally,
**validate the residuals** — they should be uncorrelated white noise; if not, revisit the orders.

Strengths, limits, and SARIMA
-------------------------------

ARIMA is **flexible** and **interpretable** for **linear, univariate** series — finance, demand,
sales — and gives stable longer-term forecasts. But it **assumes a linear autocorrelation structure**,
**requires stationarity**, and struggles with non-linear patterns where **LSTMs or Transformers**
do better. For periodic data, the **SARIMA** extension adds seasonal terms, written
:math:`\text{ARIMA}(p, d, q)(P, D, Q)_m` with season length :math:`m`.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`Time Series <010-time-series>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Seasonality <329-seasonality>` · :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `ARIMA (AutoRegressive Integrated Moving Average) <https://insightful-data-lab.com/2025/08/23/arima-autoregressive-integrated-moving-average/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
