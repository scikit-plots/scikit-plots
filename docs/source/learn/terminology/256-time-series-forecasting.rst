:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-time-series-forecasting:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Time Series Forecasting</b></div>`

=========================
Time Series Forecasting
=========================

*Predicting future values of a time-ordered series.*

What it is
----------

**Time series forecasting** predicts **future values** of a **time-ordered** series from its past. Unlike
ordinary supervised learning — where rows are independent and can be shuffled — a time series has
**memory**: order matters, and each observation is **dependent** on its neighbors.

The components
--------------

A series decomposes into a **trend** (long-term drift up or down), **seasonality** (regular patterns at a
**fixed** period), **cycles** (longer, **aperiodic** swings with no fixed length), and **noise** (the
irregular residual). Modeling means **separating** these.

Stationarity and diagnostics
----------------------------

Many methods need **stationarity** (constant mean and variance); non-stationary series are made modelable
by **differencing** (removes trend / seasonality from the mean) and a **log / Box-Cox** transform
(stabilizes variance). Choose the model **after** diagnostics — time, seasonal and **lag** plots, and the
**ACF / PACF** — and evaluate with **rolling-origin backtesting**, reporting error by **horizon** and
publishing **intervals**, not just points.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Log-Space <257-log-space>` · :doc:`Seasonality <329-seasonality>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Point Forecasts <233-point-forecasts>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Time Series Forecasting <https://insightful-data-lab.com/2025/08/22/time-series-forecasting/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
