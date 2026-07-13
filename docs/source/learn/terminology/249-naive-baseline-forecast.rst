:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-naive-baseline-forecast:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Naïve Baseline Forecast</b></div>`

=========================
Naïve Baseline Forecast
=========================

*A baseline predicting the next value equals the most recent observation.*

What it is
----------

A **naïve baseline forecast** is the simplest possible forecast — use the **last observed value** as the
prediction for the next period (the random-walk forecast). Its seasonal cousin, the **seasonal naïve**,
uses the value from one season ago:

.. math::

   \hat{y}_{t+1} = y_t \qquad\text{(naïve)}, \qquad\qquad \hat{y}_{t+h} = y_{t+h-m} \quad\text{(seasonal naïve, period } m).

Why it matters
--------------

It is the **benchmark every model must beat**. If a complex model cannot outperform *"just repeat the
last value"*, the complexity is not paying off. It is also the reference in the **denominator of MASE**,
computed on the **in-sample (training)** series so the benchmark does not leak future information.

Which variant
-------------

The plain naïve suits **non-seasonal** data; the **seasonal naïve** is the right reference when there is a
clear period (set :math:`m = 12` for monthly data with yearly seasonality, not :math:`m = 1`). Cheap,
robust, and — for noisy or short series — surprisingly hard to beat.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Forecast Error <250-forecast-error>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Point Forecasts <233-point-forecasts>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Naïve Baseline Forecast <https://insightful-data-lab.com/2025/08/22/naive-baseline-forecast/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
