:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-log-space:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Log-Space</b></div>`

===========
Log-Space
===========

*Working with log-transformed values to stabilise variance or handle scale.*

What it is
----------

Working in **log-space** means transforming a series to its **logarithm**, :math:`w_t = \log(y_t)`,
instead of the raw values — a **variance-stabilizing** transform for when fluctuations **grow with the
level** of the series (multiplicative or heteroscedastic behavior).

What it does
------------

It **compresses large values** while leaving small ones nearly untouched, turns **multiplicative**
structure into **additive**, and makes **relative (percentage) change** the natural unit — a difference in
log-space is approximately a proportional change:

.. math::

   \log(y_t) - \log(y_{t-1}) \approx \frac{y_t - y_{t-1}}{y_{t-1}}.

Caveats
-------

The logarithm is **undefined for zero or negative** values — use :math:`\log(y + 1)` or a **Box-Cox**
transform instead — and any forecast made in log-space must be **back-transformed** (exponentiated) to the
original scale.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Time Series Forecasting <256-time-series-forecasting>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Normal Distribution <238-normal-distribution>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Relative accuracy <258-relative-accuracy>` · :doc:`Seasonality <329-seasonality>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Log-Space <https://insightful-data-lab.com/2025/08/22/log-space/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
