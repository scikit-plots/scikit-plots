:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. _term-seasonal-lag:

:raw-html:`<div align="center" style="font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Seasonal Lag</b></div>`

==============
Seasonal Lag
==============

*The offset to the same point in a previous season, used in seasonal models.*

What it is
----------

A **lag** is a past value of the series, :math:`y_{t-k}`; the **seasonal lag** is the lag equal to the
**seasonal period** :math:`m` — the value from the **same point one cycle ago**:

.. math::

   \text{seasonal lag: } y_{t-m}, \qquad \text{seasonal difference: } y_t - y_{t-m}.

Common periods
--------------

The period is set by the calendar of the data: :math:`m = 12` for **monthly** data with yearly
seasonality, :math:`m = 7` for **daily** data with weekly seasonality, :math:`m = 24` for **hourly** data
with daily cycles.

Where it's used
---------------

The seasonal lag underlies the **seasonal naïve** forecast (:math:`\hat{y}_t = y_{t-m}`), **seasonal
differencing** (which strips out seasonality), and **lag features** in ML forecasting. A large
**autocorrelation** at the seasonal lag is the signature of seasonality.

----

**Mind map — connected ideas**

   :doc:`Time Series Forecasting <256-time-series-forecasting>` · :doc:`Seasonality <329-seasonality>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Log-Space <257-log-space>` · :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

----

**More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

.. seealso::

   Adapted in our own words from `Seasonal Lag <https://insightful-data-lab.com/2025/08/22/seasonal-lag/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, level: advanced
