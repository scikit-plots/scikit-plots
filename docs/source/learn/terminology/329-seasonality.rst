:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-seasonality:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Seasonality</b></div>`

=============
Seasonality
=============

*Regular, calendar-linked cycles in a time series.*

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

**Seasonality** is a pattern in a time series that **repeats at fixed, regular intervals** — driven by
seasonal factors such as the time of year, the month, the day of the week or the hour of the day.
Unlike a **trend** (a long-term rise or fall), seasonality is **periodic**, recurring with a fixed
frequency. In a decomposition, a series splits into trend, seasonal and residual parts,
:math:`y_t = T_t + S_t + \varepsilon_t`, with :math:`S_t` the seasonal component.

Examples
--------

Retail sales spike every December, ice-cream demand rises each summer, website traffic dips every
weekend, and electricity load peaks on hot afternoons. In each case the same shape returns on a
predictable cycle.

Detecting it
------------

Seasonality shows up as a **repeating shape** in a line plot, and is confirmed with **seasonal
subseries** or **decomposition** plots and with the **autocorrelation function (ACF)**, which spikes at
the **seasonal lag** — for example lag 12 for monthly data with yearly seasonality.

Handling it in models
---------------------

Several tools absorb it: **seasonal differencing** removes it, **SARIMA** adds seasonal
:math:`(P, D, Q)_m` terms, **Prophet** fits it with a Fourier series, and simpler models use
**seasonal dummy variables**. Getting seasonality right is essential for accurate forecasting —
ignoring it leaves systematic, repeating errors in the residuals.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Time Series <010-time-series>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Temporal autocorrelation (Serial Correlation) <127-temporal-autocorrelation-serial-correlation>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Seasonality <https://insightful-data-lab.com/2025/08/20/seasonality/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
