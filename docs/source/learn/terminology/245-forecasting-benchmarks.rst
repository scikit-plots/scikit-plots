:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-forecasting-benchmarks:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Forecasting Benchmarks</b></div>`

========================
Forecasting Benchmarks
========================

*Standard datasets and baselines for comparing forecasting methods.*

.. important::

   **✨ AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What it is
----------

A **forecasting benchmark** is the **reference forecast** a proposed model must **outperform** to be worth
using. In practice the **simple baseline methods** (naïve, seasonal naïve, mean, drift) fill this role —
the **naïve** forecast is the standard reference, and the basis of **MASE**.

The discipline
--------------

Always establish a benchmark **before** reaching for complex models — a step that is often skipped. *Any
complex model must be better than the baseline to be considered.* A model that only **marginally** beats
the naïve forecast probably is not worth its added complexity and maintenance.

Beyond one series
-----------------

Shared **datasets** (such as the M-competition series) act as **community benchmarks**, letting different
methods be compared on **common ground** rather than on each author's private data.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Simple Baseline Methods <248-simple-baseline-methods>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Relative accuracy <258-relative-accuracy>` · :doc:`Forecasting Competitions <251-forecasting-competitions>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Forecasting Benchmarks <https://insightful-data-lab.com/2025/08/22/forecasting-benchmarks/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
