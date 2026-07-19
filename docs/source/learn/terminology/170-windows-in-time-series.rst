:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-windows-in-time-series:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Windows (in Time-Series)</b></div>`

==========================
Windows (in Time-Series)
==========================

*Fixed spans over which time-series features or aggregates are computed.*

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

In time-series analysis and monitoring, a **window** is a **slice of time** over which you compute a
statistic — a mean, sum, baseline or anomaly score. The window defines **how much past data** counts,
and it can **stay fixed**, **slide forward**, or **expand**.

The three types
---------------

A **fixed window** always covers a set period — "weekly sales, Monday to Sunday". A **rolling (or
sliding) window** moves forward step by step, recomputing as it goes — "a 7-day rolling average of
temperature" — and is the workhorse for smoothing, baselines and anomaly detection. An **expanding
window** starts at a point and **grows** as data arrives — "cumulative average sales since launch".

Baseline vs current
-------------------

Monitoring usually compares two windows. A **baseline window** — a rolling average over the last N
weeks — defines what "normal" looks like. A short **current window** — the last 24 hours to 7 days —
shows whether recent behaviour has **deviated** from that baseline.

An example
----------

For API monitoring, a rolling **8-week baseline** gives an average latency of **200ms**. The
**current 24-hour window** shows **350ms**. Because 350ms far exceeds the baseline, the comparison
**flags performance degradation** — the same windowed-comparison logic that underlies drift detection
and leading indicators.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>` · :doc:`Expanding Window Cross-Validation <130-expanding-window-cross-validation>` · :doc:`Time Series <010-time-series>` · :doc:`Monitoring Pipelines <162-monitoring-pipelines>` · :doc:`Drift Detection <138-drift-detection>` · :doc:`Leading Indicators <169-leading-indicators>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Windows (in Time-Series) <https://insightful-data-lab.com/2025/08/23/windows-in-time-series-data-analysis/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
