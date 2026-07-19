:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-simple-baseline-methods:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Simple Baseline Methods</b></div>`

=========================
Simple Baseline Methods
=========================

*Easy reference forecasts (last value, mean) that stronger models must beat.*

.. important::

   **AI-generated content.** This page was written with the assistance of an
   AI language model and is provided as a learning aid. Despite careful
   review, it may still contain mistakes, omissions, or out-of-date
   information. Whether you are new to the topic, a team lead, or a senior
   practitioner, treat it as a starting point rather than an authoritative
   reference: read it critically and independently verify anything you act on
   (code, commands, figures, and factual claims) against official
   documentation and primary sources before relying on it.

What they are
-------------

**Simple baseline methods** are a small family of trivially simple forecasts used to set a **statistical
baseline** before anything complex. The staples: the **mean** method (forecast the historical **average**,
which smooths away seasonality), the **naïve** method (carry the **last value** forward — a random walk),
the **seasonal naïve** (carry the value from **one season ago**), and the **drift** method (naïve plus a
**trend** equal to the average historical change).

The drift formula
-----------------

.. math::

   \hat{y}_{T+h} = y_T + h \cdot \frac{y_T - y_1}{T-1},

equivalent to drawing a line through the **first and last** observations and extrapolating it forward.

Why use them
------------

They are **cheap and transparent**, and any sophisticated model must **beat** them to earn its
complexity. Choose by structure — the **mean** for a flat series, the **naïve** for a random walk, the
**seasonal naïve** for strong seasonality, **drift** for a trend — and note that **averaging** several
often improves accuracy. *KISS: keep it sophisticatedly simple.*

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Time Series Forecasting <256-time-series-forecasting>` · :doc:`Forecasting Competitions <251-forecasting-competitions>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Simple Baseline Methods <https://insightful-data-lab.com/2025/08/22/simple-baseline-methods/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
