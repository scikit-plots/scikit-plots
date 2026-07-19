:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-prophet-time-series-forecasting-by-facebook-meta:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Prophet — Time Series Forecasting by Facebook (Meta)</b></div>`

======================================================
Prophet — Time Series Forecasting by Facebook (Meta)
======================================================

*An open-source library for decomposable time-series forecasting with trend and seasonality.*

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

**Prophet** is an **open-source forecasting library** from Facebook (now Meta) that makes time-series
forecasting **simple, scalable and interpretable** — designed for business data with **trends,
seasonality and holidays**, and usable without deep statistical expertise.

The decomposable model
------------------------

Prophet models a series as a sum of interpretable components,

.. math::

   y(t) = g(t) + s(t) + h(t) + \varepsilon_t,

where :math:`g(t)` is the **trend** (linear, or logistic with saturation
:math:`g(t) = \frac{C}{1 + \exp(-k(t - m))}`, plus automatic **changepoints**), :math:`s(t)` is
**seasonality** (a Fourier series for weekly, yearly or custom cycles), :math:`h(t)` captures
**holidays and events** from a supplied list, and :math:`\varepsilon_t` is noise.

Strengths and limits
--------------------

Prophet is **user-friendly** (just a ``ds``/``y`` DataFrame), **interpretable** (each component is
separable), **robust** to missing data and outliers, detects changepoints automatically, and
**scales** across many series. Its limits follow from its **additive** design: it does not model
autoregressive correlations, it suits **daily/weekly/monthly** business data rather than
high-frequency signals, and it is **less powerful than LSTMs or Transformers** on complex patterns.

In practice
-----------

.. code-block:: python

   from prophet import Prophet

   # df has two columns: ds (datestamp) and y (value)
   model = Prophet()
   model.fit(df)

   future = model.make_future_dataframe(periods=90)
   forecast = model.predict(future)
   model.plot(forecast)
   model.plot_components(forecast)

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Time Series <010-time-series>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Stockout Rate <221-stockout-rate>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Simple Baseline Methods <248-simple-baseline-methods>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Prophet — Time Series Forecasting by Facebook (Meta) <https://insightful-data-lab.com/2025/08/23/prophet-time-series-forecasting-by-facebook-meta/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
