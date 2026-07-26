:html_theme.sidebar_secondary.remove:

..
   GENERATED FILE - do not edit by hand.
   Full body lives in term_content.py (CONTENT[title]).
   Regenerate: python build_terminology.py

.. role:: raw-html(raw)
   :format: html

.. |br| raw:: html

   <br/>

.. _term-temporal-autocorrelation-serial-correlation:

:raw-html:`<div align="center" style="text-align:center;font-size:1.12rem;margin:0.45rem 0 0.2rem">📈&nbsp;&nbsp;<b>Temporal autocorrelation (Serial Correlation)</b></div>`

===============================================
Temporal autocorrelation (Serial Correlation)
===============================================

*Correlation of a time series with its own past values.*

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

**Temporal autocorrelation** (or **serial correlation**) is when the values of a **time
series correlate with their own past** — the value at time :math:`t` depends partly on
:math:`t-1, t-2, \dots`. It is the defining property of time-series data and the most common
way the IID assumption breaks.

The measure
-----------

The **autocorrelation function (ACF)** at lag :math:`k` is

.. math::

   \rho_k = \frac{\operatorname{Cov}(X_t, X_{t-k})}{\sigma^2},

the correlation between the series and its own lag-:math:`k` copy, where :math:`\sigma^2` is
the series variance. Stock prices (today near yesterday), temperature and weekly website
traffic all show it.

Why it matters
--------------

First, it **violates IID** — past strongly influences future, so models that assume
independence are wrong. Second, it **drives forecasting**: ARIMA and SARIMA explicitly model
autocorrelation, and **ACF/PACF** plots reveal the AR and MA orders. Third, it is a
**diagnostic**: the **Durbin-Watson** test checks regression residuals, and autocorrelated
residuals signal a misspecified model.

Positive, negative, and reading the plot
----------------------------------------

**Positive** autocorrelation means high tends to follow high (**momentum**); **negative**
means high tends to follow low (**mean-reversion**). On an ACF plot, strong spikes at lags
1, 2 and 7 would suggest **weekly seasonality**.

----

*Theme:* :ref:`Signal Processing & Time Series <term-theme-signal>` :raw-html:`&nbsp;·&nbsp;` :doc:`All terminology <index>`

----

.. hint::
   **Mind map — connected ideas**

   :doc:`IID (Independent and Identically Distributed) <126-iid-independent-and-identically-distributed>` · :doc:`Time Series <010-time-series>` · :doc:`Blocked Splits (Single Holdout) <128-blocked-splits-single-holdout>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Signal Processing <009-signal-processing>` · :doc:`Sliding Window (Rolling Window) Cross-Validation <129-sliding-window-rolling-window-cross-validation>`

----

.. hint::
   **More in Signal Processing & Time Series**

   :doc:`ARIMA (AutoRegressive Integrated Moving Average) <224-arima-autoregressive-integrated-moving-average>` · :doc:`Bayesian Time Series <052-bayesian-time-series>` · :doc:`Forecast Error <250-forecast-error>` · :doc:`Forecasting Benchmarks <245-forecasting-benchmarks>` · :doc:`Forecasting Competitions <251-forecasting-competitions>` · :doc:`Log-Space <257-log-space>` · :doc:`Low-pass Filtering <005-low-pass-filtering>` · :doc:`LSTM — Long Short-Term Memory Networks <223-lstm-long-short-term-memory-networks>` · :doc:`M-Competitions (Makridakis Competitions) <244-m-competitions-makridakis-competitions>` · :doc:`Naïve Baseline Forecast <249-naive-baseline-forecast>` · :doc:`Prophet — Time Series Forecasting by Facebook (Meta) <222-prophet-time-series-forecasting-by-facebook-meta>` · :doc:`Seasonal Lag <247-seasonal-lag>` · :doc:`Seasonality <329-seasonality>` · :doc:`Signal Processing <009-signal-processing>`

.. seealso::

   **Source article** Adapted (context, re-expressed) in our own words from: `Temporal autocorrelation (Serial Correlation) <https://insightful-data-lab.com/2025/08/24/temporal-autocorrelation-serial-correlation/>`__ (insightful-data-lab.com).

.. tags:: purpose: reference, topic: terminology, level: advanced
